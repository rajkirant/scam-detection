#!/usr/bin/env python3
"""
blind_rewrite.py - normalise surface style across both classes without
letting the rewriter know which class it is handling.

WHY BLIND
---------
Rewriting a dataset until a particular baseline fails is circular: the
resulting comparison measures how the data was built, not how the systems
perform. The safeguard here is that the rewriting model never sees the label.
It is given one identical instruction for every transcript, scam or
legitimate alike, so it has no way to style the two classes differently. Any
lexical difference that survives is therefore content, not register.

Do not iterate this script against a detector's score. Run it once, measure,
and report whatever comes out. If bag-of-words still separates the classes,
that is a finding about the corpus, not a reason to rewrite again.

WHAT IT NORMALISES
------------------
  - grammatical mood (the your/our/we/will vs would/if/this is split)
  - opening and closing conventions
  - sentence length and punctuation density
  - target word count, which removes the length artifact

WHAT IT PRESERVES
-----------------
  - every factual claim, request, and instruction
  - who is calling and what they want
  - the sequence of events
  - whether the call is in fact a scam, which is never stated to the model

Usage:
    python scripts/blind_rewrite.py \
        --in datasets/zhi_balanced_333x333.csv \
        --out datasets/zhi_rewritten.csv \
        --model qwen2.5:14b

    # resume after an interruption - already-done rows are skipped
    python scripts/blind_rewrite.py --in ... --out ... --resume

    # try 10 rows first and eyeball them
    python scripts/blind_rewrite.py --in ... --out /tmp/sample.csv --limit 10 --show
"""

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# One instruction, used for every transcript. The label is never interpolated
# into this prompt and never reaches the model.
REWRITE_PROMPT = """Rewrite this telephone call transcript, changing only the
wording while keeping everything else identical.

The transcript is raw speech-to-text output. Your rewrite MUST look exactly
like speech-to-text output too. This is the most important requirement.

FORM RULES - the output is unusable if any is broken:
- all lowercase, no capital letters anywhere
- no punctuation at all: no full stops, no commas, no question marks, no apostrophes
- write contractions as separate letters the way the input does: i m, you re, it s, we ll, don t, that s
- keep some disfluencies like uh and um, about one every forty words, no more
- both speakers run together as one continuous stream, no speaker labels, no line breaks
- keep every placeholder in square brackets exactly as given: [Company] [Title] [Number] [Product] [Greetings] [Money]

KEEP UNCHANGED:
- every fact, figure, request, and instruction
- what the caller wants and what they ask the person to do
- the order events happen in
- the caller's stated role or organisation
- the length, within twenty percent of the original

Do not judge, label, or comment on the call. Do not add warnings. Do not
clean it up into proper writing. Return only the rewritten transcript.

Transcript:
{text}"""


# Label-aware variant. The label is used ONLY to protect content, never to set
# style: the style block below is byte-identical to the blind prompt's. This
# guards against the rewriter quietly sanitising a scam call - tidying away the
# pressure tactics as if they were clumsy phrasing - which would leave a row
# labelled scam that no longer is one.
INTENT = {
    "scam": ("This call is an attempt to manipulate or defraud the person. "
             "Keep that attempt fully intact: every pressure tactic, every "
             "request, every false claim must survive the rewrite."),
    "nonscam": ("This call is a genuine, ordinary piece of business. Keep it "
                "genuine: do not add urgency, threats, or requests that were "
                "not already there."),
}

LABEL_AWARE_PROMPT = """Rewrite the following phone call transcript.

{intent}

Keep all of this exactly as it is:
- every fact, figure, name placeholder in square brackets, and amount
- what the caller wants and every action they ask for
- the order in which things happen
- the caller's stated role or organisation

Change only the surface style, so that the rewritten call:
- is spoken naturally, as one side of a real telephone conversation
- uses plain everyday wording, not marketing or promotional language
- uses a mix of statements and questions
- is between {lo} and {hi} words
- opens and closes the way an ordinary phone call does

Do not make the call more suspicious or less suspicious than it already is.
Do not add any warning sign that was not in the original.
Do not judge, label, or comment on the call.

Return only the rewritten transcript, with no preamble and no quotation marks.

Transcript:
{text}"""


def call_model(prompt, model, max_tokens=600, timeout=180):
    import requests
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.4, "num_predict": max_tokens},
              "keep_alive": "6h"},
        timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def clean_output(text):
    """Strip the wrappers models add despite being told not to."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t)
    t = re.sub(r'^(here is|here\'s|sure[,!]?|rewritten transcript:?)\s*'
               r'(the )?(rewritten )?(transcript:?)?\s*', "", t, flags=re.I)
    t = t.strip().strip('"').strip()
    return fix_placeholder_case(re.sub(r"\s+", " ", t))


CONTENT_STOP = set(
    "a an the this that these those i me my we us our you your he him his she "
    "her it its they them their is am are was were be been being do does did "
    "have has had will would shall should can could may might must and or but "
    "if because so than then when while as of to in on at by for with from "
    "into about over under not no very too also just only both each any some "
    "all there here what which who how why".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in CONTENT_STOP and len(w) > 2}


def content_drift(original, rewritten):
    """How much meaning-bearing vocabulary the rewrite invented or lost.

    In label-aware mode this is the guard against the model drifting toward
    its own stereotype of a scam instead of restating the original call.
    """
    o, r = content_words(original), content_words(rewritten)
    if not o:
        return 0.0, 0.0, set()
    added = r - o
    kept = len(o & r) / len(o)
    return kept, len(added) / max(1, len(r)), added


CANONICAL_PH = {"company": "[Company]", "title": "[Title]", "number": "[Number]",
                "product": "[Product]", "greetings": "[Greetings]", "money": "[Money]"}


def fix_placeholder_case(text):
    """qwen lowercases [Company] to [company] because we asked for lowercase.
    Restore the canonical casing so the placeholder is preserved, not dropped."""
    def repl(m):
        return CANONICAL_PH.get(m.group(1).lower(), m.group(0))
    return re.sub(r"\[([A-Za-z]+)\]", repl, text)


def placeholders(text):
    return sorted(set(re.findall(r"\[[A-Za-z]+\]", text)))


def check(original, rewritten, lo, hi, max_added=0.55, min_kept=0.30):
    """Cheap sanity checks. Returns a list of problems, empty if fine."""
    problems = []
    kept, added, _ = content_drift(original, rewritten)
    if kept < min_kept:
        problems.append("lost %.0f%% of content words" % (100 * (1 - kept)))
    if added > max_added:
        problems.append("invented %.0f%% new content words" % (100 * added))
    n = len(rewritten.split())
    if n < lo * 0.6:
        problems.append("too short (%d words)" % n)
    if n > hi * 1.8:
        problems.append("too long (%d words)" % n)
    lost = set(placeholders(original)) - set(placeholders(rewritten))
    if lost:
        problems.append("dropped placeholders: %s" % ",".join(sorted(lost)))
    if re.search(r"\b(scam|fraudulent|phishing|legitimate call|this is a scam)\b",
                 rewritten, re.I) and not re.search(r"\bscam\b", original, re.I):
        problems.append("model editorialised about scam status")
    if len(rewritten) < 40:
        problems.append("output nearly empty")
    # ASR register: input has no capitals/punctuation, output must not either
    if re.search(r"[A-Z]", re.sub(r"\[[A-Za-z]+\]", "", rewritten)):
        problems.append("introduced capital letters")
    if re.search(r"[.?!,]", rewritten):
        problems.append("introduced punctuation")
    if re.search(r"[a-z]'[a-z]", rewritten):
        problems.append("introduced apostrophe contractions")
    return problems


def style_report(rows, label_key="label", text_key="text"):
    """Compare surface markers across classes. Run before and after."""
    print("  %-10s %6s %10s %10s %10s %10s" %
          ("class", "n", "med words", "full stop", "question", "'would'"))
    for lab in sorted({r[label_key] for r in rows}):
        sub = [r[text_key] for r in rows if r[label_key] == lab]
        if not sub:
            continue
        w = [len(t.split()) for t in sub]
        dot = 100.0 * sum(1 for t in sub if "." in t) / len(sub)
        q = 100.0 * sum(1 for t in sub if "?" in t) / len(sub)
        wd = 100.0 * sum(1 for t in sub if re.search(r"\bwould\b", t, re.I)) / len(sub)
        print("  %-10s %6d %10.0f %9.0f%% %9.0f%% %9.0f%%"
              % (lab, len(sub), st.median(w), dot, q, wd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default=os.environ.get("SCAM_MODEL", "qwen2.5:14b"))
    ap.add_argument("--target-words", type=int, default=0,
                    help="0 = use the median length of the whole corpus, so both "
                         "classes converge on one length and the length artifact goes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--show", action="store_true", help="print each before/after pair")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--label-aware", action="store_true",
                    help="tell the model whether the call is a scam, so it "
                         "preserves the intent rather than sanitising it. The "
                         "style instruction stays identical across classes.")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    target = args.target_words or int(st.median(len(r["text"].split()) for r in rows))
    lo, hi = int(target * 0.8), int(target * 1.2)

    print("=" * 74)
    print("Blind rewrite - the model is never told the label")
    print("=" * 74)
    print("  input : %s (%d rows)" % (args.src, len(rows)))
    print("  model : %s" % args.model)
    print("  mode  : %s" % ("LABEL-AWARE (intent preserved, style identical "
                            "across classes)" if args.label_aware else "BLIND"))
    print("  target: %d words per call (range %d-%d)" % (target, lo, hi))
    print("\n  style BEFORE:")
    style_report(rows)
    print()

    done = {}
    if args.resume and Path(args.out).exists():
        for r in csv.DictReader(open(args.out, encoding="utf-8")):
            done[r["id"]] = r
        print("  resuming, %d rows already written\n" % len(done))

    out_rows, failures = [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        if r["id"] in done:
            out_rows.append(done[r["id"]])
            continue

        if args.label_aware:
            prompt = LABEL_AWARE_PROMPT.format(
                intent=INTENT.get(r["label"], INTENT["nonscam"]),
                lo=lo, hi=hi, text=r["text"])
        else:
            prompt = REWRITE_PROMPT.format(lo=lo, hi=hi, text=r["text"])
        rewritten, problems = "", ["not attempted"]
        for attempt in range(args.retries + 1):
            try:
                rewritten = clean_output(call_model(prompt, args.model))
            except Exception as exc:
                problems = ["request failed: %s" % exc]
                time.sleep(2)
                continue
            problems = check(r["text"], rewritten, lo, hi)
            if not problems:
                break

        if problems:
            failures.append((r["id"], problems))
            rewritten = rewritten or r["text"]

        out_rows.append({"pair_id": r.get("pair_id", ""), "id": r["id"],
                         "label": r["label"], "source": r.get("source", ""),
                         "text": rewritten,
                         "original_text": r["text"],
                         "rewrite_issues": ";".join(problems)})

        if args.show:
            print("  --- %s ---" % r["id"])
            print("  BEFORE: %s" % r["text"][:180])
            print("  AFTER : %s" % rewritten[:180])
            if problems:
                print("  ISSUES: %s" % problems)
            print()
        if i % 25 == 0:
            rate = (time.time() - t0) / i
            print("    %d/%d  (%.1fs each, ~%.0f min left)"
                  % (i, len(rows), rate, rate * (len(rows) - i) / 60))
            # checkpoint so an interruption does not lose everything
            _write(args.out, out_rows)

    _write(args.out, out_rows)

    print("\n" + "=" * 74)
    print("  rewritten %d rows in %.0f min" % (len(out_rows), (time.time() - t0) / 60))
    if failures:
        print("  %d rows had issues after retries:" % len(failures))
        for fid, probs in failures[:10]:
            print("    %s: %s" % (fid, probs))
        print("  (their original text was kept; check rewrite_issues in the CSV)")
    print("\n  style AFTER:")
    style_report(out_rows)
    print("\n  wrote %s" % args.out)
    print("\n  Now measure, and report whatever you get:")
    print("    python scripts/combined_evaluate.py --csv %s --trivial-only --bow-features"
          % args.out)
    print("  Do not rewrite again in response to that number.")
    print("=" * 74)


def _write(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "id", "label", "source", "text", "original_text", "rewrite_issues"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
