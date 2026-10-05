#!/usr/bin/env python3
"""
mcq_ontology.py - the MCQ ontology LLM: one question, the call's category.

It starts from a JSON file (knowledge/mcq_ontology.json unless told
otherwise) holding ONE multiple-choice question and a few options. Each option
is a category of call and carries a verdict, scam or legit:

    {"prompt": "Which category does this call belong to?",
     "options": [{"id": "ssn", "text": "Social Security: ...", "verdict": "scam"},
                 {"id": "delivery", "text": "Delivery: ...", "verdict": "legit"},
                 ...]}

No retrieval of any kind. The transcript goes to the model on its own, with
the question and the options lettered A, B, C ..., and the model answers one
letter. Ollama returns the probability the model put on every letter, so the
answer is a distribution over the categories rather than a single pick:

  category   the option with the most probability
  P(scam)    the total probability on the options whose verdict is scam
  verdict    Fraud when P(scam) is 0.5 or more

One request per call, generating a token or two.

The options come from the transcripts. `build` makes them from a dataset's
category column (type, scam_type, topic, ...): one option per category, its
verdict the label most of that category's calls carry, and - with --describe -
a one-line description the model writes after reading a few of those calls.
Edit the JSON by hand afterwards if a description needs it; nothing else
reads the descriptions.

An Ollama too old to return logprobs still answers the letter: the category
is then that letter, P(scam) 1 or 0, and the run says so.

    python scripts/mcq_ontology.py ask --text "Hello, this is your bank ..."
    python scripts/mcq_ontology.py evaluate --csv datasets/zhi_english_646.csv --limit 40
    python scripts/mcq_ontology.py build --csv datasets/huggingface_1600.csv \\
        --out knowledge/mcq_huggingface.json --describe

Your own question can be trained on a dataset too: the model answers it in a
few words about a sample of calls, groups the answers into a few options, and
counts the scam and legitimate calls on each; the result is saved in
knowledge/questions/ and asking it later uses the options in that file.

    python scripts/mcq_ontology.py train-question --csv datasets/huggingface_1600.csv \\
        --question "What does the caller ask for?" --out knowledge/questions/asks.json
    python scripts/mcq_ontology.py question --saved knowledge/questions/asks.json \\
        --text "Hello, this is your bank ..."

Standard library only (Ollama is spoken to through llm_judge), so web_ui.py
can import it without the venv.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_io                                              # noqa: E402
import llm_judge                                               # noqa: E402
import ollama_ctx                                              # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
DEFAULT_ONTOLOGY = KNOWLEDGE_DIR / "mcq_ontology.json"

VERDICTS = ("scam", "legit")
MAX_OPTIONS = 12                 # letters A-L: past that it stops being "a few"
LETTERS = "ABCDEFGHIJKL"
ANSWER_TOKENS = 3                # the letter, and room for a space before it
TOP_LOGPROBS = 20

# columns a dataset may keep each call's category in, most specific first
CATEGORY_COLS = ("category", "type", "scam_type", "topic", "call_type",
                 "subtype", "class_name")

HEAD = """You are a scam detection analyst. Read this phone call transcript.

Transcript:
{transcript}

"""

_LOGPROBS_OK = None          # None = not tried yet; False = this Ollama has none


# --------------------------------------------------------------- the file

def check_ontology(obj):
    """Every problem with an ontology, as a list of sentences; [] if none."""
    if not isinstance(obj, dict):
        return ["the file must hold one JSON object, {...}"]
    probs = []
    if not str(obj.get("prompt") or "").strip():
        probs.append('"prompt" - the question - is missing or empty')
    opts = obj.get("options")
    if not isinstance(opts, list):
        return probs + ['"options" must be a list of options']
    if not 2 <= len(opts) <= MAX_OPTIONS:
        probs.append("there must be between 2 and %d options; there are %d"
                     % (MAX_OPTIONS, len(opts)))
    seen = set()
    for i, o in enumerate(opts, 1):
        if not isinstance(o, dict):
            probs.append("option %d is not an object" % i)
            continue
        if not str(o.get("text") or "").strip():
            probs.append('option %d has no "text"' % i)
        if o.get("verdict") not in VERDICTS:
            probs.append('option %d: "verdict" must be "scam" or "legit", not %r'
                         % (i, o.get("verdict")))
        oid = str(o.get("id") or "opt%d" % i)
        if oid in seen:
            probs.append('option %d: the id "%s" is used twice' % (i, oid))
        seen.add(oid)
    verdicts = {o.get("verdict") for o in opts if isinstance(o, dict)}
    if isinstance(opts, list) and len(opts) >= 2 and not set(VERDICTS) <= verdicts:
        probs.append("at least one option must be scam and one legit - "
                     "otherwise every call gets the same verdict")
    return probs


def normalise(obj):
    """The ontology with ids filled in and text trimmed. Assumes it checked."""
    out = dict(obj)
    out["prompt"] = str(obj["prompt"]).strip()
    out["options"] = []
    for i, o in enumerate(obj["options"], 1):
        o = dict(o)
        o["id"] = str(o.get("id") or "opt%d" % i)
        o["text"] = " ".join(str(o["text"]).split())
        out["options"].append(o)
    return out


def load_ontology(path=None):
    """The ontology at `path`, checked. ValueError naming every problem."""
    path = Path(path or DEFAULT_ONTOLOGY)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("no ontology file at %s" % path)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON: %s (line %d, column %d)"
                         % (path.name, e.msg, e.lineno, e.colno))
    probs = check_ontology(obj)
    if probs:
        raise ValueError("%s: %s" % (path.name, "; ".join(probs)))
    return normalise(obj)


def is_mcq_ontology(path):
    """Is this JSON file one of these (rather than another knowledge file)?"""
    try:
        return not check_ontology(json.loads(Path(path).read_text(
            encoding="utf-8")))
    except (OSError, ValueError):
        return False


def list_ontologies():
    """Every one-question ontology in knowledge/, the default first."""
    out = []
    for p in sorted(KNOWLEDGE_DIR.glob("*.json")):
        if not is_mcq_ontology(p):
            continue
        o = load_ontology(p)
        out.append({"path": "knowledge/" + p.name, "name": p.name,
                    "prompt": o["prompt"], "options": len(o["options"]),
                    "default": p.name == DEFAULT_ONTOLOGY.name})
    out.sort(key=lambda d: (not d["default"], d["name"]))
    return out


# ------------------------------------------------------------- the question

def question(onto):
    """The question and its lettered options, as the model sees them."""
    n = len(onto["options"])
    lines = [onto["prompt"]]
    lines += ["%s - %s" % (LETTERS[i], o["text"])
              for i, o in enumerate(onto["options"])]
    lines.append("Answer with exactly one letter, A to %s." % LETTERS[n - 1])
    lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(transcript, onto):
    return HEAD.format(transcript=transcript) + question(onto)


def answer_letter(token, n):
    """The letter (A..) a token answers with among n options, or None."""
    t = token.strip().strip("*\"'`().:").upper()
    return t if len(t) == 1 and t in LETTERS[:n] else None


def generate(prompt, n_tokens, top=TOP_LOGPROBS, timeout=300):
    """(text, [(token, [(alternative, logprob), ...]) per generated token]).
    The alternatives are empty when this Ollama returns no logprobs."""
    num_ctx = ollama_ctx.fit_num_ctx(prompt, n_tokens + 2, where="mcq")
    payload = {
        "model": llm_judge.DEFAULT_MODEL, "prompt": prompt, "stream": False,
        "logprobs": True, "top_logprobs": top,
        "options": {"temperature": 0.0, "num_predict": n_tokens,
                    "num_ctx": int(num_ctx)},
    }
    try:
        body = llm_judge._post("/api/generate", payload, timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError("ollama refused that (%s): %s"
                           % (e.code, e.read().decode("utf-8", "replace")[:300]))
    except OSError as e:           # URLError, a refused connection, a timeout
        raise llm_judge._unreachable(e)
    steps = []
    for pos in body.get("logprobs") or []:
        alts = [(a.get("token", ""), a["logprob"])
                for a in (pos.get("top_logprobs") or [])
                if a.get("logprob") is not None]
        if not alts and pos.get("logprob") is not None:
            alts = [(pos.get("token", ""), pos["logprob"])]
        steps.append((pos.get("token", ""), alts))
    return body.get("response", ""), steps


def _note_no_logprobs():
    global _LOGPROBS_OK
    if _LOGPROBS_OK is None:
        _LOGPROBS_OK = False
        print("    NOTE this Ollama returns no logprobs (it needs a recent "
              "version): the category is the letter it answered and P(scam) "
              "is 1 or 0")


def judge(transcript, onto):
    """Ask the question about one transcript. One request.

    Returns a dict:
      probs     probability of each option, in order (sums to 1 when read)
      choice    index of the most likely option, or None if unreadable
      category  that option's id, text its text, category_verdict its verdict
      p_scam    total probability on the scam options, or None
      verdict   "Fraud" | "Normal" | None (None: no letter could be read)
      how       "logprobs" (measured) | "letter" (no logprobs: the answer
                alone) | "unreadable"
      answered  what the model actually wrote
    """
    global _LOGPROBS_OK
    opts = onto["options"]
    n = len(opts)
    text, steps = generate(build_prompt(transcript, onto), ANSWER_TOKENS)
    has_lp = any(alts for _, alts in steps)
    if has_lp:
        _LOGPROBS_OK = True
    else:
        _note_no_logprobs()

    probs, how = None, "unreadable"
    # the first token that is an option letter; a leading space or "**" may
    # come before it
    li = next((i for i, (tok, _) in enumerate(steps) if answer_letter(tok, n)),
              None)
    if li is not None:
        mass = [0.0] * n
        for tok, lp in steps[li][1]:
            k = answer_letter(tok, n)
            if k:
                mass[LETTERS.index(k)] += math.exp(lp)
        total = sum(mass)
        if has_lp and total > 0:
            probs, how = [m / total for m in mass], "logprobs"
        else:
            probs, how = [0.0] * n, "letter"
            probs[LETTERS.index(answer_letter(steps[li][0], n))] = 1.0
    elif not steps:
        k = answer_letter((text.split() or [""])[0], n)
        if k:
            probs, how = [0.0] * n, "letter"
            probs[LETTERS.index(k)] = 1.0

    if probs is None:
        return {"probs": [0.0] * n, "choice": None, "category": None,
                "text": None, "category_verdict": None, "p_scam": None,
                "verdict": None, "how": "unreadable", "answered": text}
    choice = max(range(n), key=lambda i: probs[i])
    p_scam = sum(p for p, o in zip(probs, opts) if o["verdict"] == "scam")
    return {"probs": probs, "choice": choice, "category": opts[choice]["id"],
            "text": opts[choice]["text"],
            "category_verdict": opts[choice]["verdict"],
            "p_scam": p_scam, "verdict": "Fraud" if p_scam >= 0.5 else "Normal",
            "how": how, "answered": text}


def presize(transcripts, onto):
    """Size the context window once, for the longest call in the run, so the
    model is loaded once rather than reloaded each time a longer call turns
    up (the window only grows - see ollama_ctx.STICKY)."""
    if not transcripts:
        return None
    longest = max(transcripts, key=lambda t: len(t or ""))
    return ollama_ctx.fit_num_ctx(build_prompt(longest, onto),
                                  ANSWER_TOKENS + 2, where="mcq")


def explain(res, onto):
    """A short account of one answer: the category, then P(scam)."""
    if res["choice"] is None:
        return "unreadable answer: %r" % (res["answered"] or "")[:60]
    o = onto["options"][res["choice"]]
    return ("%s - %s (%s, %.0f%%); P(scam) %.0f%%%s"
            % (LETTERS[res["choice"]], o["id"], o["verdict"],
               100 * res["probs"][res["choice"]], 100 * res["p_scam"],
               "" if res["how"] == "logprobs" else ", no logprobs"))


def category_table(onto, chosen, truths):
    """Which category the model put each call in, against the true label.

    The one table that says what the model actually did: whether the scam
    calls landed in scam categories, and which legitimate category a missed
    scam was taken for. `chosen` holds option indexes (None = unreadable),
    `truths` booleans (True = scam).
    """
    opts = onto["options"]
    c = Counter((ch, bool(t)) for ch, t in zip(chosen, truths))
    lines = ["%-3s %-26s %-6s %7s %7s" % ("", "category the model chose",
                                           "means", "scam", "legit")]
    for i, o in enumerate(opts):
        s, l = c[(i, True)], c[(i, False)]
        if s or l:
            lines.append("%-3s %-26s %-6s %7d %7d"
                         % (LETTERS[i], o["id"][:26], o["verdict"], s, l))
    s, l = c[(None, True)], c[(None, False)]
    if s or l:
        lines.append("%-3s %-26s %-6s %7d %7d" % ("", "(unreadable)", "", s, l))
    return lines


def judge_all(transcripts, onto, parallel=None):
    """judge() over many calls, in order. SCAM_LLM_PARALLEL (or `parallel`)
    sends that many at once - which only helps if Ollama serves requests in
    parallel (OLLAMA_NUM_PARALLEL) and the GPU has room for it."""
    from concurrent.futures import ThreadPoolExecutor
    parallel = max(1, int(parallel or os.environ.get("SCAM_LLM_PARALLEL") or 1))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for res in pool.map(lambda t: judge(t, onto), transcripts):
            yield res


# ----------------------------------------------- options from the transcripts

def category_columns(header):
    """The columns of a dataset that may hold each call's category."""
    by = {(h or "").lstrip("\ufeff").strip().lower(): h for h in header}
    return [by[c] for c in CATEGORY_COLS if c in by]


def _name(cat):
    words = cat.replace("_", " ").replace("-", " ").strip()
    return words.upper() if len(words) <= 3 else words[:1].upper() + words[1:]


def describe(texts, timeout=300):
    """What a category of call is about, in one line, written by the model
    after reading a few of its calls. None when it gave nothing usable."""
    calls = "\n\n".join("Call %d:\n%s" % (i, " ".join(t.split()[:220]))
                        for i, t in enumerate(texts, 1))
    prompt = ("Here are %d phone call transcripts that all belong to the same "
              "category of call.\n\n%s\n\nIn one short line of at most 15 "
              "words, say what this category of call is about: who is calling "
              "and what they want. Do not say whether the calls are scams or "
              "legitimate.\nCategory:" % (len(texts), calls))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, 60, where="mcq describe")
    try:
        out = llm_judge.generate(prompt, max_tokens=60, num_ctx=num_ctx,
                                 timeout=timeout)
    except RuntimeError as e:
        print("    could not describe it: %s" % e)
        return None
    line = (out.get("response") or "").strip().split("\n")[0]
    line = line.strip().strip('"').strip()
    return line or None


def build(csv_path, column=None, describe_with_model=False, examples=3,
          max_options=MAX_OPTIONS, seed=42):
    """An ontology whose options are the categories in a dataset's transcripts.

    One option per category, biggest first; past max_options the smallest are
    merged into one "other" option. Each verdict is the label most of that
    category's calls carry; a category split exactly evenly is marked
    "mixed" and given scam, the cautious side, and a column in which every
    category is split that way is refused - it does not separate scam from
    legitimate at all. Scam and legitimate options are interleaved so that a
    preference for early letters cannot line up with one verdict.
    """
    rows = dataset_io.read_rows(Path(csv_path))
    tcol, lcol, _ = dataset_io.columns(rows, csv_path)
    cols = category_columns(rows[0].keys())
    if column:
        col = next((h for h in rows[0].keys()
                    if h.lstrip("\ufeff").strip().lower() == column.lower()), None)
        if col is None:
            raise ValueError("%s has no column %r (it has: %s)"
                             % (csv_path, column, ", ".join(rows[0].keys())))
    elif cols:
        col = cols[0]
    else:
        raise ValueError(
            "%s has no category column (looked for: %s), so there is nothing "
            "in it to build options from" % (csv_path, ", ".join(CATEGORY_COLS)))

    groups = {}
    blank = 0
    for r in rows:
        text = (r[tcol] or "").strip()
        cat = (r[col] or "").strip()
        if not text:
            continue
        if not cat:
            blank += 1
            continue
        groups.setdefault(cat, []).append((text, dataset_io.is_scam(r[lcol])))
    if len(groups) < 2:
        raise ValueError("column %r has %d categor%s - a question needs at "
                         "least two options" % (col, len(groups),
                                                "y" if len(groups) == 1 else "ies"))

    cats = sorted(groups, key=lambda c: (-len(groups[c]), c))
    merged = None
    if len(cats) > max_options:
        keep, rest = cats[:max_options - 1], cats[max_options - 1:]
        merged = "other" if "other" not in keep else "other_merged"
        pooled = [x for c in rest for x in groups.pop(c)]
        groups[merged] = pooled
        cats = keep + [merged]

    options = []
    for cat in cats:
        calls = groups[cat]
        s = sum(1 for _, y in calls if y)
        l = len(calls) - s
        o = {"id": cat, "text": _name(cat),
             "verdict": "scam" if s >= l else "legit",
             "calls": {"scam": s, "legit": l}}
        if s == l:
            o["mixed"] = True
        if cat == merged:
            o["text"] = "Something else"
            o["merged"] = sorted(rest)
        options.append(o)
    if all(o.get("mixed") for o in options):
        raise ValueError(
            "every category in %r is split evenly between scam and legitimate "
            "calls, so the category says nothing about the label - this "
            "column cannot be turned into a verdict" % col)
    if not {"scam", "legit"} <= {o["verdict"] for o in options}:
        raise ValueError(
            "every category in %r is mostly %s, so every call would get the "
            "same verdict" % (col, options[0]["verdict"]))

    if describe_with_model:
        rng = random.Random(seed)
        for o in options:
            texts = [t for t, _ in groups[o["id"]]]
            pick = rng.sample(texts, min(examples, len(texts)))
            print("  describing %-24s from %d of its %d calls ..."
                  % (o["id"][:24], len(pick), len(texts)), flush=True)
            d = describe(pick)
            if d:
                o["text"] = "%s: %s" % (_name(o["id"]), d)
                print("    %s" % o["text"])

    scam = [o for o in options if o["verdict"] == "scam"]
    legit = [o for o in options if o["verdict"] == "legit"]
    mixed = []
    for i in range(max(len(scam), len(legit))):
        mixed += scam[i:i + 1] + legit[i:i + 1]

    return {
        "id": "call_category",
        "prompt": "Which category does this call belong to?",
        "about": ("One question: the category of the call. The options are "
                  "the categories in the %r column of %s; each verdict is the "
                  "label most of that category's calls carry. Scam and "
                  "legitimate options alternate." % (col, csv_path)),
        "built_from": {"dataset": str(csv_path), "column": col,
                       "calls": sum(len(v) for v in groups.values()),
                       "no_category": blank,
                       "described_by_model": bool(describe_with_model)},
        "options": mixed,
    }


# ------------------------------------------------- your own question
# Any question about one call, typed on the page. If the question lists its
# own options - "A) ... B) ...", one per line, or "options: x / y / z" - it is
# put as a multiple choice and answered like the ontology's question: one
# letter, with the probability on every option read back. If it lists none,
# the model answers in its own words, from the transcript and, where the
# transcript does not say, from what it knows.
MAX_ANSWER_TOKENS = 400

_OPT_LINE = re.compile(r"^\s*(?:\(?([A-La-l]|\d{1,2})[).:]|[-*\u2022])\s+(\S.*?)\s*$")
_OPT_INLINE = re.compile(r"(?:(?<=\s)|^)\(?([A-La-l])\)\s*")
_OPT_LIST = re.compile(r"\boptions?\s*(?:are|:|-)\s*(.+)$", re.I | re.S)


def parse_options(text):
    """(question, [option text, ...]) - the options the question lists, if
    it lists at least two; otherwise (the whole text, []).

    Three ways of listing them are understood: one per line ("A) yes",
    "1. yes", "- yes"), inline lettered ("... A) yes B) no"), and
    "options: yes / no / not said" (split on / | ; or commas).
    """
    text = (text or "").strip()
    lines = text.split("\n")
    stem, opts = [], []
    for ln in lines:
        m = _OPT_LINE.match(ln)
        if m:
            opts.append(m.group(2))
        elif opts and ln.strip():
            opts[-1] += " " + ln.strip()      # an option's text run onto a new line
        else:
            stem.append(ln)
    if len(opts) >= 2:
        return " ".join(" ".join(stem).split()), opts

    marks = list(_OPT_INLINE.finditer(text))
    # inline letters only count as options when they run A, B, C ... in order
    run = []
    for m in marks:
        if m.group(1).upper() == LETTERS[len(run)]:
            run.append(m)
    if len(run) >= 2:
        opts = [text[a.end():(b.start() if b else len(text))].strip(" ,;")
                for a, b in zip(run, run[1:] + [None])]
        if all(opts):
            return " ".join(text[:run[0].start()].split()), opts

    m = _OPT_LIST.search(text)
    if m:
        body = m.group(1).strip().rstrip("?.")
        for sep in ("/", "|", ";", ","):
            parts = [x.strip() for x in body.split(sep) if x.strip()]
            if len(parts) >= 2:
                return " ".join(text[:m.start()].split()), parts
    return " ".join(text.split()), []


def _mcq_prompt(transcript, stem, opts):
    n = len(opts)
    return HEAD.format(transcript=transcript) + "\n".join(
        [stem or "Which of these is right?"]
        + ["%s - %s" % (LETTERS[i], o) for i, o in enumerate(opts)]
        + ["Answer with exactly one letter, A to %s." % LETTERS[n - 1],
           "Answer:"])


def choose(transcript, stem, opts, timeout=300):
    """Put `stem` to the model as a multiple choice over `opts` (texts) about
    one call. One request. (probs, choice, how, answered, prompt): probs
    sums to 1 when read; choice is None when no letter could be read."""
    n = len(opts)
    prompt = _mcq_prompt(transcript, stem, opts)
    text, steps = generate(prompt, ANSWER_TOKENS, timeout=timeout)
    has_lp = any(alts for _, alts in steps)
    if not has_lp:
        _note_no_logprobs()
    probs, how = None, "unreadable"
    li = next((i for i, (tok, _) in enumerate(steps)
               if answer_letter(tok, n)), None)
    if li is not None:
        mass = [0.0] * n
        for tok, lp in steps[li][1]:
            k = answer_letter(tok, n)
            if k:
                mass[LETTERS.index(k)] += math.exp(lp)
        if has_lp and sum(mass) > 0:
            probs, how = [x / sum(mass) for x in mass], "logprobs"
        else:
            probs, how = [0.0] * n, "letter"
            probs[LETTERS.index(answer_letter(steps[li][0], n))] = 1.0
    elif not steps:
        k = answer_letter((text.split() or [""])[0], n)
        if k:
            probs, how = [0.0] * n, "letter"
            probs[LETTERS.index(k)] = 1.0
    probs = probs or [0.0] * n
    choice = (max(range(n), key=lambda i: probs[i])
              if how != "unreadable" else None)
    return probs, choice, how, text, prompt


def ask_question(transcript, question_text, options=None, timeout=300):
    """Answer one typed question about one call. One request.

    `options` - option texts, e.g. a saved question's - makes it multiple
    choice over those, and the question text is then taken as it is. Without
    them the question's own listed options are used, if it lists any.

    Returns {"mode": "options" | "free", "question": the stem, ...}:
      options   "options": [{"letter", "text", "p"}], "choice", "how"
                ("logprobs" | "letter" | "unreadable"), "answered"
      free      "answer": the model's own words, "truncated": whether it ran
                out of room
    """
    if options:
        stem, opts = " ".join(str(question_text or "").split()), list(options)
    else:
        stem, opts = parse_options(question_text)
    if not stem and not opts:
        raise ValueError("type a question")
    if len(opts) > MAX_OPTIONS:
        raise ValueError("at most %d options - that question lists %d"
                         % (MAX_OPTIONS, len(opts)))
    if opts:
        probs, choice, how, text, prompt = choose(transcript, stem, opts,
                                                  timeout)
        return {"mode": "options", "question": stem,
                "options": [{"letter": LETTERS[i], "text": o, "p": probs[i]}
                            for i, o in enumerate(opts)],
                "choice": choice, "how": how, "answered": text,
                "prompt_tokens_estimated": ollama_ctx.estimate_tokens(prompt)}

    prompt = HEAD.format(transcript=transcript) + (
        "Question: %s\n\nAnswer the question about this call. "
        "Use what the transcript says; where it does not say, "
        "answer from your own knowledge and say that you are. "
        "Keep it to a few sentences.\nAnswer:" % stem)
    num_ctx = ollama_ctx.fit_num_ctx(prompt, MAX_ANSWER_TOKENS,
                                     where="mcq question")
    out = llm_judge.generate(prompt, max_tokens=MAX_ANSWER_TOKENS,
                             num_ctx=num_ctx, timeout=timeout)
    return {"mode": "free", "question": stem,
            "answer": (out.get("response") or "").strip(),
            "truncated": out.get("done_reason") == "length",
            "prompt_tokens_estimated": ollama_ctx.estimate_tokens(prompt)}


# ------------------------------------------- a question trained on a dataset
# A question typed on the page can be trained on a dataset, and saved: the
# model answers it in a few words about a balanced sample of the dataset's
# calls, then groups those answers into a few options, and each sampled call
# is put back to the model as a multiple choice over them, which counts how
# many scam and legitimate calls land on each option. The question, its
# options and those counts go to knowledge/questions/<name>.json; asking the
# saved question later uses the options in that file.
#
# A question that already lists its own options keeps them: training then
# only does the last step, the counting.
QUESTIONS_DIR = KNOWLEDGE_DIR / "questions"
TRAIN_CALLS = 20
TRAIN_OPTIONS = 6
SHORT_ANSWER_TOKENS = 40
GROUP_TOKENS = 300
NOT_SAID = "not said"


def short_answer(transcript, stem, timeout=300):
    """The model's answer to the question about one call, in a few words."""
    prompt = HEAD.format(transcript=transcript) + (
        "Question: %s\n\nAnswer in a few words - at most 12 - from what the "
        "transcript says. If the transcript does not say, answer \"%s\".\n"
        "Short answer:" % (stem, NOT_SAID))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, SHORT_ANSWER_TOKENS,
                                     where="question training")
    out = llm_judge.generate(prompt, max_tokens=SHORT_ANSWER_TOKENS,
                             num_ctx=num_ctx, timeout=timeout)
    line = (out.get("response") or "").strip().split("\n")[0]
    return line.strip().strip("\"'*").strip().rstrip(".") or NOT_SAID


def _clean_option(line):
    m = _OPT_LINE.match(line)
    text = m.group(2) if m else line
    text = re.sub(r"\s*\(\s*\d+\s*(?:answers?|calls?)?\s*\)\s*$", "", text)
    text = text.strip().strip("\"'*").strip().rstrip(".")
    return " ".join(text.split())[:120]


def group_answers(stem, answers, max_options=TRAIN_OPTIONS, timeout=300):
    """A few options that cover the short answers, written by the model.

    Falls back to the most common answers themselves when the model's list
    has fewer than two usable lines.
    """
    listed = "\n".join("%d. %s" % (i, a) for i, a in enumerate(answers, 1))
    prompt = (
        "A model was asked this question about %d different phone calls:\n\n"
        "  %s\n\nIts short answers, one per call:\n\n%s\n\n"
        "Group these answers into at most %d options for a multiple-choice "
        "version of the question. Each option is a short phrase of at most 8 "
        "words that covers several of the answers. Together the options must "
        "cover every answer, and no two may mean the same thing. If some "
        "answers say the call does not say, keep one option for that.\n"
        "Write one option per line, each starting with \"- \", and nothing "
        "else.\nOptions:" % (len(answers), stem, listed, max_options))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, GROUP_TOKENS,
                                     where="question training")
    out = llm_judge.generate(prompt, max_tokens=GROUP_TOKENS, num_ctx=num_ctx,
                             timeout=timeout)
    opts, seen = [], set()
    for line in (out.get("response") or "").split("\n"):
        o = _clean_option(line)
        if not o or o.lower().rstrip(":") in ("options", "option") \
                or o.lower() in seen:
            continue
        seen.add(o.lower())
        opts.append(o)
    opts = opts[:max_options]
    if len(opts) >= 2:
        return opts, "grouped by the model"
    first = {}
    for a in answers:
        first.setdefault(a.lower(), a)
    common = [first[a] for a, _ in Counter(a.lower() for a in answers)
              .most_common(max_options)]
    if len(common) < 2:
        raise ValueError("every call got the same answer (%r), so there is "
                         "nothing to make options from - ask something the "
                         "calls differ on" % (answers[0] if answers else ""))
    return common, "the most common answers (the model's grouping was unusable)"


def sample_calls(csv_path, calls=TRAIN_CALLS, seed=42):
    """A balanced random sample: [(text, is_scam)], half of each label where
    the dataset has enough of both."""
    rows = dataset_io.read_rows(Path(csv_path))
    tcol, lcol, _ = dataset_io.columns(rows, csv_path)
    scam, legit = [], []
    for r in rows:
        t = (r[tcol] or "").strip()
        if t:
            (scam if dataset_io.is_scam(r[lcol]) else legit).append(t)
    if not scam and not legit:
        raise ValueError("%s has no transcripts" % csv_path)
    rng = random.Random(seed)
    k = min(calls // 2, len(scam))
    j = min(calls - k, len(legit))
    k = min(calls - j, len(scam))          # top up from scam if legit ran short
    pick = ([(t, True) for t in rng.sample(scam, k)]
            + [(t, False) for t in rng.sample(legit, j)])
    rng.shuffle(pick)
    return pick


def train_question(csv_path, question_text, calls=TRAIN_CALLS,
                   max_options=TRAIN_OPTIONS, seed=42, log=print):
    """Train a question on a dataset; returns what goes in its JSON file."""
    stem, listed = parse_options(question_text)
    if not stem:
        raise ValueError("type a question")
    if len(listed) > MAX_OPTIONS:
        raise ValueError("at most %d options - that question lists %d"
                         % (MAX_OPTIONS, len(listed)))
    if not 2 <= max_options <= MAX_OPTIONS:
        raise ValueError("options must be between 2 and %d" % MAX_OPTIONS)
    if not 2 <= calls <= 500:
        raise ValueError("train on between 2 and 500 calls")
    sample = sample_calls(csv_path, calls, seed)
    n_scam = sum(1 for _, y in sample if y)
    log("  question: %s" % stem)
    log("  %d calls from %s (%d scam, %d not), seed %d"
        % (len(sample), csv_path, n_scam, len(sample) - n_scam, seed))
    longest = max((t for t, _ in sample), key=len)
    t0 = time.time()

    answers = []
    if listed:
        opts, source = listed, "listed in the question"
        log("  the question lists its own %d options, so they are kept: "
            "training only counts\n  which calls land on each" % len(opts))
    else:
        log("\n  step 1 of 3: the model answers the question about each call "
            "in a few words")
        for i, (text, y) in enumerate(sample, 1):
            a = short_answer(text, stem)
            answers.append(a)
            log("    %2d/%d  %-5s  %s" % (i, len(sample),
                                          "scam" if y else "legit", a[:90]))
        log("\n  step 2 of 3: the model groups the %d answers into at most %d "
            "options" % (len(answers), max_options))
        opts, source = group_answers(stem, answers, max_options)
        for i, o in enumerate(opts):
            log("    %s  %s" % (LETTERS[i], o))
        if source != "grouped by the model":
            log("    (%s)" % source)

    log("\n  step %s: each call is put back as a multiple choice over the "
        "options" % ("3 of 3" if not listed else "1 of 1"))
    ollama_ctx.fit_num_ctx(_mcq_prompt(longest, stem, opts), ANSWER_TOKENS + 2,
                           where="question training")
    counts = [{"scam": 0, "legit": 0} for _ in opts]
    picked, unread, measured = [], 0, 0
    for i, (text, y) in enumerate(sample, 1):
        probs, choice, how, _, _ = choose(text, stem, opts)
        measured += how == "logprobs"
        picked.append(choice)
        if choice is None:
            unread += 1
            log("    %2d/%d  %-5s  (no readable letter)"
                % (i, len(sample), "scam" if y else "legit"))
            continue
        counts[choice]["scam" if y else "legit"] += 1
        log("    %2d/%d  %-5s  %s %5.1f%%  %s"
            % (i, len(sample), "scam" if y else "legit", LETTERS[choice],
               100 * probs[choice], opts[choice][:70]))

    options = []
    for i, (o, c) in enumerate(zip(opts, counts)):
        options.append({"letter": LETTERS[i], "text": o, "calls": c})
    log("\n  %-3s %-50s %5s %5s" % ("", "option", "scam", "legit"))
    for o in options:
        log("  %-3s %-50s %5d %5d" % (o["letter"], o["text"][:50],
                                      o["calls"]["scam"], o["calls"]["legit"]))
    if unread:
        log("  %d call%s gave no readable letter" % (unread,
                                                    "" if unread == 1 else "s"))
    log("  %.0fs" % (time.time() - t0))
    return {
        "kind": "question",
        "prompt": stem,
        "asked": " ".join(str(question_text).split()),
        "options": options,
        "options_from": source,
        "trained_on": {"dataset": str(csv_path), "calls": len(sample),
                       "scam": n_scam, "legit": len(sample) - n_scam,
                       "seed": seed, "max_options": max_options,
                       "model": llm_judge.DEFAULT_MODEL,
                       "measured": measured, "unreadable": unread,
                       "date": time.strftime("%Y-%m-%d %H:%M")},
        "answers": [{"label": "scam" if y else "legit", "answer": a,
                     "option": None if c is None else LETTERS[c]}
                    for (_, y), a, c in zip(sample, answers or
                                            [None] * len(sample), picked)],
    }


def check_question(obj):
    """Every problem with a saved question, as sentences; [] if none."""
    if not isinstance(obj, dict) or obj.get("kind") != "question":
        return ['not a saved question (it needs "kind": "question")']
    probs = []
    if not str(obj.get("prompt") or "").strip():
        probs.append('"prompt" - the question - is missing or empty')
    opts = obj.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        probs.append('"options" must be a list of 2 to %d options'
                     % MAX_OPTIONS)
    else:
        for i, o in enumerate(opts, 1):
            if not isinstance(o, dict) or not str(o.get("text") or "").strip():
                probs.append('option %d has no "text"' % i)
    return probs


def load_question(path):
    """A saved question, checked, with each option's scam share worked out
    (None when no training call landed on it). ValueError if broken."""
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("no saved question at %s" % path)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON: %s (line %d, column %d)"
                         % (path.name, e.msg, e.lineno, e.colno))
    probs = check_question(obj)
    if probs:
        raise ValueError("%s: %s" % (path.name, "; ".join(probs)))
    obj["prompt"] = " ".join(str(obj["prompt"]).split())
    for i, o in enumerate(obj["options"]):
        o["letter"] = LETTERS[i]
        o["text"] = " ".join(str(o["text"]).split())
        c = o.get("calls") or {}
        s, l = int(c.get("scam") or 0), int(c.get("legit") or 0)
        o["calls"] = {"scam": s, "legit": l}
        o["scam_share"] = s / (s + l) if s + l else None
    return obj


def list_questions():
    """Every saved question in knowledge/questions/, newest first."""
    out = []
    for p in QUESTIONS_DIR.glob("*.json") if QUESTIONS_DIR.is_dir() else []:
        try:
            q = load_question(p)
        except ValueError:
            continue
        t = q.get("trained_on") or {}
        out.append({"path": "knowledge/questions/" + p.name, "name": p.name,
                    "prompt": q["prompt"], "options": len(q["options"]),
                    "dataset": Path(t.get("dataset") or "").name or None,
                    "calls": t.get("calls"), "mtime": p.stat().st_mtime})
    out.sort(key=lambda d: -d["mtime"])
    return out


def ask_saved(transcript, path, timeout=300):
    """Ask a saved question about one call: multiple choice over the options
    in its file. Adds each option's training counts, and `scam_lean` - the
    answer's probabilities weighted by the share of scam calls among the
    training calls on each option (None when no option it put weight on had
    any)."""
    q = load_question(path)
    res = ask_question(transcript, q["prompt"],
                       options=[o["text"] for o in q["options"]],
                       timeout=timeout)
    num = den = 0.0
    for o, a in zip(q["options"], res["options"]):
        a["calls"], a["scam_share"] = o["calls"], o["scam_share"]
        if o["scam_share"] is not None and res["how"] != "unreadable":
            num += a["p"] * o["scam_share"]
            den += a["p"]
    res["scam_lean"] = num / den if den > 0 else None
    res["saved"] = {"path": "knowledge/questions/" + Path(path).name,
                    "trained_on": q.get("trained_on"),
                    "options_from": q.get("options_from")}
    return res


# ----------------------------------------------------------------------- CLI

def run_ask(args):
    onto = load_ontology(args.ontology)
    res = judge(args.text, onto)
    print(onto["prompt"])
    for i, (o, p) in enumerate(zip(onto["options"], res["probs"])):
        mark = "<-" if i == res["choice"] else ""
        print("  %s %5.1f%%  %-5s  %s %s"
              % (LETTERS[i], 100 * p, o["verdict"], o["text"][:70], mark))
    print(explain(res, onto))
    print("verdict  %s" % (res["verdict"] or "unreadable"))
    return 0 if res["verdict"] else 1


def run_question(args):
    if args.saved:
        res = ask_saved(args.text, args.saved)
    elif args.question:
        res = ask_question(args.text, args.question)
    else:
        raise ValueError("pass --question, or --saved with a trained question")
    print(res["question"])
    if res["mode"] == "free":
        print(res["answer"])
        return 0
    for o in res["options"]:
        print("  %s %5.1f%%  %s%s" % (o["letter"], 100 * o["p"], o["text"],
                                      "  <-" if o["letter"] == (
                                          res["choice"] is not None
                                          and LETTERS[res["choice"]]) else ""))
    if res.get("scam_lean") is not None:
        print("  calls in training that gave these answers: %.0f%% scam"
              % (100 * res["scam_lean"]))
    return 0 if res["choice"] is not None else 1


def run_train_question(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit("%s already exists - pass --force to replace it" % out)
    text = (Path(args.question_file).read_text(encoding="utf-8")
            if args.question_file else args.question)
    if not text:
        raise ValueError("pass --question or --question-file")
    print("==> train a question on %s" % args.csv)
    q = train_question(args.csv, text, args.calls, args.options, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
    print("  ok wrote %s" % out)
    return 0


def run_evaluate(args):
    """Score a whole dataset: what the MCQ page's Score a dataset tab runs."""
    import eval_common as EC
    onto = load_ontology(args.ontology)
    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    built = (onto.get("built_from") or {}).get("dataset")
    print()
    print("==> score  %s over %d calls" % (args.ontology, len(rows)))
    print("  %s  (%d options)" % (onto["prompt"], len(onto["options"])))
    if built and Path(built).name == Path(args.csv).name:
        print("  NOTE these options were built from this dataset's own "
              "categories, so the model\n       only has to recognise the "
              "topic - read the score with that in mind.")
    ctx = presize([r["text"] for r in rows], onto)
    print("  one request per call, context window %s" % ctx)

    truths = [r["label"] for r in rows]
    prog = EC.Progress(len(rows), every=1)
    preds, cats, probs, chosen, reasons = [], [], [], [], []
    hows = Counter()
    t0 = time.time()
    try:
        for r, res in zip(rows, judge_all([r["text"] for r in rows], onto)):
            hows[res["how"]] += 1
            pred = None if res["verdict"] is None else int(res["verdict"] == "Fraud")
            preds.append(pred)
            chosen.append(res["choice"])
            cats.append(res["category"] or "")
            probs.append("" if res["p_scam"] is None
                         else round(res["p_scam"], 4))
            reasons.append(explain(res, onto))
            prog.tick(r, pred)
    except RuntimeError as e:
        print("    ERROR %s" % e)
    elapsed = time.time() - t0
    rows = rows[:len(preds)]
    truths = truths[:len(preds)]

    m = EC.metrics(truths, preds)
    base = EC.baselines(truths)
    extra = ["P(scam) measured from logprobs on %d/%d calls"
             % (hows["logprobs"], len(preds))]
    extra += [""] + category_table(onto, chosen, truths)
    EC.report(m, base, elapsed, len(preds), extra)
    if args.out:
        EC.write_results(args.out, Path(args.ontology).name, args.csv, rows,
                         preds, m, base, elapsed,
                         {"category": cats, "prob_scam": probs,
                          "reason": reasons},
                         {"kind": "mcq", "ontology": str(args.ontology),
                          "question": onto["prompt"], "built_from": built,
                          "options": [{"letter": LETTERS[i], "id": o["id"],
                                       "verdict": o["verdict"],
                                       "scam": sum(1 for c, t in zip(chosen, truths)
                                                   if c == i and t),
                                       "legit": sum(1 for c, t in zip(chosen, truths)
                                                    if c == i and not t)}
                                      for i, o in enumerate(onto["options"])],
                          "measured": hows["logprobs"]})
    return m


def run_build(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit("%s already exists - pass --force to replace it" % out)
    print("==> build options from %s" % args.csv)
    onto = build(args.csv, args.column, args.describe, args.examples,
                 args.max_options)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(onto, indent=2) + "\n", encoding="utf-8")
    print("  column %r, %d calls, %d options:"
          % (onto["built_from"]["column"], onto["built_from"]["calls"],
             len(onto["options"])))
    for i, o in enumerate(onto["options"]):
        print("    %s  %-5s  %-30s %5d scam %5d legit%s"
              % (LETTERS[i], o["verdict"], o["text"][:30], o["calls"]["scam"],
                 o["calls"]["legit"], "  (split evenly)" if o.get("mixed")
                 else ""))
    print("  ok wrote %s" % out)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="one transcript")
    a.add_argument("--text", required=True)
    a.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    a.set_defaults(func=run_ask)

    qq = sub.add_parser("question", help="your own question about one call")
    qq.add_argument("--text", required=True)
    qq.add_argument("--question", default=None,
                    help='list options to make it multiple choice: '
                         '"... A) yes B) no", or "options: yes / no"')
    qq.add_argument("--saved", default=None,
                    help="a trained question's JSON (knowledge/questions/...): "
                         "its options are used")
    qq.set_defaults(func=run_question)

    tq = sub.add_parser("train-question",
                        help="options for your question, from a dataset")
    tq.add_argument("--csv", required=True)
    tq.add_argument("--question", default=None)
    tq.add_argument("--question-file", default=None,
                    help="read the question from this file instead")
    tq.add_argument("--out", required=True,
                    help="knowledge/questions/<name>.json")
    tq.add_argument("--calls", type=int, default=TRAIN_CALLS,
                    help="calls to train on, half scam half not "
                         "(default %d)" % TRAIN_CALLS)
    tq.add_argument("--options", type=int, default=TRAIN_OPTIONS,
                    help="at most this many options (default %d)"
                         % TRAIN_OPTIONS)
    tq.add_argument("--seed", type=int, default=42)
    tq.add_argument("--force", action="store_true")
    tq.set_defaults(func=run_train_question)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    ev.add_argument("--csv", required=True)
    ev.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    ev.add_argument("--limit", type=int, default=None,
                    help="a class-balanced head of the dataset")
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    b = sub.add_parser("build", help="options from a dataset's categories")
    b.add_argument("--csv", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--column", default=None,
                   help="the category column (default: the first of %s)"
                        % ", ".join(CATEGORY_COLS))
    b.add_argument("--describe", action="store_true",
                   help="have the model describe each category from a few of "
                        "its calls")
    b.add_argument("--examples", type=int, default=3)
    b.add_argument("--max-options", type=int, default=MAX_OPTIONS)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=run_build)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        out = args.func(args)
    except ValueError as e:
        raise SystemExit("ERROR %s" % e)
    return 0 if out is None or isinstance(out, dict) else out


if __name__ == "__main__":
    sys.exit(main())
