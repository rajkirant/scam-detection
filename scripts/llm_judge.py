#!/usr/bin/env python3
"""
llm_judge.py - put one transcript to the local LLM and get back a verdict and
the reason for it.

This is the `llm_only` baseline out of combined_evaluate.py, lifted out of the
benchmark loop so it can be pointed at a single call: no retrieval, no
ontology, no fine-tuned anything - the model reads the transcript and says
Fraud or Normal, then says why. It is the control the other systems on the
Benchmark page are measured against, and it is what the LLM judge page in
web_ui.py runs.

The prompt and the parsing are imported from combined_evaluate rather than
copied, so a verdict here is the same verdict the benchmark would record for
the same call. What differs is the one place where a benchmark and a person
want opposite things: when the reply cannot be read, the benchmark has to
score something and settles for Normal, which quietly turns an unreadable
answer into a legitimate call. Here it is reported as unreadable, because
nobody reading one call wants "Normal" to sometimes mean "the model did not
answer".

Ollama is spoken to over plain HTTP with urllib rather than through
credibility.call_ollama, which needs `requests`: web_ui.py runs on the system
python so it can start without the venv, and this has to work there too.

--guidance is where a correction goes when the model gets one wrong: standing
instructions that are put in the prompt, fenced off from the transcript, every
time it is asked. Nothing is stored and nothing is learned by it - a model
told the same thing twice is being told, not taught - and a verdict reached
under instructions is flagged `guided`, because it is no longer the llm_only
control.

Usage:
    python scripts/llm_judge.py --text "Hello, this is your bank..."
    python scripts/llm_judge.py --csv datasets/scambait_bank_422.csv --idx 3
    python scripts/llm_judge.py --text "..." --model qwen2.5:14b --json
    python scripts/llm_judge.py --text "..." \
        --guidance "A bank asking to confirm a card number is normal here."
    python scripts/llm_judge.py models          # what ollama has pulled
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# OLLAMA_URL is what web_ui.py already probes with, so it wins; OLLAMA_HOST is
# the variable ollama's own CLI reads, so a box that only sets that one works
# too.
OLLAMA_HOST = (os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
               or "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
DEFAULT_MODEL = os.environ.get("SCAM_MODEL", "qwen2.5:14b")

# Ollama truncates a prompt that overflows the context window from the FRONT,
# which drops the instructions and leaves the model staring at a headless wall
# of transcript - so the window is set explicitly rather than left to whatever
# the model's default happens to be.
DEFAULT_NUM_CTX = int(os.environ.get("QWEN_NUM_CTX", "8192"))
DEFAULT_MAX_TOKENS = 300
MAX_CHARS = 400_000


def _post(path, payload, timeout):
    req = urllib.request.Request(
        OLLAMA_HOST.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _unreachable(e):
    return RuntimeError(
        "cannot reach ollama at %s (%s). Start it with `ollama serve`, and "
        "check the model is pulled with `ollama list`." % (OLLAMA_HOST, e))


def _get(path, timeout=10):
    req = urllib.request.Request(OLLAMA_HOST.rstrip("/") + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def list_models():
    """What ollama has pulled, newest first. An empty list means ollama
    answered and has nothing; an error means it did not answer."""
    try:
        out = _get("/api/tags")
    except Exception as e:
        raise _unreachable(e)
    models = [{"name": m.get("name") or m.get("model", ""),
               "size": m.get("size"),
               "modified": m.get("modified_at")}
              for m in out.get("models", [])]
    models.sort(key=lambda m: m.get("modified") or "", reverse=True)
    return models


def ask(prompt, model=None, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.0,
        num_ctx=DEFAULT_NUM_CTX, timeout=300):
    """The model's reply, as text."""
    return (generate(prompt, model, max_tokens, temperature, num_ctx,
                     timeout).get("response") or "").strip()


def generate(prompt, model=None, max_tokens=DEFAULT_MAX_TOKENS,
             temperature=0.0, num_ctx=DEFAULT_NUM_CTX, timeout=300):
    """One completion out of ollama, whole. Temperature 0 by default: this is
    a judgement, and the benchmark asks for it the same way.

    The reply carries prompt_eval_count, which is how many tokens the model
    actually read. That number is the only non-guessed way to tell whether a
    prompt was truncated: everything else here is an estimate, and Ollama
    truncates without saying so.
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if num_ctx:
        payload["options"]["num_ctx"] = int(num_ctx)
    try:
        out = _post("/api/generate", payload, timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        if e.code == 404:
            raise RuntimeError(
                "ollama has no model called %r - pull it first: `ollama pull "
                "%s`" % (payload["model"], payload["model"]))
        raise RuntimeError("ollama refused that (%s): %s" % (e.code, body))
    except urllib.error.URLError as e:
        raise _unreachable(e.reason)
    except TimeoutError:
        raise RuntimeError(
            "ollama did not answer within %ds. A 14B model on CPU is far "
            "slower than on a GPU - check `nvidia-smi`, or raise the "
            "timeout." % timeout)
    return out


MAX_GUIDANCE = 20_000

OPENING = ("You are a scam detection analyst. Read this phone call "
           "transcript and decide whether the caller is attempting a scam.")


def build_prompt(transcript, guidance=None, profile=None):
    """The benchmark's llm_only prompt, with anything fitted folded in.

    With no guidance and no profile the string is byte for byte what
    combined_evaluate sends, which is what keeps a verdict here comparable to
    a benchmark row. That property is worth protecting: the moment either is
    in play this is no longer the llm_only control, and the reply says so.

    The order of the parts is not cosmetic. Ollama truncates an overlong
    prompt from the FRONT, so everything is laid out worst-to-best: the
    worked examples first, because losing them costs calibration; then the
    transcript; then the rules and the answer format, which are the last
    things that may be lost. A fitted prompt that overflows degrades into the
    unfitted one rather than into a headless wall of transcript.

    Every fitted part is fenced and labelled as material for the analyst
    rather than as speech. A transcript is full of people telling each other
    what to do, and a model that cannot tell an instruction from the call it
    is reading will start taking orders from the caller.
    """
    import combined_evaluate as CE

    profile = profile or {}
    shots = profile.get("shots") or []
    rubric = (profile.get("rubric") or "").strip()

    prompt = OPENING + "\n\n"
    if shots:
        prompt += (
            "Worked examples: past calls from this corpus with the answer "
            "already known. They are for calibration only - none of them is "
            "the call you are being asked about.\n<<<EXAMPLES\n")
        for i, sh in enumerate(shots, 1):
            prompt += ("[%d] Answer: %s\n%s\n\n"
                       % (i, sh.get("verdict", "Normal"),
                          (sh.get("excerpt") or "").strip()))
        prompt += "EXAMPLES>>>\n\n"
    if rubric:
        prompt += (
            "What separates the two classes in this corpus, written from the "
            "labelled calls above. Treat it as a guide to this corpus, not as "
            "a rule that overrides what you read:\n"
            "<<<RUBRIC\n%s\nRUBRIC>>>\n\n" % rubric)

    prompt += "Transcript:\n%s\n\n" % transcript

    # The analyst's own corrections go last of all, because they are the one
    # thing here a person typed on purpose.
    for text, head in ((profile.get("guidance"),
                        "Standing instructions carried in the fitted prompt"),
                       (guidance,
                        "Standing instructions from the analyst")):
        if (text or "").strip():
            prompt += (
                "%s. These are rules for you to apply, not part of the call, "
                "and they take precedence over your own defaults where the "
                "two disagree:\n<<<INSTRUCTIONS\n%s\nINSTRUCTIONS>>>\n\n"
                % (head, text.strip()))
    return prompt + CE.VERDICT_FORMAT


def estimate_tokens(text):
    """Shared with every other Ollama client in the project, so the number the
    page shows is the number the benchmark sizes its windows by."""
    import ollama_ctx
    return ollama_ctx.estimate_tokens(text)


def judge(transcript, model=None, max_tokens=DEFAULT_MAX_TOKENS,
          num_ctx=DEFAULT_NUM_CTX, temperature=0.0, timeout=300,
          guidance=None, profile=None):
    """Scam or not, and why, for one transcript.

    Mirrors combined_evaluate.ask_verdict, including its retry: when the first
    reply cannot be parsed the model is asked again for one bare word. The
    reason always comes from the FIRST reply, because the retry is told not to
    explain itself and so has none to give.

    `guidance` is standing instructions to apply to this call - the place to
    put a correction after the model gets one wrong. Nothing is stored and
    nothing is learned: the text is put in the prompt, every time, and a model
    told the same thing twice is being told, not taught.

    `profile` is a fitted prompt out of llm_fit.py: worked examples drawn from
    a dataset, a rubric the model wrote from them, or both. The same applies -
    the weights do not move, the prompt gets longer. What it buys is measured
    at fit time against this same function with profile=None, which is the
    only way to know whether it bought anything.
    """
    import combined_evaluate as CE

    text = (transcript or "").strip()
    if not text:
        raise ValueError("there is no transcript to judge")
    if len(text) > MAX_CHARS:
        raise ValueError("that is longer than any call in the datasets - "
                         "paste one call, not a whole file")
    guidance = (guidance or "").strip()
    if len(guidance) > MAX_GUIDANCE:
        raise ValueError("that is a lot of instructions - keep them under "
                         "%d characters" % MAX_GUIDANCE)

    prompt = build_prompt(text, guidance, profile)

    t0 = time.time()
    out = generate(prompt, model=model, max_tokens=max_tokens,
                   temperature=temperature, num_ctx=num_ctx, timeout=timeout)
    raw = (out.get("response") or "").strip()
    read = out.get("prompt_eval_count")
    verdict = CE.parse_verdict(raw)
    reason = CE.parse_reason(raw)
    truncated = CE.looks_truncated(raw)
    retried = False

    if verdict is None:
        retried = True
        raw2 = ask(prompt + "\n\nReply with exactly one word, either Fraud or "
                            "Normal. No explanation.",
                   model=model, max_tokens=10, temperature=temperature,
                   num_ctx=num_ctx, timeout=timeout)
        verdict = CE.parse_verdict(raw2)
        raw = raw + "\n\n--- retry ---\n" + raw2

    prompt_tokens = estimate_tokens(prompt)
    return {
        "ok": True,
        "model": model or DEFAULT_MODEL,
        "verdict": verdict,                      # "Fraud", "Normal", or None
        "scam": None if verdict is None else (verdict == "Fraud"),
        "unreadable": verdict is None,
        "reason": reason or ("(the model gave no reason)" if verdict
                             else "(the model's answer could not be read)"),
        "raw": raw,
        "retried": retried,
        "truncated": truncated,
        # a verdict reached under instructions or a fitted prompt is not the
        # llm_only control any more, and anything reading this reply should be
        # able to tell
        "guided": bool(guidance),
        "guidance": guidance,
        "profile": (profile or {}).get("name"),
        "shots": len((profile or {}).get("shots") or []),
        "rubric": bool(((profile or {}).get("rubric") or "").strip()),
        "fitted": bool(profile),
        "words": len(text.split()),
        "prompt_tokens_estimated": prompt_tokens,
        # What Ollama says it actually read: the only number here that is not
        # a guess, and the one that shows how much of the call survived.
        #
        # It does NOT come back level with the window when a prompt overflows.
        # Measured on qwen2.5:14b, an overlong prompt reads back at exactly
        # num_ctx/2 + 2 - 4098 of 8192, 1026 of 2048 - because Ollama keeps
        # about half the window and throws the rest away, from the front. So
        # a window merely close to the prompt size is not enough: missing by
        # one token costs half the window, not one token.
        "prompt_tokens_read": read,
        # Capped at 100: the estimate runs a few percent low against a real
        # tokenizer (15,933 read against 15,293 estimated, on the call this
        # was calibrated on), and "kept 104%" reads like a bug rather than
        # like a heuristic being a heuristic.
        "prompt_kept_pct": (min(100.0, round(100.0 * read / prompt_tokens, 1))
                            if read and prompt_tokens else None),
        # Certain when the prompt could not have fitted. A prompt that did fit
        # can also read short, which is a cached prefix rather than a loss.
        "prompt_truncated": bool(read and num_ctx and prompt_tokens > num_ctx
                                 and read < prompt_tokens),
        "num_ctx": num_ctx,
        # the failure that looks like a bad model rather than a bad setting:
        # over the window, ollama drops the FRONT of the prompt, instructions
        # and all, and the reply that comes back is answering something else
        "over_context": bool(num_ctx and prompt_tokens > num_ctx),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


def transcript_from_csv(path, idx=None, row_id=None):
    import csv
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("that dataset is empty")
    cols = {c.lower(): c for c in rows[0]}
    tcol = next((cols[c] for c in ("transcript", "text", "call", "conversation",
                                   "dialogue", "content", "body") if c in cols),
                None)
    if tcol is None:
        raise SystemExit("no transcript column in that dataset")
    if row_id is not None:
        idcol = next((cols[c] for c in ("id", "call_id", "conv_id")
                      if c in cols), None)
        if idcol is None:
            raise SystemExit("that dataset has no id column")
        hit = [r for r in rows if str(r[idcol]) == str(row_id)]
        if not hit:
            raise SystemExit("no row with id %s" % row_id)
        return hit[0][tcol]
    return rows[int(idx or 0)][tcol]


def print_result(r):
    bar = "=" * 74
    print(bar)
    if r["unreadable"]:
        head = "UNREADABLE - the model did not give an answer that can be read"
    else:
        head = "SCAM" if r["scam"] else "LEGITIMATE"
    print("%s   (%s)" % (head, r["model"]))
    print(bar)
    print("  reason     %s" % r["reason"])
    if r.get("profile"):
        print("  prompt     fitted: models/%s, %d worked example(s)%s"
              % (r["profile"], r["shots"],
                 " and a rubric" if r.get("rubric") else ""))
    if r["guided"] or r.get("fitted"):
        print("  note       judged under %s, so this is not the\n"
              "             llm_only control any more"
              % ("a fitted prompt" if r.get("fitted") and not r["guided"]
                 else "your standing instructions"))
    if r["retried"]:
        print("  note       the first reply could not be parsed; the model was "
              "asked again for one word")
    if r["truncated"]:
        print("  note       the reply looks cut off - raise --max-tokens")
    if r["over_context"]:
        print("  WARNING    the prompt is about %d tokens and the window is "
              "%d.\n             Ollama drops the FRONT of an overlong "
              "prompt, instructions\n             and all - raise --num-ctx."
              % (r["prompt_tokens_estimated"], r["num_ctx"]))
    if r["prompt_truncated"]:
        print("  TRUNCATED  ollama read %d of the ~%d tokens sent - %.0f%% of "
              "this call was\n             thrown away, from the beginning. "
              "The verdict above is about\n             whatever was left, "
              "which on a long call is the goodbyes.\n"
              "             Note it read about half the %d window, not all of "
              "it: a window\n             merely close to the prompt is no "
              "use, it has to exceed it."
              % (r["prompt_tokens_read"], r["prompt_tokens_estimated"],
                 100 - (r["prompt_kept_pct"] or 0), r["num_ctx"]))
    print("  %d words, ~%d prompt tokens estimated%s, window %d, %d ms"
          % (r["words"], r["prompt_tokens_estimated"],
             (", %d read by ollama (%.0f%%)"
              % (r["prompt_tokens_read"], r["prompt_kept_pct"] or 0))
             if r["prompt_tokens_read"] else "",
             r["num_ctx"], r["elapsed_ms"]))
    print(bar)


def main():
    ap = argparse.ArgumentParser(
        description="ask the local LLM whether one call is a scam, and why")
    sub = ap.add_subparsers(dest="mode")

    ls = sub.add_parser("models", help="what ollama has pulled")
    ls.add_argument("--json", action="store_true")

    ap.add_argument("--text", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--guidance", default=None,
                    help="standing instructions to apply to this call - where "
                         "a correction goes after the model gets one wrong. "
                         "Nothing is stored; it goes in the prompt each time")
    ap.add_argument("--guidance-file", default=None,
                    help="read the standing instructions from a file instead")
    ap.add_argument("--profile", default=None,
                    help="a fitted prompt from llm_fit.py, by name - its "
                         "worked examples and rubric go in the prompt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.mode == "models":
        try:
            models = list_models()
        except RuntimeError as e:
            raise SystemExit(str(e))
        if args.json:
            print(json.dumps(models, indent=2))
        elif not models:
            print("  ollama has nothing pulled yet:  ollama pull qwen2.5:14b")
        else:
            for m in models:
                gb = (m["size"] or 0) / 1e9
                print("  %-28s %5.1f GB" % (m["name"], gb))
        return

    text = args.text
    if not text and args.csv:
        text = transcript_from_csv(args.csv, args.idx, args.id)
    if not text:
        raise SystemExit("pass --text, or --csv with --idx/--id")

    guidance = args.guidance
    if args.guidance_file:
        guidance = Path(args.guidance_file).read_text(encoding="utf-8")

    profile = None
    if args.profile:
        import llm_fit
        profile = llm_fit.load_profile(args.profile)

    try:
        r = judge(text, model=args.model, max_tokens=args.max_tokens,
                  num_ctx=args.num_ctx, temperature=args.temperature,
                  timeout=args.timeout, guidance=guidance, profile=profile)
    except RuntimeError as e:
        raise SystemExit(str(e))
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_result(r)


if __name__ == "__main__":
    main()
