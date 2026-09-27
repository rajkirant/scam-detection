#!/usr/bin/env python3
"""
llm_prob.py - the plain LLM judge, with P(scam) measured rather than asked for.

No retrieval of any kind: no knowledge base, no vector database, no web search.
The transcript goes to the model on its own - exactly what llm_only does - and
the model answers one word, Scam or Legit. P(scam) is read from the
probability the model puts on each word, via the logprobs Ollama returns.

Why measure it: a score the model writes out ("Confidence: 75") is a number it
picks as text, and at temperature 0 it picks the same few round numbers again
and again. The split of probability between the two answer words is the
model's actual uncertainty, and it varies continuously from call to call - so
the benchmark's Scam chance plot shows where the right and wrong calls really
sit.

An Ollama too old to return logprobs still answers the word; P(scam) is then
1 or 0 from that word, and the run says so.

    python scripts/llm_prob.py --text "Hello, this is your bank's fraud team..."
"""

import argparse
import math
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_judge                                               # noqa: E402
import ollama_ctx                                              # noqa: E402

PROMPT = """You are a scam detection analyst. Read this phone call transcript and decide
whether the caller is attempting a scam.

Transcript:
{transcript}

Is the caller attempting a scam? Answer with exactly one word: Scam or Legit.
Answer:"""

TOP_LOGPROBS = 20
_LOGPROBS_OK = None          # None = not tried yet; False = this Ollama has none


def answer_word(token):
    """'scam', 'legit' or None for one candidate first token of the answer.
    Qwen may split a word ("Sc" + "am") or lead with a space or markdown, so a
    token counts when it is the start of one answer word and cannot be the
    start of the other."""
    t = token.strip().strip("*\"'`").lower()
    if len(t) < 2:
        return None
    if "scam".startswith(t) or t.startswith("scam"):
        return "scam"
    if "legit".startswith(t) or t.startswith("legit"):
        return "legit"
    return None


def generate_one(prompt, top=TOP_LOGPROBS, timeout=300):
    """One token, and the alternatives Ollama weighed for it:
    (text, [(token, logprob), ...]). The list is empty when this Ollama does
    not return logprobs."""
    num_ctx = ollama_ctx.fit_num_ctx(prompt, 4, where="llm_prob")
    payload = {
        "model": llm_judge.DEFAULT_MODEL, "prompt": prompt, "stream": False,
        "logprobs": True, "top_logprobs": top,
        "options": {"temperature": 0.0, "num_predict": 1,
                    "num_ctx": int(num_ctx)},
    }
    try:
        body = llm_judge._post("/api/generate", payload, timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError("ollama refused that (%s): %s"
                           % (e.code, e.read().decode("utf-8", "replace")[:300]))
    except OSError as e:           # URLError, a refused connection, a timeout
        raise llm_judge._unreachable(e)
    first = (body.get("logprobs") or [None])[0] or {}
    alts = [(a.get("token", ""), a["logprob"])
            for a in (first.get("top_logprobs") or [])
            if a.get("logprob") is not None]
    if not alts and first.get("logprob") is not None:
        alts = [(first.get("token", ""), first["logprob"])]
    return body.get("response", ""), alts


def scam_probability(transcript):
    """(P(scam) in 0..1 or None, how, the word it answered).

    how is "logprobs" when P(scam) was measured, "word" when this Ollama gives
    no logprobs and the answer word alone (1 or 0) had to stand in, and
    "unreadable" when the model answered neither word."""
    global _LOGPROBS_OK
    text, alts = generate_one(PROMPT.format(transcript=transcript))
    word = answer_word(text)
    mass = {"scam": 0.0, "legit": 0.0}
    for tok, lp in alts:
        w = answer_word(tok)
        if w:
            mass[w] += math.exp(lp)
    if alts:
        _LOGPROBS_OK = True
    elif _LOGPROBS_OK is None:
        _LOGPROBS_OK = False
        print("    NOTE this Ollama returns no logprobs (it needs a recent "
              "version): P(scam) is 1 or 0 from the answer word alone, so the "
              "plot will only show the two ends")
    total = mass["scam"] + mass["legit"]
    if total > 0:
        return mass["scam"] / total, "logprobs", word
    if word:
        return (1.0 if word == "scam" else 0.0), "word", word
    return None, "unreadable", text


def main():
    ap = argparse.ArgumentParser(description="P(scam) for one transcript")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    p, how, word = scam_probability(args.text)
    if p is None:
        print("unreadable answer: %r" % word)
        return 1
    print("P(scam) %.1f%%  (%s, answered %r)" % (100 * p, how, word))
    return 0


if __name__ == "__main__":
    sys.exit(main())
