#!/usr/bin/env python3
"""
llm_novelty.py - the plain LLM judge, scoring each call by how novel it is.

No retrieval of any kind: no knowledge base, no vector database, no web search.
The transcript goes to the model on its own, as in llm_only, and it is asked
two one-token questions on the same prompt:

  1. the verdict - one word, Scam or Legit. The call is Fraud when the model
     puts at least half its probability on "Scam".
  2. novelty - how familiar the call's pattern is, as one letter:
        A  textbook: a very common, well-known script              0
        B  familiar: a common pattern with small variations       25
        C  somewhat unusual                                       50
        D  unusual: an uncommon approach                          75
        E  novel: unlike calls the model has seen                100
     The score is the probability-weighted average of those five values,
     read from the logprobs Ollama returns - so it is measured, continuous
     0-100, not a round number the model types.

Novelty is scored for every call, scam or not, and says nothing about
whether it is a scam: it is there to see whether the calls the model gets
wrong are the unfamiliar ones.

An Ollama too old to return logprobs still answers both questions: the
verdict is then the word, novelty the chosen letter's value (5 levels only),
and the run says so.

    python scripts/llm_novelty.py --text "Hello, this is your bank's fraud team..."
"""

import argparse
import math
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_judge                                               # noqa: E402
import ollama_ctx                                              # noqa: E402

# Both questions share everything up to and including the transcript, so
# Ollama can reuse the processed prompt for the second one.
HEAD = """You are a scam detection analyst. Read this phone call transcript.

Transcript:
{transcript}

"""

VERDICT_Q = """Is the caller attempting a scam? Answer with exactly one word: Scam or Legit.
Answer:"""

NOVELTY_Q = """Set aside whether it is a scam. How familiar is the pattern of this call -
what the caller wants and how they go about it?
A - textbook: a very common, well-known script
B - familiar: a common pattern with small variations
C - somewhat unusual
D - unusual: an uncommon approach
E - novel: unlike calls you have seen before
Answer with exactly one letter, A to E.
Answer:"""

LEVELS = {"A": 0.0, "B": 25.0, "C": 50.0, "D": 75.0, "E": 100.0}
TOP_LOGPROBS = 20
_LOGPROBS_OK = None          # None = not tried yet; False = this Ollama has none


def _note_no_logprobs():
    global _LOGPROBS_OK
    if _LOGPROBS_OK is None:
        _LOGPROBS_OK = False
        print("    NOTE this Ollama returns no logprobs (it needs a recent "
              "version): the verdict is the answer word and novelty the "
              "chosen letter's value, so novelty has only 5 levels")


def answer_word(token):
    """'scam', 'legit' or None for one candidate first token of the answer.
    Qwen may split a word ("Sc" + "am") or lead with a space or markdown."""
    t = token.strip().strip("*\"'`").lower()
    if len(t) < 2:
        return None
    if "scam".startswith(t) or t.startswith("scam"):
        return "scam"
    if "legit".startswith(t) or t.startswith("legit"):
        return "legit"
    return None


def answer_letter(token):
    """'A'..'E' or None: a lone letter, perhaps with a space, markdown or a
    trailing bracket or full stop."""
    t = token.strip().strip("*\"'`().:").upper()
    return t if t in LEVELS else None


def generate_one(prompt, top=TOP_LOGPROBS, timeout=300):
    """One token, and the alternatives Ollama weighed for it:
    (text, [(token, logprob), ...]). The list is empty when this Ollama does
    not return logprobs."""
    num_ctx = ollama_ctx.fit_num_ctx(prompt, 4, where="llm_novelty")
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


def verdict(transcript):
    """("Fraud" | "Normal" | None, the word answered). None when the answer is
    neither word."""
    global _LOGPROBS_OK
    text, alts = generate_one(HEAD.format(transcript=transcript) + VERDICT_Q)
    mass = {"scam": 0.0, "legit": 0.0}
    for tok, lp in alts:
        w = answer_word(tok)
        if w:
            mass[w] += math.exp(lp)
    if alts:
        _LOGPROBS_OK = True
    else:
        _note_no_logprobs()
    word = answer_word(text)
    if mass["scam"] + mass["legit"] > 0:
        return ("Fraud" if mass["scam"] >= mass["legit"] else "Normal"), word
    if word:
        return ("Fraud" if word == "scam" else "Normal"), word
    return None, text


def novelty(transcript):
    """(novelty 0..100 or None, how, the letter answered).

    how is "logprobs" when it was measured, "letter" when this Ollama gives no
    logprobs and the chosen letter's value stands in, and "unreadable" when
    the answer was no letter A-E."""
    global _LOGPROBS_OK
    text, alts = generate_one(HEAD.format(transcript=transcript) + NOVELTY_Q)
    mass = dict.fromkeys(LEVELS, 0.0)
    for tok, lp in alts:
        k = answer_letter(tok)
        if k:
            mass[k] += math.exp(lp)
    if alts:
        _LOGPROBS_OK = True
    else:
        _note_no_logprobs()
    letter = answer_letter(text)
    total = sum(mass.values())
    if total > 0:
        return (sum(LEVELS[k] * m for k, m in mass.items()) / total,
                "logprobs", letter)
    if letter:
        return LEVELS[letter], "letter", letter
    return None, "unreadable", text


def main():
    ap = argparse.ArgumentParser(description="verdict and novelty for one transcript")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    v, word = verdict(args.text)
    n, how, letter = novelty(args.text)
    print("verdict  %s  (answered %r)" % (v or "unreadable", word))
    print("novelty  %s  (%s, answered %r)"
          % ("%.1f" % n if n is not None else "unreadable", how, letter))
    return 0


if __name__ == "__main__":
    sys.exit(main())
