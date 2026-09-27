#!/usr/bin/env python3
"""
Offline tests for llm_prob: the plain LLM judge with P(scam) from logprobs.

No Ollama: llm_judge._post is replaced by a fake that answers the way
Ollama's /api/generate does with logprobs on (and, for the fallback, the way
an older Ollama does without them). Nothing here touches a knowledge base, a
vector database or the web - which is the point of the baseline.

    python scripts/test_llm_prob.py
"""
import contextlib
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_judge                                               # noqa: E402
import llm_prob as LP                                          # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-64s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


# what the fake model believes, keyed on a word in the transcript
BELIEF = {"GIFTCARD": 0.97, "BORDERLINE": 0.55, "DENTIST": 0.03}
STATE = {"logprobs": True, "calls": 0, "payloads": []}


def fake_post(path, payload, timeout):
    STATE["calls"] += 1
    STATE["payloads"].append(payload)
    prompt = payload["prompt"]
    if "GIBBERISH" in prompt:
        alts = [{"token": "The", "logprob": math.log(0.9)}]
        return {"response": "The", "logprobs": [{"token": "The",
                "logprob": alts[0]["logprob"], "top_logprobs": alts}]}
    p = next(v for k, v in BELIEF.items() if k in prompt)
    word = "Scam" if p >= .5 else "Legit"
    if not STATE["logprobs"]:
        return {"response": word}
    # the ways each word can start, plus a little junk
    alts = [{"token": "Sc", "logprob": math.log(p * 0.7)},
            {"token": " Scam", "logprob": math.log(p * 0.2)},
            {"token": "Leg", "logprob": math.log((1 - p) * 0.8)},
            {"token": "Legit", "logprob": math.log((1 - p) * 0.1)},
            {"token": "The", "logprob": math.log(0.1)}]
    return {"response": word, "logprobs": [{"token": word, "logprob": 0.0,
                                            "top_logprobs": alts}]}


llm_judge._post = fake_post

print("\nthe answer word")
for tok, want in (("Sc", "scam"), (" Scam", "scam"), ("**Scam", "scam"),
                  ("Leg", "legit"), ("Legitimate", "legit"), ("S", None),
                  ("The", None)):
    check("%r reads as %r" % (tok, want), LP.answer_word(tok), want)

print("\nP(scam) from logprobs")
p, how, word = LP.scam_probability("GIFTCARD call")
check("measured from logprobs", how, "logprobs")
check("P(scam) is the scam share of the answer mass (0.97)", round(p, 3), 0.97)
p, _, _ = LP.scam_probability("BORDERLINE call")
check("an unsure call reads as unsure (0.55)", round(p, 3), 0.55)
pl = STATE["payloads"][-1]
check("one token, temperature 0, logprobs asked for",
      (pl["options"]["num_predict"], pl["options"]["temperature"],
       pl["logprobs"]), (1, 0.0, True))
check("the prompt is the transcript alone - no evidence, no retrieval",
      "REFERENCE" not in pl["prompt"] and "BORDERLINE call" in pl["prompt"])
check("the prompt ends on the one-word question",
      pl["prompt"].rstrip().endswith("Answer:"))
p, how, word = LP.scam_probability("GIBBERISH call")
check("an answer that is neither word is unreadable", (p, how), (None, "unreadable"))

print("\nan Ollama without logprobs")
STATE["logprobs"] = False
LP._LOGPROBS_OK = None
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    p, how, _ = LP.scam_probability("BORDERLINE call")
check("falls back to the answer word: 1 or 0", (p, how), (1.0, "word"))
check("and says so, once", buf.getvalue().count("no logprobs"), 1)
STATE["logprobs"] = True
LP._LOGPROBS_OK = None

print("\nthe benchmark runner")
sys.argv = sys.argv[:1]
import combined_evaluate as C                                  # noqa: E402
data = [("GIFTCARD call", "Fraud"), ("DENTIST call", "Normal"),
        ("BORDERLINE call", "Fraud"), ("GIBBERISH call", "Fraud")]
STATE["calls"] = 0
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out, raws, reasons, scores = C.run_llm_prob(data)
check("one Ollama call per transcript", STATE["calls"], 4)
check("verdicts at 0.5", [p for p, _ in out],
      ["Fraud", "Normal", "Fraud", "Normal"])
check("scores are P(scam) in percent, unreadable left blank",
      scores, [97.0, 3.0, 55.0, None])
check("the log counts how P(scam) was measured",
      "measured from logprobs on 3/4 calls" in buf.getvalue()
      and "1 unreadable" in buf.getvalue())
r = C.score_ranges(out, scores)
check("the band table places each scored call", (r["TP"][9], r["TN"][0],
                                                 r["TP"][5]), (1, 1, 1))

print("\n" + ("all good - P(scam) is measured, from the transcript alone"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
