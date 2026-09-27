#!/usr/bin/env python3
"""
Offline tests for llm_novelty: the Web-RAG baseline is now the plain LLM
judge, each call scored 0-100 by how novel it is.

No Ollama: llm_judge._post is replaced by a fake that answers the way
Ollama's /api/generate does with logprobs on (and, for the fallback, the way
an older Ollama does without them). Nothing here touches a knowledge base, a
vector database or the web - which is the point of the baseline.

    python scripts/test_llm_novelty.py
"""
import contextlib
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_judge                                               # noqa: E402
import llm_novelty as LN                                       # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-64s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


# what the fake model believes, keyed on a word in the transcript:
# (P(scam), probability over the novelty letters A-E)
BELIEF = {
    "GIFTCARD": (0.97, {"A": .8, "B": .2}),                 # textbook scam
    "DENTIST": (0.03, {"A": .6, "B": .4}),                  # ordinary call
    "CRYPTOPIG": (0.45, {"C": .2, "D": .5, "E": .3}),       # novel, missed
}
STATE = {"logprobs": True, "calls": 0, "payloads": []}


def fake_post(path, payload, timeout):
    STATE["calls"] += 1
    STATE["payloads"].append(payload)
    prompt = payload["prompt"]
    if "GIBBERISH" in prompt:
        return {"response": "The", "logprobs": [{"token": "The", "logprob": 0.0,
                "top_logprobs": [{"token": "The", "logprob": 0.0}]}]}
    p, letters = next(v for k, v in BELIEF.items() if k in prompt)
    if prompt.rstrip().endswith("A to E.\nAnswer:"):
        alts = [{"token": (" " if k == "B" else "") + k, "logprob": math.log(v)}
                for k, v in letters.items()]
        alts.append({"token": "The", "logprob": math.log(0.05)})   # junk
    else:
        alts = [{"token": "Sc", "logprob": math.log(p * 0.8)},
                {"token": " Scam", "logprob": math.log(p * 0.2)},
                {"token": "Leg", "logprob": math.log(1 - p)}]
    top = max(alts, key=lambda a: a["logprob"])
    if not STATE["logprobs"]:
        return {"response": top["token"]}
    return {"response": top["token"], "logprobs": [{"token": top["token"],
            "logprob": top["logprob"], "top_logprobs": alts}]}


llm_judge._post = fake_post

print("\nreading the answers")
for tok, want in (("Sc", "scam"), (" Scam", "scam"), ("Leg", "legit"),
                  ("S", None), ("The", None)):
    check("word %r reads as %r" % (tok, want), LN.answer_word(tok), want)
for tok, want in (("A", "A"), (" B", "B"), ("**E", "E"), ("C)", "C"),
                  ("F", None), ("The", None)):
    check("letter %r reads as %r" % (tok, want), LN.answer_letter(tok), want)

print("\nnovelty from logprobs")
n, how, letter = LN.novelty("GIFTCARD call")
check("measured from logprobs", how, "logprobs")
check("a textbook scam is low novelty (0.8*0 + 0.2*25 = 5)", round(n, 2), 5.0)
n, _, _ = LN.novelty("CRYPTOPIG call")
check("a novel call is high novelty (.2*50 + .5*75 + .3*100 = 77.5)",
      round(n, 2), 77.5)
pl = STATE["payloads"][-1]
check("one token, temperature 0, logprobs asked for",
      (pl["options"]["num_predict"], pl["options"]["temperature"],
       pl["logprobs"]), (1, 0.0, True))
check("the prompt is the transcript alone - no evidence, no retrieval",
      "REFERENCE" not in pl["prompt"] and "CRYPTOPIG call" in pl["prompt"])
check("novelty is asked apart from the scam question",
      "Set aside whether it is a scam" in pl["prompt"])
check("both questions share the prompt head (for Ollama's cache)",
      LN.HEAD.format(transcript="x") in LN.HEAD.format(transcript="x") + LN.VERDICT_Q
      and pl["prompt"].startswith(LN.HEAD.format(transcript="CRYPTOPIG call")))

print("\nthe verdict")
check("a scam is Fraud", LN.verdict("GIFTCARD call")[0], "Fraud")
check("P(scam) 0.45 is Normal", LN.verdict("CRYPTOPIG call")[0], "Normal")
check("neither word is unreadable", LN.verdict("GIBBERISH call")[0], None)
check("no letter is unreadable novelty",
      LN.novelty("GIBBERISH call")[:2], (None, "unreadable"))

print("\nan Ollama without logprobs")
STATE["logprobs"] = False
LN._LOGPROBS_OK = None
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    n, how, letter = LN.novelty("CRYPTOPIG call")
    v, _ = LN.verdict("GIFTCARD call")
check("novelty falls back to the letter's value", (n, how, letter),
      (75.0, "letter", "D"))
check("the verdict falls back to the word", v, "Fraud")
check("and it says so, once", buf.getvalue().count("no logprobs"), 1)
STATE["logprobs"] = True
LN._LOGPROBS_OK = None

print("\nthe benchmark runner")
sys.argv = sys.argv[:1]
import combined_evaluate as C                                  # noqa: E402
data = [("GIFTCARD call", "Fraud"), ("DENTIST call", "Normal"),
        ("CRYPTOPIG call", "Fraud"), ("GIBBERISH call", "Fraud")]
STATE["calls"] = 0
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out, raws, reasons, scores = C.run_llm_novelty(data)
log = buf.getvalue()
check("two one-token Ollama calls per transcript", STATE["calls"], 8)
check("verdicts", [p for p, _ in out], ["Fraud", "Normal", "Normal", "Normal"])
check("scores are novelty 0-100, unreadable left blank",
      scores, [5.0, 10.0, 77.5, None])
check("the log compares novelty of right and wrong calls",
      "median novelty: 5.0 on the 2 calls it got right, 77.5 on the 1" in log)
check("a novelty system writes a _novelty column, not _pct",
      (C.is_novelty("webrag"), C.is_novelty("webrag__stripped"),
       C.is_novelty("hybrid")), (True, True, False))

print("\n" + ("all good - novelty is measured, from the transcript alone"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
