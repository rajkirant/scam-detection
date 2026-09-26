#!/usr/bin/env python3
"""
Offline tests for Web-RAG adaptive: P(scam) from logprobs, and the web search
only for the calls the model is unsure about.

No Ollama, no chromadb, no Tavily: requests.post is replaced by a fake that
answers the way Ollama's /api/generate (with logprobs) and Tavily's /search
do, and chromadb is stubbed so webrag_system imports.

    python scripts/test_webrag_adaptive.py
"""
import json
import math
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# chromadb is only needed for the real KB; stub it so the module imports
fake_chroma = types.ModuleType("chromadb")
fake_chroma.utils = types.ModuleType("chromadb.utils")
fake_chroma.utils.embedding_functions = types.ModuleType("embedding_functions")
sys.modules.setdefault("chromadb", fake_chroma)
sys.modules.setdefault("chromadb.utils", fake_chroma.utils)
sys.modules.setdefault("chromadb.utils.embedding_functions",
                       fake_chroma.utils.embedding_functions)

import webrag_system as W                                      # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-66s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


class Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


# What the fake model believes about each call, keyed on a word in it.
# (P(scam) from the KB alone, P(scam) once web evidence is in the prompt)
BELIEF = {"GIFTCARD": (0.97, 0.97), "BORDERLINE": (0.55, 0.92),
          "DENTIST": (0.03, 0.03), "UNSURE_NOWEB": (0.40, 0.40)}
CALLS = {"generate": 0, "logprob_generate": 0, "tavily": 0}
LOGPROBS = {"on": True}


def fake_post(url, json=None, timeout=None):
    if "tavily" in url:
        CALLS["tavily"] += 1
        return Resp({"results": [{"title": "Gift card scam warning",
                                  "url": "https://www.ftc.gov/x",
                                  "content": "Scammers ask for gift cards.",
                                  "published_date": None, "score": 0.9}]})
    prompt = json["prompt"]
    if json.get("logprobs"):
        CALLS["logprob_generate"] += 1
        key = next(k for k in BELIEF if k in prompt)
        p = BELIEF[key][1 if "Gift card scam warning" in prompt else 0]
        if not LOGPROBS["on"]:
            return Resp({"response": "Scam" if p >= .5 else "Legit"})
        # Qwen splits "Scam" and leads with a space sometimes; spread the
        # probability over the ways each word can start, plus some junk
        alts = [{"token": "Sc", "logprob": math.log(p * 0.7)},
                {"token": " Scam", "logprob": math.log(p * 0.2)},
                {"token": "Leg", "logprob": math.log((1 - p) * 0.8)},
                {"token": "Legit", "logprob": math.log((1 - p) * 0.1)},
                {"token": "The", "logprob": math.log(0.1)}]
        top = max(alts, key=lambda a: a["logprob"])
        return Resp({"response": top["token"],
                     "logprobs": [{"token": top["token"],
                                   "logprob": top["logprob"],
                                   "top_logprobs": alts}]})
    CALLS["generate"] += 1
    if "Confidence:" in prompt and "Respond in exactly this format" in prompt:
        key = next(k for k in BELIEF if k in prompt)
        return Resp({"response": "Confidence: %d\nReason: x"
                     % round(100 * BELIEF[key][0])})
    return Resp({"response": "asks for payment"})       # extract_signals


W.requests.post = fake_post
W.OLLAMA_URL = "http://fake/api/generate"
W.TAVILY_URL = "http://tavily/search"
tmp = tempfile.mkdtemp()
W.WEB_CACHE_DIR = __import__("pathlib").Path(tmp) / "cache"
os.environ["TAVILY_API_KEY"] = "test"
# the KB side: one relevant pattern for every call, no LLM gate
W.retrieve_kb = lambda coll, signals: [{"text": "pattern", "scam_type": "x",
                                         "distance": 0.2, "similarity": 0.8}]
W.filter_relevant_kb = lambda t, c, **k: (c, [], "")
W.filter_relevant_web = lambda t, items, **k: (items, [])
W.build_evidence_block = lambda kb, web: (
    "KB PATTERN" + "".join("\n" + w["title"] for w in web))

print("\nthe answer word")
for tok, want in (("Sc", "scam"), (" Scam", "scam"), ("scam", "scam"),
                  ("Leg", "legit"), ("Legitimate", "legit"), ("S", None),
                  ("The", None), ("**Scam", "scam")):
    check("%r reads as %r" % (tok, want), W._answer_word(tok), want)

print("\nP(scam) from logprobs")
p, how = W.scam_probability("GIFTCARD call", "KB PATTERN")
check("measured from logprobs", how, "logprobs")
check("P(scam) is the scam share of the answer mass (0.97)", round(p, 3), 0.97)
p, _ = W.scam_probability("BORDERLINE call", "KB PATTERN")
check("an unsure call reads as unsure (0.55)", round(p, 3), 0.55)
check("the question is asked on the same prompt head as the graded score",
      W._judge_head("T", "E") in (W._judge_head("T", "E") + W.PROB_QUESTION))

print("\nadaptive: only the unsure calls go to the web")
CALLS.update(generate=0, logprob_generate=0, tavily=0)
r1 = W.detect_adaptive("GIFTCARD call", None, confident=0.9)
check("a confident scam keeps its KB verdict", (r1["escalated"], r1["predicted"]),
      (False, "Fraud"))
r2 = W.detect_adaptive("DENTIST call", None, confident=0.9)
check("a confident legit call keeps its KB verdict",
      (r2["escalated"], r2["predicted"]), (False, "Normal"))
check("neither of them searched the web", CALLS["tavily"], 0)
r3 = W.detect_adaptive("BORDERLINE call", None, confident=0.9)
check("an unsure call is sent to the web", r3["escalated"], True)
check("and judged again with the web evidence (0.55 -> 0.92)",
      (round(r3["p_kb"], 2), round(r3["p_scam"], 2)), (0.55, 0.92))
check("its verdict comes from the second judgement", r3["predicted"], "Fraud")
check("one live web search", CALLS["tavily"], 1)
check("the score column is P(scam) in percent", r3["confidence"], 92.0)
W.detect_adaptive("BORDERLINE call", None, confident=0.9)
check("the same search again comes from the cache", CALLS["tavily"], 1)
check("cache hit counted", W.WEB_STATS["cached"] >= 1)
r4 = W.detect_adaptive("BORDERLINE call", None, confident=0.5)
check("at confident=0.5 nothing is ever sent to the web", r4["escalated"], False)
check("no verbal-score call was needed anywhere", CALLS["generate"],
      5)                                  # 5 extract_signals, 0 graded scores

print("\nwithout a Tavily key")
del os.environ["TAVILY_API_KEY"]
shutil.rmtree(W.WEB_CACHE_DIR, ignore_errors=True)
r5 = W.detect_adaptive("UNSURE_NOWEB call", None, confident=0.9)
check("unsure, but no key: keeps the KB verdict",
      (r5["escalated"], r5["n_web"], r5["predicted"]), (True, 0, "Normal"))
os.environ["TAVILY_API_KEY"] = "test"

print("\nan Ollama without logprobs")
LOGPROBS["on"] = False
W._LOGPROBS_OK = None
p, how = W.scam_probability("GIFTCARD call", "KB PATTERN")
check("falls back to the graded score", (how, round(p, 2)), ("verbal", 0.97))
check("and stops asking for logprobs after the first miss", W._LOGPROBS_OK, False)

print("\nthe benchmark runner")
LOGPROBS["on"] = True
W._LOGPROBS_OK = None
W.get_kb_collection = lambda: None
import combined_evaluate as C                                  # noqa: E402
data = [("GIFTCARD call", "Fraud"), ("DENTIST call", "Normal"),
        ("BORDERLINE call", "Fraud"), ("UNSURE_NOWEB call", "Normal")]
out, raws, reasons, scores = C.run_webrag_adaptive(data, confident=0.9)
check("every call scored", [p for p, _ in out],
      ["Fraud", "Normal", "Fraud", "Normal"])
check("scores are P(scam) in percent", scores, [97.0, 3.0, 92.0, 40.0])
check("the reason says whether it searched",
      ["-> web" in r for r in reasons], [False, False, True, True])

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all good - the web is searched only when the model is unsure"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
