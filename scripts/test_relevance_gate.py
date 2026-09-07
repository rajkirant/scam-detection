"""
Offline tests for the Web-RAG relevance gate.

No Ollama, no network, no ChromaDB: the LLM is stubbed and the collection is a
fake, so this runs anywhere and in under a second. It checks the logic that
decides what reaches the prompt, which is the part that was silently feeding
unrelated scam patterns into every verdict.

Run from project root:
    python scripts/test_relevance_gate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import webrag_system as W

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


class StubLLM:
    """Replaces call_ollama. Returns queued replies and records every prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt, max_tokens=300, temperature=0.0):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class FakeCollection:
    """Minimal Chroma stand-in returning fixed hits at chosen similarities."""

    metadata = {"hnsw:space": "cosine"}

    def __init__(self, hits):
        self.hits = hits  # [(scam_type, similarity)]

    def query(self, query_texts=None, n_results=5, **kw):
        hits = self.hits[:n_results]
        return {
            "ids": [[f"p{i}" for i, _ in enumerate(hits)]],
            "documents": [[f"pattern text for {t}" for t, _ in hits]],
            "metadatas": [[{"scam_type": t, "domain": "ftc.gov",
                            "url": "https://ftc.gov/x", "credibility": 0.85}
                           for t, _ in hits]],
            "distances": [[1.0 - s for _, s in hits]],
        }


def item(sim, scam_type="bank_impersonation"):
    return {"text": f"pattern text for {scam_type}", "scam_type": scam_type,
            "domain": "ftc.gov", "url": "https://ftc.gov/x",
            "credibility": 0.85, "distance": 1.0 - sim, "similarity": sim}


# --------------------------------------------------------------- unit tests

def test_similarity_conversion():
    check("cosine distance converts directly",
          abs(W.similarity_from_distance(0.4, "cosine") - 0.6) < 1e-9)
    check("squared L2 converts on the unit-vector assumption",
          abs(W.similarity_from_distance(1.0, "l2") - 0.5) < 1e-9)
    check("missing distance is not treated as a perfect match",
          W.similarity_from_distance(None, "cosine") == 0.0)


def test_similarity_floor():
    items = [item(0.60), item(0.36, "prize_lottery"), item(0.20, "romance_relationship")]
    kept, dropped, _ = W.filter_relevant_kb("t", items, min_similarity=0.40,
                                            use_llm_gate=False)
    check("floor keeps only hits at or above it", [k["similarity"] for k in kept] == [0.60])
    check("floor drops the rest", len(dropped) == 2)
    check("drop reason is recorded",
          all(d["dropped_by"] == "similarity" for d in dropped))


def test_llm_gate_discards_irrelevant():
    W.call_ollama = StubLLM(["1: NO\n2: YES\n3: NO"])
    items = [item(0.55, "tech_support"), item(0.52, "bank_impersonation"),
             item(0.50, "prize_lottery")]
    kept, dropped, note = W.filter_relevant_kb("t", items, min_similarity=0.0,
                                               use_llm_gate=True)
    check("only the YES candidate survives",
          [k["scam_type"] for k in kept] == ["bank_impersonation"], kept)
    check("gate note is clean", note == "ok", note)
    check("NO candidates are marked as gate drops",
          sorted(d["dropped_by"] for d in dropped) == ["llm-gate", "llm-gate"])


def test_llm_gate_can_discard_everything():
    """The case that matters: a legitimate call must end up with NO evidence."""
    W.call_ollama = StubLLM(["1: NO\n2: NO"])
    items = [item(0.45, "government_impersonation"), item(0.44, "prize_lottery")]
    kept, dropped, _ = W.filter_relevant_kb("t", items, min_similarity=0.0,
                                            use_llm_gate=True)
    check("nothing survives when nothing is relevant", kept == [])
    check("build_evidence_block returns None, not an empty heading",
          W.build_evidence_block(kept, []) is None)


def test_gate_parses_loose_formats():
    W.call_ollama = StubLLM(["1. YES\n2) no\n3 - Yes"])
    flags, note = W.llm_relevance_gate("t", [item(0.5), item(0.5), item(0.5)])
    check("numbered variants and mixed case parse", flags == [True, False, True], flags)
    check("full coverage reports ok", note == "ok", note)


def test_gate_fails_open_when_unreadable():
    W.call_ollama = StubLLM(["I cannot determine relevance from this."])
    flags, note = W.llm_relevance_gate("t", [item(0.5), item(0.5)])
    check("unreadable judge keeps candidates rather than guessing",
          flags == [True, True])
    check("and says so, so it shows up in the run log", note == "unreadable", note)


def test_missing_verdict_defaults_to_keep():
    W.call_ollama = StubLLM(["1: NO"])                    # says nothing about 2
    flags, note = W.llm_relevance_gate("t", [item(0.5), item(0.5)])
    check("unanswered candidate is kept", flags == [False, True], flags)
    check("partial coverage is flagged", note == "partial", note)


def test_keep_cap_and_ordering():
    W.call_ollama = StubLLM(["1: YES\n2: YES\n3: YES\n4: YES"])
    items = [item(0.41, "a"), item(0.62, "b"), item(0.55, "c"), item(0.48, "d")]
    kept, _, _ = W.filter_relevant_kb("t", items, min_similarity=0.0,
                                      use_llm_gate=True, keep=2)
    check("survivors are ordered best match first and capped",
          [k["scam_type"] for k in kept] == ["b", "c"], kept)


# ---------------------------------------------------------- prompt selection

def test_no_evidence_prompt_is_neutral():
    W.call_ollama = StubLLM(["Confidence: 20\nReason: routine call."])
    score, _ = W.assess_confidence("some transcript", None)
    prompt = W.call_ollama.prompts[0]
    check("no-evidence prompt is used", "on its own merits" in prompt)
    check("absence of a match is explicitly not incriminating",
          "NOT\nevidence either way" in prompt or "not\nevidence" in prompt.lower())
    check("no scam-pattern heading is shown when nothing matched",
          "SCAM PATTERNS" not in prompt)
    check("score parses", score == 20)


def test_evidence_prompt_does_not_assert_guilt():
    W.call_ollama = StubLLM(["Confidence: 80\nReason: matches the pattern."])
    W.assess_confidence("some transcript", "MATCHING SCAM PATTERNS:\nblah")
    prompt = W.call_ollama.prompts[0]
    check("evidence is framed as background, not proof",
          "BACKGROUND, NOT PROOF" in prompt)
    check("the old 'ONLY the evidence provided' framing is gone",
          "ONLY the evidence" not in prompt)


# ------------------------------------------------------------ end-to-end

def test_detect_legitimate_call_gets_no_evidence():
    """A legit call whose nearest neighbours are all rejected must be judged alone."""
    W.call_ollama = StubLLM([
        "Caller confirms a dental appointment.",   # extract_signals
        "1: NO\n2: NO\n3: NO",                     # relevance gate
        "Confidence: 10\nReason: routine appointment call.",
    ])
    coll = FakeCollection([("government_impersonation", 0.40),
                           ("prize_lottery", 0.38),
                           ("bank_impersonation", 0.36)])
    res = W.detect("Caller: your dentist appointment is Tuesday.", coll,
                   use_web=False, min_similarity=0.0)
    check("no evidence was used", res["evidence_used"] is False)
    check("all candidates were discarded",
          res["n_kb_kept"] == 0 and res["n_kb_dropped"] == 3, res)
    check("verdict is Normal", res["predicted"] == "Normal", res["predicted"])
    check("verdict prompt was the no-evidence one",
          "on its own merits" in W.call_ollama.prompts[2])


def test_detect_real_match_keeps_evidence():
    W.call_ollama = StubLLM([
        "Caller demands card number and threatens to freeze the account.",
        "1: YES\n2: NO\n3: NO",
        "Confidence: 90\nReason: classic bank impersonation.",
    ])
    coll = FakeCollection([("bank_impersonation", 0.52),
                           ("prize_lottery", 0.41),
                           ("romance_relationship", 0.39)])
    res = W.detect("Caller: give me your CVV or we freeze the account.", coll,
                   use_web=False, min_similarity=0.0)
    check("the matching pattern survives", res["n_kb_kept"] == 1)
    check("evidence was used", res["evidence_used"] is True)
    check("top type is reported", res["kb_top_type"] == "bank_impersonation")
    check("verdict is Fraud", res["predicted"] == "Fraud")
    check("the surviving pattern is in the prompt",
          "bank_impersonation" in W.call_ollama.prompts[2])


def test_detect_reports_gate_diagnostics():
    W.call_ollama = StubLLM(["signals", "1: NO\n2: NO", "Confidence: 30\nReason: x."])
    coll = FakeCollection([("prize_lottery", 0.50), ("tech_support", 0.45)])
    res = W.detect("t", coll, use_web=False, min_similarity=0.0)
    for field in ("n_kb_candidates", "n_kb_kept", "n_kb_dropped",
                  "kb_best_candidate_similarity", "gate_note", "evidence_used"):
        check(f"detect() reports {field}", field in res)
    check("best candidate similarity survives the drop for logging",
          abs(res["kb_best_candidate_similarity"] - 0.50) < 1e-9)


def main():
    original = W.call_ollama
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            print(f"\n{t.__name__}:")
            t()
    finally:
        W.call_ollama = original

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print(f"all {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
