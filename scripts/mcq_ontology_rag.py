#!/usr/bin/env python3
"""
mcq_ontology_rag.py

MCQ-scored ontology detector.

Scoring is a plain sum, with no per-question weight. Every question counts
equally, and the strength of a signal lives in the option values themselves
(-1.0 to +1.0), not in a separate weight field. The claimed_identity question
present in every branch is recorded but always scores 0.0 - it exists so the
system can explain the impersonation, not so it can vote.

    score > scam_cutoff   -> "scam"
    score < legit_cutoff  -> "legitimate"
    otherwise             -> "uncertain"

Both cutoffs default to 0.0 in the ontology JSON's "bands" section, so by
default the verdict is simply the sign of the sum. RQ2 (personalised alert
thresholds) moves the scam cutoff per user profile; nothing else about the
scoring needs to change to support that.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_ONTOLOGY = (Path(__file__).resolve().parent.parent
                    / "knowledge" / "mcq_ontology.json")


# --------------------------------------------------------------- utilities

def _norm(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_json(raw):
    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def quote_supported(quote, transcript, min_words=3):
    if not quote:
        return False
    q, t = _norm(quote), _norm(transcript)
    if len(q.split()) < min_words:
        return False
    if q in t:
        return True
    q_bare = re.sub(r"[^a-z0-9 ]", "", q)
    t_bare = re.sub(r"[^a-z0-9 ]", "", t)
    if q_bare in t_bare:
        return True
    words = q_bare.split()
    for size in range(len(words), min_words - 1, -1):
        for i in range(len(words) - size + 1):
            if " ".join(words[i:i + size]) in t_bare:
                return size >= max(min_words, int(0.6 * len(words)))
    return False


# ------------------------------------------------------------- the detector

class DetectionStats:
    def __init__(self):
        self.calls = 0
        self.routing_unread = 0
        self.answers_unread = 0
        self.quote_missing = 0
        self.quote_unsupported = 0
        self.invalid_option = 0
        self.branch_counts = Counter()
        self.band_counts = Counter()

    def report(self):
        print("    branches: %s" % dict(self.branch_counts))
        print("    verdicts: %s" % dict(self.band_counts))
        other = self.branch_counts.get("other", 0)
        if self.calls:
            pct = 100.0 * other / self.calls
            if pct > 15:
                print("    WARNING: %.0f%% routed to 'other'. Branch coverage "
                      "has gaps; those calls score 0 and read as uncertain." % pct)
        issues = (self.routing_unread + self.answers_unread
                  + self.quote_unsupported + self.invalid_option)
        if issues:
            print("    parsing: %d routing unread, %d answer blocks unread, "
                  "%d quotes not found in transcript, %d invalid option ids"
                  % (self.routing_unread, self.answers_unread,
                     self.quote_unsupported, self.invalid_option))
            print("    (unread and unsupported answers score 0, not Fraud)")
        else:
            print("    parsing: clean")


class MCQOntologyDetector:

    def __init__(self, ontology_path=None, model=None, max_tokens=700,
                 verify_quotes=True, debug=False):
        path = Path(ontology_path or DEFAULT_ONTOLOGY)
        if not path.exists():
            raise SystemExit("ontology not found: %s" % path)
        self.ont = json.loads(path.read_text(encoding="utf-8"))
        self.model = model or os.environ.get("SCAM_MODEL", "llama3.1:8b")
        self.max_tokens = max_tokens
        self.verify_quotes = verify_quotes
        self.debug = debug
        self.stats = DetectionStats()
        self._branches = {o["id"]: o for o in self.ont["options"]}

    def _ask(self, prompt, max_tokens=None):
        import credibility as C
        return C.call_ollama(prompt, max_tokens=max_tokens or self.max_tokens)

    def _routing_prompt(self, transcript):
        lines = ["You are classifying a phone call transcript.", "",
                 self.ont.get("prompt", "What kind of call is this?"), ""]
        for o in self.ont["options"]:
            lines.append("  %s = %s" % (o["id"], o["text"]))
        lines += [
            "",
            "Transcript:", transcript, "",
            "This question is only about the SUBJECT of the call. It is not "
            "about whether the call is a scam. Legitimate and fraudulent calls "
            "both occur in every category.",
            "",
            'Reply with JSON only: {"call_type": "<id>"}',
        ]
        return "\n".join(lines)

    def _route(self, transcript):
        raw = self._ask(self._routing_prompt(transcript), max_tokens=60)
        data = extract_json(raw)
        cid = (data or {}).get("call_type")
        if cid in self._branches:
            return cid, raw
        low = (raw or "").lower()
        for bid in self._branches:
            if bid in low:
                return bid, raw
        self.stats.routing_unread += 1
        return "other", raw

    def _questions_prompt(self, transcript, branch):
        lines = ["You are analysing a phone call transcript.",
                 "Answer each question by choosing exactly ONE option id.",
                 "",
                 "For every answer you MUST give a short verbatim quote from "
                 "the transcript that supports it. If the transcript does not "
                 "state something, answer not_mentioned with an empty quote. "
                 "Do not guess and do not infer.",
                 ""]
        for q in branch["questions"]:
            lines.append("%s: %s" % (q["id"], q["prompt"]))
            for o in q["options"]:
                lines.append("    %s = %s" % (o["id"], o["text"]))
            lines.append("")
        lines += [
            "Transcript:", transcript, "",
            "Reply with JSON only, in exactly this shape:", "{",
        ]
        lines += ['  "%s": {"option": "<id>", "quote": "<verbatim text or empty>"},'
                  % q["id"] for q in branch["questions"]]
        lines += ["}"]
        return "\n".join(lines)

    def _score(self, branch, answers, transcript):
        """Plain sum. No weight field is read anywhere in this function."""
        score, detail = 0.0, []
        for q in branch["questions"]:
            ans = answers.get(q["id"]) or {}
            oid = ans.get("option")
            quote = ans.get("quote") or ""

            opt = next((o for o in q["options"] if o["id"] == oid), None)
            reason = ""

            if opt is None:
                if oid not in (None, "", "not_mentioned"):
                    self.stats.invalid_option += 1
                    reason = "invalid option id %r" % oid
                oid, value = "not_mentioned", 0.0
            else:
                value = opt["value"]
                if oid != "not_mentioned":
                    if not quote:
                        self.stats.quote_missing += 1
                        oid, value = "not_mentioned", 0.0
                        reason = "no quote given"
                    elif self.verify_quotes and not quote_supported(quote, transcript):
                        self.stats.quote_unsupported += 1
                        oid, value = "not_mentioned", 0.0
                        reason = "quote not found in transcript"

            score += value
            detail.append({"question": q["id"], "option": oid,
                           "value": value, "contribution": round(value, 3),
                           "quote": quote[:160], "downgraded": reason})

        return score, detail

    def _band(self, score):
        lo = next((b["max"] for b in self.ont["bands"]
                   if b["label"] == "legitimate"), 0.0)
        hi = next((b["min"] for b in self.ont["bands"]
                   if b["label"] == "scam"), 0.0)
        if score > hi:
            return "scam"
        if score < lo:
            return "legitimate"
        return "uncertain"

    def detect(self, transcript):
        self.stats.calls += 1
        call_type, routing_raw = self._route(transcript)
        self.stats.branch_counts[call_type] += 1
        branch = self._branches[call_type]

        if not branch.get("questions"):
            band = self._band(0.0)
            self.stats.band_counts[band] += 1
            return {"call_type": call_type, "score": 0.0, "band": band,
                    "predicted": "Fraud" if band == "scam" else "Normal",
                    "detail": [], "note": "no questions for this branch"}

        raw = self._ask(self._questions_prompt(transcript, branch))
        data = extract_json(raw)
        if data is None:
            self.stats.answers_unread += 1
            data = {}
            if self.debug:
                print("      answers unread: %r" % (raw or "")[:200])

        score, detail = self._score(branch, data, transcript)
        band = self._band(score)
        self.stats.band_counts[band] += 1

        return {
            "call_type": call_type,
            "score": round(score, 3),
            "band": band,
            "predicted": "Fraud" if band == "scam" else "Normal",
            "detail": detail,
            "raw_response": (raw or "")[:800] if self.debug else None,
        }

    def explain(self, result):
        out = ["call type: %s" % result["call_type"],
               "verdict: %s (score %+.2f)" % (result["band"], result["score"])]
        for d in result["detail"]:
            if d["contribution"] == 0:
                continue
            direction = "toward scam" if d["contribution"] > 0 else "toward legitimate"
            out.append("  %-24s %-20s %+.2f  %s"
                       % (d["question"], d["option"], d["contribution"], direction))
            if d["quote"]:
                out.append("      evidence: %r" % d["quote"][:90])
        skipped = [d for d in result["detail"] if d["downgraded"]]
        for d in skipped:
            out.append("  %-24s dropped: %s" % (d["question"], d["downgraded"]))
        return "\n".join(out)