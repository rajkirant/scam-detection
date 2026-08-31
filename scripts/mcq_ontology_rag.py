#!/usr/bin/env python3
"""
mcq_ontology_rag.py

MCQ-scored ontology detector, with a speaker-labelling pre-pass.

Pipeline is now three LLM calls per transcript instead of two:
  1. route      - which call-type branch applies
  2. label_turns- split the single-stream transcript into Agent:/Caller:
                  turns, so the questions call does not have to solve
                  "who said this" and "what does it mean" at the same time
  3. questions  - answer the branch's MCQ questions on the labelled text

Labelling is strictly improve-or-do-nothing: if the model's labelled output
does not match the original wording closely enough (checked with
difflib.SequenceMatcher on the word sequence, threshold 0.92), the original
unlabelled transcript is used instead and nothing about the rest of the
pipeline changes. This step can only help or be a no-op; it cannot corrupt
the transcript, because the fidelity check runs before the labelled version
is trusted for anything.

Quote verification always checks against the ORIGINAL transcript, never the
labelled one, so labels can never be used to manufacture a passing quote.

Scoring is a plain sum, as before. No per-question weight.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

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
    """Checked against the ORIGINAL transcript. Callers must never pass a
    labelled version here - see the note in MCQOntologyDetector.detect()."""
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


# ---------------------------------------------------------- turn labelling

LABEL_PROMPT = """This is a single-stream transcript of a phone call, with no
speaker markers. Split it into turns and label each one, alternating between
the two parties.

Use exactly these labels:
  Agent:  - whoever answers or represents an organisation
  Caller: - whoever placed the call, or is reporting a problem

Rules:
  - Do not change, add, or remove a single word of the transcript.
  - Do not paraphrase, correct grammar, or add punctuation beyond what is
    needed to mark a new turn.
  - If you cannot tell where a turn ends, keep that stretch under whichever
    label it more plausibly belongs to rather than guessing at word level.
  - Output only the labelled transcript, nothing else.

Transcript:
{text}"""


def _words(text):
    """Word tokens for the fidelity check.

    These transcripts write contractions with a space instead of an
    apostrophe - "it s" rather than "it's" - because that is how the ASR
    that produced them worked. A labelling model writes ordinary English
    and puts the apostrophe back. Splitting "it's" into "it" + "s" (the
    same shape the original already has) makes both sides comparable,
    rather than merging the original's two tokens into one and creating a
    new mismatch, which made an earlier version of this function worse
    than not normalising apostrophes at all.

    This is only a tokenisation choice for comparing wording; quotes are
    still matched against the raw transcript text exactly as written.
    """
    text = re.sub(r"([a-z])'([a-z])", r"\1 \2", text.lower())
    return re.findall(r"[a-z]+", text)


def _strip_preamble(text):
    """Models often add a line like 'Here is the labelled transcript:'
    before the real content, despite being told not to. Cut everything
    before the first real label so that wrapper text cannot fail the
    fidelity check on its own."""
    m = re.search(r"\b(Agent|Caller):", text)
    return text[m.start():] if m else text


def label_fidelity(original, labelled):
    """Word-sequence similarity, ignoring our own Agent:/Caller: tags.
    A labelling pass that paraphrases will score low here and be rejected."""
    stripped = re.sub(r"\b(Agent|Caller):\s*", "", labelled)
    a, b = _words(original), _words(stripped)
    if not a:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def label_turns(transcript, ask_fn, min_fidelity=0.92, max_tokens=900, debug=False):
    """Return (text_for_questions, was_labelled, reason).

    Strictly improve-or-do-nothing: if the model's output does not match the
    original wording closely enough, the original transcript is returned
    unchanged and was_labelled is False. The rest of the pipeline proceeds
    exactly as it did before this feature existed.

    A common, harmless failure is the model adding a preamble sentence
    ("Here is the labelled transcript:") before the real content. That
    preamble is stripped before the fidelity check, so it cannot sink an
    otherwise faithful labelling.
    """
    try:
        raw = ask_fn(LABEL_PROMPT.format(text=transcript), max_tokens=max_tokens)
    except Exception as exc:
        return transcript, False, "request failed: %s" % exc

    labelled = _strip_preamble((raw or "").strip())
    if not labelled:
        return transcript, False, "empty response"

    fid = label_fidelity(transcript, labelled)
    if debug:
        print("      label fidelity: %.3f (need >= %.2f)" % (fid, min_fidelity))
        print("      label output  : %r" % labelled[:200])
    if fid < min_fidelity:
        return transcript, False, "fidelity %.2f below threshold" % fid
    return labelled, True, None


# ------------------------------------------------------------- the detector

class DetectionStats:
    def __init__(self):
        self.calls = 0
        self.routing_unread = 0
        self.answers_unread = 0
        self.quote_missing = 0
        self.quote_unsupported = 0
        self.invalid_option = 0
        self.labelling_accepted = 0
        self.labelling_rejected = 0
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
        total_lab = self.labelling_accepted + self.labelling_rejected
        if total_lab:
            print("    turn labelling: %d accepted, %d rejected (fell back "
                  "to unlabelled) of %d"
                  % (self.labelling_accepted, self.labelling_rejected, total_lab))
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
                 verify_quotes=True, debug=False, label_speakers=True):
        path = Path(ontology_path or DEFAULT_ONTOLOGY)
        if not path.exists():
            raise SystemExit("ontology not found: %s" % path)
        self.ont = json.loads(path.read_text(encoding="utf-8"))
        self.model = model or os.environ.get("SCAM_MODEL", "llama3.1:8b")
        self.max_tokens = max_tokens
        self.verify_quotes = verify_quotes
        self.debug = debug
        self.label_speakers = label_speakers
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

    def _questions_prompt(self, text_for_questions, branch, was_labelled):
        lines = ["You are analysing a phone call transcript."]
        if was_labelled:
            lines.append(
                "Turns are marked Agent: and Caller:. Agent is whoever "
                "answers or represents an organisation; Caller is whoever "
                "placed the call or is reporting something. Use these "
                "labels to decide who said what - do not re-guess turn "
                "boundaries from wording alone.")
        lines += [
            "Answer each question by choosing exactly ONE option id.",
            "",
            "For every answer you MUST give a short verbatim quote from "
            "the transcript that supports it. If the transcript does not "
            "state something, answer not_mentioned with an empty quote. "
            "Do not guess and do not infer.",
            "",
        ]
        for q in branch["questions"]:
            lines.append("%s: %s" % (q["id"], q["prompt"]))
            for o in q["options"]:
                lines.append("    %s = %s" % (o["id"], o["text"]))
            lines.append("")
        lines += [
            "Transcript:", text_for_questions, "",
            "Reply with JSON only, in exactly this shape:", "{",
        ]
        lines += ['  "%s": {"option": "<id>", "quote": "<verbatim text or empty>"},'
                  % q["id"] for q in branch["questions"]]
        lines += ["}"]
        return "\n".join(lines)

    def _score(self, branch, answers, original_transcript):
        """Plain sum. Quotes are checked against original_transcript, which
        must be the raw unlabelled text even when labelling was used for the
        questions prompt - labels must never be able to satisfy a quote."""
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
                    elif self.verify_quotes and not quote_supported(quote, original_transcript):
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
                    "detail": [], "labelled": False,
                    "note": "no questions for this branch"}

        text_for_questions, was_labelled, label_reason = transcript, False, None
        if self.label_speakers:
            text_for_questions, was_labelled, label_reason = label_turns(
                transcript, self._ask, debug=self.debug)
            if was_labelled:
                self.stats.labelling_accepted += 1
            else:
                self.stats.labelling_rejected += 1
                if self.debug and label_reason:
                    print("      labelling rejected: %s" % label_reason)

        raw = self._ask(self._questions_prompt(text_for_questions, branch, was_labelled))
        data = extract_json(raw)
        if data is None:
            self.stats.answers_unread += 1
            data = {}
            if self.debug:
                print("      answers unread: %r" % (raw or "")[:200])

        # quotes always checked against the ORIGINAL transcript
        score, detail = self._score(branch, data, transcript)
        band = self._band(score)
        self.stats.band_counts[band] += 1

        return {
            "call_type": call_type,
            "score": round(score, 3),
            "band": band,
            "predicted": "Fraud" if band == "scam" else "Normal",
            "detail": detail,
            "labelled": was_labelled,
            "labelled_text": text_for_questions if (self.debug and was_labelled) else None,
            "raw_response": (raw or "")[:800] if self.debug else None,
        }

    def explain(self, result):
        out = ["call type: %s" % result["call_type"],
               "verdict: %s (score %+.2f)" % (result["band"], result["score"]),
               "turns labelled: %s" % result.get("labelled", False)]
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
        if result.get("labelled_text"):
            out.append("  --- labelled transcript used ---")
            out.append("  " + result["labelled_text"][:500])
        return "\n".join(out)