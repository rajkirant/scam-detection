
#!/usr/bin/env python3
"""
ontology_rag.py - ontology-based RAG scam detector (alternative to vector RAG).

Pipeline per call:
  1. EXTRACT: LLM pulls structured attributes from the transcript
     (claimed identity, who initiated, requested action, payment channel, pressure).
  2. MATCH: score the attributes against every ontology node by weighted
     attribute overlap. No embeddings, no distances - explicit fields.
  3. ASSESS: LLM judges FIT vs CONTRADICTION: does the call exhibit the
     matched scam's defining behaviours, or does it match the node's
     legit_contrast (the legitimate mirror image) instead?
  4. If no node matches well -> novel_flag=True (in the full system this
     would trigger web search; here we fall back to LLM-only judgement).

Usage (on cs25003ay, Ollama running):
    from ontology_rag import OntologyRAG
    d = OntologyRAG("scam_ontology.json")
    out = d.detect(transcript)
    # out: {predicted, confidence, node, node_score, matched, contradicted,
    #        novel_flag, reason, signals}
"""

import json, re, requests, os
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("SCAM_MODEL", "llama3.1:8b")

MATCH_WEIGHTS = {         # how much each attribute contributes to a node match
    "claimed_identity": 0.40,
    "requested_action": 0.30,
    "payment_channel":  0.20,
    "pressure":         0.10,
}
NODE_MATCH_THRESHOLD = 0.35   # below this -> no known pattern (novel_flag)

EXTRACT_PROMPT = """Read this phone call transcript and extract ONLY what is actually present.
Do not guess or invent. If something is absent, use "none".

Transcript:
{call}

Answer in JSON with exactly these keys:
{{"claimed_identity": "who the caller claims to be, in a few words, or none",
  "who_initiated": "which side placed the call: caller_contacted_recipient, recipient_called_in, or unclear",
  "requested_action": "what the caller wants the other person to DO, or none",
  "money_or_data_destination": "where any money, codes, or personal data would END UP (own account, the caller, a third party, or none)",
  "payment_channel": "gift card, wire, crypto, card over phone, own-account transfer, or none",
  "pressure": "any urgency, threat, or secrecy tactic actually stated, or none"}}
JSON only, no other text."""

ASSESS_PROMPT = """You are judging whether a phone call is a scam.

Call transcript:
{call}

Extracted signals:
{signals}

Closest known scam pattern: {node_name}
Its defining behaviours: identity like {ids}; asks for {acts}; payment via {pays}; pressure like {press}.
What the LEGITIMATE version of this interaction looks like: {legit}

Decide by FIT, not resemblance:
- The call is only a scam if it actually EXHIBITS the defining behaviours
  (unsolicited contact, impersonation, pressure, money/codes/data flowing
  AWAY from the customer to the caller or a third party).
- If the call matches the legitimate version instead (customer initiated it,
  money stays in the customer's own accounts, codes go TO the customer,
  no pressure), it is Normal, EVEN IF the topic resembles the scam.
- Topic similarity alone is NOT evidence of fraud.

Answer in JSON: {{"verdict": "Fraud" or "Normal", "confidence": 0-100,
"reason": "one sentence naming the decisive fit or contradiction"}}
JSON only."""

FALLBACK_PROMPT = """Is this phone call a scam? No known scam pattern matched it,
so judge from the transcript alone. A call is a scam only if someone is being
manipulated into sending money, revealing codes/credentials, or giving access
to an attacker. Routine business between a customer and a company they contacted
is Normal.

Transcript:
{call}

Answer in JSON: {{"verdict": "Fraud" or "Normal", "confidence": 0-100,
"reason": "one sentence"}} JSON only."""


def call_ollama(prompt, timeout=180):
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0}
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_json(text):
    """Best-effort JSON extraction from an LLM reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        # common repair: single quotes
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return {}


class OntologyRAG:
    def __init__(self, ontology_path):
        self.nodes = json.load(open(ontology_path, encoding="utf-8"))["nodes"]

    # ---------- step 1 ----------
    def extract(self, call):
        out = parse_json(call_ollama(EXTRACT_PROMPT.format(call=call[:6000])))
        keys = ["claimed_identity", "who_initiated", "requested_action",
                "money_or_data_destination", "payment_channel", "pressure"]
        return {k: str(out.get(k, "none")).lower() for k in keys}

    # ---------- step 2 ----------
    @staticmethod
    def _overlap(text, terms):
        """1 if any ontology term appears in the extracted text, else 0."""
        if not text or text == "none":
            return 0.0
        return 1.0 if any(t in text for t in terms) else 0.0

    def match(self, sig):
        scored = []
        for node in self.nodes:
            s = (MATCH_WEIGHTS["claimed_identity"] * self._overlap(sig["claimed_identity"], node["claimed_identity"])
               + MATCH_WEIGHTS["requested_action"] * self._overlap(sig["requested_action"], [t.lower() for t in node["requested_action"]])
               + MATCH_WEIGHTS["payment_channel"]  * self._overlap(sig["payment_channel"],  [t.lower() for t in node["payment_channel"]])
               + MATCH_WEIGHTS["pressure"]         * self._overlap(sig["pressure"],          [t.lower() for t in node["pressure"]]))
            scored.append((s, node))
        scored.sort(key=lambda x: -x[0])
        return scored[0]      # (best_score, best_node)

    # ---------- steps 3/4 ----------
    def detect(self, call, threshold=50):
        sig = self.extract(call)
        score, node = self.match(sig)
        novel = score < NODE_MATCH_THRESHOLD

        if novel:
            out = parse_json(call_ollama(FALLBACK_PROMPT.format(call=call[:6000])))
        else:
            out = parse_json(call_ollama(ASSESS_PROMPT.format(
                call=call[:6000], signals=json.dumps(sig),
                node_name=node["name"],
                ids=", ".join(node["claimed_identity"][:4]),
                acts=", ".join(node["requested_action"][:4]),
                pays=", ".join(node["payment_channel"][:3]),
                press=", ".join(node["pressure"][:3]),
                legit=node["legit_contrast"])))

        conf = int(out.get("confidence", 50) or 50)
        verdict = str(out.get("verdict", "Normal"))
        if verdict not in ("Fraud", "Normal"):
            verdict = "Fraud" if conf >= threshold else "Normal"
        return {
            "predicted": verdict,
            "confidence": conf,
            "node": None if novel else node["id"],
            "node_score": round(score, 3),
            "novel_flag": novel,
            "reason": str(out.get("reason", ""))[:300],
            "signals": sig,
        }


if __name__ == "__main__":
    import sys
    d = OntologyRAG(sys.argv[1] if len(sys.argv) > 1 else "scam_ontology.json")
    call = sys.stdin.read().strip()
    print(json.dumps(d.detect(call), indent=2))
