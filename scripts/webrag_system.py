"""
Web-RAG scam detection system.

Pipeline per call:
    1. Extract scam signals from transcript        (local LLM)
    2. Retrieve from local scam-pattern KB         (ChromaDB)
    3. Retrieve from live web                      (Tavily, optional)
    4. Score each web item for credibility         (source trust / recency / corroboration)
    5. Produce graded confidence 0-100             (local LLM)
    6. Apply threshold -> decision                 (fixed for now, learned later)

Web retrieval is OPTIONAL. Without a TAVILY_API_KEY the system runs KB-only,
which is itself a useful ablation condition.
"""

import os
import re
import math
from datetime import datetime, timezone

import os
import re
import math
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "./chroma_db"
KB_COLLECTION = "scam_patterns"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_MODEL = os.environ.get("SCAM_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/generate"

TAVILY_URL = "https://api.tavily.com/search"
from urllib.parse import urlparse

import requests
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = "./chroma_db"
KB_COLLECTION = "scam_patterns"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TAVILY_URL = "https://api.tavily.com/search"
N_KB_CHUNKS = 3
N_WEB_RESULTS = 5

# --- credibility layer configuration -------------------------------------

TRUST_TIERS = {
    1.00: [  # government / national CERT / consumer protection
        "cert.govt.nz", "consumerprotection.govt.nz", "netsafe.org.nz",
        "actionfraud.police.uk", "ncsc.gov.uk", "ftc.gov", "consumer.ftc.gov",
        "scamwatch.gov.au", "accc.gov.au", "cisa.gov", "ic3.gov",
    ],
    0.75: [  # established news / major banks / recognised security vendors
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "theguardian.com",
        "rnz.co.nz", "nzherald.co.nz", "stuff.co.nz", "abc.net.au",
        "krebsonsecurity.com",
    ],
}
DEFAULT_TRUST = 0.35          # anything not listed
RECENCY_HALFLIFE_DAYS = 45.0  # weight halves every 45 days
W_TRUST, W_RECENCY, W_CORROB = 0.5, 0.3, 0.2


# =========================================================================
# LOCAL LLM
# =========================================================================

def call_ollama(prompt, max_tokens=300, temperature=0.0):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach Ollama at localhost:11434. Try: ollama serve")


# =========================================================================
# STEP 1 - SIGNAL EXTRACTION
# =========================================================================

def extract_signals(transcript):
    """Turn a transcript into a short search-friendly description of the tactic."""
    prompt = f"""Read this phone call transcript and describe, in one short sentence,
what the caller is asking the recipient to do and what pressure tactic they use.
Do not judge whether it is a scam. Just describe the behaviour factually.

Transcript:
{transcript}

Respond with the single sentence only, no preamble."""
    return call_ollama(prompt, max_tokens=60)


# =========================================================================
# STEP 2 - LOCAL KB RETRIEVAL
# =========================================================================

def get_kb_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=KB_COLLECTION, embedding_function=ef)


def retrieve_kb(collection, query, n=N_KB_CHUNKS):
    """
    Retrieve cached patterns. Every KB entry is web-derived and carries its
    own provenance (source domain, url, credibility at harvest time).
    """
    res = collection.query(query_texts=[query], n_results=n)
    out = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        out.append({
            "text": res["documents"][0][i],
            "scam_type": md.get("scam_type", ""),
            "domain": md.get("domain", ""),
            "url": md.get("url", ""),
            "credibility": md.get("credibility", 0.0),
            "retrieved_at": md.get("retrieved_at", ""),
            "distance": res["distances"][0][i],
        })
    return out


# =========================================================================
# STEP 3 - WEB RETRIEVAL (optional)
# =========================================================================

def retrieve_web(query, n=N_WEB_RESULTS):
    """Query Tavily. Returns [] if no API key is configured."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    payload = {
        "api_key": api_key,
        "query": f"phone scam {query}",
        "search_depth": "advanced",
        "max_results": n,
        "days": 365,
    }
    try:
        r = requests.post(TAVILY_URL, json=payload, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"    [web retrieval failed: {e}]")
        return []

    out = []
    for item in results:
        out.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "published_date": item.get("published_date"),
            "relevance": item.get("score", 0.0),
        })
    return out


# =========================================================================
# STEP 4 - CREDIBILITY LAYER  (pure python, no external calls)
# =========================================================================

def domain_of(url):
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def source_trust(url):
    d = domain_of(url)
    if not d:
        return DEFAULT_TRUST
    for score, domains in TRUST_TIERS.items():
        for trusted in domains:
            if d == trusted or d.endswith("." + trusted):
                return score
    return DEFAULT_TRUST


def recency_weight(published_date, now=None):
    """Exponential decay. Unknown date is treated as moderately stale."""
    if not published_date:
        return 0.5
    now = now or datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(published_date, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    else:
        return 0.5
    days = max((now - dt).days, 0)
    return math.exp(-math.log(2) * days / RECENCY_HALFLIFE_DAYS)


def corroboration_scores(items):
    """
    Independent-domain agreement. An item is corroborated if other items from
    DIFFERENT domains are present. Multiple items from one domain do not
    corroborate each other - that is the duplication problem.
    """
    domains = [domain_of(i["url"]) for i in items]
    distinct = len({d for d in domains if d})
    if distinct <= 1:
        return [0.0] * len(items)
    scores = []
    for d in domains:
        others = len({x for x in domains if x and x != d})
        scores.append(min(others / 2.0, 1.0))  # 2+ other domains = full score
    return scores


def score_credibility(items, now=None):
    """Attach trust, recency, corroboration and a combined score to each item."""
    if not items:
        return []
    corrob = corroboration_scores(items)
    for item, c in zip(items, corrob):
        t = source_trust(item["url"])
        r = recency_weight(item.get("published_date"), now=now)
        item["trust"] = t
        item["recency"] = r
        item["corroboration"] = c
        item["credibility"] = W_TRUST * t + W_RECENCY * r + W_CORROB * c
    return sorted(items, key=lambda x: x["credibility"], reverse=True)


# =========================================================================
# STEP 5 - GRADED CONFIDENCE
# =========================================================================

def build_evidence_block(kb_items, web_items, min_credibility=0.40):
    parts = []
    if kb_items:
        parts.append("CACHED SCAM PATTERNS (harvested from the web at last refresh):")
        for k in kb_items:
            src = k.get("domain") or "unknown source"
            cred = k.get("credibility", 0.0)
            parts.append(f"[credibility {cred:.2f} | source {src}]\n{k['text']}")
    kept = [w for w in web_items if w["credibility"] >= min_credibility]
    if kept:
        parts.append("\nRECENT WEB REPORTS (with credibility scores):")
        for w in kept:
            parts.append(
                f"[credibility {w['credibility']:.2f} | source {domain_of(w['url'])}] "
                f"{w['title']}: {w['content'][:400]}")
    return "\n\n".join(parts) if parts else "No supporting evidence retrieved."


def assess_confidence(transcript, evidence):
    prompt = f"""You are a scam detection analyst. Using ONLY the evidence provided,
assess how likely it is that the caller in this transcript is attempting a scam.

Evidence may include known scam patterns and recent web reports. Web reports carry
a credibility score between 0 and 1; weight low-credibility evidence less heavily.

EVIDENCE:
{evidence}

TRANSCRIPT:
{transcript}

Give a confidence score from 0 to 100, where 0 means certainly legitimate and 100
means certainly a scam.
Respond in exactly this format:
Confidence: <number>
Reason: <one sentence>"""

    text = call_ollama(prompt, max_tokens=150)
    m = re.search(r"confidence:\s*(\d{1,3})", text, re.IGNORECASE)
    score = min(int(m.group(1)), 100) if m else None
    if score is None:
        m2 = re.search(r"\b(\d{1,3})\b", text)
        score = min(int(m2.group(1)), 100) if m2 else 50
    rm = re.search(r"reason:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return score, (rm.group(1).strip() if rm else text)


# =========================================================================
# STEP 6 - DECISION
# =========================================================================

def decide(confidence, threshold=50):
    return "Fraud" if confidence >= threshold else "Normal"


# =========================================================================
# FULL PIPELINE
# =========================================================================

def detect(transcript, collection, use_web=True, threshold=50):
    signals = extract_signals(transcript)
    kb_items = retrieve_kb(collection, signals)
    web_items = score_credibility(retrieve_web(signals)) if use_web else []
    evidence = build_evidence_block(kb_items, web_items)
    confidence, reason = assess_confidence(transcript, evidence)
    return {
        "signals": signals,
        "kb_top_type": kb_items[0]["scam_type"] if kb_items else "",
        "kb_top_distance": kb_items[0]["distance"] if kb_items else None,
        "n_web": len(web_items),
        "mean_web_credibility": (
            sum(w["credibility"] for w in web_items) / len(web_items)
            if web_items else 0.0),
        "confidence": confidence,
        "reason": reason,
        "predicted": decide(confidence, threshold),
    }
