"""
Web-RAG scam detection system.

Pipeline per call:
    1. Extract scam signals from transcript           (local LLM)
    2. Retrieve candidates from local scam-pattern KB (ChromaDB)
    3. Retrieve from live web                         (Tavily, optional)
    4. RELEVANCE GATE - discard retrieved items that do not actually
       match this transcript                          (embedding + local LLM)
    5. Score surviving web items for credibility      (trust / recency / corroboration)
    6. Produce graded confidence 0-100                (local LLM)
    7. Apply threshold -> decision                    (fixed for now, learned later)

Web retrieval is OPTIONAL. Without a TAVILY_API_KEY the system runs KB-only,
which is itself a useful ablation condition.

WHY THE GATE EXISTS
-------------------
The KB contains scam patterns only. A vector search always returns its nearest
neighbours, so a perfectly legitimate call still came back with three documents
headed "CACHED SCAM PATTERNS", and the prompt then told the model to judge
"using ONLY the evidence provided". That primes Fraud on every call, and is why
Web-RAG scored WORSE than the LLM-only control on precision. Nearest is not the
same as relevant: an item now has to clear a similarity floor AND survive a
yes/no relevance check against the transcript before it reaches the prompt.
When nothing survives, the model is told so explicitly and judges the
transcript alone - the same footing as the LLM-only control, never worse.
"""

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

# Pull a few more candidates than we intend to use, then filter down. The gate
# needs something to reject; retrieving exactly N and gating leaves too little.
N_KB_CANDIDATES = 5
N_KB_KEEP = 3
N_WEB_RESULTS = 5


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name, default=True):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


# --- relevance gate configuration ----------------------------------------
# Cosine similarity floor for KB hits. This is a CHEAP PRE-FILTER ONLY - do not
# mistake it for the relevance decision. Measured on 200 transcripts from the
# 1000-row run, top-1 similarity is:
#     Fraud   p25 0.371  median 0.414  p75 0.444
#     Normal  p25 0.374  median 0.416  p75 0.449
# i.e. the two classes are indistinguishable by distance. MiniLM compresses
# these long pattern paragraphs into a narrow band: a genuine bank-impersonation
# match scores 0.50, a novel AI-voice-clone scam 0.49, and an ordinary dentist
# appointment 0.37. No threshold separates relevant from irrelevant on its own,
# which is why the LLM gate below is the part that actually decides. The floor's
# job is only to strip the obvious tail before spending a call on it.
MIN_KB_SIMILARITY = _env_float("WEBRAG_MIN_SIMILARITY", 0.35)
# Tavily's own relevance score for web hits (0-1).
MIN_WEB_RELEVANCE = _env_float("WEBRAG_MIN_WEB_RELEVANCE", 0.50)
# Second-stage LLM check. Costs one extra Ollama call per transcript that has
# any surviving candidate. Set WEBRAG_LLM_GATE=0 to use the similarity floor
# alone - faster, and a useful ablation in its own right.
USE_LLM_GATE = _env_flag("WEBRAG_LLM_GATE", True)

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


def collection_space(collection):
    """
    Which distance metric the collection was built with. Chroma's default is
    'l2'; build_index.py now asks for 'cosine'. Older indexes predate that, so
    both have to be readable - hence the fallbacks rather than a constant.
    """
    try:
        md = collection.metadata or {}
        if md.get("hnsw:space"):
            return str(md["hnsw:space"]).lower()
    except Exception:
        pass
    try:
        cfg = getattr(collection, "configuration_json", None) or {}
        space = (cfg.get("hnsw") or {}).get("space")
        if space:
            return str(space).lower()
    except Exception:
        pass
    return "l2"


def similarity_from_distance(distance, space="l2"):
    """
    Convert a Chroma distance into a cosine similarity in [-1, 1].

    all-MiniLM-L6-v2 emits unit-length vectors (its sentence-transformers
    config ends in a Normalize module), so for those:
        squared L2 = 2 - 2*cos   ->   cos = 1 - d/2
    The 'cosine' and 'ip' spaces already return 1 - cos and 1 - dot.
    """
    if distance is None:
        return 0.0
    if space in ("cosine", "ip"):
        return 1.0 - distance
    return 1.0 - (distance / 2.0)


def retrieve_kb(collection, query, n=N_KB_CANDIDATES):
    """
    Retrieve CANDIDATE patterns - nearest neighbours, not yet judged relevant.
    Every KB entry is web-derived and carries its own provenance (source
    domain, url, credibility at harvest time). Pass the result through
    filter_relevant_kb() before putting any of it in a prompt.
    """
    res = collection.query(query_texts=[query], n_results=n)
    space = collection_space(collection)
    out = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        out.append({
            "text": res["documents"][0][i],
            "scam_type": md.get("scam_type", ""),
            "domain": md.get("domain", ""),
            "url": md.get("url", ""),
            "credibility": md.get("credibility", 0.0),
            "retrieved_at": md.get("retrieved_at", ""),
            # Where this pattern came from. The persistent KB is entirely
            # web-harvested and says nothing, hence the default; the hybrid
            # baseline in combined_evaluate.py mixes in patterns generalised
            # from a labelled training split and marks those "training", so
            # the prompt can tell the judge which is which instead of calling
            # all of it "harvested from the web".
            "origin": md.get("origin", "web"),
            "support": md.get("support", 0),
            "distance": dist,
            "similarity": similarity_from_distance(dist, space),
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
# STEP 4 - RELEVANCE GATE
# =========================================================================

def _candidate_summary(item, limit=500):
    """The text the relevance judge sees for one candidate."""
    if "text" in item:                                    # KB pattern
        label = item.get("scam_type") or "unlabelled pattern"
        return f"({label}) {item['text'][:limit]}"
    return f"{item.get('title', '')}: {item.get('content', '')[:limit]}"   # web item


def llm_relevance_gate(transcript, candidates, max_transcript_chars=2500):
    """
    Ask the local LLM, in ONE call, which candidates actually describe what
    happens in this transcript. Returns (list of bools, note).

    Fails OPEN: if the response cannot be parsed the candidates are kept, since
    they already cleared the similarity floor. The note is returned so an
    unreadable judge shows up in the logs instead of silently moving results.
    """
    if not candidates:
        return [], "no-candidates"

    listing = "\n\n".join(
        f"{i}. {_candidate_summary(c)}" for i, c in enumerate(candidates, 1))

    prompt = f"""You are checking whether retrieved reference material is RELEVANT
to one specific phone call. You are NOT deciding whether the call is a scam.

An item is RELEVANT only if this call actually shows the behaviour it describes:
the same kind of request, the same tactic, the same kind of caller. An item
about a different scam type is NOT relevant even though both are phone calls
about money. When in doubt, answer NO.

TRANSCRIPT:
{transcript[:max_transcript_chars]}

CANDIDATE ITEMS:
{listing}

For each candidate, answer on its own line in exactly this format:
<number>: YES
or
<number>: NO
Output nothing else."""

    raw = call_ollama(prompt, max_tokens=10 * len(candidates) + 40)

    verdicts = {}
    for m in re.finditer(r"^\s*(\d+)\s*[:.)\-]?\s*(YES|NO)\b",
                         raw, re.IGNORECASE | re.MULTILINE):
        verdicts[int(m.group(1))] = m.group(2).upper() == "YES"

    if not verdicts:
        return [True] * len(candidates), "unreadable"

    keep = [verdicts.get(i, True) for i in range(1, len(candidates) + 1)]
    note = "ok" if len(verdicts) >= len(candidates) else "partial"
    return keep, note


def filter_relevant_kb(transcript, items, min_similarity=None,
                       keep=N_KB_KEEP, use_llm_gate=None):
    """
    Two-stage relevance gate over KB candidates.

    Stage A  similarity floor  - cheap, catches "nearest, but nothing like it"
    Stage B  LLM yes/no judge  - catches "similar wording, different situation"

    Returns (kept, dropped, note). Each dropped item carries a 'dropped_by'
    field, so the reason is inspectable rather than guessed at.
    """
    min_similarity = MIN_KB_SIMILARITY if min_similarity is None else min_similarity
    use_llm_gate = USE_LLM_GATE if use_llm_gate is None else use_llm_gate

    kept, dropped = [], []
    for it in items:
        if it.get("similarity", 0.0) >= min_similarity:
            kept.append(it)
        else:
            it["dropped_by"] = "similarity"
            dropped.append(it)

    note = "similarity-only"
    if kept and use_llm_gate:
        flags, note = llm_relevance_gate(transcript, kept)
        survivors = []
        for it, ok in zip(kept, flags):
            if ok:
                survivors.append(it)
            else:
                it["dropped_by"] = "llm-gate"
                dropped.append(it)
        kept = survivors

    # Best-matching survivors first, and never more than `keep` of them.
    kept.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return kept[:keep], dropped, note


def filter_relevant_web(transcript, items, min_relevance=None, use_llm_gate=None):
    """Same gate for live web hits, using Tavily's relevance score as stage A."""
    min_relevance = MIN_WEB_RELEVANCE if min_relevance is None else min_relevance
    use_llm_gate = USE_LLM_GATE if use_llm_gate is None else use_llm_gate

    kept, dropped = [], []
    for it in items:
        if it.get("relevance", 0.0) >= min_relevance:
            kept.append(it)
        else:
            it["dropped_by"] = "relevance"
            dropped.append(it)

    if kept and use_llm_gate:
        flags, _ = llm_relevance_gate(transcript, kept)
        survivors = []
        for it, ok in zip(kept, flags):
            if ok:
                survivors.append(it)
            else:
                it["dropped_by"] = "llm-gate"
                dropped.append(it)
        kept = survivors
    return kept, dropped


# =========================================================================
# STEP 5 - CREDIBILITY LAYER  (pure python, no external calls)
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
# STEP 6 - GRADED CONFIDENCE
# =========================================================================

def build_evidence_block(kb_items, web_items, min_credibility=0.40):
    """
    Render the evidence section of the prompt. Both lists are expected to have
    been through the relevance gate already - anything in here is asserted to
    match this transcript. Returns None when there is nothing to show, which is
    the caller's signal to use the no-evidence prompt.
    """
    parts = []
    harvested = [k for k in kb_items if k.get("origin", "web") != "training"]
    learned = [k for k in kb_items if k.get("origin", "web") == "training"]
    if harvested:
        parts.append("MATCHING SCAM PATTERNS (harvested from the web, kept only "
                     "because they match this call):")
        for k in harvested:
            src = k.get("domain") or "unknown source"
            cred = k.get("credibility", 0.0)
            sim = k.get("similarity", 0.0)
            parts.append(f"[match {sim:.2f} | credibility {cred:.2f} | source {src}]\n"
                         f"{k['text']}")
    if learned:
        # A different kind of evidence, so it is labelled as one. These carry
        # no source domain and no harvest-time credibility - what they have
        # instead is how many training batches independently produced them,
        # which is the only support claim that can honestly be made for them.
        parts.append("MATCHING SCAM PATTERNS (generalised from separate "
                     "labelled training calls, kept only because they match "
                     "this call):")
        for k in learned:
            sim = k.get("similarity", 0.0)
            sup = k.get("support", 0)
            parts.append(f"[match {sim:.2f} | generalised from {sup} training "
                         f"batches]\n{k['text']}")
    kept = [w for w in web_items if w["credibility"] >= min_credibility]
    if kept:
        parts.append("\nRECENT WEB REPORTS (with credibility scores):")
        for w in kept:
            parts.append(
                f"[credibility {w['credibility']:.2f} | source {domain_of(w['url'])}] "
                f"{w['title']}: {w['content'][:400]}")
    return "\n\n".join(parts) if parts else None


def assess_confidence(transcript, evidence):
    """
    Graded 0-100 confidence. Two prompts, because "nothing relevant was
    retrieved" and "here is matching evidence" are different questions. The old
    single prompt said "using ONLY the evidence provided" even when the evidence
    was three unrelated scam patterns, which pushed every call toward Fraud.
    """
    if evidence:
        prompt = f"""You are a scam detection analyst. Decide whether the caller in this
transcript is attempting a scam.

The reference material below describes known scam tactics that a retrieval step
judged similar to this call. It is BACKGROUND, NOT PROOF. Its presence does not
mean this call is a scam. Rely on it only where the transcript genuinely shows
that behaviour, and ignore any part of it that does not fit. Web reports carry a
credibility score between 0 and 1; weight low-credibility evidence less heavily.
Your verdict must rest on what the caller actually says and asks for.

REFERENCE MATERIAL:
{evidence}

TRANSCRIPT:
{transcript}

Give a confidence score from 0 to 100, where 0 means certainly legitimate and 100
means certainly a scam.
Respond in exactly this format:
Confidence: <number>
Reason: <one sentence>"""
    else:
        prompt = f"""You are a scam detection analyst. Decide whether the caller in this
transcript is attempting a scam.

No stored scam pattern or web report matched this call closely enough to be
useful, so judge the transcript on its own merits. Finding no match is NOT
evidence either way - do not treat it as suspicious, and do not treat it as
proof the call is legitimate.

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
# STEP 7 - DECISION
# =========================================================================

def decide(confidence, threshold=50):
    return "Fraud" if confidence >= threshold else "Normal"


# =========================================================================
# FULL PIPELINE
# =========================================================================

def detect(transcript, collection, use_web=True, threshold=50,
           min_similarity=None, use_llm_gate=None):
    signals = extract_signals(transcript)

    kb_candidates = retrieve_kb(collection, signals)
    kb_items, kb_dropped, gate_note = filter_relevant_kb(
        transcript, kb_candidates,
        min_similarity=min_similarity, use_llm_gate=use_llm_gate)

    if use_web:
        web_candidates = retrieve_web(signals)
        web_items, web_dropped = filter_relevant_web(
            transcript, web_candidates, use_llm_gate=use_llm_gate)
        web_items = score_credibility(web_items)
    else:
        web_candidates, web_items, web_dropped = [], [], []

    evidence = build_evidence_block(kb_items, web_items)
    confidence, reason = assess_confidence(transcript, evidence)

    best = max((c.get("similarity", 0.0) for c in kb_candidates), default=None)
    return {
        "signals": signals,
        "kb_top_type": kb_items[0]["scam_type"] if kb_items else "",
        "kb_top_distance": kb_items[0]["distance"] if kb_items else None,
        "kb_top_similarity": kb_items[0]["similarity"] if kb_items else None,
        "kb_best_candidate_similarity": best,
        "n_kb_candidates": len(kb_candidates),
        "n_kb_kept": len(kb_items),
        "n_kb_harvested": sum(1 for k in kb_items
                              if k.get("origin", "web") != "training"),
        "n_kb_learned": sum(1 for k in kb_items
                            if k.get("origin", "web") == "training"),
        "n_kb_dropped": len(kb_dropped),
        "gate_note": gate_note,
        "evidence_used": evidence is not None,
        "n_web_candidates": len(web_candidates),
        "n_web": len(web_items),
        "n_web_dropped": len(web_dropped),
        "mean_web_credibility": (
            sum(w["credibility"] for w in web_items) / len(web_items)
            if web_items else 0.0),
        "confidence": confidence,
        "reason": reason,
        "predicted": decide(confidence, threshold),
    }
