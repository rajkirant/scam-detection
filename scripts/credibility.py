"""
Shared utilities: local LLM access and the credibility scoring layer.

Imported by both harvest_patterns.py (KB building) and webrag_system.py
(detection), so the credibility logic exists in exactly one place.
"""

import json, os
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import requests

# --- local LLM -----------------------------------------------------------

OLLAMA_MODEL = os.environ.get("SCAM_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- web search + quota protection ---------------------------------------

TAVILY_URL = "https://api.tavily.com/search"

CACHE_DIR = Path("./cache/tavily")
QUOTA_LOG = Path("./cache/tavily_usage.json")
CACHE_TTL_DAYS = 7          # cached results reused for a week
CREDITS = {"basic": 1, "advanced": 2}


def _cache_key(query, max_results, depth):
    raw = f"{query}|{max_results}|{depth}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _log_usage(depth, cached):
    """Track cumulative API usage so quota consumption is visible."""
    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = {"calls": 0, "credits": 0, "cache_hits": 0, "first": None, "last": None}
    if QUOTA_LOG.exists():
        try:
            log = json.loads(QUOTA_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    now = datetime.now(timezone.utc).isoformat()
    log["first"] = log.get("first") or now
    log["last"] = now
    if cached:
        log["cache_hits"] = log.get("cache_hits", 0) + 1
    else:
        log["calls"] = log.get("calls", 0) + 1
        log["credits"] = log.get("credits", 0) + CREDITS.get(depth, 1)
    QUOTA_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def tavily_search(query, api_key, max_results=6, depth="advanced",
                  use_cache=True, ttl_days=CACHE_TTL_DAYS):
    """
    Search Tavily with disk caching.

    Caching serves two purposes:
      1. Protects the free-tier quota - repeated runs cost nothing
      2. Freezes the evidence, so two evaluation runs compare your system
         changes rather than changes in what the web happened to return

    Returns (results, was_cached).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(query, max_results, depth)}.json"

    if use_cache and path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(blob["fetched_at"])
            if datetime.now(timezone.utc) - fetched < timedelta(days=ttl_days):
                _log_usage(depth, cached=True)
                return blob["results"], True
        except Exception:
            pass  # corrupt or unreadable cache entry - refetch

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": depth,
        "max_results": max_results,
    }
    r = requests.post(TAVILY_URL, json=payload, timeout=30)
    r.raise_for_status()
    raw = r.json().get("results", [])

    results = [{
        "title": it.get("title", ""),
        "url": it.get("url", ""),
        "content": it.get("content", ""),
        "published_date": it.get("published_date"),
    } for it in raw]

    path.write_text(json.dumps({
        "query": query,
        "depth": depth,
        "max_results": max_results,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "raw_first_result": raw[0] if raw else None,  # for field inspection
    }, indent=2), encoding="utf-8")

    _log_usage(depth, cached=False)
    return results, False


def print_quota():
    if not QUOTA_LOG.exists():
        print("  no Tavily usage recorded yet")
        return
    log = json.loads(QUOTA_LOG.read_text(encoding="utf-8"))
    n_cached = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
    print(f"  Tavily API calls:  {log.get('calls', 0)}")
    print(f"  credits used:      {log.get('credits', 0)}")
    print(f"  cache hits:        {log.get('cache_hits', 0)}  (credits saved: "
          f"{log.get('cache_hits', 0) * 2})")
    print(f"  cached queries:    {n_cached}")
    print(f"  since:             {log.get('first', '-')[:10]}")


# --- credibility configuration -------------------------------------------
#
# Trust is assigned by TLD/suffix pattern first, then by explicit domain list.
# Pattern matching is essential: a literal-only whitelist silently demotes
# every government source it does not happen to name.

# Suffix patterns -> trust. Checked in order, first match wins.
TRUST_PATTERNS = [
    # Tier 1: government, regulators, military, national CERTs
    (1.00, (".gov", ".gov.au", ".gov.uk", ".govt.nz", ".gov.sg", ".gc.ca",
            ".mil", ".police.uk", ".europa.eu")),
    # Tier 2: academic
    (0.75, (".edu", ".ac.uk", ".ac.nz", ".edu.au")),
]

# Explicit domains -> trust. Checked after patterns.
TRUST_DOMAINS = {
    1.00: [  # non-.gov consumer protection / national CERT
        "cert.govt.nz", "netsafe.org.nz", "actionfraud.police.uk",
        "consumerprotection.govt.nz", "ic3.gov", "scamwatch.gov.au",
    ],
    0.75: [  # established news, major banks, recognised security research
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "theguardian.com",
        "rnz.co.nz", "nzherald.co.nz", "stuff.co.nz", "abc.net.au",
        "krebsonsecurity.com", "which.co.uk", "ft.com", "wsj.com",
    ],
    0.50: [  # vendor / industry publications - informative but commercial
        "bitdefender.com", "microsoft.com", "support.microsoft.com",
        "kaspersky.com", "norton.com", "mcafee.com", "trendmicro.com",
        "en.wikipedia.org",
    ],
    0.15: [  # user-generated - actively downweighted
        "reddit.com", "youtube.com", "quora.com", "facebook.com",
        "twitter.com", "x.com", "tiktok.com", "medium.com",
    ],
}

DEFAULT_TRUST = 0.35
RECENCY_HALFLIFE_DAYS = 45.0
W_TRUST, W_RECENCY, W_CORROB = 0.5, 0.3, 0.2


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


def domain_of(url):
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def source_trust(url):
    """
    Trust score for a source. Suffix patterns are checked first so that any
    government or academic domain is recognised, then explicit domain lists.
    """
    d = domain_of(url)
    if not d:
        return DEFAULT_TRUST

    # explicit domains take precedence (lets us downweight youtube.com etc.)
    for score, domains in TRUST_DOMAINS.items():
        for listed in domains:
            if d == listed or d.endswith("." + listed):
                return score

    # then suffix patterns
    for score, suffixes in TRUST_PATTERNS:
        for suf in suffixes:
            if d == suf.lstrip(".") or d.endswith(suf):
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
    Independent-domain agreement. Multiple items from ONE domain do not
    corroborate each other - that is the duplication problem.
    """
    domains = [domain_of(i["url"]) for i in items]
    distinct = len({d for d in domains if d})
    if distinct <= 1:
        return [0.0] * len(items)
    scores = []
    for d in domains:
        others = len({x for x in domains if x and x != d})
        scores.append(min(others / 2.0, 1.0))
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
