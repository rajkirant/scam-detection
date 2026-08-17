"""
Harvest scam patterns from the web into JSON.

JSON is the source of truth. The vector index is derived from it later.

Pipeline:
    1. Seed queries -> Tavily (cached, to protect free-tier quota)
    2. Credibility-score each article (trust / recency / corroboration)
    3. Drop anything below the credibility floor
    4. Skip articles already in the KB (no wasted LLM extraction)
    5. Local LLM extracts a structured pattern from each new survivor
    6. Merge into scam_patterns.json, preserving provenance and history

Re-running IS the weekly refresh.

RQ1 NOTE: seed queries deliberately cover only established scam categories.
Types in NOVEL_CATEGORIES are excluded, simulating a KB built before those
scams emerged. Live web retrieval at detection time is what should close that
gap - and measuring whether it does is RQ1.

Requires:
    TAVILY_API_KEY in .env or the environment

Usage:
    python scripts/harvest_patterns.py --dry-run    show plan, no network
    python scripts/harvest_patterns.py              harvest
    python scripts/harvest_patterns.py --stats      summarise existing JSON
    python scripts/harvest_patterns.py --no-cache   force fresh fetch
"""

import os
import re
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import credibility as W

KB_JSON = Path("./scam_patterns.json")

KB_CREDIBILITY_FLOOR = 0.55
RESULTS_PER_QUERY = 6

VALID_FLAGS = {"--dry-run", "--stats", "--no-cache"}

# Multiple seeds per category are fine - merge is keyed on URL, so overlapping
# results update rather than duplicate.
SEED_QUERIES = [
    # bank impersonation - weighted more heavily, it is the dominant test case
    ("bank_impersonation",
     "how bank impersonation phone scam works tactics victims"),
    ("bank_impersonation",
     "FTC consumer alert bank impersonation scam how to spot"),
    ("bank_impersonation",
     "scammers pretending to be your bank move money to safe account warning"),
    ("bank_impersonation",
     "regulator warning fake bank fraud department caller verification"),

    ("tech_support",
     "tech support phone scam remote access software tactics"),

    ("government_impersonation",
     "tax office police impersonation phone scam threats arrest"),

    ("prize_lottery",
     "lottery prize advance fee phone scam how it works"),

    ("romance_relationship",
     "romance scam phone money request tactics warning signs"),
    ("romance_relationship",
     "FTC romance scam consumer alert warning signs money request"),

    ("invoice_supplier",
     "invoice redirection supplier bank detail change fraud phone"),
]

# Deliberately NOT seeded - defines the novel split for RQ1.
NOVEL_CATEGORIES = [
    "ai_voice_cloning",
    "crypto_recovery",
    "qr_code_payment",
    "parcel_delivery_fee",
]


# =========================================================================
# ENV
# =========================================================================

def load_env():
    """Read .env if present, without needing python-dotenv."""
    env_file = Path("./.env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# =========================================================================
# HARVEST
# =========================================================================

def tavily_search(query, n=RESULTS_PER_QUERY, use_cache=True):
    """Cached search. Returns (results, was_cached)."""
    return W.tavily_search(query, api_key=os.environ["TAVILY_API_KEY"],
                           max_results=n, depth="advanced",
                           use_cache=use_cache)


def extract_pattern(article):
    """Local LLM turns an article into structured behaviours + signals."""
    prompt = f"""Read this article about telephone scams and extract the scam
pattern it describes.

ARTICLE TITLE: {article['title']}
ARTICLE TEXT: {article['content'][:2500]}

Answer in exactly this structure, no preamble:

BEHAVIOURS: <what the caller does and asks for, 2-3 sentences>
SIGNALS: <what distinguishes this from a legitimate call, 2-3 sentences>

If the article does not describe a telephone scam pattern, respond with exactly:
NOT_APPLICABLE"""

    text = W.call_ollama(prompt, max_tokens=350)
    if "NOT_APPLICABLE" in text.upper():
        return None

    beh = re.search(r"BEHAVIOURS:\s*(.+?)(?=SIGNALS:|$)", text,
                    re.IGNORECASE | re.DOTALL)
    sig = re.search(r"SIGNALS:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not beh:
        return None

    behaviours = beh.group(1).strip()
    signals = sig.group(1).strip() if sig else ""
    if len(behaviours) < 40:
        return None
    return behaviours, signals


def embeddable_text(category, behaviours, signals):
    return (f"Scam category: {category.replace('_', ' ')}\n\n"
            f"Typical caller behaviours: {behaviours}\n\n"
            f"Distinguishing signals: {signals}")


# =========================================================================
# JSON STORE
# =========================================================================

def load_kb():
    if KB_JSON.exists():
        return json.loads(KB_JSON.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created": None, "last_refresh": None,
            "refresh_count": 0, "patterns": []}


def save_kb(kb):
    KB_JSON.write_text(json.dumps(kb, indent=2, ensure_ascii=False),
                       encoding="utf-8")


def find_by_url(kb, url):
    return next((p for p in kb["patterns"] if p["source"]["url"] == url), None)


def cred_block(art, cred):
    return {
        "trust": round(art["trust"], 3),
        "recency": round(art["recency"], 3),
        "corroboration": round(art["corroboration"], 3),
        "combined": round(cred, 3),
    }


def next_id(kb, category):
    n = sum(1 for p in kb["patterns"] if p["category"] == category)
    return f"{category}_{n + 1:04d}"


# =========================================================================
# REPORTING
# =========================================================================

def print_stats(kb):
    pats = kb["patterns"]
    print("=" * 74)
    print(f"KB: {KB_JSON}")
    print("=" * 74)
    if not pats:
        print("  empty - run a harvest first")
        return
    print(f"  patterns:      {len(pats)}")
    print(f"  refreshes:     {kb.get('refresh_count', 0)}")
    print(f"  last refresh:  {kb.get('last_refresh', 'never')}")

    by_cat, by_dom, creds = {}, {}, []
    for p in pats:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
        d = p["source"]["domain"]
        by_dom[d] = by_dom.get(d, 0) + 1
        creds.append(p["credibility"]["combined"])

    print("\n  by category:")
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}  {c}")

    print(f"\n  distinct source domains: {len(by_dom)}")
    for d, n in sorted(by_dom.items(), key=lambda x: -x[1])[:12]:
        print(f"    {n:>3}  {d}")

    print(f"\n  credibility: min {min(creds):.3f}  "
          f"mean {sum(creds)/len(creds):.3f}  max {max(creds):.3f}")

    # flag categories that look under-covered
    thin = [c for c, n in by_cat.items() if n < 2]
    if thin:
        print(f"\n  WARNING - thin coverage (<2 patterns): {', '.join(thin)}")
    print("=" * 74)


# =========================================================================
# MAIN
# =========================================================================

def main():
    load_env()

    unknown = [a for a in sys.argv[1:] if a not in VALID_FLAGS]
    if unknown:
        print(f"Unknown option(s): {' '.join(unknown)}")
        print(f"Valid options: {' '.join(sorted(VALID_FLAGS))}")
        return

    dry_run = "--dry-run" in sys.argv
    stats_only = "--stats" in sys.argv
    use_cache = "--no-cache" not in sys.argv

    kb = load_kb()

    if stats_only:
        print_stats(kb)
        return

    print("=" * 74)
    print("Harvesting scam patterns from the web")
    print("=" * 74)

    by_cat = {}
    for cat, _ in SEED_QUERIES:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"\nSeed queries ({len(SEED_QUERIES)}) across "
          f"{len(by_cat)} categories:")
    for cat, q in SEED_QUERIES:
        print(f"  {cat:<26} <- \"{q[:52]}\"")

    print("\nExcluded (novel split for RQ1):")
    for c in NOVEL_CATEGORIES:
        print(f"  - {c}")

    print(f"\nCredibility floor: {KB_CREDIBILITY_FLOOR}")
    print(f"Cache: {'enabled' if use_cache else 'DISABLED (--no-cache)'}"
          f"  ttl {W.CACHE_TTL_DAYS} days")
    print(f"Existing patterns in KB: {len(kb['patterns'])}")

    if dry_run:
        n_new = sum(1 for q in SEED_QUERIES)
        print(f"\n--dry-run: stopping before any network call.")
        print(f"  a full run would issue up to {n_new} queries "
              f"({n_new * 2} credits if none are cached)")
        return

    if not os.environ.get("TAVILY_API_KEY"):
        print("\nERROR: TAVILY_API_KEY not found in .env or environment.")
        return

    print("\nChecking Ollama...")
    try:
        W.call_ollama("Reply with one word: ready", 5)
        print("  ok")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    stats = {"retrieved": 0, "below_floor": 0, "no_pattern": 0,
             "new": 0, "updated": 0, "api_queries": 0, "cached_queries": 0}

    for category, query in SEED_QUERIES:
        print(f"\n[{category}] \"{query[:46]}\"")
        try:
            articles, was_cached = tavily_search(query, use_cache=use_cache)
        except Exception as e:
            print(f"  search failed: {e}")
            continue

        print(f"  {'(from cache)' if was_cached else '(API call - 2 credits)'}")
        stats["retrieved"] += len(articles)
        if was_cached:
            stats["cached_queries"] += 1
        else:
            stats["api_queries"] += 1

        scored = W.score_credibility(articles, now=now)

        for art in scored:
            dom = W.domain_of(art["url"])
            cred = art["credibility"]

            if cred < KB_CREDIBILITY_FLOOR:
                stats["below_floor"] += 1
                print(f"  drop  {dom:<32} {cred:.2f}")
                continue

            # already known? refresh metadata, skip the LLM entirely
            existing = find_by_url(kb, art["url"])
            if existing:
                existing["credibility"] = cred_block(art, cred)
                existing["last_seen"] = now_iso
                existing["seen_count"] = existing.get("seen_count", 1) + 1
                stats["updated"] += 1
                print(f"  seen  {dom:<32} {cred:.2f}")
                continue

            extracted = extract_pattern(art)
            if not extracted:
                stats["no_pattern"] += 1
                print(f"  skip  {dom:<32} {cred:.2f}  (no pattern)")
                continue

            behaviours, signals = extracted
            kb["patterns"].append({
                "id": next_id(kb, category),
                "category": category,
                "behaviours": behaviours,
                "signals": signals,
                "text": embeddable_text(category, behaviours, signals),
                "source": {
                    "url": art["url"],
                    "domain": dom,
                    "title": art["title"],
                    "published_date": art.get("published_date") or "",
                },
                "credibility": cred_block(art, cred),
                "first_seen": now_iso,
                "last_seen": now_iso,
                "seen_count": 1,
            })
            stats["new"] += 1
            print(f"  NEW   {dom:<32} {cred:.2f}")

    kb["created"] = kb.get("created") or now_iso
    kb["last_refresh"] = now_iso
    kb["refresh_count"] = kb.get("refresh_count", 0) + 1
    save_kb(kb)

    print("\n" + "=" * 74)
    print("HARVEST COMPLETE")
    print("=" * 74)
    print(f"  articles retrieved:        {stats['retrieved']}")
    print(f"  dropped below {KB_CREDIBILITY_FLOOR}:         {stats['below_floor']}")
    print(f"  already known (skipped):   {stats['updated']}")
    print(f"  no pattern extractable:    {stats['no_pattern']}")
    print(f"  new patterns added:        {stats['new']}")
    print(f"\n  queries served from cache: {stats['cached_queries']}")
    print(f"  queries hitting the API:   {stats['api_queries']}"
          f"  ({stats['api_queries'] * 2} credits)")
    print(f"\n  KB now holds {len(kb['patterns'])} patterns")
    print(f"  saved to {KB_JSON}")
    print("\nTavily usage to date:")
    W.print_quota()
    print("\nRun --stats to see category coverage.")
    print("=" * 74)


if __name__ == "__main__":
    main()
