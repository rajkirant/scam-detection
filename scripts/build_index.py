"""
Build the vector index from knowledge/scam_patterns.json.

The JSON is the source of truth. This script derives the searchable index
from it - no web calls, no LLM, no re-harvesting. Re-run it whenever the JSON
changes (after a harvest) or when switching embedding models.

Each pattern becomes ONE vector. Patterns are already short and self-contained
(behaviours + signals, ~100 words), so they are not sub-chunked - the pattern
IS the retrieval unit.

Run from project root:
    python scripts/build_index.py
    python scripts/build_index.py --test    # rebuild, then run sample queries
"""

import sys
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

KB_JSON = Path(__file__).resolve().parent.parent / "knowledge" / "scam_patterns.json"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "scam_patterns"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_patterns():
    if not KB_JSON.exists():
        print(f"ERROR: {KB_JSON} not found. Run harvest_patterns.py first.")
        return None
    kb = json.loads(KB_JSON.read_text(encoding="utf-8"))
    return kb.get("patterns", [])


def build_index(patterns):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)

    # Rebuild from scratch so the index always matches the JSON exactly.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"  cleared existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    # Cosine space, so a distance converts to a similarity as plain 1 - d.
    # Chroma's default is squared L2, which needs the unit-vector assumption to
    # be interpretable - and webrag_system's relevance gate thresholds on
    # similarity, so the metric has to be unambiguous.
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=ef,
        metadata={"hnsw:space": "cosine"})

    documents, metadatas, ids = [], [], []
    for p in patterns:
        documents.append(p["text"])
        metadatas.append({
            "pattern_id": p["id"],
            "scam_type": p["category"],
            "domain": p["source"]["domain"],
            "url": p["source"]["url"],
            "credibility": p["credibility"]["combined"],
            "trust": p["credibility"]["trust"],
            "recency": p["credibility"]["recency"],
            "corroboration": p["credibility"]["corroboration"],
        })
        ids.append(p["id"])

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def test_queries(collection):
    """
    Sanity queries: known categories should hit, novel ones and ordinary
    legitimate calls should miss. PASS/drop shows what webrag_system's
    similarity floor would do with each hit - this is how you calibrate it.
    """
    from webrag_system import (MIN_KB_SIMILARITY, collection_space,
                               similarity_from_distance)

    checks = [
        ("caller demands full card number and CVV, threatens account freeze",
         "known - bank_impersonation should be top and PASS"),
        ("caller says my computer has a virus, asks to install remote software",
         "known - tech_support should be top and PASS"),
        ("caller used an AI clone of my grandson's voice asking for bail money",
         "NOVEL (ai_voice_cloning) - expect a POOR match"),
        ("caller says I must pay to release a parcel held at customs",
         "NOVEL (parcel_delivery_fee) - expect a POOR match"),
        ("caller confirms a dentist appointment and offers to reschedule it",
         "LEGITIMATE - every hit should be dropped, or the KB biases the prompt"),
    ]
    space = collection_space(collection)
    print(f"    (space={space}, similarity floor={MIN_KB_SIMILARITY:.2f})")
    for query, note in checks:
        res = collection.query(query_texts=[query], n_results=3)
        print(f"\n  query: \"{query[:60]}\"")
        print(f"    ({note})")
        for i in range(len(res["ids"][0])):
            md = res["metadatas"][0][i]
            dist = res["distances"][0][i]
            sim = similarity_from_distance(dist, space)
            verdict = "PASS" if sim >= MIN_KB_SIMILARITY else "drop"
            print(f"      {i+1}. {md['scam_type']:<26} "
                  f"sim {sim:.3f} [{verdict}]  dist {dist:.3f}  "
                  f"cred {md['credibility']:.2f}  [{md['domain']}]")


def main():
    do_test = "--test" in sys.argv

    print("=" * 70)
    print("Building vector index from knowledge/scam_patterns.json")
    print("=" * 70)

    patterns = load_patterns()
    if not patterns:
        return

    print(f"\n  {len(patterns)} patterns in JSON")

    by_cat = {}
    for p in patterns:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}  {c}")

    print(f"\n  Embedding ({EMBEDDING_MODEL_NAME}) and indexing...")
    collection = build_index(patterns)
    count = collection.count()
    print(f"  indexed {count} vectors into '{COLLECTION_NAME}'")

    if count != len(patterns):
        print(f"  WARNING: indexed {count} but JSON had {len(patterns)} - "
              f"check for duplicate ids")

    if do_test:
        print("\n  Running sanity queries...")
        test_queries(collection)
        print("\n  Interpretation: known categories should PASS the floor; the two")
        print("  NOVEL queries and the legitimate call should be dropped. That gap")
        print("  is what live web retrieval is meant to close (RQ1), and the floor")
        print("  is what stops an unmatched call being handed scam patterns anyway.")

    print("\n" + "=" * 70)
    print("Index built. Source of truth remains knowledge/scam_patterns.json.")
    print("Re-run this after any harvest, or when changing embedding model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
