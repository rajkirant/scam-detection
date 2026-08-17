#!/usr/bin/env python3
"""
combined_evaluate.py

Run all FIVE systems on the combined delexicalised mixed dataset
(combined_delex_dataset.csv: 243 YouTube-delex scam + 400 Zhi-template non-scam)
and report accuracy / precision / recall / F1 for each.

Systems (same escalation ladder as zhi_evaluate.py):
    1. length-only     trivial word-count threshold
    2. bag-of-words    TF-IDF + LogisticRegression, 5-fold CV
    3. LLM-only        llama3.1:8b decides alone, NO retrieval  (the control)
    4. Singh           policy-compliance vs bank_policies collection
    5. Web-RAG         your system, KB-only (use_web=False)

READ THIS BEFORE TRUSTING THE NUMBERS
-------------------------------------
This dataset was shown to be separable at ~97% by BOTH bag-of-words AND
length-only, because the two halves differ in REGISTER (spoken YouTube speech
vs written Zhi templates), not in scam content. So a high score from ANY system
here is very likely reading the source artefact, not detecting scams. The whole
point of running length-only and bag-of-words alongside the LLM systems is to
make that visible: if the trivial baselines score as high as the LLM systems,
the comparison is measuring style, not detection. Report the baselines next to
the systems every time.

Runs on the RAW templated text as-is (slots like [Company] left unfilled) per
the chosen configuration. Web retrieval OFF (KB-only) for all LLM systems.

Usage (from project root, dataset in ./datasets/):
    python scripts/combined_evaluate.py --csv ./datasets/combined_delex_dataset.csv --limit 40
    python scripts/combined_evaluate.py --csv ./datasets/combined_delex_dataset.csv           # full 643 (slow)
    python scripts/combined_evaluate.py --csv ./datasets/combined_delex_dataset.csv --trivial-only
    python scripts/combined_evaluate.py --csv ./datasets/combined_delex_dataset.csv --skip singh   # skip a system
"""

import csv
import re
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")


# ------------------------------------------------------------------ loading
def load_combined(csv_path, limit=None):
    """Read combined_delex_dataset.csv -> list of (text, true_label).

    label column is 'scam'/'nonscam'; map to 'Fraud'/'Normal' to match the
    other eval scripts' convention.
    """
    p = Path(csv_path)
    if not p.exists():
        sys.exit("ERROR: not found: %s" % csv_path)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    data = []
    for r in rows:
        lab = "Fraud" if r["label"].strip().lower() == "scam" else "Normal"
        text = r["text"].strip()
        if text:
            data.append((text, lab))
    if limit:
        # balanced sample
        scam = [d for d in data if d[1] == "Fraud"]
        norm = [d for d in data if d[1] == "Normal"]
        k = limit // 2
        random.seed(42)
        scam = random.sample(scam, min(k, len(scam)))
        norm = random.sample(norm, min(k, len(norm)))
        data = scam + norm
    random.seed(42)
    random.shuffle(data)
    return data


# ------------------------------------------------------------------ metrics
def metrics(rows):
    tp = fp = fn = tn = 0
    for pred, true in rows:
        if pred == "Fraud" and true == "Fraud":     tp += 1
        elif pred == "Fraud" and true == "Normal":  fp += 1
        elif pred == "Normal" and true == "Fraud":  fn += 1
        elif pred == "Normal" and true == "Normal": tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def show(name, m):
    print("  %-24s acc %5.1f%%  P %.3f  R %.3f  F1 %.3f   (TP%d FP%d FN%d TN%d)" %
          (name, m["acc"] * 100, m["prec"], m["rec"], m["f1"],
           m["tp"], m["fp"], m["fn"], m["tn"]))


# ------------------------------------------ trivial baselines (no LLM, instant)
def trivial_length(data, threshold=45):
    return [("Fraud" if len(t.split()) > threshold else "Normal", lab)
            for t, lab in data]


def trivial_bow(data):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        print("  (sklearn not installed - skipping bag-of-words)")
        return None
    texts = [t for t, _ in data]
    y = np.array([1 if lab == "Fraud" else 0 for _, lab in data])
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5)
    return [("Fraud" if p == 1 else "Normal", "Fraud" if t == 1 else "Normal")
            for p, t in zip(pred, y)]


# --------------------------------------------------- LLM systems (need Ollama)
def run_llm_only(data):
    """Control: model decides alone, no retrieval. Answers 'is this a scam call?'"""
    import credibility as C
    out = []
    for i, (text, true) in enumerate(data, 1):
        prompt = (
            "You are a scam detection analyst. Read this phone call transcript and "
            "decide whether the caller is attempting a scam.\n\n"
            "Transcript:\n%s\n\n"
            "Answer 'Fraud' or 'Normal' on the first line, starting with 'Answer:'."
            % text)
        r = C.call_ollama(prompt, max_tokens=60)
        pred = "Fraud" if re.search(r"answer:\s*fraud", r, re.I) or \
               (("fraud" in r.lower()) and ("normal" not in r.lower())) else "Normal"
        out.append((pred, true))
        if i % 20 == 0:
            print("    LLM-only: %d/%d" % (i, len(data)))
    return out


def run_singh(data):
    """Singh baseline: policy-compliance check vs bank_policies collection."""
    import chromadb
    from chromadb.utils import embedding_functions
    import credibility as C

    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    coll = client.get_collection(name="bank_policies", embedding_function=ef)

    out = []
    for i, (text, true) in enumerate(data, 1):
        res = coll.query(query_texts=[text], n_results=3)
        policy = "\n\n".join(res["documents"][0])
        prompt = ("You are a policy inspector. Using ONLY these policies:\n%s\n\n"
                  "Conversation: %s\n\n"
                  "Does the conversation break any policy? Answer 'Fraud' or 'Normal' "
                  "on the first line starting with 'Answer:'." % (policy, text))
        r = C.call_ollama(prompt, max_tokens=60)
        pred = "Fraud" if re.search(r"answer:\s*fraud", r, re.I) or \
               (("fraud" in r.lower()) and ("normal" not in r.lower())) else "Normal"
        out.append((pred, true))
        if i % 20 == 0:
            print("    Singh: %d/%d" % (i, len(data)))
    return out


def run_webrag(data):
    """Web-RAG system, KB-only (use_web=False)."""
    import webrag_system as W
    coll = W.get_kb_collection()
    out = []
    for i, (text, true) in enumerate(data, 1):
        res = W.detect(text, coll, use_web=False, threshold=50)
        out.append((res["predicted"], true))
        if i % 20 == 0:
            print("    WebRAG: %d/%d" % (i, len(data)))
    return out


# ------------------------------------------------------------------ main
def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    csv_path = "./datasets/combined_delex_dataset.csv"
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]
    trivial_only = "--trivial-only" in sys.argv
    skip = set()
    if "--skip" in sys.argv:
        skip = set(sys.argv[sys.argv.index("--skip") + 1].split(","))

    data = load_combined(csv_path, limit=limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")
    print("=" * 74)
    print("Combined dataset evaluation - %d calls (%d scam, %d non-scam)" %
          (len(data), n_fraud, len(data) - n_fraud))
    print("  source: %s" % csv_path)
    print("  NOTE: this mix separates ~97%% on register (spoken vs written).")
    print("  Watch whether the LLM systems beat the trivial baselines or just match them.")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\nTrivial reference classifiers (no LLM):")
    results["length"] = trivial_length(data)
    show("length-only (>45 words)", metrics(results["length"]))
    bow = trivial_bow(data)
    if bow:
        results["bow"] = bow
        show("bag-of-words (TF-IDF+LR)", metrics(bow))

    if trivial_only:
        print("\n--trivial-only: stopping before the LLM systems.")
        _save(results, data)
        return

    if "llm_only" not in skip:
        print("\nLLM-only (no retrieval):")
        t0 = time.time()
        results["llm_only"] = run_llm_only(data)
        show("LLM-only", metrics(results["llm_only"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "singh" not in skip:
        print("\nSingh baseline (policy compliance):")
        t0 = time.time()
        results["singh"] = run_singh(data)
        show("Singh baseline", metrics(results["singh"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "webrag" not in skip:
        print("\nWeb-RAG system (KB-only):")
        t0 = time.time()
        results["webrag"] = run_webrag(data)
        show("Web-RAG (KB-only)", metrics(results["webrag"]))
        print("    (%.0fs)" % (time.time() - t0))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in results:
        show(k, metrics(results[k]))
    _save(results, data)


def _save(results, data):
    if not results:
        return
    keys = list(results.keys())
    out_csv = RESULTS_DIR / ("combined_results_%d.csv" % len(data))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true"] + keys)
        for i in range(len(data)):
            true = results[keys[0]][i][1]
            w.writerow([i, true] + [results[k][i][0] for k in keys])
    print("\n  per-call results: %s" % out_csv)
    print("=" * 74)


if __name__ == "__main__":
    main()
