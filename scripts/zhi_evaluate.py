"""
Run both systems on the Zhi et al. dataset (single-line monologues).

The Zhi files are one scam/non-scam monologue per line, not multi-turn
transcripts. This adapter treats each line as one call, runs it through both
the Singh baseline and the Web-RAG system, and scores against the file label
(English_Scam.txt -> Fraud, English_NonScam.txt -> Normal).

Also runs two trivial reference classifiers - length-threshold and
bag-of-words - so the results show how hard the dataset actually is.

Web retrieval is OFF by default (800 calls would burn the Tavily quota and
KB-only is the cleaner comparison anyway).

Usage (from project root):
    python scripts/zhi_evaluate.py --limit 40     # quick sanity run, 20+20
    python scripts/zhi_evaluate.py                # full 800 (slow: ~1.5-2h)
    python scripts/zhi_evaluate.py --trivial-only # just the cheap baselines, no LLM
"""

import re
import sys
import csv
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SCAM_FILE = Path("./datasets/English_Scam.txt")
NONSCAM_FILE = Path("./datasets/English_NonScam.txt")
RESULTS_DIR = Path("./results")


def load_zhi(limit=None):
    """Return list of (text, true_label). Optionally balanced-sample `limit` total."""
    def read(path):
        return [re.sub(r'^\d+[\.\)]\s*', '', l.strip())
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    scam = [(t, "Fraud") for t in read(SCAM_FILE)]
    normal = [(t, "Normal") for t in read(NONSCAM_FILE)]
    if limit:
        k = limit // 2
        random.seed(42)
        scam = random.sample(scam, min(k, len(scam)))
        normal = random.sample(normal, min(k, len(normal)))
    data = scam + normal
    random.seed(42)
    random.shuffle(data)
    return data


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
    print(f"  {name:<28} acc {m['acc']*100:5.1f}%  P {m['prec']:.3f}  "
          f"R {m['rec']:.3f}  F1 {m['f1']:.3f}   "
          f"(TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']})")


# ---- trivial reference classifiers (no LLM, instant, no cost) -----------

def trivial_length(data, threshold=45):
    """Longer than threshold words -> guess Fraud. Measures length confound."""
    return [("Fraud" if len(t.split()) > threshold else "Normal", lab)
            for t, lab in data]


def trivial_bow(data):
    """Bag-of-words logistic regression, 5-fold CV pooled predictions."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        import numpy as np
    except ImportError:
        print("  (sklearn not installed - skipping bag-of-words baseline)")
        return None
    texts = [t for t, _ in data]
    y = np.array([1 if lab == "Fraud" else 0 for _, lab in data])
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5)
    return [("Fraud" if p == 1 else "Normal", "Fraud" if t == 1 else "Normal")
            for p, t in zip(pred, y)]


# ---- LLM-based systems --------------------------------------------------

def run_singh(data):
    """Singh baseline: policy-compliance check. Needs bank_policies collection."""
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
        prompt = (f"You are a policy inspector. Using ONLY these policies:\n{policy}\n\n"
                  f"Conversation: {text}\n\n"
                  f"Does the conversation break any policy? Answer 'Fraud' or 'Normal' "
                  f"on the first line starting with 'Answer:'.")
        r = C.call_ollama(prompt, max_tokens=60)
        pred = "Fraud" if re.search(r"answer:\s*fraud", r, re.I) or \
               (("fraud" in r.lower()) and ("normal" not in r.lower())) else "Normal"
        out.append((pred, true))
        if i % 20 == 0:
            print(f"    Singh: {i}/{len(data)}")
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
            print(f"    WebRAG: {i}/{len(data)}")
    return out


def run_llm_only(data):
    """
    LLM-only baseline: the model decides with NO retrieval, NO KB, NO policy.
    This is the control that shows whether retrieval adds anything over the
    bare model. If Singh and Web-RAG don't beat this, their retrieval isn't
    doing the work - the model is deciding from the transcript alone.
    """
    import credibility as C
    out = []
    for i, (text, true) in enumerate(data, 1):
        prompt = (f"Is the following phone call a scam attempt? "
                  f"Answer with only one word, 'Fraud' or 'Normal', "
                  f"on the first line.\n\nCall: {text}")
        r = C.call_ollama(prompt, max_tokens=10)
        low = r.lower()
        pred = "Fraud" if ("fraud" in low and "normal" not in low) or \
               low.strip().startswith("fraud") else "Normal"
        out.append((pred, true))
        if i % 20 == 0:
            print(f"    LLM-only: {i}/{len(data)}")
    return out


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    trivial_only = "--trivial-only" in sys.argv

    if not SCAM_FILE.exists():
        print(f"ERROR: {SCAM_FILE} not found. Put the Zhi files in ./datasets/")
        return

    data = load_zhi(limit=limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")
    print("=" * 74)
    print(f"Zhi dataset evaluation - {len(data)} calls "
          f"({n_fraud} scam, {len(data)-n_fraud} non-scam)")
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
        return

    print("\nLLM-only baseline (no retrieval):")
    t0 = time.time()
    results["llm_only"] = run_llm_only(data)
    show("LLM-only", metrics(results["llm_only"]))
    print(f"    ({time.time()-t0:.0f}s)")

    print("\nSingh baseline (policy compliance):")
    t0 = time.time()
    results["singh"] = run_singh(data)
    show("Singh baseline", metrics(results["singh"]))
    print(f"    ({time.time()-t0:.0f}s)")

    print("\nWeb-RAG system (KB-only):")
    t0 = time.time()
    results["webrag"] = run_webrag(data)
    show("Web-RAG (KB-only)", metrics(results["webrag"]))
    print(f"    ({time.time()-t0:.0f}s)")

    # save per-call predictions
    out_csv = RESULTS_DIR / f"zhi_results_{len(data)}.csv"
    keys = list(results.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true"] + keys)
        for i in range(len(data)):
            true = results[keys[0]][i][1]
            row = [i, true] + [results[k][i][0] for k in keys]
            w.writerow(row)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in keys:
        show(k, metrics(results[k]))
    print(f"\n  per-call results: {out_csv}")
    print("=" * 74)


if __name__ == "__main__":
    main()
