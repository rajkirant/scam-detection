"""
Five-system evaluation on the Singh-style multi-turn call transcripts.

Reads the 15 (or however many) transcript files in ./transcripts/ plus
labels.csv, and runs all five systems on them - the same five compared on the
Zhi dataset, so the two results are directly comparable:

    1. length-only     guess Fraud if transcript is long (no reading)
    2. bag-of-words    TF-IDF + logistic regression, leave-one-out (tiny set)
    3. LLM-only        model decides alone, no retrieval
    4. Singh baseline  model + retrieved bank policy (Singh's exact prompt)
    5. Web-RAG         your system, KB-only (use_web=False)

Web retrieval is OFF (KB-only), matching the Zhi comparison.

NOTE: this set is small (15) and easy - expect every LLM-based system to score
high and the systems not to separate much. Its value is the matched, like-for-
like comparison against the Zhi table, not discrimination between approaches.

Run from project root:
    python scripts/singh_style_evaluate.py
    python scripts/singh_style_evaluate.py --trivial-only
"""

import re
import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import credibility as C

TRANSCRIPTS_DIR = Path("./transcripts")
LABELS_FILE = Path("./labels.csv")
RESULTS_DIR = Path("./results")

CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_CHUNKS = 3


# =========================================================================
# LOAD
# =========================================================================

def load_data():
    """Return list of (filename, transcript_text, true_label, bank)."""
    if not LABELS_FILE.exists():
        print(f"ERROR: {LABELS_FILE} not found.")
        return None
    if not TRANSCRIPTS_DIR.exists():
        print(f"ERROR: {TRANSCRIPTS_DIR} not found.")
        return None

    labels = {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = row

    data = []
    for tf in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        fn = tf.name
        if fn not in labels:
            print(f"  (skipping {fn} - not in labels.csv)")
            continue
        text = tf.read_text(encoding="utf-8")
        data.append((fn, text, labels[fn]["label"], labels[fn].get("bank", "")))
    return data


# =========================================================================
# METRICS
# =========================================================================

def metrics(rows):
    """rows = list of (pred, true)."""
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
    print(f"  {name:<24} acc {m['acc']*100:5.1f}%  P {m['prec']:.3f}  "
          f"R {m['rec']:.3f}  F1 {m['f1']:.3f}   "
          f"(TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']})")


# =========================================================================
# 1. LENGTH-ONLY
# =========================================================================

def run_length(data, threshold_words=120):
    """Multi-turn transcripts are longer than single lines, so the threshold
    is higher than the Zhi one. Fraud calls tend to run longer (more back and
    forth to extract information)."""
    return [("Fraud" if len(text.split()) > threshold_words else "Normal", true)
            for _, text, true, _ in data]


# =========================================================================
# 2. BAG-OF-WORDS  (leave-one-out, because the set is tiny)
# =========================================================================

def run_bow(data):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        import numpy as np
    except ImportError:
        print("  (sklearn not installed - skipping bag-of-words)")
        return None

    texts = [t for _, t, _, _ in data]
    y = np.array([1 if lab == "Fraud" else 0 for _, _, lab, _ in data])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts).toarray()

    # Leave-one-out: with only ~15 samples, k-fold CV is unstable, LOO is honest.
    loo = LeaveOneOut()
    preds = [0] * len(y)
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = clf.predict(X[test_idx])[0]

    return [("Fraud" if p == 1 else "Normal", "Fraud" if t == 1 else "Normal")
            for p, t in zip(preds, y)]


# =========================================================================
# 3. LLM-ONLY  (no retrieval)
# =========================================================================

def run_llm_only(data):
    out = []
    for i, (fn, text, true, _) in enumerate(data, 1):
        prompt = (f"Is the following phone call a scam attempt? "
                  f"Answer with only one word, 'Fraud' or 'Normal', on the "
                  f"first line.\n\nCall:\n{text}")
        r = C.call_ollama(prompt, max_tokens=10)
        low = r.lower()
        pred = "Fraud" if (("fraud" in low and "normal" not in low)
                           or low.strip().startswith("fraud")) else "Normal"
        out.append((pred, true))
        print(f"    LLM-only: {i}/{len(data)}  {fn}  -> {pred}")
    return out


# =========================================================================
# 4. SINGH BASELINE  (Singh's exact logic, reused)
# =========================================================================

def _get_collection(name):
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=name, embedding_function=ef)


def _extract_bank(text):
    prompt = (f"Identify which bank the caller claims to represent. "
              f"One of: Bank1, Bank2, Bank3.\n\nTranscript:\n{text}\n\n"
              f"Respond with ONLY the bank name, or Unknown.")
    ans = C.call_ollama(prompt, max_tokens=10)
    for b in ["Bank1", "Bank2", "Bank3"]:
        if b.lower() in ans.lower():
            return b
    return "Unknown"


def run_singh(data):
    coll = _get_collection("bank_policies")
    out = []
    for i, (fn, text, true, _) in enumerate(data, 1):
        bank = _extract_bank(text)
        if bank == "Unknown":
            res = coll.query(query_texts=[text], n_results=N_CHUNKS)
        else:
            res = coll.query(query_texts=[text], n_results=N_CHUNKS,
                             where={"bank": bank})
        policy = "\n\n---\n\n".join(res["documents"][0])
        prompt = (f"You are a policy inspector. Ensure the conversation complies "
                  f"with the policies provided. Don't use any external source of "
                  f"information other than the policies.\n\n"
                  f"Policies: {policy}\n\nConversation: {text}\n\n"
                  f"Q: Does the conversation break any policies? If yes, return "
                  f"'Fraud'. If no, return 'Normal'. Provide only the label in the "
                  f"first line starting with \"Answer:\".")
        r = C.call_ollama(prompt, max_tokens=200)
        if re.search(r"answer:\s*fraud", r, re.I):
            pred = "Fraud"
        elif re.search(r"answer:\s*normal", r, re.I):
            pred = "Normal"
        else:
            pred = "Fraud" if ("fraud" in r.lower()
                               and "normal" not in r.lower()) else "Normal"
        out.append((pred, true))
        print(f"    Singh: {i}/{len(data)}  {fn}  bank={bank}  -> {pred}")
    return out


# =========================================================================
# 5. WEB-RAG  (your system, KB-only)
# =========================================================================

def run_webrag(data):
    import webrag_system as W
    coll = W.get_kb_collection()
    out = []
    for i, (fn, text, true, _) in enumerate(data, 1):
        res = W.detect(text, coll, use_web=False, threshold=50)
        out.append((res["predicted"], true))
        print(f"    WebRAG: {i}/{len(data)}  {fn}  conf={res['confidence']}  "
              f"-> {res['predicted']}")
    return out


# =========================================================================
# MAIN
# =========================================================================

def main():
    trivial_only = "--trivial-only" in sys.argv

    data = load_data()
    if not data:
        return

    n_fraud = sum(1 for _, _, l, _ in data if l == "Fraud")
    print("=" * 74)
    print(f"Singh-style evaluation - {len(data)} calls "
          f"({n_fraud} Fraud, {len(data)-n_fraud} Normal)")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\nTrivial reference classifiers (no LLM):")
    results["length"] = run_length(data)
    show("length-only", metrics(results["length"]))
    bow = run_bow(data)
    if bow:
        results["bow"] = bow
        show("bag-of-words (LOO)", metrics(bow))

    if trivial_only:
        print("\n--trivial-only: stopping before the LLM systems.")
        return

    print("\nLLM-only (no retrieval):")
    t0 = time.time()
    results["llm_only"] = run_llm_only(data)
    show("LLM-only", metrics(results["llm_only"]))
    print(f"    ({time.time()-t0:.0f}s)")

    print("\nSingh baseline (policy compliance):")
    t0 = time.time()
    results["singh"] = run_singh(data)
    show("Singh baseline", metrics(results["singh"]))
    print(f"    ({time.time()-t0:.0f}s)")

    print("\nWeb-RAG (KB-only):")
    t0 = time.time()
    results["webrag"] = run_webrag(data)
    show("Web-RAG (KB-only)", metrics(results["webrag"]))
    print(f"    ({time.time()-t0:.0f}s)")

    # save per-call predictions
    keys = list(results.keys())
    out_csv = RESULTS_DIR / f"singh_style_results_{len(data)}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true"] + keys)
        for i in range(len(data)):
            fn = data[i][0]
            true = results[keys[0]][i][1]
            w.writerow([fn, true] + [results[k][i][0] for k in keys])

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in keys:
        show(k, metrics(results[k]))
    print(f"\n  per-call results: {out_csv}")
    print("=" * 74)


if __name__ == "__main__":
    main()
