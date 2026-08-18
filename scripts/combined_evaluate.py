#!/usr/bin/env python3
"""
combined_evaluate.py

Run all FIVE systems on a scam/non-scam transcript dataset and report
accuracy / precision / recall / F1 for each.

Systems (escalation ladder):
    1. length-only     trivial word-count threshold
    2. bag-of-words    TF-IDF + LogisticRegression, 5-fold CV
    3. LLM-only        the model decides alone, NO retrieval  (the control)
    4. Singh           policy-compliance vs bank_policies collection
    5. Web-RAG         this project's system, KB-only (use_web=False)

CHANGES IN THIS VERSION
-----------------------
1. BAG-OF-WORDS LEAK FIXED. The TF-IDF vectoriser used to be fitted on the
   whole dataset before cross-validation, so vocabulary selection and IDF
   weights had seen the held-out fold. It is now inside a Pipeline, refitted
   from scratch on each training fold. Splits are also explicitly stratified
   and seeded.

2. VERDICT PARSING FIXED. The old rule was:
       pred = "Fraud" if <marker or fraud-and-not-normal> else "Normal"
   Any answer the regex could not read silently became "Normal", i.e. a scam
   call recorded as safe. Combined with max_tokens=60, a verbose model
   (qwen2.5:14b) could still be reasoning when it ran out of tokens, never
   reach its verdict, and be scored as a false negative. That is the most
   likely cause of Singh's recall collapsing from ~0.99 on llama3.1:8b to
   0.757 on qwen2.5:14b.
   Now: a shared parser tries several formats, retries once with a
   one-word-answer prompt if it fails, and every unreadable response is
   COUNTED AND REPORTED rather than quietly becoming "Normal".

3. TOKEN BUDGET RAISED. max_tokens default 60 -> 300 for the LLM systems.
   NOTE: this changes results relative to earlier runs. Re-run any baseline
   you intend to compare against.

Usage:
    python scripts/combined_evaluate.py --csv datasets/zhi_scam_vs_legit_794.csv
    python scripts/combined_evaluate.py --csv datasets/... --limit 40
    python scripts/combined_evaluate.py --csv datasets/... --debug
    python scripts/combined_evaluate.py --csv datasets/... --bow-features
    python scripts/combined_evaluate.py --csv datasets/... --trivial-only
    python scripts/combined_evaluate.py --csv datasets/... --skip singh
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")
SEED = 42


# ------------------------------------------------------------------ loading
def load_combined(csv_path, limit=None):
    """Read a dataset CSV -> list of (text, true_label).

    Accepts label values scam/nonscam/fraud/normal/legit and maps to
    'Fraud'/'Normal' to match the other eval scripts' convention.
    """
    p = Path(csv_path)
    if not p.exists():
        sys.exit("ERROR: not found: %s" % csv_path)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        sys.exit("ERROR: empty CSV: %s" % csv_path)
    if "label" not in rows[0] or "text" not in rows[0]:
        sys.exit("ERROR: CSV needs 'label' and 'text' columns. Found: %s"
                 % list(rows[0].keys()))

    scam_words = {"scam", "fraud", "fraudulent", "1", "true", "yes"}
    data = []
    for r in rows:
        lab_raw = (r["label"] or "").strip().lower()
        lab = "Fraud" if lab_raw in scam_words else "Normal"
        text = (r["text"] or "").strip()
        if text:
            data.append((text, lab))

    if limit:
        # balanced sample
        scam = [d for d in data if d[1] == "Fraud"]
        norm = [d for d in data if d[1] == "Normal"]
        k = limit // 2
        random.seed(SEED)
        scam = random.sample(scam, min(k, len(scam)))
        norm = random.sample(norm, min(k, len(norm)))
        data = scam + norm
    random.seed(SEED)
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


# ------------------------------------------------------- verdict parsing
FRAUD_WORDS = r"fraud|scam|fraudulent|suspicious"
NORMAL_WORDS = r"normal|legitimate|legit|genuine|safe|not\s+a\s+scam"


def parse_verdict(raw):
    """Read a Fraud/Normal verdict out of a model response.

    Returns "Fraud", "Normal", or None if the response cannot be read.
    None is the important case: the old code turned it into "Normal", which
    silently converted unreadable answers into false negatives.
    """
    if not raw or not raw.strip():
        return None
    low = raw.lower()

    # 1. explicit answer marker, tolerating markdown and punctuation
    m = re.search(r"answer\s*[:\-]?\s*[*_`\"'\s]*(" + FRAUD_WORDS + "|"
                  + NORMAL_WORDS + r")\b", low)
    if m:
        return "Fraud" if re.match(FRAUD_WORDS, m.group(1)) else "Normal"

    # 2. first non-empty line only (the prompt asks for the verdict there)
    lines = [ln for ln in low.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        f = re.search(r"\b(" + FRAUD_WORDS + r")\b", first)
        n = re.search(r"\b(" + NORMAL_WORDS + r")\b", first)
        if f and not n:
            return "Fraud"
        if n and not f:
            return "Normal"

    # 3. whole response, but only if exactly one side appears
    f = re.search(r"\b(" + FRAUD_WORDS + r")\b", low)
    n = re.search(r"\b(" + NORMAL_WORDS + r")\b", low)
    if f and not n:
        return "Fraud"
    if n and not f:
        return "Normal"

    return None


def looks_truncated(raw):
    """Heuristic: response ended mid-thought rather than concluding."""
    if not raw:
        return False
    t = raw.strip()
    return bool(t) and t[-1] not in ".!?\"')}]" and len(t.split()) > 20


class VerdictStats:
    """Track how the parser coped, so a parsing bug can never hide again."""

    def __init__(self, name):
        self.name = name
        self.ok = 0
        self.retried = 0
        self.failed = 0
        self.truncated = 0
        self.samples = []

    def record(self, status, raw, truncated):
        setattr(self, status, getattr(self, status) + 1)
        if truncated:
            self.truncated += 1
        if status in ("retried", "failed") and len(self.samples) < 5:
            self.samples.append(raw[:300])

    def report(self):
        total = self.ok + self.retried + self.failed
        if total == 0:
            return
        if self.retried or self.failed or self.truncated:
            print("    parsing: %d clean, %d needed retry, %d UNREADABLE, "
                  "%d look truncated" %
                  (self.ok, self.retried, self.failed, self.truncated))
            if self.failed:
                print("    WARNING: %d unreadable responses were scored 'Normal'."
                      % self.failed)
                print("    If these were scam calls they are counted as false "
                      "negatives. Raise --max-tokens or inspect with --debug.")
            for s in self.samples[:2]:
                print("      sample: %r" % s)
        else:
            print("    parsing: all %d responses read cleanly" % total)


def ask_verdict(prompt, stats, max_tokens=300, debug=False, idx=None):
    """Call the model, parse a verdict, retry once if unreadable."""
    import credibility as C

    raw = C.call_ollama(prompt, max_tokens=max_tokens)
    truncated = looks_truncated(raw)
    verdict = parse_verdict(raw)

    if verdict is not None:
        stats.record("ok", raw, truncated)
        if debug and idx is not None and idx <= 5:
            print("      [%d] %s <- %r" % (idx, verdict, (raw or "")[:160]))
        return verdict, raw

    # retry, forcing a one-word answer
    retry_prompt = prompt + "\n\nReply with exactly one word, either Fraud or Normal. No explanation."
    raw2 = C.call_ollama(retry_prompt, max_tokens=10)
    verdict2 = parse_verdict(raw2)
    if verdict2 is not None:
        stats.record("retried", raw, truncated)
        if debug:
            print("      [%s] retry rescued -> %s (first reply: %r)"
                  % (idx, verdict2, (raw or "")[:120]))
        return verdict2, raw

    stats.record("failed", raw, truncated)
    if debug:
        print("      [%s] UNREADABLE -> defaulting Normal: %r" % (idx, (raw or "")[:160]))
    return "Normal", raw


# ------------------------------------------ trivial baselines (no LLM, instant)
def trivial_length(data, threshold=45):
    return [("Fraud" if len(t.split()) > threshold else "Normal", lab)
            for t, lab in data]


def trivial_bow(data, folds=5, show_features=False):
    """TF-IDF + LogisticRegression, properly cross-validated.

    The vectoriser lives inside the Pipeline so it is refitted on each
    training fold. Fitting it on the full dataset first (the previous
    behaviour) leaked test-fold vocabulary and IDF weights into training.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        import numpy as np
    except ImportError:
        print("  (sklearn not installed - skipping bag-of-words)")
        return None

    texts = [t for t, _ in data]
    y = np.array([1 if lab == "Fraud" else 0 for _, lab in data])

    smallest_class = int(min(y.sum(), len(y) - y.sum()))
    n_splits = max(2, min(folds, smallest_class))
    if n_splits < folds:
        print("  (only %d per smallest class, using %d folds)"
              % (smallest_class, n_splits))

    def build(min_df):
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=min_df),
            LogisticRegression(max_iter=2000))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    try:
        pred = cross_val_predict(build(2), texts, y, cv=cv)
    except ValueError:
        # min_df=2 can empty the vocabulary on very small folds
        print("  (min_df=2 left no vocabulary in a fold, falling back to min_df=1)")
        pred = cross_val_predict(build(1), texts, y, cv=cv)

    if show_features:
        _bow_top_features(texts, y)

    return [("Fraud" if p == 1 else "Normal", "Fraud" if t == 1 else "Normal")
            for p, t in zip(pred, y)]


def _bow_top_features(texts, y, k=15):
    """Diagnostic only: which words is BoW actually keying on?

    Fitted on the full dataset deliberately - this is for inspection, not
    for scoring, and is never used to produce predictions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_scam = names[np.argsort(coefs)[-k:]][::-1]
    top_legit = names[np.argsort(coefs)[:k]]
    print("\n  BoW top features (diagnostic, fitted on full data):")
    print("    -> Fraud : %s" % ", ".join(top_scam))
    print("    -> Normal: %s" % ", ".join(top_legit))
    print("    If these look like scam concepts, the signal is real.")
    print("    If they look like style artefacts, the dataset is separable on source.\n")


# --------------------------------------------------- LLM systems (need Ollama)
def run_llm_only(data, max_tokens=300, debug=False):
    """Control: model decides alone, no retrieval."""
    stats = VerdictStats("llm_only")
    out, raws = [], []
    for i, (text, true) in enumerate(data, 1):
        prompt = (
            "You are a scam detection analyst. Read this phone call transcript and "
            "decide whether the caller is attempting a scam.\n\n"
            "Transcript:\n%s\n\n"
            "Answer 'Fraud' or 'Normal' on the first line, starting with 'Answer:'."
            % text)
        pred, raw = ask_verdict(prompt, stats, max_tokens, debug, i)
        out.append((pred, true))
        raws.append(raw)
        if i % 20 == 0:
            print("    LLM-only: %d/%d" % (i, len(data)))
    stats.report()
    return out, raws


def run_singh(data, max_tokens=300, debug=False):
    """Singh baseline: policy-compliance check vs bank_policies collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    coll = client.get_collection(name="bank_policies", embedding_function=ef)

    stats = VerdictStats("singh")
    out, raws = [], []
    for i, (text, true) in enumerate(data, 1):
        res = coll.query(query_texts=[text], n_results=3)
        policy = "\n\n".join(res["documents"][0])
        prompt = ("You are a policy inspector. Using ONLY these policies:\n%s\n\n"
                  "Conversation: %s\n\n"
                  "Does the conversation break any policy? Answer 'Fraud' or 'Normal' "
                  "on the first line starting with 'Answer:'." % (policy, text))
        pred, raw = ask_verdict(prompt, stats, max_tokens, debug, i)
        out.append((pred, true))
        raws.append(raw)
        if i % 20 == 0:
            print("    Singh: %d/%d" % (i, len(data)))
    stats.report()
    return out, raws


def run_webrag(data, debug=False):
    """Web-RAG system, KB-only (use_web=False).

    NOTE: this delegates to webrag_system.detect(), which does its own
    prompting and parsing. The fixes in this file do NOT reach inside it.
    If Web-RAG's numbers also look odd on a verbose model, check the
    verdict parsing and token budget in webrag_system.py too.
    """
    import webrag_system as W
    coll = W.get_kb_collection()
    out, raws = [], []
    for i, (text, true) in enumerate(data, 1):
        res = W.detect(text, coll, use_web=False, threshold=50)
        out.append((res["predicted"], true))
        raws.append(json.dumps(res, default=str)[:500])
        if i % 20 == 0:
            print("    WebRAG: %d/%d" % (i, len(data)))
    return out, raws


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="./datasets/combined_delex_dataset.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trivial-only", action="store_true")
    ap.add_argument("--skip", default="", help="comma list: llm_only,singh,webrag")
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="token budget for LLM verdicts (was 60, which truncated "
                         "verbose models mid-answer)")
    ap.add_argument("--debug", action="store_true",
                    help="print raw model responses and save them to results/")
    ap.add_argument("--bow-features", action="store_true",
                    help="show which words BoW is keying on")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    data = load_combined(args.csv, limit=args.limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")

    print("=" * 74)
    print("Dataset evaluation - %d calls (%d scam, %d non-scam)" %
          (len(data), n_fraud, len(data) - n_fraud))
    print("  source: %s" % args.csv)
    print("  LLM max_tokens: %d" % args.max_tokens)
    print("  Compare every LLM system against the trivial baselines below.")
    print("  If length or bag-of-words matches the LLM systems, the dataset is")
    print("  separable without understanding scams, and the comparison is")
    print("  measuring the dataset, not the systems.")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    raw_log = {}

    print("\nTrivial reference classifiers (no LLM):")
    results["length"] = trivial_length(data)
    show("length-only (>45 words)", metrics(results["length"]))
    bow = trivial_bow(data, folds=args.folds, show_features=args.bow_features)
    if bow:
        results["bow"] = bow
        show("bag-of-words (TF-IDF+LR)", metrics(bow))

    if args.trivial_only:
        print("\n--trivial-only: stopping before the LLM systems.")
        _save(results, data, raw_log, args.debug)
        return

    if "llm_only" not in skip:
        print("\nLLM-only (no retrieval):")
        t0 = time.time()
        results["llm_only"], raw_log["llm_only"] = run_llm_only(
            data, args.max_tokens, args.debug)
        show("LLM-only", metrics(results["llm_only"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "singh" not in skip:
        print("\nSingh baseline (policy compliance):")
        t0 = time.time()
        results["singh"], raw_log["singh"] = run_singh(
            data, args.max_tokens, args.debug)
        show("Singh baseline", metrics(results["singh"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "webrag" not in skip:
        print("\nWeb-RAG system (KB-only):")
        t0 = time.time()
        results["webrag"], raw_log["webrag"] = run_webrag(data, args.debug)
        show("Web-RAG (KB-only)", metrics(results["webrag"]))
        print("    (%.0fs)" % (time.time() - t0))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in results:
        show(k, metrics(results[k]))
    _save(results, data, raw_log, args.debug)


def _save(results, data, raw_log=None, debug=False):
    if not results:
        return
    keys = list(results.keys())
    out_csv = RESULTS_DIR / ("combined_results_%d.csv" % len(data))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true", "text"] + keys)
        for i in range(len(data)):
            true = results[keys[0]][i][1]
            text = data[i][0].replace("\n", " ")[:400]
            w.writerow([i, true, text] + [results[k][i][0] for k in keys])
    print("\n  per-call results: %s" % out_csv)

    if debug and raw_log:
        out_raw = RESULTS_DIR / ("combined_raw_%d.json" % len(data))
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(raw_log, f, indent=2)
        print("  raw model responses: %s" % out_raw)
    print("=" * 74)


if __name__ == "__main__":
    main()