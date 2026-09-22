#!/usr/bin/env python3
"""
bow_classify.py - train a bag-of-words classifier on scam calls, keep it, and
ask it whether one call is a scam.

The same shape as bert_classify.py, and deliberately so: train a model on a
dataset, keep it, put a transcript to it, get back the probability that the
call is a scam. Two pages, one question, and the comparison is the point.

What is under it is TF-IDF over word unigrams and bigrams into a logistic
regression - the same vectoriser and classifier the `bow` baseline in
combined_evaluate.py cross-validates, so a verdict here is the kind of verdict
that baseline produces. No embeddings, no attention, no GPU. It trains in
about a second.

Two things follow from that, and both matter when reading the two pages side
by side:

  it reads the whole call   BERT takes 512 tokens, so bert_classify cuts a
                            long transcript into windows and scores each.
                            TF-IDF has no length limit: every word in the
                            call is counted. So the two are not being asked
                            the same question of a long call, and BoW is the
                            one that saw all of it.

  it can be read back       A linear model over TF-IDF decomposes exactly:
                            the decision is the intercept plus, for every
                            word, its tf-idf weight times its coefficient.
                            So "why" is not an approximation here - the
                            words that decided this call are listed with
                            what each contributed, and they sum to the
                            score. That is the thing BERT cannot give you,
                            and it is what makes this baseline worth having
                            rather than just worth beating.

If this scores near a fine-tuned BERT on a dataset, that dataset is separable
on vocabulary and neither number is evidence about understanding scams.

Usage:
    python scripts/bow_classify.py train --csv datasets/zhi_english_646.csv \
        --name zhi-bow
    python scripts/bow_classify.py classify --name zhi-bow --text "Hello..."
    python scripts/bow_classify.py classify --name zhi-bow \
        --csv datasets/scambait_bank_422.csv --idx 3 --json
    python scripts/bow_classify.py models
    python scripts/bow_classify.py serve --name zhi-bow    # one JSON per line

scripts/web_ui.py drives all of these from the Bag of words page.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared with the BERT page rather than copied: the tone-tag map is a list of
# real strings that would drift if it lived in two files, and a row is a row.
# Both are standard library only, so this costs nothing.
from bert_classify import NAME_RE, checked_name, model_dir, strip_tags, \
    transcript_from_csv                                    # noqa: E402

# What marks a directory in models/ as one of these rather than a BERT
# checkpoint. bert_classify looks for config.json and skips anything without
# it, so the two kinds sit side by side without seeing each other.
MODEL_FILE = "bow.joblib"
META_FILE = "meta.json"


def read_meta(d):
    try:
        return json.loads((d / META_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_models():
    """Every bag-of-words model in models/, newest first."""
    out = []
    if not MODELS_DIR.is_dir():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / MODEL_FILE).exists():
            continue
        meta = read_meta(d)
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out.append({
            "name": d.name,
            "dataset": meta.get("dataset"),
            "rows": meta.get("rows"),
            "holdout": meta.get("holdout"),
            "trained_at": meta.get("trained_at"),
            "elapsed_s": meta.get("elapsed_s"),
            "features": meta.get("features"),
            "ngram_max": meta.get("ngram_max"),
            "min_df": meta.get("min_df"),
            "threshold": meta.get("threshold", 0.5),
            "bytes": size,
            "described": bool(meta),
        })
    out.sort(key=lambda m: m.get("trained_at") or "", reverse=True)
    return out


def delete_model(name):
    import shutil
    d = model_dir(name)
    if d.resolve().parent != MODELS_DIR.resolve() or not d.is_dir():
        raise SystemExit("no such model: %s" % name)
    if not (d / MODEL_FILE).exists():
        raise SystemExit("models/%s is not a bag-of-words model" % name)
    shutil.rmtree(d)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(args):
    import bert_baseline as bb
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    name = checked_name(args.name)
    dest = model_dir(name)
    if dest.exists() and not args.overwrite:
        raise SystemExit("models/%s already exists - pick another name, or "
                         "delete it first" % name)

    print("Loading dataset")
    _, texts, labels = bb.load_dataset(args.csv, args.text_col, args.label_col,
                                       args.limit)
    if len(set(labels)) < 2:
        raise SystemExit("the dataset has only one class in it - nothing to learn")
    if args.strip_tags:
        texts = [strip_tags(t) for t in texts]
        print("  tone tags stripped before fitting")
    print()

    idx = list(range(len(texts)))
    if args.holdout > 0 and len(texts) >= 10:
        tr_idx, te_idx = train_test_split(idx, test_size=args.holdout,
                                          random_state=args.seed,
                                          stratify=labels)
    else:
        tr_idx, te_idx = idx, []
    tr_texts = [texts[i] for i in tr_idx]
    tr_labels = [labels[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    te_labels = [labels[i] for i in te_idx]

    print("==> train")
    print("  %d calls to train on, %d held out" % (len(tr_texts), len(te_texts)))

    def build(min_df):
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, args.ngram_max), min_df=min_df),
            LogisticRegression(max_iter=2000, random_state=args.seed))

    t0 = time.time()
    min_df = args.min_df
    try:
        pipe = build(min_df).fit(tr_texts, tr_labels)
    except ValueError:
        # min_df=2 can empty the vocabulary on a very small training set -
        # the same fallback combined_evaluate.py makes
        min_df = 1
        print("  (min_df=%d left no vocabulary, falling back to 1)" % args.min_df)
        pipe = build(1).fit(tr_texts, tr_labels)
    elapsed = time.time() - t0

    vec = pipe.named_steps["tfidfvectorizer"]
    n_features = len(vec.vocabulary_)

    holdout = None
    if te_texts:
        preds = [int(p) for p in pipe.predict(te_texts)]
        m = bb.metrics(te_labels, preds)
        holdout = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items()}
        print()
        print(bb.fmt("holdout", m))

    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, dest / MODEL_FILE)
    meta = {
        "name": name,
        "kind": "bow",
        "dataset": args.csv,
        "rows": len(texts),
        "scam": int(sum(labels)),
        "legit": int(len(labels) - sum(labels)),
        "trained_rows": len(tr_texts),
        "holdout_rows": len(te_texts),
        "holdout": holdout,
        "features": n_features,
        "ngram_max": args.ngram_max,
        "min_df": min_df,
        "threshold": args.threshold,
        "seed": args.seed,
        "stripped_tags": bool(args.strip_tags),
        "elapsed_s": round(elapsed, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print("  ok trained  models/%s  (%.1fs, %d features)"
          % (name, elapsed, n_features))
    print("  ask it about a call:  python scripts/bow_classify.py classify "
          "--name %s --text \"...\"" % name)
    return meta


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

class Classifier:
    """One loaded bag-of-words model, answering scam or not.

    Held open by serve() so the web UI pays the load once, though unlike a
    BERT checkpoint this is a few hundred kilobytes and loads instantly.
    """

    def __init__(self, name, strip=False, top=12):
        import joblib

        self.name = checked_name(name)
        d = model_dir(self.name)
        if not (d / MODEL_FILE).exists():
            raise SystemExit("no bag-of-words model at models/%s - train one "
                             "first" % self.name)
        self.pipe = joblib.load(d / MODEL_FILE)
        self.vec = self.pipe.named_steps["tfidfvectorizer"]
        self.clf = self.pipe.named_steps["logisticregression"]
        self.meta = read_meta(d) or {"name": self.name}
        self.strip = strip
        self.top = int(top)
        self.threshold = float(self.meta.get("threshold", 0.5) or 0.5)

    def classify(self, transcript, threshold=None, strip=None, top=None):
        """Scam or not, plus the words that decided it.

        A linear model over TF-IDF decomposes exactly: the decision function
        is the intercept plus, for every feature present, its tf-idf weight
        times its coefficient. Those contributions are returned, largest
        first, and they sum to the score - so the explanation is the
        arithmetic, not a story told about it.
        """
        import numpy as np

        t0 = time.time()
        raw = (transcript or "").strip()
        if not raw:
            raise ValueError("there is no transcript to classify")
        strip = self.strip if strip is None else bool(strip)
        text = strip_tags(raw) if strip else raw
        if not text:
            raise ValueError("there is nothing left of that transcript once "
                             "the tone tags are taken out")
        k = int(top or self.top)
        cut = float(self.threshold if threshold is None else threshold)

        X = self.vec.transform([text])
        prob = float(self.clf.predict_proba(X)[0, 1])
        intercept = float(self.clf.intercept_[0])
        coefs = self.clf.coef_[0]
        names = self.vec.get_feature_names_out()

        row = X.tocoo()
        contributions = [(names[j], float(v) * float(coefs[j]), float(v))
                         for j, v in zip(row.col, row.data)]
        contributions.sort(key=lambda c: c[1])
        toward_legit = [c for c in contributions if c[1] < 0][:k]
        toward_scam = [c for c in contributions if c[1] > 0][-k:][::-1]

        words = text.split()
        # how much of the call the model could see at all: a call made of
        # words the training set never had is decided by the intercept
        known = sum(1 for w in set(w.lower().strip(".,!?;:'\"") for w in words)
                    if w in self.vec.vocabulary_)
        distinct = len(set(w.lower().strip(".,!?;:'\"") for w in words))

        return {
            "ok": True,
            "model": {"name": self.name, "dataset": self.meta.get("dataset"),
                      "trained_at": self.meta.get("trained_at"),
                      "features": self.meta.get("features"),
                      "holdout": self.meta.get("holdout")},
            "prob_scam": round(prob, 4),
            "verdict": "scam" if prob >= cut else "legitimate",
            "threshold": round(cut, 4),
            "score": round(float(self.clf.decision_function(X)[0]), 4),
            "intercept": round(intercept, 4),
            "matched": int(row.nnz),
            "vocab_known": known,
            "vocab_distinct": distinct,
            "toward_scam": [{"term": t, "contribution": round(c, 4),
                             "tfidf": round(v, 4)} for t, c, v in toward_scam],
            "toward_legit": [{"term": t, "contribution": round(c, 4),
                              "tfidf": round(v, 4)} for t, c, v in toward_legit],
            "stripped_tags": strip,
            "words": len(words),
            "elapsed_ms": int((time.time() - t0) * 1000),
        }


def classifier_from(args):
    return Classifier(args.name, strip=args.strip_tags, top=args.top)


def run_classify(args):
    text = args.text
    if not text and args.csv:
        text = transcript_from_csv(args.csv, args.idx, args.id)
    if not text:
        raise SystemExit("pass --text, or --csv with --idx/--id")
    out = classifier_from(args).classify(text, args.threshold)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_result(out)
    return out


def print_result(r):
    bar = "=" * 74
    print(bar)
    print("%s   %.1f%% scam   (models/%s, bag of words)"
          % (r["verdict"].upper(), 100 * r["prob_scam"], r["model"]["name"]))
    print(bar)
    print("  score %+.3f = intercept %+.3f + the terms below, which sum to "
          "%+.3f" % (r["score"], r["intercept"], r["score"] - r["intercept"]))
    print("  %d of the call's %d distinct words are in the model's vocabulary; "
          "%d features matched" % (r["vocab_known"], r["vocab_distinct"],
                                   r["matched"]))
    if r["stripped_tags"]:
        print("  tone tags were stripped first")
    print()
    print("  towards SCAM")
    for c in r["toward_scam"]:
        print("    %-34s %+.4f" % (c["term"][:34], c["contribution"]))
    print()
    print("  towards LEGITIMATE")
    for c in r["toward_legit"]:
        print("    %-34s %+.4f" % (c["term"][:34], c["contribution"]))
    print()
    print("  %d words, %d ms" % (r["words"], r["elapsed_ms"]))
    print(bar)


# ---------------------------------------------------------------------------
# evaluate - the whole dataset, not one call
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Score every call in a dataset with one fitted model.

    Note what this is NOT: the `bow` row on the Benchmark page is five-fold
    cross-validation, where every call is predicted by a model that never saw
    it. This scores calls with one already-fitted model, so any call that was
    in that model's training split is being marked by a model that has read
    it. The run says which dataset the model was fitted on, and says it
    loudly when that is the dataset being scored - a 100% there means the
    model remembers, not that it generalises.
    """
    import eval_common as EC

    clf = classifier_from(args)
    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    print()
    print("==> score  models/%s over %d calls" % (clf.name, len(rows)))
    trained_on = clf.meta.get("dataset")
    if trained_on and trained_on == args.csv:
        print("  WARNING this model was fitted on this same dataset. Unless "
              "you held rows\n          back, it has read these calls before "
              "and the score below is a\n          memory test.")

    prog = EC.Progress(len(rows), every=max(1, len(rows) // 40))
    preds, probs = [], []
    t0 = time.time()
    for r in rows:
        try:
            out = clf.classify(r["text"], args.threshold, strip=args.strip_tags)
            pred = 1 if out["verdict"] == "scam" else 0
            probs.append("%.4f" % out["prob_scam"])
        except ValueError:
            pred = None
            probs.append("")
        preds.append(pred)
        prog.tick(r, pred)
    elapsed = time.time() - t0

    m = EC.metrics([r["label"] for r in rows], preds)
    base = EC.baselines([r["label"] for r in rows])
    EC.report(m, base, elapsed, len(rows))
    if args.out:
        EC.write_results(args.out, clf.name, args.csv, rows, preds, m, base,
                         elapsed, {"prob_scam": probs},
                         {"kind": "bow", "trained_on": trained_on,
                          "threshold": args.threshold or clf.threshold,
                          "same_dataset": bool(trained_on == args.csv)})
    return m


# ---------------------------------------------------------------------------
# serve - one JSON request per line, one JSON reply per line
# ---------------------------------------------------------------------------

def run_serve(args):
    try:
        clf = classifier_from(args)
    except BaseException as e:      # SystemExit included - report, do not exit
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        return
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "model": clf.name,
                                 "device": "cpu"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                return
            thr = req.get("threshold")
            out = clf.classify(req.get("transcript", ""),
                               None if thr in (None, "") else float(thr),
                               strip=req.get("strip_tags"),
                               top=req.get("top"))
        except Exception as e:
            out = {"ok": False,
                   "error": str(e) if isinstance(e, ValueError)
                            else "%s: %s" % (type(e).__name__, e)}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def run_models(args):
    models = list_models()
    if args.json:
        print(json.dumps(models, indent=2))
        return models
    if not models:
        print("  no bag-of-words models in models/ yet")
        return models
    for m in models:
        acc = m["holdout"] and m["holdout"].get("acc")
        print("  %-24s %-34s %7s features  %s"
              % (m["name"], m["dataset"] or "", m["features"] or "?",
                 ("holdout acc %.1f%%" % (100 * acc)) if acc else ""))
    return models


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="train a bag-of-words classifier on scam calls, then ask "
                    "it whether a call is one",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train", help="fit and keep a model")
    tr.add_argument("--csv", required=True)
    tr.add_argument("--name", required=True, help="saved as models/<name>/")
    tr.add_argument("--ngram-max", type=int, default=2,
                    help="1 for words only, 2 to add bigrams (the benchmark "
                         "uses 2)")
    tr.add_argument("--min-df", type=int, default=2,
                    help="ignore terms in fewer than this many calls")
    tr.add_argument("--threshold", type=float, default=0.5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--limit", type=int, default=None)
    tr.add_argument("--holdout", type=float, default=0.2,
                    help="fraction kept back to score the model (0 = none)")
    tr.add_argument("--text-col", default=None)
    tr.add_argument("--label-col", default=None)
    tr.add_argument("--strip-tags", action="store_true",
                    help="drop the [curious]/[long pause] annotations before "
                         "fitting")
    tr.add_argument("--overwrite", action="store_true")
    tr.set_defaults(func=run_train)

    def scoring(sp):
        sp.add_argument("--name", required=True)
        sp.add_argument("--top", type=int, default=12,
                        help="how many terms to show on each side")
        sp.add_argument("--strip-tags", action="store_true",
                        help="drop the tone tags before scoring")

    cl = sub.add_parser("classify", help="scam or not, for one transcript")
    scoring(cl)
    cl.add_argument("--text", default=None)
    cl.add_argument("--csv", default=None)
    cl.add_argument("--idx", type=int, default=None)
    cl.add_argument("--id", default=None)
    cl.add_argument("--threshold", type=float, default=None,
                    help="defaults to the model's own")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=run_classify)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    scoring(ev)
    ev.add_argument("--csv", required=True)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--threshold", type=float, default=None)
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    sv = sub.add_parser("serve", help="stay loaded, one JSON request per line")
    scoring(sv)
    sv.set_defaults(func=run_serve)

    ls = sub.add_parser("models", help="what bag-of-words models are in models/")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=run_models)

    rm = sub.add_parser("delete", help="remove one model")
    rm.add_argument("--name", required=True)
    rm.set_defaults(func=lambda a: print(json.dumps(delete_model(a.name))))

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
