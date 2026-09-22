#!/usr/bin/env python3
"""
length_classify.py - the shortest baseline there is: count the words, compare
the count to one number, call it a scam or not.

This is `trivial_length` out of combined_evaluate.py - "Fraud if the call is
longer than 45 words" - given the same shape as bert_classify.py and
bow_classify.py so it can sit beside them on the same page and be asked the
same question about the same call.

It exists to be the floor. A model is only interesting to the extent it beats
what you get from the length of the transcript alone, and on several of the
datasets here that floor is embarrassingly high:

    scambait_bank_422.csv    scam calls are scam-baiting videos, legitimate
                             calls are HarperValleyBank task recordings. The
                             two have different typical lengths, so a single
                             threshold separates much of the set without
                             looking at a single word.

If a fine-tuned BERT beats this by four points on a dataset, the dataset is
measuring transcript length with extra steps.

A note on the word "train". There is nothing learned here - one number is
fitted, by trying every threshold the training calls suggest and keeping the
one that scores best. That is a sweep, not learning, and it will overfit a
single number to a single dataset happily. The page says so, `fit` prints the
runner-up thresholds so you can see how flat or sharp the choice was, and
--threshold pins it to the benchmark's own 45 if you would rather not fit
anything at all.

Usage:
    python scripts/length_classify.py fit --csv datasets/zhi_english_646.csv \
        --name zhi-length
    python scripts/length_classify.py fit --csv datasets/zhi_english_646.csv \
        --name zhi-45 --threshold 45          # no sweep, the benchmark's rule
    python scripts/length_classify.py classify --name zhi-length --text "..."
    python scripts/length_classify.py models
    python scripts/length_classify.py serve --name zhi-length

scripts/web_ui.py drives all of these from the Length only page.
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

# Shared with the other two pages rather than copied - a name is a name, a
# row is a row, and the tone-tag map is a list of real strings that would
# drift if it lived in three files. All standard library.
from bert_classify import NAME_RE, checked_name, model_dir, strip_tags, \
    transcript_from_csv                                    # noqa: E402

# What marks a directory in models/ as one of these. bert_classify looks for
# config.json and bow_classify for bow.joblib, so the three kinds sit side by
# side in models/ without seeing each other.
MODEL_FILE = "length.json"
META_FILE = "meta.json"

# combined_evaluate.trivial_length's own threshold, used when --threshold is
# given no value and as the fallback for a model fitted on one class.
BENCHMARK_THRESHOLD = 45

# Which way round the rule goes. Fitting tries both, because "longer is the
# scam" is an assumption about the corpus and not a fact about scams: a
# dataset whose legitimate side is long support calls flips it.
LONGER = "longer"
SHORTER = "shorter"


def word_count(text):
    """The same count trivial_length uses: whitespace-split tokens."""
    return len((text or "").split())


def read_meta(d):
    try:
        return json.loads((d / META_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_rule(d):
    return json.loads((d / MODEL_FILE).read_text(encoding="utf-8"))


def list_models():
    """Every length-only model in models/, newest first."""
    out = []
    if not MODELS_DIR.is_dir():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / MODEL_FILE).exists():
            continue
        meta = read_meta(d)
        out.append({
            "name": d.name,
            "dataset": meta.get("dataset"),
            "rows": meta.get("rows"),
            "holdout": meta.get("holdout"),
            "trained_at": meta.get("trained_at"),
            "elapsed_s": meta.get("elapsed_s"),
            "threshold": meta.get("threshold"),
            "direction": meta.get("direction"),
            "swept": meta.get("swept"),
            "fit_f1": meta.get("fit_f1"),
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
        raise SystemExit("models/%s is not a length-only model" % name)
    shutil.rmtree(d)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def score(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"acc": acc, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ladder(counts, labels):
    """[(word count, scam calls at or below it, legitimate calls at or below)].

    One sorted walk of the fitting set, which is the whole of what a
    length-only model needs to know: from it, any threshold's confusion
    matrix and any threshold's class balance are lookups rather than another
    pass. A dataset of 7,000 calls has about 2,000 distinct lengths, so this
    is small enough to keep in the model file - which is what lets the page
    move the threshold and show honest numbers at the new one instead of the
    fitted one's.
    """
    pairs = sorted(zip(counts, labels))
    if not pairs:
        return []
    # a threshold below the shortest call: "everything is longer", the
    # always-scam rule under LONGER, which is worth having as the thing a
    # real threshold has to beat
    rows = [(pairs[0][0] - 1, 0, 0)]
    pos_below = neg_below = 0
    i = 0
    while i < len(pairs):
        w = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == w:
            if pairs[i][1]:
                pos_below += 1
            else:
                neg_below += 1
            i += 1
        rows.append((w, pos_below, neg_below))
    return rows


def below_at(rows, threshold):
    """(scam, legitimate) calls at or below `threshold`, from a ladder."""
    import bisect
    if not rows:
        return 0, 0
    i = bisect.bisect_right([r[0] for r in rows], int(threshold)) - 1
    return (0, 0) if i < 0 else (rows[i][1], rows[i][2])


def sweep(counts, labels, metric="f1", rows=None):
    """Every threshold the data suggests, both directions, scored.

    Returns (best, curve) where a row is
    (threshold, direction, metrics dict).
    """
    P = sum(labels)
    N = len(labels) - P
    curve = []
    candidates = rows if rows is not None else ladder(counts, labels)

    for t, pb, nb in candidates:
        # LONGER: scam if w > t. Everything at or below t is called legit.
        curve.append((t, LONGER, score(P - pb, N - nb, pb, nb)))
        # SHORTER: scam if w <= t.
        curve.append((t, SHORTER, score(pb, nb, P - pb, N - nb)))

    if not curve:
        return None, []
    # ties broken towards accuracy, then towards the smaller threshold, so a
    # flat curve gives a repeatable answer instead of whichever row sorted
    # first
    best = max(curve, key=lambda row: (row[2][metric], row[2]["acc"], -row[0]))
    return best, curve


def percentiles(values, steps=(0, 10, 25, 50, 75, 90, 100)):
    if not values:
        return {}
    s = sorted(values)
    out = {}
    for p in steps:
        k = (len(s) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        out["p%d" % p] = round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)
    return out


def side_rates(rows, total_scam, total_legit, threshold, direction):
    """The fitting-set scam rate on each side of a line, at any threshold.

    This is the only honest number a threshold can offer in place of a
    probability: not "the model thinks this is 82% a scam", but "82% of the
    fitting calls on this side of the line were scams". Computed from the
    ladder rather than stored for one threshold, so moving the line on the
    page moves these with it instead of quietly keeping the fitted line's
    numbers under a different cut.
    """
    pb, nb = below_at(rows, threshold)
    below_n = pb + nb
    above_n = (total_scam - pb) + (total_legit - nb)
    return {
        "above": round((total_scam - pb) / above_n, 4) if above_n else None,
        "above_n": above_n,
        "below": round(pb / below_n, 4) if below_n else None,
        "below_n": below_n,
        "scam_side": "above" if direction == LONGER else "below",
    }


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def run_train(args):
    import bert_baseline as bb
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
        raise SystemExit("the dataset has only one class in it - there is no "
                         "line to draw")
    if args.strip_tags:
        texts = [strip_tags(t) for t in texts]
        print("  tone tags stripped before counting")
    print()

    idx = list(range(len(texts)))
    if args.holdout > 0 and len(texts) >= 10:
        tr_idx, te_idx = train_test_split(idx, test_size=args.holdout,
                                          random_state=args.seed,
                                          stratify=labels)
    else:
        tr_idx, te_idx = idx, []
    counts = [word_count(t) for t in texts]
    tr_counts = [counts[i] for i in tr_idx]
    tr_labels = [labels[i] for i in tr_idx]
    te_counts = [counts[i] for i in te_idx]
    te_labels = [labels[i] for i in te_idx]

    print("==> fit")
    print("  %d calls to fit on, %d held out" % (len(tr_counts), len(te_counts)))
    scam_len = [c for c, y in zip(tr_counts, tr_labels) if y]
    legit_len = [c for c, y in zip(tr_counts, tr_labels) if not y]
    ps, pl = percentiles(scam_len), percentiles(legit_len)
    print("  scam calls        median %6.0f words   (p10 %.0f - p90 %.0f)"
          % (ps["p50"], ps["p10"], ps["p90"]))
    print("  legitimate calls  median %6.0f words   (p10 %.0f - p90 %.0f)"
          % (pl["p50"], pl["p10"], pl["p90"]))
    print()

    t0 = time.time()
    curve = []
    rows = ladder(tr_counts, tr_labels)
    if args.threshold is not None:
        threshold = int(args.threshold)
        direction = args.direction
        fit = score(*_confusion(tr_counts, tr_labels, threshold, direction))
        swept = False
        print("  threshold pinned at %d words, %s is scam (no sweep)"
              % (threshold, direction))
    else:
        best, curve = sweep(tr_counts, tr_labels, args.metric, rows)
        threshold, direction, fit = best
        swept = True
        print("  swept %d thresholds, both directions" % (len(curve) // 2))
        print("  best: %s than %d words is scam   (%s %.3f on the fitting set)"
              % (direction, threshold, args.metric, fit[args.metric]))
        print()
        print("  runner-up thresholds - how sharp that choice was:")
        ranked = sorted(curve, key=lambda r: -r[2][args.metric])
        for t, d, m in ranked[1:6]:
            print("    %-8s than %6d words   %s %.3f   acc %.1f%%"
                  % (d, t, args.metric, m[args.metric], 100 * m["acc"]))
        flat = sum(1 for _t, _d, m in curve
                   if m[args.metric] >= fit[args.metric] - 0.01)
        if flat > max(3, len(curve) // 20):
            print("    %d of %d thresholds come within a point of the best - "
                  "this curve is flat, so the exact number is not meaningful"
                  % (flat, len(curve)))
    elapsed = time.time() - t0

    holdout = None
    if te_counts:
        preds = [predict(w, threshold, direction) for w in te_counts]
        m = bb.metrics(te_labels, preds)
        holdout = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items()}
        print()
        print(bb.fmt("holdout", m))
        always = bb.metrics(te_labels, [1] * len(te_labels))
        never = bb.metrics(te_labels, [0] * len(te_labels))
        floor = max(always["acc"], never["acc"])
        print(bb.fmt("always scam", always))
        print(bb.fmt("never scam", never))
        if m["acc"] <= floor:
            print("  NOTE the threshold does not beat calling every call the "
                  "same thing - on this dataset length carries nothing.")

    n_scam, n_legit = len(scam_len), len(legit_len)
    rates = side_rates(rows, n_scam, n_legit, threshold, direction)

    dest.mkdir(parents=True, exist_ok=True)
    rule = {
        "threshold": threshold,
        "direction": direction,
        "rates": rates,
        # kept so a threshold moved on the page gets its own honest rates
        # rather than the fitted line's under a different cut
        "ladder": [list(r) for r in rows],
        "fit_scam": n_scam,
        "fit_legit": n_legit,
        "dist": {"scam": ps, "legit": pl,
                 "scam_n": n_scam, "legit_n": n_legit},
        # thinned so a 7,000-call sweep does not write a 300 KB file; the
        # page draws this to show how flat the choice was
        "curve": [{"t": t, "d": d, "acc": round(m["acc"], 4),
                   "f1": round(m["f1"], 4)}
                  for t, d, m in _thin(curve, 120)],
    }
    (dest / MODEL_FILE).write_text(json.dumps(rule, indent=2), encoding="utf-8")
    meta = {
        "name": name,
        "kind": "length",
        "dataset": args.csv,
        "rows": len(texts),
        "scam": int(sum(labels)),
        "legit": int(len(labels) - sum(labels)),
        "trained_rows": len(tr_counts),
        "holdout_rows": len(te_counts),
        "holdout": holdout,
        "threshold": threshold,
        "direction": direction,
        "swept": swept,
        "metric": args.metric,
        "fit_f1": round(fit["f1"], 4),
        "fit_acc": round(fit["acc"], 4),
        "seed": args.seed,
        "stripped_tags": bool(args.strip_tags),
        "elapsed_s": round(elapsed, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print("  ok fitted  models/%s  (%s than %d words is scam)"
          % (name, direction, threshold))
    print("  ask it about a call:  python scripts/length_classify.py classify "
          "--name %s --text \"...\"" % name)
    return meta


def _confusion(counts, labels, threshold, direction):
    tp = fp = fn = tn = 0
    for w, y in zip(counts, labels):
        p = predict(w, threshold, direction)
        tp += 1 if (p and y) else 0
        fp += 1 if (p and not y) else 0
        fn += 1 if (not p and y) else 0
        tn += 1 if (not p and not y) else 0
    return tp, fp, fn, tn


def _thin(curve, keep):
    """Every nth threshold of the sweep, keeping both of its directions.

    sweep() emits the two directions of a threshold as adjacent rows, so the
    stride is even: thinning by an odd step would hand the page a curve made
    of half of one rule and half of the other.
    """
    if len(curve) <= keep:
        return curve
    pairs = [curve[i:i + 2] for i in range(0, len(curve), 2)]
    step = len(pairs) / float(max(1, keep // 2))
    out = []
    for i in range(max(1, keep // 2)):
        out.extend(pairs[min(int(i * step), len(pairs) - 1)])
    return out


def predict(words, threshold, direction):
    return int(words > threshold if direction == LONGER else words <= threshold)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

class Classifier:
    """One fitted threshold, answering scam or not.

    Held open by serve() for symmetry with the other two pages rather than
    for speed: this is a JSON file of about two kilobytes and the decision is
    a comparison. It is the one page where the model genuinely does fit in
    the sentence that describes it.
    """

    def __init__(self, name, strip=False):
        self.name = checked_name(name)
        d = model_dir(self.name)
        if not (d / MODEL_FILE).exists():
            raise SystemExit("no length-only model at models/%s - fit one "
                             "first" % self.name)
        self.rule = read_rule(d)
        self.meta = read_meta(d) or {"name": self.name}
        self.strip = strip
        self.threshold = int(self.rule.get("threshold", BENCHMARK_THRESHOLD))
        self.direction = self.rule.get("direction", LONGER)
        self.ladder = [tuple(r) for r in (self.rule.get("ladder") or [])]
        self.fit_scam = int(self.rule.get("fit_scam") or 0)
        self.fit_legit = int(self.rule.get("fit_legit") or 0)

    def classify(self, transcript, threshold=None, strip=None, direction=None):
        """Scam or not, and the whole of the reasoning.

        Nothing is hidden here, because there is nothing to hide: the call is
        a number of words, the model is a number, and the verdict is which
        side of the other one is on. What the reply adds is context for that
        number - where the call sits in the two training distributions, and
        what share of training calls on this side of the line were actually
        scams - so the page can say how much the comparison is worth.
        """
        t0 = time.time()
        raw = (transcript or "").strip()
        if not raw:
            raise ValueError("there is no transcript to classify")
        strip = self.strip if strip is None else bool(strip)
        text = strip_tags(raw) if strip else raw
        if not text:
            raise ValueError("there is nothing left of that transcript once "
                             "the tone tags are taken out")
        cut = int(self.threshold if threshold is None else threshold)
        way = direction if direction in (LONGER, SHORTER) else self.direction

        words = word_count(text)
        verdict = "scam" if predict(words, cut, way) else "legitimate"

        # at the model's own line the stored rates are these; away from it
        # they are not, which is why this is recomputed rather than read
        rates = (side_rates(self.ladder, self.fit_scam, self.fit_legit, cut, way)
                 if self.ladder else (self.rule.get("rates") or {}))
        on_side = "above" if words > cut else "below"
        rate = rates.get(on_side)
        rate_n = rates.get(on_side + "_n") or 0
        # the rate is the training-set scam share on this side of the line.
        # When the verdict is scam and that share is under half, the
        # threshold is making a call its own fitting data does not support,
        # which the page says out loud rather than hiding behind a number.
        disagrees = rate is not None and (
            (verdict == "scam" and rate < 0.5)
            or (verdict == "legitimate" and rate >= 0.5))

        dist = self.rule.get("dist") or {}
        return {
            "ok": True,
            "model": {"name": self.name, "dataset": self.meta.get("dataset"),
                      "trained_at": self.meta.get("trained_at"),
                      "threshold": self.threshold,
                      "direction": self.direction,
                      "swept": self.meta.get("swept"),
                      "holdout": self.meta.get("holdout")},
            "words": words,
            "threshold": cut,
            "direction": way,
            "verdict": verdict,
            "margin": words - cut,
            "prob_scam": rate,
            "side": on_side,
            "side_n": rate_n,
            "rates": rates,
            "moved": cut != self.threshold or way != self.direction,
            "rate_disagrees": disagrees,
            "dist": dist,
            "curve": self.rule.get("curve") or [],
            "stripped_tags": strip,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }


def classifier_from(args):
    return Classifier(args.name, strip=args.strip_tags)


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
    print("%s   %d words   (models/%s, length only)"
          % (r["verdict"].upper(), r["words"], r["model"]["name"]))
    print(bar)
    print("  the rule: %s than %d words is a scam. This call is %d, which is "
          "%+d." % (r["direction"], r["threshold"], r["words"], r["margin"]))
    if r["prob_scam"] is not None:
        print("  %.1f%% of the %d fitting calls on this side of the line were "
              "scams" % (100 * r["prob_scam"], r["side_n"]))
    if r["rate_disagrees"]:
        print("  NOTE most calls on this side of the line were NOT what the "
              "rule just called this one. The threshold is past the point "
              "where it carries anything.")
    d = r.get("dist") or {}
    if d.get("scam") and d.get("legit"):
        print()
        print("  in the fitting set:")
        print("    scam calls        median %6.0f   p10 %.0f  p90 %.0f"
              % (d["scam"]["p50"], d["scam"]["p10"], d["scam"]["p90"]))
        print("    legitimate calls  median %6.0f   p10 %.0f  p90 %.0f"
              % (d["legit"]["p50"], d["legit"]["p10"], d["legit"]["p90"]))
    if r["stripped_tags"]:
        print()
        print("  tone tags were stripped before counting")
    print()
    print("  nothing in the call was read. Only how much of it there was.")
    print(bar)


# ---------------------------------------------------------------------------
# evaluate - the whole dataset, not one call
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Score every call in a dataset against one fitted threshold.

    Cheap enough that this is the honest way to read this baseline: a
    threshold fitted on one corpus and run over another is the clearest test
    there is of whether call length carries anything general, and the answer
    is usually no. The run reports the direction as well as the number,
    because a threshold pointing the wrong way scores worse than its own
    mirror image and that is worth seeing rather than inferring.
    """
    import eval_common as EC

    clf = classifier_from(args)
    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    print()
    cut = int(args.threshold if args.threshold is not None else clf.threshold)
    way = args.direction or clf.direction
    print("==> score  %s than %d words is scam, over %d calls"
          % (way, cut, len(rows)))
    trained_on = clf.meta.get("dataset")
    if trained_on and trained_on == args.csv:
        print("  NOTE this threshold was fitted on this same dataset, so this "
              "is how well one\n       number fits the calls it was chosen "
              "from, not how well it travels.")

    prog = EC.Progress(len(rows), every=max(1, len(rows) // 40))
    preds, words = [], []
    t0 = time.time()
    for r in rows:
        text = strip_tags(r["text"]) if args.strip_tags else r["text"]
        w = word_count(text)
        pred = predict(w, cut, way)
        words.append(w)
        preds.append(pred)
        prog.tick(r, pred)
    elapsed = time.time() - t0

    truths = [r["label"] for r in rows]
    m = EC.metrics(truths, preds)
    base = EC.baselines(truths)
    # The same rule pointing the other way, since a threshold that is worse
    # than its own mirror image is a direction error rather than a bad number.
    other = LONGER if way == SHORTER else SHORTER
    mirror = EC.metrics(truths, [predict(w, cut, other) for w in words])
    EC.report(m, base, elapsed, len(rows), [
        "the same threshold pointing the other way (%s than %d) would get "
        "%.1f%%" % (other, cut, 100 * mirror["acc"])])
    if mirror["acc"] > m["acc"]:
        print("  NOTE the rule is pointing the wrong way for this dataset - "
              "on these calls the\n       %s ones are the scams." % other)
    if args.out:
        EC.write_results(args.out, clf.name, args.csv, rows, preds, m, base,
                         elapsed, {"words_counted": words},
                         {"kind": "length", "trained_on": trained_on,
                          "threshold": cut, "direction": way,
                          "mirror": mirror,
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
                               None if thr in (None, "") else int(float(thr)),
                               strip=req.get("strip_tags"),
                               direction=req.get("direction"))
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
        print("  no length-only models in models/ yet")
        return models
    for m in models:
        acc = m["holdout"] and m["holdout"].get("acc")
        print("  %-24s %-34s %-8s than %5s words  %s"
              % (m["name"], m["dataset"] or "", m["direction"] or "?",
                 m["threshold"], ("holdout acc %.1f%%" % (100 * acc))
                 if acc else ""))
    return models


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="the length-only baseline: count the words, compare to "
                    "one number, call it",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    # "fit" rather than "train", and "train" kept as an alias: nothing is
    # learned, and the web UI calls the same endpoint for all three pages.
    for word in ("fit", "train"):
        tr = sub.add_parser(word, help="pick a threshold and keep it")
        tr.add_argument("--csv", required=True)
        tr.add_argument("--name", required=True, help="saved as models/<name>/")
        tr.add_argument("--threshold", type=int, default=None,
                        help="pin the threshold instead of sweeping for one "
                             "(the benchmark's own rule is 45)")
        tr.add_argument("--direction", choices=[LONGER, SHORTER], default=LONGER,
                        help="which side is the scam, when --threshold pins "
                             "it; a sweep decides this for itself")
        tr.add_argument("--metric", choices=["f1", "acc"], default="f1",
                        help="what the sweep maximises")
        tr.add_argument("--seed", type=int, default=42)
        tr.add_argument("--limit", type=int, default=None)
        tr.add_argument("--holdout", type=float, default=0.2,
                        help="fraction kept back to score it (0 = none)")
        tr.add_argument("--text-col", default=None)
        tr.add_argument("--label-col", default=None)
        tr.add_argument("--strip-tags", action="store_true",
                        help="drop the [curious]/[long pause] annotations "
                             "before counting")
        tr.add_argument("--overwrite", action="store_true")
        tr.set_defaults(func=run_train)

    def scoring(sp):
        sp.add_argument("--name", required=True)
        sp.add_argument("--strip-tags", action="store_true",
                        help="drop the tone tags before counting")

    cl = sub.add_parser("classify", help="scam or not, for one transcript")
    scoring(cl)
    cl.add_argument("--text", default=None)
    cl.add_argument("--csv", default=None)
    cl.add_argument("--idx", type=int, default=None)
    cl.add_argument("--id", default=None)
    cl.add_argument("--threshold", type=int, default=None,
                    help="defaults to the model's own")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=run_classify)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    scoring(ev)
    ev.add_argument("--csv", required=True)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--threshold", type=int, default=None)
    ev.add_argument("--direction", choices=[LONGER, SHORTER], default=None)
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    sv = sub.add_parser("serve", help="stay loaded, one JSON request per line")
    scoring(sv)
    sv.set_defaults(func=run_serve)

    ls = sub.add_parser("models", help="what length-only models are in models/")
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
