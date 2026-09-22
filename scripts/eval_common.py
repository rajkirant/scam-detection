#!/usr/bin/env python3
"""
eval_common.py - the shared parts of scoring a whole dataset.

Four pages can now put a whole dataset to a model rather than one call at a
time, and all four have to agree on what a score means or the numbers cannot
be read side by side. That agreement lives here: one dataset loader, one
confusion matrix, one set of trivial baselines to measure against, and one
results file.

Standard library only. web_ui.py imports it, and this server starts without
the venv.

The one opinion in here worth arguing about is `baselines`. Every score is
printed next to what always-scam and never-scam get on the same calls,
because an accuracy on its own is not a result: on a set that is 77%
legitimate, a model at 75% is worse than answering "Normal" to everything, and
that fact should not need working out. The numbers are there whether or not
anyone wants them.
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

TEXT_COLS = ("transcript", "text", "call", "conversation", "dialogue",
             "content", "body")
LABEL_COLS = ("label", "is_scam", "scam", "target", "class", "y",
              "ground_truth")
ID_COLS = ("id", "call_id", "conv_id")

SCAM_WORDS = {"1", "scam", "fraud", "fraudulent", "true", "yes", "spam",
              "phishing", "malicious"}


def to_binary(value):
    v = str(value).strip().lower()
    if v in SCAM_WORDS:
        return 1
    try:
        return 1 if float(v) >= 0.5 else 0
    except ValueError:
        return 0


def load_rows(path, text_col=None, label_col=None, limit=None, quiet=False):
    """[{id, row, text, label, words}] out of a dataset CSV.

    stdlib csv rather than pandas, and the field size limit is lifted first:
    the longest transcript in datasets/ is 225,978 characters and the default
    limit is 131,072, so without this the hard subsets raise rather than load.
    """
    csv.field_size_limit(sys.maxsize)
    p = Path(path)
    if not p.exists():
        raise SystemExit("dataset not found: %s" % path)
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit("that dataset is empty")

    cols = {c.lower(): c for c in raw[0]}
    tcol = text_col or next((cols[c] for c in TEXT_COLS if c in cols), None)
    lcol = label_col or next((cols[c] for c in LABEL_COLS if c in cols), None)
    if tcol is None:
        raise SystemExit("no transcript column in that dataset")
    if lcol is None:
        raise SystemExit("no label column in that dataset")
    icol = next((cols[c] for c in ID_COLS if c in cols), None)

    rows = []
    for i, r in enumerate(raw):
        text = (r.get(tcol) or "").strip()
        if not text:
            continue
        rows.append({"id": (r.get(icol) if icol else None) or i,
                     "row": i,
                     "text": text,
                     "label": to_binary(r.get(lcol)),
                     "words": len(text.split())})
    if limit:
        # a class-balanced head, so a smoke test is not all one class
        pos = [r for r in rows if r["label"]][: limit // 2]
        neg = [r for r in rows if not r["label"]][: limit - limit // 2]
        rows = sorted(pos + neg, key=lambda r: r["row"])
    if not quiet:
        print("  loaded %d calls from %s" % (len(rows), path))
        print("  text column: '%s'   label column: '%s'" % (tcol, lcol))
        print("  class balance: %d scam / %d legitimate"
              % (sum(r["label"] for r in rows),
                 len(rows) - sum(r["label"] for r in rows)))
    return rows


def metrics(y_true, y_pred):
    """The confusion matrix and what comes off it.

    A prediction of None - the model gave an answer that could not be read -
    counts as neither class and is reported separately rather than being
    quietly folded into Normal, which is what turns an unreadable answer into
    a false negative. Scores are over what was actually scored; `scored` and
    `unreadable` say how many that was.
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    n = len(pairs)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # Balanced accuracy, because plain accuracy on a lopsided set flatters a
    # model that has learnt the class balance and nothing else.
    bal = (rec + spec) / 2
    return {"acc": acc, "precision": prec, "recall": rec, "specificity": spec,
            "f1": f1, "balanced_acc": bal,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "unreadable": len(y_pred) - n, "scored": n, "total": len(y_pred)}


def fmt(name, m):
    return ("  %-22s acc %5.1f%%  P %.3f  R %.3f  F1 %.3f   (TP%d FP%d FN%d "
            "TN%d)%s" % (name, m["acc"] * 100, m["precision"], m["recall"],
                         m["f1"], m["tp"], m["fp"], m["fn"], m["tn"],
                         "  %d unreadable" % m["unreadable"]
                         if m["unreadable"] else ""))


def baselines(y_true):
    """What answering the same thing every time gets on these calls.

    The floor a result has to clear. Anything at or under it was measured on
    the class balance, not on the calls.
    """
    return {"always_scam": metrics(y_true, [1] * len(y_true)),
            "never_scam": metrics(y_true, [0] * len(y_true))}


def verdict_of(pred):
    return "unreadable" if pred is None else ("scam" if pred else "legitimate")


def say_which_experiment(trained_on, scoring, verb="fitted"):
    """Name the experiment this run is, before it prints a number for it.

    Three different things get run from the same button and they are not
    comparable, which is easy to miss and expensive to miss:

      same dataset        the model has read these calls. Unless rows were
                          held back this is a memory test.
      a different one     a transfer test: how far what it learned on one
                          corpus carries to another. This is usually the
                          number worth having, and usually far lower.
      unknown             an old model with no dataset recorded.

    None of the three is what the Benchmark page reports for a dataset. That
    is k-fold cross-validation WITHIN one dataset - trained on part of it,
    scored on the rest, every call predicted by a model that never saw it -
    so a 100% there and a 50% here are not a contradiction and not the same
    experiment. Saying so costs two lines and saves reading one as the other.
    """
    if not trained_on:
        print("  NOTE this model does not record what it was %s on, so "
              "whether it has seen\n       these calls cannot be told from "
              "here." % verb)
        return "unknown"
    if trained_on == scoring:
        print("  WARNING this model was %s on this same dataset. Unless rows "
              "were held back\n          it has read these calls before, and "
              "the score below is a memory\n          test rather than a "
              "measurement." % verb)
        return "same"
    print("  This is a TRANSFER test.")
    print("    %s on   %s" % (verb.rjust(7), trained_on))
    print("    scored on  %s" % scoring)
    print("  It asks how far what the model learned on the first carries to "
          "the second,\n  which is a harder question than the Benchmark "
          "page's figure for either.\n  That one is k-fold cross-validation "
          "within a single dataset, so it trains\n  and scores on the same "
          "kind of text; this does not. A low number here next\n  to a high "
          "one there is a finding, not a disagreement.")
    return "transfer"


class Progress:
    """One line per call while a slow run is going, or a tick every so often
    while a fast one is.

    A bag-of-words run over 7,000 calls finishes before a per-call line has
    been read, and an LLM run over 20 takes ten minutes; the same printer has
    to serve both, so `every` is set by the caller that knows which it is.
    """

    def __init__(self, total, every=1, out=None):
        self.total, self.every = total, max(1, int(every))
        self.out = out or sys.stdout
        self.t0 = time.time()
        self.n = self.right = 0

    def tick(self, row, pred, note=""):
        self.n += 1
        ok = pred is not None and pred == row["label"]
        self.right += bool(ok)
        if self.n % self.every and self.n != self.total:
            return
        done = time.time() - self.t0
        rate = self.n / done if done else 0
        left = (self.total - self.n) / rate if rate else 0
        self.out.write(
            "    %5d/%d  %-2s  said %-11s really %-11s  %5.1f%% right so far"
            "   %s%s\n"
            % (self.n, self.total,
               "ok" if ok else ("?" if pred is None else "X"),
               verdict_of(pred),
               "scam" if row["label"] else "legitimate",
               100.0 * self.right / self.n,
               ("%.1f/s" % rate) if rate >= 1 else ("%.1fs/call" % (1 / rate)
                                                    if rate else "-"),
               ("  ~%s left" % human(left)) if self.n < self.total else ""))
        self.out.flush()


def human(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def report(m, base, elapsed, total, extra_lines=()):
    """The block every evaluate run ends with."""
    print()
    print("  " + "-" * 70)
    print(fmt("this model", m))
    print(fmt("always scam", base["always_scam"]))
    print(fmt("never scam", base["never_scam"]))
    floor = max(base["always_scam"]["acc"], base["never_scam"]["acc"])
    print("  " + "-" * 70)
    print("  balanced accuracy %.1f%%   specificity %.3f"
          % (100 * m["balanced_acc"], m["specificity"]))
    if m["acc"] <= floor:
        print("  NOTE this does not beat answering the same thing every time "
              "(%.1f%%). On this\n       dataset the model is not adding "
              "anything to the class balance." % (100 * floor))
    if m["unreadable"]:
        print("  NOTE %d call(s) got no readable answer and are left out of "
              "the scores above,\n       rather than counted as legitimate."
              % m["unreadable"])
    for line in extra_lines:
        print("  " + line)
    print("  %d calls in %s (%.1f/s)"
          % (total, human(elapsed), total / elapsed if elapsed else 0))
    print("  " + "-" * 70)


def short_path(path):
    """Project-relative if it is inside the project, absolute if it is not.

    RESULTS_DIR is inside the project in every real run; it is not under test,
    and a path that cannot be made relative should read oddly rather than
    raise on the last line of a run that has already done all the work.
    """
    try:
        return str(Path(path).relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def write_results(out_json, name, dataset, rows, preds, m, base, elapsed,
                  extra_cols=None, meta=None):
    """A metrics JSON beside the run log, and a per-call CSV in results/.

    The JSON is what the page reads to draw its numbers; the CSV is what you
    open when the numbers are surprising and you want to see which calls went
    wrong. Both are written even when the run is cancelled part way, because
    a partial score on 300 of 7,000 calls is still worth the calls it cost.
    """
    extra_cols = extra_cols or {}
    json_path = Path(out_json)
    if json_path.suffix != ".json":
        json_path = Path(str(json_path) + ".json")
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # results/eval_<run id>.csv, whatever shape the json path was given in -
    # .with_suffix would eat a dotted run id, and the run's own meta file is
    # already <run id>.json, which this must not land on top of
    stem = json_path.name[: -len(".json")]
    if stem.endswith(".metrics"):
        stem = stem[: -len(".metrics")]
    csv_path = Path(RESULTS_DIR) / ("eval_%s.csv" % stem)
    header = ["idx", "id", "true", "predicted", "correct", "words"]
    header += list(extra_cols.keys())
    header.append("text")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (r, p) in enumerate(zip(rows, preds)):
            row = [i, r["id"], "scam" if r["label"] else "legitimate",
                   verdict_of(p),
                   "" if p is None else int(p == r["label"]), r["words"]]
            for k in extra_cols:
                vals = extra_cols[k]
                row.append(vals[i] if i < len(vals) else "")
            row.append(" ".join(r["text"].split())[:400])
            w.writerow(row)

    # A sample of the calls it got wrong, carried in the results file so the
    # page can show them under the numbers. An accuracy tells you how often a
    # model is wrong; these tell you what it is wrong about, which is the
    # part that goes in a write-up. Both kinds of mistake are sampled rather
    # than the first dozen of whichever comes first in the file.
    misses = {"false_scam": [], "missed_scam": [], "unreadable": []}
    for i, (r, p) in enumerate(zip(rows, preds)):
        if p is not None and p == r["label"]:
            continue
        where = ("unreadable" if p is None
                 else "false_scam" if p == 1 else "missed_scam")
        if len(misses[where]) >= 10:
            continue
        misses[where].append({
            "idx": i, "id": r["id"], "words": r["words"],
            "excerpt": " ".join(r["text"].split())[:220],
            **{k: (v[i] if i < len(v) else "") for k, v in extra_cols.items()
               if k in ("prob_scam", "verdict", "reason", "words_counted")}})

    payload = {
        "model": name,
        "dataset": dataset,
        "calls": len(rows),
        "metrics": m,
        "baselines": base,
        "elapsed_s": round(elapsed, 2),
        "per_call_csv": short_path(csv_path),
        "misses": misses,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload.update(meta or {})
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print("  per-call results: %s" % payload["per_call_csv"])
    return payload
