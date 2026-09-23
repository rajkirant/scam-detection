#!/usr/bin/env python3
"""
test_eval_common.py - offline checks on the shared dataset-scoring code.

All four model pages score a dataset through eval_common, which is the point:
four confusion matrices computed four ways could not be compared, and
comparing them is the only reason to have four pages. So this checks the one
implementation rather than four.

What it mostly guards is the handling of a prediction that is not a
prediction. A model that gives an unreadable answer has not said "legitimate",
and counting it as one turns every such answer into a false negative and
flatters recall. That mistake is invisible in an accuracy figure, which is
exactly why it needs a test.

    python3 scripts/test_eval_common.py

Exits non-zero on the first failure.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_common as EC                                    # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s" % (name, "ok" if ok else "FAIL"))
    if not ok:
        print("      got  %r\n      want %r" % (got, want))
        FAILS.append(name)


def head(title):
    print("\n" + title)


# ---------------------------------------------------------------------------
head("labels are read the way every script in the project reads them")
for v in ("1", "scam", "Fraud", "TRUE", "yes", 1, 1.0):
    check("%r is a scam" % (v,), EC.to_binary(v), 1)
for v in ("0", "normal", "legit", "false", "no", 0, "", "anything else"):
    check("%r is not" % (v,), EC.to_binary(v), 0)

# ---------------------------------------------------------------------------
head("the confusion matrix, on a case worked by hand")
#           true: 1 1 1 0 0 0
#           pred: 1 0 1 1 0 0
m = EC.metrics([1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0])
check("tp", m["tp"], 2)
check("fn", m["fn"], 1)
check("fp", m["fp"], 1)
check("tn", m["tn"], 2)
check("accuracy is 4 of 6", round(m["acc"], 4), round(4 / 6, 4))
check("precision is 2 of 3", round(m["precision"], 4), round(2 / 3, 4))
check("recall is 2 of 3", round(m["recall"], 4), round(2 / 3, 4))
check("specificity is 2 of 3", round(m["specificity"], 4), round(2 / 3, 4))
check("f1", round(m["f1"], 4), round(2 / 3, 4))
check("balanced accuracy is the mean of the two rates",
      round(m["balanced_acc"], 4), round(2 / 3, 4))

head("an unreadable answer is not a legitimate one")
# the failure this exists to stop: three scams, one unreadable. Counting the
# unreadable as Normal would make it a false negative and drop recall to 2/3.
u = EC.metrics([1, 1, 1, 0], [1, 1, None, 0])
check("recall is over what was answered, not what was asked", u["recall"], 1.0)
check("it is counted, not dropped silently", u["unreadable"], 1)
check("and the scores say how many they are over", u["scored"], 3)
check("the total is still the total", u["total"], 4)
check("accuracy is 3 of 3", u["acc"], 1.0)

folded = EC.metrics([1, 1, 1, 0], [1, 1, 0, 0])
check("folding it into legitimate would have said 0.667",
      round(folded["recall"], 3), 0.667)

head("an empty run does not divide by zero")
e = EC.metrics([], [])
check("accuracy", e["acc"], 0.0)
check("f1", e["f1"], 0.0)
none = EC.metrics([1, 0], [None, None])
check("nothing readable at all", (none["acc"], none["scored"],
                                  none["unreadable"]), (0.0, 0, 2))

# ---------------------------------------------------------------------------
head("the baselines are what answering the same thing every time gets")
truths = [1] * 3 + [0] * 7
b = EC.baselines(truths)
check("always scam gets the scam share", b["always_scam"]["acc"], 0.3)
check("never scam gets the rest", b["never_scam"]["acc"], 0.7)
check("always scam finds every scam", b["always_scam"]["recall"], 1.0)
check("and gets no legitimate call right", b["always_scam"]["tn"], 0)

# ---------------------------------------------------------------------------
head("a dataset loads, and a limit takes a balanced head")
with tempfile.TemporaryDirectory() as tmp:
    ds = Path(tmp) / "d.csv"
    with open(ds, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "label", "text"])
        w.writeheader()
        for i in range(30):
            w.writerow({"id": "r%d" % i, "label": 1 if i < 20 else 0,
                        "text": "word " * (i + 1)})
        w.writerow({"id": "blank", "label": 0, "text": "   "})

    rows = EC.load_rows(ds, quiet=True)
    check("the blank transcript is dropped", len(rows), 30)
    check("ids are kept", rows[0]["id"], "r0")
    check("labels are read", sum(r["label"] for r in rows), 20)
    check("word counts are counted", rows[0]["words"], 1)

    cut = EC.load_rows(ds, limit=10, quiet=True)
    check("a limit is honoured", len(cut), 10)
    check("and is balanced, not the first ten",
          sum(r["label"] for r in cut), 5)
    check("still in file order", [r["row"] for r in cut],
          sorted(r["row"] for r in cut))

    # a transcript longer than csv's own field limit, which is the bug that
    # stopped the hard subsets loading at all
    big = Path(tmp) / "big.csv"
    with open(big, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "label", "text"])
        w.writeheader()
        w.writerow({"id": 1, "label": 1, "text": "w " * 90_000})
        w.writerow({"id": 2, "label": 0, "text": "short one"})
    check("a 180kB transcript loads", len(EC.load_rows(big, quiet=True)), 2)

# ---------------------------------------------------------------------------
head("results are written where the page and a person can each find them")
with tempfile.TemporaryDirectory() as tmp:
    EC.RESULTS_DIR = Path(tmp) / "results"
    rows = [{"id": "a", "row": 0, "text": "one two", "label": 1, "words": 2},
            {"id": "b", "row": 1, "text": "three", "label": 0, "words": 1},
            {"id": "c", "row": 2, "text": "four", "label": 1, "words": 1},
            {"id": "d", "row": 3, "text": "five", "label": 0, "words": 1}]
    preds = [1, 1, None, 0]          # right, false positive, unreadable, right
    m = EC.metrics([r["label"] for r in rows], preds)
    base = EC.baselines([r["label"] for r in rows])

    out = Path(tmp) / "20260101_120000_eval_bow_x.metrics.json"
    payload = EC.write_results(out, "x", "d.csv", rows, preds, m, base, 1.5,
                               {"prob_scam": ["0.9", "0.8", "", "0.1"]},
                               {"kind": "bow"})

    check("the json landed where the page looks", out.exists(), True)
    check("it did not land on the run's own meta file",
          (Path(tmp) / "20260101_120000_eval_bow_x.json").exists(), False)
    csv_path = Path(tmp) / "results" / "eval_20260101_120000_eval_bow_x.csv"
    check("the per-call csv is named for the run", csv_path.exists(), True)

    back = json.loads(out.read_text())
    check("the metrics round-trip", back["metrics"]["tp"], 1)
    check("the extra meta is carried", back["kind"], "bow")

    with open(csv_path, newline="") as f:
        got = list(csv.DictReader(f))
    check("one row per call", len(got), 4)
    check("the truth is spelled out", got[0]["true"], "scam")
    check("a right answer is marked", got[0]["correct"], "1")
    check("a wrong one too", got[1]["correct"], "0")
    check("and an unreadable one is neither", got[2]["correct"], "")
    check("the unreadable verdict is named", got[2]["predicted"], "unreadable")
    check("the page's own column came through", got[0]["prob_scam"], "0.9")

    misses = payload["misses"]
    check("the false positive is sampled",
          [x["id"] for x in misses["false_scam"]], ["b"])
    check("the unreadable call is sampled",
          [x["id"] for x in misses["unreadable"]], ["c"])
    check("nothing was missed as a scam", misses["missed_scam"], [])
    check("a miss carries enough to read it",
          misses["false_scam"][0]["excerpt"], "three")

head("at most ten of each kind are sampled")
with tempfile.TemporaryDirectory() as tmp:
    EC.RESULTS_DIR = Path(tmp) / "results"
    many = [{"id": i, "row": i, "text": "t", "label": 0, "words": 1}
            for i in range(40)]
    p = EC.write_results(Path(tmp) / "m.json", "x", "d", many, [1] * 40,
                         EC.metrics([0] * 40, [1] * 40),
                         EC.baselines([0] * 40), 1.0)
    check("capped at ten", len(p["misses"]["false_scam"]), 10)

# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
if FAILS:
    print("%d check(s) failed: %s" % (len(FAILS), ", ".join(FAILS)))
    sys.exit(1)
print("all good - one confusion matrix for four pages, and an unreadable "
      "answer\nis never quietly counted as a legitimate call")
