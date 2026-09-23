#!/usr/bin/env python3
"""
test_length_classify.py - offline checks on the length-only baseline.

No dataset, no sklearn, no server: the sweep, the ladder and the verdict are
arithmetic over lists of numbers, so they can be checked against hand-worked
answers. What is being guarded here is the part that is easy to get quietly
wrong - a threshold moved on the page must move the rates with it, rather
than keeping the fitted line's numbers under a different cut and looking
perfectly reasonable while doing it.

    python3 scripts/test_length_classify.py

Exits non-zero on the first failure.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import length_classify as L                                # noqa: E402

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
head("word_count is the same count trivial_length uses")
check("plain", L.word_count("one two three"), 3)
check("ragged whitespace", L.word_count("  one\n\ttwo   three  "), 3)
check("empty", L.word_count(""), 0)
check("None", L.word_count(None), 0)

# ---------------------------------------------------------------------------
head("predict points the way the direction says")
check("longer, above", L.predict(50, 45, L.LONGER), 1)
check("longer, on the line", L.predict(45, 45, L.LONGER), 0)
check("longer, below", L.predict(10, 45, L.LONGER), 0)
check("shorter, below", L.predict(10, 45, L.SHORTER), 1)
check("shorter, on the line", L.predict(45, 45, L.SHORTER), 1)
check("shorter, above", L.predict(50, 45, L.SHORTER), 0)

# ---------------------------------------------------------------------------
head("the ladder counts each class at or below every distinct length")
counts = [10, 10, 20, 30, 30, 40]
labels = [0,  0,  1,  0,  1,  1]
rows = L.ladder(counts, labels)
check("one row per distinct length, plus the floor", len(rows), 5)
check("floor row is below the shortest call", rows[0], (9, 0, 0))
check("at 10: no scams, two legitimate", rows[1], (10, 0, 2))
check("at 30: two scams, three legitimate", rows[3], (30, 2, 3))
check("at 40: every call", rows[4], (40, 3, 3))

check("below_at under everything", L.below_at(rows, 5), (0, 0))
check("below_at between rungs", L.below_at(rows, 25), (1, 2))
check("below_at above everything", L.below_at(rows, 9999), (3, 3))

# ---------------------------------------------------------------------------
head("the sweep finds a line that is really there, and points it right way")
# scams short, legitimate long - the direction trivial_length assumes is the
# wrong one here, which is the case that matters
short_scam = [5, 6, 7, 8, 9]
long_legit = [80, 90, 100, 110, 120]
c = short_scam + long_legit
y = [1] * 5 + [0] * 5
best, curve = L.sweep(c, y, "f1")
t, d, m = best
check("perfectly separable, so f1 is 1", round(m["f1"], 4), 1.0)
check("direction flipped to shorter", d, L.SHORTER)
check("threshold sits in the gap", 9 <= t < 80, True)
check("both directions in the curve", len({row[1] for row in curve}), 2)

# a set with no signal at all: every length appears in both classes
flat_c = [10, 10, 20, 20, 30, 30]
flat_y = [0, 1, 0, 1, 0, 1]
fbest, _ = L.sweep(flat_c, flat_y, "acc")
check("no signal, so no rule beats a coin", round(fbest[2]["acc"], 4) <= 0.5, True)

# ---------------------------------------------------------------------------
head("side_rates are computed at the threshold asked for, not the fitted one")
rows2 = L.ladder(c, y)
at_gap = L.side_rates(rows2, 5, 5, 40, L.SHORTER)
check("below the gap is all scam", at_gap["below"], 1.0)
check("above the gap is no scam", at_gap["above"], 0.0)
check("five calls each side", (at_gap["below_n"], at_gap["above_n"]), (5, 5))

moved = L.side_rates(rows2, 5, 5, 9999, L.SHORTER)
check("a line past everything puts all calls below", moved["below_n"], 10)
check("and its rate is the whole set's balance", moved["below"], 0.5)
check("with nothing above it, the rate there is unknown", moved["above"], None)

# ---------------------------------------------------------------------------
head("a fitted model classifies, and moving its line moves the rates")
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "t-len"
    d.mkdir()
    (d / L.MODEL_FILE).write_text(json.dumps({
        "threshold": 40, "direction": L.SHORTER,
        "rates": L.side_rates(rows2, 5, 5, 40, L.SHORTER),
        "ladder": [list(r) for r in rows2],
        "fit_scam": 5, "fit_legit": 5,
        "dist": {"scam": L.percentiles(short_scam),
                 "legit": L.percentiles(long_legit),
                 "scam_n": 5, "legit_n": 5},
        "curve": [],
    }), encoding="utf-8")
    (d / L.META_FILE).write_text(json.dumps(
        {"name": "t-len", "dataset": "made up", "swept": True}), encoding="utf-8")

    # model_dir is MODELS_DIR-relative, so point the module at the temp tree
    real = L.MODELS_DIR
    L.MODELS_DIR = Path(tmp)
    try:
        import bert_classify
        real_bert = bert_classify.MODELS_DIR
        bert_classify.MODELS_DIR = Path(tmp)
        try:
            clf = L.Classifier("t-len")
            a = clf.classify("one two three four five six")
            check("six words is under the line, so scam", a["verdict"], "scam")
            check("and the fitting calls there were all scams",
                  a["prob_scam"], 1.0)
            check("not moved", a["moved"], False)
            check("no disagreement", a["rate_disagrees"], False)

            b = clf.classify("one two three four five six", threshold=9999)
            check("the same call under a line past everything",
                  b["verdict"], "scam")
            check("rate follows the moved line, it is not the fitted one",
                  b["prob_scam"], 0.5)
            check("and the move is reported", b["moved"], True)

            c2 = clf.classify(" ".join(["w"] * 200))
            check("200 words is over the line, so legitimate",
                  c2["verdict"], "legitimate")
            check("no fitting call up there was a scam, so no contradiction",
                  c2["rate_disagrees"], False)

            flipped = clf.classify("one two three", direction=L.LONGER)
            check("asking the other direction flips the verdict",
                  flipped["verdict"], "legitimate")
            check("and that counts as moved", flipped["moved"], True)

            try:
                clf.classify("   ")
                check("an empty transcript is refused", "no error", "an error")
            except ValueError as e:
                check("an empty transcript is refused",
                      "no transcript" in str(e), True)
        finally:
            bert_classify.MODELS_DIR = real_bert
    finally:
        L.MODELS_DIR = real

# ---------------------------------------------------------------------------
head("the thinned curve keeps both directions of each threshold")
_, big = L.sweep(list(range(200)), [i % 2 for i in range(200)], "f1")
thin = L._thin(big, 40)
check("thinned to about what was asked for", len(thin) <= 42, True)
check("still both directions", len({row[1] for row in thin}), 2)
check("and they stay paired", all(thin[i][1] != thin[i + 1][1]
                                  for i in range(0, len(thin) - 1, 2)), True)

# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
if FAILS:
    print("%d check(s) failed: %s" % (len(FAILS), ", ".join(FAILS)))
    sys.exit(1)
print("all good - the floor holds, and a moved line moves its numbers with it")
