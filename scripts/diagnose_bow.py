#!/usr/bin/env python3
"""
diagnose_bow.py  -  Is the bag-of-words 100% an ARTEFACT (Case A) or REAL/SATURATED (Case B)?

Check 1: prints the words most driving 'scam' vs 'legit'.
Check 2: strips corpus-identity words (e.g. 'harper valley national bank') and re-probes.
         - stays ~100%  -> separation is general scam-vs-legit language  = Case B
         - drops sharply -> a chunk of the 100% was corpus boilerplate    = Case A

Usage:
    python diagnose_bow.py --csv ./datasets/scam_vs_bank_243x243.csv
    python diagnose_bow.py --csv ./datasets/scam_vs_bank_243x243.csv --kill harper,valley,national,bank,robert
"""
import argparse, csv, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--kill", default="harper,valley,national,bank,robert",
                help="comma-separated corpus-identity words to ablate in Check 2")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
X = [r["text"] for r in rows]
y = np.array([1 if r["label"].strip().lower() == "scam" else 0 for r in rows])
print("loaded %d rows: %d scam, %d legit\n" % (len(rows), int(y.sum()), int((y == 0).sum())))

# ---- Check 1: top features -------------------------------------------------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
Xf = vec.fit_transform(X)
lr = LogisticRegression(max_iter=2000).fit(Xf, y)
names = np.array(vec.get_feature_names_out())
coef = lr.coef_[0]
print("=== CHECK 1: what bag-of-words keys on ===")
print("MOST SCAM-INDICATIVE :", ", ".join(names[np.argsort(coef)[-20:]][::-1]))
print("MOST LEGIT-INDICATIVE:", ", ".join(names[np.argsort(coef)[:20]]))
print("  -> fraud-meaningful words = leaning Case B; corpus boilerplate = leaning Case A\n")

# ---- baseline score (no scrub) ---------------------------------------------
def score(texts):
    res = cross_validate(
        make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                      LogisticRegression(max_iter=2000)),
        texts, y, cv=5, scoring=["accuracy"])
    return 100 * res["test_accuracy"].mean()

base = score(X)

# ---- Check 2: ablate corpus-identity words ---------------------------------
kill = [w.strip().lower() for w in args.kill.split(",") if w.strip()]
def scrub(t):
    for w in kill:
        t = re.sub(r"\b" + re.escape(w) + r"\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t)
Xs = [scrub(t) for t in X]
scrubbed = score(Xs)

print("=== CHECK 2: ablation ===")
print("killed words: %s" % ", ".join(kill))
print("bag-of-words BEFORE scrub: %.1f%%" % base)
print("bag-of-words AFTER  scrub: %.1f%%" % scrubbed)
print("drop: %.1f points\n" % (base - scrubbed))

if base - scrubbed < 3:
    print("VERDICT: barely moved -> separation is GENERAL scam-vs-legit language.")
    print("         Case B (real signal / saturated). The dataset is too EASY to rank systems,")
    print("         not artefactual. Pivot evaluation to novel + adversarial scams.")
elif scrubbed < 80:
    print("VERDICT: big drop -> much of the 100%% was corpus boilerplate.")
    print("         Case A (artefact). Neutralise these words and re-evaluate.")
else:
    print("VERDICT: partial drop -> MOSTLY real signal with some corpus-identity leakage.")
    print("         Mixed A/B. Report both; still too easy/saturated to rank systems.")
