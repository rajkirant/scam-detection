#!/usr/bin/env python3
"""
evaluate_mcq_ontology.py

Evaluate the MCQ-scored ontology detector on a labelled transcript CSV.
Output format matches combined_evaluate.py so the numbers drop straight into
the benchmark table.

Usage:
    python scripts/evaluate_mcq_ontology.py --csv datasets/zhi_scam_vs_legit_794.csv --limit 20 --debug
    python scripts/evaluate_mcq_ontology.py --csv datasets/zhi_scam_vs_legit_794.csv --model qwen2.5:14b

Two calls to the model per transcript (routing, then the branch questions),
so expect roughly twice the wall time of a single-prompt baseline.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcq_ontology_rag import MCQOntologyDetector, DEFAULT_ONTOLOGY

SEED = 42
SCAM_WORDS = {"scam", "fraud", "fraudulent", "1", "true", "yes"}


def load(csv_path, limit=None):
    p = Path(csv_path)
    if not p.exists():
        sys.exit("ERROR: not found: %s" % csv_path)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows or "text" not in rows[0] or "label" not in rows[0]:
        sys.exit("ERROR: CSV needs 'text' and 'label' columns")
    data = []
    for r in rows:
        lab = "Fraud" if (r["label"] or "").strip().lower() in SCAM_WORDS else "Normal"
        txt = (r["text"] or "").strip()
        if txt:
            data.append((txt, lab))
    if limit:
        scam = [d for d in data if d[1] == "Fraud"]
        norm = [d for d in data if d[1] == "Normal"]
        k = limit // 2
        random.seed(SEED)
        data = (random.sample(scam, min(k, len(scam)))
                + random.sample(norm, min(limit - k, len(norm))))
    random.seed(SEED)
    random.shuffle(data)
    return data


def metrics(rows):
    tp = sum(1 for p, t in rows if p == "Fraud" and t == "Fraud")
    fp = sum(1 for p, t in rows if p == "Fraud" and t == "Normal")
    fn = sum(1 for p, t in rows if p == "Normal" and t == "Fraud")
    tn = sum(1 for p, t in rows if p == "Normal" and t == "Normal")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    ap.add_argument("--model", default=os.environ.get("SCAM_MODEL", "llama3.1:8b"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--no-verify-quotes", action="store_true",
                    help="skip checking quotes against the transcript "
                         "(use to measure how much the check is worth)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.environ["SCAM_MODEL"] = args.model
    data = load(args.csv, args.limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")

    print("=" * 74)
    print("MCQ ontology evaluation - %d calls (%d scam, %d non-scam)"
          % (len(data), n_fraud, len(data) - n_fraud))
    print("  dataset : %s" % args.csv)
    print("  ontology: %s" % args.ontology)
    print("  model   : %s" % args.model)
    print("  quote verification: %s" % ("OFF" if args.no_verify_quotes else "on"))
    print("=" * 74)

    det = MCQOntologyDetector(args.ontology, model=args.model,
                              max_tokens=args.max_tokens,
                              verify_quotes=not args.no_verify_quotes,
                              debug=args.debug)

    rows, records = [], []
    t0 = time.time()
    for i, (text, true) in enumerate(data, 1):
        try:
            res = det.detect(text)
        except Exception as exc:
            print("    ! call %d failed: %s" % (i, exc))
            res = {"call_type": "other", "raw_score": 0.0, "normalised": 0.0,
                   "band": "uncertain", "predicted": "Normal", "detail": []}
        rows.append((res["predicted"], true))
        records.append((i - 1, true, text, res))

        if args.debug and i <= 3:
            print("\n  --- call %d (true: %s) ---" % (i, true))
            print(det.explain(res))
            print()
        if i % 20 == 0:
            print("    MCQ: %d/%d" % (i, len(data)))

    elapsed = time.time() - t0
    m = metrics(rows)

    print("\n" + "=" * 74)
    show("mcq_ontology", m)
    print("    (%.0fs, %.1fs per call)" % (elapsed, elapsed / max(1, len(data))))
    det.stats.report()

    out = args.out or ("results/mcq_ontology_results_%d.csv" % len(data))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true", "predicted", "call_type", "raw_score",
                    "normalised", "band", "text", "detail_json"])
        for idx, true, text, res in records:
            w.writerow([idx, true, res["predicted"], res["call_type"],
                        res.get("raw_score"), res.get("normalised"),
                        res.get("band"), text.replace("\n", " ")[:400],
                        json.dumps(res.get("detail", []))])
    print("  per-call results: %s" % out)
    print("=" * 74)


if __name__ == "__main__":
    main()
