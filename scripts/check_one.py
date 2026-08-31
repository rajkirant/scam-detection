#!/usr/bin/env python3
"""
check_one.py - run the MCQ ontology detector on a single row from a dataset.

Usage:
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19 --runs 5
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19 --model qwen2.5:14b
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19 --no-label

--idx uses the SAME shuffled order as evaluate_mcq_ontology.py (seed 42, same
--limit), so idx 19 here is the same call as idx 19 in a prior --limit 40 run.
If you instead want a specific row from the raw CSV as it sits on disk,
use --raw-row instead of --idx.

--runs repeats the same transcript N times so you can see how much the
verdict moves between calls - useful after last session's finding that
answers on this exact transcript were not stable run to run.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcq_ontology_rag import MCQOntologyDetector, DEFAULT_ONTOLOGY


def load_shuffled(csv_path, limit=None, seed=42):
    """Reproduces evaluate_mcq_ontology.py's load() ordering exactly, so
    --idx here lines up with idx values from a prior evaluation run."""
    import random
    SCAM_WORDS = {"scam", "fraud", "fraudulent", "1", "true", "yes"}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
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
        random.seed(seed)
        data = (random.sample(scam, min(k, len(scam)))
                + random.sample(norm, min(limit - k, len(norm))))
    random.seed(seed)
    random.shuffle(data)
    return data


def load_raw_row(csv_path, row_num):
    """Row as it sits in the file, 0-indexed, ignoring shuffle entirely."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    r = rows[row_num]
    lab = "Fraud" if (r["label"] or "").strip().lower() in {"scam", "fraud", "1"} else "Normal"
    return r["text"].strip(), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--idx", type=int, help="index into the shuffled --limit 40 style order")
    ap.add_argument("--raw-row", type=int, help="row number in the file as-is, 0-indexed")
    ap.add_argument("--limit", type=int, default=40,
                    help="must match the --limit used when --idx was read off a prior run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-label", action="store_true",
                    help="disable the speaker-labelling pre-pass, for comparison")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the same transcript N times to check stability")
    args = ap.parse_args()

    if args.idx is None and args.raw_row is None:
        sys.exit("give either --idx (shuffled order) or --raw-row (file order)")

    if args.raw_row is not None:
        text, true_label = load_raw_row(args.csv, args.raw_row)
        print("row %d (file order)" % args.raw_row)
    else:
        data = load_shuffled(args.csv, limit=args.limit, seed=args.seed)
        if not (0 <= args.idx < len(data)):
            sys.exit("idx %d out of range for %d loaded rows (check --limit matches)"
                     % (args.idx, len(data)))
        text, true_label = data[args.idx]
        print("idx %d of a --limit %d, seed %d load" % (args.idx, args.limit, args.seed))

    print("true label:", true_label)
    print("text      :", text[:200] + ("..." if len(text) > 200 else ""))
    print("=" * 74)

    det = MCQOntologyDetector(args.ontology, model=args.model, debug=True,
                              label_speakers=not args.no_label)

    verdicts = []
    for i in range(args.runs):
        if args.runs > 1:
            print("\n--- run %d/%d ---" % (i + 1, args.runs))
        result = det.detect(text)
        print(det.explain(result))
        correct = result["predicted"] == true_label
        print("predicted:", result["predicted"],
              " correct" if correct else " WRONG (true: %s)" % true_label)
        verdicts.append(result["predicted"])

    if args.runs > 1:
        print("\n" + "=" * 74)
        from collections import Counter
        c = Counter(verdicts)
        print("verdicts across %d runs: %s" % (args.runs, dict(c)))
        if len(c) > 1:
            print("UNSTABLE - the same transcript produced different verdicts "
                  "on different calls. Treat any single run as noisy.")
        else:
            print("stable across all runs")


if __name__ == "__main__":
    main()