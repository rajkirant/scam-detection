#!/usr/bin/env python3
"""
check_one.py - run the MCQ ontology LLM on a single row from a dataset, and
show the probability it put on every option.

Usage:
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19
    python scripts/check_one.py --csv datasets/paired_scam_legit_198.csv --idx 19 --runs 5
    python scripts/check_one.py --csv datasets/huggingface_1600.csv --raw-row 0

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

# Transcripts run to a quarter of a million characters in
# datasets/scamai_hard_subset.csv, and the csv module refuses any field
# over 131,072 by default - with an error that names no row and no file.
csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcq_ontology as MCQ                                     # noqa: E402


def load_shuffled(csv_path, limit=None, seed=42):
    """Reproduces evaluate_mcq_ontology.py's load() ordering exactly, so
    --idx here lines up with idx values from a prior evaluation run."""
    import random
    SCAM_WORDS = {"scam", "fraud", "fraudulent", "1", "true", "yes"}
    import dataset_io
    rows = dataset_io.read_rows(csv_path)
    tcol, lcol, _ = dataset_io.columns(rows, csv_path)
    data = []
    for r in rows:
        lab = "Fraud" if dataset_io.is_scam(r[lcol]) else "Normal"
        txt = (r[tcol] or "").strip()
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
    import dataset_io
    rows = dataset_io.read_rows(csv_path)
    tcol, lcol, _ = dataset_io.columns(rows, csv_path)
    r = rows[row_num]
    return r[tcol].strip(), "Fraud" if dataset_io.is_scam(r[lcol]) else "Normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--idx", type=int, help="index into the shuffled --limit 40 style order")
    ap.add_argument("--raw-row", type=int, help="row number in the file as-is, 0-indexed")
    ap.add_argument("--limit", type=int, default=40,
                    help="must match the --limit used when --idx was read off a prior run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ontology", default=str(MCQ.DEFAULT_ONTOLOGY))
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

    try:
        onto = MCQ.load_ontology(args.ontology)
    except ValueError as e:
        sys.exit("ERROR %s" % e)
    print(onto["prompt"])

    verdicts = []
    for i in range(args.runs):
        if args.runs > 1:
            print("\n--- run %d/%d ---" % (i + 1, args.runs))
        result = MCQ.judge(text, onto)
        for j, (o, p) in enumerate(zip(onto["options"], result["probs"])):
            print("  %s %5.1f%%  %-5s  %s%s"
                  % (MCQ.LETTERS[j], 100 * p, o["verdict"], o["text"][:64],
                     "  <-" if j == result["choice"] else ""))
        print(MCQ.explain(result, onto))
        predicted = result["verdict"] or "Normal"
        correct = predicted == true_label
        print("predicted:", predicted,
              " correct" if correct else " WRONG (true: %s)" % true_label)
        verdicts.append(predicted)

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