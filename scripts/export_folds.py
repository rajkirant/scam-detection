#!/usr/bin/env python3
"""
export_folds.py - write the benchmark's five train/test splits out as CSVs.

The benchmark never writes its folds down: each learner is trained on four of
them and scored on the fifth, five times over, in memory. This writes the same
ten sets to disk so any one fold can be trained and tested on its own - on the
BERT, bag-of-words or length pages, or by hand:

    datasets/<name>_train1.csv   datasets/<name>_test1.csv
    ...
    datasets/<name>_train5.csv   datasets/<name>_test5.csv

The folds are taken from combined_evaluate.py's own loader and splitter - the
code the benchmark runs for bag of words, Qwen-KB and the hybrid - and checked
against dataset_io.benchmark_folds, which BERT uses, before anything is
written. Rows keep the original file's columns and order, so each file loads
like any other dataset.

    python scripts/export_folds.py --csv datasets/huggingface_1600.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_io                                            # noqa: E402
import combined_evaluate as C                                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", required=True, help="the dataset to split")
    ap.add_argument("--out", default="datasets",
                    help="directory for the ten files (default: datasets/)")
    args = ap.parse_args()

    src = Path(args.csv)
    rows = dataset_io.read_rows(src)
    tcol, lcol, id_key = dataset_io.columns(rows, args.csv)
    if not id_key:
        sys.exit("ERROR: %s has no id column, so a fold cannot be traced back "
                 "to the calls in it" % args.csv)
    kept = [r for r in rows if (r[tcol] or "").strip()]
    ids = [r[id_key] for r in kept]
    if len(set(ids)) != len(ids):
        sys.exit("ERROR: %s has duplicate ids" % args.csv)

    # 1. what the benchmark itself does: load, shuffle, split
    data = C.load_combined_ids(str(src))
    _, splits = C._stratified_folds([(t, lab) for _, t, lab in data],
                                    dataset_io.FOLDS)
    bench = [[data[i][0] for i in te] for _, te in splits]

    # 2. the shared helper BERT uses - it must name exactly the same calls
    shared = dataset_io.benchmark_folds(
        [dataset_io.is_scam(r[lcol]) for r in kept])
    for k, ((_, te), want) in enumerate(zip(shared, bench), 1):
        if {ids[i] for i in te} != set(want):
            sys.exit("ERROR: fold %d differs between combined_evaluate and "
                     "dataset_io.benchmark_folds - not writing anything" % k)
    if len(shared) != len(bench):
        sys.exit("ERROR: the two disagree on the number of folds")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    stem = src.stem
    print("%s: %d calls, %d folds (seed %d)" % (src.name, len(kept),
                                                len(shared), dataset_io.SEED))
    for k, (tr, te) in enumerate(shared, 1):
        for part, idx in (("train", tr), ("test", te)):
            path = out / ("%s_%s%d.csv" % (stem, part, k))
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for i in idx:
                    w.writerow(kept[i])
            scam = sum(dataset_io.is_scam(kept[i][lcol]) for i in idx)
            print("  %-34s %5d calls  (%d scam / %d not)"
                  % (path.name, len(idx), scam, len(idx) - scam))
    # every call is in exactly one test set, and never in its own fold's train
    seen = sorted(i for _, te in shared for i in te)
    assert seen == list(range(len(kept))), "test sets do not cover the data once"
    assert all(not set(tr) & set(te) for tr, te in shared), "train/test overlap"
    print("ok  every call is in exactly one test set, and no test call is in "
          "its own fold's training set")


if __name__ == "__main__":
    main()
