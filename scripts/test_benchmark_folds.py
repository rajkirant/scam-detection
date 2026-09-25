#!/usr/bin/env python3
"""
Every learner in the benchmark holds out the same calls in fold k.

bag of words, Qwen-KB and the hybrid get their folds from combined_evaluate.py
(shuffle with seed 42, then StratifiedKFold); BERT gets them from
dataset_io.benchmark_folds on the file as it is. This checks the two name the
same calls, fold by fold, on every dataset in datasets/ - and that
export_folds.py's files, where they exist, are exactly those folds.

    python scripts/test_benchmark_folds.py
"""
import contextlib
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import combined_evaluate as C                                  # noqa: E402
import dataset_io                                              # noqa: E402
import bert_baseline as B                                      # noqa: E402

DS = os.path.join(HERE, "..", "datasets")
bad = 0
for f in sorted(glob.glob(os.path.join(DS, "*.csv"))):
    name = os.path.basename(f)
    if re.search(r"_(train|test)\d+\.csv$", name):
        continue
    try:
        data = C.load_combined_ids(f)
        _, splits = C._stratified_folds([(t, l) for _, t, l in data], 5)
        with contextlib.redirect_stdout(io.StringIO()):
            _, texts, labels = B.load_dataset(f)
    except (SystemExit, RuntimeError) as e:
        print("skip  %s (%s)" % (name, str(e).splitlines()[0][:60]))
        continue
    same = len(texts) == len(data)
    bert = dataset_io.benchmark_folds(labels) if same else []
    for (_, te), (_, te2) in zip(splits, bert):
        if sorted(data[i][1] for i in te) != sorted(texts[i].strip() for i in te2):
            same = False
    # the exported files, if this dataset has them
    stem = name[:-4]
    for k, (_, te) in enumerate(splits, 1):
        path = os.path.join(DS, "%s_test%d.csv" % (stem, k))
        if os.path.exists(path):
            got = {i for i, _, _ in C.load_combined_ids(path)}
            if got != {data[i][0] for i in te}:
                print("FAIL  %s does not match fold %d" % (os.path.basename(path), k))
                bad += 1
    print("%s  %s" % ("ok  " if same else "FAIL", name))
    bad += not same

print("all good - every learner holds out the same calls in each fold"
      if not bad else "%d problem(s)" % bad)
sys.exit(1 if bad else 0)
