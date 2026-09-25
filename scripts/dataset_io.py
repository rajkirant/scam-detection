#!/usr/bin/env python3
"""
dataset_io.py - reading a dataset CSV, the same way everywhere.

Seven scripts read datasets and each had grown its own idea of what one looks
like. Four insisted on columns called exactly "text" and "label"; the rest
took other names but opened the file without stripping a byte-order mark, so
a first column Excel had saved as "\\ufeffdialogue" matched nothing. And one
counted a call as a scam only if its label was the word "scam", which on a
dataset labelled 1/0 would have scored every call as legitimate without a
word of complaint.

This is the one answer to all three. Standard library only.
"""

import csv
import sys

# Column names accepted for each role, most specific first, compared without
# case and without a byte-order mark.
TEXT_COLS = ("transcript", "text", "dialogue", "conversation", "call",
             "content", "body")
LABEL_COLS = ("label", "is_scam", "scam", "target", "class", "y",
              "ground_truth")
ID_COLS = ("id", "call_id", "conv_id")

# A label counts as a scam if it is one of these (lowercased, stripped).
# Everything else is legitimate - "nonscam", "0", "normal", "legit", ...
SCAM_WORDS = frozenset({"scam", "fraud", "fraudulent", "1", "1.0", "true",
                        "yes", "spam", "phishing", "malicious"})


def is_scam(label):
    return str(label or "").strip().lower() in SCAM_WORDS


def read_rows(path):
    """Every row as a dict. utf-8-sig drops a byte-order mark if there is one
    and reads a plain UTF-8 file unchanged; the field limit is lifted because
    the longest transcript here is 225,978 characters and csv stops at
    131,072."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def pick(header, candidates):
    """The column in `header` that matches the first candidate, or None."""
    by_name = {(h or "").lstrip("﻿").strip().lower(): h for h in header}
    return next((by_name[c] for c in candidates if c in by_name), None)


def columns(rows, path="the dataset", need_label=True):
    """(text column, label column, id column or None), or a clear exit.

    The error names the columns the file does have and the names that would
    have been accepted, which is the thing to know when it goes wrong.
    """
    if not rows:
        raise SystemExit("ERROR: %s has no rows" % path)
    header = list(rows[0].keys())
    tcol = pick(header, TEXT_COLS)
    lcol = pick(header, LABEL_COLS)
    if tcol is None or (need_label and lcol is None):
        found = [h.lstrip("﻿") for h in header]
        missing = []
        if tcol is None:
            missing.append("a transcript column (one of: %s)" % ", ".join(TEXT_COLS))
        if need_label and lcol is None:
            missing.append("a label column (one of: %s)" % ", ".join(LABEL_COLS))
        raise SystemExit("ERROR: %s needs %s. It has: %s"
                         % (path, " and ".join(missing), ", ".join(found)))
    return tcol, lcol, pick(header, ID_COLS)


# The benchmark's cross-validation split. Every learner that trains on the
# dataset - bag of words, Qwen-KB, the hybrid and BERT - must hold out the SAME
# calls in each fold, or a per-fold comparison between them compares different
# test sets. combined_evaluate.py shuffles the calls with random.seed(SEED)
# before StratifiedKFold sees them; this reproduces that exactly for any list
# of labels in file order, and hands back indices in file order.
FOLDS = 5
SEED = 42


def benchmark_folds(labels, folds=FOLDS, seed=SEED):
    """[(train_idx, test_idx), ...] in file-order indices, fold 1 first.

    `labels` is one truthy/falsy value per call, in the order the calls appear
    in the file (empty-text rows already dropped, as combined_evaluate does).
    random.shuffle's permutation depends only on the list's length, so
    shuffling range(n) gives the same reordering combined_evaluate applies to
    its list of calls. The fold count shrinks to the smaller class when that
    is under `folds`, the same rule combined_evaluate uses.
    """
    import random
    from sklearn.model_selection import StratifiedKFold

    n = len(labels)
    perm = list(range(n))
    random.seed(seed)
    random.shuffle(perm)
    y = [1 if labels[p] else 0 for p in perm]
    smallest = min(sum(y), n - sum(y))
    n_splits = max(2, min(folds, smallest))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(sorted(perm[i] for i in tr), sorted(perm[i] for i in te))
            for tr, te in skf.split([[0]] * n, y)]
