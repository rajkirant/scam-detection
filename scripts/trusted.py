#!/usr/bin/env python3
"""
trusted.py - the content-deletion test, in one place.

Every system is scored twice: on the transcripts as given, and on a stripped
copy with every content word removed (nouns, verbs, adjectives, adverbs), so
that only function words such as "the", "of" and "is" remain. Whatever a
detector still gets right on the stripped copy was not earned by reading what
the caller said.

    trusted accuracy   A = a_full - max(0, a_stripped - 0.5)

0.5 is chance on a balanced set, so anything the stripped run earns above it
is subtracted from the full score. The formula lives here and nowhere else:
combined_evaluate.py, bert_baseline.py and collect_results.py all import it,
so the number in a log, in the results table and in the web UI is always the
same computation.

HOW A SYSTEM MUST BE SCORED ON THE STRIPPED COPY
------------------------------------------------
A system that learns from the dataset - bag-of-words, BERT, Qwen-KB, the
hybrid - is trained on the ORIGINAL text of each training fold, and that same
trained model then scores both the original and the stripped text of the
held-out fold. It is never trained on stripped text. Training on stripped
text asks a different question (can a fresh model learn the stripped data?),
which is a property of the dataset, not of the detector, and it cannot be
compared with a system that does not train at all.

A system that does not learn from the dataset - length, LLM-only, Singh,
Web-RAG, the ontology systems - is simply run on the stripped file.

WHY PAIRING IS BY ID
--------------------
The two files are matched on their `id` column and nothing else. Some loaders
shuffle and some sample, and matching by row position would silently pair a
call with a different call's stripped twin the moment either file was
reordered. check_pair() refuses a pair whose ids or labels disagree.

    python3 scripts/trusted.py check datasets/a.csv datasets/a_stripped.csv
"""

import csv
import sys

csv.field_size_limit(sys.maxsize)

CHANCE = 0.5
STRIPPED_SUFFIX = "__stripped"


def trusted_accuracy(a_full, a_stripped):
    """A = a_full - max(0, a_stripped - chance). Both arguments in [0, 1]."""
    return a_full - max(0.0, a_stripped - CHANCE)


def accuracy_from_counts(tp, fp, fn, tn):
    """Exact accuracy from a confusion matrix, rather than a rounded print."""
    n = tp + fp + fn + tn
    return (tp + tn) / n if n else 0.0


SCAM_WORDS = {"scam", "fraud", "fraudulent", "1", "true", "yes"}


def is_scam(label):
    """The label convention combined_evaluate.py uses for Fraud."""
    return (label or "").strip().lower() in SCAM_WORDS


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_pair(full_path, stripped_path, id_col="id"):
    """Return a list of problems with using stripped_path as full_path's twin.

    Empty list means the pair is sound: the same ids, in the same order, with
    the same labels, and no stripped transcript left empty. Order is checked
    as well as membership, because the ontology and MCQ loaders select rows by
    position; a pair in a different order would score different calls.
    """
    problems = []
    full = read_rows(full_path)
    strip = read_rows(stripped_path)
    for name, rows in (("original", full), ("stripped", strip)):
        if not rows:
            problems.append("%s file is empty" % name)
        elif id_col not in rows[0]:
            problems.append("%s file has no '%s' column, so its calls cannot be "
                            "matched to their twins" % (name, id_col))
    if problems:
        return problems

    ids_f = [r[id_col] for r in full]
    ids_s = [r[id_col] for r in strip]
    for name, ids in (("original", ids_f), ("stripped", ids_s)):
        dup = len(ids) - len(set(ids))
        if dup:
            problems.append("%s file has %d duplicate id(s)" % (name, dup))
    if len(full) != len(strip):
        problems.append("row counts differ: %d original, %d stripped"
                        % (len(full), len(strip)))
    missing = set(ids_f) - set(ids_s)
    if missing:
        problems.append("%d original id(s) have no stripped twin, e.g. %s"
                        % (len(missing), sorted(missing)[:3]))
    if not problems and ids_f != ids_s:
        problems.append("same ids but in a different order; the ontology and "
                        "MCQ loaders select rows by position, so the files "
                        "must be in the same row order")
    if problems:
        return problems

    by_id = {r[id_col]: r for r in strip}
    bad_label = [i for i, r in zip(ids_f, full)
                 if is_scam(r.get("label")) != is_scam(by_id[i].get("label"))]
    if bad_label:
        problems.append("%d call(s) have a different label in the two files, "
                        "e.g. id %s" % (len(bad_label), bad_label[0]))
    empty = [i for i in ids_f if not (by_id[i].get("text") or "").strip()]
    if empty:
        problems.append("%d stripped transcript(s) are empty, e.g. id %s"
                        % (len(empty), empty[0]))
    return problems


def main(argv):
    if len(argv) != 3 or argv[0] != "check":
        print("usage: trusted.py check <original.csv> <stripped.csv>")
        return 2
    problems = check_pair(argv[1], argv[2])
    if problems:
        for p in problems:
            print("  stripped pair rejected: " + p)
        return 1
    print("  stripped pair ok: %s <-> %s" % (argv[1], argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
