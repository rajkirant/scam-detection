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
from pathlib import Path

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


# ---------------------------------------------------------------- stripping
# The closed class. Every word a transcript can contain that is NOT in this
# set is treated as a content word and deleted.
#
# This is a whitelist rather than a part-of-speech tagger on purpose. A tagger
# means spaCy or NLTK, a model download, and an answer that changes when the
# model does - and the stripped text is not an intermediate here, it IS the
# measurement, so it has to be reproducible from this file alone and identical
# on every machine. The categories below are the standard closed classes:
# determiners, pronouns, prepositions, conjunctions, the auxiliary and modal
# verbs, negation, and the handful of particles that carry no topic.
#
# Auxiliaries are kept even though they are verbs. "is", "have" and "will"
# appear in every call ever recorded and say nothing about what one is about;
# dropping them would remove the grammar that makes the remainder readable
# without removing any information about the caller.
FUNCTION_WORDS = frozenset("""
a an the this that these those
i me my mine myself we us our ours ourselves
you your yours yourself yourselves
he him his himself she her hers herself it its itself
they them their theirs themselves
who whom whose which what one oneself
of in on at by for with to from into onto upon within without
about above across after against along among around before behind below
beneath beside besides between beyond during except inside near off out
outside over past since through throughout toward towards under underneath
until up upon via while
and or but nor so yet because as if though although unless whether than
when where why how whenever wherever however
am is are was were be been being
have has had having
do does did doing done
will would shall should can could may might must ought
not no nor none nothing never neither either both all any some each every
few many much more most other another such own same
there here then now again further too very just only even still also
""".split())

# Contractions are expanded before anything is matched. Splitting them on the
# apostrophe instead loses the word: "don't" becomes "don" + "t", neither of
# which is a function word, so a negation - which is exactly the kind of
# thing the stripped text should keep - would silently disappear.
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "shan't": "shall not", "shouldn't": "should not", "can't": "can not",
    "cannot": "can not", "couldn't": "could not", "mustn't": "must not",
    "mightn't": "might not", "ain't": "is not", "needn't": "need not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "we're": "we are", "we've": "we have",
    "we'll": "we will", "we'd": "we would", "they're": "they are",
    "they've": "they have", "they'll": "they will", "they'd": "they would",
    "he's": "he is", "he'll": "he will", "he'd": "he would",
    "she's": "she is", "she'll": "she will", "she'd": "she would",
    "it's": "it is", "it'll": "it will", "it'd": "it would",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "let's": "let us",
    "would've": "would have", "could've": "could have",
    "should've": "should have", "might've": "might have",
}

# Tokens that are not words at all - the transcripts are ASR, so these are
# rare, but a digit string is content (a card number, an amount) and goes.
_WORD = __import__("re").compile(r"[a-z]+'[a-z]+|[a-z]+")


def strip_content_words(text):
    """`text` with every content word deleted, leaving the function words.

    Deterministic, dependency-free and defined entirely by FUNCTION_WORDS
    above, so the same transcript strips to the same thing on every machine
    and in every run. Word order is kept; nothing is substituted for what is
    removed.

    Contractions are expanded first, so "don't" keeps its negation and
    "you're" keeps both its pronoun and its verb. A possessive or an unlisted
    contraction ("bank's", "gonna've") falls through and is deleted with the
    content word it is attached to, which is the right answer.
    """
    out = []
    for tok in _WORD.findall((text or "").lower()):
        for word in CONTRACTIONS.get(tok, tok).split():
            if word in FUNCTION_WORDS:
                out.append(word)
    return " ".join(out)


def strip_all(texts):
    """Strip a list of transcripts, and say what it cost.

    Returns (stripped, empties) - how many calls had no function words left
    at all matters, because those are scored on an empty string and whatever
    a model then says is its prior, not a reading of the call.
    """
    out = [strip_content_words(t) for t in texts]
    return out, sum(1 for t in out if not t)


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


def write_stripped(src, dest):
    """Write `src` with its transcripts stripped, keeping every other column.

    For the systems that take a CSV path rather than a list of transcripts -
    the ontology runners - so they can be pointed at the stripped copy
    without any of them growing a flag. Same strip_content_words as the
    in-memory path, so the two cannot drift.
    """
    rows = read_rows(src)
    if not rows:
        raise SystemExit("nothing to strip in %s" % src)
    cols = {c.lower().lstrip("\ufeff"): c for c in rows[0]}
    tcol = next((cols[c] for c in ("transcript", "text", "call",
                                   "conversation", "dialogue", "content",
                                   "body") if c in cols), None)
    if tcol is None:
        raise SystemExit("no transcript column in %s" % src)
    kept = whole = empties = 0
    for r in rows:
        original = r.get(tcol) or ""
        r[tcol] = strip_content_words(original)
        whole += len(original.split())
        kept += len(r[tcol].split())
        empties += not r[tcol]
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  stripped copy: %s  (%d of %d words kept, %.0f%%%s)"
          % (dest, kept, whole, 100.0 * kept / whole if whole else 0,
             ", %d calls left empty" % empties if empties else ""))
    return 0


def main(argv):
    if len(argv) == 3 and argv[0] == "strip":
        return write_stripped(argv[1], argv[2])
    if len(argv) != 3 or argv[0] != "check":
        print("usage: trusted.py check <original.csv> <stripped.csv>")
        print("       trusted.py strip <original.csv> <out.csv>")
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
