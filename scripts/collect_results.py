#!/usr/bin/env python3
"""Parse the per-baseline logs from a run and print one aligned table."""
import re, sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trusted                                              # noqa: E402

# --json prints the same numbers as a JSON object instead of the table, so
# the web UI can render real HTML rows without a second copy of the regex.
AS_JSON = "--json" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
LOGDIR = ARGS[0] if ARGS else "."

METRIC = re.compile(
    r"acc\s+([\d.]+)%\s+P\s+([\d.]+)\s+R\s+([\d.]+)\s+F1\s+([\d.]+)"
    r"\s*\(TP(\d+)\s+FP(\d+)\s+FN(\d+)\s+TN(\d+)\)")

# label -> (logfile, how to pick the line)
WANT = [
    ("length",       "combined.log", r"^\s*length\b"),
    # --trivial-only stops before the SUMMARY block, so the only bow line in
    # the log is the friendly one: "bag-of-words (TF-IDF+LR)"
    ("bow",          "combined.log", r"^\s*(bow|bag-of-words)\b"),
    ("llm_only",     "combined.log", r"^\s*llm_only\b"),
    ("singh",        "combined.log", r"^\s*singh\b"),
    ("webrag",       "combined.log", r"^\s*webrag\b"),
    ("qwen_kb",       "combined.log", r"^\s*qwen_kb\b"),
    ("hybrid",       "combined.log", r"^\s*hybrid\b"),
    ("ontology_rag", "ontology.log", None),          # last metric line
    ("mcq_ontology", "mcq.log",      r"^\s*mcq_ontology\b"),
    ("bert",         "bert.log",     r"pooled OOF"),
]

# The content-deletion test: each system's score on the stripped twin of the
# dataset, from a run given --stripped. Learning systems were trained on the
# original text and only scored on the stripped copy (see scripts/trusted.py).
# The <key>__stripped patterns cannot match a full row: "\b" after "bow" fails
# on "bow__stripped" because "_" is a word character. BERT's stripped row says
# "stripped OOF", never "pooled OOF". The two ontology systems learn nothing
# from the data, so run_all.sh simply runs them again on the stripped file,
# into their own logs.
WANT_STRIPPED = {
    "length":       ("combined.log",          r"^\s*length__stripped\b"),
    "bow":          ("combined.log",          r"^\s*bow__stripped\b"),
    "llm_only":     ("combined.log",          r"^\s*llm_only__stripped\b"),
    "singh":        ("combined.log",          r"^\s*singh__stripped\b"),
    "webrag":       ("combined.log",          r"^\s*webrag__stripped\b"),
    "qwen_kb":      ("combined.log",          r"^\s*qwen_kb__stripped\b"),
    "hybrid":       ("combined.log",          r"^\s*hybrid__stripped\b"),
    "ontology_rag": ("ontology_stripped.log", None),
    "mcq_ontology": ("mcq_stripped.log",      r"^\s*mcq_ontology\b"),
    "bert":         ("bert.log",              r"stripped OOF"),
}


def find(fname, pattern):
    """The last metric line in fname matching pattern (any line if None)."""
    path = os.path.join(LOGDIR, fname)
    if not os.path.exists(path):
        return None
    hit = None
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        if not METRIC.search(ln):
            continue
        if pattern is None or re.search(pattern, ln):
            hit = METRIC.search(ln)       # keep last match
    return hit


def counts(m):
    """(tp, fp, fn, tn) from a matched metric line."""
    return tuple(int(x) for x in m.groups()[4:8])


def trusted_pct(full, stripped):
    """Trusted accuracy in percent, from exact counts rather than the rounded
    accuracy printed on each line, so it matches the log's own table."""
    af = trusted.accuracy_from_counts(*counts(full))
    ast = trusted.accuracy_from_counts(*counts(stripped))
    return 100.0 * trusted.trusted_accuracy(af, ast)


rows = []
for label, fname, pattern in WANT:
    full = find(fname, pattern)
    strip = None
    if full is not None and label in WANT_STRIPPED:
        strip = find(*WANT_STRIPPED[label])
    rows.append((label, full, strip))
PAIRED = any(st is not None for _, _, st in rows)

def as_dict(m):
    acc, pr, rc, f1, tp, fp, fn, tn = m.groups()
    return {"acc": float(acc), "p": float(pr), "r": float(rc), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


if AS_JSON:
    out = {"logdir": LOGDIR, "systems": [], "paired": PAIRED}
    for label, m, st in rows:
        if m is None:
            out["systems"].append({"system": label, "ran": False})
            continue
        entry = {"system": label, "ran": True, **as_dict(m)}
        if st is not None:
            entry["stripped"] = as_dict(st)
            entry["trusted"] = round(trusted_pct(m, st), 1)
        out["systems"].append(entry)
    bert_log = os.path.join(LOGDIR, "bert.log")
    if os.path.exists(bert_log):
        txt = open(bert_log, encoding="utf-8", errors="replace").read()
        out["bert_spread"] = [e.strip() for e in
                              re.findall(r"(mean accuracy.*|mean F1.*)", txt)]
    print(json.dumps(out))
    sys.exit(0)

w = 74
print("=" * w)
print(f" ALL RESULTS   logs: {LOGDIR}")
print("=" * w)
print(f"  {'SYSTEM':<14}{'ACC':>7}  {'P':>5}  {'R':>5}  {'F1':>5}   "
      f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
print("  " + "-" * (w - 4))
for label, m, _ in rows:
    if m is None:
        print(f"  {label:<14}{'--':>7}   (not run or failed)")
        continue
    acc, p, r, f1, tp, fp, fn, tn = m.groups()
    print(f"  {label:<14}{acc:>6}%  {p:>5}  {r:>5}  {f1:>5}   "
          f"{tp:>4} {fp:>4} {fn:>4} {tn:>4}")

if PAIRED:
    print()
    print("  CONTENT-DELETION TEST   trained on original text, scored on stripped")
    print("  A = a_full - max(0, a_stripped - 0.5)")
    print(f"  {'SYSTEM':<14}{'a_full':>8}{'a_stripped':>12}{'A':>8}"
          f"   {'FP':>4} {'FN':>4}  (stripped)")
    print("  " + "-" * (w - 4))
    for label, m, st in rows:
        if m is None or st is None:
            continue
        s_acc = st.groups()[0]
        _, fp_s, fn_s, _ = counts(st)
        print(f"  {label:<14}{m.groups()[0]:>7}%{s_acc:>11}%"
              f"{trusted_pct(m, st):>7.1f}%   {fp_s:>4} {fn_s:>4}")

# BERT fold spread, if present
bert_log = os.path.join(LOGDIR, "bert.log")
if os.path.exists(bert_log):
    txt = open(bert_log, encoding="utf-8", errors="replace").read()
    extra = re.findall(r"(mean accuracy.*|mean F1.*)", txt)
    if extra:
        print()
        for e in extra:
            print(f"  bert cv spread: {e.strip()}")
print("=" * w)

# suppress BrokenPipeError when output is piped to head
try:
    sys.stdout.flush()
except BrokenPipeError:
    os._exit(0)
