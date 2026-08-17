#!/usr/bin/env python3
"""Parse the per-baseline logs from a run and print one aligned table."""
import re, sys, os

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "."

METRIC = re.compile(
    r"acc\s+([\d.]+)%\s+P\s+([\d.]+)\s+R\s+([\d.]+)\s+F1\s+([\d.]+)"
    r"\s*\(TP(\d+)\s+FP(\d+)\s+FN(\d+)\s+TN(\d+)\)")

# label -> (logfile, how to pick the line)
WANT = [
    ("length",       "combined.log", r"^\s*length\b"),
    ("bow",          "combined.log", r"^\s*bow\b"),
    ("llm_only",     "combined.log", r"^\s*llm_only\b"),
    ("singh",        "combined.log", r"^\s*singh\b"),
    ("webrag",       "combined.log", r"^\s*webrag\b"),
    ("ontology_rag", "ontology.log", None),          # last metric line
    ("bert",         "bert.log",     r"pooled OOF"),
]

rows = []
for label, fname, pattern in WANT:
    path = os.path.join(LOGDIR, fname)
    if not os.path.exists(path):
        rows.append((label, None)); continue
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    hit = None
    for ln in lines:
        if not METRIC.search(ln):
            continue
        if pattern is None or re.search(pattern, ln):
            hit = METRIC.search(ln)       # keep last match
    rows.append((label, hit))

w = 74
print("=" * w)
print(f" ALL RESULTS   logs: {LOGDIR}")
print("=" * w)
print(f"  {'SYSTEM':<14}{'ACC':>7}  {'P':>5}  {'R':>5}  {'F1':>5}   "
      f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
print("  " + "-" * (w - 4))
for label, m in rows:
    if m is None:
        print(f"  {label:<14}{'--':>7}   (not run or failed)")
        continue
    acc, p, r, f1, tp, fp, fn, tn = m.groups()
    print(f"  {label:<14}{acc:>6}%  {p:>5}  {r:>5}  {f1:>5}   "
          f"{tp:>4} {fp:>4} {fn:>4} {tn:>4}")

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
