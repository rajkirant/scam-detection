#!/usr/bin/env python3
"""
evaluate_mcq_ontology.py

The benchmark's `mcq` baseline: the MCQ ontology LLM over a labelled
transcript CSV. Each call is walked through the ontology's tree of questions
(knowledge/mcq_ontology.json unless --ontology says otherwise) - about a
dozen requests per call - and its score is the sum of the values of the
answers chosen: scam above 0, legitimate below, neutral at 0 (counted as not
scam). See mcq_ontology.py. The metric line matches combined_evaluate.py so
the numbers drop straight into the benchmark table, and the per-call CSV
carries <system>_pct (the score on 0-100, 50 at a score of 0) so the run's
Scores tab plots it.

Usage:
    python scripts/evaluate_mcq_ontology.py --csv datasets/zhi_english_646.csv --limit 20 --debug
    python scripts/evaluate_mcq_ontology.py --csv datasets/huggingface_1600.csv
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

# Transcripts run to a quarter of a million characters in
# datasets/scamai_hard_subset.csv, and the csv module refuses any field
# over 131,072 by default - with an error that names no row and no file.
csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcq_ontology as MCQ                                     # noqa: E402

SEED = 42


def load(csv_path, limit=None):
    """(text, "Fraud"|"Normal") in the benchmark's order.

    The order is fixed - seed 42, a balanced sample for --limit, then a
    shuffle - because run_all.sh's idx:<n> and check_one.py --idx reproduce
    it to pick out the n-th call of a run. Do not change it.
    """
    p = Path(csv_path)
    if not p.exists():
        sys.exit("ERROR: not found: %s" % csv_path)
    import dataset_io
    rows = dataset_io.read_rows(p)
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
    ap.add_argument("--ontology", default=str(MCQ.DEFAULT_ONTOLOGY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--system", default="mcq_ontology",
                    help="column name in the per-call CSV (run_all.sh passes "
                         "mcq_ontology__stripped for the content-deletion pass)")
    ap.add_argument("--no-quotes", action="store_true",
                    help="do not require a supporting quote for each answer")
    ap.add_argument("--debug", action="store_true",
                    help="print every answer for the first calls")
    args = ap.parse_args()
    quotes = not args.no_quotes

    try:
        onto = MCQ.load_ontology(args.ontology)
    except ValueError as e:
        sys.exit("ERROR %s" % e)
    data = load(args.csv, args.limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")

    print("=" * 74)
    print("MCQ ontology LLM - %d calls (%d scam, %d non-scam)"
          % (len(data), n_fraud, len(data) - n_fraud))
    print("  dataset : %s" % args.csv)
    print("  ontology: %s" % args.ontology)
    print("  tree    : %s - %d subjects, %d common questions, %d in all"
          % (onto["prompt"], len(onto["options"]),
             len(onto["common_questions"]), MCQ.count_questions(onto)))
    print("  answers : quotes %s; no retrieval, no learned knowledge"
          % ("required" if quotes else "off"))
    print("  verdict : the sign of the summed answer values (0 = neutral, "
          "counted as not scam)")
    import ollama_ctx
    print("  model   : %s" % ollama_ctx.MODEL)
    trained = [t.get("dataset") for t in onto.get("training") or []]
    if any(t and Path(t).name == Path(args.csv).name for t in trained):
        print("  NOTE this ontology was trained on this dataset: some of the "
              "calls scored here\n       set its options and values - a "
              "held-out set is the fairer read.")
    print("  context window %s for the run"
          % MCQ.presize([t for t, _ in data], onto, quotes))
    print("=" * 74)

    rows, results = [], []
    t0 = time.time()
    answers = MCQ.classify_all([t for t, _ in data], onto, quotes)
    for i, (text, true) in enumerate(data, 1):
        try:
            res = next(answers)
        except StopIteration:
            break
        except RuntimeError as exc:
            # Ollama went away: stop rather than score the rest as Normal
            print("    ! call %d failed: %s" % (i, exc))
            break
        pred = "Fraud" if res["verdict"] == "scam" else "Normal"
        rows.append((pred, true))
        results.append(res)
        if args.debug and i <= 3:
            print("\n  --- call %d (true: %s) ---" % (i, true))
            for r in res["answers"]:
                print("    %+.1f  %-40s %s" % (r["value"], r["path"][:40],
                                             (r["text"] or "unreadable")[:50]))
            print("    " + MCQ.explain(res))
        if i % 20 == 0:
            el = time.time() - t0
            print("    MCQ: %d/%d  (%.2fs a call)" % (i, len(data), el / i),
                  flush=True)

    elapsed = time.time() - t0
    m = metrics(rows)
    print("\n" + "=" * 74)
    show("mcq_ontology", m)
    print("    (%.0fs, %.2fs per call)" % (elapsed, elapsed / max(1, len(rows))))
    print("    %d calls neutral (score 0, counted as not scam); %d questions "
          "asked, %d answers without a quote counted as not stated"
          % (sum(1 for r in results if r["verdict"] == "neutral"),
             sum(r["requests"] for r in results),
             sum(1 for r in results for a in r["answers"]
                 if a["quoted"] is False)))
    print()
    for line in MCQ.subject_table(onto, results,
                                  [t == "Fraud" for _, t in rows]):
        print("    " + line)
    try:
        from combined_evaluate import show_score_ranges
        show_score_ranges(args.system, rows,
                          [MCQ.pct(r["score"]) for r in results])
    except ImportError:
        pass

    out = args.out or ("results/mcq_ontology_results_%d.csv" % len(data))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    s = args.system
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # <s>_pct and <s>_category are read by the web UI: the first is
        # plotted on the Scores tab, the second shown as plain text
        w.writerow(["idx", "true", "text", s, s + "_pct", s + "_category",
                    s + "_why"])
        for idx, ((text, true), (pred, _), res) in enumerate(
                zip(data, rows, results)):
            w.writerow([idx, true, text.replace("\n", " ")[:400], pred,
                        round(MCQ.pct(res["score"]), 2),
                        res["subject"] or "", MCQ.explain(res)])
    print("\n  per-call results: %s" % out)
    print("=" * 74)


if __name__ == "__main__":
    main()
