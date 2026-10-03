#!/usr/bin/env python3
"""
evaluate_mcq_ontology.py

The benchmark's `mcq` baseline: the MCQ ontology LLM over a labelled
transcript CSV. One question - the call's category - from the ontology JSON,
one request per call; see mcq_ontology.py for how the answer becomes a
category, a P(scam) and a verdict. The metric line matches combined_evaluate.py
so the numbers drop straight into the benchmark table, and the per-call CSV
carries <system>_pct (P(scam)) so the run's Scores tab plots it.

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
    ap.add_argument("--debug", action="store_true",
                    help="print every option's probability for the first calls")
    args = ap.parse_args()

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
    print("  question: %s  (%d options, one request per call)"
          % (onto["prompt"], len(onto["options"])))
    for i, o in enumerate(onto["options"]):
        print("    %s %-5s %s" % (MCQ.LETTERS[i], o["verdict"], o["text"][:62]))
    import ollama_ctx
    print("  model   : %s" % ollama_ctx.MODEL)
    built = (onto.get("built_from") or {}).get("dataset")
    if built and Path(built).name == Path(args.csv).name:
        print("  NOTE the options were built from this dataset's own "
              "categories: the model only has\n       to recognise the topic, "
              "so read the score with that in mind.")
    print("  context window %s for the run"
          % MCQ.presize([t for t, _ in data], onto))
    print("=" * 74)

    rows, results = [], []
    t0 = time.time()
    answers = MCQ.judge_all([t for t, _ in data], onto)
    for i, (text, true) in enumerate(data, 1):
        try:
            res = next(answers)
        except StopIteration:
            break
        except RuntimeError as exc:
            # Ollama went away: stop rather than score the rest as Normal
            print("    ! call %d failed: %s" % (i, exc))
            break
        pred = res["verdict"] or "Normal"     # unreadable scores Normal, as
        rows.append((pred, true))             # every other LLM system does
        results.append(res)
        if args.debug and i <= 3:
            print("\n  --- call %d (true: %s) ---" % (i, true))
            for j, (o, p) in enumerate(zip(onto["options"], res["probs"])):
                print("    %s %5.1f%%  %-5s %s" % (MCQ.LETTERS[j], 100 * p,
                                               o["verdict"], o["id"]))
            print("    " + MCQ.explain(res, onto))
        if i % 20 == 0:
            el = time.time() - t0
            print("    MCQ: %d/%d  (%.2fs a call)" % (i, len(data), el / i),
                  flush=True)

    elapsed = time.time() - t0
    m = metrics(rows)
    hows = [r["how"] for r in results]
    print("\n" + "=" * 74)
    show("mcq_ontology", m)
    print("    (%.0fs, %.2fs per call)" % (elapsed, elapsed / max(1, len(rows))))
    print("    P(scam) measured from logprobs on %d/%d calls%s"
          % (hows.count("logprobs"), len(hows),
             "; %d unreadable, scored Normal" % hows.count("unreadable")
             if "unreadable" in hows else ""))
    print()
    for line in MCQ.category_table(onto, [r["choice"] for r in results],
                                   [t == "Fraud" for _, t in rows]):
        print("    " + line)
    try:
        from combined_evaluate import show_score_ranges
        show_score_ranges(args.system, rows,
                          [None if r["p_scam"] is None else 100 * r["p_scam"]
                           for r in results])
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
                        "" if res["p_scam"] is None
                        else round(100 * res["p_scam"], 2),
                        res["category"] or "", MCQ.explain(res, onto)])
    print("\n  per-call results: %s" % out)
    print("=" * 74)


if __name__ == "__main__":
    main()
