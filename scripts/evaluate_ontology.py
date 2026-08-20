#!/usr/bin/env python3
"""
evaluate_ontology.py - run the ontology-RAG detector over a labelled CSV and
report the same metrics as combined_evaluate.py, so numbers are directly
comparable with the vector-RAG results.

CSV needs columns: label (scam/nonscam) and text.

Usage (on cs25003ay):
    python evaluate_ontology.py --csv datasets/scam_vs_bank_243x243.csv --limit 40
    python evaluate_ontology.py --csv datasets/scam_vs_bank_243x243.csv          # full run (tmux!)

Reference numbers to compare against (from earlier verified runs, KB-only vector RAG):
    zhi 800:   acc 92.9%  P 0.934 R 0.922 F1 0.928
    bank 486:  acc 73.5%  P 0.887 R 0.938 F1 0.912   (113 false positives)
The interesting question: does ontology matching + legit-contrast cut the
bank false positives below 113?
"""
import argparse, csv, os, random, time
from ontology_rag import OntologyRAG, DEFAULT_ONTOLOGY

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    ap.add_argument("--limit", type=int, default=0, help="run only N calls (balanced sample)")
    ap.add_argument("--model", default=None,
                help="Ollama model name (default: llama3.1:8b, or $SCAM_MODEL)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import ontology_rag
    if args.model:
        ontology_rag.MODEL = args.model
    print("model:", ontology_rag.MODEL)

    # --- default output filename includes the model so runs don't overwrite ---
    if not args.out:
        args.out = "results/ontology_results_%s.csv" % ontology_rag.MODEL.replace(":", "-")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))

    data = [(r["text"], "Fraud" if r["label"].strip().lower() == "scam" else "Normal") for r in rows]
    random.seed(42)
    random.shuffle(data)          # same seed/order convention as combined_evaluate
    if args.limit:
        fraud = [d for d in data if d[1] == "Fraud"][: args.limit // 2]
        norm  = [d for d in data if d[1] == "Normal"][: args.limit // 2]
        data = fraud + norm
        random.shuffle(data)

    det = OntologyRAG(args.ontology)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tp = fp = fn = tn = 0
    novel_count = 0
    t0 = time.time()
    results = []
    for i, (call, true) in enumerate(data):
        try:
            out = det.detect(call)
        except Exception as e:
            out = {"predicted": "Normal", "confidence": 0, "node": "ERR",
                   "node_score": 0, "novel_flag": False, "reason": str(e)[:120], "signals": {}}
        pred = out["predicted"]
        if   true == "Fraud"  and pred == "Fraud":  tp += 1
        elif true == "Normal" and pred == "Fraud":  fp += 1
        elif true == "Fraud"  and pred == "Normal": fn += 1
        else:                                       tn += 1
        novel_count += int(out["novel_flag"])
        results.append({"idx": i, "true": true, "pred": pred,
                        "confidence": out["confidence"], "node": out["node"],
                        "node_score": out["node_score"], "novel_flag": out["novel_flag"],
                        "reason": out["reason"]})
        if (i + 1) % 20 == 0:
            print("  ontology: %d/%d  (%.0fs)" % (i + 1, len(data), time.time() - t0), flush=True)

    n = len(data)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print()
    print("=" * 60)
    print("ONTOLOGY-RAG on %s  (%d calls)" % (os.path.basename(args.csv), n))
    print("=" * 60)
    print("  acc %5.1f%%  P %.3f  R %.3f  F1 %.3f   (TP%d FP%d FN%d TN%d)"
          % (100 * acc, prec, rec, f1, tp, fp, fn, tn))
    print("  novel_flag raised on %d calls (no ontology node matched)" % novel_count)
    print()
    print("  compare (vector RAG, KB-only, earlier verified runs):")
    print("    zhi 800:  acc 92.9%  F1 0.928")
    print("    bank 486: acc 73.5%  F1 0.912  (FP 113)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("\n  per-call results: %s" % args.out)

if __name__ == "__main__":
    main()
