"""
Evaluate the Web-RAG system.

Mirrors singh_evaluate_ollama.py so the numbers are directly comparable:
same transcripts, same labels, same metrics.

Run from project root:
    python scripts/webrag_evaluate.py                 # KB + web (needs TAVILY_API_KEY)
    python scripts/webrag_evaluate.py --no-web        # KB only (ablation)
    python scripts/webrag_evaluate.py --threshold 60  # vary decision threshold
"""

import re
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import webrag_system as W

TRANSCRIPTS_DIR = Path("./transcripts")
LABELS_FILE = Path("./labels.csv")
RESULTS_DIR = Path("./results")


def load_labels():
    with open(LABELS_FILE, encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def compute_metrics(rows):
    tp = fp = fn = tn = 0
    for r in rows:
        p, t = r["predicted"], r["true_label"]
        if p == "Fraud" and t == "Fraud":     tp += 1
        elif p == "Fraud" and t == "Normal":  fp += 1
        elif p == "Normal" and t == "Fraud":  fn += 1
        elif p == "Normal" and t == "Normal": tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                accuracy=acc, precision=prec, recall=rec, f1=f1)


def main():
    use_web = "--no-web" not in sys.argv
    threshold = 50
    if "--threshold" in sys.argv:
        threshold = int(sys.argv[sys.argv.index("--threshold") + 1])

    mode = "KB+WEB" if use_web else "KB-ONLY"
    tag = f"{'web' if use_web else 'kbonly'}_t{threshold}"

    print("=" * 74)
    print(f"Web-RAG system evaluation  [{mode}]  threshold={threshold}")
    print("=" * 74)

    if use_web:
        import os
        if not os.environ.get("TAVILY_API_KEY"):
            print("\n  NOTE: TAVILY_API_KEY not set - web retrieval will return")
            print("        nothing. This run is effectively KB-only.\n")

    print("\nChecking Ollama...")
    try:
        print(f"  responded: '{W.call_ollama('Reply with one word: ready', 5)[:40]}'")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    collection = W.get_kb_collection()
    labels = load_labels()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        print(f"ERROR: no transcripts in {TRANSCRIPTS_DIR}")
        return

    print(f"\nEvaluating {len(files)} calls...\n")
    rows = []
    t0 = time.time()

    for i, tf in enumerate(files, 1):
        name = tf.name
        if name not in labels:
            print(f"  [{i:>3}] SKIP {name} - not in labels.csv")
            continue

        transcript = tf.read_text(encoding="utf-8")
        transcript = re.sub(r"GROUND TRUTH LABEL:.*", "", transcript,
                            flags=re.IGNORECASE).strip()
        true_label = labels[name]["label"]

        try:
            out = W.detect(transcript, collection,
                           use_web=use_web, threshold=threshold)
            mark = "OK" if out["predicted"] == true_label else "WRONG"
            print(f"  [{i:>3}] {name:<36} conf {out['confidence']:>3} "
                  f"-> {out['predicted']:<7} (true {true_label:<6}) {mark:<5} "
                  f"kb={out['kb_top_type'][:18]:<18} web={out['n_web']}")

            rows.append({
                "filename": name,
                "true_label": true_label,
                "confidence": out["confidence"],
                "predicted": out["predicted"],
                "correct": mark,
                "kb_top_type": out["kb_top_type"],
                "kb_top_distance": (f"{out['kb_top_distance']:.4f}"
                                    if out["kb_top_distance"] is not None else ""),
                "n_web": out["n_web"],
                "mean_web_credibility": f"{out['mean_web_credibility']:.3f}",
                "signals": out["signals"][:150],
                "reason": out["reason"][:200],
            })
        except Exception as e:
            print(f"  [{i:>3}] {name}: ERROR - {e}")
            rows.append({
                "filename": name, "true_label": true_label, "confidence": "",
                "predicted": "ERROR", "correct": "ERROR", "kb_top_type": "",
                "kb_top_distance": "", "n_web": 0, "mean_web_credibility": "",
                "signals": "", "reason": str(e)[:200],
            })

    elapsed = time.time() - t0
    out_file = RESULTS_DIR / f"webrag_results_{tag}.csv"
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["predicted"] in ("Fraud", "Normal")]
    m = compute_metrics(valid)

    # confidence distribution, split by true label
    fraud_conf = [r["confidence"] for r in valid if r["true_label"] == "Fraud"]
    norm_conf = [r["confidence"] for r in valid if r["true_label"] == "Normal"]

    print("\n" + "=" * 74)
    print(f"RESULTS  [{mode}]  threshold={threshold}")
    print("=" * 74)
    print(f"  model:      {W.OLLAMA_MODEL}")
    print(f"  evaluated:  {len(valid)} of {len(rows)}")
    print(f"  time:       {elapsed:.1f}s ({elapsed/max(len(rows),1):.1f}s per call)")
    print(f"\n  TP {m['TP']}   FP {m['FP']}   FN {m['FN']}   TN {m['TN']}")
    print(f"\n  Accuracy   {m['accuracy']:.4f}   ({m['accuracy']*100:.2f}%)")
    print(f"  Precision  {m['precision']:.4f}")
    print(f"  Recall     {m['recall']:.4f}")
    print(f"  F1         {m['f1']:.4f}")

    if fraud_conf and norm_conf:
        print(f"\n  Confidence separation:")
        print(f"    true Fraud  calls: min {min(fraud_conf):>3}  "
              f"mean {sum(fraud_conf)/len(fraud_conf):>5.1f}  max {max(fraud_conf):>3}")
        print(f"    true Normal calls: min {min(norm_conf):>3}  "
              f"mean {sum(norm_conf)/len(norm_conf):>5.1f}  max {max(norm_conf):>3}")
        gap = min(fraud_conf) - max(norm_conf)
        if gap > 0:
            print(f"    clean separation - any threshold in ({max(norm_conf)}, "
                  f"{min(fraud_conf)}] gives 100% accuracy")
        else:
            print(f"    overlap of {-gap} points - no threshold separates perfectly")

    print(f"\n  Singh baseline on same data: Accuracy 86.67%, F1 0.857, "
          f"Precision 0.75, Recall 1.00")
    print(f"  results: {out_file}")
    print("=" * 74)


if __name__ == "__main__":
    main()
