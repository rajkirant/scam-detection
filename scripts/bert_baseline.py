#!/usr/bin/env python3
"""
bert_baseline.py - fine-tuned BERT baseline for scam call detection.

Two modes:

  cv          Stratified k-fold cross-validation on one dataset. Produces
              out-of-fold predictions for every call, so the per-call CSV is
              directly comparable with combined_results_486.csv.

  transfer    Train on one dataset, evaluate on a different one. This is the
              mode that matters for RQ1: a fine-tuned classifier is expected to
              do well in-distribution and poorly on scam types it never saw.

Why k-fold rather than a single split: with 486 calls a 80/20 split leaves ~97
test calls, and accuracy on 97 calls has a confidence interval wide enough to
hide most effects worth measuring. Cross-validation uses every call for testing
exactly once and gives a spread across folds.

Usage:
    python scripts/bert_baseline.py cv --csv datasets/scam_vs_bank_243x243.csv
    python scripts/bert_baseline.py cv --csv datasets/scam_vs_bank_243x243.csv --limit 40
    python scripts/bert_baseline.py transfer \
        --train-csv datasets/scam_vs_bank_243x243.csv \
        --test-csv  datasets/zhi_800.csv

Before running: unload the Ollama model or you will run out of VRAM.
    curl -s http://localhost:11434/api/generate \
      -d '{"model":"qwen2.5:14b","prompt":"","keep_alive":0}' > /dev/null
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# torch and transformers are imported lazily inside main() so that --help works
# on a machine without them installed.


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TEXT_CANDIDATES = ["transcript", "text", "call", "conversation", "dialogue", "content", "body"]
LABEL_CANDIDATES = ["label", "is_scam", "scam", "target", "class", "y", "ground_truth"]

SCAM_STRINGS = {"scam", "fraud", "fraudulent", "1", "true", "yes", "positive", "spam"}
LEGIT_STRINGS = {"legit", "legitimate", "nonscam", "non-scam", "non_scam", "not_scam", "notscam", "normal", "bank", "ham", "0", "false", "no", "negative"}


def pick_column(df, candidates, kind):
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # fall back: for text, the column with the longest average string
    if kind == "text":
        best, best_len = None, 0
        for c in df.columns:
            if df[c].dtype == object:
                avg = df[c].astype(str).str.len().mean()
                if avg > best_len:
                    best, best_len = c, avg
        if best is not None and best_len > 40:
            return best
    raise SystemExit(
        f"Could not identify the {kind} column. Columns present: {list(df.columns)}.\n"
        f"Pass it explicitly with --{kind}-col"
    )


def to_binary(value):
    """Map a label cell to 1 (scam) or 0 (legitimate)."""
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, float) and not pd.isna(value):
        return int(value != 0)
    s = str(value).strip().lower()
    if s in SCAM_STRINGS:
        return 1
    if s in LEGIT_STRINGS:
        return 0
    raise SystemExit(f"Cannot interpret label value: {value!r}")


def load_dataset(path, text_col=None, label_col=None, limit=None):
    if not os.path.exists(path):
        raise SystemExit(f"dataset not found: {path}")
    df = pd.read_csv(path)
    tcol = text_col or pick_column(df, TEXT_CANDIDATES, "text")
    lcol = label_col or pick_column(df, LABEL_CANDIDATES, "label")
    texts = df[tcol].astype(str).tolist()
    labels = [to_binary(v) for v in df[lcol].tolist()]
    if limit:
        # take a class-balanced head so a smoke test is not all one class
        idx_pos = [i for i, y in enumerate(labels) if y == 1][: limit // 2]
        idx_neg = [i for i, y in enumerate(labels) if y == 0][: limit - limit // 2]
        keep = sorted(idx_pos + idx_neg)
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]
        df = df.iloc[keep].reset_index(drop=True)
    print(f"  loaded {len(texts)} calls from {path}")
    print(f"  text column: '{tcol}'   label column: '{lcol}'")
    print(f"  class balance: {sum(labels)} scam / {len(labels) - sum(labels)} legitimate")
    return df, texts, labels


# ---------------------------------------------------------------------------
# Metrics, printed in the same shape as combined_evaluate.py
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fmt(name, m, suffix=""):
    return (f"  {name:<24} acc {m['acc']*100:5.1f}%  P {m['precision']:.3f}  "
            f"R {m['recall']:.3f}  F1 {m['f1']:.3f}   "
            f"(TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']}){suffix}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, max_length):
    return tokenizer(texts, truncation=True, padding="max_length",
                     max_length=max_length, return_tensors="pt")


def train_and_predict(model_name, train_texts, train_labels, eval_texts,
                      args, tokenizer=None, model_init=None):
    """Fine-tune on train_*, return (predictions, probabilities) for eval_texts."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_init is None:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        model = model_init()
    model.to(device)

    enc = encode(tokenizer, train_texts, args.max_length)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(train_labels, dtype=torch.long))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * args.epochs)
    warmup = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for input_ids, mask, labels in loader:
            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            optimiser.zero_grad()
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += out.loss.item()
        print(f"      epoch {epoch + 1}/{args.epochs}  loss {running / len(loader):.4f}")

    model.eval()
    enc_eval = encode(tokenizer, eval_texts, args.max_length)
    eval_ds = TensorDataset(enc_eval["input_ids"], enc_eval["attention_mask"])
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    preds, probs = [], []
    with torch.no_grad():
        for input_ids, mask in eval_loader:
            logits = model(input_ids=input_ids.to(device),
                           attention_mask=mask.to(device)).logits
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
            preds.extend((p >= args.threshold).long().cpu().tolist())

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"      saved model -> {args.save_model}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, probs


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_cv(args, tokenizer=None, model_init=None):
    from sklearn.model_selection import StratifiedKFold

    print("Loading dataset")
    df, texts, labels = load_dataset(args.csv, args.text_col, args.label_col, args.limit)
    print()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred = [None] * len(texts)
    oof_prob = [None] * len(texts)
    fold_metrics = []

    t_start = time.time()
    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"  fold {fold}/{args.folds}  train {len(tr_idx)}  test {len(te_idx)}")
        set_seed(args.seed + fold)
        tr_texts = [texts[i] for i in tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        te_texts = [texts[i] for i in te_idx]
        te_labels = [labels[i] for i in te_idx]

        preds, probs = train_and_predict(args.model, tr_texts, tr_labels, te_texts,
                                         args, tokenizer, model_init)
        for j, i in enumerate(te_idx):
            oof_pred[i] = preds[j]
            oof_prob[i] = probs[j]

        m = metrics(te_labels, preds)
        fold_metrics.append(m)
        print(fmt(f"fold {fold}", m))
        print()

    elapsed = time.time() - t_start
    overall = metrics(labels, oof_pred)

    print("=" * 74)
    print("BERT BASELINE  (stratified {}-fold cross-validation)".format(args.folds))
    print("=" * 74)
    for i, m in enumerate(fold_metrics, start=1):
        print(fmt(f"fold {i}", m))
    accs = [m["acc"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    print()
    print(f"  mean accuracy  {np.mean(accs)*100:5.1f}%  (sd {np.std(accs)*100:.1f})")
    print(f"  mean F1        {np.mean(f1s):.3f}  (sd {np.std(f1s):.3f})")
    print()
    print(fmt("bert (pooled OOF)", overall))
    print(f"  model: {args.model}   seed: {args.seed}   elapsed: {elapsed:.0f}s")

    out_df = df.copy()
    out_df["bert_pred"] = oof_pred
    out_df["bert_prob_scam"] = [round(p, 4) for p in oof_prob]
    out_df["bert_correct"] = [int(p == t) for p, t in zip(oof_pred, labels)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"  per-call results: {args.out}")
    print("=" * 74)

    write_summary(args, {"mode": "cv", "folds": args.folds,
                         "per_fold": fold_metrics, "pooled": overall,
                         "mean_acc": float(np.mean(accs)), "sd_acc": float(np.std(accs)),
                         "mean_f1": float(np.mean(f1s)), "sd_f1": float(np.std(f1s)),
                         "elapsed_s": round(elapsed)})
    return overall


def run_transfer(args, tokenizer=None, model_init=None):
    print("Loading training dataset")
    _, tr_texts, tr_labels = load_dataset(args.train_csv, args.text_col, args.label_col, args.limit)
    print("\nLoading test dataset")
    te_df, te_texts, te_labels = load_dataset(args.test_csv, args.text_col, args.label_col, None)
    print()

    set_seed(args.seed)
    t_start = time.time()
    preds, probs = train_and_predict(args.model, tr_texts, tr_labels, te_texts,
                                     args, tokenizer, model_init)
    elapsed = time.time() - t_start
    m = metrics(te_labels, preds)

    print("=" * 74)
    print("BERT BASELINE  (cross-dataset transfer)")
    print("=" * 74)
    print(f"  trained on: {args.train_csv}  ({len(tr_texts)} calls)")
    print(f"  tested on : {args.test_csv}  ({len(te_texts)} calls)")
    print(fmt("bert (transfer)", m))
    print(f"  model: {args.model}   seed: {args.seed}   elapsed: {elapsed:.0f}s")

    out_df = te_df.copy()
    out_df["bert_pred"] = preds
    out_df["bert_prob_scam"] = [round(p, 4) for p in probs]
    out_df["bert_correct"] = [int(p == t) for p, t in zip(preds, te_labels)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"  per-call results: {args.out}")
    print("=" * 74)

    write_summary(args, {"mode": "transfer", "train_csv": args.train_csv,
                         "test_csv": args.test_csv, "metrics": m,
                         "elapsed_s": round(elapsed)})
    return m


def write_summary(args, payload):
    path = os.path.splitext(args.out)[0] + "_summary.json"
    payload["model"] = args.model
    payload["seed"] = args.seed
    payload["epochs"] = args.epochs
    payload["max_length"] = args.max_length
    payload["batch_size"] = args.batch_size
    payload["lr"] = args.lr
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  summary: {path}")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(args):
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch is not installed. On cs25003ay:\n"
            "  pip install torch transformers scikit-learn"
        )
    if args.cpu:
        print("  device: cpu (forced)")
        return
    if not torch.cuda.is_available():
        print("  warning: CUDA not available, this will be slow on CPU")
        return
    free, total = torch.cuda.mem_get_info()
    free_mb, total_mb = free / 1024**2, total / 1024**2
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"{free_mb:.0f} MB free of {total_mb:.0f} MB")
    if free_mb < 4000:
        print("  warning: low free VRAM. If Ollama still holds a model, unload it:")
        print("    curl -s http://localhost:11434/api/generate \\")
        print("      -d '{\"model\":\"qwen2.5:14b\",\"prompt\":\"\",\"keep_alive\":0}' > /dev/null")


def build_parser():
    p = argparse.ArgumentParser(
        description="Fine-tuned BERT baseline for scam call detection",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--model", default="bert-base-uncased",
                        help="HF model id, e.g. bert-base-uncased, distilbert-base-uncased, roberta-base")
        sp.add_argument("--epochs", type=int, default=4)
        sp.add_argument("--batch-size", type=int, default=8)
        sp.add_argument("--max-length", type=int, default=256,
                        help="tokens per call; transcripts longer than this are truncated")
        sp.add_argument("--lr", type=float, default=2e-5)
        sp.add_argument("--threshold", type=float, default=0.5)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--limit", type=int, default=None, help="smoke test on N calls")
        sp.add_argument("--text-col", default=None)
        sp.add_argument("--label-col", default=None)
        sp.add_argument("--cpu", action="store_true")
        sp.add_argument("--save-model", default=None)

    cv = sub.add_parser("cv", help="stratified k-fold cross-validation")
    cv.add_argument("--csv", default="datasets/scam_vs_bank_243x243.csv")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--out", default="results/bert_results_486.csv")
    common(cv)
    cv.set_defaults(func=run_cv)

    tr = sub.add_parser("transfer", help="train on one dataset, test on another")
    tr.add_argument("--train-csv", required=True)
    tr.add_argument("--test-csv", required=True)
    tr.add_argument("--out", default="results/bert_transfer.csv")
    common(tr)
    tr.set_defaults(func=run_transfer)

    return p


def main():
    args = build_parser().parse_args()
    print("Preflight")
    preflight(args)
    print()
    args.func(args)


if __name__ == "__main__":
    main()
