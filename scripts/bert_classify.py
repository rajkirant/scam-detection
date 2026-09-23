#!/usr/bin/env python3
"""
bert_classify.py - train a BERT on scam calls, keep the checkpoint, and ask it
whether one call is a scam.

Two halves:

  train     Fine-tunes a BERT for sequence classification on a labelled
            dataset - the same job bert_baseline.py does, but trained once on
            the whole set rather than per fold, and kept. The checkpoint lands
            in models/<name>/ with a meta.json beside it recording what it was
            trained on and how it scored on a held-out slice.

  classify  Loads one of those checkpoints and puts a transcript to it. The
            answer is the one thing training actually fits: the probability
            that this call is a scam.

BERT reads at most 512 tokens and these checkpoints are trained at 256, which
is about 180 words - and the calls in datasets/scamai_hard_307_ordered.csv run
to eleven thousand. Handing the whole transcript to the tokenizer would score
the first two minutes of the call and silently ignore the rest, which is the
same failure the Ollama context window had (see scripts/ollama_ctx.py). So the
transcript is cut into overlapping windows the size training used, every
window is scored, and the call's probability is the aggregate:

  max    the highest-scoring stretch of the call decides it. A scam signal
         anywhere is a scam signal, and most of a scam call is small talk.
  mean   the average across windows, which is steadier but dilutes a short
         telling exchange in a long friendly call.

Both are reported either way, along with what a single truncated read would
have said, so the difference is visible rather than assumed.

Usage:
    python scripts/bert_classify.py train --csv datasets/zhi_english_646.csv \
        --name zhi-bert --epochs 4
    python scripts/bert_classify.py classify --name zhi-bert --text "Hello..."
    python scripts/bert_classify.py classify --name zhi-bert \
        --csv datasets/scambait_bank_422.csv --idx 3 --json
    python scripts/bert_classify.py models
    python scripts/bert_classify.py serve --name zhi-bert   # one JSON per line

scripts/web_ui.py drives all of these from the BERT page; nothing here needs
the web UI to be useful.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A saved model is addressed by name from the web UI, so the name has to be a
# single path segment and nothing else - it is joined onto models/.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")

# Newer checkpoints write meta.json; the ones trained before this file was
# split out wrote mcq_meta.json. Read either, so an existing models/ keeps
# working.
META_NAMES = ("meta.json", "mcq_meta.json")

# ASR transcripts carry annotator tone tags - [curious], [long pause] - that
# nobody said out loud. --strip-tags takes them out, which is an experiment
# rather than a default: training read them, so a checkpoint asked about text
# without them is being asked about text of a kind it never saw. Worth running
# both ways, because in scamai_full_1000.csv [satisfied] sits on 54.8% of
# legitimate calls and 31.0% of scams, and a model can score well off that
# rather than off the call.
TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z _-]{0,18})\]")
ENTITY_WORDS = {
    "name": "name", "title": "mr", "org": "company", "organisation": "company",
    "company": "company", "bank": "bank", "address": "address", "city": "city",
    "location": "location", "card": "card number", "phone": "phone number",
    "ssn": "social security number", "dob": "date of birth", "zip": "postcode",
    "postcode": "postcode", "email": "email address", "url": "website",
    "number": "number", "amount": "amount", "money": "amount",
    "date": "date", "time": "time", "account": "account number",
    "sortcode": "sort code", "sort code": "sort code", "id": "id number",
    "pin": "pin", "password": "password", "otp": "one time passcode",
    "code": "code", "reference": "reference number",
}


def strip_tags(text):
    """Redacted entities keep the plain words for what they were; tone tags
    and [noise] become nothing."""
    def sub(m):
        key = m.group(1).lower().replace("_", " ").strip()
        return " %s " % ENTITY_WORDS[key] if key in ENTITY_WORDS else " "
    return re.sub(r"\s+", " ", TAG_RE.sub(sub, text or "")).strip()


def checked_name(name):
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise SystemExit(
            "model name must be letters, digits, dot, dash or underscore "
            "(got %r)" % name)
    return name


def model_dir(name):
    return MODELS_DIR / checked_name(name)


def read_meta(d):
    for fname in META_NAMES:
        try:
            return json.loads((d / fname).read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


# ---------------------------------------------------------------------------
# Saved models
# ---------------------------------------------------------------------------

def list_models():
    """Every checkpoint in models/, newest first, with whatever metadata it
    kept. A directory with no config.json is not a model and is skipped."""
    out = []
    if not MODELS_DIR.is_dir():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        meta = read_meta(d)
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out.append({
            "name": d.name,
            "base": meta.get("base_model", "?"),
            "dataset": meta.get("dataset"),
            "rows": meta.get("rows"),
            "epochs": meta.get("epochs"),
            "holdout": meta.get("holdout"),
            "trained_at": meta.get("trained_at"),
            "elapsed_s": meta.get("elapsed_s"),
            "device": meta.get("device"),
            "max_length": meta.get("max_length"),
            "threshold": meta.get("threshold", 0.5),
            "bytes": size,
            "described": bool(meta),
        })
    out.sort(key=lambda m: m.get("trained_at") or "", reverse=True)
    return out


def delete_model(name):
    import shutil
    d = model_dir(name)
    if d.resolve().parent != MODELS_DIR.resolve() or not d.is_dir():
        raise SystemExit("no such model: %s" % name)
    shutil.rmtree(d)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(args):
    import bert_baseline as bb

    name = checked_name(args.name)
    dest = model_dir(name)
    if dest.exists() and not args.overwrite:
        raise SystemExit("models/%s already exists - pick another name, or "
                         "delete it first" % name)

    print("Preflight")
    bb.preflight(args)
    print()

    print("Loading dataset")
    _, texts, labels = bb.load_dataset(args.csv, args.text_col, args.label_col,
                                       args.limit)
    if len(set(labels)) < 2:
        raise SystemExit("the dataset has only one class in it - nothing to learn")
    print()

    bb.set_seed(args.seed)

    # A held-out slice, stratified, purely so the saved model comes with a
    # number attached. It is not the benchmark - run_all.sh --baseline bert
    # does k-fold for that - it is the check that this checkpoint learnt
    # anything at all before it is asked about a call.
    from sklearn.model_selection import train_test_split
    idx = list(range(len(texts)))
    if args.holdout > 0 and len(texts) >= 10:
        tr_idx, te_idx = train_test_split(idx, test_size=args.holdout,
                                          random_state=args.seed,
                                          stratify=labels)
    else:
        tr_idx, te_idx = idx, []
    tr_texts = [texts[i] for i in tr_idx]
    tr_labels = [labels[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    te_labels = [labels[i] for i in te_idx]

    print("==> train")
    print("  %d calls to train on, %d held out" % (len(tr_texts), len(te_texts)))
    dest.mkdir(parents=True, exist_ok=True)
    args.save_model = str(dest)

    t0 = time.time()
    # train_and_predict always predicts something; with no holdout it is given
    # one training row and the predictions are thrown away - the side effect,
    # the saved checkpoint, is the point in that case.
    preds, _ = bb.train_and_predict(args.model, tr_texts, tr_labels,
                                    te_texts or tr_texts[:1], args)
    elapsed = time.time() - t0

    holdout = None
    if te_texts:
        m = bb.metrics(te_labels, preds)
        holdout = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items()}
        print()
        print(bb.fmt("holdout", m))

    import torch
    meta = {
        "name": name,
        "base_model": args.model,
        "dataset": args.csv,
        "rows": len(texts),
        "scam": int(sum(labels)),
        "legit": int(len(labels) - sum(labels)),
        "trained_rows": len(tr_texts),
        "holdout_rows": len(te_texts),
        "holdout": holdout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "seed": args.seed,
        "threshold": args.threshold,
        "device": "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda",
        "elapsed_s": round(elapsed),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print("  ok trained  models/%s  (%.0fs)" % (name, elapsed))
    print("  ask it about a call:  python scripts/bert_classify.py classify "
          "--name %s --text \"...\"" % name)
    return meta


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def windows_of(text, size, stride, cap=64):
    """Overlapping word windows.

    A checkpoint trained at 256 tokens reads about 180 words. Feeding it a
    ten-thousand-word call would score the opening and drop the rest, so the
    call is cut into pieces the size training used and every piece is scored.
    """
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    out = []
    for i in range(0, len(words), stride):
        chunk = words[i:i + size]
        if len(chunk) < min(size, 12) and out:
            break
        out.append(" ".join(chunk))
        if i + size >= len(words):
            break
    if len(out) > cap:                      # keep both ends, thin the middle
        step = max(1, (len(out) - 2) // (cap - 2))
        out = [out[0]] + out[1:-1:step][:cap - 2] + [out[-1]]
    return out


class Classifier:
    """One loaded checkpoint, answering scam or not for any transcript.

    Held open by serve() so the web UI pays the model load once rather than
    once per question. Inference is on CPU unless gpu=True: a benchmark run
    wants the whole GPU, and scoring one transcript on CPU takes well under a
    second once the model is in memory.
    """

    def __init__(self, name, gpu=False, max_length=None, window=None,
                 stride=None, aggregate="max", strip=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.name = checked_name(name)
        d = model_dir(self.name)
        if not (d / "config.json").exists():
            raise SystemExit("no trained model at models/%s - train one first"
                             % self.name)
        self.torch = torch
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available()
                                   else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(d))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(d)).to(self.device).eval()
        self.meta = read_meta(d) or {"name": self.name}

        # Default to the window training used, so every piece scored is a
        # piece of the kind the model was fitted on.
        self.max_length = int(max_length or self.meta.get("max_length") or 256)
        self.window = int(window or max(32, int(self.max_length / 1.4)))
        self.stride = int(stride or max(8, self.window // 2))
        self.aggregate = aggregate if aggregate in ("max", "mean") else "max"
        self.strip = strip
        self.threshold = float(self.meta.get("threshold", 0.5) or 0.5)

    def _scores(self, texts):
        """P(scam) for each piece, in order."""
        torch = self.torch
        out = []
        for i in range(0, len(texts), 16):
            enc = self.tokenizer(texts[i:i + 16], truncation=True, padding=True,
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            out += [float(p) for p in probs]
        return out

    def classify(self, transcript, threshold=None, aggregate=None, strip=None):
        """aggregate, threshold and strip are per-call rather than per-load:
        they change what is done with the scores, not how the model reads, so
        toggling one should not cost a model reload."""
        t0 = time.time()
        raw = (transcript or "").strip()
        if not raw:
            raise ValueError("there is no transcript to classify")
        strip = self.strip if strip is None else bool(strip)
        how = aggregate if aggregate in ("max", "mean") else self.aggregate
        text = strip_tags(raw) if strip else raw
        if not text:
            raise ValueError("there is nothing left of that transcript once "
                             "the tone tags are taken out")

        wins = windows_of(text, self.window, self.stride)
        scores = self._scores(wins)
        cut = float(self.threshold if threshold is None else threshold)

        top = max(scores)
        avg = sum(scores) / len(scores)
        prob = top if how == "max" else avg
        hottest = scores.index(top)

        return {
            "ok": True,
            "model": {"name": self.name, "base": self.meta.get("base_model"),
                      "dataset": self.meta.get("dataset"),
                      "trained_at": self.meta.get("trained_at"),
                      "holdout": self.meta.get("holdout")},
            "device": str(self.device),
            "prob_scam": round(prob, 4),
            "verdict": "scam" if prob >= cut else "legitimate",
            "threshold": round(cut, 4),
            "aggregate": how,
            "max": round(top, 4),
            "mean": round(avg, 4),
            # what a single read would have said, which is what this used to
            # do: the opening of the call and nothing else
            "first_window": round(scores[0], 4),
            "windows": len(wins),
            "window_words": self.window,
            "stride": self.stride,
            "max_length": self.max_length,
            "stripped_tags": strip,
            "words": len(text.split()),
            "hottest": {"index": hottest, "prob": round(top, 4),
                        "text": wins[hottest]},
            "profile": [round(s, 4) for s in scores],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }


def transcript_from_csv(path, idx=None, row_id=None):
    import csv
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("that dataset is empty")
    cols = {c.lower(): c for c in rows[0]}
    tcol = next((cols[c] for c in ("transcript", "text", "call", "conversation",
                                   "dialogue", "content", "body")
                 if c in cols), None)
    if tcol is None:
        raise SystemExit("no transcript column in that dataset")
    if row_id is not None:
        idcol = next((cols[c] for c in ("id", "call_id", "conv_id")
                      if c in cols), None)
        if idcol is None:
            raise SystemExit("that dataset has no id column")
        hit = [r for r in rows if str(r[idcol]) == str(row_id)]
        if not hit:
            raise SystemExit("no row with id %s" % row_id)
        return hit[0][tcol]
    return rows[int(idx or 0)][tcol]


def classifier_from(args):
    return Classifier(args.name, gpu=args.gpu, max_length=args.max_length,
                      window=args.window, stride=args.stride,
                      aggregate=args.aggregate, strip=args.strip_tags)


# ---------------------------------------------------------------------------
# evaluate - the whole dataset, not one call
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Score every call in a dataset with one trained checkpoint.

    Note what this is NOT: the `bert` row on the Benchmark page is k-fold
    cross-validation, where every call is predicted by a model that never saw
    it. This scores calls with one already-trained checkpoint, so any call
    that was in that checkpoint's training split is being marked by a model
    that has read it. The run says so when the dataset being scored is the
    one it was trained on - a very high number there means the model
    remembers, not that it generalises.

    The useful thing to do with it is the opposite: point a checkpoint at a
    dataset it has never seen. That is the number the write-up needs, and
    it is usually a good deal lower than the holdout one.
    """
    import eval_common as EC

    clf = classifier_from(args)
    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    print()
    print("==> score  models/%s over %d calls, on the %s"
          % (clf.name, len(rows), clf.device))
    trained_on = clf.meta.get("dataset")
    EC.say_which_experiment(trained_on, args.csv, "trained")
    long_calls = sum(1 for r in rows if r["words"] > (clf.window or 200))
    if long_calls:
        print("  %d call(s) are longer than one window and will be scored in "
              "windows, taking\n  the %s" % (long_calls, clf.aggregate))

    prog = EC.Progress(len(rows), every=max(1, len(rows) // 60))
    preds, probs, wins = [], [], []
    t0 = time.time()
    for r in rows:
        try:
            out = clf.classify(r["text"], args.threshold, strip=args.strip_tags)
            pred = 1 if out["verdict"] == "scam" else 0
            probs.append("%.4f" % out["prob_scam"])
            wins.append(out.get("windows"))
        except ValueError:
            pred = None
            probs.append("")
            wins.append("")
        preds.append(pred)
        prog.tick(r, pred)
    elapsed = time.time() - t0

    truths = [r["label"] for r in rows]
    m = EC.metrics(truths, preds)
    base = EC.baselines(truths)
    EC.report(m, base, elapsed, len(rows))
    if args.out:
        EC.write_results(args.out, clf.name, args.csv, rows, preds, m, base,
                         elapsed, {"prob_scam": probs, "windows": wins},
                         {"kind": "bert", "trained_on": trained_on,
                          "base_model": clf.meta.get("base_model"),
                          "threshold": args.threshold or clf.threshold,
                          "aggregate": clf.aggregate,
                          "device": str(clf.device),
                          "same_dataset": bool(trained_on == args.csv)})
    return m


def run_classify(args):
    text = args.text
    if not text and args.csv:
        text = transcript_from_csv(args.csv, args.idx, args.id)
    if not text:
        raise SystemExit("pass --text, or --csv with --idx/--id")
    out = classifier_from(args).classify(text, args.threshold)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_result(out)
    return out


def print_result(r):
    bar = "=" * 74
    print(bar)
    print("%s   %.1f%% scam   (models/%s)"
          % (r["verdict"].upper(), 100 * r["prob_scam"], r["model"]["name"]))
    print(bar)
    print("  aggregate  %s of %d window%s, at threshold %.2f"
          % (r["aggregate"], r["windows"], "" if r["windows"] == 1 else "s",
             r["threshold"]))
    print("  max %.3f   mean %.3f   first window %.3f"
          % (r["max"], r["mean"], r["first_window"]))
    if r["windows"] > 1 and abs(r["max"] - r["first_window"]) >= 0.2:
        print("  NOTE       the opening of this call scores %.3f and the "
              "strongest stretch\n             %.3f - reading only the first "
              "%d words would have said\n             something else."
              % (r["first_window"], r["max"], r["window_words"]))
    if r["stripped_tags"]:
        print("  NOTE       tone tags were stripped, which is not how this "
              "model was trained")
    print()
    print("  strongest stretch (window %d of %d):"
          % (r["hottest"]["index"] + 1, r["windows"]))
    print("    ...%s..." % r["hottest"]["text"][:260])
    print()
    print("  %d words, %d ms on %s"
          % (r["words"], r["elapsed_ms"], r["device"]))
    print(bar)


# ---------------------------------------------------------------------------
# serve - one JSON request per line, one JSON reply per line
# ---------------------------------------------------------------------------

def run_serve(args):
    """A long-lived classifier on stdin/stdout, so the model is loaded once.

    web_ui.py keeps one of these per selected model and stops it when the
    model changes, when Unload is pressed, or when it has sat idle. The
    protocol is deliberately dumb: a request is one line of JSON, a reply is
    one line of JSON, and every reply carries "ok".
    """
    try:
        clf = classifier_from(args)
    except BaseException as e:      # SystemExit included - report, do not exit
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        return
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "model": clf.name,
                                 "device": str(clf.device)}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                return
            thr = req.get("threshold")
            out = clf.classify(req.get("transcript", ""),
                               None if thr in (None, "") else float(thr),
                               aggregate=req.get("aggregate"),
                               strip=req.get("strip_tags"))
        except Exception as e:
            # a ValueError here is this module saying what is wrong with the
            # request, and it is shown to someone on a web page - the class
            # name in front of it is noise. Anything else keeps its type.
            out = {"ok": False,
                   "error": str(e) if isinstance(e, ValueError)
                            else "%s: %s" % (type(e).__name__, e)}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def run_models(args):
    models = list_models()
    if args.json:
        print(json.dumps(models, indent=2))
        return models
    if not models:
        print("  nothing in models/ yet")
        return models
    for m in models:
        acc = m["holdout"] and m["holdout"].get("acc")
        print("  %-24s %-22s %-34s %s"
              % (m["name"], m["base"], m["dataset"] or "",
                 ("holdout acc %.1f%%" % (100 * acc)) if acc else ""))
    return models


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="train a BERT on scam calls, then ask it whether a call "
                    "is one",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train", help="fine-tune and keep a checkpoint")
    tr.add_argument("--csv", required=True)
    tr.add_argument("--name", required=True, help="saved as models/<name>/")
    tr.add_argument("--model", default="bert-base-uncased",
                    help="HF model id to start from")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--max-length", type=int, default=256)
    tr.add_argument("--lr", type=float, default=2e-5)
    tr.add_argument("--threshold", type=float, default=0.5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--limit", type=int, default=None)
    tr.add_argument("--holdout", type=float, default=0.2,
                    help="fraction kept back to score the checkpoint (0 = none)")
    tr.add_argument("--text-col", default=None)
    tr.add_argument("--label-col", default=None)
    tr.add_argument("--cpu", action="store_true")
    tr.add_argument("--overwrite", action="store_true")
    tr.set_defaults(func=run_train, save_model=None)

    def scoring(sp):
        sp.add_argument("--name", required=True)
        sp.add_argument("--gpu", action="store_true",
                        help="score on the GPU; by default inference is on "
                             "CPU so a benchmark run keeps the VRAM")
        sp.add_argument("--max-length", type=int, default=None,
                        help="tokens per window; defaults to what the "
                             "checkpoint was trained at")
        sp.add_argument("--window", type=int, default=None,
                        help="words per window; defaults to whatever fills "
                             "max-length")
        sp.add_argument("--stride", type=int, default=None)
        sp.add_argument("--aggregate", choices=("max", "mean"), default="max",
                        help="how window scores become one number")
        sp.add_argument("--strip-tags", action="store_true",
                        help="take the [curious]/[long pause] annotations out "
                             "first - an experiment, not a default: training "
                             "read them")

    cl = sub.add_parser("classify", help="scam or not, for one transcript")
    scoring(cl)
    cl.add_argument("--text", default=None)
    cl.add_argument("--csv", default=None)
    cl.add_argument("--idx", type=int, default=None)
    cl.add_argument("--id", default=None)
    cl.add_argument("--threshold", type=float, default=None,
                    help="defaults to the checkpoint's own")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=run_classify)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    scoring(ev)
    ev.add_argument("--csv", required=True)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--threshold", type=float, default=None)
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    sv = sub.add_parser("serve", help="stay loaded, one JSON request per line")
    scoring(sv)
    sv.set_defaults(func=run_serve)

    ls = sub.add_parser("models", help="what is in models/")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=run_models)

    rm = sub.add_parser("delete", help="remove one checkpoint")
    rm.add_argument("--name", required=True)
    rm.set_defaults(func=lambda a: print(json.dumps(delete_model(a.name))))

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
