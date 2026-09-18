#!/usr/bin/env python3
"""
bert_mcq.py - train a BERT encoder on scam calls, then make it answer the
MCQ ontology.

Two halves, and it matters which is which:

  train    Fine-tunes a BERT for sequence classification on a labelled
           dataset - the same job bert_baseline.py does, but trained once on
           the whole set rather than per fold, and kept. The checkpoint lands
           in models/<name>/ with a mcq_meta.json beside it recording what it
           was trained on and how it scored on a held-out slice.

  answer   Loads one of those checkpoints and walks knowledge/mcq_ontology.json
           with it: route the call to a branch, then answer every question on
           that branch, then sum the values of the chosen options into the
           ontology's verdict.

How BERT answers a multiple-choice question it was never given labels for:
the fine-tuned encoder is used as an embedding model. The transcript is cut
into overlapping windows; every window and every option text is mean-pooled
into a vector; an option's score is its best cosine similarity against any
window. Scores are centred per question - the mean across that question's own
options is subtracted, which strips out the similarity every option shares
simply by being about the same subject - and turned into probabilities with a
softmax. When the best option does not clear --min-confidence the question
abstains to its "not stated" option, so a call that never mentions payment
does not get an answer invented for it.

That makes the MCQ answers zero-shot in the ontology's terms and supervised
only in the domain sense: what training changes is the embedding space the
options are matched in, so a model fine-tuned on bank scams answers these
questions differently from stock bert-base-uncased. The binary head is
reported alongside as prob_scam - that number, and only that number, is what
the model was directly trained to produce.

Usage:
    python scripts/bert_mcq.py train --csv datasets/zhi_english_646.csv \
        --name zhi-bert --epochs 4
    python scripts/bert_mcq.py answer --name zhi-bert --text "Hello, this is..."
    python scripts/bert_mcq.py answer --name zhi-bert --csv datasets/x.csv --idx 3
    python scripts/bert_mcq.py models --json
    python scripts/bert_mcq.py serve --name zhi-bert      # one JSON request per line

scripts/web_ui.py drives all of these from the BERT + MCQ page; nothing here
needs the web UI to be useful.
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
ONTOLOGY = PROJECT_DIR / "knowledge" / "mcq_ontology.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A saved model is addressed by name from the web UI, so the name has to be a
# single path segment and nothing else - it is joined onto models/.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")

# The option a question falls back to when nothing in the call answers it.
# Every branch in mcq_ontology.json ends its questions with one of these.
ABSTAIN_IDS = ("not_mentioned", "unclear", "none", "no_action")


def checked_name(name):
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise SystemExit(
            "model name must be letters, digits, dot, dash or underscore "
            "(got %r)" % name)
    return name


def model_dir(name):
    return MODELS_DIR / checked_name(name)


def load_ontology(path=None):
    return json.loads(Path(path or ONTOLOGY).read_text(encoding="utf-8"))


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
        meta = {}
        try:
            meta = json.loads((d / "mcq_meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass
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
    # anything at all before it is used to answer questions.
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
    (dest / "mcq_meta.json").write_text(json.dumps(meta, indent=2),
                                        encoding="utf-8")
    print()
    print("  ok trained  models/%s  (%.0fs)" % (name, elapsed))
    print("  answer questions with it:  python scripts/bert_mcq.py answer "
          "--name %s --text \"...\"" % name)
    return meta


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------

def windows_of(text, size, stride, cap=64):
    """Overlapping word windows.

    A transcript is far longer than BERT's context, and the answer to "how is
    the payment made" lives in one stretch of it - so an option is matched
    against the best window rather than against an average of the whole call,
    which would wash the one telling exchange out.
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


def softmax(xs, temperature):
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(temperature, 1e-6)) for x in xs]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def band_for(score, bands):
    """The ontology's own bands, read rather than assumed.

    They ship configured with min == max == 0, which is the "verdict is the
    sign of the sum" default the scoring block describes: below zero is
    legitimate, above it is scam, exactly zero is the call giving nothing to
    go on. Written generally, so moving a cut-off in the JSON - which is what
    RQ2 does - changes the verdict here with no code change.
    """
    for b in bands or []:
        lo, hi = b.get("min"), b.get("max")
        if lo is not None and hi is not None:
            if lo == hi:
                if score == lo:
                    return b.get("label", "uncertain")
            elif lo < score < hi:
                return b.get("label", "uncertain")
        elif hi is not None and score < hi:
            return b.get("label", "legitimate")
        elif lo is not None and score > lo:
            return b.get("label", "scam")
    return "uncertain"


def abstain_option(question):
    by_id = {o["id"]: o for o in question["options"]}
    for key in ABSTAIN_IDS:
        if key in by_id:
            return by_id[key]
    return None


class MCQAnswerer:
    """One loaded checkpoint, answering the ontology for any transcript.

    Held open by serve() so the web UI pays the model load once rather than
    once per question set. Inference is on CPU unless gpu=True: a benchmark
    run wants the whole GPU, and answering one transcript on CPU takes well
    under a second once the model is in memory.
    """

    def __init__(self, name, ontology=None, gpu=False, max_length=256,
                 window=60, stride=30, temperature=0.02, min_confidence=0.30):
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
        # One model doing both jobs: the classification head gives prob_scam,
        # and the hidden states it is asked to keep give the embedding space
        # the options are matched in.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(d), output_hidden_states=True)
        self.model.to(self.device).eval()

        self.ontology = ontology or load_ontology()
        self.max_length = max_length
        self.window = window
        self.stride = stride
        self.temperature = temperature
        self.min_confidence = min_confidence
        try:
            self.meta = json.loads((d / "mcq_meta.json").read_text(encoding="utf-8"))
        except Exception:
            self.meta = {"name": self.name}
        self._cache = {}        # option text -> vector, reused across calls

    # ---------------------------------------------------------- embeddings
    def _encode(self, texts):
        torch = self.torch
        out = []
        for i in range(0, len(texts), 16):
            enc = self.tokenizer(texts[i:i + 16], truncation=True, padding=True,
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model(**enc)
            hidden = res.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros((0, 1))

    def _option_vectors(self, keys):
        """Cached: the ontology's option texts never change between
        transcripts, and re-embedding ninety short strings per call would
        otherwise be most of the work."""
        missing = [k for k in keys if k not in self._cache]
        if missing:
            for k, v in zip(missing, self._encode(missing)):
                self._cache[k] = v
        return self.torch.stack([self._cache[k] for k in keys])

    def _prob_scam(self, text):
        """The trained head's own answer, on the transcript as training saw
        it: truncated to max_length, not windowed."""
        torch = self.torch
        enc = self.tokenizer([text], truncation=True, padding=True,
                             max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])

    # ------------------------------------------------------------- scoring
    def _pick(self, win_vecs, wins, prompt, options):
        """Score every option of one question against the transcript.

        similarity - the option's best cosine against any window
        margin     - that, minus the mean across this question's options,
                     which removes the similarity every option of a question
                     shares by being about the same subject at all
        """
        keys = [(prompt + " " + o.get("text", o["id"])).strip() for o in options]
        sims = win_vecs @ self._option_vectors(keys).T      # windows x options
        best, arg = sims.max(dim=0)
        best = [float(x) for x in best]
        arg = [int(x) for x in arg]
        mean = sum(best) / len(best)
        centred = [b - mean for b in best]
        probs = softmax(centred, self.temperature)
        scored = [{"id": o["id"], "text": o.get("text", o["id"]),
                   "value": o.get("value"),
                   "similarity": round(s, 4), "margin": round(c, 4),
                   "confidence": round(p, 4),
                   "evidence": wins[w] if wins else ""}
                  for o, s, c, p, w in zip(options, best, centred, probs, arg)]
        return scored, max(scored, key=lambda s: s["confidence"])

    def answer(self, transcript, branch="auto", cutoff=0.0):
        t0 = time.time()
        transcript = (transcript or "").strip()
        if not transcript:
            raise ValueError("there is no transcript to answer about")
        wins = windows_of(transcript, self.window, self.stride)
        win_vecs = self._encode(wins)

        # ---- route: which branch of the ontology this call belongs to
        root_opts = self.ontology["options"]
        routed, top = self._pick(win_vecs, wins,
                                 self.ontology.get("prompt", ""), root_opts)
        forced = bool(branch) and branch != "auto"
        if forced:
            top = next((o for o in routed if o["id"] == branch), None)
            if top is None:
                raise ValueError("no such branch: %s" % branch)
        chosen_branch = next(o for o in root_opts if o["id"] == top["id"])

        # ---- the questions that branch asks
        answers, total = [], 0.0
        for q in chosen_branch.get("questions", []):
            scored, pick = self._pick(win_vecs, wins, q.get("prompt", ""),
                                      q["options"])
            abstained = False
            fallback = abstain_option(q)
            if pick["confidence"] < self.min_confidence and fallback is not None:
                pick = next(s for s in scored if s["id"] == fallback["id"])
                abstained = True
            recorded = q.get("role") == "recorded"
            value = 0.0 if recorded else float(pick.get("value") or 0.0)
            total += value
            answers.append({
                "id": q["id"], "prompt": q.get("prompt", ""),
                "role": q.get("role"), "note": q.get("note"),
                "recorded": recorded,
                "chosen": pick["id"], "chosen_text": pick["text"],
                "confidence": pick["confidence"], "abstained": abstained,
                "contributes": round(value, 3),
                "evidence": pick["evidence"],
                "options": scored,
            })

        score = round(total, 3)
        verdict = band_for(round(score - cutoff, 6), self.ontology.get("bands"))
        return {
            "ok": True,
            "model": {"name": self.name, "base": self.meta.get("base_model"),
                      "dataset": self.meta.get("dataset"),
                      "trained_at": self.meta.get("trained_at"),
                      "holdout": self.meta.get("holdout")},
            "device": str(self.device),
            "windows": len(wins),
            "classifier": {"prob_scam": round(self._prob_scam(transcript), 4),
                           "threshold": self.meta.get("threshold", 0.5)},
            "route": {"prompt": self.ontology.get("prompt", ""),
                      "forced": forced,
                      "chosen": chosen_branch["id"],
                      "chosen_text": chosen_branch.get("text", ""),
                      "legit_contrast": chosen_branch.get("legit_contrast"),
                      "confidence": top["confidence"],
                      "options": routed},
            "questions": answers,
            "score": {"sum": score, "cutoff": cutoff, "verdict": verdict,
                      "method": (self.ontology.get("scoring") or {}).get("formula", "")},
            "elapsed_ms": int((time.time() - t0) * 1000),
        }


def transcript_from_csv(path, idx=None, row_id=None):
    import pandas as pd
    import bert_baseline as bb
    df = pd.read_csv(path)
    tcol = bb.pick_column(df, bb.TEXT_CANDIDATES, "text")
    if row_id is not None:
        idcol = next((c for c in df.columns
                      if c.lower() in ("id", "call_id", "conv_id")), None)
        if idcol is None:
            raise SystemExit("that dataset has no id column")
        hit = df[df[idcol].astype(str) == str(row_id)]
        if hit.empty:
            raise SystemExit("no row with id %s" % row_id)
        return str(hit.iloc[0][tcol])
    return str(df.iloc[int(idx or 0)][tcol])


def run_answer(args):
    text = args.text
    if not text and args.csv:
        text = transcript_from_csv(args.csv, args.idx, args.id)
    if not text:
        raise SystemExit("pass --text, or --csv with --idx/--id")
    ans = MCQAnswerer(args.name, gpu=args.gpu, max_length=args.max_length,
                      window=args.window, stride=args.stride,
                      temperature=args.temperature,
                      min_confidence=args.min_confidence).answer(
        text, args.branch, args.cutoff)
    if args.json:
        print(json.dumps(ans, indent=2))
    else:
        print_answer(ans)
    return ans


def print_answer(a):
    bar = "=" * 74
    print(bar)
    print("MCQ ONTOLOGY, answered by models/%s" % a["model"]["name"])
    print(bar)
    print("  route      %s  (%.0f%% confident)"
          % (a["route"]["chosen_text"], 100 * a["route"]["confidence"]))
    print()
    for q in a["questions"]:
        mark = "  abstained" if q["abstained"] else ""
        val = "recorded" if q["recorded"] else "%+.1f" % q["contributes"]
        print("  %s" % q["prompt"])
        print("      %-50s %8s  %3.0f%%%s"
              % (q["chosen_text"][:50], val, 100 * q["confidence"], mark))
    print()
    print("  score      %+.2f   verdict: %s"
          % (a["score"]["sum"], a["score"]["verdict"].upper()))
    print("  prob_scam  %.3f   (the trained classification head, for contrast)"
          % a["classifier"]["prob_scam"])
    print("  %d windows, %d ms on %s"
          % (a["windows"], a["elapsed_ms"], a["device"]))
    print(bar)


# ---------------------------------------------------------------------------
# serve - one JSON request per line, one JSON reply per line
# ---------------------------------------------------------------------------

def run_serve(args):
    """A long-lived answerer on stdin/stdout, so the model is loaded once.

    web_ui.py keeps one of these per selected model and stops it when the
    model changes, when Unload is pressed, or when it has sat idle. The
    protocol is deliberately dumb: a request is one line of JSON, a reply is
    one line of JSON, and every reply carries "ok".
    """
    try:
        ans = MCQAnswerer(args.name, gpu=args.gpu, max_length=args.max_length,
                          window=args.window, stride=args.stride,
                          temperature=args.temperature,
                          min_confidence=args.min_confidence)
    except BaseException as e:      # SystemExit included - report, do not exit
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        return
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "model": ans.name,
                                 "device": str(ans.device)}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                return
            out = ans.answer(req.get("transcript", ""),
                             req.get("branch", "auto"),
                             float(req.get("cutoff", 0.0)))
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
        description="train a BERT on scam calls, then answer the MCQ ontology "
                    "with it",
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

    def answering(sp):
        sp.add_argument("--name", required=True)
        sp.add_argument("--gpu", action="store_true",
                        help="answer on the GPU; by default inference is on "
                             "CPU so a benchmark run keeps the VRAM")
        sp.add_argument("--max-length", type=int, default=256)
        sp.add_argument("--window", type=int, default=60,
                        help="words per transcript window")
        sp.add_argument("--stride", type=int, default=30)
        sp.add_argument("--temperature", type=float, default=0.02)
        sp.add_argument("--min-confidence", type=float, default=0.30,
                        help="below this a question abstains to 'not stated'")

    an = sub.add_parser("answer", help="answer the ontology for one transcript")
    answering(an)
    an.add_argument("--text", default=None)
    an.add_argument("--csv", default=None)
    an.add_argument("--idx", type=int, default=None)
    an.add_argument("--id", default=None)
    an.add_argument("--branch", default="auto")
    an.add_argument("--cutoff", type=float, default=0.0)
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=run_answer)

    sv = sub.add_parser("serve", help="stay loaded, one JSON request per line")
    answering(sv)
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
