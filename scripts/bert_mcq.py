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

How a question with no labels gets answered: by similarity, in an embedding
space. The transcript is cut into overlapping windows; every window and every
option text is mean-pooled into a vector; an option's score is its best cosine
similarity against any window. The best option is taken when it is far enough
clear of the runner-up, and otherwise the question abstains to its "not
stated" option, so a call that never mentions payment does not get an answer
invented for it.

Four things about that matching are load-bearing, and each was a source of
noise-driven answers before it was fixed:

  the encoder     Options are matched with a sentence-similarity encoder
                  (--encoder, all-MiniLM-L6-v2 by default), not with the
                  fine-tuned checkpoint's own hidden states. Fine-tuning on a
                  binary scam/legitimate objective never asks the token states
                  to tell "a courier" from "a customs agency", which is the
                  distinction every MCQ here turns on, and the states it does
                  leave behind are anisotropic: any two short phrases sit at
                  cosine .9 of each other and the differences that remain are
                  mostly noise. --encoder self restores the old behaviour to
                  compare against.

  the key         An option is embedded as its own text. Prefixing the
                  question prompt, which is identical across that question's
                  options, made the strings being compared ~90% the same
                  characters and buried the part that differs.

  centring        The mean vector of the whole option corpus is subtracted
                  from both sides before the cosine (--no-center to disable).
                  That direction is "generic short English phrase" and carries
                  no information; removing it is what turns a 0.02 spread
                  between options into a readable one.

  the gate        A question is only answered when the best option clears the
                  runner-up by --min-margin in raw cosine, and clears
                  --min-confidence after the softmax. Margin is the honest
                  test: a softmax over near-identical scores reports high
                  confidence for what is a coin flip.

Transcripts are repaired before any of that (--raw to disable): ASR output
arrives lower-cased and unpunctuated, with apostrophes gone ("i m", "don t"),
fillers left in and entities blanked to [Number]/[Title], none of which is
what a sentence encoder was trained to read.

That makes the MCQ answers zero-shot in the ontology's terms. The binary head
is reported alongside as prob_scam - that number, and only that number, is
what the model was directly trained to produce.

Usage:
    python scripts/bert_mcq.py train --csv datasets/zhi_english_646.csv \
        --name zhi-bert --epochs 4
    python scripts/bert_mcq.py answer --name zhi-bert --text "Hello, this is..."
    python scripts/bert_mcq.py answer --name zhi-bert --csv datasets/x.csv --idx 3
    python scripts/bert_mcq.py diagnose --name zhi-bert --csv datasets/x.csv --idx 3
    python scripts/bert_mcq.py models --json
    python scripts/bert_mcq.py serve --name zhi-bert      # one JSON request per line

`diagnose` is the command to run first if the answers look arbitrary: it
prints the raw similarities behind every pick, and --control answers a
word-shuffled copy of the same call. Shuffling destroys word order and every
local phrase; answers that survive it were never reading the call.

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

# Of those, the ones that are non-answers rather than answers. "Not stated"
# describes the transcript, not the call, so no window can resemble it and
# letting it compete on similarity only adds a way for noise to win. It is
# reached through the gate instead. The other two ("Nothing, or information
# only", "No consequence described") are real answers carrying real values, so
# they compete like anything else.
NON_ANSWER_IDS = ("not_mentioned", "unclear")

# Matched in a space built for sentence similarity rather than in the
# fine-tuned checkpoint's own. Already this project's embedding model -
# build_index.py and webrag_system.py index the policy KB with it - so it is
# in the venv and usually in the HF cache already.
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# --- repairing ASR transcripts -------------------------------------------
# The calls arrive like this:
#
#   "ah hi uh how s it going uh i had a transaction for the john lewis for
#    [Number] pounds and i ll be honest we don t even shop at john lewis"
#
# Lower-cased, unpunctuated, apostrophes dropped by the recogniser, fillers
# left in, entities blanked. "i m" and "don t" are two tokens of nothing much
# to an encoder that has only ever read "I'm" and "don't", so they are put
# back before anything is embedded.
PLACEHOLDER_RE = re.compile(r"\[([A-Za-z][A-Za-z _-]{0,18})\]")

# Square brackets in these datasets hold two unrelated things. One is a
# redacted entity, and what kind it was is worth keeping: "[CARD]" means a
# card number was read out, which is most of the answer to "what is the caller
# asking for". The other is the annotator's note on tone - [curious], [calm],
# [sighing] - which nobody said out loud. Those are far the commoner of the
# two (77k [curious] and 73k [calm] across datasets/) and they are dropped:
# left in, every window is padded with mood words that drag its vector towards
# whichever option happens to sound emotional.
#
# They are worth keeping away from the classifier too, and for a sharper
# reason: in scamai_full_1000.csv [satisfied] is on 54.8% of legitimate calls
# and 31.0% of scams, so a model reading them can score well off the labelling
# convention instead of off the call. That is bert_baseline.py's business
# rather than this file's, but the same call goes through both.
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
CLITIC_RE = re.compile(
    r"\b(i|you|we|they|he|she|it|that|there|here|what|who|where|how|let|"
    r"don|doesn|didn|isn|aren|wasn|weren|won|can|couldn|shouldn|wouldn|"
    r"haven|hasn|hadn|ain|could|would|should|must|might|somebody|someone|"
    r"nobody|everybody|one|company|bank)\s+(m|s|re|ve|ll|d|t)\b", re.I)
FILLER_RE = re.compile(r"\b(u+h+|u+m+|erm|er|a+h+|eh|mm+|hm+|mhm)\b", re.I)


def _placeholder(match):
    """A redacted entity becomes the plain words for what it was; anything
    else in brackets - tone notes, [noise], [long pause] - becomes nothing."""
    key = match.group(1).lower().replace("_", " ").strip()
    return " %s " % ENTITY_WORDS[key] if key in ENTITY_WORDS else " "


def normalise_transcript(text):
    """Put back what the recogniser dropped, take out what it added.

    Nothing is invented: every word out is either a word that was in or the
    plain name of an entity the dataset had redacted.
    """
    t = PLACEHOLDER_RE.sub(_placeholder, text or "")
    t = CLITIC_RE.sub(lambda m: "%s'%s" % (m.group(1), m.group(2)), t)
    t = FILLER_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


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


def option_key(option):
    """What an option is embedded as: its own text, and nothing else.

    Not the question prompt plus the text, which is what this used to be. The
    prompt is identical across that question's options, so including it made
    every string being compared about 90% the same characters and left the
    difference between "A courier or delivery company" and "A customs or
    border agency" - the entire content of the question - to fight its way out
    from under a shared prefix.
    """
    return (option.get("text") or option["id"]).strip()


class MCQAnswerer:
    """One loaded checkpoint, answering the ontology for any transcript.

    Held open by serve() so the web UI pays the model load once rather than
    once per question set. Inference is on CPU unless gpu=True: a benchmark
    run wants the whole GPU, and answering one transcript on CPU takes well
    under a second once the model is in memory.
    """

    def __init__(self, name, ontology=None, gpu=False, max_length=256,
                 window=45, stride=15, temperature=0.10, min_confidence=0.30,
                 min_margin=0.04, encoder=DEFAULT_ENCODER, center=True,
                 normalise=True):
        import torch
        from transformers import (AutoModel, AutoTokenizer,
                                  AutoModelForSequenceClassification)

        self.name = checked_name(name)
        d = model_dir(self.name)
        if not (d / "config.json").exists():
            raise SystemExit("no trained model at models/%s - train one first"
                             % self.name)
        self.torch = torch
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available()
                                   else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(d))

        # Two jobs, and they are not the same job. prob_scam is the trained
        # classification head, which only this checkpoint can give. The option
        # match wants a space where "a courier" and "a customs agency" sit
        # apart, and a binary scam/legitimate objective never asked for one.
        self.encoder_id = (encoder or "self").strip()
        own = self.encoder_id in ("self", "own", "checkpoint")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(d), output_hidden_states=own).to(self.device).eval()
        if own:
            self.encoder_id = "self"
            self.embed_tokenizer, self.embed_model = self.tokenizer, self.model
        else:
            try:
                self.embed_tokenizer = AutoTokenizer.from_pretrained(self.encoder_id)
                self.embed_model = AutoModel.from_pretrained(
                    self.encoder_id).to(self.device).eval()
            except Exception as e:
                raise SystemExit(
                    "could not load the matching encoder %r (%s).\n"
                    "It downloads from Hugging Face the first time and is "
                    "cached after that, so this is usually no network. Pass "
                    "--encoder self to match with the checkpoint's own hidden "
                    "states instead - that is what this did before, and what "
                    "made the answers arbitrary." % (self.encoder_id, e))

        self.ontology = ontology or load_ontology()
        self.max_length = max_length
        self.window = window
        self.stride = stride
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.center = center
        self.normalise = normalise
        try:
            self.meta = json.loads((d / "mcq_meta.json").read_text(encoding="utf-8"))
        except Exception:
            self.meta = {"name": self.name}
        self._cache = {}        # option text -> vector, reused across calls
        self._centre = None     # the ontology's own mean direction, computed once

    # ---------------------------------------------------------- embeddings
    def _encode(self, texts):
        """Mean-pooled, L2-normalised, uncentred."""
        torch = self.torch
        out = []
        for i in range(0, len(texts), 16):
            enc = self.embed_tokenizer(texts[i:i + 16], truncation=True,
                                       padding=True, max_length=self.max_length,
                                       return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.embed_model(**enc)
            hidden = (res.hidden_states[-1] if self.encoder_id == "self"
                      else res.last_hidden_state)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros((0, 1))

    def _vectors(self, keys):
        """Cached: the ontology's option texts never change between
        transcripts, and re-embedding a hundred and fifty short strings per
        call would otherwise be most of the work."""
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            for k, v in zip(missing, self._encode(missing)):
                self._cache[k] = v
        return self.torch.stack([self._cache[k] for k in keys])

    def _centre_vector(self):
        """The mean direction of the ontology's own option texts.

        Sentence vectors out of any BERT are anisotropic: they occupy a narrow
        cone, so two unrelated short phrases still score .85 against each
        other and the part that carries the meaning is a thin margin on top of
        a large shared component. That shared component is this vector.
        Subtracting it from both sides before the cosine is what gives the
        options room to differ; without it the spread across a question's
        options is a couple of hundredths and a softmax over that is reading
        noise. Computed once from a fixed corpus - the ontology, not the call
        - so the same transcript always scores the same.
        """
        if self._centre is None:
            corpus = []
            for branch in self.ontology["options"]:
                corpus.append(option_key(branch))
                for q in branch.get("questions", []):
                    corpus += [option_key(o) for o in q["options"]]
            self._centre = self._vectors(list(dict.fromkeys(corpus))).mean(
                0, keepdim=True)
        return self._centre

    def _centred(self, vecs):
        if not self.center or vecs.shape[0] == 0:
            return vecs
        return self.torch.nn.functional.normalize(
            vecs - self._centre_vector(), dim=-1)

    def _prob_scam(self, text):
        """The trained head's own answer, on the transcript as training saw
        it: the raw text, truncated to max_length, not windowed and not
        repaired. Training read the tone tags and the missing apostrophes, so
        this has to as well, or the head is being asked about text of a kind
        it never saw."""
        torch = self.torch
        enc = self.tokenizer([text], truncation=True, padding=True,
                             max_length=self.max_length,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])

    # ------------------------------------------------------------- scoring
    def _pick(self, win_vecs, wins, options):
        """Score one question's options against the transcript.

        similarity - the option's best cosine against any window, centred
        centred    - that, minus the mean across this question's options
        margin     - how far the winner is clear of the runner-up. This is the
                     number that says whether there was anything to choose
                     between them; the softmax confidence beside it cannot,
                     because a softmax over four near-identical scores still
                     has to hand out its probability to somebody.
        """
        answerable = [o for o in options if o["id"] not in NON_ANSWER_IDS]
        if not answerable:
            answerable = list(options)
        keys = [option_key(o) for o in answerable]
        sims = win_vecs @ self._centred(self._vectors(keys)).T  # windows x options
        best, arg = sims.max(dim=0)
        best = [float(x) for x in best]
        arg = [int(x) for x in arg]
        ranked = sorted(best, reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        mean = sum(best) / len(best)
        centred = [b - mean for b in best]
        probs = softmax(centred, self.temperature)
        won = {o["id"]: {"id": o["id"], "text": option_key(o),
                         "value": o.get("value"),
                         "similarity": round(s, 4), "centred": round(c, 4),
                         "confidence": round(p, 4), "scored": True,
                         "evidence": wins[w] if wins else ""}
               for o, s, c, p, w in zip(answerable, best, centred, probs, arg)}
        # the non-answers stay in the list for the table, out of the contest
        scored = [won.get(o["id"]) or
                  {"id": o["id"], "text": option_key(o), "value": o.get("value"),
                   "similarity": 0.0, "centred": 0.0, "confidence": 0.0,
                   "scored": False, "evidence": ""}
                  for o in options]
        # by similarity, not by the rounded confidence, so the winner is the
        # same option the margin was measured from even when two confidences
        # round to the same four places
        top = max((s for s in scored if s["scored"]),
                  key=lambda s: s["similarity"])
        return scored, top, round(margin, 4)

    def _abstain_reason(self, pick, margin):
        """Why this question should not be answered, or None to answer it."""
        if pick["similarity"] <= 0:
            return "nothing in the call resembled any of the options"
        if margin < self.min_margin:
            return ("the best two options were %.3f apart, under the %.3f "
                    "needed to tell them apart" % (margin, self.min_margin))
        if pick["confidence"] < self.min_confidence:
            return ("the best option reached %.0f%%, under the %.0f%% needed"
                    % (100 * pick["confidence"], 100 * self.min_confidence))
        return None

    def answer(self, transcript, branch="auto", cutoff=0.0):
        t0 = time.time()
        raw = (transcript or "").strip()
        if not raw:
            raise ValueError("there is no transcript to answer about")
        text = normalise_transcript(raw) if self.normalise else raw
        if not text:
            raise ValueError("there is nothing left of that transcript once "
                             "the tone tags and fillers are taken out")
        wins = windows_of(text, self.window, self.stride)
        win_vecs = self._centred(self._encode(wins))

        # ---- route: which branch of the ontology this call belongs to
        root_opts = self.ontology["options"]
        routed, top, route_margin = self._pick(win_vecs, wins, root_opts)
        forced = bool(branch) and branch != "auto"
        if forced:
            top = next((o for o in routed if o["id"] == branch), None)
            if top is None:
                raise ValueError("no such branch: %s" % branch)
        chosen_branch = next(o for o in root_opts if o["id"] == top["id"])

        # ---- the questions that branch asks
        answers, total = [], 0.0
        for q in chosen_branch.get("questions", []):
            scored, pick, margin = self._pick(win_vecs, wins, q["options"])
            fallback = abstain_option(q)
            why = self._abstain_reason(pick, margin) if fallback else None
            abstained = why is not None
            best_text = pick["text"]
            if abstained:
                pick = next(s for s in scored if s["id"] == fallback["id"])
            recorded = q.get("role") == "recorded"
            # An abstained question contributes nothing. Most fall back to a
            # "Not stated" worth 0.0 anyway; the three questions that would
            # otherwise fall back to a -0.5 option are why this is explicit.
            # Not knowing whether the caller asked for anything is not
            # evidence that they asked for nothing.
            value = 0.0 if (recorded or abstained) else float(pick.get("value") or 0.0)
            total += value
            answers.append({
                "id": q["id"], "prompt": q.get("prompt", ""),
                "role": q.get("role"), "note": q.get("note"),
                "recorded": recorded,
                "chosen": pick["id"], "chosen_text": pick["text"],
                "confidence": pick["confidence"], "abstained": abstained,
                "why_abstained": why, "best_text": best_text,
                "margin": margin,
                "contributes": round(value, 3),
                "evidence": pick["evidence"],
                "options": scored,
            })

        score = round(total, 3)
        verdict = band_for(round(score - cutoff, 6), self.ontology.get("bands"))
        answered = sum(1 for a in answers if not a["abstained"])
        return {
            "ok": True,
            "model": {"name": self.name, "base": self.meta.get("base_model"),
                      "dataset": self.meta.get("dataset"),
                      "trained_at": self.meta.get("trained_at"),
                      "holdout": self.meta.get("holdout")},
            "device": str(self.device),
            "windows": len(wins),
            "matching": {"encoder": self.encoder_id, "centred": self.center,
                         "normalised": self.normalise,
                         "temperature": self.temperature,
                         "min_margin": self.min_margin,
                         "min_confidence": self.min_confidence,
                         "answered": answered, "asked": len(answers)},
            "classifier": {"prob_scam": round(self._prob_scam(raw), 4),
                           "threshold": self.meta.get("threshold", 0.5)},
            "route": {"prompt": self.ontology.get("prompt", ""),
                      "forced": forced,
                      "chosen": chosen_branch["id"],
                      "chosen_text": chosen_branch.get("text", ""),
                      "legit_contrast": chosen_branch.get("legit_contrast"),
                      "confidence": top["confidence"],
                      "margin": route_margin,
                      "uncertain": (not forced) and route_margin < self.min_margin,
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


def answerer_from(args):
    return MCQAnswerer(args.name, gpu=args.gpu, max_length=args.max_length,
                       window=args.window, stride=args.stride,
                       temperature=args.temperature,
                       min_confidence=args.min_confidence,
                       min_margin=args.min_margin, encoder=args.encoder,
                       center=not args.no_center, normalise=not args.raw)


def transcript_from(args):
    text = args.text
    if not text and args.csv:
        text = transcript_from_csv(args.csv, args.idx, args.id)
    if not text:
        raise SystemExit("pass --text, or --csv with --idx/--id")
    return text


def run_answer(args):
    ans = answerer_from(args).answer(transcript_from(args), args.branch,
                                     args.cutoff)
    if args.json:
        print(json.dumps(ans, indent=2))
    else:
        print_answer(ans)
    return ans


def print_answer(a):
    bar = "=" * 74
    m = a["matching"]
    print(bar)
    print("MCQ ONTOLOGY, answered by models/%s" % a["model"]["name"])
    print(bar)
    print("  route      %s  (%.0f%%, margin %.3f)%s"
          % (a["route"]["chosen_text"], 100 * a["route"]["confidence"],
             a["route"]["margin"],
             "  TOO CLOSE TO CALL" if a["route"].get("uncertain") else ""))
    print()
    for q in a["questions"]:
        val = "recorded" if q["recorded"] else "%+.1f" % q["contributes"]
        print("  %s" % q["prompt"])
        print("      %-50s %8s  %3.0f%%  margin %.3f"
              % (q["chosen_text"][:50], val, 100 * q["confidence"], q["margin"]))
        if q["abstained"]:
            print("      abstained: %s" % q["why_abstained"])
            print("      the best option was %r" % q["best_text"][:60])
    print()
    print("  score      %+.2f   verdict: %s"
          % (a["score"]["sum"], a["score"]["verdict"].upper()))
    print("  prob_scam  %.3f   (the trained classification head, for contrast)"
          % a["classifier"]["prob_scam"])
    print("  answered   %d of %d questions; the rest had nothing to go on"
          % (m["answered"], m["asked"]))
    print("  matched in %s%s, %d windows, %d ms on %s"
          % (m["encoder"], ", centred" if m["centred"] else ", uncentred",
             a["windows"], a["elapsed_ms"], a["device"]))
    print(bar)


# ---------------------------------------------------------------------------
# diagnose - is there any signal in the match at all?
# ---------------------------------------------------------------------------

def run_diagnose(args):
    """The numbers behind every pick, and a control that can fail.

    Read the spread column first: if a question's options all score within a
    few thousandths of each other, whichever one won it won on noise, and no
    threshold further down can undo that. A healthy question has its winner
    clear of the field by a margin you can see without squinting.

    --control answers a word-shuffled copy of the same call. Shuffling keeps
    every word and destroys every phrase, so an answer that survives it was
    never reading the call, only reading which words were in it. Low agreement
    is the good result here.
    """
    import random
    ans = answerer_from(args)
    text = transcript_from(args)
    a = ans.answer(text, args.branch, args.cutoff)
    m = a["matching"]

    bar = "=" * 78
    print(bar)
    print("DIAGNOSE  models/%s" % a["model"]["name"])
    print("  matched in  %s" % m["encoder"])
    print("  centring    %s        transcript repair  %s"
          % ("on" if m["centred"] else "OFF", "on" if m["normalised"] else "OFF"))
    print("  gates       margin >= %.3f, confidence >= %.2f, temperature %.3f"
          % (m["min_margin"], m["min_confidence"], m["temperature"]))
    print(bar)

    r = a["route"]
    print("\nROUTE  %s   margin %.3f  %s"
          % (r["chosen_text"], r["margin"],
             "TOO CLOSE TO CALL" if r.get("uncertain") else "clear"))
    for o in sorted(r["options"], key=lambda o: -o["similarity"])[:4]:
        print("    %-46s cos %+.4f  %3.0f%%"
              % (o["text"][:46], o["similarity"], 100 * o["confidence"]))

    print("\nQUESTIONS")
    for q in a["questions"]:
        live = [o for o in q["options"] if o.get("scored")]
        sims = [o["similarity"] for o in live]
        spread = (max(sims) - min(sims)) if sims else 0.0
        flag = "ABSTAIN" if q["abstained"] else ("weak" if q["margin"] < 2 *
                                                 m["min_margin"] else "ok")
        print("\n  %s" % q["prompt"])
        print("    margin %.3f   spread %.3f   %s"
              % (q["margin"], spread, flag))
        for o in sorted(live, key=lambda o: -o["similarity"])[:3]:
            mark = "<-" if o["id"] == q["chosen"] else "  "
            print("    %s %-42s cos %+.4f  %3.0f%%"
                  % (mark, o["text"][:42], o["similarity"], 100 * o["confidence"]))
        if q["abstained"]:
            print("       abstained: %s" % q["why_abstained"])

    margins = [q["margin"] for q in a["questions"]]
    print("\n" + bar)
    print("  answered %d of %d   mean margin %.3f   score %+.2f   verdict %s"
          % (m["answered"], m["asked"],
             sum(margins) / max(len(margins), 1), a["score"]["sum"],
             a["score"]["verdict"].upper()))
    print("  prob_scam %.3f" % a["classifier"]["prob_scam"])

    if args.control:
        words = (normalise_transcript(text) if ans.normalise else text).split()
        random.Random(args.seed).shuffle(words)
        # forced to the same branch, so the two runs answer the same questions
        b = ans.answer(" ".join(words), a["route"]["chosen"], args.cutoff)
        pairs = list(zip(a["questions"], b["questions"]))
        same = sum(1 for x, y in pairs if x["chosen"] == y["chosen"])
        both = sum(1 for x, y in pairs
                   if not x["abstained"] and not y["abstained"]
                   and x["chosen"] == y["chosen"])
        answered_ctl = sum(1 for y in b["questions"] if not y["abstained"])
        ctl_route = max(b["route"]["options"], key=lambda o: o["confidence"])
        print(bar)
        print("  CONTROL, the same call with its words shuffled")
        print("    route          %s" % ctl_route["text"][:52])
        print("    answered       %d of %d (real call: %d)"
              % (answered_ctl, m["asked"], m["answered"]))
        print("    same answer    %d of %d questions, %d of them answered "
              "in both runs" % (same, len(pairs), both))
        print("    Word order carries the meaning, so a high count here means "
              "the\n    match is running on vocabulary alone and the answers "
              "are not\n    reading the call.")
    print(bar)
    return a


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
        ans = answerer_from(args)
    except BaseException as e:      # SystemExit included - report, do not exit
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        return
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "model": ans.name,
                                 "device": str(ans.device),
                                 "encoder": ans.encoder_id}) + "\n")
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
        sp.add_argument("--window", type=int, default=45,
                        help="words per transcript window")
        sp.add_argument("--stride", type=int, default=15)
        sp.add_argument("--temperature", type=float, default=0.10)
        sp.add_argument("--min-confidence", type=float, default=0.30,
                        help="below this a question abstains to 'not stated'")
        sp.add_argument("--min-margin", type=float, default=0.04,
                        help="how far the best option must be clear of the "
                             "runner-up, in raw cosine, before the question "
                             "is answered at all")
        sp.add_argument("--encoder", default=DEFAULT_ENCODER,
                        help="the model the options are matched in. 'self' "
                             "uses the fine-tuned checkpoint's own hidden "
                             "states, which is what this did before")
        sp.add_argument("--no-center", action="store_true",
                        help="skip subtracting the option corpus mean")
        sp.add_argument("--raw", action="store_true",
                        help="match against the transcript as it came, tone "
                             "tags and dropped apostrophes and all")

    def transcripts(sp):
        sp.add_argument("--text", default=None)
        sp.add_argument("--csv", default=None)
        sp.add_argument("--idx", type=int, default=None)
        sp.add_argument("--id", default=None)

    an = sub.add_parser("answer", help="answer the ontology for one transcript")
    answering(an)
    transcripts(an)
    an.add_argument("--branch", default="auto")
    an.add_argument("--cutoff", type=float, default=0.0)
    an.add_argument("--json", action="store_true")
    an.set_defaults(func=run_answer)

    dg = sub.add_parser("diagnose", help="show the numbers behind every pick")
    answering(dg)
    transcripts(dg)
    dg.add_argument("--branch", default="auto")
    dg.add_argument("--cutoff", type=float, default=0.0)
    dg.add_argument("--control", action="store_true",
                    help="also answer a word-shuffled copy of the call and "
                         "report how many answers survive it")
    dg.add_argument("--seed", type=int, default=42)
    dg.set_defaults(func=run_diagnose)

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
