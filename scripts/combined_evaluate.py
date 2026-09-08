#!/usr/bin/env python3
"""
combined_evaluate.py

Run all SEVEN systems on a scam/non-scam transcript dataset and report
accuracy / precision / recall / F1 for each.

Systems (escalation ladder):
    1. length-only     trivial word-count threshold
    2. bag-of-words    TF-IDF + LogisticRegression, 5-fold CV
    3. LLM-only        the model decides alone, NO retrieval  (the control)
    4. Singh           policy-compliance vs bank_policies collection
    5. Web-RAG         this project's system, KB-only (use_web=False)
    6. Qwen-KB         the LLM generalises a TRAINING SPLIT of this dataset
                      into scam patterns, indexes them as a knowledge base,
                      and judges held-out calls against what it retrieves
                      from it. Stratified k-fold like BERT and bag-of-words,
                      so every call is scored exactly once while held out.
    7. Hybrid          Web-RAG and Qwen-KB over ONE knowledge base: the
                      web-harvested patterns and the patterns learned from the
                      training split go into the same collection, and Web-RAG's
                      pipeline (relevance gate, graded confidence, threshold)
                      runs over the mix. Same folds and the same learned
                      patterns as system 6, so the two differ only in whether
                      the web KB is present.

CHANGES IN THIS VERSION
-----------------------
1. BAG-OF-WORDS LEAK FIXED. The TF-IDF vectoriser used to be fitted on the
   whole dataset before cross-validation, so vocabulary selection and IDF
   weights had seen the held-out fold. It is now inside a Pipeline, refitted
   from scratch on each training fold. Splits are also explicitly stratified
   and seeded.

2. VERDICT PARSING FIXED. The old rule was:
       pred = "Fraud" if <marker or fraud-and-not-normal> else "Normal"
   Any answer the regex could not read silently became "Normal", i.e. a scam
   call recorded as safe. Combined with max_tokens=60, a verbose model
   (qwen2.5:14b) could still be reasoning when it ran out of tokens, never
   reach its verdict, and be scored as a false negative. That is the most
   likely cause of Singh's recall collapsing from ~0.99 on llama3.1:8b to
   0.757 on qwen2.5:14b.
   Now: a shared parser tries several formats, retries once with a
   one-word-answer prompt if it fails, and every unreadable response is
   COUNTED AND REPORTED rather than quietly becoming "Normal".

3. TOKEN BUDGET RAISED. max_tokens default 60 -> 300 for the LLM systems.
   NOTE: this changes results relative to earlier runs. Re-run any baseline
   you intend to compare against.

4. EVERY LLM SYSTEM NOW RECORDS WHY. Each verdict prompt asks for a
   "Reason:" line under the "Answer:" line, and the per-call CSV gains a
   <system>_why column beside each <system> column. Web-RAG and the hybrid
   already had a reason inside detect() - it just never left the function -
   so theirs costs no extra tokens and comes with the confidence score.
   NOTE: asking for a reason changes the llm_only, singh and qwen_kb prompts,
   so their numbers are not directly comparable with runs made before this.
   Re-run any baseline you intend to compare against.

5. QWEN-KB MADE COMPARABLE, AND MADE TO RUN AT ALL. It used to take one
   80/20 split, so it was scored on a fifth of the calls while every other
   system was scored on all of them, and its row was not comparable with
   theirs. It is now stratified k-fold with pooled out-of-fold predictions,
   the same protocol BERT and bag-of-words already used.
   It also used to put forty whole transcripts in one KB-building prompt.
   Ollama truncates an over-long prompt from the FRONT, so the instructions
   and the JSON schema were the first thing dropped, no JSON came back, and
   the step died on "Qwen returned no usable training patterns" - taking the
   four baselines that had already finished down with it. The KB is now built
   in small batches with an explicit num_ctx, the learned patterns are indexed
   and retrieved per call rather than pasted in whole, and a failure here is
   reported and skipped instead of ending the run.

Usage:
    python scripts/combined_evaluate.py --csv datasets/zhi_scam_vs_legit_794.csv
    python scripts/combined_evaluate.py --csv datasets/... --limit 40
    python scripts/combined_evaluate.py --csv datasets/... --debug
    python scripts/combined_evaluate.py --csv datasets/... --bow-features
    python scripts/combined_evaluate.py --csv datasets/... --trivial-only
    python scripts/combined_evaluate.py --csv datasets/... --skip singh
    python scripts/combined_evaluate.py --csv datasets/... --skip qwen_kb
    python scripts/combined_evaluate.py --csv datasets/... --skip length,bow,llm_only,singh
    python scripts/combined_evaluate.py --csv one_row.csv \n        --skip length,bow --qwen-train-csv datasets/zhi_english_646.csv
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")
SEED = 42


# ------------------------------------------------------------------ loading
def load_combined(csv_path, limit=None):
    """Read a dataset CSV -> list of (text, true_label).

    Accepts label values scam/nonscam/fraud/normal/legit and maps to
    'Fraud'/'Normal' to match the other eval scripts' convention.
    """
    p = Path(csv_path)
    if not p.exists():
        sys.exit("ERROR: not found: %s" % csv_path)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        sys.exit("ERROR: empty CSV: %s" % csv_path)
    if "label" not in rows[0] or "text" not in rows[0]:
        sys.exit("ERROR: CSV needs 'label' and 'text' columns. Found: %s"
                 % list(rows[0].keys()))

    scam_words = {"scam", "fraud", "fraudulent", "1", "true", "yes"}
    data = []
    for r in rows:
        lab_raw = (r["label"] or "").strip().lower()
        lab = "Fraud" if lab_raw in scam_words else "Normal"
        text = (r["text"] or "").strip()
        if text:
            data.append((text, lab))

    if limit:
        # balanced sample
        scam = [d for d in data if d[1] == "Fraud"]
        norm = [d for d in data if d[1] == "Normal"]
        k = limit // 2
        random.seed(SEED)
        scam = random.sample(scam, min(k, len(scam)))
        norm = random.sample(norm, min(k, len(norm)))
        data = scam + norm
    random.seed(SEED)
    random.shuffle(data)
    return data


# ------------------------------------------------------------------ metrics
def metrics(rows):
    tp = fp = fn = tn = 0
    for pred, true in rows:
        if pred == "Fraud" and true == "Fraud":     tp += 1
        elif pred == "Fraud" and true == "Normal":  fp += 1
        elif pred == "Normal" and true == "Fraud":  fn += 1
        elif pred == "Normal" and true == "Normal": tn += 1
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


# ------------------------------------------------------- verdict parsing
FRAUD_WORDS = r"fraud|scam|fraudulent|suspicious"
NORMAL_WORDS = r"normal|legitimate|legit|genuine|safe|not\s+a\s+scam"


def parse_verdict(raw):
    """Read a Fraud/Normal verdict out of a model response.

    Returns "Fraud", "Normal", or None if the response cannot be read.
    None is the important case: the old code turned it into "Normal", which
    silently converted unreadable answers into false negatives.
    """
    if not raw or not raw.strip():
        return None
    low = raw.lower()

    # 1. explicit answer marker, tolerating markdown and punctuation
    m = re.search(r"answer\s*[:\-]?\s*[*_`\"'\s]*(" + FRAUD_WORDS + "|"
                  + NORMAL_WORDS + r")\b", low)
    if m:
        return "Fraud" if re.match(FRAUD_WORDS, m.group(1)) else "Normal"

    # 2. first non-empty line only (the prompt asks for the verdict there)
    lines = [ln for ln in low.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        f = re.search(r"\b(" + FRAUD_WORDS + r")\b", first)
        n = re.search(r"\b(" + NORMAL_WORDS + r")\b", first)
        if f and not n:
            return "Fraud"
        if n and not f:
            return "Normal"

    # 3. whole response, but only if exactly one side appears
    f = re.search(r"\b(" + FRAUD_WORDS + r")\b", low)
    n = re.search(r"\b(" + NORMAL_WORDS + r")\b", low)
    if f and not n:
        return "Fraud"
    if n and not f:
        return "Normal"

    return None


# Every LLM system asks for its verdict in the same shape, so one constant
# keeps them in step. The Reason line is what fills the *_why columns of the
# per-call CSV: without asking for it, all that can be shown is whatever prose
# the model happened to volunteer, which is often nothing.
VERDICT_FORMAT = (
    "Answer 'Fraud' or 'Normal' on the first line, starting with 'Answer:'.\n"
    "On the second line give one short sentence starting with 'Reason:', "
    "naming what in the call decided it.")

# Tolerant of markdown the way parse_verdict is: qwen2.5 likes to answer
# "**Reason:** ..." and the asterisks are not part of the reason.
REASON_RE = re.compile(r"reason[*_`\s]*[:\-]\s*[*_`\"'\s]*(.+)",
                       re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(r"^[*_`\s]*answer\s*[:\-]", re.IGNORECASE)


def parse_reason(raw, limit=300):
    """The model's own account of why, as one line.

    Falls back to whatever it said either side of the verdict, so a model that
    ignores the format still contributes something rather than an empty cell.
    """
    if not raw:
        return ""
    m = REASON_RE.search(raw)
    if m:
        text = m.group(1)
    else:
        text = " ".join(ln for ln in raw.splitlines()
                        if ln.strip() and not ANSWER_LINE_RE.match(ln))
    text = " ".join(text.split()).strip("*_` ")
    return text[:limit].rstrip()


def looks_truncated(raw):
    """Heuristic: response ended mid-thought rather than concluding."""
    if not raw:
        return False
    t = raw.strip()
    return bool(t) and t[-1] not in ".!?\"')}]" and len(t.split()) > 20


class VerdictStats:
    """Track how the parser coped, so a parsing bug can never hide again."""

    def __init__(self, name):
        self.name = name
        self.ok = 0
        self.retried = 0
        self.failed = 0
        self.truncated = 0
        self.samples = []

    def record(self, status, raw, truncated):
        setattr(self, status, getattr(self, status) + 1)
        if truncated:
            self.truncated += 1
        if status in ("retried", "failed") and len(self.samples) < 5:
            self.samples.append(raw[:300])

    def report(self):
        total = self.ok + self.retried + self.failed
        if total == 0:
            return
        if self.retried or self.failed or self.truncated:
            print("    parsing: %d clean, %d needed retry, %d UNREADABLE, "
                  "%d look truncated" %
                  (self.ok, self.retried, self.failed, self.truncated))
            if self.failed:
                print("    WARNING: %d unreadable responses were scored 'Normal'."
                      % self.failed)
                print("    If these were scam calls they are counted as false "
                      "negatives. Raise --max-tokens or inspect with --debug.")
            for s in self.samples[:2]:
                print("      sample: %r" % s)
        else:
            print("    parsing: all %d responses read cleanly" % total)


def ask_verdict(prompt, stats, max_tokens=300, debug=False, idx=None):
    """Call the model, parse a verdict and its reason, retry once if unreadable.

    Returns (verdict, raw, reason). The reason always comes from the FIRST
    reply: the retry deliberately asks for one word and no explanation, so it
    has none to give.
    """
    import credibility as C

    raw = C.call_ollama(prompt, max_tokens=max_tokens)
    truncated = looks_truncated(raw)
    verdict = parse_verdict(raw)
    # A model that answers with the bare word and nothing else leaves nothing
    # to quote. Say that, rather than leave a blank cell that reads as though
    # the column failed to fill.
    reason = parse_reason(raw) or "(no reason given)"

    if verdict is not None:
        stats.record("ok", raw, truncated)
        if debug and idx is not None and idx <= 5:
            print("      [%d] %s <- %r" % (idx, verdict, (raw or "")[:160]))
        return verdict, raw, reason

    # retry, forcing a one-word answer
    retry_prompt = prompt + "\n\nReply with exactly one word, either Fraud or Normal. No explanation."
    raw2 = C.call_ollama(retry_prompt, max_tokens=10)
    verdict2 = parse_verdict(raw2)
    if verdict2 is not None:
        stats.record("retried", raw, truncated)
        if debug:
            print("      [%s] retry rescued -> %s (first reply: %r)"
                  % (idx, verdict2, (raw or "")[:120]))
        return verdict2, raw, reason or "(verdict came from a one-word retry)"

    stats.record("failed", raw, truncated)
    if debug:
        print("      [%s] UNREADABLE -> defaulting Normal: %r" % (idx, (raw or "")[:160]))
    return "Normal", raw, "(unreadable answer, scored Normal)"


# ------------------------------------------ trivial baselines (no LLM, instant)
def trivial_length(data, threshold=45):
    return [("Fraud" if len(t.split()) > threshold else "Normal", lab)
            for t, lab in data]


def trivial_bow(data, folds=5, show_features=False):
    """TF-IDF + LogisticRegression, properly cross-validated.

    The vectoriser lives inside the Pipeline so it is refitted on each
    training fold. Fitting it on the full dataset first (the previous
    behaviour) leaked test-fold vocabulary and IDF weights into training.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        import numpy as np
    except ImportError:
        print("  (sklearn not installed - skipping bag-of-words)")
        return None

    texts = [t for t, _ in data]
    y = np.array([1 if lab == "Fraud" else 0 for _, lab in data])

    smallest_class = int(min(y.sum(), len(y) - y.sum()))
    n_splits = max(2, min(folds, smallest_class))
    if n_splits < folds:
        print("  (only %d per smallest class, using %d folds)"
              % (smallest_class, n_splits))

    def build(min_df):
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=min_df),
            LogisticRegression(max_iter=2000))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    try:
        pred = cross_val_predict(build(2), texts, y, cv=cv)
    except ValueError:
        # min_df=2 can empty the vocabulary on very small folds
        print("  (min_df=2 left no vocabulary in a fold, falling back to min_df=1)")
        pred = cross_val_predict(build(1), texts, y, cv=cv)

    if show_features:
        _bow_top_features(texts, y)

    return [("Fraud" if p == 1 else "Normal", "Fraud" if t == 1 else "Normal")
            for p, t in zip(pred, y)]


def _bow_top_features(texts, y, k=15):
    """Diagnostic only: which words is BoW actually keying on?

    Fitted on the full dataset deliberately - this is for inspection, not
    for scoring, and is never used to produce predictions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_scam = names[np.argsort(coefs)[-k:]][::-1]
    top_legit = names[np.argsort(coefs)[:k]]
    print("\n  BoW top features (diagnostic, fitted on full data):")
    print("    -> Fraud : %s" % ", ".join(top_scam))
    print("    -> Normal: %s" % ", ".join(top_legit))
    print("    If these look like scam concepts, the signal is real.")
    print("    If they look like style artefacts, the dataset is separable on source.\n")


# --------------------------------------------------- LLM systems (need Ollama)
def run_llm_only(data, max_tokens=300, debug=False):
    """Control: model decides alone, no retrieval."""
    stats = VerdictStats("llm_only")
    out, raws, reasons = [], [], []
    for i, (text, true) in enumerate(data, 1):
        prompt = (
            "You are a scam detection analyst. Read this phone call transcript and "
            "decide whether the caller is attempting a scam.\n\n"
            "Transcript:\n%s\n\n%s" % (text, VERDICT_FORMAT))
        pred, raw, why = ask_verdict(prompt, stats, max_tokens, debug, i)
        out.append((pred, true))
        raws.append(raw)
        reasons.append(why)
        if i % 20 == 0:
            print("    LLM-only: %d/%d" % (i, len(data)))
    stats.report()
    return out, raws, reasons


def run_singh(data, max_tokens=300, debug=False):
    """Singh baseline: policy-compliance check vs bank_policies collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    coll = client.get_collection(name="bank_policies", embedding_function=ef)

    stats = VerdictStats("singh")
    out, raws, reasons = [], [], []
    for i, (text, true) in enumerate(data, 1):
        res = coll.query(query_texts=[text], n_results=3)
        policy = "\n\n".join(res["documents"][0])
        prompt = ("You are a policy inspector. Using ONLY these policies:\n%s\n\n"
                  "Conversation: %s\n\n"
                  "Does the conversation break any policy?\n%s"
                  % (policy, text, VERDICT_FORMAT))
        pred, raw, why = ask_verdict(prompt, stats, max_tokens, debug, i)
        out.append((pred, true))
        raws.append(raw)
        reasons.append(why)
        if i % 20 == 0:
            print("    Singh: %d/%d" % (i, len(data)))
    stats.report()
    return out, raws, reasons


def run_webrag(data, debug=False):
    """Web-RAG system, KB-only (use_web=False).

    NOTE: this delegates to webrag_system.detect(), which does its own
    prompting and parsing. The fixes in this file do NOT reach inside it.
    If Web-RAG's numbers also look odd on a verbose model, check the
    verdict parsing and token budget in webrag_system.py too.
    """
    import webrag_system as W
    coll = W.get_kb_collection()
    print("    gate: min similarity %.2f, LLM relevance check %s"
          % (W.MIN_KB_SIMILARITY, "on" if W.USE_LLM_GATE else "off"))

    gate = Counter()
    out, raws, reasons = [], [], []
    for i, (text, true) in enumerate(data, 1):
        res = W.detect(text, coll, use_web=False, threshold=50)
        out.append((res["predicted"], true))
        raws.append(json.dumps(res, default=str)[:500])
        reasons.append(webrag_reason(res))
        gate["with_evidence" if res["evidence_used"] else "no_evidence"] += 1
        gate["kept"] += res["n_kb_kept"]
        gate["dropped"] += res["n_kb_dropped"]
        if res["gate_note"] == "unreadable":
            gate["judge_unreadable"] += 1
        if i % 20 == 0:
            print("    WebRAG: %d/%d" % (i, len(data)))

    # How often retrieval actually contributed. If with_evidence is ~100% the
    # gate is not biting and the Fraud prior is back; if it is ~0% this is the
    # LLM-only control wearing a different name. Either extreme is a finding.
    n = max(len(data), 1)
    print("    retrieval gate: %d/%d transcripts got evidence (%.0f%%), "
          "%d chunks kept / %d discarded as irrelevant"
          % (gate["with_evidence"], n, 100.0 * gate["with_evidence"] / n,
             gate["kept"], gate["dropped"]))
    if gate["judge_unreadable"]:
        print("    WARNING: relevance judge unreadable on %d transcripts "
              "(kept their candidates rather than guessing)"
              % gate["judge_unreadable"])
    return out, raws, reasons


def webrag_reason(res):
    """The graded verdict as one line: score, then detect()'s own sentence."""
    why = (res.get("reason") or "").strip()
    why = " ".join(why.split())[:300]
    conf = res.get("confidence")
    evid = "no matching evidence" if not res.get("evidence_used") else None
    bits = []
    if conf is not None:
        bits.append("confidence %s/100" % conf)
    if evid:
        bits.append(evid)
    head = "[%s] " % ", ".join(bits) if bits else ""
    return head + why


def _qwen_json(raw):
    """Extract the JSON object from a Qwen response, including fenced JSON."""
    if not raw:
        return None
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text,
                       re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    # a bare object inside a chattier answer ("Here is the JSON: {...}")
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


# --------------------------------------------------------------- Qwen-KB
# The learning baseline: the LLM is shown a TRAINING SPLIT of this dataset,
# generalises the scams in it into reusable patterns, those patterns are
# indexed as a knowledge base, and held-out calls are then judged against what
# is retrieved from it. Same protocol as BERT and bag-of-words - stratified
# k-fold, every row predicted exactly once while it was held out - so its row
# in the results table is comparable with theirs rather than being scored on a
# different, smaller set of calls.
#
# WHY THE KB IS BUILT IN BATCHES
# One prompt holding forty transcripts is tens of thousands of tokens. Ollama
# does not refuse an over-long prompt, it truncates it FROM THE FRONT, which
# deletes the instructions and the JSON schema and leaves the model staring at
# half a transcript. It then answers in prose, no JSON parses, and the step
# died on "Qwen returned no usable training patterns". Small batches with an
# explicit num_ctx keep every prompt inside the window.

QWEN_BATCH_SIZE = 6           # training transcripts per KB-building call
QWEN_EXAMPLE_CHARS = 1500     # per transcript, inside a batch
QWEN_NUM_CTX = 8192           # context window asked of Ollama for those calls
QWEN_N_RETRIEVE = 3           # learned patterns put in front of the judge
QWEN_MIN_SIMILARITY = 0.20    # below this a retrieved pattern is dropped

_KB_SCHEMA = (
    'Return JSON only, no prose, in exactly this shape:\n'
    '{"patterns":[{"category":"short name","behaviours":"what the scammer '
    'does","signals":"what distinguishes it from a legitimate call"}]}'
)


def _kb_patterns_from_batch(batch, max_patterns, num_ctx, debug=False):
    """One KB-building call over a handful of training scams."""
    import credibility as C

    examples = "\n\n".join(
        "TRAINING SCAM %d:\n%s" % (i, text[:QWEN_EXAMPLE_CHARS])
        for i, text in enumerate(batch, 1))
    prompt = (
        "You are building a scam-detection knowledge base from labelled scam "
        "phone transcripts.\n\n%s\n\nGeneralise the recurring tactics in the "
        "transcripts above into at most %d distinct patterns. Describe the "
        "tactic, never the specific call: no names, phone numbers, amounts, "
        "URLs, or copied wording.\n\n%s"
        % (examples, max_patterns, _KB_SCHEMA))
    raw = C.call_ollama(prompt, max_tokens=800, num_ctx=num_ctx)
    parsed = _qwen_json(raw)
    if parsed is None:
        if debug:
            print("      KB batch unreadable: %r" % ((raw or "")[:200],))
        return []
    out = []
    for item in parsed.get("patterns", []):
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "") or "training_scam").strip()
        behaviours = str(item.get("behaviours", "") or "").strip()
        signals = str(item.get("signals", "") or "").strip()
        if behaviours and signals:
            out.append({"category": category, "behaviours": behaviours,
                        "signals": signals})
    return out


def _merge_patterns(raw_patterns, max_patterns):
    """Fold duplicate categories together and keep the best-supported ones.

    Batching means the same tactic comes back from several batches. Merging on
    the category name and ranking by how many batches produced it is a cheap
    consensus: a tactic four batches agreed on is a real regularity of the
    training split, one batch's one-off is probably one transcript's detail.
    """
    merged = {}
    for p in raw_patterns:
        key = re.sub(r"[^a-z0-9]+", "_", p["category"].lower()).strip("_")
        if not key:
            key = "training_scam"
        slot = merged.setdefault(key, {"category": p["category"], "support": 0,
                                       "behaviours": [], "signals": []})
        slot["support"] += 1
        if p["behaviours"] not in slot["behaviours"]:
            slot["behaviours"].append(p["behaviours"])
        if p["signals"] not in slot["signals"]:
            slot["signals"].append(p["signals"])

    ordered = sorted(merged.items(), key=lambda kv: -kv[1]["support"])
    out = []
    for i, (_key, slot) in enumerate(ordered[:max_patterns], 1):
        behaviours = " ".join(slot["behaviours"][:3])
        signals = " ".join(slot["signals"][:3])
        out.append({
            "id": "qwen_training_%04d" % i,
            "category": slot["category"],
            "support": slot["support"],
            "behaviours": behaviours,
            "signals": signals,
            "text": ("Scam category: %s\n\nTypical behaviours: %s\n\n"
                     "Distinguishing signals: %s"
                     % (slot["category"], behaviours, signals)),
            "source": "dataset training split",
        })
    return out


def build_qwen_training_kb(train_data, max_examples=40, max_patterns=8,
                           num_ctx=QWEN_NUM_CTX, debug=False, label=""):
    """Generalise the scams in a training split into KB patterns.

    Returns [] rather than raising, so the caller decides whether an empty
    result is fatal.
    """
    scam_texts = [text for text, lab in train_data if lab == "Fraud"]
    if not scam_texts:
        print("    %sno scam calls in the training split - nothing to "
              "generalise" % label)
        return []

    random.seed(SEED)
    sample = random.sample(scam_texts, min(max_examples, len(scam_texts)))
    batches = [sample[i:i + QWEN_BATCH_SIZE]
               for i in range(0, len(sample), QWEN_BATCH_SIZE)]
    # a single batch only has a few tactics in it; the real cap is applied
    # after merging, across all of them
    per_batch = max(2, min(4, max_patterns))

    raw, unreadable = [], 0
    for i, batch in enumerate(batches, 1):
        got = _kb_patterns_from_batch(batch, per_batch, num_ctx, debug)
        if not got:
            unreadable += 1
        raw.extend(got)
        print("    %sKB build: batch %d/%d, %d raw patterns so far"
              % (label, i, len(batches), len(raw)), flush=True)

    if unreadable:
        print("    %sWARNING: %d/%d KB batches returned no readable JSON"
              % (label, unreadable, len(batches)))
    return _merge_patterns(raw, max_patterns)


class QwenPatternKB:
    """The learned patterns, indexed so they can be retrieved per transcript.

    This is the knowledge-base half of the baseline. It uses an IN-MEMORY
    Chroma client with the same embedding model as the real KB, so it can
    never write into chroma_db/ and disturb the scam_patterns collection the
    Web-RAG baseline reads from.
    """

    def __init__(self, patterns, tag="fold"):
        self.patterns = patterns
        self.collection = None
        self.retrieval = True
        if not patterns:
            return
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            client = chromadb.EphemeralClient()
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2")
            name = "qwen_training_kb_%s" % tag
            # Chroma hands back the same in-memory instance for identical
            # settings, so a second run in one process would collide on the
            # fold names. Start each fold from an empty collection.
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
            self.collection = client.create_collection(
                name=name, embedding_function=ef,
                metadata={"hnsw:space": "cosine"})
            self.collection.add(
                documents=[p["text"] for p in patterns],
                metadatas=[{"category": p["category"]} for p in patterns],
                ids=[p["id"] for p in patterns])
        except Exception as exc:
            # Without an index the baseline still runs, it just shows the judge
            # every learned pattern instead of the nearest few. Say so rather
            # than silently changing what is being measured.
            print("    could not index the learned patterns (%s) - falling "
                  "back to showing all %d in the prompt" % (exc, len(patterns)))
            self.collection = None
            self.retrieval = False

    def evidence_for(self, transcript, n=QWEN_N_RETRIEVE,
                     min_similarity=QWEN_MIN_SIMILARITY):
        """Nearest learned patterns for this call: (evidence, kept, dropped)."""
        if not self.patterns:
            return "", 0, 0
        if self.collection is None:
            return ("\n\n".join(p["text"] for p in self.patterns),
                    len(self.patterns), 0)

        n = min(n, len(self.patterns))
        res = self.collection.query(query_texts=[transcript], n_results=n)
        kept, dropped = [], 0
        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i]
            # the collection is built in cosine space, so similarity is 1 - d
            sim = 1.0 - dist if dist is not None else 0.0
            if sim >= min_similarity:
                kept.append(res["documents"][0][i])
            else:
                dropped += 1
        return "\n\n".join(kept), len(kept), dropped


def _qwen_prompt(evidence, text):
    if evidence:
        return (
            "You are a scam detection analyst. The knowledge base below was "
            "generalised from SEPARATE labelled scam calls, not from this one. "
            "Use it as evidence, but require the behaviour to actually match - "
            "an ordinary legitimate call must still be called Normal.\n\n"
            "LEARNED KNOWLEDGE BASE:\n%s\n\nTRANSCRIPT:\n%s\n\n%s"
            % (evidence, text, VERDICT_FORMAT))
    # Nothing retrieved: judge the call alone rather than hand it scam patterns
    # that did not match. Same footing as the LLM-only control, never worse -
    # the reasoning webrag_system.py gives for its own gate.
    return (
        "You are a scam detection analyst. No learned scam pattern matched "
        "this call, so judge it on its own content.\n\nTRANSCRIPT:\n%s\n\n%s"
        % (text, VERDICT_FORMAT))


def _qwen_judge(indices, data, kb, stats, out, raws, reasons, gate, max_tokens,
                debug, label=""):
    indices = list(indices)
    for n, i in enumerate(indices, 1):
        text, true = data[i]
        evidence, kept, dropped = kb.evidence_for(text)
        gate["with_evidence" if kept else "no_evidence"] += 1
        gate["kept"] += kept
        gate["dropped"] += dropped
        pred, raw, why = ask_verdict(_qwen_prompt(evidence, text), stats,
                                     max_tokens, debug, n)
        out[i] = (pred, true)
        raws[i] = raw
        reasons[i] = ("%s learned pattern%s matched. %s"
                      % (kept, "" if kept == 1 else "s", why)) if kept else why
        if n % 20 == 0:
            print("    %sQwen-KB: %d/%d held-out calls"
                  % (label, n, len(indices)), flush=True)


# ------------------------------------------- the split the learners share
# Qwen-KB and the hybrid must run over the SAME folds and learn from the SAME
# training patterns, otherwise comparing them measures two different random
# knowledge bases rather than the one thing that differs between them - which
# is whether the web-harvested KB is in the mix.

def _stratified_folds(data, folds):
    """(n_splits, [(train_idx, test_idx), ...]) - deterministic for a dataset."""
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    y = np.array([1 if lab == "Fraud" else 0 for _, lab in data])
    smallest = int(min(y.sum(), len(y) - y.sum()))
    if smallest < 2:
        raise RuntimeError(
            "need at least 2 scam and 2 non-scam calls to hold any out (the "
            "smaller class here has %d). Pass --qwen-train-csv to learn the "
            "patterns from a separate file instead." % smallest)
    n_splits = max(2, min(folds, smallest))
    if n_splits < folds:
        print("    (only %d calls in the smaller class, using %d folds)"
              % (smallest, n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return n_splits, list(skf.split(np.zeros(len(data)), y))


# Fold -> learned patterns, so the second system to ask for a fold pays
# nothing. Building a fold's KB is the expensive part of both baselines.
_LEARNED_KB = {}


def learned_patterns_for_fold(data, train_idx, fold, n_splits, max_examples,
                              max_patterns, num_ctx, debug, label=""):
    key = (id(data), fold, n_splits, max_examples, max_patterns, num_ctx)
    if key in _LEARNED_KB:
        patterns = _LEARNED_KB[key]
        print("    %sreusing the %d patterns already learned from this fold"
              % (label, len(patterns)), flush=True)
        return patterns
    train_data = [data[i] for i in train_idx]
    patterns = build_qwen_training_kb(train_data, max_examples, max_patterns,
                                      num_ctx, debug, label)
    if not patterns:
        raise RuntimeError(
            "the LLM produced no usable training patterns on fold %d - every "
            "KB-building prompt came back unreadable. Re-run with --debug to "
            "see them." % fold)
    _LEARNED_KB[key] = patterns
    return patterns


def _patterns_from_train_csv(train_csv, data, max_examples, max_patterns,
                             num_ctx, debug):
    """The no-folds path: learn from a separate file, score every row of data."""
    key = (train_csv, id(data), max_examples, max_patterns, num_ctx)
    if key in _LEARNED_KB:
        patterns = _LEARNED_KB[key]
        print("    reusing the %d patterns already learned from %s"
              % (len(patterns), train_csv), flush=True)
        return patterns
    train_data = load_combined(train_csv)
    held_out = {text for text, _ in data}
    train_data = [(t, l) for t, l in train_data if t not in held_out]
    if not train_data:
        raise RuntimeError(
            "--qwen-train-csv %s has no rows left once the evaluated calls are "
            "removed from it" % train_csv)
    print("    learning from %s (%d calls, none of them scored below)"
          % (train_csv, len(train_data)), flush=True)
    patterns = build_qwen_training_kb(train_data, max_examples, max_patterns,
                                      num_ctx, debug)
    if not patterns:
        raise RuntimeError("the LLM produced no usable training patterns")
    _LEARNED_KB[key] = patterns
    return patterns


def _tag_fold(patterns, fold):
    """Copies of a fold's patterns with fold-unique ids, for the saved KB."""
    out = []
    for p in patterns:
        q = dict(p)
        q["fold"] = fold
        q["id"] = "f%d_%s" % (fold, p["id"])
        out.append(q)
    return out


def run_qwen_kb(data, folds=5, max_examples=40, max_patterns=8,
                max_tokens=300, debug=False, train_csv=None,
                num_ctx=QWEN_NUM_CTX):
    """Learn a KB from a training split, judge held-out calls against it.

    Two shapes, because the runner has two:
      * normal run - stratified k-fold over `data`. Every call is predicted
        exactly once, while it was held out, so the metrics line up with
        BERT's pooled out-of-fold row and with cross-validated bag-of-words.
      * train_csv given - patterns are learned from a separate file and every
        row of `data` is scored against them. This is what makes the single
        transcript mode work: one row cannot be split into train and test, but
        it can be judged against a KB learned from the dataset it came from.
    """
    stats = VerdictStats("qwen_kb")
    gate = Counter()
    out = [(None, true) for _, true in data]
    raws = [None] * len(data)
    reasons = [""] * len(data)

    if train_csv:
        patterns = _patterns_from_train_csv(train_csv, data, max_examples,
                                            max_patterns, num_ctx, debug)
        kb = QwenPatternKB(patterns, "single")
        _qwen_judge(range(len(data)), data, kb, stats, out, raws, reasons, gate,
                    max_tokens, debug)
        all_patterns = patterns
    else:
        n_splits, splits = _stratified_folds(data, folds)
        all_patterns = []
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            label = "fold %d/%d " % (fold, n_splits)
            print("    %strain %d, held out %d"
                  % (label, len(train_idx), len(test_idx)), flush=True)
            patterns = learned_patterns_for_fold(
                data, train_idx, fold, n_splits, max_examples, max_patterns,
                num_ctx, debug, label)
            all_patterns.extend(_tag_fold(patterns, fold))
            kb = QwenPatternKB(patterns, "f%d" % fold)
            _qwen_judge(test_idx, data, kb, stats, out, raws, reasons, gate,
                        max_tokens, debug, label)

    unscored = sum(1 for pred, _ in out if pred is None)
    if unscored:
        raise RuntimeError("Qwen-KB left %d calls unscored" % unscored)

    stats.report()
    n = max(len(data), 1)
    # How often the learned KB actually contributed. 0% means this is the
    # LLM-only control under another name; 100% means the similarity floor is
    # not biting and every call is being shown scam patterns. Either extreme
    # is a finding about the KB, not about the calls.
    print("    retrieval: %d/%d calls got a learned pattern (%.0f%%), "
          "%d kept / %d below the similarity floor"
          % (gate["with_evidence"], n, 100.0 * gate["with_evidence"] / n,
             gate["kept"], gate["dropped"]))
    if train_csv:
        print("    learned %d patterns from %s; all %d calls scored against "
              "them" % (len(all_patterns), train_csv, len(data)))
    else:
        print("    learned %d patterns across %d folds; all %d calls scored "
              "while held out" % (len(all_patterns), n_splits, len(data)))
    return out, raws, reasons, all_patterns


# ----------------------------------------------------------------- hybrid
# Web-RAG and Qwen-KB, over ONE knowledge base.
#
# The two systems differ in where their knowledge comes from, not in what they
# do with it: Web-RAG retrieves from patterns harvested off the web, Qwen-KB
# from patterns the model generalised out of a labelled training split. So the
# hybrid puts both kinds of pattern in the same collection and runs the full
# Web-RAG pipeline over it - the same signal extraction, the same two-stage
# relevance gate, the same graded 0-100 confidence and threshold. Retrieval
# decides per call which kind of knowledge is worth showing, and the prompt
# says which is which (see build_evidence_block in webrag_system.py).
#
# It is scored on exactly the folds Qwen-KB was scored on, reusing exactly the
# patterns Qwen-KB learned, so hybrid-vs-Qwen-KB isolates one variable: the
# web KB. hybrid-vs-Web-RAG isolates the other: the learned patterns.

def build_merged_kb(learned_patterns, tag="hyb", include_web=True):
    """One in-memory collection holding the web KB and the learned patterns.

    In-memory on purpose: the persistent chroma_db/ collection is what the
    Web-RAG baseline reads, and a benchmark must not write into the thing it
    is measuring.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.EphemeralClient()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    name = "hybrid_kb_%s" % tag
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    coll = client.create_collection(name=name, embedding_function=ef,
                                    metadata={"hnsw:space": "cosine"})

    n_web = 0
    if include_web:
        import webrag_system as W
        src = W.get_kb_collection()
        got = src.get(include=["documents", "metadatas"])
        docs = got.get("documents") or []
        if docs:
            metas = []
            for md in (got.get("metadatas") or [{}] * len(docs)):
                md = dict(md or {})
                md["origin"] = "web"
                metas.append(md)
            coll.add(documents=docs, metadatas=metas,
                     ids=["web_" + str(i) for i in got["ids"]])
            n_web = len(docs)

    coll.add(
        documents=[p["text"] for p in learned_patterns],
        metadatas=[{
            "pattern_id": p["id"],
            "scam_type": p["category"],
            "domain": "dataset training split",
            "url": "",
            # No harvest-time credibility exists for a learned pattern and
            # inventing one would put a fabricated number in the prompt. What
            # it has instead is consensus across the KB-building batches.
            "credibility": 0.0,
            "support": int(p.get("support", 1)),
            "origin": "training",
        } for p in learned_patterns],
        ids=["learned_" + p["id"] for p in learned_patterns])

    return coll, n_web, len(learned_patterns)


def _hybrid_judge(indices, data, coll, out, raws, reasons, gate, threshold,
                  label=""):
    import webrag_system as W

    indices = list(indices)
    for n, i in enumerate(indices, 1):
        text, true = data[i]
        res = W.detect(text, coll, use_web=False, threshold=threshold)
        out[i] = (res["predicted"], true)
        raws[i] = json.dumps(res, default=str)[:500]
        # which half of the shared KB was actually in front of the judge is
        # part of why it said what it said, so it goes in the reason
        mix = []
        if res.get("n_kb_harvested"):
            mix.append("%d web" % res["n_kb_harvested"])
        if res.get("n_kb_learned"):
            mix.append("%d learned" % res["n_kb_learned"])
        reasons[i] = (("[%s] " % " + ".join(mix)) if mix else "") + webrag_reason(res)
        gate["with_evidence" if res["evidence_used"] else "no_evidence"] += 1
        gate["kept"] += res["n_kb_kept"]
        gate["dropped"] += res["n_kb_dropped"]
        gate["from_web"] += res.get("n_kb_harvested", 0)
        gate["from_training"] += res.get("n_kb_learned", 0)
        if res["gate_note"] == "unreadable":
            gate["judge_unreadable"] += 1
        if n % 20 == 0:
            print("    %sHybrid: %d/%d held-out calls" % (label, n, len(indices)),
                  flush=True)


def run_hybrid(data, folds=5, max_examples=40, max_patterns=8, debug=False,
               train_csv=None, num_ctx=QWEN_NUM_CTX, threshold=50):
    """Web-RAG's pipeline over a KB holding both web and learned patterns."""
    import webrag_system as W

    print("    gate: min similarity %.2f, LLM relevance check %s"
          % (W.MIN_KB_SIMILARITY, "on" if W.USE_LLM_GATE else "off"))

    gate = Counter()
    out = [(None, true) for _, true in data]
    raws = [None] * len(data)
    reasons = [""] * len(data)

    if train_csv:
        patterns = _patterns_from_train_csv(train_csv, data, max_examples,
                                            max_patterns, num_ctx, debug)
        coll, n_web, n_learned = build_merged_kb(patterns, "single")
        print("    merged KB: %d web-harvested + %d learned patterns"
              % (n_web, n_learned), flush=True)
        _hybrid_judge(range(len(data)), data, coll, out, raws, reasons, gate,
                      threshold)
        all_patterns = patterns
    else:
        n_splits, splits = _stratified_folds(data, folds)
        all_patterns = []
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            label = "fold %d/%d " % (fold, n_splits)
            print("    %strain %d, held out %d"
                  % (label, len(train_idx), len(test_idx)), flush=True)
            patterns = learned_patterns_for_fold(
                data, train_idx, fold, n_splits, max_examples, max_patterns,
                num_ctx, debug, label)
            all_patterns.extend(_tag_fold(patterns, fold))
            coll, n_web, n_learned = build_merged_kb(patterns, "f%d" % fold)
            print("    %smerged KB: %d web-harvested + %d learned patterns"
                  % (label, n_web, n_learned), flush=True)
            _hybrid_judge(test_idx, data, coll, out, raws, reasons, gate,
                          threshold, label)

    unscored = sum(1 for pred, _ in out if pred is None)
    if unscored:
        raise RuntimeError("Hybrid left %d calls unscored" % unscored)

    n = max(len(data), 1)
    print("    retrieval gate: %d/%d calls got evidence (%.0f%%), "
          "%d chunks kept / %d discarded as irrelevant"
          % (gate["with_evidence"], n, 100.0 * gate["with_evidence"] / n,
             gate["kept"], gate["dropped"]))
    # The number this baseline exists to produce. If the surviving evidence is
    # nearly all web, the learned patterns are not earning their place and the
    # hybrid is Web-RAG; if it is nearly all learned, it is Qwen-KB with a
    # slower pipeline. A real mix is what would justify the combination.
    print("    evidence mix: %d chunks from the web KB, %d from the learned "
          "patterns" % (gate["from_web"], gate["from_training"]))
    if gate["judge_unreadable"]:
        print("    WARNING: relevance judge unreadable on %d transcripts "
              "(kept their candidates rather than guessing)"
              % gate["judge_unreadable"])
    return out, raws, reasons, all_patterns


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="./datasets/combined_delex_dataset.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trivial-only", action="store_true")
    ap.add_argument("--skip", default="",
                    help="comma list: length,bow,llm_only,singh,webrag,qwen_kb,hybrid")
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="token budget for LLM verdicts (was 60, which truncated "
                         "verbose models mid-answer)")
    ap.add_argument("--debug", action="store_true",
                    help="print raw model responses and save them to results/")
    ap.add_argument("--bow-features", action="store_true",
                    help="show which words BoW is keying on")
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-validation folds for bag-of-words and Qwen-KB")
    ap.add_argument("--qwen-folds", type=int, default=None,
                    help="folds for Qwen-KB alone (default: --folds). Each fold "
                         "costs one set of KB-building calls; the verdict calls "
                         "are one per transcript either way")
    ap.add_argument("--qwen-max-examples", type=int, default=40,
                    help="training scams sampled per fold to generalise from")
    ap.add_argument("--qwen-patterns", type=int, default=8,
                    help="patterns kept in the learned KB per fold")
    ap.add_argument("--qwen-train-csv", default=None,
                    help="learn the Qwen KB from this file instead of holding "
                         "folds out of --csv. Rows that also appear in --csv are "
                         "dropped from it first. This is how a single-transcript "
                         "run can still use the baseline")
    ap.add_argument("--qwen-num-ctx", type=int, default=QWEN_NUM_CTX,
                    help="context window for the KB-building prompts. Ollama "
                         "truncates an over-long prompt from the front, which "
                         "silently removes the instructions")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    print("Reading %s ..." % args.csv, flush=True)
    data = load_combined(args.csv, limit=args.limit)
    n_fraud = sum(1 for _, l in data if l == "Fraud")

    print("=" * 74)
    print("Dataset evaluation - %d calls (%d scam, %d non-scam)" %
          (len(data), n_fraud, len(data) - n_fraud))
    print("  source: %s" % args.csv)
    print("  LLM max_tokens: %d" % args.max_tokens)
    print("  Compare every LLM system against the trivial baselines below.")
    print("  If length or bag-of-words matches the LLM systems, the dataset is")
    print("  separable without understanding scams, and the comparison is")
    print("  measuring the dataset, not the systems.")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    raw_log = {}
    # system -> one sentence per call saying why. Only the LLM systems have
    # one; length and bag-of-words have no account to give of themselves.
    reasons = {}

    if {"length", "bow"} - skip:
        print("\nTrivial reference classifiers (no LLM):")
    if "length" not in skip:
        print("  length-only ...", flush=True)
        results["length"] = trivial_length(data)
        show("length-only (>45 words)", metrics(results["length"]))
    if "bow" not in skip:
        print("  bag-of-words: %d-fold TF-IDF + logistic regression over %d calls ..."
              % (args.folds, len(data)), flush=True)
        bow = trivial_bow(data, folds=args.folds, show_features=args.bow_features)
        if bow:
            results["bow"] = bow
            show("bag-of-words (TF-IDF+LR)", metrics(bow))

    if args.trivial_only:
        print("\n--trivial-only: stopping before the LLM systems.")
        _save(results, data, reasons, raw_log, args.debug)
        return

    if "llm_only" not in skip:
        print("\nLLM-only (no retrieval):")
        t0 = time.time()
        print("    %s calls to go, one per transcript" % len(data), flush=True)
        (results["llm_only"], raw_log["llm_only"],
         reasons["llm_only"]) = run_llm_only(data, args.max_tokens, args.debug)
        show("LLM-only", metrics(results["llm_only"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "singh" not in skip:
        print("\nSingh baseline (policy compliance):")
        t0 = time.time()
        print("    %s calls to go, one per transcript" % len(data), flush=True)
        (results["singh"], raw_log["singh"],
         reasons["singh"]) = run_singh(data, args.max_tokens, args.debug)
        show("Singh baseline", metrics(results["singh"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "webrag" not in skip:
        print("\nWeb-RAG system (KB-only):")
        t0 = time.time()
        print("    %s calls to go, one per transcript" % len(data), flush=True)
        (results["webrag"], raw_log["webrag"],
         reasons["webrag"]) = run_webrag(data, args.debug)
        show("Web-RAG (KB-only)", metrics(results["webrag"]))
        print("    (%.0fs)" % (time.time() - t0))

    if "qwen_kb" not in skip:
        print("\nQwen-KB baseline (patterns learned from a training split):")
        t0 = time.time()
        try:
            (results["qwen_kb"], raw_log["qwen_kb"], reasons["qwen_kb"],
             qwen_patterns) = run_qwen_kb(
                data, args.qwen_folds or args.folds, args.qwen_max_examples,
                args.qwen_patterns, args.max_tokens, args.debug,
                args.qwen_train_csv, args.qwen_num_ctx)
        except RuntimeError as exc:
            # One baseline failing must not throw away the four that already
            # ran: the whole step used to exit here, so a run that included
            # qwen_kb finished with no results table at all.
            print("    SKIPPED: %s" % exc)
            results.pop("qwen_kb", None)
            raw_log.pop("qwen_kb", None)
            reasons.pop("qwen_kb", None)
        else:
            show("Qwen-KB (held-out)", metrics(results["qwen_kb"]))
            pattern_out = RESULTS_DIR / ("qwen_training_patterns_%d.json"
                                         % len(data))
            with open(pattern_out, "w", encoding="utf-8") as f:
                json.dump(qwen_patterns, f, indent=2)
            print("    learned KB: %s" % pattern_out)
        print("    (%.0fs)" % (time.time() - t0))

    if "hybrid" not in skip:
        print("\nHybrid baseline (Web-RAG pipeline over web + learned KB):")
        t0 = time.time()
        try:
            (results["hybrid"], raw_log["hybrid"], reasons["hybrid"],
             hybrid_patterns) = run_hybrid(
                data, args.qwen_folds or args.folds, args.qwen_max_examples,
                args.qwen_patterns, args.debug, args.qwen_train_csv,
                args.qwen_num_ctx)
        except RuntimeError as exc:
            print("    SKIPPED: %s" % exc)
            results.pop("hybrid", None)
            raw_log.pop("hybrid", None)
            reasons.pop("hybrid", None)
        else:
            show("Hybrid (web + learned KB)", metrics(results["hybrid"]))
            pattern_out = RESULTS_DIR / ("hybrid_learned_patterns_%d.json"
                                         % len(data))
            with open(pattern_out, "w", encoding="utf-8") as f:
                json.dump(hybrid_patterns, f, indent=2)
            print("    learned half of the KB: %s" % pattern_out)
        print("    (%.0fs)" % (time.time() - t0))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in results:
        show(k, metrics(results[k]))
    _save(results, data, reasons, raw_log, args.debug)


def _save(results, data, reasons=None, raw_log=None, debug=False):
    if not results:
        return
    reasons = reasons or {}
    keys = list(results.keys())
    # Each system's verdict, and directly after it the reason it gave, so the
    # two read together. A system with nothing to say (length, bag-of-words)
    # contributes no _why column rather than an empty one. The web UI keys off
    # the _why suffix, so it is part of the interface, not just a name.
    header = ["idx", "true", "text"]
    for k in keys:
        header.append(k)
        if k in reasons:
            header.append(k + "_why")
    out_csv = RESULTS_DIR / ("combined_results_%d.csv" % len(data))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(data)):
            true = results[keys[0]][i][1]
            row = [i, true, data[i][0].replace("\n", " ")[:400]]
            for k in keys:
                row.append(results[k][i][0])
                if k in reasons:
                    why = reasons[k][i] if i < len(reasons[k]) else ""
                    row.append(" ".join((why or "").split()))
            w.writerow(row)
    print("\n  per-call results: %s" % out_csv)
    if reasons:
        print("  with a reason column for: %s" % ", ".join(sorted(reasons)))

    if debug and raw_log:
        out_raw = RESULTS_DIR / ("combined_raw_%d.json" % len(data))
        with open(out_raw, "w", encoding="utf-8") as f:
            json.dump(raw_log, f, indent=2)
        print("  raw model responses: %s" % out_raw)
    print("=" * 74)


if __name__ == "__main__":
    main()