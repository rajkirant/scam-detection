#!/usr/bin/env python3
"""
mcq_ontology.py - the MCQ ontology LLM: a tree of multiple-choice questions.

It starts from a JSON file (knowledge/mcq_ontology.json unless told
otherwise) holding a tree of questions:

  the root question    the subject of the call (bank, tech support,
                       government ...). It does not decide the verdict.
  common questions     asked of every call: how the call came about, which
                       details are asked for, how money would move ...
  subject questions    asked of calls on that subject
  follow-ups           asked only when the option that opens them is chosen

Every option has a value from -1 to 1: positive points toward scam, negative
toward legitimate, not_mentioned 0, and every option of a "recorded"
question 0. One request per question - the transcript, the question and its
options lettered A, B, C ... - and the model answers one letter, and where
the answer needs it a short quote from the transcript. An answer whose quote
is not in the transcript counts as Not stated. Ollama returns the
probability on every letter, which is kept with each answer.

  score     the sum of the values of the options chosen
  verdict   scam above 0, legitimate below 0, neutral at exactly 0

About a dozen requests per call. No retrieval and no learned knowledge:
the transcript, the question and its options are all the model sees.

`train` starts from an ontology and a dataset (a training fold) and writes a
new file: it adds options where calls answered Not stated but had something
to say, and moves each option's value toward the labels of the calls that
chose it. See train_ontology().

An older one-question file - options with a verdict instead of a value -
still loads: each verdict becomes a value of +1 or -1 on the root question.

    python scripts/mcq_ontology.py ask --text "Hello, this is your bank ..."
    python scripts/mcq_ontology.py evaluate --csv datasets/huggingface_1600_test1.csv --limit 40
    python scripts/mcq_ontology.py train --csv datasets/huggingface_1600_train1.csv \\
        --out knowledge/mcq_trained_fold1.json

Your own question can be trained on a dataset too: the model answers it in a
few words about a sample of calls, groups the answers into a few options, and
counts the scam and legitimate calls on each; the result is saved in
knowledge/questions/ and asking it later uses the options in that file.

    python scripts/mcq_ontology.py train-question --csv datasets/huggingface_1600.csv \\
        --question "What does the caller ask for?" --out knowledge/questions/asks.json
    python scripts/mcq_ontology.py question --saved knowledge/questions/asks.json \\
        --text "Hello, this is your bank ..."

Standard library only (Ollama is spoken to through llm_judge), so web_ui.py
can import it without the venv.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_io                                              # noqa: E402
import llm_judge                                               # noqa: E402
import ollama_ctx                                              # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
DEFAULT_ONTOLOGY = KNOWLEDGE_DIR / "mcq_ontology.json"

VERDICTS = ("scam", "legit")
MAX_OPTIONS = 12                 # your own question: past that it is not "a few"
LETTERS = "ABCDEFGHIJKLMNOPQRST"  # one per option; MAX_TREE_OPTIONS of them
ANSWER_TOKENS = 3                # the letter, and room for a space before it
TOP_LOGPROBS = 20

HEAD = """You are a scam detection analyst. Read this phone call transcript.

Transcript:
{transcript}

"""

_LOGPROBS_OK = None          # None = not tried yet; False = this Ollama has none


# ------------------------------------------------------------------- the file
# The ontology is a tree (see knowledge/mcq_ontology.json):
#
#   prompt, options        the root question: the subject of the call. A
#                          subject carries `questions` of its own, and may
#                          carry a `value` (a file of the older one-question
#                          kind does: its options' verdicts become +1 / -1).
#   common_questions       asked of every call
#   ask                    a subject's order for its call: its own question
#                          ids and common/<id>, e.g. urgency first for a bank
#                          call, then where the money is to go
#   question               {id, prompt, role?, options}; role "recorded"
#                          scores 0 on every option
#   option                 {id, text, value -1..1, absence?, follow_up?}
#                          follow_up questions are asked only when that
#                          option is chosen; not_mentioned always scores 0
#
# A call's score is the sum of the values of the options chosen, and its
# verdict is the sign of the score: above 0 scam, below 0 legitimate, 0
# neutral.
NOT_MENTIONED = "not_mentioned"
MAX_TREE_OPTIONS = 20            # one letter each, all inside top_logprobs
QUOTE_TOKENS = 60                # the letter, then "Quote: ..." on its own line
ROLE_RECORDED = "recorded"


def _is_flat(obj):
    """A file of the older kind: one question whose options carry a
    verdict, scam or legit, instead of a value."""
    opts = obj.get("options") if isinstance(obj, dict) else None
    return (isinstance(opts, list) and not obj.get("common_questions")
            and any(isinstance(o, dict) and "verdict" in o for o in opts))


def _from_flat(obj):
    """The older one-question file as a tree: no other questions, and each
    option's verdict as its value (+1 scam, -1 legit, 0 split evenly)."""
    out = {k: v for k, v in obj.items() if k != "options"}
    out["converted_from"] = "one question with verdicts"
    out["common_questions"] = []
    out["options"] = []
    for o in obj["options"]:
        o = dict(o)
        v = o.get("verdict")
        o["value"] = 0.0 if o.get("mixed") else (1.0 if v == "scam" else
                                                  -1.0 if v == "legit" else
                                                  o.get("value", 0.0))
        o.setdefault("questions", [])
        out["options"].append(o)
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_question(q, where, probs, root=False):
    if not isinstance(q, dict):
        probs.append("%s is not an object" % where)
        return
    if not root and not str(q.get("id") or "").strip():
        probs.append('%s has no "id"' % where)
    if not str(q.get("prompt") or "").strip():
        probs.append('%s: "prompt" - the question - is missing or empty' % where)
    opts = q.get("options")
    if not isinstance(opts, list):
        probs.append('%s: "options" must be a list' % where)
        return
    if not 2 <= len(opts) <= MAX_TREE_OPTIONS:
        probs.append("%s: there must be between 2 and %d options; there are %d"
                     % (where, MAX_TREE_OPTIONS, len(opts)))
    seen = set()
    recorded = q.get("role") == ROLE_RECORDED
    for i, o in enumerate(opts, 1):
        ow = "%s, option %d" % (where, i)
        if not isinstance(o, dict):
            probs.append("%s is not an object" % ow)
            continue
        oid = str(o.get("id") or "")
        if not oid:
            probs.append('%s has no "id"' % ow)
        elif oid in seen:
            probs.append('%s: the id "%s" is used twice' % (ow, oid))
        seen.add(oid)
        if not str(o.get("text") or "").strip():
            probs.append('%s has no "text"' % ow)
        v = o.get("value", 0.0)
        if not _num(v) or not -1.0 <= v <= 1.0:
            probs.append('%s (%s): "value" must be a number from -1 to 1, not %r'
                         % (ow, oid, v))
        elif v and oid == NOT_MENTIONED:
            probs.append("%s: not_mentioned must score 0" % ow)
        elif v and recorded:
            probs.append("%s (%s): a recorded question scores 0 on every "
                         "option" % (ow, oid))
        fu = o.get("follow_up", [])
        if not isinstance(fu, list):
            probs.append('%s: "follow_up" must be a list of questions' % ow)
            continue
        for j, f in enumerate(fu, 1):
            _check_question(f, "%s/%s/%s" % (where, oid,
                                             (isinstance(f, dict) and f.get("id"))
                                             or "follow-up %d" % j), probs)


def _check_list(qs, where, probs):
    if not isinstance(qs, list):
        probs.append('%s must be a list of questions' % where)
        return
    ids = set()
    for i, q in enumerate(qs, 1):
        qid = isinstance(q, dict) and q.get("id")
        if qid in ids:
            probs.append('%s: the question id "%s" is used twice' % (where, qid))
        ids.add(qid)
        _check_question(q, "%s/%s" % (where, qid or "question %d" % i), probs)


def check_ontology(obj):
    """Every problem with an ontology file, as a list of sentences; [] if
    none. Takes the tree, or a file of the older one-question kind."""
    if not isinstance(obj, dict):
        return ["the file must hold one JSON object, {...}"]
    if obj.get("kind") == "question":
        return ["this is a saved question, not an ontology"]
    if _is_flat(obj):
        probs = []
        for i, o in enumerate(obj["options"], 1):
            if isinstance(o, dict) and o.get("verdict") not in VERDICTS:
                probs.append('option %d: "verdict" must be "scam" or "legit", '
                             'not %r' % (i, o.get("verdict")))
        if probs:
            return probs
        obj = _from_flat(obj)
    probs = []
    _check_question(obj, "the root question", probs, root=True)
    common = {q.get("id") for q in obj.get("common_questions") or []
              if isinstance(q, dict)}
    for o in obj.get("options") or []:
        if isinstance(o, dict):
            _check_list(o.get("questions", []), str(o.get("id")), probs)
            _check_ask(o, common, probs)
    _check_list(obj.get("common_questions", []), "common", probs)
    return probs


def _check_ask(subject, common, probs):
    """A subject's `ask` order may name only its own questions and the
    common ones (as common/<id>), each once."""
    ask = subject.get("ask")
    if ask is None:
        return
    sid = subject.get("id")
    if not isinstance(ask, list):
        probs.append('%s: "ask" must be a list of question ids' % sid)
        return
    own = {q.get("id") for q in subject.get("questions") or []
           if isinstance(q, dict)}
    seen = set()
    for ref in ask:
        if ref in seen:
            probs.append('%s: "ask" names %r twice' % (sid, ref))
        seen.add(ref)
        ok = (isinstance(ref, str) and (ref in own or (
            ref.startswith("common/") and ref[7:] in common)))
        if not ok:
            probs.append('%s: "ask" names %r, which is neither one of its '
                         'questions nor common/<a common question>'
                         % (sid, ref))


def _norm_question(q):
    q = dict(q)
    q["prompt"] = " ".join(str(q["prompt"]).split())
    opts = []
    for o in q["options"]:
        o = dict(o)
        o["id"] = str(o["id"])
        o["text"] = " ".join(str(o["text"]).split())
        o["value"] = float(o.get("value", 0.0))
        o["follow_up"] = [_norm_question(f) for f in o.get("follow_up", [])]
        opts.append(o)
    q["options"] = opts
    return q


def normalise(obj):
    """The tree with defaults filled in. Assumes it checked."""
    if _is_flat(obj):
        obj = _from_flat(obj)
    out = dict(obj)
    out["prompt"] = " ".join(str(obj["prompt"]).split())
    out["common_questions"] = [_norm_question(q)
                               for q in obj.get("common_questions", [])]
    subs = []
    for o in obj["options"]:
        o = dict(o)
        o["id"] = str(o["id"])
        o["text"] = " ".join(str(o["text"]).split())
        o["value"] = float(o.get("value", 0.0))
        o["questions"] = [_norm_question(q) for q in o.get("questions", [])]
        subs.append(o)
    out["options"] = subs
    return out


def load_ontology(path=None):
    """The ontology at `path`, checked. ValueError naming every problem."""
    path = Path(path or DEFAULT_ONTOLOGY)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("no ontology file at %s" % path)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON: %s (line %d, column %d)"
                         % (path.name, e.msg, e.lineno, e.colno))
    probs = check_ontology(obj)
    if probs:
        raise ValueError("%s: %s" % (path.name, "; ".join(probs[:8])
                                     + ("; and %d more" % (len(probs) - 8)
                                        if len(probs) > 8 else "")))
    return normalise(obj)


def is_mcq_ontology(path):
    """Is this JSON file one of these (rather than another knowledge file)?"""
    try:
        return not check_ontology(json.loads(Path(path).read_text(
            encoding="utf-8")))
    except (OSError, ValueError):
        return False


def iter_questions(onto):
    """(path, question, subject id or None) for every question in the tree,
    follow-ups included. Paths: common/<q>, <subject>/<q>, and
    <parent path>/<option>/<q> for a follow-up."""
    def walk(qs, prefix, subject):
        for q in qs:
            path = "%s/%s" % (prefix, q["id"])
            yield path, q, subject
            for o in q["options"]:
                yield from walk(o["follow_up"], "%s/%s" % (path, o["id"]),
                                subject)
    yield from walk(onto["common_questions"], "common", None)
    for s in onto["options"]:
        yield from walk(s["questions"], s["id"], s["id"])


def count_questions(onto):
    return sum(1 for _ in iter_questions(onto))


def list_ontologies():
    """Every ontology in knowledge/, the default first."""
    out = []
    for p in sorted(KNOWLEDGE_DIR.glob("*.json")):
        if not is_mcq_ontology(p):
            continue
        o = load_ontology(p)
        out.append({"path": "knowledge/" + p.name, "name": p.name,
                    "prompt": o["prompt"], "options": len(o["options"]),
                    "questions": count_questions(o),
                    "trained": len(o.get("training") or []),
                    "default": p.name == DEFAULT_ONTOLOGY.name})
    out.sort(key=lambda d: (not d["default"], d["name"]))
    return out


# ---------------------------------------------------------- asking a question

def answer_letter(token, n):
    """The letter (A..) a token answers with among n options, or None."""
    t = token.strip().strip("*\"'`().:").upper()
    return t if len(t) == 1 and t in LETTERS[:n] else None


def generate(prompt, n_tokens, top=TOP_LOGPROBS, timeout=300, stop=None):
    """(text, [(token, [(alternative, logprob), ...]) per generated token]).
    The alternatives are empty when this Ollama returns no logprobs."""
    num_ctx = ollama_ctx.fit_num_ctx(prompt, n_tokens + 2, where="mcq")
    payload = {
        "model": llm_judge.DEFAULT_MODEL, "prompt": prompt, "stream": False,
        "logprobs": True, "top_logprobs": top,
        "options": {"temperature": 0.0, "num_predict": n_tokens,
                    "num_ctx": int(num_ctx)},
    }
    if stop:
        payload["options"]["stop"] = list(stop)
    try:
        body = llm_judge._post("/api/generate", payload, timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError("ollama refused that (%s): %s"
                           % (e.code, e.read().decode("utf-8", "replace")[:300]))
    except OSError as e:           # URLError, a refused connection, a timeout
        raise llm_judge._unreachable(e)
    steps = []
    for pos in body.get("logprobs") or []:
        alts = [(a.get("token", ""), a["logprob"])
                for a in (pos.get("top_logprobs") or [])
                if a.get("logprob") is not None]
        if not alts and pos.get("logprob") is not None:
            alts = [(pos.get("token", ""), pos["logprob"])]
        steps.append((pos.get("token", ""), alts))
    return body.get("response", ""), steps


def _note_no_logprobs():
    global _LOGPROBS_OK
    if _LOGPROBS_OK is None:
        _LOGPROBS_OK = False
        print("    NOTE this Ollama returns no logprobs (it needs a recent "
              "version): each answer is the letter it gave, at 100%")


def read_letter(text, steps, n):
    """(probs, how) from one answer: the probability on each of n letters,
    measured from the logprobs at the first letter the model wrote.
    how: "logprobs" | "letter" (no logprobs: the answer alone, at 1.0) |
    "unreadable" (probs all 0)."""
    global _LOGPROBS_OK
    has_lp = any(alts for _, alts in steps)
    if has_lp:
        _LOGPROBS_OK = True
    else:
        _note_no_logprobs()
    # the first token that is an option letter; a leading space or "**" may
    # come before it
    li = next((i for i, (tok, _) in enumerate(steps) if answer_letter(tok, n)),
              None)
    if li is not None:
        mass = [0.0] * n
        for tok, lp in steps[li][1]:
            k = answer_letter(tok, n)
            if k:
                mass[LETTERS.index(k)] += math.exp(lp)
        if has_lp and sum(mass) > 0:
            return [m / sum(mass) for m in mass], "logprobs"
        probs = [0.0] * n
        probs[LETTERS.index(answer_letter(steps[li][0], n))] = 1.0
        return probs, "letter"
    if not steps:
        k = answer_letter((re.split(r"[\s\n]", text.strip()) or [""])[0], n)
        if k:
            probs = [0.0] * n
            probs[LETTERS.index(k)] = 1.0
            return probs, "letter"
    return [0.0] * n, "unreadable"


_QUOTE = re.compile(r"quote\s*:\s*(.+)", re.I)
_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def find_quote(text):
    """The quote after "Quote:" in an answer, or None."""
    m = _QUOTE.search(text or "")
    if not m:
        return None
    q = m.group(1).strip().split("\n")[0].strip().strip("\"'“”").strip()
    return None if not q or q.lower().rstrip(".") in ("none", "n/a", "-") else q


def quote_found(quote, transcript):
    """Is the quote in the transcript? Compared word by word, ignoring case
    and punctuation; a quote of three words or more may differ a little - at
    least 80% of its words must run on unbroken in the transcript."""
    q = _WORDS.findall((quote or "").lower())
    if not q:
        return False
    t = _WORDS.findall((transcript or "").lower())
    if " %s " % " ".join(q) in " %s " % " ".join(t):
        return True
    if len(q) < 3:
        return False
    import difflib
    m = difflib.SequenceMatcher(None, t, q, autojunk=False).find_longest_match(
        0, len(t), 0, len(q))
    return m.size >= 0.8 * len(q)


def _rules(onto):
    """The file's how_to_ask, worded for the model: it sees option texts,
    never ids, so not_mentioned is "Not stated" to it."""
    h = {k: (v.replace("not_mentioned", '"Not stated"')
             if isinstance(v, str) else v)
         for k, v in (onto.get("how_to_ask") or {}).items()}
    return {
        "route": h.get("route") or ("This question is only about the subject "
                                    "of the call, not about whether it is a "
                                    "scam."),
        "order": h.get("order") or ("Options run from the strongest scam sign "
                                    "to the strongest legitimate sign, with "
                                    "Not stated last. When several options "
                                    "fit, choose the first one that fits."),
        "absence": h.get("absence") or ("An option that says something did not "
                                        "happen needs no quote. Choose it only "
                                        "when the transcript covers the part "
                                        "of the call where it would have "
                                        "happened."),
    }


def question_prompt(transcript, q, onto, root=False, quotes=True):
    """The prompt for one question about one call: the transcript, the
    question, its options and the file's rules - nothing else."""
    n = len(q["options"])
    rules = _rules(onto)
    lines = ["Question: " + q["prompt"]]
    # absence options are marked, as the file's own rule calls them
    lines += ["%s - %s%s" % (LETTERS[i], o["text"],
                             " (absence)" if o.get("absence") else "")
              for i, o in enumerate(q["options"])]
    lines.append("")
    if root:
        lines.append(rules["route"])
    else:
        lines.append(rules["order"])
        if any(o.get("absence") for o in q["options"]):
            lines.append(rules["absence"])
    if quotes and not root and q.get("role") != ROLE_RECORDED:
        lines.append("Answer with exactly one letter, A to %s. Then, on the "
                     "next line, write \"Quote:\" and a few words copied "
                     "exactly from the transcript that support the answer "
                     "(\"Quote: none\" for Not stated, or for an option that "
                     "says something did not happen)." % LETTERS[n - 1])
    else:
        lines.append("Answer with exactly one letter, A to %s."
                     % LETTERS[n - 1])
    lines.append("Answer:")
    return HEAD.format(transcript=transcript) + "\n".join(lines)


def _needs_quote(q, o, root, quotes):
    return (quotes and not root and q.get("role") != ROLE_RECORDED
            and o["id"] != NOT_MENTIONED and not o.get("absence"))


def ask(transcript, q, onto, root=False, quotes=True, timeout=300):
    """Put one question about one call to the model. One request.

    Returns {"probs", "how", "choice": the option the model picked (None:
    unreadable), "effective": the option that counts - not_mentioned when
    the pick needs a quote and none was found in the transcript, "quote",
    "quoted": True | False | None (no quote needed), "answered": the raw
    reply}.
    """
    n = len(q["options"])
    quoting = quotes and not root and q.get("role") != ROLE_RECORDED
    prompt = question_prompt(transcript, q, onto, root, quotes)
    text, steps = generate(prompt, QUOTE_TOKENS if quoting else ANSWER_TOKENS,
                           timeout=timeout,
                           stop=["\n\n", "\nQuestion"] if quoting else None)
    probs, how = read_letter(text, steps, n)
    choice = None if how == "unreadable" else max(range(n),
                                                  key=lambda i: probs[i])
    effective, quote, quoted = choice, None, None
    if choice is not None:
        o = q["options"][choice]
        if _needs_quote(q, o, root, quotes):
            quote = find_quote(text)
            quoted = bool(quote) and quote_found(quote, transcript)
            if not quoted:
                effective = next((i for i, x in enumerate(q["options"])
                                  if x["id"] == NOT_MENTIONED), None)
        elif quoting:
            quote = find_quote(text)
    return {"probs": probs, "how": how, "choice": choice,
            "effective": effective, "quote": quote, "quoted": quoted,
            "answered": text}


# ------------------------------------------------------------ one whole call

def verdict_of(score):
    """The sign of the score: scam above 0, legit below, neutral at 0."""
    s = round(score, 6)
    return "scam" if s > 0 else "legit" if s < 0 else "neutral"


def pct(score):
    """The score on 0-100 for the Scores tab: 50 at 0, 73 at +1, 27 at -1."""
    return 100.0 / (1.0 + math.exp(-score))


def _answer_record(path, q, a, root=False):
    opts = q["options"]
    eff = a["effective"]
    o = opts[eff] if eff is not None else None
    return {"path": path, "question": q["prompt"], "role": q.get("role"),
            "root": root,
            "choice": a["choice"], "effective": eff,
            # what the model picked, when a missing quote overruled it
            "picked": (opts[a["choice"]]["text"]
                       if a["choice"] is not None and a["choice"] != eff
                       else None),
            "option": o["id"] if o else None,
            "text": o["text"] if o else None,
            "value": o["value"] if o else 0.0,
            "p": a["probs"][a["choice"]] if a["choice"] is not None else None,
            "probs": [round(p, 4) for p in a["probs"]],
            "expected": sum(p * x["value"] for p, x in zip(a["probs"], opts)),
            "how": a["how"], "quote": a["quote"], "quoted": a["quoted"]}


def question_order(onto, subject):
    """[(path, question)] in the order a call on this subject is asked them.

    A subject's `ask` list sets the order: its own question ids, and the
    common questions as common/<id> - for a bank call, urgency first, then
    whether money is to go to a different account, and so on. Questions it
    does not list follow, the common ones first, in file order. Without
    `ask`: the common questions, then the subject's."""
    common = {q["id"]: q for q in onto["common_questions"]}
    own = {q["id"]: q for q in (subject or {}).get("questions", [])}
    sid = (subject or {}).get("id")
    out, seen = [], set()
    for ref in (subject or {}).get("ask") or []:
        if ref.startswith("common/") and ref[7:] in common:
            out.append((ref, common[ref[7:]]))
        elif ref in own:
            out.append(("%s/%s" % (sid, ref), own[ref]))
        seen.add(ref)
    out += [("common/" + k, q) for k, q in common.items()
            if "common/" + k not in seen]
    out += [("%s/%s" % (sid, k), q) for k, q in own.items() if k not in seen]
    return out


def classify(transcript, onto, quotes=True, timeout=300):
    """Walk the tree for one call: the root question, then the common
    questions and the subject's, each follow-up straight after the answer
    that opens it. One request per question.

    Returns {"subject", "subject_text", "answers": [one record per question
    asked, the root first], "score", "expected" (the same sum with each
    answer weighted by its probabilities), "verdict": scam | legit |
    neutral, "requests", "measured": answers read from logprobs}.
    """
    root = {"prompt": onto["prompt"], "options": onto["options"]}
    a = ask(transcript, root, onto, root=True, quotes=quotes, timeout=timeout)
    answers = [_answer_record("root", root, a, root=True)]
    subject = None
    if a["choice"] is not None:
        subject = onto["options"][a["choice"]]
    else:
        subject = next((s for s in onto["options"] if s["id"] == "other"), None)
    queue = question_order(onto, subject)
    i = 0
    while i < len(queue):
        path, q = queue[i]
        a = ask(transcript, q, onto, quotes=quotes, timeout=timeout)
        rec = _answer_record(path, q, a)
        answers.append(rec)
        if a["effective"] is not None:
            o = q["options"][a["effective"]]
            queue[i + 1:i + 1] = [("%s/%s/%s" % (path, o["id"], f["id"]), f)
                                  for f in o["follow_up"]]
        i += 1
    return _summarise(answers, subject)


def _summarise(answers, subject):
    score = sum(r["value"] for r in answers)
    return {"subject": subject["id"] if subject else None,
            "subject_text": subject["text"] if subject else None,
            "answers": answers, "score": round(score, 4),
            "expected": round(sum(r["expected"] for r in answers), 4),
            "verdict": verdict_of(score), "requests": len(answers),
            "measured": sum(1 for r in answers if r["how"] == "logprobs")}


def classify_all(transcripts, onto, quotes=True, parallel=None):
    """classify() over many calls, in order. SCAM_LLM_PARALLEL (or
    `parallel`) sends that many calls at once - which only helps if Ollama
    serves requests in parallel (OLLAMA_NUM_PARALLEL) and the GPU has room."""
    from concurrent.futures import ThreadPoolExecutor
    parallel = max(1, int(parallel or os.environ.get("SCAM_LLM_PARALLEL") or 1))
    if parallel == 1:
        # one at a time in this thread, so Stop (SIGTERM) interrupts the
        # request in flight rather than waiting on a worker
        for t in transcripts:
            yield classify(t, onto, quotes)
        return
    pool = ThreadPoolExecutor(max_workers=parallel)
    try:
        for res in pool.map(lambda t: classify(t, onto, quotes), transcripts):
            yield res
    finally:
        # on Stop, drop the calls not started yet instead of finishing them
        pool.shutdown(wait=False, cancel_futures=True)


def presize(transcripts, onto, quotes=True):
    """Size the context window once, for the longest call and the longest
    question, so the model is loaded once rather than reloaded each time a
    longer prompt turns up (the window only grows - see ollama_ctx.STICKY)."""
    if not transcripts:
        return None
    longest = max(transcripts, key=lambda t: len(t or ""))
    qs = [q for _, q, _ in iter_questions(onto)]
    qs.append({"prompt": onto["prompt"], "options": onto["options"]})
    biggest = max(qs, key=lambda q: len(q["prompt"]) + sum(
        len(o["text"]) + 6 for o in q["options"]))
    return ollama_ctx.fit_num_ctx(
        question_prompt(longest, biggest, onto, False, quotes),
        QUOTE_TOKENS + 2, where="mcq")


def explain(res):
    """A short account of one call: subject, score, the answers that moved
    it."""
    moved = ["%s %+.1f" % (r["option"], r["value"]) for r in res["answers"]
             if r["value"] and not r["root"]]
    unq = sum(1 for r in res["answers"] if r["quoted"] is False)
    return ("%s; score %+.2f (%s)%s%s"
            % (res["subject"] or "no subject", res["score"], res["verdict"],
               ": " + ", ".join(moved) if moved else "",
               "; %d answer%s without a quote counted as not stated"
               % (unq, "" if unq == 1 else "s") if unq else ""))


def subject_table(onto, results, truths):
    """Which subject each call was routed to, against the true label, with
    how many of each the score called scam."""
    c = Counter((r["subject"], bool(t)) for r, t in zip(results, truths))
    lines = ["%-3s %-22s %6s %6s" % ("", "subject", "scam", "legit")]
    for i, s in enumerate(onto["options"]):
        a, b = c[(s["id"], True)], c[(s["id"], False)]
        if a or b:
            lines.append("%-3s %-22s %6d %6d" % (LETTERS[i], s["id"][:22], a, b))
    a, b = c[(None, True)], c[(None, False)]
    if a or b:
        lines.append("%-3s %-22s %6d %6d" % ("", "(unreadable)", a, b))
    return lines


# ------------------------------------------------------- personalisation
# The same words can deserve a different label. A courier asking for import
# tax on a parcel worth 80 is normal in Ireland, where VAT is due on all goods
# from outside the EU, and can only be a scam in New Zealand, where nothing is
# collected at the border below NZ$1,000. Where the person lives is not in
# the transcript. A country file (knowledge/countries/<name>.json) holds that
# country's law, a few questions put to the call, and rules that turn the
# answers into a label:
#
#   {"name", "law", "source", "applies_to"?: [subject ids],
#    "questions": [questions, as in the ontology, without values],
#    "rules": [{"if": {<question id>: [option ids]},
#               "label": "scam" | "legit" | "unsure", "why", "advice"?}]}
#
# The first rule whose every condition holds decides; when none does, the
# verdict from the words stands.
COUNTRIES_DIR = KNOWLEDGE_DIR / "countries"
LABELS = ("scam", "legit", "unsure")


def check_country(obj):
    """Every problem with a country file, as sentences; [] if none."""
    if not isinstance(obj, dict):
        return ["the file must hold one JSON object, {...}"]
    probs = []
    if not str(obj.get("name") or "").strip():
        probs.append('"name" is missing')
    qs = obj.get("questions")
    _check_list(qs if isinstance(qs, list) else None, "questions", probs)
    answers = {}
    for q in qs if isinstance(qs, list) else []:
        if isinstance(q, dict) and isinstance(q.get("options"), list):
            answers[q.get("id")] = {o.get("id") for o in q["options"]
                                    if isinstance(o, dict)}
    rules = obj.get("rules")
    if not isinstance(rules, list) or not rules:
        probs.append('"rules" must be a list of at least one rule')
        rules = []
    for i, r in enumerate(rules, 1):
        where = "rule %d" % i
        if not isinstance(r, dict) or not isinstance(r.get("if"), dict) \
                or not r["if"]:
            probs.append('%s needs an "if": {question id: [answers]}' % where)
            continue
        if r.get("label") not in LABELS:
            probs.append('%s: "label" must be scam, legit or unsure, not %r'
                         % (where, r.get("label")))
        if not str(r.get("why") or "").strip():
            probs.append('%s has no "why"' % where)
        for k, v in r["if"].items():
            if k not in answers:
                probs.append('%s: "%s" is not a question of this country'
                             % (where, k))
            elif not isinstance(v, list) or not v:
                probs.append('%s: "%s" must list the answers it accepts'
                             % (where, k))
            else:
                bad = [x for x in v if x not in answers[k]]
                if bad:
                    probs.append("%s: %s has no answer %s"
                                 % (where, k, ", ".join(map(str, bad))))
    return probs


def load_country(path):
    """A country file, checked. ValueError naming every problem."""
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("no country file at %s" % path)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON: %s (line %d, column %d)"
                         % (path.name, e.msg, e.lineno, e.colno))
    probs = check_country(obj)
    if probs:
        raise ValueError("%s: %s" % (path.name, "; ".join(probs)))
    obj = dict(obj)
    obj["questions"] = [_norm_question(q) for q in obj["questions"]]
    return obj


def list_countries():
    """Every country file in knowledge/countries/ that loads, by file name."""
    out = []
    for p in sorted(COUNTRIES_DIR.glob("*.json")) if COUNTRIES_DIR.is_dir() \
            else []:
        try:
            c = load_country(p)
        except ValueError:
            continue
        out.append({"path": "knowledge/countries/" + p.name, "name": c["name"],
                    "law": c.get("law", ""), "source": c.get("source", "")})
    return out


def find_country(name):
    """A country file by path, or by its name ("New Zealand") or file name
    ("new_zealand") - what a dataset's country column holds."""
    for p in (Path(name), PROJECT_DIR / str(name)):
        if p.is_file():
            return p
    key = re.sub(r"[\s_-]+", " ", str(name)).strip().lower()
    for c in list_countries():
        stem = Path(c["path"]).stem.replace("_", " ").lower()
        if key in (c["name"].lower(), stem):
            return PROJECT_DIR / c["path"]
    raise ValueError("no country file for %r - the ones there are: %s"
                     % (name, ", ".join(c["name"] for c in list_countries())
                        or "none"))


def personalise(transcript, res, country, quotes=True, timeout=300):
    """The label for a person in this country: a classified call (`res`, from
    classify) put through the country's questions and rules.

    Returns {"country", "law", "source", "applies": whether the call is on a
    subject the country's rules are for, "answers": its questions' answer
    records, "label": scam | legit | unsure, or None when no rule applies,
    "why", "advice", "rule", "final": the label, or the words' verdict when
    there is none, "from": "country" | "words"}.
    """
    out = {"country": country["name"], "law": country.get("law", ""),
           "source": country.get("source", ""), "answers": [], "label": None,
           "advice": None, "rule": None}
    applies = country.get("applies_to")
    if applies and res.get("subject") not in applies:
        out.update(applies=False, final=res["verdict"], **{"from": "words"},
                   why="This call is not on a subject %s's rules cover, so "
                       "the verdict from the words stands." % country["name"])
        return out
    out["applies"] = True
    chosen = {}
    for q in country["questions"]:
        a = ask(transcript, q, {}, quotes=quotes, timeout=timeout)
        rec = _answer_record("country/" + q["id"], q, a)
        out["answers"].append(rec)
        chosen[q["id"]] = rec["option"]
    for i, rule in enumerate(country["rules"], 1):
        if all(chosen.get(k) in v for k, v in rule["if"].items()):
            out.update(label=rule["label"], why=rule["why"], rule=i,
                       advice=rule.get("advice"), final=rule["label"],
                       **{"from": "country"})
            return out
    out.update(final=res["verdict"], why="None of %s's rules applies, so the "
               "verdict from the words stands." % country["name"],
               **{"from": "words"})
    return out


# ------------------------------------------------- your own question
# Any question about one call, typed on the page. If the question lists its
# own options - "A) ... B) ...", one per line, or "options: x / y / z" - it is
# put as a multiple choice and answered like the ontology's question: one
# letter, with the probability on every option read back. If it lists none,
# the model answers in its own words, from the transcript and, where the
# transcript does not say, from what it knows.
MAX_ANSWER_TOKENS = 400

_OPT_LINE = re.compile(r"^\s*(?:\(?([A-La-l]|\d{1,2})[).:]|[-*\u2022])\s+(\S.*?)\s*$")
_OPT_INLINE = re.compile(r"(?:(?<=\s)|^)\(?([A-La-l])\)\s*")
_OPT_LIST = re.compile(r"\boptions?\s*(?:are|:|-)\s*(.+)$", re.I | re.S)


def parse_options(text):
    """(question, [option text, ...]) - the options the question lists, if
    it lists at least two; otherwise (the whole text, []).

    Three ways of listing them are understood: one per line ("A) yes",
    "1. yes", "- yes"), inline lettered ("... A) yes B) no"), and
    "options: yes / no / not said" (split on / | ; or commas).
    """
    text = (text or "").strip()
    lines = text.split("\n")
    stem, opts = [], []
    for ln in lines:
        m = _OPT_LINE.match(ln)
        if m:
            opts.append(m.group(2))
        elif opts and ln.strip():
            opts[-1] += " " + ln.strip()      # an option's text run onto a new line
        else:
            stem.append(ln)
    if len(opts) >= 2:
        return " ".join(" ".join(stem).split()), opts

    marks = list(_OPT_INLINE.finditer(text))
    # inline letters only count as options when they run A, B, C ... in order
    run = []
    for m in marks:
        if m.group(1).upper() == LETTERS[len(run)]:
            run.append(m)
    if len(run) >= 2:
        opts = [text[a.end():(b.start() if b else len(text))].strip(" ,;")
                for a, b in zip(run, run[1:] + [None])]
        if all(opts):
            return " ".join(text[:run[0].start()].split()), opts

    m = _OPT_LIST.search(text)
    if m:
        body = m.group(1).strip().rstrip("?.")
        for sep in ("/", "|", ";", ","):
            parts = [x.strip() for x in body.split(sep) if x.strip()]
            if len(parts) >= 2:
                return " ".join(text[:m.start()].split()), parts
    return " ".join(text.split()), []


def _mcq_prompt(transcript, stem, opts):
    n = len(opts)
    return HEAD.format(transcript=transcript) + "\n".join(
        [stem or "Which of these is right?"]
        + ["%s - %s" % (LETTERS[i], o) for i, o in enumerate(opts)]
        + ["Answer with exactly one letter, A to %s." % LETTERS[n - 1],
           "Answer:"])


def choose(transcript, stem, opts, timeout=300):
    """Put `stem` to the model as a multiple choice over `opts` (texts) about
    one call. One request. (probs, choice, how, answered, prompt): probs
    sums to 1 when read; choice is None when no letter could be read."""
    n = len(opts)
    prompt = _mcq_prompt(transcript, stem, opts)
    text, steps = generate(prompt, ANSWER_TOKENS, timeout=timeout)
    probs, how = read_letter(text, steps, n)
    choice = (max(range(n), key=lambda i: probs[i])
              if how != "unreadable" else None)
    return probs, choice, how, text, prompt


def ask_question(transcript, question_text, options=None, timeout=300):
    """Answer one typed question about one call. One request.

    `options` - option texts, e.g. a saved question's - makes it multiple
    choice over those, and the question text is then taken as it is. Without
    them the question's own listed options are used, if it lists any.

    Returns {"mode": "options" | "free", "question": the stem, ...}:
      options   "options": [{"letter", "text", "p"}], "choice", "how"
                ("logprobs" | "letter" | "unreadable"), "answered"
      free      "answer": the model's own words, "truncated": whether it ran
                out of room
    """
    if options:
        stem, opts = " ".join(str(question_text or "").split()), list(options)
    else:
        stem, opts = parse_options(question_text)
    if not stem and not opts:
        raise ValueError("type a question")
    if len(opts) > MAX_OPTIONS:
        raise ValueError("at most %d options - that question lists %d"
                         % (MAX_OPTIONS, len(opts)))
    if opts:
        probs, choice, how, text, prompt = choose(transcript, stem, opts,
                                                  timeout)
        return {"mode": "options", "question": stem,
                "options": [{"letter": LETTERS[i], "text": o, "p": probs[i]}
                            for i, o in enumerate(opts)],
                "choice": choice, "how": how, "answered": text,
                "prompt_tokens_estimated": ollama_ctx.estimate_tokens(prompt)}

    prompt = HEAD.format(transcript=transcript) + (
        "Question: %s\n\nAnswer the question about this call. "
        "Use what the transcript says; where it does not say, "
        "answer from your own knowledge and say that you are. "
        "Keep it to a few sentences.\nAnswer:" % stem)
    num_ctx = ollama_ctx.fit_num_ctx(prompt, MAX_ANSWER_TOKENS,
                                     where="mcq question")
    out = llm_judge.generate(prompt, max_tokens=MAX_ANSWER_TOKENS,
                             num_ctx=num_ctx, timeout=timeout)
    return {"mode": "free", "question": stem,
            "answer": (out.get("response") or "").strip(),
            "truncated": out.get("done_reason") == "length",
            "prompt_tokens_estimated": ollama_ctx.estimate_tokens(prompt)}


# ------------------------------------------- a question trained on a dataset
# A question typed on the page can be trained on a dataset, and saved: the
# model answers it in a few words about a balanced sample of the dataset's
# calls, then groups those answers into a few options, and each sampled call
# is put back to the model as a multiple choice over them, which counts how
# many scam and legitimate calls land on each option. The question, its
# options and those counts go to knowledge/questions/<name>.json; asking the
# saved question later uses the options in that file.
#
# A question that already lists its own options keeps them: training then
# only does the last step, the counting.
QUESTIONS_DIR = KNOWLEDGE_DIR / "questions"
TRAIN_CALLS = 20
TRAIN_OPTIONS = 6
SHORT_ANSWER_TOKENS = 40
GROUP_TOKENS = 300
NOT_SAID = "not said"


def short_answer(transcript, stem, instruction=None, timeout=300):
    """The model's answer to the question about one call, in a few words."""
    prompt = HEAD.format(transcript=transcript) + (
        "Question: %s\n\n%s\nShort answer:"
        % (stem, instruction or ("Answer in a few words - at most 12 - from "
                                 "what the transcript says. If the transcript "
                                 "does not say, answer \"%s\"." % NOT_SAID)))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, SHORT_ANSWER_TOKENS,
                                     where="question training")
    out = llm_judge.generate(prompt, max_tokens=SHORT_ANSWER_TOKENS,
                             num_ctx=num_ctx, timeout=timeout)
    line = (out.get("response") or "").strip().split("\n")[0]
    return line.strip().strip("\"'*").strip().rstrip(".") or NOT_SAID


def _clean_option(line):
    m = _OPT_LINE.match(line)
    text = m.group(2) if m else line
    text = re.sub(r"\s*\(\s*\d+\s*(?:answers?|calls?)?\s*\)\s*$", "", text)
    text = text.strip().strip("\"'*").strip().rstrip(".")
    return " ".join(text.split())[:120]


def group_answers(stem, answers, max_options=TRAIN_OPTIONS, timeout=300):
    """A few options that cover the short answers, written by the model.

    Falls back to the most common answers themselves when the model's list
    has fewer than two usable lines.
    """
    listed = "\n".join("%d. %s" % (i, a) for i, a in enumerate(answers, 1))
    prompt = (
        "A model was asked this question about %d different phone calls:\n\n"
        "  %s\n\nIts short answers, one per call:\n\n%s\n\n"
        "Group these answers into at most %d options for a multiple-choice "
        "version of the question. Each option is a short phrase of at most 8 "
        "words that covers several of the answers. Together the options must "
        "cover every answer, and no two may mean the same thing. If some "
        "answers say the call does not say, keep one option for that.\n"
        "Write one option per line, each starting with \"- \", and nothing "
        "else.\nOptions:" % (len(answers), stem, listed, max_options))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, GROUP_TOKENS,
                                     where="question training")
    out = llm_judge.generate(prompt, max_tokens=GROUP_TOKENS, num_ctx=num_ctx,
                             timeout=timeout)
    opts, seen = [], set()
    for line in (out.get("response") or "").split("\n"):
        o = _clean_option(line)
        if not o or o.lower().rstrip(":") in ("options", "option") \
                or o.lower() in seen:
            continue
        seen.add(o.lower())
        opts.append(o)
    opts = opts[:max_options]
    if len(opts) >= 2:
        return opts, "grouped by the model"
    first = {}
    for a in answers:
        first.setdefault(a.lower(), a)
    common = [first[a] for a, _ in Counter(a.lower() for a in answers)
              .most_common(max_options)]
    if len(common) < 2:
        raise ValueError("every call got the same answer (%r), so there is "
                         "nothing to make options from - ask something the "
                         "calls differ on" % (answers[0] if answers else ""))
    return common, "the most common answers (the model's grouping was unusable)"


def sample_calls(csv_path, calls=TRAIN_CALLS, seed=42):
    """A balanced random sample: [(text, is_scam)], half of each label where
    the dataset has enough of both."""
    rows = dataset_io.read_rows(Path(csv_path))
    tcol, lcol, _ = dataset_io.columns(rows, csv_path)
    scam, legit = [], []
    for r in rows:
        t = (r[tcol] or "").strip()
        if t:
            (scam if dataset_io.is_scam(r[lcol]) else legit).append(t)
    if not scam and not legit:
        raise ValueError("%s has no transcripts" % csv_path)
    rng = random.Random(seed)
    k = min(calls // 2, len(scam))
    j = min(calls - k, len(legit))
    k = min(calls - j, len(scam))          # top up from scam if legit ran short
    pick = ([(t, True) for t in rng.sample(scam, k)]
            + [(t, False) for t in rng.sample(legit, j)])
    rng.shuffle(pick)
    return pick


def learn_options(sample, stem, listed=None, max_options=TRAIN_OPTIONS,
                  instruction=None, log=print):
    """The three steps shared by training a question and building an
    ontology, over a sample of [(transcript, is_scam)]:

      1. the model answers `stem` about each call in a few words
      2. it groups those answers into at most max_options options
      3. each call is put back as a multiple choice over the options, which
         counts the scam and legitimate calls that land on each

    `listed` options skip steps 1 and 2. Returns {"options": texts,
    "source": where they came from, "counts": [{"scam", "legit"}],
    "answers": step 1's answers (None each when skipped), "picked": the
    letter index each call landed on (None: unreadable), "measured",
    "unreadable"}.
    """
    answers = [None] * len(sample)
    steps = 1 if listed else 3
    if listed:
        opts, source = list(listed), "listed in the question"
        log("  the question lists its own %d options, so they are kept: "
            "training only counts\n  which calls land on each" % len(opts))
    else:
        log("\n  step 1 of 3: the model %s in a few words"
            % "answers the question about each call")
        for i, (text, y) in enumerate(sample, 1):
            answers[i - 1] = short_answer(text, stem, instruction)
            log("    %2d/%d  %-5s  %s" % (i, len(sample),
                                          "scam" if y else "legit",
                                          answers[i - 1][:90]))
        log("\n  step 2 of 3: the model groups the %d answers into at most %d "
            "options" % (len(answers), max_options))
        opts, source = group_answers(stem, answers, max_options)
        for i, o in enumerate(opts):
            log("    %s  %s" % (LETTERS[i], o))
        if source != "grouped by the model":
            log("    (%s)" % source)

    log("\n  step %d of %d: each call is put back as a multiple choice over "
        "the options" % (steps, steps))
    longest = max((t for t, _ in sample), key=len)
    ollama_ctx.fit_num_ctx(_mcq_prompt(longest, stem, opts), ANSWER_TOKENS + 2,
                           where="question training")
    counts = [{"scam": 0, "legit": 0} for _ in opts]
    picked, unread, measured = [], 0, 0
    for i, (text, y) in enumerate(sample, 1):
        probs, choice, how, _, _ = choose(text, stem, opts)
        measured += how == "logprobs"
        picked.append(choice)
        if choice is None:
            unread += 1
            log("    %2d/%d  %-5s  (no readable letter)"
                % (i, len(sample), "scam" if y else "legit"))
            continue
        counts[choice]["scam" if y else "legit"] += 1
        log("    %2d/%d  %-5s  %s %5.1f%%  %s"
            % (i, len(sample), "scam" if y else "legit", LETTERS[choice],
               100 * probs[choice], opts[choice][:70]))
    if unread:
        log("  %d call%s gave no readable letter"
            % (unread, "" if unread == 1 else "s"))
    return {"options": opts, "source": source, "counts": counts,
            "answers": answers, "picked": picked, "measured": measured,
            "unreadable": unread}


def _check_training(calls, max_options):
    if not 2 <= max_options <= MAX_OPTIONS:
        raise ValueError("options must be between 2 and %d" % MAX_OPTIONS)
    if not 2 <= calls <= 500:
        raise ValueError("train on between 2 and 500 calls")


def _trained_on(csv_path, sample, seed, max_options, learned):
    n_scam = sum(1 for _, y in sample if y)
    return {"dataset": str(csv_path), "calls": len(sample), "scam": n_scam,
            "legit": len(sample) - n_scam, "seed": seed,
            "max_options": max_options, "model": llm_judge.DEFAULT_MODEL,
            "measured": learned["measured"],
            "unreadable": learned["unreadable"],
            "date": time.strftime("%Y-%m-%d %H:%M")}


def _answers(sample, learned):
    return [{"label": "scam" if y else "legit", "answer": a,
             "option": None if c is None else LETTERS[c]}
            for (_, y), a, c in zip(sample, learned["answers"],
                                    learned["picked"])]


def train_question(csv_path, question_text, calls=TRAIN_CALLS,
                   max_options=TRAIN_OPTIONS, seed=42, log=print):
    """Train a question on a dataset; returns what goes in its JSON file."""
    stem, listed = parse_options(question_text)
    if not stem:
        raise ValueError("type a question")
    if len(listed) > MAX_OPTIONS:
        raise ValueError("at most %d options - that question lists %d"
                         % (MAX_OPTIONS, len(listed)))
    _check_training(calls, max_options)
    sample = sample_calls(csv_path, calls, seed)
    n_scam = sum(1 for _, y in sample if y)
    log("  question: %s" % stem)
    log("  %d calls from %s (%d scam, %d not), seed %d"
        % (len(sample), csv_path, n_scam, len(sample) - n_scam, seed))
    t0 = time.time()
    learned = learn_options(sample, stem, listed, max_options, log=log)
    options = [{"letter": LETTERS[i], "text": o, "calls": c}
               for i, (o, c) in enumerate(zip(learned["options"],
                                              learned["counts"]))]
    log("\n  %-3s %-50s %5s %5s" % ("", "option", "scam", "legit"))
    for o in options:
        log("  %-3s %-50s %5d %5d" % (o["letter"], o["text"][:50],
                                      o["calls"]["scam"], o["calls"]["legit"]))
    log("  %.0fs" % (time.time() - t0))
    return {
        "kind": "question",
        "prompt": stem,
        "asked": " ".join(str(question_text).split()),
        "options": options,
        "options_from": learned["source"],
        "trained_on": _trained_on(csv_path, sample, seed, max_options, learned),
        "answers": _answers(sample, learned),
    }


def check_question(obj):
    """Every problem with a saved question, as sentences; [] if none."""
    if not isinstance(obj, dict) or obj.get("kind") != "question":
        return ['not a saved question (it needs "kind": "question")']
    probs = []
    if not str(obj.get("prompt") or "").strip():
        probs.append('"prompt" - the question - is missing or empty')
    opts = obj.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        probs.append('"options" must be a list of 2 to %d options'
                     % MAX_OPTIONS)
    else:
        for i, o in enumerate(opts, 1):
            if not isinstance(o, dict) or not str(o.get("text") or "").strip():
                probs.append('option %d has no "text"' % i)
    return probs


def load_question(path):
    """A saved question, checked, with each option's scam share worked out
    (None when no training call landed on it). ValueError if broken."""
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("no saved question at %s" % path)
    except json.JSONDecodeError as e:
        raise ValueError("%s is not valid JSON: %s (line %d, column %d)"
                         % (path.name, e.msg, e.lineno, e.colno))
    probs = check_question(obj)
    if probs:
        raise ValueError("%s: %s" % (path.name, "; ".join(probs)))
    obj["prompt"] = " ".join(str(obj["prompt"]).split())
    for i, o in enumerate(obj["options"]):
        o["letter"] = LETTERS[i]
        o["text"] = " ".join(str(o["text"]).split())
        c = o.get("calls") or {}
        s, l = int(c.get("scam") or 0), int(c.get("legit") or 0)
        o["calls"] = {"scam": s, "legit": l}
        o["scam_share"] = s / (s + l) if s + l else None
    return obj


def list_questions():
    """Every saved question in knowledge/questions/, newest first."""
    out = []
    for p in QUESTIONS_DIR.glob("*.json") if QUESTIONS_DIR.is_dir() else []:
        try:
            q = load_question(p)
        except ValueError:
            continue
        t = q.get("trained_on") or {}
        out.append({"path": "knowledge/questions/" + p.name, "name": p.name,
                    "prompt": q["prompt"], "options": len(q["options"]),
                    "dataset": Path(t.get("dataset") or "").name or None,
                    "calls": t.get("calls"), "mtime": p.stat().st_mtime})
    out.sort(key=lambda d: -d["mtime"])
    return out


def ask_saved(transcript, path, timeout=300):
    """Ask a saved question about one call: multiple choice over the options
    in its file. Adds each option's training counts, and `scam_lean` - the
    answer's probabilities weighted by the share of scam calls among the
    training calls on each option (None when no option it put weight on had
    any)."""
    q = load_question(path)
    res = ask_question(transcript, q["prompt"],
                       options=[o["text"] for o in q["options"]],
                       timeout=timeout)
    num = den = 0.0
    for o, a in zip(q["options"], res["options"]):
        a["calls"], a["scam_share"] = o["calls"], o["scam_share"]
        if o["scam_share"] is not None and res["how"] != "unreadable":
            num += a["p"] * o["scam_share"]
            den += a["p"]
    res["scam_lean"] = num / den if den > 0 else None
    res["saved"] = {"path": "knowledge/questions/" + Path(path).name,
                    "trained_on": q.get("trained_on"),
                    "options_from": q.get("options_from")}
    return res


# ------------------------------------------------ training on a dataset
# Training starts from an ontology (the baseline file, or one trained
# before) and a balanced sample of a dataset's calls, and writes a new file:
#
#   1. every sampled call is walked through the tree, as classify() does
#   2. new options: for each scored question, the calls that answered Not
#      stated are asked it openly, in a few words; when some of them have a
#      real answer, the model proposes up to N new options that the existing
#      ones do not cover, and those calls are asked the question again with
#      the new options in
#   3. new values: each option's value moves toward what the calls that
#      chose it say - +1 for every scam call, -1 for every legitimate one -
#      with the old value counting as `prior` calls' worth:
#
#          value = (prior * old + scam - legit) / (prior + scam + legit)
#
#      (scam and legit are weighted so each class counts half when the
#      sample is not balanced). not_mentioned and recorded questions stay 0.
#      The root question's subjects are not given values unless they already
#      carry them.
#   4. each scored question's options are put back in order, strongest scam
#      sign first and Not stated last, since the model is told to choose the
#      first that fits
TRAIN_ONTOLOGY_CALLS = 40
NEW_OPTIONS = 2
PRIOR_WEIGHT = 4.0
PROBE_CALLS = 8          # Not-stated calls asked openly, per question
_EMPTY_ANSWERS = ("not said", "not stated", "not mentioned", "none", "no",
                  "n/a", "unknown", "nothing", "not specified", "unclear",
                  "does not say", "doesn't say", "not applicable")


def _scored(q):
    return q.get("role") != ROLE_RECORDED


def _root_scored(onto):
    return any(s["value"] for s in onto["options"])


def _substantive(answer):
    a = " ".join((answer or "").lower().split()).strip(" .!\"'")
    return bool(a) and a not in _EMPTY_ANSWERS and not any(
        a.startswith(x) for x in ("not said", "not stated", "not mentioned",
                                  "the transcript does not", "it does not say",
                                  "no mention"))


def propose_options(q, answers, k, timeout=300):
    """Up to k new option texts for question q that cover `answers` (open
    answers from calls that chose Not stated) and that its options do not
    already cover. [] when the model says they are covered."""
    existing = "\n".join("- " + o["text"] for o in q["options"]
                         if o["id"] != NOT_MENTIONED)
    listed = "\n".join("%d. %s" % (i, a) for i, a in enumerate(answers, 1))
    prompt = (
        "A multiple-choice question about phone calls:\n\n  %s\n\n"
        "Its options are:\n%s\n- Not stated\n\n"
        "For some calls the answer chosen was Not stated, but asked openly "
        "the answers were:\n\n%s\n\n"
        "Write up to %d new options that cover these answers and that none "
        "of the existing options already covers. Each new option is a short "
        "phrase of at most 15 words, written like the existing ones. Do not "
        "say whether the call is a scam. If the existing options already "
        "cover these answers, write NONE.\n"
        "One option per line, each starting with \"- \", and nothing else.\n"
        "New options:" % (q["prompt"], existing, listed, k))
    num_ctx = ollama_ctx.fit_num_ctx(prompt, GROUP_TOKENS, where="mcq training")
    out = llm_judge.generate(prompt, max_tokens=GROUP_TOKENS, num_ctx=num_ctx,
                             timeout=timeout)
    have = {o["text"].lower().rstrip(".") for o in q["options"]}
    new = []
    for line in (out.get("response") or "").split("\n"):
        t = _clean_option(line)
        low = t.lower().rstrip(".")
        if (not t or low in have or low.startswith("none")
                or low in ("not stated", "new options", "options")):
            continue
        have.add(low)
        new.append(t)
    return new[:k]


def _slug(text, taken):
    base = (re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30]
            .strip("_") or "option")
    sid, k = base, 2
    while sid in taken:
        sid, k = "%s_%d" % (base, k), k + 1
    taken.add(sid)
    return sid


def _insert_option(q, text, added):
    """A new option, value 0, just before Not stated."""
    o = {"id": _slug(text, {x["id"] for x in q["options"]}), "text": text,
         "value": 0.0, "follow_up": [], "added": added}
    at = next((i for i, x in enumerate(q["options"])
               if x["id"] == NOT_MENTIONED), len(q["options"]))
    q["options"].insert(at, o)
    return o


def _reorder(q):
    """Strongest scam sign first, Not stated last; equal values keep their
    order."""
    nm = [o for o in q["options"] if o["id"] == NOT_MENTIONED]
    rest = [o for o in q["options"] if o["id"] != NOT_MENTIONED]
    rest.sort(key=lambda o: -o["value"])
    q["options"] = rest + nm


def _accuracy(results, labels):
    right = sum(1 for r, y in zip(results, labels)
                if r["verdict"] == ("scam" if y else "legit"))
    neutral = sum(1 for r in results if r["verdict"] == "neutral")
    return right / max(1, len(results)), neutral


def train_ontology(csv_path, onto, calls=TRAIN_ONTOLOGY_CALLS,
                   new_options=NEW_OPTIONS, prior=PRIOR_WEIGHT, seed=42,
                   quotes=True, log=print):
    """Train an ontology on a dataset; returns the trained tree (a new
    object - `onto` is not changed). See the comment above for the steps."""
    if not 2 <= calls <= 500:
        raise ValueError("train on between 2 and 500 calls")
    if not 0 <= new_options <= 5:
        raise ValueError("new options per question must be between 0 and 5")
    if prior < 0:
        raise ValueError("the prior weight cannot be negative")
    onto = json.loads(json.dumps(onto))             # a copy to change
    sample = sample_calls(csv_path, calls, seed)
    labels = [y for _, y in sample]
    n_scam = sum(labels)
    n_legit = len(sample) - n_scam
    log("  %d calls from %s (%d scam, %d not), seed %d"
        % (len(sample), csv_path, n_scam, n_legit, seed))
    log("  %d subjects, %d questions; quotes %s"
        % (len(onto["options"]), count_questions(onto),
           "required" if quotes else "off"))
    t0 = time.time()
    qmap = {path: q for path, q, _ in iter_questions(onto)}
    root = {"prompt": onto["prompt"], "options": onto["options"]}
    qmap["root"] = root

    # 1. walk every call
    log("\n  step 1 of 4: every call through the tree")
    presize([t for t, _ in sample], onto, quotes)
    results = []
    for i, ((text, y), res) in enumerate(zip(sample, classify_all(
            [t for t, _ in sample], onto, quotes)), 1):
        results.append(res)
        log("    %2d/%d  %-5s  %-20s score %+5.2f  %-7s (%d questions)"
            % (i, len(sample), "scam" if y else "legit",
               (res["subject"] or "-")[:20], res["score"], res["verdict"],
               res["requests"]))
    acc0, neu0 = _accuracy(results, labels)
    log("    on these calls: %.0f%% right, %d neutral" % (100 * acc0, neu0))

    # 2. new options from the calls that answered Not stated
    added = []
    if new_options:
        log("\n  step 2 of 4: new options, from the calls that answered "
            "Not stated")
        rng = random.Random(seed)
        stamp = {"dataset": str(csv_path), "date": time.strftime("%Y-%m-%d")}
        for path, q in list(qmap.items()):
            if path == "root" or not _scored(q):
                continue
            idle = [(ci, ai) for ci, r in enumerate(results)
                    for ai, a in enumerate(r["answers"])
                    if a["path"] == path and a["option"] == NOT_MENTIONED]
            if len(idle) < 2:
                continue
            if len(q["options"]) >= MAX_TREE_OPTIONS:
                continue
            probe = rng.sample(idle, min(PROBE_CALLS, len(idle)))
            heard = []
            for ci, _ in probe:
                a = short_answer(sample[ci][0], q["prompt"])
                if _substantive(a):
                    heard.append(a)
            if len(heard) < 2:
                continue
            room = min(new_options, MAX_TREE_OPTIONS - len(q["options"]))
            texts = propose_options(q, heard, room)
            if not texts:
                log("    %-40s %d open answers, all covered already"
                    % (path[:40], len(heard)))
                continue
            for t in texts:
                o = _insert_option(q, t, dict(stamp, heard=heard[:8]))
                added.append({"path": path, "id": o["id"], "text": t})
                log("    %-40s + %s" % (path[:40], t))
            # ask this question again of every call that was asked it
            for ci, r in enumerate(results):
                for ai, a in enumerate(r["answers"]):
                    if a["path"] != path:
                        continue
                    old = a["option"]
                    new = _answer_record(path, q, ask(
                        sample[ci][0], q, onto, quotes=quotes))
                    r["answers"][ai] = new
                    if new["option"] != old:
                        # the old option's follow-ups no longer apply
                        drop = "%s/%s/" % (path, old)
                        r["answers"] = [x for x in r["answers"]
                                        if not x["path"].startswith(drop)]
                    break
        if not added:
            log("    none: the options already cover what the calls said")

    # 3. new values
    log("\n  step 3 of 4: new values from the calls that chose each option "
        "(prior weight %g)" % prior)
    ws = len(sample) / (2.0 * n_scam) if n_scam else 0.0
    wl = len(sample) / (2.0 * n_legit) if n_legit else 0.0
    counts = Counter()
    for r, y in zip(results, labels):
        for a in r["answers"]:
            if a["option"] is not None:
                counts[(a["path"], a["option"], y)] += 1
    changed = []
    for path, q in qmap.items():
        if not _scored(q) or (path == "root" and not _root_scored(onto)):
            continue
        for o in q["options"]:
            if o["id"] == NOT_MENTIONED:
                continue
            s, l = counts[(path, o["id"], True)], counts[(path, o["id"], False)]
            if not s + l:
                continue
            before = o["value"]
            after = (prior * before + ws * s - wl * l) / (prior + ws * s + wl * l)
            after = round(max(-1.0, min(1.0, after)), 2)
            o["value"] = after
            o.setdefault("training", []).append(
                {"dataset": str(csv_path), "scam": s, "legit": l,
                 "before": before, "after": after})
            if after != before:
                changed.append({"path": path, "option": o["id"],
                                "before": before, "after": after,
                                "scam": s, "legit": l})
    for c in sorted(changed, key=lambda c: -abs(c["after"] - c["before"]))[:25]:
        log("    %-46s %+.2f -> %+.2f   (%d scam, %d legit)"
            % (("%s %s" % (c["path"], c["option"]))[:46], c["before"],
               c["after"], c["scam"], c["legit"]))
    if len(changed) > 25:
        log("    ... and %d more" % (len(changed) - 25))
    if not changed:
        log("    no value moved")

    # 4. put the options back in order
    for path, q in qmap.items():
        if path != "root" and _scored(q):
            _reorder(q)

    # the same answers, scored with the new values
    vals = {(path, o["id"]): o["value"] for path, q in qmap.items()
            for o in q["options"]}
    after = []
    for r in results:
        for a in r["answers"]:
            a["value"] = vals.get((a["path"], a["option"]), 0.0) \
                if a["option"] is not None else 0.0
        after.append(_summarise(r["answers"], next(
            (s for s in onto["options"] if s["id"] == r["subject"]), None)))
    acc1, neu1 = _accuracy(after, labels)
    log("\n  step 4 of 4: options put back in order, strongest scam sign "
        "first")
    log("\n  on the training calls: %.0f%% right before, %.0f%% after "
        "(%d -> %d neutral)" % (100 * acc0, 100 * acc1, neu0, neu1))
    log("  (these are the calls it learned from - score a held-out set for a "
        "fair number)")
    log("  %.0fs" % (time.time() - t0))

    onto.setdefault("training", []).append({
        "dataset": str(csv_path), "calls": len(sample), "scam": n_scam,
        "legit": n_legit, "seed": seed, "prior_weight": prior,
        "new_options_per_question": new_options, "quotes": quotes,
        "model": llm_judge.DEFAULT_MODEL,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "added": added, "changed": len(changed),
        "training_calls_right_before": round(acc0, 4),
        "training_calls_right_after": round(acc1, 4),
        "neutral_before": neu0, "neutral_after": neu1})
    return onto


# ----------------------------------------------------------------------- CLI

def _show_call(res, onto):
    """One classified call, as the CLI prints it."""
    print("%s\n  -> %s" % (onto["prompt"], res["subject_text"] or "unreadable"))
    total = res["answers"][0]["value"]
    for r in res["answers"][1:]:
        total += r["value"]
        mark = ("" if r["quoted"] is None else "  quote ok" if r["quoted"]
                else "  NO QUOTE FOUND - counted as not stated")
        depth = (len(r["path"].split("/")) - 2) // 2
        print("  %+.1f  =%+5.1f  %s%-40s %s%s"
              % (r["value"], total, "  " * depth, r["question"][:40 - 2 * depth],
                 (r["text"] or "unreadable")[:56], mark))
        if r["quote"]:
            print("                \"%s\"" % r["quote"][:90])
    print("score %+.2f  ->  %s   (%d questions)"
          % (res["score"], res["verdict"], res["requests"]))


def run_ask(args):
    onto = load_ontology(args.ontology)
    res = classify(args.text, onto, not args.no_quotes)
    _show_call(res, onto)
    if args.country:
        p = personalise(args.text, res,
                        load_country(find_country(args.country)),
                        not args.no_quotes)
        print("\npersonalised for a person in %s: %s" % (p["country"], p["law"]))
        for r in p["answers"]:
            print("  %-40s %s" % (r["question"][:40], r["text"] or "unreadable"))
        print("-> %s  (%s)" % (p["final"], p["why"]))
        if p["advice"]:
            print("   %s" % p["advice"])
    return 0


def run_question(args):
    if args.saved:
        res = ask_saved(args.text, args.saved)
    elif args.question:
        res = ask_question(args.text, args.question)
    else:
        raise ValueError("pass --question, or --saved with a trained question")
    print(res["question"])
    if res["mode"] == "free":
        print(res["answer"])
        return 0
    for o in res["options"]:
        print("  %s %5.1f%%  %s%s" % (o["letter"], 100 * o["p"], o["text"],
                                      "  <-" if o["letter"] == (
                                          res["choice"] is not None
                                          and LETTERS[res["choice"]]) else ""))
    if res.get("scam_lean") is not None:
        print("  calls in training that gave these answers: %.0f%% scam"
              % (100 * res["scam_lean"]))
    return 0 if res["choice"] is not None else 1


def run_train_question(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit("%s already exists - pass --force to replace it" % out)
    text = (Path(args.question_file).read_text(encoding="utf-8")
            if args.question_file else args.question)
    if not text:
        raise ValueError("pass --question or --question-file")
    print("==> train a question on %s" % args.csv)
    q = train_question(args.csv, text, args.calls, args.options, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
    print("  ok wrote %s" % out)
    return 0


def run_evaluate(args):
    """Score a whole dataset: what the MCQ page's Score a dataset tab runs."""
    import eval_common as EC
    onto = load_ontology(args.ontology)
    quotes = not args.no_quotes
    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    print()
    print("==> score  %s over %d calls" % (args.ontology, len(rows)))
    print("  %s  (%d subjects, %d questions; quotes %s)"
          % (onto["prompt"], len(onto["options"]), count_questions(onto),
             "required" if quotes else "off"))
    trained = [t.get("dataset") for t in onto.get("training") or []]
    if any(t and Path(t).name == Path(args.csv).name for t in trained):
        print("  NOTE this ontology was trained on this dataset: some of the "
              "calls scored here\n       set its options and values - a "
              "held-out set is the fairer read.")
    ctx = presize([r["text"] for r in rows], onto, quotes)
    print("  about a dozen requests per call, context window %s" % ctx)

    truths = [r["label"] for r in rows]
    prog = EC.Progress(len(rows), every=1)
    preds, results = [], []
    stopped = None
    t0 = time.time()
    try:
        for r, res in zip(rows, classify_all([r["text"] for r in rows], onto,
                                             quotes)):
            results.append(res)
            # neutral (a score of exactly 0) is not called a scam
            pred = int(res["verdict"] == "scam")
            preds.append(pred)
            prog.tick(r, pred)
    except RuntimeError as e:
        print("    ERROR %s" % e)
    except EC.Stopped:
        stopped = EC.stopped_after(len(preds), len(rows))
    elapsed = time.time() - t0
    rows = rows[:len(preds)]
    truths = truths[:len(preds)]
    results = results[:len(preds)]

    m = EC.metrics(truths, preds)
    base = EC.baselines(truths)
    neutral = sum(1 for r in results if r["verdict"] == "neutral")
    asked = sum(r["requests"] for r in results)
    unquoted = sum(1 for r in results for a in r["answers"]
                   if a["quoted"] is False)
    extra = ["%d of %d calls scored exactly 0 (neutral), counted as not scam"
             % (neutral, len(results)),
             "%d questions asked, %d answered from logprobs, %d answers "
             "without a quote counted as not stated"
             % (asked, sum(r["measured"] for r in results), unquoted)]
    extra += [""] + subject_table(onto, results, truths)
    EC.report(m, base, elapsed, len(preds), extra)
    if args.out:
        subj = Counter((r["subject"], bool(t)) for r, t in zip(results, truths))
        EC.write_results(args.out, Path(args.ontology).name, args.csv, rows,
                         preds, m, base, elapsed,
                         {"category": [r["subject"] or "" for r in results],
                          "score": [r["score"] for r in results],
                          "verdict3": [r["verdict"] for r in results],
                          "prob_scam": [round(pct(r["score"]) / 100, 4)
                                        for r in results],
                          "reason": [explain(r) for r in results]},
                         {"kind": "mcq", "ontology": str(args.ontology),
                          "question": onto["prompt"],
                          "trained_on": trained,
                          "options": [{"letter": LETTERS[i], "id": s["id"],
                                       "scam": subj[(s["id"], True)],
                                       "legit": subj[(s["id"], False)]}
                                      for i, s in enumerate(onto["options"])],
                          "neutral": neutral, "questions_asked": asked,
                          "unquoted": unquoted, "quotes": quotes,
                          "stopped": stopped,
                          "measured": sum(r["measured"] for r in results)})
    return m


def dumps(obj, width=180, indent=0):
    """JSON for an ontology file, easy to read: anything that fits on one
    line (an option, an ask list) is written on one line, the rest is
    indented by two."""
    flat = json.dumps(obj, ensure_ascii=False)
    if len(flat) + indent <= width or not isinstance(obj, (dict, list)) \
            or not obj:
        return flat
    pad = " " * (indent + 2)
    if isinstance(obj, list):
        items = [pad + dumps(v, width, indent + 2) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    items = ["%s%s: %s" % (pad, json.dumps(k, ensure_ascii=False),
                           dumps(v, width, indent + 2)) for k, v in obj.items()]
    return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"


# what an ontology file keeps: the questionnaire, and one line per training run
_Q_KEYS = ("id", "role", "prompt", "options")
_O_KEYS = ("id", "text", "value", "absence", "follow_up")
_RUN_KEYS = ("dataset", "calls", "date", "added", "changed",
             "training_calls_right_before", "training_calls_right_after")


def to_file(onto):
    """The tree as it is written to disk: only the questionnaire - questions,
    options, values, ask orders - and a one-line summary of each training
    run. Per-option training history stays in the run's log."""
    def q_out(q):
        out = {k: q[k] for k in _Q_KEYS if k in q}
        out["options"] = [o_out(o) for o in q["options"]]
        return out

    def o_out(o):
        out = {k: o[k] for k in _O_KEYS if k in o}
        if o.get("follow_up"):
            out["follow_up"] = [q_out(f) for f in o["follow_up"]]
        else:
            out.pop("follow_up", None)
        return out
    subs = []
    for s in onto["options"]:
        n = {"id": s["id"], "text": s["text"]}
        if s.get("value"):
            n["value"] = s["value"]
        if s.get("ask"):
            n["ask"] = s["ask"]
        n["questions"] = [q_out(q) for q in s.get("questions", [])]
        subs.append(n)
    out = {"prompt": onto["prompt"], "options": subs,
           "common_questions": [q_out(q) for q in onto["common_questions"]]}
    runs = []
    for t in onto.get("training") or []:
        r = {k: t[k] for k in _RUN_KEYS if k in t}
        if isinstance(r.get("added"), list):
            r["added"] = len(r["added"])
        runs.append(r)
    if runs:
        out["training"] = runs
    return out


def run_train(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit("%s already exists - pass --force to replace it" % out)
    onto = load_ontology(args.ontology)
    print("==> train %s on %s" % (args.ontology, args.csv))
    trained = train_ontology(args.csv, onto, args.calls, args.new_options,
                             args.prior, args.seed, not args.no_quotes)
    trained.setdefault("trained_from", str(args.ontology))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(to_file(trained)) + "\n", encoding="utf-8")
    t = trained["training"][-1]
    print("  %d options added, %d values changed" % (len(t["added"]),
                                                     t["changed"]))
    print("  ok wrote %s" % out)
    return 0


def run_build_gone():
    raise SystemExit(
        "ERROR `build` is gone: the ontology is a tree of questions now, and "
        "`train` adds options and updates values from a dataset instead -\n"
        "      python scripts/mcq_ontology.py train --csv <dataset> --out "
        "knowledge/<name>.json\n"
        "If this came from the web page, its server is still running the old "
        "code: restart web_ui.sh and reload the page.")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def walk_flags(p):
        p.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
        p.add_argument("--no-quotes", action="store_true",
                       help="do not require a supporting quote for each "
                            "answer (faster: one letter per question)")

    a = sub.add_parser("ask", help="one transcript through the tree")
    a.add_argument("--text", required=True)
    walk_flags(a)
    a.add_argument("--country", default=None,
                   help="where the person lives: a country name (\"New "
                        "Zealand\") or file in knowledge/countries/, whose "
                        "law personalises the label")
    a.set_defaults(func=run_ask)

    qq = sub.add_parser("question", help="your own question about one call")
    qq.add_argument("--text", required=True)
    qq.add_argument("--question", default=None,
                    help='list options to make it multiple choice: '
                         '"... A) yes B) no", or "options: yes / no"')
    qq.add_argument("--saved", default=None,
                    help="a trained question's JSON (knowledge/questions/...): "
                         "its options are used")
    qq.set_defaults(func=run_question)

    tq = sub.add_parser("train-question",
                        help="options for your question, from a dataset")
    tq.add_argument("--csv", required=True)
    tq.add_argument("--question", default=None)
    tq.add_argument("--question-file", default=None,
                    help="read the question from this file instead")
    tq.add_argument("--out", required=True,
                    help="knowledge/questions/<name>.json")
    tq.add_argument("--calls", type=int, default=TRAIN_CALLS,
                    help="calls to train on, half scam half not "
                         "(default %d)" % TRAIN_CALLS)
    tq.add_argument("--options", type=int, default=TRAIN_OPTIONS,
                    help="at most this many options (default %d)"
                         % TRAIN_OPTIONS)
    tq.add_argument("--seed", type=int, default=42)
    tq.add_argument("--force", action="store_true")
    tq.set_defaults(func=run_train_question)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    ev.add_argument("--csv", required=True)
    walk_flags(ev)
    ev.add_argument("--limit", type=int, default=None,
                    help="a class-balanced head of the dataset")
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    t = sub.add_parser("train", help="add options and update values from a "
                                     "dataset")
    t.add_argument("--csv", required=True,
                   help="a training set - a train fold, never the calls you "
                        "will score")
    t.add_argument("--out", required=True)
    walk_flags(t)
    t.add_argument("--calls", type=int, default=TRAIN_ONTOLOGY_CALLS,
                   help="calls to train on, half scam half not (default %d)"
                        % TRAIN_ONTOLOGY_CALLS)
    t.add_argument("--new-options", type=int, default=NEW_OPTIONS,
                   help="at most this many new options per question "
                        "(default %d; 0 updates the values only)" % NEW_OPTIONS)
    t.add_argument("--prior", type=float, default=PRIOR_WEIGHT,
                   help="how many calls' worth the old value counts for "
                        "(default %g)" % PRIOR_WEIGHT)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=run_train)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # `build` was replaced by `train`. A web page still running the old code
    # calls it, so say what to do rather than leave argparse to say
    # "invalid choice".
    if argv[:1] == ["build"]:
        run_build_gone()
    args = build_parser().parse_args(argv)
    try:
        out = args.func(args)
    except ValueError as e:
        raise SystemExit("ERROR %s" % e)
    return 0 if out is None or isinstance(out, dict) else out


if __name__ == "__main__":
    sys.exit(main())
