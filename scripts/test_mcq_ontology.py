#!/usr/bin/env python3
"""
Offline tests for mcq_ontology: the MCQ ontology LLM, one question - the
call's category - out of a JSON file.

No Ollama: llm_judge._post is replaced by a fake that answers the way Ollama's
/api/generate does with logprobs on (and, for the fallback, the way an older
Ollama does without them). Nothing here touches a knowledge base, a vector
database or the web.

    python scripts/test_mcq_ontology.py
"""
import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_judge                                               # noqa: E402
import mcq_ontology as M                                       # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-66s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


ONTO = {"prompt": "Which category does this call belong to?",
        "options": [{"id": "ssn", "text": "Social Security problem", "verdict": "scam"},
                    {"id": "delivery", "text": "Delivery update", "verdict": "legit"},
                    {"id": "refund", "text": "Refund owed", "verdict": "scam"},
                    {"id": "wrong", "text": "Wrong number", "verdict": "legit"}]}

# what the fake model believes about a call, keyed on a word in it:
# probability on each letter A-D
BELIEF = {"SSNCALL": [0.85, 0.05, 0.08, 0.02],
          "PARCEL": [0.02, 0.90, 0.03, 0.05],
          "SPLIT": [0.40, 0.35, 0.00, 0.25],     # top pick scam, total legit
          "GIBBERISH": None}
STATE = {"logprobs": True, "calls": 0, "payloads": [], "lead": ""}


def fake_post(path, payload, timeout):
    STATE["calls"] += 1
    STATE["payloads"].append(payload)
    prompt = payload["prompt"]
    if prompt.rstrip().endswith("Category:"):          # build --describe
        return {"response": " People calling about it\nsecond line"}
    if prompt.rstrip().endswith("Short answer:"):       # training, step 1
        tr = prompt.split("Transcript:\n", 1)[1]
        return {"response": " Their SSN.\nmore" if "SSNCALL" in tr
                else " A delivery time" if "PARCEL" in tr else " not said"}
    if prompt.rstrip().endswith("Options:"):            # training, step 2
        if STATE.get("group_bad"):
            return {"response": " I cannot group these."}
        return {"response": " - SSN (3)\n- A delivery time\n- not said\n"
                            "- a delivery time"}
    if "\nQuestion: " in prompt:                        # an open question
        return {"response": " They want the person's bank details.",
                "done_reason": "stop"}
    belief = next(v for k, v in BELIEF.items() if k in prompt)
    if belief is None:
        return {"response": "The", "logprobs": [{"token": "The", "logprob": 0.0,
                "top_logprobs": [{"token": "The", "logprob": 0.0}]}]}
    alts = [{"token": (" " if i == 1 else "") + "ABCD"[i], "logprob": math.log(p)}
            for i, p in enumerate(belief) if p > 0]
    alts.append({"token": "The", "logprob": math.log(0.01)})
    top = max(alts, key=lambda a: a["logprob"])
    steps = []
    if STATE["lead"]:                    # a token before the letter
        steps.append({"token": STATE["lead"], "logprob": -0.1,
                      "top_logprobs": [{"token": STATE["lead"], "logprob": -0.1}]})
    steps.append({"token": top["token"], "logprob": top["logprob"],
                  "top_logprobs": alts})
    text = STATE["lead"] + top["token"]
    if not STATE["logprobs"]:
        return {"response": text}
    return {"response": text, "logprobs": steps}


llm_judge._post = fake_post
tmp = tempfile.mkdtemp()

print("\nthe file")
check("a good ontology has no problems", M.check_ontology(ONTO), [])
bad = dict(ONTO, options=[dict(o, verdict="maybe") if i == 0 else o
                          for i, o in enumerate(ONTO["options"])])
check("a verdict that is not scam/legit is named",
      any("verdict" in p for p in M.check_ontology(bad)))
check("all-scam options are refused (every call would get one verdict)",
      any("one option must be scam and one legit" in p for p in M.check_ontology(
          dict(ONTO, options=[dict(o, verdict="scam") for o in ONTO["options"]]))))
check("one option is not a question",
      any("between 2 and" in p for p in M.check_ontology(
          dict(ONTO, options=ONTO["options"][:1]))))
check("13 options is too many",
      any("between 2 and" in p for p in M.check_ontology(
          dict(ONTO, options=ONTO["options"] * 4))))
check("no prompt is named", any("prompt" in p for p in
                                M.check_ontology(dict(ONTO, prompt=" "))))
path = os.path.join(tmp, "onto.json")
open(path, "w").write(json.dumps(ONTO))
onto = M.load_ontology(path)
check("load_ontology reads it", (onto["prompt"], len(onto["options"])),
      (ONTO["prompt"], 4))
open(os.path.join(tmp, "broken.json"), "w").write('{"prompt": "x", ')
try:
    M.load_ontology(os.path.join(tmp, "broken.json"))
    check("broken JSON is refused", False)
except ValueError as e:
    check("broken JSON is refused, with where", "line" in str(e))
check("the shipped ontology loads", len(M.load_ontology()["options"]) >= 2)
check("other knowledge files are not mistaken for one",
      M.is_mcq_ontology(os.path.join(HERE, "..", "knowledge", "scam_ontology.json")),
      False)

print("\nthe question")
q = M.build_prompt("SSNCALL transcript", onto)
check("letters the options A-D", "\nA - Social Security problem\n" in q
      and "\nD - Wrong number\n" in q)
check("asks for one letter, A to D", q.rstrip().endswith(
      "Answer with exactly one letter, A to D.\nAnswer:"))
check("the transcript alone - no retrieval", "SSNCALL transcript" in q
      and "REFERENCE" not in q)
for tok, n, want in (("A", 4, "A"), (" B", 4, "B"), ("**C", 4, "C"),
                     ("D)", 4, "D"), ("E", 4, None), ("The", 4, None),
                     ("a", 4, "A")):
    check("token %r among %d options reads as %r" % (tok, n, want),
          M.answer_letter(tok, n), want)

print("\njudging one call, from logprobs")
STATE["calls"] = 0
r = M.judge("SSNCALL transcript", onto)
check("one request", STATE["calls"], 1)
check("measured from logprobs", r["how"], "logprobs")
check("the most likely category", (r["choice"], r["category"]), (0, "ssn"))
check("P(scam) is the total on the scam options (0.85 + 0.08)",
      round(r["p_scam"], 3), 0.93)
check("verdict Fraud", r["verdict"], "Fraud")
check("probabilities sum to 1", round(sum(r["probs"]), 6), 1.0)
r = M.judge("PARCEL transcript", onto)
check("a delivery call is legitimate", (r["category"], r["verdict"]),
      ("delivery", "Normal"))
r = M.judge("SPLIT transcript", onto)
check("the top category can be scam while the total is legit",
      (r["category"], round(r["p_scam"], 2), r["verdict"]),
      ("ssn", 0.4, "Normal"))
pl = STATE["payloads"][-1]
check("a few tokens at temperature 0, logprobs asked for",
      (pl["options"]["num_predict"], pl["options"]["temperature"],
       pl["logprobs"]), (M.ANSWER_TOKENS, 0.0, True))
STATE["lead"] = " "
r = M.judge("SSNCALL transcript", onto)
check("a token before the letter is skipped", (r["category"], r["how"]),
      ("ssn", "logprobs"))
STATE["lead"] = ""
r = M.judge("GIBBERISH transcript", onto)
check("no letter at all is unreadable", (r["verdict"], r["how"], r["p_scam"]),
      (None, "unreadable", None))

print("\nan Ollama without logprobs")
STATE["logprobs"] = False
M._LOGPROBS_OK = None
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = M.judge("PARCEL transcript", onto)
    r2 = M.judge("SSNCALL transcript", onto)
check("the answered letter stands in", (r["category"], r["how"], r["p_scam"]),
      ("delivery", "letter", 0.0))
check("and P(scam) is 1 for a scam letter", r2["p_scam"], 1.0)
check("it says so, once", buf.getvalue().count("no logprobs"), 1)
STATE["logprobs"] = True
M._LOGPROBS_OK = None

print("\nyour own question")
for q, want in (
        ("What does the caller want?", ("What does the caller want?", [])),
        ("Who are they? A) a bank B) a government office C) not said",
         ("Who are they?", ["a bank", "a government office", "not said"])),
        ("Who are they?\nA) a bank\nB) a shop", ("Who are they?", ["a bank", "a shop"])),
        ("Urgent?\n1. yes\n2. no", ("Urgent?", ["yes", "no"])),
        ("How urgent? options: very / a little / not at all",
         ("How urgent?", ["very", "a little", "not at all"])),
        ("Did they say a) the bank?", ("Did they say a) the bank?", [])),
        ("Is this a scam or legit?", ("Is this a scam or legit?", []))):
    check("parse %r" % q[:34], M.parse_options(q), want)
STATE["calls"] = 0
r = M.ask_question("SSNCALL transcript",
                   "Which is it? A) social security B) delivery C) refund D) wrong")
check("options in the question: one request, multiple choice",
      (STATE["calls"], r["mode"], r["how"]), (1, "options", "logprobs"))
check("the probabilities come back per option, the pick is the top one",
      (r["choice"], [round(o["p"], 2) for o in r["options"]]),
      (0, [0.85, 0.05, 0.08, 0.02]))
check("the options are put to the model lettered A-D",
      "\nA - social security\nB - delivery\n" in STATE["payloads"][-1]["prompt"])
r = M.ask_question("SSNCALL transcript", "What does the caller want?")
check("no options: the model answers in its own words",
      (r["mode"], r["answer"]), ("free", "They want the person's bank details."))
check("the open question invites its own knowledge",
      "your own knowledge" in STATE["payloads"][-1]["prompt"])
try:
    M.ask_question("x", "Which?\n" + "\n".join("%d. option %d" % (i, i)
                                                for i in range(1, 14)))
    check("13 options are refused", False)
except ValueError as e:
    check("13 options are refused", "at most 12" in str(e))

print("\noptions from a dataset's categories")
ds = os.path.join(tmp, "ds.csv")
with open(ds, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "type", "text"])
    n = 0
    for cat, lab, k in (("ssn", "scam", 5), ("delivery", "nonscam", 4),
                        ("refund", "scam", 3), ("wrong", "nonscam", 2),
                        ("mixed_topic", "scam", 1), ("mixed_topic", "nonscam", 1)):
        for _ in range(k):
            n += 1
            w.writerow([n, lab, cat, "a %s call number %d" % (cat, n)])
    w.writerow([n + 1, "scam", "", "a call with no category"])
b = M.build(ds)
ids = [o["id"] for o in b["options"]]
check("one option per category", sorted(ids),
      sorted(["ssn", "delivery", "refund", "wrong", "mixed_topic"]))
check("verdict = the label most of its calls carry",
      {o["id"]: o["verdict"] for o in b["options"]}["delivery"], "legit")
check("an evenly split category is marked mixed and leans scam",
      [(o["verdict"], o.get("mixed")) for o in b["options"]
       if o["id"] == "mixed_topic"], [("scam", True)])
check("scam and legitimate options alternate",
      [o["verdict"] for o in b["options"]][:4], ["scam", "legit", "scam", "legit"])
check("the calls with no category are counted, not used",
      b["built_from"]["no_category"], 1)
check("what it builds is a valid ontology", M.check_ontology(b), [])
small = M.build(ds, max_options=3)
check("past max_options the smallest merge into 'Something else'",
      ([o["id"] for o in small["options"]].count("other"),
       next(o["text"] for o in small["options"] if o["id"] == "other")),
      (1, "Something else"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    d = M.build(ds, describe_with_model=True)
check("--describe has the model write each option's text",
      next(o["text"] for o in d["options"] if o["id"] == "ssn"),
      "SSN: People calling about it")
nocat = os.path.join(tmp, "nocat.csv")
with open(nocat, "w", newline="") as f:
    csv.writer(f).writerows([["id", "label", "text"], [1, "scam", "x"]])
try:
    M.build(nocat)
    check("a dataset without a category column is refused", False)
except ValueError as e:
    check("a dataset without a category column is refused",
          "no category column" in str(e))
even = os.path.join(tmp, "even.csv")
with open(even, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "topic", "text"])
    for i, (t, lab) in enumerate((("a", "scam"), ("a", "nonscam"),
                                  ("b", "scam"), ("b", "nonscam"))):
        w.writerow([i, lab, t, "call %d" % i])
try:
    M.build(even)
    check("categories that say nothing about the label are refused", False)
except ValueError as e:
    check("categories that say nothing about the label are refused",
          "exactly as many scam calls as legitimate" in str(e)
          and "paired with one legitimate call on the same topic" in str(e))
# Paired-196's shape: many topics of one scam + one legitimate call, and a
# couple left unpaired. Too many topics, so the smallest would be merged into
# "other" - the refusal has to come before that, and point to training
paired = os.path.join(tmp, "paired.csv")
with open(paired, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "topic", "text"])
    for i in range(20):
        w.writerow([2 * i, "scam", "t%02d" % i, "call"])
        w.writerow([2 * i + 1, "nonscam", "t%02d" % i, "call"])
    w.writerow([99, "scam", "lone", "call"])
try:
    M.build(paired)
    check("a topic-paired dataset is refused before merging", False)
except ValueError as e:
    check("a topic-paired dataset is refused before merging, pointing to "
          "training", ("20 of the 21 values" in str(e),
                       "train it on this dataset" in str(e)), (True, True))

print("\nscoring a dataset (the page's Score a dataset)")
out = os.path.join(tmp, "run.metrics.json")
ev = os.path.join(tmp, "ev.csv")
with open(ev, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    w.writerows([[1, "scam", "SSNCALL one"], [2, "nonscam", "PARCEL two"],
                 [3, "scam", "SPLIT three"], [4, "nonscam", "PARCEL four"]])
import eval_common as EC                                       # noqa: E402
EC.RESULTS_DIR = tmp
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M.main(["evaluate", "--csv", ev, "--ontology", path, "--out", out])
res = json.load(open(out))
check("metrics.json carries the MCQ details",
      (res["kind"], res["question"], len(res["options"]), res["measured"]),
      ("mcq", ONTO["prompt"], 4, 4))
check("accuracy over the calls (SPLIT is a missed scam)",
      (res["metrics"]["tp"], res["metrics"]["fn"], res["metrics"]["tn"]),
      (1, 1, 2))
check("the category table counts where each call went",
      [(o["id"], o["scam"], o["legit"]) for o in res["options"]],
      [("ssn", 2, 0), ("delivery", 0, 2), ("refund", 0, 0), ("wrong", 0, 0)])

print("\nthe benchmark runner")
import evaluate_mcq_ontology as E                              # noqa: E402
bench_csv = os.path.join(tmp, "bench.csv")
argv = sys.argv
sys.argv = ["evaluate_mcq_ontology.py", "--csv", ev, "--ontology", path,
            "--out", bench_csv, "--system", "mcq_ontology__stripped"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    E.main()
sys.argv = argv
log = buf.getvalue()
check("prints the metric line collect_results reads",
      "  mcq_ontology             acc  75.0%" in log)
rows = list(csv.DictReader(open(bench_csv)))
check("per-call CSV columns for the Scores and Per-call tabs",
      list(rows[0].keys()),
      ["idx", "true", "text", "mcq_ontology__stripped",
       "mcq_ontology__stripped_pct", "mcq_ontology__stripped_category",
       "mcq_ontology__stripped_why"])
check("the order is the benchmark's shuffled order (seed 42)",
      [r["text"] for r in rows], [t for t, _ in E.load(ev)])

print("\na question trained on a dataset")
tds = os.path.join(tmp, "train.csv")
with open(tds, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    for i in range(4):
        w.writerow([i, "scam", "SSNCALL number %d" % i])
        w.writerow([10 + i, "nonscam", "PARCEL number %d" % i])
check("the sample is balanced", sorted(y for _, y in M.sample_calls(tds, 6)),
      [False] * 3 + [True] * 3)
check("the sample is the same each time (seeded)",
      M.sample_calls(tds, 6), M.sample_calls(tds, 6))
STATE["calls"] = 0
STATE["payloads"] = []
logged = []
q = M.train_question(tds, "What does the caller ask for?", calls=6,
                     max_options=4, log=logged.append)
check("2n+1 requests: an answer per call, a grouping, a choice per call",
      STATE["calls"], 13)
check("short answers are asked for in a few words",
      "at most 12" in STATE["payloads"][0]["prompt"]
      and STATE["payloads"][0]["options"]["num_predict"] == M.SHORT_ANSWER_TOKENS)
check("the grouping prompt sees every answer and the option limit",
      "6. " in STATE["payloads"][6]["prompt"]
      and "at most 4 options" in STATE["payloads"][6]["prompt"])
check("options: counts stripped, duplicates dropped",
      [o["text"] for o in q["options"]], ["SSN", "A delivery time", "not said"])
check("scam and legitimate calls counted on each option",
      [o["calls"] for o in q["options"]],
      [{"scam": 3, "legit": 0}, {"scam": 0, "legit": 3},
       {"scam": 0, "legit": 0}])
check("what it saves is a valid saved question", M.check_question(q), [])
check("the answers it grouped are kept, with the label and pick",
      sorted((a["label"], a["answer"], a["option"]) for a in q["answers"])[0],
      ("legit", "A delivery time", "B"))
check("where it came from is recorded",
      (q["trained_on"]["dataset"], q["trained_on"]["calls"],
       q["trained_on"]["scam"], q["options_from"]),
      (tds, 6, 3, "grouped by the model"))

STATE["calls"] = 0
q2 = M.train_question(tds, "Who is it? A) a government office B) a courier",
                      calls=6, log=lambda *_: None)
check("a question with its own options keeps them: n requests",
      (STATE["calls"], [o["text"] for o in q2["options"]], q2["answers"][0]["answer"]),
      (6, ["a government office", "a courier"], None))
STATE["group_bad"] = True
q3 = M.train_question(tds, "What does the caller ask for?", calls=6,
                      log=lambda *_: None)
STATE["group_bad"] = False
check("an unusable grouping falls back to the most common answers",
      (sorted(o["text"] for o in q3["options"]), q3["options_from"][:15]),
      (["A delivery time", "Their SSN"], "the most common"))
same = os.path.join(tmp, "same.csv")
with open(same, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    for i in range(4):
        w.writerow([i, "scam" if i % 2 else "nonscam", "SSNCALL %d" % i])
STATE["group_bad"] = True
try:
    M.train_question(same, "What?", calls=4, log=lambda *_: None)
    check("one answer for every call is refused", False)
except ValueError as e:
    check("one answer for every call is refused", "same answer" in str(e))
STATE["group_bad"] = False

M.QUESTIONS_DIR = type(M.QUESTIONS_DIR)(tmp) / "questions"
out = M.QUESTIONS_DIR / "asks.json"
argv, sys.argv = sys.argv, ["x"]
with contextlib.redirect_stdout(io.StringIO()):
    rc = M.main(["train-question", "--csv", tds, "--question",
                 "What does the caller ask for?", "--calls", "6",
                 "--out", str(out)])
sys.argv = argv
check("train-question writes the file", (rc, out.exists()), (0, True))
lst = M.list_questions()
check("it is listed as a saved question",
      [(d["name"], d["options"], d["calls"]) for d in lst], [("asks.json", 3, 6)])
saved = M.load_question(out)
check("loading works out each option's scam share",
      [o["scam_share"] for o in saved["options"]], [1.0, 0.0, None])
STATE["calls"] = 0
r = M.ask_saved("SSNCALL transcript", out)
check("asking it: one request, multiple choice over the file's options",
      (STATE["calls"], r["mode"], [o["text"] for o in r["options"]]),
      (1, "options", ["SSN", "A delivery time", "not said"]))
check("the saved question's own text is asked, the options lettered",
      "What does the caller ask for?\nA - SSN\nB - A delivery time\nC - not said\n"
      in STATE["payloads"][-1]["prompt"])
check("the pick, with the training counts alongside",
      (r["choice"], r["options"][0]["calls"]), (0, {"scam": 3, "legit": 0}))
check("scam lean: the answer weighted by each option's scam share",
      round(r["scam_lean"], 3), round(0.85 / 0.90, 3))
for bad, why in (({"prompt": "x", "options": []}, "kind"),
                 ({"kind": "question", "prompt": "x", "options": [{"text": "a"}]},
                  "2 to 12"),
                 ({"kind": "question", "prompt": "", "options": [{"text": "a"},
                                                                 {"text": "b"}]},
                  "prompt")):
    check("a broken saved question is refused (%s)" % why,
          any(why in p for p in M.check_question(bad)))
check("an ontology is not mistaken for a saved question",
      bool(M.check_question(ONTO)))

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all good - one question, the category, measured"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
