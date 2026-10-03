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
          "split evenly" in str(e))

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

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all good - one question, the category, measured"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
