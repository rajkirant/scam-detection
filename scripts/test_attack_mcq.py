#!/usr/bin/env python3
"""
Offline tests for attack_mcq: no Ollama. The attacker, the goal check and
the tree are replaced by fakes, so only the loop, the counting and the files
are tested here.

    python scripts/test_attack_mcq.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_mcq as A                                         # noqa: E402
import mcq_ontology as M                                       # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-66s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


FILL = " ".join("receiver: okay i see\ncaller: thank you for waiting on the line"
                for _ in range(8))
ASK = "caller: please pay the bill now into this account"
LOUD = "caller: this is urgent act immediately\n%s\n%s" % (ASK, FILL)
QUIET = "caller: hello this is a routine call\n%s\n%s" % (ASK, FILL)
HONEST = "caller: hello this is a routine call\n%s" % FILL
prompts = []
plan = {}


def fake_llm(prompt, max_tokens, temperature=0.0, seed=None, model=None,
             timeout=600):
    prompts.append(prompt)
    if prompt.startswith("Read this phone call"):          # the goal check
        call = prompt.split("Transcript:\n")[1].split("\n\nQuestion")[0]
        if "pay the bill now" in call:
            return "YES\nQuote: please pay the bill now into this account"
        return "NO\nQuote: none"
    if "Your last call" not in prompt:
        return LOUD
    return plan["rewrite"]


def fake_classify(call, onto, quotes=True, timeout=300):
    score = 2.0 if "urgent" in call else plan.get("quiet_score", -1.0)
    return {"subject": "utility", "subject_text": "Electricity, gas, or water supply",
            "score": score, "verdict": M.verdict_of(score), "requests": 2,
            "answers": [
                {"root": True, "path": "root", "question": "?", "text": "x",
                 "option": "utility", "value": 0.0, "quote": None, "quoted": None},
                {"root": False, "path": "common/pressure", "question": "Pressure?",
                 "text": "Act now" if score > 0 else "No hurry",
                 "option": "now" if score > 0 else "no_hurry", "value": score,
                 "quote": "urgent" if score > 0 else "routine call",
                 "quoted": True}]}


A.llm = fake_llm
A.MCQ.classify = fake_classify
tmp = tempfile.mkdtemp()


def run(*extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.main(["--only", "utility_pay_now", "--variants", "2", "--rounds", "3",
                "--out", tmp] + list(extra))
    return buf.getvalue(), A.load_attempts(os.path.join(tmp, "attempts.jsonl"))


print("the loop")
plan.update(rewrite=QUIET)
out, att = run()
check("round 0 is caught, round 1 evades, and the variant stops there",
      [(a["variant"], a["round"], a["verdict"], a["goal_kept"]) for a in att],
      [(0, 0, "scam", True), (0, 1, "legit", True),
       (1, 0, "scam", True), (1, 1, "legit", True)])
row = A.summarise(att)[0]
check("summary: both variants evaded, in one rewrite, from a score of +2",
      (row["evaded"], row["caught"], row["rounds_to_evade_median"],
       row["naive_score_median"], row["naive_caught"]), (2, 0, 1, 2.0, 2))
check("the answers behind the evasions are listed",
      A.exploited(att)[0], (("common/pressure", "no_hurry", -1.0), 2))
check("white box: the rewrite prompt shows the tree and the answers",
      any("You know how the detector works" in p and "Supply will be cut off"
          in p and "Pressure? -> Act now" in p for p in prompts))
n = len(prompts)
out, att2 = run()
check("a finished run is not run again", (len(prompts), len(att2)), (n, 4))

print("\na call that drops the goal does not count")
shutil.rmtree(tmp); os.mkdir(tmp)
plan.update(rewrite=HONEST)
out, att = run()
row = A.summarise(att)[0]
check("an honest rewrite scores below 0 but is not an evasion",
      (row["evaded"], row["caught"], row["goal_dropped"], len(att)),
      (0, 2, 6, 8))
check("the attacker is told why", any("no longer clearly asks" in p
                                      for p in prompts))

print("\nneutral, black box, malformed")
shutil.rmtree(tmp); os.mkdir(tmp)
plan.update(rewrite=QUIET, quiet_score=0.0)
out, att = run()
check("a score of 0 with the goal kept is neutral, and the rounds go on",
      (A.summarise(att)[0]["neutral"], len(att)), (2, 8))
shutil.rmtree(tmp); os.mkdir(tmp)
plan.update(quiet_score=-1.0)
del prompts[:]
out, att = run("--knowledge", "score")
check("black box: no tree and no answers in the prompt, the score is",
      (any("You know how the detector works" in p or "Pressure?" in p
           for p in prompts),
       any("verdict scam, score +2.00" in p for p in prompts)), (False, True))
shutil.rmtree(tmp); os.mkdir(tmp)
plan.update(rewrite="Sure! Here is a better call.")
out, att = run()
check("a reply that is not a transcript is not scored",
      (att[1]["malformed"] != "", att[1]["score"], A.outcome(att[:4])),
      (True, None, "caught"))

print("\na run cut off half way, and replay")
shutil.rmtree(tmp); os.mkdir(tmp)
plan.update(rewrite=QUIET)
out, att = run()
with open(os.path.join(tmp, "attempts.jsonl"), "w") as f:
    for a in att[:3]:                       # variant 1 lost its last call
        f.write(json.dumps(a) + "\n")
out, att = run()
check("the cut-off variant is written again from its start",
      [(a["variant"], a["round"]) for a in att], [(0, 0), (0, 1), (1, 0), (1, 1)])
plan.update(quiet_score=1.5)                # the "repaired" tree
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    A.main(["--replay", tmp])
check("replay: the two evasions are caught by the repaired tree",
      "was legit    -> now scam 2, neutral 0, legit 0" in buf.getvalue())
check("every scenario names a subject of the released tree",
      {s[1] for s in A.SCENARIOS} <= {o["id"] for o in M.load_ontology()["options"]})

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all good - the attack loop, its counting and its files"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
