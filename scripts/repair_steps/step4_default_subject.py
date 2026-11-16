#!/usr/bin/env python3
"""Step 4 of scripts/repair_mcq.py, from the tree of step 3.

The default subject. When the root question finds no subject that fits, the
call goes to "Something else, or not clear from the call". Until now that
subject asked one question, about time pressure, so a call of a type the
tree has no subject for was barely asked anything: in the held-out test of
the released tree, 34 technical-support scams went there, and 19 of the 27
scams it missed scored exactly 0.

The default subject now asks every common question, as every other subject
does, and three questions of its own, written for a call of any kind: who
the caller says they are (recorded, scores 0), whether the caller says
something is wrong that must be dealt with on this call, and whether the
caller sends the person to a website, link, app or number that the caller
gives. As in the earlier repairs, no new option gives credit for the absence
of a sign.

The step also writes <output>_no_tech.json, the same tree without the
technical-support subject, for the held-out test.

    run by scripts/repair_mcq.py, which gives it the input and output files
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src, encoding="utf-8"))
other = next(s for s in t["options"] if s["id"] == "other")
assert other.get("ask") == ["common/pressure"] and not other.get("questions"), other

other["questions"] = [
    {
        "id": "claimed_identity", "role": "recorded",
        "prompt": "Who does the caller say they are?",
        "options": [
            {"id": "named_organisation", "text": "Someone from a named company, government body, or other organisation", "value": 0.0},
            {"id": "vague_department", "text": "A vague department or team, with no clear organisation", "value": 0.0},
            {"id": "private_person", "text": "A private person, speaking for themselves", "value": 0.0},
            {"id": "not_mentioned", "text": "Not stated", "value": 0.0},
        ],
    },
    {
        "id": "problem_to_fix_now",
        "prompt": "Does the caller say that something is wrong that the person must deal with during this call?",
        "hint": "Count a problem the caller brings up, such as a virus, a hacked account, a debt, a fine, a fault, or a suspended service, when the caller wants it dealt with on this call. A problem the person raised themselves, or one they can deal with later in their own way, is not this.",
        "options": [
            {"id": "must_act_now", "value": 0.8,
             "text": "Yes: the caller says something is wrong, and the person must act on this call to fix it"},
            {"id": "can_wait", "value": 0.0,
             "text": "A problem is mentioned, but the person can deal with it later in their own way"},
            {"id": "no_problem", "value": 0.0, "text": "No problem is brought up"},
            {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
        ],
    },
    {
        "id": "sent_elsewhere",
        "prompt": "Does the caller send the person to a website, link, app, or phone number that the caller gives during this call?",
        "hint": "A website or number that the person already uses, or that is printed on their own card, bill, or letter, is not given by the caller.",
        "options": [
            {"id": "caller_given", "value": 0.5,
             "text": "Yes: a website, link, app, or number that the caller gives, for the person to use now"},
            {"id": "own_known", "value": 0.0,
             "text": "Only a website or number that the person already uses or holds"},
            {"id": "none", "value": 0.0, "text": "No website, link, app, or number"},
            {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
        ],
    },
]
other["ask"] = ["claimed_identity", "common/pressure", "problem_to_fix_now", "sent_elsewhere",
                "common/caller_format", "common/contact_origin", "common/sensitive_details",
                "common/code_asked", "common/payment_asked", "common/remote_access",
                "common/verification"]

t["version_note"] = ("Final tree: knowledge/mcq_ontology.json with the repairs of "
                     "steps 1 to 3 of scripts/repair_mcq.py, and the default subject of "
                     "step 4.")
json.dump(t, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
log = ["default subject 'other': asks every common question (was: common/pressure only)",
       "new  other/claimed_identity (recorded, 0)",
       "new  other/problem_to_fix_now (+0.8 when the caller says it must be fixed on this call)",
       "new  other/sent_elsewhere (+0.5 when the caller gives a website, link, app or number to use now)"]
open(dst.replace(".json", "_changes.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")

nt = dict(t)
nt["options"] = [o for o in t["options"] if o["id"] != "tech_support"]
json.dump(nt, open(dst.replace(".json", "_no_tech.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("wrote", dst, "and", dst.replace(".json", "_no_tech.json"))
