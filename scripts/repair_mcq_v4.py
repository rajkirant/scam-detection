#!/usr/bin/env python3
"""Builds knowledge/mcq_ontology_v4.json from knowledge/mcq_ontology_v3.json.

Repair 3, from the honest-call test of 14 Nov 2026: the 4 honest calls that
v3 did not pass (2 false alarms, 2 neutral) were all about a bill or a tax
amount the person already knew of. The caller-raised-a-problem option (+0.8)
and the tax-or-fine claim (+0.3) were chosen although the person said they
had received the notice, or had called first.

Both now go through a follow-up question that asks whether the person shows
they already knew. As in repair 2, the credit depends on the person's own
words, never on the caller's.

    python scripts/repair_mcq_v4.py knowledge/mcq_ontology_v3.json knowledge/mcq_ontology_v4.json
"""
import copy
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src, encoding="utf-8"))
log = []

KNEW = {
    "id": "person_knew",
    "prompt": "Does the person show, in their own words, that they already knew about this before the call?",
    "hint": "Knowing means the person says they received the bill, the notice, or the letter, that they called about it first, or that they were already dealing with it. The person only saying \"okay\" or \"I see\" to the caller is not knowing.",
    "options": [
        {"id": "knew", "value": -0.8,
         "text": "Yes: they say they received it, called about it, or were already dealing with it"},
        {"id": "did_not_know", "value": 0.0,
         "text": "No: it is news to them, or they only agree with the caller"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
}


def walk(q, path):
    for o in q["options"]:
        if (path, o["id"]) in TARGETS:
            o.setdefault("follow_up", []).append(copy.deepcopy(KNEW))
            log.append("follow-up %s -> %s (%+.1f): person_knew, -0.8 when the person already knew"
                       % (path, o["id"], o["value"]))
        for f in o.get("follow_up") or []:
            walk(f, path + "/" + o["id"] + "/" + f["id"])


TARGETS = {("common/contact_origin/caller_rang/caller_reason", "caller_raised_problem"),
           ("government/claim", "money_owed")}
for q in t["common_questions"]:
    walk(q, "common/" + q["id"])
for s in t["options"]:
    for q in s.get("questions") or []:
        walk(q, s["id"] + "/" + q["id"])
assert len(log) == 2, log

t["version_note"] = ("Repair 3 of knowledge/mcq_ontology.json, from the honest-call test of "
                     "14 Nov 2026: built from mcq_ontology_v3.json by scripts/repair_mcq_v4.py.")
json.dump(t, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(dst.replace(".json", "_changes.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
print(len(log), "changes")
