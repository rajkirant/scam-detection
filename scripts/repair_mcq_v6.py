#!/usr/bin/env python3
"""Builds knowledge/mcq_ontology_v6.json from knowledge/mcq_ontology_v5.json.

A fuller default subject. The default subject, "Something else, or not clear
from the call", is where a call of a type the tree has no subject for is
judged, so its questions are written for a call of any kind, from what scam
calls of every type do, and none names a scam type:

- who the caller says they are (recorded) now opens a follow-up when the
  caller names an organisation: does the person show, in their own words,
  that they deal with it? A person who says they have no dealings with it
  counts toward a scam (+0.5), one who says they do counts a little toward
  legitimate (-0.3), and the caller's word alone counts for nothing;
- a problem that must be dealt with on this call now earns its credit only
  through a follow-up on who raised it (+0.8 when the caller did, 0 when
  the person rang about it): in v5 the question also fired on ordinary
  support calls in which the person had rung about a fault of their own;
- being sent to a website, link, app or number that the caller gives gets a
  hint that steps such as restarting or clearing a cache are not that;
- new: what the caller does when the person doubts them or wants to check
  (+0.8 when the caller insists, gets angry or threatens);
- new: whether the reason for the call changes into a request for money,
  details or access (+0.3).

As in the earlier repairs, no option gives credit for the absence of a sign.

knowledge/mcq_ontology_v6_no_tech.json is the same tree for the held-out
test: the technical-support subject keeps its place in the root question,
so calls are routed as with the full tree, but it is marked "held_out" and
has no questions of its own, so a call routed there is asked the default
subject's questions.

    python scripts/repair_mcq_v6.py knowledge/mcq_ontology_v5.json knowledge/mcq_ontology_v6.json
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcq_ontology as M                                       # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src, encoding="utf-8"))
other = next(s for s in t["options"] if s["id"] == "other")
own = {q["id"]: q for q in other["questions"]}
assert set(own) == {"claimed_identity", "problem_to_fix_now", "sent_elsewhere"}, own.keys()
log = []

# who the caller says they are: does the person deal with that organisation?
named = next(o for o in own["claimed_identity"]["options"] if o["id"] == "named_organisation")
named["follow_up"] = [{
    "id": "person_deals_with_it",
    "prompt": "Does the person show, in their own words, that they deal with this organisation?",
    "hint": "Go by the person's own words. The person only saying \"okay\" or \"yes\" to what the caller says is not dealing with it. Owning a product the organisation makes is not dealing with it unless the person says they have an account, an order, or a service with it.",
    "options": [
        {"id": "person_confirms", "value": -0.3,
         "text": "Yes: they say they have an account, an order, or a service with it"},
        {"id": "person_denies", "value": 0.5,
         "text": "No: they say they have no account or dealings with it, or have never heard of it"},
        {"id": "caller_only", "value": 0.0, "text": "Only the caller says so"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
}]
log.append("graft other/claimed_identity/named_organisation/person_deals_with_it "
           "(person confirms -0.3, person denies +0.5, caller only 0)")

# a problem to deal with now: the credit moves to who raised it
p = own["problem_to_fix_now"]
must = next(o for o in p["options"] if o["id"] == "must_act_now")
assert must["value"] == 0.8
must["value"] = 0.0
must["follow_up"] = [{
    "id": "who_raised",
    "prompt": "Who first raised the problem in this call?",
    "hint": "The caller raised it when they bring it up and the person did not know about it before the call. The person raised it when they rang, or asked for help, about it themselves.",
    "options": [
        {"id": "caller_raised", "value": 0.8,
         "text": "The caller: the person did not know about it before the call"},
        {"id": "person_raised", "value": 0.0,
         "text": "The person: they rang or asked for help about it themselves"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
}]
log.append("prune other/problem_to_fix_now/must_act_now +0.8 -> 0, and graft "
           "who_raised (caller +0.8, person 0)")

# sent elsewhere: a step to take is not a website, link, app or number
s = own["sent_elsewhere"]
s["hint"] += (" Steps the person takes on their own device, such as restarting, clearing a cache, "
              "or changing a setting, are not a website, link, app, or number.")
log.append("hint other/sent_elsewhere: steps on the person's own device are not a link")

other["questions"] += [
    {
        "id": "response_to_doubt",
        "prompt": "When the person doubts the caller, asks who they are, or wants to check or think it over, what does the caller do?",
        "hint": "Count only what the caller does after the person shows doubt. A person who never doubts or questions the caller is the third option.",
        "options": [
            {"id": "pushes_on", "value": 0.8,
             "text": "Pushes on: insists, gets angry, or threatens the person"},
            {"id": "lets_check", "value": 0.0,
             "text": "Lets the person check, think it over, or call back on a number they already have"},
            {"id": "no_doubt", "value": 0.0, "text": "The person does not doubt or question the caller"},
            {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
        ],
    },
    {
        "id": "story_shift",
        "prompt": "Does the reason for the call change into a request for money, personal details, or access to a device?",
        "hint": "Choose Yes when the call starts with one reason, such as a check, a problem, a refund, or a prize, and later turns into such a request. A call that asks for the same thing from the start is No.",
        "options": [
            {"id": "shifts", "value": 0.3,
             "text": "Yes: it starts with one reason and turns into a request for money, details, or access"},
            {"id": "same_reason", "value": 0.0, "text": "No: the reason stays the same"},
            {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
        ],
    },
]
log.append("new  other/response_to_doubt (caller pushes on +0.8)")
log.append("new  other/story_shift (reason turns into a request +0.3)")

other["ask"] = ["claimed_identity", "common/contact_origin", "problem_to_fix_now",
                "common/pressure", "sent_elsewhere", "common/caller_format",
                "common/sensitive_details", "common/code_asked", "common/payment_asked",
                "common/remote_access", "common/verification", "response_to_doubt",
                "story_shift"]
log.append("ask order of 'other': who is calling and why first, then the requests, "
           "then doubt and a change of story")

t["version_note"] = ("Final tree: knowledge/mcq_ontology.json with the repairs of "
                     "mcq_ontology_v2 to v5, and the fuller default subject of "
                     "scripts/repair_mcq_v6.py.")

probs = M.check_ontology(t)
assert not probs, probs
json.dump(t, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# the held-out tree: technical support routes to the default subject
h = copy.deepcopy(t)
tech = next(s for s in h["options"] if s["id"] == "tech_support")
tech["held_out"] = True
tech["questions"] = []
tech.pop("ask", None)
tech.pop("skip", None)
probs = M.check_ontology(h)
assert not probs, probs
base = dst[:-len(".json")]
json.dump(h, open(base + "_no_tech.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(base + "_changes.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\n".join(log))
print("wrote", dst, "and", base + "_no_tech.json")
