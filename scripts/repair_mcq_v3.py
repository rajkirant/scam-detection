#!/usr/bin/env python3
"""Builds knowledge/mcq_ontology_v3.json from knowledge/mcq_ontology_v2.json.

Repair 2 after the automated attack of 13 Nov 2026. Repair 1 (v2) changed
values and hints; in the calls that still passed, the model did not follow
the hints. This repair puts the same checks into the tree as questions:

  1. A claim of the caller (a letter, an existing policy, an overdue bill, a
     bill the person owes) earns its credit only through a follow-up that
     asks whether the person confirms it.
  2. An official channel, and an invitation to check the caller, earn their
     credit only through a follow-up that asks who named the website, the
     number or the place: the person already had it, or the caller gave it.
  3. A password, PIN or one-time code is asked about in a question of its
     own, because the question on details takes one answer and lost the code
     when a card number was asked for as well.

One rule for every new question: it gives no credit for the absence of a
sign, and none for something only the caller says.

    python scripts/repair_mcq_v3.py knowledge/mcq_ontology_v2.json knowledge/mcq_ontology_v3.json
"""
import copy
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src, encoding="utf-8"))
log = []


def walk(q, fn, path):
    fn(q, path)
    for o in q["options"]:
        for f in o.get("follow_up") or []:
            walk(f, fn, path + "/" + o["id"] + "/" + f["id"])


def every(fn):
    for q in t["common_questions"]:
        walk(q, fn, "common/" + q["id"])
    for s in t["options"]:
        for q in s.get("questions") or []:
            walk(q, fn, s["id"] + "/" + q["id"])


def gate(qid, oid, follow, where=None):
    """Move the credit of option `oid` into a follow-up question: the option
    itself becomes worth 0, and `follow` decides."""
    n = [0]

    def fn(q, path):
        if q["id"] == qid and (where is None or path.startswith(where)):
            for o in q["options"]:
                if o["id"] == oid:
                    log.append("gate  %s -> %s: %+.1f to 0, decided by %s"
                               % (path, oid, o["value"], follow["id"]))
                    o["value"] = 0.0
                    o.setdefault("follow_up", []).append(copy.deepcopy(follow))
                    n[0] += 1
    every(fn)
    assert n[0], (qid, oid)


def confirmed(qid, credit, what):
    return {
        "id": qid,
        "prompt": "Does the person confirm %s, or does only the caller say so?" % what,
        "hint": "Go by the person's own words. The person only saying \"okay\", \"yes\", or \"I see\" to what the caller says is not confirming it.",
        "options": [
            {"id": "person_confirms", "value": credit,
             "text": "The person confirms it in their own words: they have it, know about it, or started it themselves"},
            {"id": "caller_only", "value": 0.0, "text": "Only the caller says so"},
            {"id": "person_denies", "value": 0.5,
             "text": "The person says they do not have it, did not receive it, or did not ask for it"},
            {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
        ],
    }


# 1. Claims of the caller.
gate("claim", "letter_followup", confirmed("letter_confirmed", -0.8, "the letter, the form, or the application"), where="government")
gate("claim", "existing_policy", confirmed("policy_confirmed", -0.8, "the policy, the claim, or the change"), where="insurance_health")
gate("claim", "overdue_with_notices", confirmed("bill_confirmed", -0.8, "the bill and the notices"), where="utility")
gate("payment_purpose", "agreed_purchase", confirmed("purchase_confirmed", -0.5, "the order, the sign-up, or the bill"))

# 2. Who named the channel, and the way to check.
gate("payment_channel", "official_channel", {
    "id": "channel_source",
    "prompt": "Who named the website, the app, the office, or the payment method?",
    "options": [
        {"id": "person_already_has", "value": -1.0,
         "text": "The person already uses it, or it is printed on a bill or a letter they hold"},
        {"id": "caller_named", "value": 0.0,
         "text": "The caller names or gives it in this call, such as a website address, a link, a payment page, or a place to go to"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
})
gate("verification_how", "encouraged", {
    "id": "check_source",
    "prompt": "What would the person use to check the caller?",
    "options": [
        {"id": "own_source", "value": -1.0,
         "text": "A number, an account, or a letter the person already has, such as the number on their card or their bill, or their own online account"},
        {"id": "caller_source", "value": 0.0,
         "text": "A phone number, a website, an email address, or a link that the caller gives or names in this call"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
})

# 3. A code is asked about on its own.
code_q = {
    "id": "code_asked",
    "prompt": "Does the caller ask the person to say, read out, or type in a password, a PIN, or a one-time code?",
    "hint": "A one-time code is a code sent to the person's phone or email during or just before the call. The three digits on the back of a card are not that, and neither is an order or reference number.",
    "options": [
        {"id": "code_requested", "value": 1.0,
         "text": "Yes: a password, a PIN, or a one-time or verification code"},
        {"id": "no_code", "value": 0.0, "text": "No code, password, or PIN is asked for"},
        {"id": "not_mentioned", "value": 0.0, "text": "Not stated"},
    ],
}
i = [q["id"] for q in t["common_questions"]].index("sensitive_details")
t["common_questions"].insert(i + 1, code_q)
for q in t["common_questions"]:
    if q["id"] == "sensitive_details":
        for o in q["options"]:
            if o["id"] == "security_codes":
                log.append("value common/sensitive_details -> security_codes: %+.1f to 0, counted by common/code_asked" % o["value"])
                o["value"] = 0.0
n = 0
for s in t["options"]:
    ask = s.get("ask")
    if ask and "common/sensitive_details" in ask and "common/code_asked" not in ask:
        ask.insert(ask.index("common/sensitive_details") + 1, "common/code_asked")
        n += 1
log.append("new   common/code_asked (+1.0 when a code is asked), asked after common/sensitive_details in %d subjects" % n)

t["version_note"] = ("Repair 2 of knowledge/mcq_ontology.json after the automated attack of "
                     "13 Nov 2026: built from mcq_ontology_v2.json by scripts/repair_mcq_v3.py.")
json.dump(t, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(dst.replace(".json", "_changes.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
print(len(log), "changes")
