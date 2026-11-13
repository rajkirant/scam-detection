"""Builds knowledge/mcq_ontology_v2.json from knowledge/mcq_ontology.json:
the repairs that the calls of the automated attack (13 Nov 2026) point to.
Every change is listed, so the repair can be read and repeated."""
import copy, json, sys
src, dst = sys.argv[1], sys.argv[2]
t = json.load(open(src))
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

def hint(qid, add, where=None):
    n = [0]
    def fn(q, path):
        if q["id"] == qid and (where is None or path.startswith(where)):
            q["hint"] = (q.get("hint", "") + " " + add).strip()
            n[0] += 1
    every(fn)
    assert n[0], qid
    log.append("hint  %-18s x%d  + %s" % (qid, n[0], add))

def option(qid, oid, value=None, text=None, where=None):
    n = [0]
    def fn(q, path):
        if q["id"] == qid and (where is None or path.startswith(where)):
            for o in q["options"]:
                if o["id"] == oid:
                    if value is not None:
                        log.append("value %s -> %s: %+.1f to %+.1f" % (path, oid, o["value"], value))
                        o["value"] = value
                    if text is not None:
                        log.append("text  %s -> %s: %s" % (path, oid, text))
                        o["text"] = text
                    n[0] += 1
    every(fn)
    assert n[0], (qid, oid)

def add_option(qid, new, before, where=None):
    n = [0]
    def fn(q, path):
        if q["id"] == qid and (where is None or path.startswith(where)):
            i = [o["id"] for o in q["options"]].index(before)
            q["options"].insert(i, copy.deepcopy(new))
            n[0] += 1
            log.append("new   %s -> %s: %+.1f %s" % (path, new["id"], new["value"], new["text"]))
    every(fn)
    assert n[0], qid

# 1. A claim of the caller that the call cannot confirm earns less, and only
#    when the person confirms it.
option("claim", "letter_followup", value=-0.3, where="government")
hint("claim", "The follow-up option also needs the person to say they have the letter, or that they sent the form or the application. The caller saying that a letter or a notice was sent, or that an application was received, is not enough: choose by what the caller asks for.", where="government")
option("claim", "existing_policy", value=-0.3, where="insurance_health")
hint("claim", "The existing policy option needs the person to say they have this policy, or to have raised the claim or the change themselves. A caller who only says the call is about the person's cover, an update, or a renewal has not shown that.", where="insurance_health")
option("claim", "overdue_with_notices", value=-0.3, where="utility")
q = next(q for s in t["options"] if s["id"] == "utility" for q in s["questions"] if q["id"] == "claim")
q["hint"] = "The overdue bill option needs the normal ways to pay: the person's online account, the details printed on the bill, or the company's office. A caller who reads out a bank account for the payment, or wants it paid on this call, is not offering the normal ways to pay: choose Not stated unless another option fits."
log.append("hint  utility/claim  + " + q["hint"])
hint("payment_purpose", "The ordered-or-owed option needs the person to say they ordered it or know the bill. A bill or a debt that only the caller says is unpaid is not that: if it must be paid on this call to keep a service on, choose the keep-a-service option.")
hint("payment_channel", "The organisation's own website, app, or office means one the person already uses or can find themselves. A website, a link, a portal, or a payment page named by a caller who rang the person is not that: choose by how the money is paid there, or Not stated.")
hint("caller_reason", "The follow-up option needs the person, not only the caller, to say they did or arranged it.")
option("parcel_expected", "expected", value=-0.5)

# 2. A request for a card number, a code or an ID number, by its purpose.
hint("sensitive_details", "When the caller asks for more than one kind of detail, choose the first option in the list that fits. A one-time or verification code sent to a phone or an email is the first option; the three digits on the back of a card belong with the card number.")
option("card_purpose", "to_verify_identity", value=1.0)
option("card_purpose", "to_fix_or_unblock", text="To pay a fee or a fine, to fix, unblock, or protect an account, or to cancel or stop a charge or a subscription")
add_option("id_purpose", {"id": "to_check_or_update_records", "value": 1.0,
    "text": "To check who the person is, or to check, confirm, or update records, cover, or benefits, in a call the person did not ask for"}, before="new_card_or_benefit")
def set_hint(qid, text):
    n = [0]
    def fn(q, path):
        if q["id"] == qid:
            q["hint"] = text; n[0] += 1
    every(fn); assert n[0]; log.append("hint  %-18s x%d  = %s" % (qid, n[0], text))
set_hint("id_purpose", "The person's-own-application option needs the person to say they applied or started it. A caller saying that an application or a request was received is not that.")
set_hint("money_movement", "Choose \"Nothing moves\" only when the person is asked for nothing but to say whether a payment is theirs. If the caller asks for a card number, the code on a card, a password, or a one-time code, choose Not stated.")
option("money_movement", "nothing_moves", value=-0.5)

# 3. Requests that were read as no request.
hint("payment_asked", "Sending money back, such as returning a refund paid in error, counts as Yes, and so does a fee to enrol or to start a service.")
set_hint("resolution", "Sending money back to the caller is sorting it out through the caller, whatever way of paying is named.")
option("remote_access", "remote_control", text="Install or open remote-control software, such as AnyDesk or TeamViewer, or any program, update, patch, or 'session' that lets the caller connect to, see, or control the device")
hint("remote_access", "Software that the caller asks the person to install so that the caller can connect, assist remotely, or run a scan is the remote-control option, whatever it is called.")
option("remote_purpose", "show_problem", text="To show, check, or fix an infection, a hack, unusual activity, or an error the caller says they found")

# 4. An invitation to verify that costs the caller nothing.
hint("verification_how", "A phone number, a website, an email address, or a link that the caller reads out or names is the caller's own contact, even when they call it official: choose the caller's-contacts option. The invited-to-check option fits only when the person is told to use a number or an account they already have, such as the number on their card or their bill.")

# 5. A caller who is greeted by name is not yet recognised.
option("identity_check", "known_voice", value=-0.3)
set_hint("identity_check", "The person greeting the caller by the name the caller has just given is not recognising them. Choose the recognises option only when the call shows something that only the two of them would know.")
option("story", "emergency_money", text="An accident, an arrest, bail, a hospital stay, a fee, or other trouble, for the caller or for someone else, that needs money now")

t["version_note"] = "Repair 1 of knowledge/mcq_ontology.json after the automated attack of 13 Nov 2026."
json.dump(t, open(dst, "w"), indent=2, ensure_ascii=False)
open(dst.replace(".json", "_changes.txt"), "w").write("\n".join(log) + "\n")
print(len(log), "changes")
