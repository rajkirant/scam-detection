#!/usr/bin/env python3
"""
Offline tests for mcq_ontology: the MCQ ontology LLM - a tree of questions
out of a JSON file, each answer a value, the call's score their sum - and the
own-question features of its page.

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
import re
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

# a small tree, and what the fake model answers about each kind of call
TREE_ONTO = {
    "id": "t", "prompt": "What is this call about?",
    "options": [
        {"id": "bank", "text": "A bank account",
         "legit_contrast": "A real bank never asks for gift cards.",
         "questions": [
             {"id": "claimed", "role": "recorded",
              "prompt": "Who do they say they are?",
              "options": [{"id": "bank", "text": "A bank", "value": 0},
                          {"id": "not_mentioned", "text": "Not stated",
                           "value": 0}]}]},
        {"id": "other", "text": "Something else", "questions": []}],
    "common_questions": [
        {"id": "money", "prompt": "How would money move?",
         "options": [
             {"id": "gift", "text": "Gift cards", "value": 1.0,
              "follow_up": [
                  {"id": "buyer", "prompt": "Who buys them?",
                   "options": [{"id": "waits", "value": 0.5,
                                "text": "The caller waits on the line while "
                                        "they are bought"},
                               {"id": "not_mentioned", "text": "Not stated",
                                "value": 0}]}]},
             {"id": "no_payment", "text": "No payment at all", "value": -0.3,
              "absence": True},
             {"id": "not_mentioned", "text": "Not stated", "value": 0}]},
        {"id": "pressure", "prompt": "Is there pressure?",
         "options": [{"id": "threat", "text": "A threat of arrest or loss",
                      "value": 1.0},
                     {"id": "calm", "text": "No time pressure", "value": -0.5,
                      "absence": True},
                     {"id": "not_mentioned", "text": "Not stated",
                      "value": 0}]}]}
GIFT = "GIFTCALL caller: buy gift cards now or be arrested. Your bank account is at risk."
NEW = "NEWCALL caller: send the money via Western Union and act today."
CALM = "CALMCALL caller: your parcel arrives Tuesday, no rush and nothing to pay."
ZERO = "ZEROCALL caller: hello."
BADROOT = "BADROOT caller: hm."
CHARGE_Q = ("Does the caller say a tax, duty or customs charge (such as VAT, "
            "GST or import tax) must be paid before a parcel is delivered?")
VALUE_Q = "What is in the parcel, and what is it worth?"
# marker -> {question: (start of the option text it picks, quote)};
# a question it has no rule for is answered Not stated
TREE = {
    "GIFTCALL": {"What is this call about?": ("A bank", None),
                 "How would money move?": ("Gift cards", "buy gift cards now"),
                 "Who buys them?": ("The caller waits", "stay on the line"),
                 "Is there pressure?": ("A threat", "or be arrested"),
                 "Who do they say they are?": ("A bank", None)},
    "NEWCALL": {"What is this call about?": ("Something else", None),
                "How would money move?": ("A money transfer", "via Western Union"),
                "Is there pressure?": ("A threat", "act today")},
    "CALMCALL": {"What is this call about?": ("Something else", None),
                 "How would money move?": ("No payment", None),
                 "Is there pressure?": ("No time pressure", None)},
    "ZEROCALL": {"What is this call about?": ("Something else", None)},
    "BADROOT": {"What is this call about?": (None, None)},
    # parcel calls, for personalisation: the same words, different
    # countries. SHOETAX is the call in datasets/personalisation_2.csv
    "SHOETAX": {"What is this call about?": ("Something else", None),
                CHARGE_Q: ("Yes: a tax", "import tax of 18.40 to pay"),
                VALUE_Q: ("Worth 1,000 New Zealand dollars or less",
                          "a pair of shoes worth 80")},
    "TVTAX": {"What is this call about?": ("Something else", None),
              CHARGE_Q: ("Yes: a tax", "a customs charge to pay"),
              VALUE_Q: ("Worth more than", "a television worth 2,400")},
    "WINETAX": {"What is this call about?": ("Something else", None),
                CHARGE_Q: ("Yes: a tax", "duty to pay"),
                VALUE_Q: ("Alcohol", "a case of wine")},
    "BOXTAX": {"What is this call about?": ("Something else", None),
               CHARGE_Q: ("Yes: a tax", "import tax to pay")},
}
with open(os.path.join(HERE, "..", "datasets", "personalisation_2.csv"),
          newline="", encoding="utf-8") as f:
    PERSONAL_ROWS = list(csv.DictReader(f))
SHOE = "SHOETAX\n" + PERSONAL_ROWS[0]["text"]
TV = ("TVTAX caller: your parcel, a television worth 2,400, is here. There is "
      "a customs charge to pay before we deliver it.")
WINE = ("WINETAX caller: your parcel, a case of wine, is here. There is duty "
        "to pay before we deliver it.")
BOX = ("BOXTAX caller: your parcel is here. There is import tax to pay before "
       "we deliver it.")


def fake_tree(prompt):
    """One tree question, answered by the rules above; None when the
    transcript is not one of these calls."""
    tr = prompt.split("Transcript:\n", 1)[1].split("\n\n", 1)[0]
    mark = next((m for m in TREE if m in tr), None)
    if mark is None:
        return None
    q = prompt.split("\nQuestion: ", 1)[1].split("\n", 1)[0]
    opts = re.findall(r"^([A-T]) - (.+)$", prompt, re.M)
    want, quote = TREE[mark].get(q, ("Not stated", None))
    if want is None:
        return {"response": "The", "logprobs": [{"token": "The", "logprob": 0.0,
                "top_logprobs": [{"token": "The", "logprob": 0.0}]}]}
    pick = next((L for L, t in opts if t.startswith(want)), None)
    if pick is None:
        pick, quote = next(L for L, t in opts if t.startswith("Not stated")), None
    rest = [L for L, _ in opts if L != pick]
    alts = [{"token": pick, "logprob": math.log(0.8)}] + [
        {"token": L, "logprob": math.log(0.2 / len(rest))} for L in rest]
    text = pick
    if 'write "Quote:"' in prompt:
        text += "\nQuote: " + ('"%s"' % quote if quote else "none")
    return {"response": text, "logprobs": [
        {"token": pick, "logprob": alts[0]["logprob"], "top_logprobs": alts}]}


def fake_post(path, payload, timeout):
    STATE["calls"] += 1
    STATE["payloads"].append(payload)
    prompt = payload["prompt"]
    if "exactly one letter" in prompt and "\nQuestion: " in prompt:
        out = fake_tree(prompt)                         # a tree question
        if out is not None:
            return out if STATE["logprobs"] else {"response": out["response"]}
    if prompt.rstrip().endswith("New options:"):        # tree training
        return {"response": " - A money transfer service such as Western "
                            "Union\n- Gift cards"}
    if prompt.rstrip().endswith("Short answer:"):       # training, step 1
        tr = prompt.split("Transcript:\n", 1)[1]
        if "NEWCALL" in tr:
            return {"response": " Wire money via Western Union"}
        return {"response": " Their SSN.\nmore" if "SSNCALL" in tr
                else " A delivery time" if "PARCEL" in tr else " not said"}
    if prompt.rstrip().endswith("Options:"):            # training, step 2
        if STATE.get("group_bad"):
            return {"response": " I cannot group these."}
        return {"response": " - SSN (3)\n- A delivery time\n- not said\n"
                            "- a delivery time"}
    if "\nQuestion: " in prompt and "exactly one letter" not in prompt:
        # an open question
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
check("a good tree has no problems", M.check_ontology(TREE_ONTO), [])


def broken(change):
    t = json.loads(json.dumps(TREE_ONTO))
    change(t)
    return M.check_ontology(t)


money = lambda t: t["common_questions"][0]
check("a value outside -1..1 is named",
      any('"value" must be a number from -1 to 1' in p for p in broken(
          lambda t: money(t)["options"][0].update(value=1.5))))
check("not_mentioned must score 0",
      any("not_mentioned must score 0" in p for p in broken(
          lambda t: money(t)["options"][2].update(value=0.2))))
check("a recorded question scores 0 on every option",
      any("recorded question scores 0" in p for p in broken(
          lambda t: t["options"][0]["questions"][0]["options"][0].update(value=0.4))))
check("an option id used twice is named",
      any("used twice" in p for p in broken(
          lambda t: money(t)["options"][1].update(id="gift"))))
check("21 options is too many",
      any("between 2 and 20" in p for p in broken(
          lambda t: money(t).update(options=[dict(money(t)["options"][0], id="o%d" % i)
                                             for i in range(21)]))))
check("a question with no prompt is named",
      any("prompt" in p for p in broken(lambda t: money(t).update(prompt=""))))
check("a broken follow-up is found, with its path",
      any("common/money/gift/buyer" in p for p in broken(
          lambda t: money(t)["options"][0]["follow_up"][0]["options"][0].update(
              value="high"))))
path = os.path.join(tmp, "tree.json")
open(path, "w").write(json.dumps(TREE_ONTO))
tree = M.load_ontology(path)
check("load_ontology reads it, defaults filled",
      (tree["prompt"], len(tree["options"]), tree["options"][1]["questions"],
       tree["common_questions"][1]["options"][0]["follow_up"]),
      (TREE_ONTO["prompt"], 2, [], []))
check("every question has a path",
      [p for p, _, _ in M.iter_questions(tree)],
      ["common/money", "common/money/gift/buyer", "common/pressure",
       "bank/claimed"])
open(os.path.join(tmp, "broken.json"), "w").write('{"prompt": "x", ')
try:
    M.load_ontology(os.path.join(tmp, "broken.json"))
    check("broken JSON is refused", False)
except ValueError as e:
    check("broken JSON is refused, with where", "line" in str(e))
shipped = M.load_ontology()
shipped_text = open(os.path.join(HERE, "..", "knowledge",
                                 "mcq_ontology.json")).read()
check("the shipped tree holds no learned knowledge or dataset evidence",
      [w for w in ("scam_patterns", "scam_ontology", "legit_contrast",
                   '"evidence"', '"knowledge"', "built_from", "honeypot",
                   "scambait", "huggingface") if w in shipped_text], [])
check("the shipped tree loads: 15 subjects, 7 common questions, 54 in all",
      (len(shipped["options"]), len(shipped["common_questions"]),
       M.count_questions(shipped)), (15, 7, 54))
pay = next(q for q in shipped["common_questions"] if q["id"] == "payment_asked")
check("whether money is to move, then how and what for, only after a yes",
      ([o["id"] for o in pay["options"]],
       [[f["id"] for f in o.get("follow_up", [])] for o in pay["options"]],
       "no_payment" in [o["id"] for o in
                        pay["options"][0]["follow_up"][0]["options"]]),
      (["yes", "not_mentioned", "no_payment"],
       [["payment_channel", "payment_purpose"], [], []],
       False))
fmt = next(q for q in shipped["common_questions"] if q["id"] == "caller_format")
check("live or recorded first, what a recording asks only after 'recorded'",
      ([o["id"] for o in fmt["options"]],
       [o["id"] for o in fmt["options"][1]["follow_up"][0]["options"]]),
      (["live_person", "recording", "not_mentioned"],
       ["recording_press_key", "recording_info_only", "not_mentioned"]))
check("other knowledge files are not mistaken for one",
      M.is_mcq_ontology(os.path.join(HERE, "..", "knowledge", "scam_ontology.json")),
      False)
check("a saved question is not mistaken for one",
      M.check_ontology({"kind": "question", "prompt": "x", "options": []}),
      ["this is a saved question, not an ontology"])
check("an older one-question file still loads",
      M.check_ontology(ONTO), [])
flat = M.normalise(ONTO)
check("... each verdict a value on the root question",
      [(o["id"], o["value"]) for o in flat["options"]],
      [("ssn", 1.0), ("delivery", -1.0), ("refund", 1.0), ("wrong", -1.0)])
check("... and a verdict that is not scam/legit is named",
      any("verdict" in p for p in M.check_ontology(dict(ONTO, options=[
          dict(o, verdict="maybe") if i == 0 else o
          for i, o in enumerate(ONTO["options"])]))))

print("\nquotes")
check("a quote found word for word, whatever the case and punctuation",
      M.quote_found("Buy gift-cards NOW", "caller: buy gift cards now, or else"))
check("a long quote may differ a little (80% of its words in a row)",
      M.quote_found("you must buy the gift cards now today",
                    "you must buy the gift cards now or else"))
check("a quote that is not there is not found",
      M.quote_found("stay on the line", "buy gift cards now"), False)
check("a short quote must match exactly",
      M.quote_found("gift card", "gift cards"), False)
check("the quote is read from the answer",
      (M.find_quote('B\nQuote: "buy gift cards now"'),
       M.find_quote("C\nQuote: none"), M.find_quote("A")),
      ("buy gift cards now", None, None))

print("\none call through the tree")
STATE["calls"] = 0
STATE["payloads"] = []
r = M.classify(GIFT, tree)
check("root, the common questions with their follow-ups, then the "
      "subject's", [a["path"] for a in r["answers"]],
      ["root", "common/money", "common/money/gift/buyer", "common/pressure",
       "bank/claimed"])
check("one request per question", (STATE["calls"], r["requests"]), (5, 5))
check("the subject", (r["subject"], r["answers"][0]["value"]), ("bank", 0.0))
check("a quote found in the transcript keeps the answer",
      [(a["option"], a["quoted"], a["value"]) for a in r["answers"]
       if a["path"] in ("common/money", "common/pressure")],
      [("gift", True, 1.0), ("threat", True, 1.0)])
buyer = r["answers"][2]
check("a quote not in the transcript counts as Not stated, and says so",
      (buyer["option"], buyer["quoted"], buyer["value"], buyer["picked"][:16]),
      ("not_mentioned", False, 0.0, "The caller waits"))
check("a recorded question needs no quote and scores 0",
      (r["answers"][4]["option"], r["answers"][4]["quoted"],
       r["answers"][4]["value"]), ("bank", None, 0.0))
check("score = the sum of the values, verdict its sign",
      (r["score"], r["verdict"]), (2.0, "scam"))
check("the probabilities are kept with each answer",
      (round(r["answers"][1]["p"], 2), round(sum(r["answers"][1]["probs"]), 3)),
      (0.8, 1.0))
check("asked with quotes: the letter, then a quote line, room for it",
      ('write "Quote:"' in STATE["payloads"][1]["prompt"],
       STATE["payloads"][1]["options"]["num_predict"]), (True, M.QUOTE_TOKENS))
check("the root question asks for a letter only",
      ('write "Quote:"' in STATE["payloads"][0]["prompt"],
       STATE["payloads"][0]["options"]["num_predict"]), (False, M.ANSWER_TOKENS))
check("no repeat penalty, so the letter written is the one the logprobs pick",
      {p["options"].get("repeat_penalty") for p in STATE["payloads"]}, {1.0})
check("absence options are marked for the model",
      "B - No payment at all (absence)" in STATE["payloads"][1]["prompt"])
check("the file's rules are in the prompt, worded for the model",
      ("choose the first one that fits" in STATE["payloads"][1]["prompt"],
       "not_mentioned" in STATE["payloads"][1]["prompt"]), (True, False))
ordered = json.loads(json.dumps(TREE_ONTO))
ordered["options"][0]["ask"] = ["common/pressure", "claimed"]
check("a subject's ask list sets the order; unlisted questions follow, "
      "follow-ups straight after their answer",
      [a["path"] for a in M.classify(GIFT, M.normalise(ordered))["answers"]],
      ["root", "common/pressure", "bank/claimed", "common/money",
       "common/money/gift/buyer"])
ordered["options"][0]["ask"] = ["common/pressure", "nope", "common/pressure"]
check("an ask list naming an unknown question, or one twice, is refused",
      sorted(p.split(":")[0] for p in M.check_ontology(ordered)),
      ["bank", "bank"])
check("the shipped file: a bank call is asked who is calling, then urgency, "
      "then where the money goes", [p for p, _ in M.question_order(
          shipped, shipped["options"][0])][:3],
      ["bank_account/claimed_identity", "common/pressure",
       "bank_account/money_movement"])
tech = next(s for s in shipped["options"] if s["id"] == "tech_support")
check("a tech support call: who is calling, then what problem, then urgency",
      [p for p, _ in M.question_order(shipped, tech)][:3],
      ["tech_support/claimed_identity", "tech_support/problem",
       "common/pressure"])


def identity_then_urgency(s):
    order = [p.split("/")[-1] for p, _ in M.question_order(shipped, s)]
    lead = [x for x in ("claimed_identity", "problem") if x in order]
    return order[:len(lead) + 1] == lead + ["pressure"]


check("... and every subject asks who is calling first, then urgency",
      [s["id"] for s in shipped["options"] if not identity_then_urgency(s)], [])
r = M.classify(GIFT, tree, quotes=False)
check("quotes off: the follow-up counts (+0.5) and no quote is asked for",
      (r["score"], 'Quote:' in STATE["payloads"][-1]["prompt"],
       STATE["payloads"][-1]["options"]["num_predict"]),
      (2.5, False, M.ANSWER_TOKENS))
r = M.classify(CALM, tree)
check("absence answers need no quote; a legitimate call scores below 0",
      ([(a["option"], a["quoted"]) for a in r["answers"][1:]], r["score"],
       r["verdict"], r["requests"]),
      ([("no_payment", None), ("calm", None)], -0.8, "legit", 3))
r = M.classify(ZERO, tree)
check("nothing said: a score of exactly 0 is neutral",
      (r["score"], r["verdict"]), (0.0, "neutral"))
r = M.classify(BADROOT, tree)
check("an unreadable subject: the common questions are still asked",
      (r["answers"][0]["choice"], r["subject"], r["requests"]),
      (None, "other", 3))
STATE["payloads"] = []
M.classify(GIFT, tree)
check("no learned knowledge in any prompt, even where the file has some",
      any("A real bank never asks" in p["prompt"] or "Background" in p["prompt"]
          for p in STATE["payloads"]), False)
check("explain() names the answers that moved the score",
      M.explain(M.classify(GIFT, tree)),
      "bank; score +2.00 (scam): gift +1.0, threat +1.0; 1 answer without a "
      "quote counted as not stated")
check("verdict and the 0-100 score at the edges",
      (M.verdict_of(0.0), M.verdict_of(1e-9), M.verdict_of(-0.1),
       round(M.pct(0.0)), round(M.pct(2.0))),
      ("neutral", "neutral", "legit", 50, 88))

print("\npersonalisation: the same words, a different label per country")
countries = {c["name"]: M.load_country(c["path"]) for c in M.list_countries()}
check("the two shipped countries load",
      sorted(countries), ["Ireland", "New Zealand"])
ie, nz = countries["Ireland"], countries["New Zealand"]
check("the dataset: the same words twice, labelled by country",
      (len({r["text"] for r in PERSONAL_ROWS}),
       {r["country"]: r["label"] for r in PERSONAL_ROWS}),
      (1, {"Ireland": "nonscam", "New Zealand": "scam"}))


def label(text, country):
    r = M.classify(text, tree)
    p = M.personalise(text, r, country)
    return p["final"], p["from"], p.get("advice")


r = M.classify(SHOE, tree)
asked = []
for c in (ie, nz):
    n0 = STATE["calls"]
    p = M.personalise(SHOE, r, c)
    asked.append((STATE["calls"] - n0, [a["path"] for a in p["answers"]]))
check("one more request in Ireland, two in New Zealand",
      asked, [(1, ["country/import_charge"]),
              (2, ["country/import_charge", "country/parcel"])])
check("the dataset's call: each row's label, from its country's law",
      [label(SHOE, countries[row["country"]])[0] for row in PERSONAL_ROWS],
      ["scam" if row["label"] == "scam" else "legit" for row in PERSONAL_ROWS])
check("New Zealand, import tax on shoes worth 80: scam, don't pay",
      label(SHOE, nz), ("scam", "country", "Don't pay. Track the parcel on "
                        "the courier's own website, typed in yourself."))
check("Ireland: VAT on all goods from outside the EU, so legit",
      label(SHOE, ie)[:2], ("legit", "country"))
check("New Zealand, worth over NZ$1,000 or alcohol: the charge is real",
      [label(t, nz)[:2] for t in (TV, WINE)],
      [("legit", "country"), ("legit", "country")])
check("New Zealand, the value not stated: unsure, check what you paid",
      label(BOX, nz)[0], "unsure")
check("no import charge: no rule applies, the words decide in both",
      [label(CALM, c)[1] for c in (ie, nz)], ["words", "words"])
p = M.personalise(CALM, M.classify(CALM, tree), dict(nz, applies_to=["bank"]))
check("a call not on a subject the rules cover: no questions asked",
      (p["applies"], p["answers"], p["final"]), (False, [], "legit"))
check("a country by name, by file name or by path",
      [M.find_country(x).name for x in ("New Zealand", "new_zealand",
                                         "knowledge/countries/ireland.json")],
      ["new_zealand.json", "new_zealand.json", "ireland.json"])
try:
    M.find_country("Atlantis")
    check("an unknown country is refused, naming the ones there", False)
except ValueError as e:
    check("an unknown country is refused, naming the ones there",
          "Ireland, New Zealand" in str(e))

bad = json.loads(json.dumps(nz))
bad["rules"] = [{"if": {"parcel": ["no_such"], "mood": ["x"]},
                 "label": "maybe", "why": ""}]
probs = M.check_country(bad)
check("a broken country file names each problem",
      [any(w in p for p in probs) for w in
       ("no answer no_such", '"mood" is not a question', "scam, legit or unsure",
        'no "why"')], [True] * 4)
check("a country file needs rules",
      any('"rules"' in p for p in M.check_country(dict(bad, rules=[]))), True)

print("\nan older one-question file, through the same walk")
r = M.classify("SSNCALL transcript", flat)
check("one request, the verdict from the option's value",
      (r["requests"], r["subject"], r["score"], r["verdict"]),
      (1, "ssn", 1.0, "scam"))
check("a legitimate option scores -1",
      M.classify("PARCEL transcript", flat)["verdict"], "legit")

print("\nhints: a note for the model on what counts")
hinted = json.loads(json.dumps(TREE_ONTO))
hinted["common_questions"][1]["hint"] = "Being polite is not pressure."
check("a hint is checked: it must be text",
      [any('"hint" must be text' in p for p in M.check_ontology(
          dict(hinted, common_questions=[dict(hinted["common_questions"][1],
                                              hint=7)])))], [True])
ht = M.normalise(hinted)
pressure = ht["common_questions"][1]
check("a question's hint goes into its prompt, after the options",
      re.search(r"C - Not stated\n\nAbout this question: Being polite is "
                r"not pressure\.\n", M.question_prompt("call", pressure, ht))
      is not None)
check("a question without one gets no hint line",
      "About this question" in M.question_prompt(
          "call", ht["common_questions"][0], ht), False)
check("without_hints takes every hint out, and leaves the rest",
      (M.count_hints(ht), M.count_hints(M.without_hints(ht)),
       M.without_hints(ht)["common_questions"][1]["prompt"]),
      (1, 0, "Is there pressure?"))
check("the file writer keeps hints",
      M.to_file(ht)["common_questions"][1].get("hint"),
      "Being polite is not pressure.")
STATE["payloads"].clear()
M.classify(GIFT, M.without_hints(ht))
check("--no-hints: no prompt carries a hint",
      any("About this question" in p["prompt"] for p in STATE["payloads"]), False)
check("the shipped tree's hints: eight, on the questions that misfired",
      sorted(path for path, q, _ in M.iter_questions(shipped) if q.get("hint")),
      ["common/caller_format", "common/contact_origin", "common/payment_asked",
       "common/remote_access",
       "common/sensitive_details", "common/verification",
       "common/verification/yes/verification_how",
       "family_personal/identity_check"])

print("\na quote after a blank line")
check("a quote on its own line after a blank line is still read",
      M.find_quote('A\n\nQuote: "a security deposit of [Money]"'),
      "a security deposit of [Money]")
STATE["payloads"].clear()
r = M.classify(GIFT, tree)
stops = [p["options"].get("stop") for p in STATE["payloads"]
         if p["options"].get("stop")]
check("quoting requests do not stop at a blank line, before the quote",
      (bool(stops), any("\n\n" in s for s in stops)), (True, False))
check("each answer keeps the model's own reply, to explain an overrule",
      all("answered" in a for a in r["answers"]) and
      any(a["answered"].startswith("A") or a["answered"][:1].isalpha()
          for a in r["answers"]), True)

print("\nreading the letter")
STATE["lead"] = "**"
a = M.ask("PARCEL transcript", {"prompt": flat["prompt"], "options": flat["options"]},
          flat, root=True)
STATE["lead"] = ""
check("a token before the letter is skipped", (a["choice"], a["how"]),
      (1, "logprobs"))
a = M.ask("GIBBERISH transcript", {"prompt": flat["prompt"],
                                   "options": flat["options"]}, flat, root=True)
check("no letter at all is unreadable", (a["choice"], a["how"]),
      (None, "unreadable"))
STATE["logprobs"] = False
M._LOGPROBS_OK = None
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a = M.ask("SSNCALL transcript", {"prompt": flat["prompt"],
                                     "options": flat["options"]}, flat, root=True)
    M.ask("SSNCALL transcript", {"prompt": flat["prompt"],
                                 "options": flat["options"]}, flat, root=True)
STATE["logprobs"] = True
check("without logprobs the answered letter stands, at 100%",
      (a["choice"], a["how"], a["probs"][0]), (0, "letter", 1.0))
check("it says so, once", buf.getvalue().count("no logprobs"), 1)

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

print("\nscoring a dataset (the page's Score a dataset)")
out = os.path.join(tmp, "run.metrics.json")
ev = os.path.join(tmp, "ev.csv")
with open(ev, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    w.writerows([[1, "scam", GIFT], [2, "nonscam", CALM],
                 [3, "scam", ZERO], [4, "scam", NEW]])
import eval_common as EC                                       # noqa: E402
EC.RESULTS_DIR = tmp
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M.main(["evaluate", "--csv", ev, "--ontology", path, "--out", out])
res = json.load(open(out))
check("metrics.json carries the tree's details",
      (res["kind"], res["question"], res["neutral"], res["unquoted"],
       res["quotes"]), ("mcq", TREE_ONTO["prompt"], 1, 1, True))
check("neutral counts as not scam (ZEROCALL is a missed scam)",
      (res["metrics"]["tp"], res["metrics"]["fn"], res["metrics"]["tn"]),
      (2, 1, 1))
check("the subject table counts where each call went",
      [(o["id"], o["scam"], o["legit"]) for o in res["options"]],
      [("bank", 1, 0), ("other", 2, 1)])
rows = list(csv.DictReader(open(os.path.join(tmp, "eval_run.csv"))))
check("the per-call CSV has the score and the three-way verdict",
      [(r["score"], r["verdict3"]) for r in rows],
      [("2.0", "scam"), ("-0.8", "legit"), ("0.0", "neutral"),
       ("1.0", "scam")])

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
check("<system>_pct is the score on 0-100",
      {r["text"][:8]: r["mcq_ontology__stripped_pct"] for r in rows}["GIFTCALL"],
      "88.08")

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

print("\ntraining the tree on a dataset")
tr_csv = os.path.join(tmp, "tree_train.csv")
with open(tr_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    for i in range(2):
        w.writerow([i, "scam", GIFT + " %d" % i])
        w.writerow([10 + i, "scam", NEW + " %d" % i])
    for i in range(4):
        w.writerow([20 + i, "nonscam", CALM + " %d" % i])
STATE["payloads"] = []
logged = []
t = M.train_ontology(tr_csv, tree, calls=8, log=logged.append)
qs = {p: q for p, q, _ in M.iter_questions(t)}
check("the tree it started from is not changed",
      len(tree["common_questions"][0]["options"]), 3)
check("a new option where calls said Not stated but had an answer",
      [(a["path"], a["text"]) for a in t["training"][-1]["added"]],
      [("common/money", "A money transfer service such as Western Union")])
check("... asked openly first, and the proposal named the existing options",
      any(p["prompt"].rstrip().endswith("New options:")
          and "- Gift cards" in p["prompt"] and "Western Union" in p["prompt"]
          for p in STATE["payloads"]))
check("... a duplicate of an existing option is dropped",
      [o["text"] for o in qs["common/money"]["options"]].count("Gift cards"), 1)
check("values move toward the labels: (4 x old + scam - legit) / (4 + n)",
      {o["id"]: o["value"] for o in qs["common/money"]["options"]},
      {"gift": 1.0, "a_money_transfer_service_such": 0.33,
       "no_payment": -0.65, "not_mentioned": 0.0})
check("... options put back in order, strongest scam sign first",
      [o["id"] for o in qs["common/money"]["options"]],
      ["gift", "a_money_transfer_service_such", "no_payment",
       "not_mentioned"])
after_nm = {"options": [{"id": "a", "value": -0.5}, {"id": "b", "value": 0.5},
                        {"id": "not_mentioned", "value": 0.0},
                        {"id": "c", "value": -0.3}, {"id": "d", "value": 0.2}]}
M._reorder(after_nm)
check("... an option the file puts after Not stated stays after it",
      [o["id"] for o in after_nm["options"]],
      ["b", "a", "not_mentioned", "d", "c"])
check("pressure: threat 4 scam calls, calm 4 legit",
      [(o["id"], o["value"]) for o in qs["common/pressure"]["options"]],
      [("threat", 1.0), ("calm", -0.75), ("not_mentioned", 0.0)])
check("an option no call landed on keeps its value (a failed quote is "
      "Not stated)", qs["common/money/gift/buyer"]["options"][0]["value"], 0.5)
check("recorded questions and the root are not given values",
      ([o["value"] for o in qs["bank/claimed"]["options"]],
       [s["value"] for s in t["options"]]), ([0.0, 0.0], [0.0, 0.0]))
check("each changed option keeps its history",
      qs["common/money"]["options"][2]["training"],
      [{"dataset": tr_csv, "scam": 0, "legit": 4, "before": -0.3,
        "after": -0.65}])
tt = t["training"][-1]
check("the training run is recorded",
      (tt["calls"], tt["scam"], tt["legit"], tt["prior_weight"],
       tt["training_calls_right_before"], tt["training_calls_right_after"]),
      (8, 4, 4, 4.0, 1.0, 1.0))
check("what it writes is a valid ontology", M.check_ontology(t), [])

un_csv = os.path.join(tmp, "uneven.csv")
with open(un_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "text"])
    w.writerows([[1, "scam", GIFT + " a"], [2, "scam", GIFT + " b"],
                 [3, "nonscam", CALM]])
u = M.train_ontology(un_csv, tree, calls=3, new_options=0, log=lambda *_: None)
check("an uneven sample: each class counts half (2 scam, 1 legit)",
      {o["id"]: o["value"] for o in u["common_questions"][0]["options"]}
      ["no_payment"], round((4 * -0.3 - 1.5) / (4 + 1.5), 2))
check("new options 0: none added", u["training"][-1]["added"], [])

out_tree = os.path.join(tmp, "trained.json")
with contextlib.redirect_stdout(io.StringIO()):
    rc = M.main(["train", "--csv", tr_csv, "--ontology", path, "--out",
                 out_tree, "--calls", "8"])
text = open(out_tree).read()
check("train writes a file that loads, without empty lists",
      (rc, len(M.load_ontology(out_tree)["training"]),
       '"follow_up": []' in text), (0, 1, False))
saved = json.loads(text)
check("the trained file is just the questionnaire and a run summary",
      (sorted(saved), "training" in text.split('"training": [')[0],
       '"added": {' in text, saved["training"][0]["added"]),
      (["common_questions", "options", "prompt", "training"], False, False, 1))
try:
    M.main(["train", "--csv", tr_csv, "--ontology", path, "--out", out_tree])
    check("an existing file is not replaced without --force", False)
except SystemExit as e:
    check("an existing file is not replaced without --force",
          "already exists" in str(e))
with contextlib.redirect_stdout(io.StringIO()):
    M.main(["train", "--csv", tr_csv, "--ontology", out_tree, "--out",
            out_tree, "--calls", "8", "--force"])
check("training a trained file adds to its history",
      len(M.load_ontology(out_tree)["training"]), 2)

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all good - the tree, its quotes, its scores and its training"
              if not fails else "%d FAILED" % fails))
sys.exit(1 if fails else 0)
