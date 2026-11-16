#!/usr/bin/env python3
"""Writes a copy of an MCQ tree with one subject held out, for the held-out
test of a scam type the tree knows nothing of.

The subject keeps its place in the root question, so calls are routed as
with the full tree, but it is marked "held_out" and loses its own questions,
so a call routed there is asked the default subject's ("other") questions.

    python scripts/hold_out.py knowledge/mcq_ontology_v5.json tech_support \
        knowledge/mcq_ontology_v5_tech_held_out.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcq_ontology as M                                       # noqa: E402

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, sid, dst = sys.argv[1:]
t = json.load(open(src, encoding="utf-8"))
subj = next((s for s in t["options"] if s["id"] == sid), None)
if subj is None:
    sys.exit("no subject %r in %s; there are: %s"
             % (sid, src, ", ".join(s["id"] for s in t["options"])))
if sid == "other":
    sys.exit("the default subject cannot be held out: held-out calls go there")
subj["held_out"] = True
subj["questions"] = []
subj.pop("ask", None)
subj.pop("skip", None)
probs = M.check_ontology(t)
if probs:
    sys.exit("; ".join(probs))
json.dump(t, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote %s: %s held out, its calls asked the default subject's questions"
      % (dst, sid))
