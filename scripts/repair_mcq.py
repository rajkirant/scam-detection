#!/usr/bin/env python3
"""Builds the final MCQ tree from the released one, and lists every change.

    python scripts/repair_mcq.py

reads knowledge/mcq_ontology.json (the released tree) and writes

    knowledge/mcq_ontology_final.json           the final tree
    knowledge/mcq_ontology_final_no_tech.json   the same tree without the
                                                technical-support subject,
                                                for the held-out test
    knowledge/mcq_ontology_final_changes.txt    every change, by its source

The repair is made in four steps, each a script in scripts/repair_steps/
that reads the tree of the step before:

  1. step1_attack_values.py    from the automated attack: values and hints
  2. step2_attack_grafts.py    from the automated attack: follow-up questions
                               grafted under the options the attack used
  3. step3_honest_calls.py     from honest calls on the same subjects: a
                               follow-up on whether the person already knew
  4. step4_default_subject.py  from the held-out test: the default subject
                               asks every common question and three of its own

Only the final tree is kept; the trees between the steps are written to a
temporary folder and thrown away. Options: --released PATH, --out PATH.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEPS = [
    ("step1_attack_values.py", "Step 1, from the automated attack: values and hints"),
    ("step2_attack_grafts.py", "Step 2, from the automated attack: grafted follow-up questions"),
    ("step3_honest_calls.py", "Step 3, from honest calls: a grafted follow-up question"),
    ("step4_default_subject.py", "Step 4, from the held-out test: the default subject"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--released", default=os.path.join(ROOT, "knowledge", "mcq_ontology.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "knowledge", "mcq_ontology_final.json"))
    a = ap.parse_args(argv)
    base = a.out[:-len(".json")]
    changes = []
    with tempfile.TemporaryDirectory() as tmp:
        src = a.released
        for i, (script, title) in enumerate(STEPS, 1):
            dst = os.path.join(tmp, "step%d.json" % i)
            subprocess.run([sys.executable, os.path.join(HERE, "repair_steps", script), src, dst],
                           check=True, stdout=subprocess.DEVNULL)
            with open(dst.replace(".json", "_changes.txt"), encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            changes += ["== %s (%d changes)" % (title, len(lines))] + lines + [""]
            src = dst
        for name in ("", "_no_tech"):
            with open(os.path.join(tmp, "step4%s.json" % name), encoding="utf-8") as f:
                t = json.load(f)
            t["version_note"] = ("Final tree: knowledge/mcq_ontology.json with every repair of "
                                 "scripts/repair_mcq.py, listed in mcq_ontology_final_changes.txt."
                                 + (" Technical-support subject removed, for the held-out test."
                                    if name else ""))
            with open(base + name + ".json", "w", encoding="utf-8") as f:
                json.dump(t, f, indent=2, ensure_ascii=False)
    with open(base + "_changes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(changes))
    print("wrote %s, %s_no_tech.json and %s_changes.txt (%d changes)"
          % (a.out, base, base, sum(1 for c in changes if c and not c.startswith("=="))))


if __name__ == "__main__":
    main()
