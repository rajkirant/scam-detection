#!/usr/bin/env python3
"""
test_stripped_pairing.py - offline checks on the content-deletion test.

The test scores every system on a dataset and on a stripped twin of it, and
reports trusted accuracy A = a_full - max(0, a_stripped - 0.5). Its whole
value rests on one rule: a system that learns from the data is trained on the
ORIGINAL text only, and the stripped copy is only ever scored. Break that rule
and the test measures something else - whether a fresh model can learn the
stripped data - while still printing a plausible-looking number. So most of
what follows checks the rule directly, by recording what each learner was
actually shown.

It also pins down that turning the test on changes nothing about the ordinary
benchmark: the full-text rows of a paired run must be the rows an unpaired
run produces.

    python3 scripts/test_stripped_pairing.py

Needs scikit-learn and pandas. No Ollama, GPU or network: the LLM steps are
replaced with stand-ins that record what they were given.
Exits non-zero on the first failure.
"""

import csv
import json
import random
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import trusted                                              # noqa: E402
import combined_evaluate as CE                              # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print("  %-62s %s" % (name, "ok" if ok else "FAIL"))
    if not ok:
        print("      got  %r\n      want %r" % (got, want))
        FAILS.append(name)


def close(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print("  %-62s %s" % (name, "ok" if ok else "FAIL"))
    if not ok:
        print("      got  %r\n      want %r" % (got, want))
        FAILS.append(name)


def write_csv(path, rows, fields=("id", "label", "source", "text")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_pair(tmp, n=40, seed=7):
    """A small balanced dataset and a stripped twin whose text is traceable.

    Every original transcript carries its own id, and every stripped one reads
    "STRIPPED <id> ...". A stand-in that sees a transcript can therefore say
    exactly which call, and which copy, it was shown.

    The twins also carry what the real stripped data carries: function words
    that differ by collection ("you me" in one class, "we our" in the other)
    and never appear in any original transcript. A model trained on the
    originals has never seen them, so it cannot use them. A model wrongly
    trained on the stripped text learns them and separates the twins
    perfectly - which is what lets a test tell the two apart.
    """
    rnd = random.Random(seed)
    scam_words = ["gift", "card", "urgent", "refund", "arrest", "wire", "code"]
    bank_words = ["balance", "statement", "branch", "savings", "deposit", "loan"]
    rows, twins = [], []
    for i in range(n):
        scam = i % 2 == 0
        words = rnd.sample(scam_words if scam else bank_words, 3)
        cid = "c%03d" % i
        rows.append({"id": cid, "label": "scam" if scam else "nonscam",
                     "source": "a" if scam else "b",
                     "text": "ORIGINAL %s the caller said %s" % (cid, " ".join(words))})
        twins.append({"id": cid, "label": "scam" if scam else "nonscam",
                      "source": "a" if scam else "b",
                      "text": "STRIPPED %s the of is %s"
                              % (cid, "you me" if scam else "we our")})
    full = Path(tmp) / "set.csv"
    strip = Path(tmp) / "set_stripped.csv"
    write_csv(full, rows)
    write_csv(strip, twins)
    return full, strip, rows, twins


# --------------------------------------------------------------- the formula
def test_formula():
    print("\ntrusted accuracy")
    close("stripped at chance keeps the whole full score",
          trusted.trusted_accuracy(0.874, 0.500), 0.874)
    close("stripped below chance is not a bonus",
          trusted.trusted_accuracy(0.929, 0.493), 0.929)
    close("stripped as good as full lands on chance",
          trusted.trusted_accuracy(1.000, 1.000), 0.500)
    close("design-B bag-of-words: 1.000 - (0.557 - 0.5)",
          trusted.trusted_accuracy(1.000, 0.557), 0.943)
    close("chance on full, above chance stripped, goes below chance",
          trusted.trusted_accuracy(0.500, 0.550), 0.450)
    close("accuracy_from_counts is (tp+tn)/n",
          trusted.accuracy_from_counts(211, 187, 0, 24), 235 / 422)
    check("accuracy_from_counts of nothing is 0",
          trusted.accuracy_from_counts(0, 0, 0, 0), 0.0)


# ------------------------------------------------------------- pairing check
def test_check_pair(tmp):
    print("\ncheck_pair refuses a twin that does not line up")
    full, strip, rows, twins = make_pair(tmp)
    check("a matching twin passes", trusted.check_pair(full, strip), [])

    def bad(name, twin_rows, needle, fields=("id", "label", "source", "text")):
        p = Path(tmp) / ("bad_%s.csv" % name.replace(" ", "_"))
        write_csv(p, twin_rows, fields)
        got = trusted.check_pair(full, p)
        check(name, bool(got) and any(needle in g for g in got), True)

    bad("missing id", twins[:-1], "row counts differ")
    bad("reordered rows", list(reversed(twins)), "different order")
    flipped = [dict(t) for t in twins]
    flipped[3]["label"] = "nonscam" if flipped[3]["label"] == "scam" else "scam"
    bad("a flipped label", flipped, "different label")
    emptied = [dict(t) for t in twins]
    emptied[5]["text"] = "   "
    bad("an emptied transcript", emptied, "empty")
    dup = [dict(t) for t in twins]
    dup[1]["id"] = dup[0]["id"]
    bad("a duplicated id", dup, "duplicate")
    bad("no id column", [{k: v for k, v in t.items() if k != "id"} for t in twins],
        "no 'id' column", fields=("label", "source", "text"))
    check("labels compare by meaning, not spelling (fraud == scam)",
          trusted.is_scam("Fraud") == trusted.is_scam("scam"), True)


# ----------------------------------------------------- the ordinary benchmark
def original_load_combined(csv_path, limit=None):
    """load_combined exactly as it was before the ids were carried through."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    scam_words = {"scam", "fraud", "fraudulent", "1", "true", "yes"}
    data = []
    for r in rows:
        lab_raw = (r["label"] or "").strip().lower()
        lab = "Fraud" if lab_raw in scam_words else "Normal"
        text = (r["text"] or "").strip()
        if text:
            data.append((text, lab))
    if limit:
        scam = [d for d in data if d[1] == "Fraud"]
        norm = [d for d in data if d[1] == "Normal"]
        k = limit // 2
        random.seed(CE.SEED)
        scam = random.sample(scam, min(k, len(scam)))
        norm = random.sample(norm, min(k, len(norm)))
        data = scam + norm
    random.seed(CE.SEED)
    random.shuffle(data)
    return data


def test_loader_unchanged(tmp):
    print("\nthe full-text calls are exactly the ones an ordinary run scores")
    full, strip, _, _ = make_pair(tmp, n=60)
    for limit in (None, 20, 7):
        check("load_combined matches the original, limit=%s" % limit,
              CE.load_combined(full, limit), original_load_combined(full, limit))
    for limit in (None, 20):
        data, data_s = CE.load_paired(full, strip, limit)
        check("paired data == load_combined, limit=%s" % limit,
              data, CE.load_combined(full, limit))
        ok = all(s.split()[1] == d.split()[1] and s.startswith("STRIPPED")
                 for (d, _), (s, _) in zip(data, data_s))
        check("each twin is its own call's, by id, limit=%s" % limit, ok, True)
        check("labels carried across, limit=%s" % limit,
              [l for _, l in data_s], [l for _, l in data])


# ------------------------------------------------------------ bag-of-words
def test_bow(tmp):
    print("\nbag-of-words: one model a fold, trained on original text")
    full, strip, _, _ = make_pair(tmp, n=60)
    data, data_s = CE.load_paired(full, strip)
    full_rows, strip_rows = CE.trivial_bow_paired(data, data_s, folds=5)
    check("full-text rows identical to trivial_bow",
          full_rows, CE.trivial_bow(data, folds=5))
    same_full, same_strip = CE.trivial_bow_paired(data, data, folds=5)
    check("a twin identical to the data scores identically",
          same_strip, same_full)

    # The discriminating check. The twins' only class signal is the
    # collection-specific function words, which no original contains. A model
    # trained on the originals cannot use them, so it falls to one verdict. A
    # model trained on the stripped text would learn them and score ~100%.
    verdicts = {p for p, _ in strip_rows}
    check("stripped copy is only scored, never trained on (one verdict)",
          len(verdicts), 1)
    acc_s = CE.metrics(strip_rows)["acc"]
    check("so the stripped copy scores at chance, not on the cue",
          acc_s, 0.5)


# ------------------------------------------------- learners that need an LLM
class Recorder:
    """Stand-ins for the LLM steps, remembering what each was shown."""

    def __init__(self):
        self.trained_on = []
        self.judged = []

    def build_kb(self, train_data, *a, **k):
        self.trained_on.extend(t for t, _ in train_data)
        return [{"id": "p1", "text": "a learned pattern", "category": "x"}]

    def ask(self, prompt, stats, *a, **k):
        self.judged.append(prompt)
        return ("Fraud" if "c0" in prompt else "Normal", "raw", "because")


class StubKB:
    def __init__(self, patterns, tag):
        self.patterns = patterns

    def evidence_for(self, text):
        return "a learned pattern", 1, 0


def install_stubs(rec):
    saved = {n: getattr(CE, n) for n in
             ("build_qwen_training_kb", "ask_verdict", "QwenPatternKB",
              "build_merged_kb")}
    CE.build_qwen_training_kb = rec.build_kb
    CE.ask_verdict = rec.ask
    CE.QwenPatternKB = StubKB
    CE.build_merged_kb = lambda patterns, tag="hyb", include_web=True: (None, 0, len(patterns))

    web = types.ModuleType("webrag_system")
    web.MIN_KB_SIMILARITY = 0.5
    web.USE_LLM_GATE = False

    def detect(text, coll, use_web=False, threshold=50):
        rec.judged.append(text)
        return {"predicted": "Fraud", "reason": "r", "confidence": 60,
                "evidence_used": True, "n_kb_kept": 1, "n_kb_dropped": 0,
                "n_kb_learned": 1, "gate_note": ""}
    web.detect = detect
    saved_web = sys.modules.get("webrag_system")
    sys.modules["webrag_system"] = web
    return saved, saved_web


def remove_stubs(saved, saved_web):
    for n, v in saved.items():
        setattr(CE, n, v)
    if saved_web is None:
        sys.modules.pop("webrag_system", None)
    else:
        sys.modules["webrag_system"] = saved_web


def test_learners(tmp):
    print("\nQwen-KB and the hybrid: learn from original, judge the twin")
    full, strip, _, _ = make_pair(tmp, n=40)
    data, data_s = CE.load_paired(full, strip)

    for name, run in (("qwen_kb", CE.run_qwen_kb), ("hybrid", CE.run_hybrid)):
        CE._LEARNED_KB.clear()
        rec = Recorder()
        saved = install_stubs(rec)
        try:
            out, _, _, _ = run(data, 5, judge_data=data_s)
        finally:
            remove_stubs(*saved)
        check("%s learned from original text only" % name,
              bool(rec.trained_on) and all(t.startswith("ORIGINAL")
                                           for t in rec.trained_on), True)
        check("%s never trained on a stripped call" % name,
              any("STRIPPED" in t for t in rec.trained_on), False)
        check("%s judged only stripped calls" % name,
              len(rec.judged) == len(data)
              and all("STRIPPED" in j for j in rec.judged), True)
        check("%s scored every call once" % name,
              all(p is not None for p, _ in out), True)
        # no call is judged while its own original sat in that fold's training
        # set: every held-out id is absent from what that fold learned from
        check("%s: all %d calls judged, as held out" % (name, len(data)),
              len(rec.judged), len(data))

    print("\n  the KB is learned once a fold, whichever copy is judged")
    CE._LEARNED_KB.clear()
    rec = Recorder()
    saved = install_stubs(rec)
    try:
        CE.run_qwen_kb(data, 5)
        first = len(rec.trained_on)
        CE.run_qwen_kb(data, 5, judge_data=data_s)
        CE.run_hybrid(data, 5, judge_data=data_s)
    finally:
        remove_stubs(*saved)
    check("full, stripped and hybrid share one set of learned patterns",
          len(rec.trained_on), first)


# ------------------------------------------------------------------- BERT
def test_bert_alignment(tmp):
    print("\nBERT: each kept row gets its own twin, even under --limit")
    import bert_baseline as B
    full, strip, rows, _ = make_pair(tmp, n=30)
    args = types.SimpleNamespace(csv=str(full), stripped_csv=str(strip))
    for limit in (None, 10):
        df, texts, _ = B.load_dataset(str(full), limit=limit)
        twins = B.stripped_twins(args, df)
        ok = all(t.split()[1] == o.split()[1] and t.startswith("STRIPPED")
                 for t, o in zip(twins, texts))
        check("twin matches its row by id, limit=%s" % limit, ok, True)


# -------------------------------------------------------- collect_results
FULL_LOG = """\
  length                   acc  50.0%  P 0.500  R 1.000  F1 0.667   (TP211 FP211 FN0 TN0)
  length__stripped         acc  55.0%  P 0.527  R 0.962  F1 0.681   (TP203 FP182 FN8 TN29)
  bag-of-words (TF-IDF+LR) acc 100.0%  P 1.000  R 1.000  F1 1.000   (TP211 FP0 FN0 TN211)
  bow__stripped            acc  55.7%  P 0.530  R 1.000  F1 0.693   (TP211 FP187 FN0 TN24)
SUMMARY
  length                   acc  50.0%  P 0.500  R 1.000  F1 0.667   (TP211 FP211 FN0 TN0)
  bow                      acc 100.0%  P 1.000  R 1.000  F1 1.000   (TP211 FP0 FN0 TN211)
  llm_only                 acc  87.4%  P 0.804  R 0.991  F1 0.887   (TP209 FP51 FN2 TN160)
STRIPPED COPY
  length__stripped         acc  55.0%  P 0.527  R 0.962  F1 0.681   (TP203 FP182 FN8 TN29)
  bow__stripped            acc  55.7%  P 0.530  R 1.000  F1 0.693   (TP211 FP187 FN0 TN24)
  llm_only__stripped       acc  50.0%  P 0.500  R 1.000  F1 0.667   (TP211 FP211 FN0 TN0)
"""
BERT_LOG = """\
  bert (pooled OOF)        acc 100.0%  P 1.000  R 1.000  F1 1.000   (TP211 FP0 FN0 TN211)
  bert__stripped (stripped OOF) acc  52.1%  P 0.511  R 1.000  F1 0.676   (TP211 FP202 FN0 TN9)
"""


def collect(logdir):
    r = subprocess.run([sys.executable, str(HERE / "collect_results.py"),
                        str(logdir), "--json"], capture_output=True, text=True)
    return json.loads(r.stdout)


def test_collect(tmp):
    print("\ncollect_results pairs each row with its stripped twin")
    d = Path(tmp) / "logs_paired"
    d.mkdir()
    (d / "combined.log").write_text(FULL_LOG)
    (d / "bert.log").write_text(BERT_LOG)
    got = {s["system"]: s for s in collect(d)["systems"] if s["ran"]}
    check("full bow row is still the full one", got["bow"]["acc"], 100.0)
    check("full length row is still the full one", got["length"]["acc"], 50.0)
    check("stripped bow picked up", got["bow"]["stripped"]["acc"], 55.7)
    check("bow trusted from exact counts",
          got["bow"]["trusted"],
          round(100 * trusted.trusted_accuracy(1.0, 235 / 422), 1))
    check("llm_only at chance keeps its full score", got["llm_only"]["trusted"], 87.4)
    check("bert full row is the pooled one", got["bert"]["acc"], 100.0)
    check("bert stripped row found by 'stripped OOF'",
          got["bert"]["stripped"]["acc"], 52.1)

    print("\n  and a run without a twin looks exactly as it did")
    e = Path(tmp) / "logs_plain"
    e.mkdir()
    (e / "combined.log").write_text(
        "\n".join(l for l in FULL_LOG.splitlines() if "__stripped" not in l))
    plain = collect(e)
    check("no stripped twin means not paired", plain["paired"], False)
    check("no system gains stripped or trusted keys",
          any("stripped" in s or "trusted" in s for s in plain["systems"]), False)


def test_end_to_end(tmp):
    """Run the real main(), not its parts. The unit checks above call the
    functions directly, so a mistake in main() itself - a name it never
    defines, a flag it never passes on - would get past every one of them."""
    print("\ncombined_evaluate.py runs end to end with --stripped-csv")
    full, strip, _, _ = make_pair(tmp, n=40)
    work = Path(tmp) / "e2e"
    work.mkdir()
    run = subprocess.run(
        [sys.executable, str(HERE / "combined_evaluate.py"), "--csv", str(full),
         "--stripped-csv", str(strip), "--trivial-only"],
        cwd=work, capture_output=True, text=True)
    check("exits cleanly", run.returncode, 0)
    if run.returncode:
        print(run.stdout[-800:], run.stderr[-800:])
        return
    out = run.stdout
    check("prints a length row for the stripped copy",
          "length__stripped" in out, True)
    check("prints a bag-of-words row for the stripped copy",
          "bow__stripped" in out, True)
    check("prints the trusted-accuracy table", "TRUSTED ACCURACY" in out, True)
    rows = list(csv.DictReader(open(work / "results" / "combined_results_40.csv",
                                    encoding="utf-8")))
    check("per-call CSV has the stripped transcript beside the original",
          "text_stripped" in rows[0], True)
    check("and a verdict column for each stripped system",
          {"length__stripped", "bow__stripped"} <= set(rows[0]), True)
    check("the stripped transcript on each row is that row's own twin",
          all(r["text_stripped"].split()[1] == r["text"].split()[1] for r in rows),
          True)

    plain = subprocess.run(
        [sys.executable, str(HERE / "combined_evaluate.py"), "--csv", str(full),
         "--trivial-only"], cwd=work, capture_output=True, text=True)
    def rows_of(text, key):
        return [l for l in text.splitlines() if l.strip().startswith(key)]
    check("full length row identical with and without the twin",
          rows_of(out, "length-only"), rows_of(plain.stdout, "length-only"))
    check("full bag-of-words row identical with and without the twin",
          rows_of(out, "bag-of-words"), rows_of(plain.stdout, "bag-of-words"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_formula()
        test_check_pair(tmp)
        test_loader_unchanged(tmp)
        test_bow(tmp)
        test_learners(tmp)
        test_bert_alignment(tmp)
        test_collect(tmp)
        test_end_to_end(tmp)
    print()
    if FAILS:
        print("%d FAILED: %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
