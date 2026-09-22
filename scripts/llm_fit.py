#!/usr/bin/env python3
"""
llm_fit.py - fit a prompt for the local LLM on a dataset, keep it, and
measure what it bought.

Read this paragraph before using it. **Nothing here fine-tunes anything.**
The weights of the model Ollama is holding do not move, and they cannot be
moved from this project. What is fitted is the prompt: worked examples drawn
from a dataset, and a rubric the model writes for itself after reading them.
That is in-context learning, and it is the only kind of training a frozen
local model can be given from a web page. The page and the CLI both say so,
because "trained" would be a lie and this baseline exists to be compared with
a BERT that really was trained.

Given that, the shape is the same as the other three pages on purpose: fit on
a dataset, keep it in models/<name>/, pick it from a list, put a call to it.

A fitted prompt has three parts, each optional:

  shots     k calls from the fitting split, balanced between the two classes,
            with their real answers attached. Excerpts, not whole calls - a
            median call here is about 5,000 tokens and four of them would
            crowd out the call being judged.
  rubric    what the model itself says separates the two classes, written
            after it is shown those examples. One extra call at fit time,
            and then it is a fixed piece of text.
  guidance  the standing instructions box from the page, carried into the
            profile so a correction survives a page reload.

What makes this worth doing rather than guessing is the last step: the
holdout is scored TWICE, once with the fitted prompt and once with the bare
llm_only prompt, on the same calls in the same order. The number that matters
is the difference. A fitted prompt that does not beat the control has cost
you context window and bought nothing, and this is the only way to find that
out.

That costs two LLM calls per holdout call, which is why the holdout here is
small by default and the run prints its own bill before it starts.

Usage:
    python scripts/llm_fit.py fit --csv datasets/zhi_english_646.csv \
        --name zhi-prompt --shots 4 --rubric
    python scripts/llm_fit.py show --name zhi-prompt
    python scripts/llm_fit.py models
    python scripts/llm_fit.py delete --name zhi-prompt

    python scripts/llm_judge.py --profile zhi-prompt --text "Hello..."

Standard library only, like llm_judge.py: web_ui.py imports this at module
level and has to start without the venv.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_judge                                           # noqa: E402

# The dataset loader and the confusion matrix live in eval_common, so all
# four pages agree on what a score means and the numbers can be read side
# by side. Re-exported here because this module's own callers use them.
import eval_common as EC                                    # noqa: E402

to_binary = EC.to_binary
load_rows = EC.load_rows
metrics = EC.metrics
fmt = EC.fmt


# What marks a directory in models/ as one of these. The other three pages
# look for config.json, bow.joblib and length.json, so the four kinds sit
# side by side in models/ without seeing each other.
PROFILE_FILE = "llm.json"
META_FILE = "meta.json"

MAX_SHOTS = 12
MAX_SHOT_WORDS = 400
RUBRIC_MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------

def checked_name(name):
    from bert_classify import checked_name as cn
    return cn(name)


def model_dir(name):
    return MODELS_DIR / checked_name(name)


def load_profile(name):
    d = model_dir(name)
    if not (d / PROFILE_FILE).exists():
        raise SystemExit("no fitted prompt at models/%s - fit one first"
                         % checked_name(name))
    prof = json.loads((d / PROFILE_FILE).read_text(encoding="utf-8"))
    prof.setdefault("name", checked_name(name))
    return prof


def read_meta(d):
    try:
        return json.loads((d / META_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_models():
    """Every fitted prompt in models/, newest first."""
    out = []
    if not MODELS_DIR.is_dir():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / PROFILE_FILE).exists():
            continue
        meta = read_meta(d)
        out.append({
            "name": d.name,
            "dataset": meta.get("dataset"),
            "base_model": meta.get("base_model"),
            "shots": meta.get("shots"),
            "rubric": meta.get("rubric"),
            "guided": meta.get("guided"),
            "holdout": meta.get("holdout"),
            "control": meta.get("control"),
            "gain": meta.get("gain"),
            "holdout_rows": meta.get("holdout_rows"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "trained_at": meta.get("trained_at"),
            "elapsed_s": meta.get("elapsed_s"),
            "described": bool(meta),
        })
    out.sort(key=lambda m: m.get("trained_at") or "", reverse=True)
    return out


def delete_model(name):
    import shutil
    d = model_dir(name)
    if d.resolve().parent != MODELS_DIR.resolve() or not d.is_dir():
        raise SystemExit("no such model: %s" % name)
    if not (d / PROFILE_FILE).exists():
        raise SystemExit("models/%s is not a fitted prompt" % name)
    shutil.rmtree(d)
    return {"deleted": checked_name(name)}


def excerpt(text, words):
    """The opening `words` words of a call.

    The opening rather than the middle because that is where the pretext is
    established - who the caller claims to be and what they claim to want -
    and that is the part a worked example is meant to demonstrate. It is a
    choice, not a law: --shot-words makes the excerpt longer at the cost of
    context window.
    """
    ws = (text or "").split()
    if len(ws) <= words:
        return " ".join(ws)
    return " ".join(ws[:words]) + " ..."


def pick_shots(rows, k, shot_words, seed):
    """k worked examples, balanced between the classes.

    Short calls are preferred, and not for tidiness: every word of an example
    is a word of context window that the call being judged does not get, and
    a prompt that overflows loses its examples first (see
    llm_judge.build_prompt). Among the short ones the choice is seeded and
    random, so a fit is repeatable without always landing on the same handful.
    """
    if k <= 0:
        return []
    rnd = random.Random(seed)
    out = []
    for label in (1, 0):
        side = [r for r in rows if r["label"] == label]
        if not side:
            continue
        side.sort(key=lambda r: r["words"])
        # the shortest half, then a seeded draw from it
        pool = side[: max(1, len(side) // 2)]
        rnd.shuffle(pool)
        out.extend(pool[: -(-k // 2)])
    rnd.shuffle(out)
    out = out[:k]
    return [{"id": r["id"],
             "verdict": "Fraud" if r["label"] else "Normal",
             "words": r["words"],
             "excerpt": excerpt(r["text"], shot_words)} for r in out]


RUBRIC_ASK = (
    "You are building a scam detection rubric for a corpus of phone call "
    "transcripts. Below are labelled excerpts from that corpus: some are "
    "scam calls, some are legitimate.\n\n"
    "Write the rules that separate the two IN THIS CORPUS. Be concrete and "
    "be specific to what you can actually see in these calls - name the "
    "moves, the requests and the pretexts that mark a scam here, and name "
    "what legitimate calls in this corpus look like so they are not "
    "mistaken for scams.\n\n"
    "Six to ten short bullet points. No preamble, no conclusion, just the "
    "rules. Do not mention these examples or their numbering - the rules "
    "will be read by someone who cannot see them.\n\n"
    "<<<EXAMPLES\n%s\nEXAMPLES>>>\n")


def write_rubric(shots, model, num_ctx, timeout, temperature=0.0):
    """Ask the model what separates the two classes, from the examples.

    This is the step that makes the thing feel like training, and it is worth
    being clear about what it actually is: the model is writing a piece of
    text that will be pasted into every later prompt. It is not learning. It
    is taking notes, and the notes are as good as the examples it saw.
    """
    import ollama_ctx

    body = "\n\n".join(
        "[%d] Answer: %s\n%s" % (i, sh["verdict"], sh["excerpt"])
        for i, sh in enumerate(shots, 1))
    prompt = RUBRIC_ASK % body
    ctx = ollama_ctx.fit_num_ctx(prompt, RUBRIC_MAX_TOKENS, cap=num_ctx,
                                 where="llm_fit rubric")
    text = llm_judge.ask(prompt, model=model, max_tokens=RUBRIC_MAX_TOKENS,
                         temperature=temperature, num_ctx=ctx,
                         timeout=timeout).strip()
    return text


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def score_split(rows, profile, model, max_tokens, num_ctx, timeout, label):
    """Run the LLM over the holdout and report how it did.

    The window is sized per call rather than fixed: a prompt with four worked
    examples in it is a good deal longer than the same call alone, and the
    whole point of measuring the fitted prompt against the control is that
    the only difference between the two runs is the prompt.
    """
    import ollama_ctx

    preds, truths = [], []
    unreadable = truncated = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        prompt = llm_judge.build_prompt(r["text"], None, profile)
        ctx = ollama_ctx.fit_num_ctx(prompt, max_tokens, cap=num_ctx,
                                     where="llm_fit %s" % label)
        try:
            out = llm_judge.judge(r["text"], model=model,
                                  max_tokens=max_tokens, num_ctx=ctx,
                                  temperature=0.0, timeout=timeout,
                                  profile=profile)
        except RuntimeError as e:
            print("    %3d/%d  ERROR %s" % (i, len(rows), e))
            preds.append(None)
            truths.append(r["label"])
            continue
        pred = None if out["unreadable"] else int(out["scam"])
        preds.append(pred)
        truths.append(r["label"])
        unreadable += out["unreadable"]
        truncated += bool(out["prompt_truncated"])
        mark = "?" if pred is None else ("ok" if pred == r["label"] else "X ")
        print("    %3d/%d  %-2s  said %-9s  really %-10s  %s"
              % (i, len(rows), mark,
                 out["verdict"] or "unreadable",
                 "Fraud" if r["label"] else "Normal",
                 "%d words" % r["words"]
                 + (", TRUNCATED" if out["prompt_truncated"] else "")))
        sys.stdout.flush()
    m = metrics(truths, preds)
    m["truncated"] = truncated
    m["elapsed_s"] = round(time.time() - t0, 1)
    return m, preds, truths


def run_fit(args):
    name = checked_name(args.name)
    dest = model_dir(name)
    if dest.exists() and not args.overwrite:
        raise SystemExit("models/%s already exists - pick another name, or "
                         "delete it first" % name)
    if args.shots > MAX_SHOTS:
        raise SystemExit("at most %d worked examples - past that the call "
                         "being judged is crowded out of the window"
                         % MAX_SHOTS)
    if args.shot_words > MAX_SHOT_WORDS:
        raise SystemExit("at most %d words per example" % MAX_SHOT_WORDS)

    guidance = (args.guidance or "")
    if args.guidance_file:
        guidance = Path(args.guidance_file).read_text(encoding="utf-8")
    guidance = guidance.strip()

    print("Loading dataset")
    rows = load_rows(args.csv, args.text_col, args.label_col, args.limit)
    if len({r["label"] for r in rows}) < 2:
        raise SystemExit("the dataset has only one class in it - there is "
                         "nothing to show the model")
    print()

    rnd = random.Random(args.seed)
    shuffled = rows[:]
    rnd.shuffle(shuffled)
    n_hold = min(args.holdout_calls, max(0, len(shuffled) - args.shots - 2))
    # balanced holdout, so an accuracy on it means something
    pos = [r for r in shuffled if r["label"]]
    neg = [r for r in shuffled if not r["label"]]
    hold = pos[: n_hold // 2] + neg[: n_hold - n_hold // 2]
    rnd.shuffle(hold)
    held_ids = {id(r) for r in hold}
    fit_rows = [r for r in shuffled if id(r) not in held_ids]

    calls = len(hold) * 2 + (1 if args.rubric else 0)
    print("==> fit")
    print("  %d calls to draw examples from, %d held out to score on"
          % (len(fit_rows), len(hold)))
    print("  this will make about %d calls to %s: %d holdout calls scored "
          "twice%s" % (calls, args.model, len(hold),
                       ", plus one to write the rubric" if args.rubric else ""))
    print("  (the second run is the bare llm_only prompt on the same calls - "
          "without it\n   there is no way to know whether the fitting bought "
          "anything)")
    print()

    t0 = time.time()
    shots = pick_shots(fit_rows, args.shots, args.shot_words, args.seed)
    if shots:
        print("  %d worked example(s) picked:" % len(shots))
        for sh in shots:
            print("    id %-14s %-7s %5d words, excerpted to %d"
                  % (sh["id"], sh["verdict"], sh["words"],
                     len(sh["excerpt"].split())))
    else:
        print("  no worked examples (--shots 0)")
    print()

    rubric = ""
    if args.rubric:
        if not shots:
            raise SystemExit("a rubric is written from the worked examples - "
                             "--rubric needs --shots")
        print("  asking %s to write the rubric from those examples..."
              % args.model)
        sys.stdout.flush()
        try:
            rubric = write_rubric(shots, args.model, args.num_ctx,
                                  args.timeout, args.temperature)
        except RuntimeError as e:
            raise SystemExit(str(e))
        print()
        for line in rubric.splitlines():
            print("    " + line)
        print()

    profile = {"name": name, "shots": shots, "rubric": rubric,
               "guidance": guidance, "base_model": args.model}

    sample = llm_judge.build_prompt("", None, profile)
    overhead = llm_judge.estimate_tokens(sample)
    print("  the fitted prompt adds about %d tokens to every call" % overhead)
    print()

    holdout = control = gain = None
    if hold:
        print("  scoring the holdout with the FITTED prompt")
        holdout, hp, truths = score_split(hold, profile, args.model,
                                          args.max_tokens, args.num_ctx,
                                          args.timeout, "fitted")
        print()
        print("  scoring the same calls with the BARE llm_only prompt")
        control, cp, _ = score_split(hold, None, args.model, args.max_tokens,
                                     args.num_ctx, args.timeout, "control")
        print()
        print(fmt("fitted prompt", holdout))
        print(fmt("llm_only control", control))
        gain = round(holdout["acc"] - control["acc"], 4)
        moved = sum(1 for a, b in zip(hp, cp) if a != b)
        print()
        print("  the fitting moved %d of %d verdicts, and changed accuracy by "
              "%+.1f points" % (moved, len(hold), 100 * gain))
        if gain <= 0:
            print("  NOTE the fitted prompt did not beat the control on this "
                  "holdout. It costs\n       ~%d tokens of context per call "
                  "and bought nothing measurable here."
                  % overhead)
        if holdout.get("truncated"):
            print("  NOTE %d fitted prompt(s) were truncated by ollama - the "
                  "examples go first,\n       so those calls were scored on "
                  "something closer to the control."
                  % holdout["truncated"])
        if len(hold) < 30:
            print("  NOTE %d calls is a small holdout. A few points either "
                  "way is noise at this\n       size - raise "
                  "--holdout-calls before quoting the difference."
                  % len(hold))
    elapsed = time.time() - t0

    dest.mkdir(parents=True, exist_ok=True)
    (dest / PROFILE_FILE).write_text(json.dumps(profile, indent=2),
                                     encoding="utf-8")
    meta = {
        "name": name,
        "kind": "llm",
        "dataset": args.csv,
        "rows": len(rows),
        "base_model": args.model,
        "shots": len(shots),
        "shot_words": args.shot_words,
        "rubric": bool(rubric),
        "guided": bool(guidance),
        "prompt_tokens": overhead,
        "holdout_rows": len(hold),
        "holdout": holdout,
        "control": control,
        "gain": gain,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print("  ok fitted  models/%s" % name)
    print("  the weights did not move. What is saved is a prompt.")
    print("  use it:  python scripts/llm_judge.py --profile %s --text \"...\""
          % name)
    return meta


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# evaluate - the whole dataset, not one call
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Put every call in a dataset to the model and score the verdicts.

    This is the `llm_only` row on the Benchmark page, run from this page and
    optionally under a fitted prompt. Two things about it are not like the
    other three evaluate runs:

      it is slow and it costs   One generation per call. On a 14B model that
                                is seconds each, so a thousand-call dataset
                                is hours. --limit takes a class-balanced head
                                and is the right default habit here.

      it can fail to answer     A reply that cannot be parsed is not a
                                verdict. Those calls are counted and left out
                                of the scores rather than folded into Normal,
                                which is what would quietly turn every
                                unreadable answer into a false negative.
    """
    import ollama_ctx

    profile = load_profile(args.profile) if args.profile else None

    print("Loading dataset")
    rows = EC.load_rows(args.csv, args.text_col, args.label_col, args.limit)
    print()
    print("==> score  %s over %d calls%s"
          % (args.model, len(rows),
             ", under models/%s" % profile["name"] if profile else
             ", bare llm_only prompt"))
    if profile and profile.get("dataset_hint") == args.csv:
        print("  NOTE this prompt was fitted on this dataset.")
    print("  one generation per call, so this is not quick. Ctrl-C or Stop "
          "leaves the\n  partial results in place.")

    truths = [r["label"] for r in rows]
    prog = EC.Progress(len(rows), every=1)
    preds, verdicts, reasons = [], [], []
    truncated = 0
    t0 = time.time()
    for r in rows:
        prompt = llm_judge.build_prompt(r["text"], None, profile)
        ctx = ollama_ctx.fit_num_ctx(prompt, args.max_tokens,
                                     cap=args.num_ctx, where="llm evaluate")
        try:
            out = llm_judge.judge(r["text"], model=args.model,
                                  max_tokens=args.max_tokens, num_ctx=ctx,
                                  temperature=args.temperature,
                                  timeout=args.timeout, profile=profile)
        except RuntimeError as e:
            print("    ERROR %s" % e)
            preds.append(None)
            verdicts.append("error")
            reasons.append(str(e)[:200])
            prog.tick(r, None)
            continue
        pred = None if out["unreadable"] else int(out["scam"])
        preds.append(pred)
        verdicts.append(out["verdict"] or "unreadable")
        reasons.append(" ".join((out["reason"] or "").split())[:300])
        truncated += bool(out["prompt_truncated"])
        prog.tick(r, pred)
    elapsed = time.time() - t0

    m = EC.metrics(truths, preds)
    base = EC.baselines(truths)
    extra = []
    if truncated:
        extra.append("%d prompt(s) were truncated by ollama - those verdicts "
                     "are about part of the call" % truncated)
    EC.report(m, base, elapsed, len(rows), extra)
    if args.out:
        EC.write_results(args.out, args.profile or args.model, args.csv, rows,
                         preds, m, base, elapsed,
                         {"verdict": verdicts, "reason": reasons},
                         {"kind": "llm", "base_model": args.model,
                          "profile": args.profile,
                          "shots": len((profile or {}).get("shots") or []),
                          "rubric": bool((profile or {}).get("rubric")),
                          "truncated": truncated})
    return m


def run_show(args):
    prof = load_profile(args.name)
    meta = read_meta(model_dir(args.name))
    if args.json:
        print(json.dumps({"profile": prof, "meta": meta}, indent=2))
        return
    bar = "=" * 74
    print(bar)
    print("models/%s   fitted prompt for %s"
          % (prof["name"], prof.get("base_model") or "?"))
    print(bar)
    print("  dataset    %s" % meta.get("dataset"))
    print("  adds       ~%s tokens to every call" % meta.get("prompt_tokens"))
    if meta.get("holdout") and meta.get("control"):
        print(fmt("fitted prompt", meta["holdout"]))
        print(fmt("llm_only control", meta["control"]))
        print("  gain       %+.1f points on %s held-out calls"
              % (100 * (meta.get("gain") or 0), meta.get("holdout_rows")))
    print()
    for sh in prof.get("shots") or []:
        print("  [%s] %s" % (sh["verdict"], sh["excerpt"][:160]))
    if prof.get("rubric"):
        print()
        print("  rubric:")
        for line in prof["rubric"].splitlines():
            print("    " + line)
    if prof.get("guidance"):
        print()
        print("  standing instructions:")
        for line in prof["guidance"].splitlines():
            print("    " + line)
    print(bar)


def run_models(args):
    models = list_models()
    if args.json:
        print(json.dumps(models, indent=2))
        return models
    if not models:
        print("  no fitted prompts in models/ yet")
        return models
    for m in models:
        acc = m["holdout"] and m["holdout"].get("acc")
        print("  %-24s %-30s %2s shot(s)%s  %s"
              % (m["name"], m["dataset"] or "", m["shots"] or 0,
                 " + rubric" if m["rubric"] else "         ",
                 ("holdout acc %.1f%%, %+.1f vs control"
                  % (100 * acc, 100 * (m["gain"] or 0))) if acc else ""))
    return models


def build_parser():
    p = argparse.ArgumentParser(
        description="fit a prompt for the local LLM on a dataset - worked "
                    "examples and a rubric, not fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    for word in ("fit", "train"):
        tr = sub.add_parser(word, help="build a prompt and measure it")
        tr.add_argument("--csv", required=True)
        tr.add_argument("--name", required=True, help="saved as models/<name>/")
        tr.add_argument("--shots", type=int, default=4,
                        help="worked examples in the prompt (0 for none)")
        tr.add_argument("--shot-words", type=int, default=120,
                        help="words kept from each example")
        tr.add_argument("--rubric", action="store_true",
                        help="have the model write the rules from those "
                             "examples, and keep them in the prompt")
        tr.add_argument("--guidance", default=None,
                        help="standing instructions to carry in the profile")
        tr.add_argument("--guidance-file", default=None)
        tr.add_argument("--holdout-calls", type=int, default=20,
                        help="calls to score on - EACH one costs two LLM "
                             "calls, fitted and control")
        tr.add_argument("--model", default=llm_judge.DEFAULT_MODEL)
        tr.add_argument("--num-ctx", type=int,
                        default=llm_judge.DEFAULT_NUM_CTX,
                        help="the ceiling; each call asks for what it needs")
        tr.add_argument("--max-tokens", type=int,
                        default=llm_judge.DEFAULT_MAX_TOKENS)
        tr.add_argument("--temperature", type=float, default=0.0)
        tr.add_argument("--timeout", type=int, default=300)
        tr.add_argument("--seed", type=int, default=42)
        tr.add_argument("--limit", type=int, default=None)
        tr.add_argument("--text-col", default=None)
        tr.add_argument("--label-col", default=None)
        tr.add_argument("--overwrite", action="store_true")
        tr.set_defaults(func=run_fit)

    ev = sub.add_parser("evaluate", help="score a whole dataset")
    ev.add_argument("--csv", required=True)
    ev.add_argument("--profile", default=None,
                    help="score under a fitted prompt; omit for the bare "
                         "llm_only control")
    ev.add_argument("--limit", type=int, default=None,
                    help="a class-balanced head - one generation per call, "
                         "so this is the flag that decides what it costs")
    ev.add_argument("--model", default=llm_judge.DEFAULT_MODEL)
    ev.add_argument("--num-ctx", type=int, default=llm_judge.DEFAULT_NUM_CTX)
    ev.add_argument("--max-tokens", type=int,
                    default=llm_judge.DEFAULT_MAX_TOKENS)
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--timeout", type=int, default=300)
    ev.add_argument("--text-col", default=None)
    ev.add_argument("--label-col", default=None)
    ev.add_argument("--out", default=None,
                    help="write <out>.json and a per-call CSV in results/")
    ev.set_defaults(func=run_evaluate)

    sh = sub.add_parser("show", help="what is in a fitted prompt")
    sh.add_argument("--name", required=True)
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=run_show)

    ls = sub.add_parser("models", help="what prompts are fitted")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=run_models)

    rm = sub.add_parser("delete", help="remove one")
    rm.add_argument("--name", required=True)
    rm.set_defaults(func=lambda a: print(json.dumps(delete_model(a.name))))

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
