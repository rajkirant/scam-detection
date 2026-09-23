#!/usr/bin/env python3
"""
ollama_ctx.py - size the context window before asking Ollama anything.

The failure this exists to stop is silent. Ollama's context window defaults to
2048 tokens unless the model's Modelfile raises it, and a prompt longer than
the window is truncated from the FRONT. The instructions are at the front. So
an overlong call does not error, does not warn, and does not come back empty:
it comes back as a fluent, plausible verdict on the first ~1,500 words of the
transcript, with the question that was asked about it cut off.

How much of a problem that is depends entirely on the dataset:

    datasets/scamai_full_1000.csv      86% of calls over 2048, median ~4,300
    datasets/everything_7013.csv       27% over, p90 ~3,200
    datasets/scambait_bank_422.csv      0% over, median ~180
    datasets/scambait_synthetic_196.csv 0% over
    datasets/zhi_english_646.csv        0% over

so a result measured on the bank or synthetic sets is unaffected, and one
measured on scamai_full_1000 without an explicit window was mostly measured on
truncated calls.

Every Ollama client in this project routes its window through fit_num_ctx, so
the window is always explicit and an overflow is always said out loud.

SCAM_NUM_CTX sets the ceiling (default 8192). It is a ceiling rather than a
fixed size because the window costs VRAM: qwen2.5:14b is already ~9.5 GB of an
11.4 GB card, so a short call asks for a small window and only a long one pays
for a big one.
"""

import os
import sys

# Ollama's own default, and the floor here: never ask for less than we would
# have got by saying nothing.
FLOOR = 2048
STEP = 1024
DEFAULT_CAP = max(FLOOR, int(os.environ.get("SCAM_NUM_CTX", "8192")))

# Headroom on the estimate before a window is sized from it.
#
# estimate_tokens is a heuristic and it runs low. Measured on row 0 of
# scamai_hard_subset.csv against qwen2.5:14b: estimated 15,293, actually read
# 15,933, so the guess was 4.2% under. Rounding up to STEP does not reliably
# cover that - a prompt estimated at 7,800 rounds to an 8,192 window and then
# really needs 8,427 - and the cost of being wrong is not the overflow, it is
# half the window (see _warn_overflow). So the sizing pays 15% up front, which
# is cheap, rather than risk losing half the call, which is not.
SAFETY = 1.15

# where -> [times it overflowed, worst prompt seen]
_overflows = {}

# The largest window asked for so far in this process. Windows only grow.
#
# Ollama reloads the model whenever a request's num_ctx differs from the one
# the loaded runner was started with - for qwen2.5:14b that is ~9 GB pushed
# back onto the GPU. Sizing every call to its own prompt, in 1024-token steps,
# changed the window on 88% of consecutive calls on scamai_full_1000 and 60%
# on everything_7013: thousands of reloads a run. A bigger window holds a
# smaller prompt just as well, so once a window has been needed it is kept,
# and a run reloads at most once per step it grows through (about 9).
#
# SCAM_STICKY_CTX=0 goes back to sizing each call on its own - worth it only
# if a big window pushes the model partly off the GPU (see `ollama ps`).
STICKY = os.environ.get("SCAM_STICKY_CTX", "1") != "0"
_sticky = 0


def estimate_tokens(text):
    """Tokens in `text`, erring high.

    Two rules of thumb disagree depending on the text, so the larger wins:
    1.4 per word is right for prose, and one per four characters is right for
    the transcripts here, which are full of short words, digits and names that
    split into several tokens each. Guessing low is the expensive mistake -
    it is the one that lets a prompt overflow quietly.
    """
    text = text or ""
    return max(int(len(text.split()) * 1.4), len(text) // 4) + 1


def fit_num_ctx(prompt, reply_tokens=300, cap=None, where=""):
    """The window to ask for, big enough for this prompt and its reply.

    Rounded up to a multiple of STEP, floored at Ollama's own default and
    capped at SCAM_NUM_CTX. When even the cap is not enough, this says so on
    stderr rather than letting the truncation happen unremarked - the caller
    still gets a number back, because a truncated answer with a warning beside
    it is more use than a crash, but nobody should be able to read the result
    and not know.
    """
    global _sticky
    cap = int(cap or DEFAULT_CAP)
    needed = int(estimate_tokens(prompt) * SAFETY) + int(reply_tokens or 0)
    want = min(cap, max(FLOOR, -(-needed // STEP) * STEP))
    if needed > cap:
        _warn_overflow(where or "an ollama call", needed, cap)
    if STICKY:
        # never shrink - but never exceed this caller's own cap either
        want = min(cap, max(want, _sticky))
        _sticky = max(_sticky, want)
    return want


def check_num_ctx(prompt, num_ctx, reply_tokens=300, where=""):
    """Same warning for a caller that has already decided its own window."""
    needed = int(estimate_tokens(prompt) * SAFETY) + int(reply_tokens or 0)
    if num_ctx and needed > int(num_ctx):
        _warn_overflow(where or "an ollama call", needed, int(num_ctx))
    return num_ctx


def _warn_overflow(where, needed, window):
    seen = _overflows.setdefault(where, [0, 0])
    seen[0] += 1
    seen[1] = max(seen[1], needed)
    # Loud the first time, then every fiftieth: a benchmark over a thousand
    # calls should not be drowned, but it must not look clean either.
    if seen[0] == 1 or seen[0] % 50 == 0:
        sys.stderr.write(
            "  WARNING [%s] prompt is about %d tokens, window is %d - Ollama "
            "cuts the FRONT off, taking the instructions with it. This is "
            "overflow #%d here. Raise SCAM_NUM_CTX, or shorten the "
            "transcript.\n" % (where, needed, window, seen[0]))
        sys.stderr.flush()


def overflows():
    """{where: (count, worst)} - for a run to report at the end."""
    return {k: tuple(v) for k, v in _overflows.items()}


def report(stream=None):
    """One line per place that overflowed, or nothing at all if none did."""
    out = stream or sys.stderr
    if not _overflows:
        return False
    out.write("\n  CONTEXT OVERFLOWS - these answers were given on truncated "
              "prompts:\n")
    for where, (count, worst) in sorted(_overflows.items()):
        out.write("    %-40s %5d call(s), worst ~%d tokens\n"
                  % (where, count, worst))
    out.write("    Re-run with a bigger SCAM_NUM_CTX before quoting any of "
              "these numbers.\n")
    out.flush()
    return True
