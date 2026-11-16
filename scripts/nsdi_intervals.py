#!/usr/bin/env python3
"""NSDI intervals and expected precision for Section 6.3 and Table 7.

Counts come from Table 4 (100 scam and 100 legitimate calls in Support-200).
Recall and specificity are drawn from beta distributions given by the counts,
with a Jeffreys prior (Beta(k + 0.5, n - k + 0.5)), and the NSDI is computed for
each draw. Expected precision uses Equation 3; for a specificity of 1.00 the
lower bound uses the 95% Wilson interval (0.963 for 100 of 100).
"""
import numpy as np

COUNTS = {"LLM-only": (97, 81), "LLM, fitted prompt": (82, 100), "ScamRAG": (76, 93)}
N, DRAWS = 100, 200_000
rng = np.random.default_rng(42)

for name, (tp, tn) in COUNTS.items():
    r = rng.beta(tp + 0.5, N - tp + 0.5, DRAWS)
    s = rng.beta(tn + 0.5, N - tn + 0.5, DRAWS)
    lo, hi = np.percentile(100 * np.sqrt(r * s), [2.5, 97.5])
    print(f"{name:20s} NSDI {100 * np.sqrt(tp / N * tn / N):5.1f}  95% interval {lo:5.1f} to {hi:5.1f}")

def precision(r, s, p):
    return r * p / (r * p + (1 - s) * (1 - p))

for name, (tp, tn) in COUNTS.items():
    s = 0.963 if tn == N else tn / N   # Wilson lower bound when no false alarm
    vals = [100 * precision(tp / N, s, p) for p in (0.001, 0.01, 0.1)]
    print(f"{name:20s} expected precision at p = 0.001, 0.01, 0.1: " + ", ".join(f"{v:.1f}%" for v in vals))
