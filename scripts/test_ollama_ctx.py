"""
Offline tests for the Ollama context window.

No Ollama, no network, no GPU: `requests` and `chromadb` are stubbed so every
real client runs and the payload it would have posted is captured. Takes under
a second and runs anywhere.

What it guards is the bug in ollama_ctx.py's docstring: Ollama defaults the
context window to 2048 and truncates an overlong prompt from the FRONT, where
the instructions are, without erroring. Every client must therefore send an
explicit num_ctx, big enough for the prompt it is sending. A regression here
does not break a test elsewhere - it quietly changes what the models were
asked - so it is checked directly.

Run from project root:
    python scripts/test_ollama_ctx.py
"""

import io
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FAILURES = []
SENT = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        FAILURES.append(name)


# --------------------------------------------------------------- the stubs

class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "Answer: Fraud\nReason: because."}


def _post(url, json=None, timeout=None, **kw):
    SENT.append(json)
    return _Resp()


_requests = types.ModuleType("requests")
_requests.post = _requests.get = _post
_requests.exceptions = types.SimpleNamespace(
    ConnectionError=type("ConnectionError", (Exception,), {}),
    HTTPError=type("HTTPError", (Exception,), {}),
    RequestException=type("RequestException", (Exception,), {}),
    Timeout=type("Timeout", (Exception,), {}))
sys.modules["requests"] = _requests

for _name in ("chromadb", "chromadb.utils"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["chromadb.utils"].embedding_functions = types.SimpleNamespace(
    SentenceTransformerEmbeddingFunction=lambda **kw: None)
sys.modules["chromadb"].PersistentClient = lambda **kw: None
sys.modules["chromadb"].utils = sys.modules["chromadb.utils"]

import ollama_ctx

# LONG is about the length of the median row in scamai_hard_subset.csv (~5k
# tokens): well past Ollama's 2048 default, but inside the 8192 cap, so it is
# the case where sizing the window has to actually work. HUGE is past the cap,
# the case that has to warn instead.
SHORT = "hello this is a short call"
LONG = "hello this is your bank calling about a payment on your card " * 350
HUGE = "hello this is your bank calling about a payment on your card " * 3000


def last_options():
    return (SENT[-1] or {}).get("options", {})


# ------------------------------------------------------------- the helper
print("\nollama_ctx")
check("a short prompt asks for Ollama's own floor",
      ollama_ctx.fit_num_ctx(SHORT, 300, where="t") == ollama_ctx.FLOOR,
      ollama_ctx.fit_num_ctx(SHORT, 300, where="t"))

want = ollama_ctx.fit_num_ctx(LONG, 300, where="t")
need = ollama_ctx.estimate_tokens(LONG) + 300
check("a long prompt gets a window that fits it", need <= want <= ollama_ctx.DEFAULT_CAP,
      "needed %d, asked %d, cap %d" % (need, want, ollama_ctx.DEFAULT_CAP))

check("a prompt past the cap is capped, not grown",
      ollama_ctx.fit_num_ctx(HUGE, 300, where="t") == ollama_ctx.DEFAULT_CAP)

# The estimate runs low against a real tokenizer - 15,933 read against 15,293
# estimated, measured on qwen2.5:14b - and missing the window costs half of
# it, so the window has to carry headroom over the estimate, not just round up.
check("the window carries headroom over the estimate",
      ollama_ctx.fit_num_ctx(LONG, 300, where="t")
      >= ollama_ctx.estimate_tokens(LONG) * 1.1 + 300,
      "estimate %d, window %d" % (ollama_ctx.estimate_tokens(LONG),
                                  ollama_ctx.fit_num_ctx(LONG, 300, where="t")))

digits = "4539 1488 0343 6467 " * 200
check("the estimate errs high on digit-heavy text",
      ollama_ctx.estimate_tokens(digits) >= len(digits) // 4,
      "chars/4=%d, estimate=%d" % (len(digits) // 4,
                                   ollama_ctx.estimate_tokens(digits)))

err, real = io.StringIO(), sys.stderr
sys.stderr = err
ollama_ctx.fit_num_ctx(HUGE, 300, where="a call over the cap")
sys.stderr = real
check("an overflow is said out loud, never silent",
      "WARNING" in err.getvalue() and "FRONT" in err.getvalue(),
      repr(err.getvalue()[:80]))

# ------------------------------------------------------------- the clients
print("\nevery client sends an explicit num_ctx")

import credibility as C

C.call_ollama(SHORT, max_tokens=300, where="t")
check("credibility, short call", "num_ctx" in last_options(), last_options())

C.call_ollama(LONG, max_tokens=300, where="t")
opts = last_options()
check("credibility, long call fits",
      opts.get("num_ctx", 0) >= ollama_ctx.estimate_tokens(LONG) + 300,
      opts)

C.call_ollama(SHORT, max_tokens=300, num_ctx=4096, where="t")
check("a window the caller names is still honoured",
      last_options().get("num_ctx") == 4096, last_options())

import ontology_rag as O

O.call_ollama(LONG)
check("ontology_rag", last_options().get("num_ctx", 0) > ollama_ctx.FLOOR,
      last_options())

import webrag_system as W

W.call_ollama(LONG, max_tokens=300)
check("webrag_system", last_options().get("num_ctx", 0) > ollama_ctx.FLOOR,
      last_options())

W.call_ollama("harvest_patterns calls it positionally", 5)
check("webrag_system, positional args still work",
      "num_ctx" in last_options(), last_options())

# --------------------------------------------- and the paths through them
print("\nthe baselines that reach those clients")

import combined_evaluate as CE

CE.ask_verdict("You are an analyst.\n" + LONG + "\n" + CE.VERDICT_FORMAT,
               CE.VerdictStats("llm_only"), 300)
check("ask_verdict (llm_only, singh, qwen_kb, hybrid)",
      last_options().get("num_ctx", 0) > ollama_ctx.FLOOR,
      "still on the 2048 default: %s" % last_options())

import mcq_ontology_rag as M

det = M.MCQOntologyDetector.__new__(M.MCQOntologyDetector)
det.max_tokens = 700
det._ask(LONG, where="mcq questions")
check("MCQOntologyDetector._ask (mcq)",
      last_options().get("num_ctx", 0) > ollama_ctx.FLOOR, last_options())

import llm_judge as J

check("llm_judge and the benchmark agree on the estimate",
      J.estimate_tokens(LONG) == ollama_ctx.estimate_tokens(LONG))
check("an unguided prompt is still the llm_only prompt",
      J.build_prompt("a call") ==
      "You are a scam detection analyst. Read this phone call transcript and "
      "decide whether the caller is attempting a scam.\n\nTranscript:\na "
      "call\n\n" + CE.VERDICT_FORMAT)

# ------------------------------------------------------------------ report
print("\nthe summary a run would print when something did not fit")
buf = io.StringIO()
ollama_ctx.report(buf)
print("".join("  " + ln + "\n" for ln in buf.getvalue().strip().splitlines()))

print("=" * 66)
if FAILURES:
    print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("all good - every Ollama client sends a window big enough for its prompt")
