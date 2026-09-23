#!/usr/bin/env python3
"""
web_ui.py - a browser front end for run_all.sh.

    python3 scripts/web_ui.py               # http://127.0.0.1:8000
    python3 scripts/web_ui.py --port 8080
    python3 scripts/web_ui.py --host 0.0.0.0    # reachable from other machines

It asks the same questions run_all.sh asks, as a form, and then runs
run_all.sh itself - the shell script stays the single source of truth for
what a baseline actually does, which model gets unloaded before BERT, and
how the results table is built.

Five pages:

  Benchmark   the form above, the output of a run, its results table and its
              prediction for every call.
  BERT        fine-tune a BERT on one of the datasets and keep the
              checkpoint, then put a transcript to it and get back the
              probability that the call is a scam. Long calls are scored in
              windows, because BERT reads 512 tokens at most.
              scripts/bert_classify.py does the work; this is its front end.
  Bag of words  TF-IDF into a logistic regression, fitted in about a second
              and readable back exactly - the control the other two are
              measured against. scripts/bow_classify.py does the work.
  Length only the floor: count the words, compare to one number, call it.
              Nothing in the call is read. scripts/length_classify.py does
              the work, and whatever a model beats it by is the whole of
              what that model is worth.
  LLM judge   one transcript to the local LLM, scam or not, with its reason.
              scripts/llm_judge.py does the work.

Over SSH, forward the port rather than binding to 0.0.0.0:

    ssh -L 8000:localhost:8000 user@your-gpu-host

Runs are detached from this server (their own session, output straight to a
file), so a run survives closing the browser, and it survives restarting or
killing this server too. Reopen the page and the run is still there.

Standard library only - nothing to install.
"""
import argparse
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUN_ALL = PROJECT_DIR / "run_all.sh"
RUNS_DIR = PROJECT_DIR / "results" / "logs" / "web"     # already gitignored
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Kept in step with run_all.sh. NEEDS_MODEL mirrors the case statement there:
# the two baselines that never call an LLM must not ask for a model.
BASELINES = [
    ("all",      "all",                   "every system below, in one run",       True),
    ("length",   "Length only",           "word count against one threshold, no LLM", False),
    ("bow",      "Bag of words",          "TF-IDF into logistic regression, no LLM", False),
    ("llm_only", "LLM-only",              "the model decides alone, no retrieval", True),
    ("singh",    "Singh",                 "policy-compliance baseline",           True),
    ("webrag",   "Web-RAG",               "KB-only retrieval",                    True),
    ("qwen_kb",  "Qwen-KB",               "learns a KB from a held-out split, k-fold", True),
    ("hybrid",   "Hybrid",                "Web-RAG + Qwen-KB over one shared KB",  True),
    ("ontology", "Ontology RAG",          "scam_ontology.json",                   True),
    ("mcq",      "BERT ontology",          "mcq_ontology.json, 2 calls per transcript", True),
    ("bert",     "BERT",                  "fine-tuned classifier, no LLM",        False),
]
MODELS = ["qwen2.5:14b", "llama3.1:8b"]

# The Web-RAG knowledge base. harvest_patterns.py writes the JSON, which is
# the source of truth, and build_index.py derives the vector index from it -
# the same two scripts section C of the README runs by hand. Each mode below
# is one combination of the flags those two already support.
#   key, label, note, harvest flags (None = do not harvest), rebuild?, exclusive?
KB_JSON = PROJECT_DIR / "knowledge" / "scam_patterns.json"
CHROMA_DB = PROJECT_DIR / "chroma_db" / "chroma.sqlite3"
KB_COLLECTION = "scam_patterns"
KB_MODES = [
    ("refresh", "Harvest the web, then rebuild the index",
     "Tavily for new articles, the local LLM extracts a pattern from each, "
     "then everything is re-embedded", [], True, True),
    ("nocache", "Harvest ignoring the cache, then rebuild",
     "the same, but every seed query is fetched fresh - spends Tavily credits",
     ["--no-cache"], True, True),
    ("index", "Rebuild the index only",
     "re-embed knowledge/scam_patterns.json as it stands - no web calls, no LLM",
     None, True, True),
    ("dry", "Dry run - show the plan",
     "the seed queries and what they would cost, stopping before any network call",
     ["--dry-run"], False, False),
    ("stats", "Stats only",
     "summarise what the knowledge base already holds", ["--stats"], False, False),
]
LIMIT_RE = re.compile(r"^(?:\d+|id:.+|idx:\d+)$")
EXIT_MARK = "__RUN_EXIT__"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_LOG_TAIL = 400_000      # bytes of a step log the page will show

# The shell run_all.sh is handed to. Overridable because "bash" on PATH is
# not always the POSIX one - on Windows it resolves to WSL, which cannot
# see the repo. On the Linux box this never needs setting.
BASH = os.environ.get("SCAM_BASH", "bash")

# How a run is cut loose from this server, so restarting or killing the
# server leaves the benchmark running. setsid() on Unix; on Windows the
# same idea needs the two creation flags, since start_new_session is a
# no-op there and the child would die with its parent.
if os.name == "nt":
    DETACHED = {"creationflags": (subprocess.DETACHED_PROCESS |
                                  subprocess.CREATE_NEW_PROCESS_GROUP)}
else:
    DETACHED = {"start_new_session": True}


# --------------------------------------------------------------- discovery
def datasets():
    """Every CSV in datasets/, with a row count that survives embedded newlines."""
    import csv
    out = []
    for p in sorted((PROJECT_DIR / "datasets").glob("*.csv")):
        try:
            csv.field_size_limit(sys.maxsize)
            with open(p, newline="", encoding="utf-8") as f:
                rows = sum(1 for _ in csv.reader(f)) - 1
        except Exception:
            rows = None
        out.append({"path": "datasets/" + p.name, "name": p.name, "rows": rows})
    return out


# dataset path -> (mtime, stats). Scanning everything_7013.csv takes a moment
# and the answer only changes when the file does.
_CTX_STATS = {}
CTX_STEPS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def dataset_context(path):
    """How big a context window this dataset's calls need.

    The Benchmark page fills its Context window box from this when a dataset
    is picked, so the window is right before the run rather than after a
    warning. Worth doing because missing the window is not a near miss:
    Ollama keeps about half of it and discards the rest from the front, so a
    verdict on an overlong call is a verdict on its goodbyes.
    """
    import csv
    import ollama_ctx

    if path not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    p = PROJECT_DIR / path
    mtime = p.stat().st_mtime
    hit = _CTX_STATS.get(path)
    if hit and hit[0] == mtime:
        return hit[1]

    csv.field_size_limit(sys.maxsize)
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        tcol = next((cols[c] for c in ("transcript", "text", "call",
                                       "conversation", "dialogue", "content",
                                       "body") if c in cols), None)
        if tcol is None:
            raise ValueError("no transcript column in that dataset")
        toks = sorted(ollama_ctx.estimate_tokens(r.get(tcol) or "")
                      for r in reader)
    if not toks:
        raise ValueError("that dataset has no rows")

    n = len(toks)
    # the prompt is the transcript plus the question and answer format, and
    # the reply needs room too
    overhead = 400
    p90 = toks[int(0.9 * n) - 1] + overhead
    biggest = toks[-1] + overhead
    fits = lambda need: next((s for s in CTX_STEPS
                              if s >= need * ollama_ctx.SAFETY), CTX_STEPS[-1])
    # Cover every call when that is affordable, and only fall back to the p90
    # when covering the longest would need a window no local model will give
    # (qwen2.5 tops out at 32768, and the KV cache for one is VRAM the weights
    # also want). Recommending the p90 by default would leave a tenth of the
    # dataset silently answered on its goodbyes, which is the whole failure.
    covers_all = fits(biggest)
    recommended = covers_all if covers_all <= 32768 else fits(p90)
    stats = {
        "dataset": path, "rows": n,
        "median": toks[n // 2] + overhead, "p90": p90, "max": biggest,
        "recommended": recommended,
        "covers_all": covers_all,
        "over_default": sum(1 for t in toks if t + overhead > 8192),
        # what is still too long even at the recommendation - the calls that
        # need a decision rather than a bigger number
        "over_recommended": sum(1 for t in toks
                                if t + overhead > recommended),
    }
    _CTX_STATS[path] = (mtime, stats)
    return stats


def idx_pool(default=40):
    """The --limit whose ordering an idx:<n> is counted against.

    run_all.sh carves that row out of a sample of this size, so indexes above
    it name nothing - and the number is defined there, in IDX_LIMIT_USED, not
    here. Read rather than duplicated so the two cannot drift apart.
    """
    try:
        m = re.search(r"^IDX_LIMIT_USED=(\d+)",
                      RUN_ALL.read_text(encoding="utf-8"), re.M)
    except OSError:
        return default
    return int(m.group(1)) if m else default


def ollama_state():
    """Which models Ollama has pulled, and which it is holding in VRAM."""
    state = {"up": False, "pulled": [], "loaded": []}
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as r:
            state["pulled"] = [m.get("name") for m in json.load(r).get("models", [])]
            state["up"] = True
    except Exception:
        return state
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=5) as r:
            state["loaded"] = [m.get("name") or m.get("model")
                               for m in json.load(r).get("models", [])]
    except Exception:
        pass
    return state


def tavily_key_present():
    """Is there a Tavily key for the harvest to use? .env is read here rather
    than loaded, so this server never changes its own environment."""
    if os.environ.get("TAVILY_API_KEY", "").strip():
        return True
    try:
        for line in (PROJECT_DIR / ".env").read_text(encoding="utf-8").splitlines():
            key, _, val = line.strip().partition("=")
            if key.strip() == "TAVILY_API_KEY" and val.strip().strip("\"'"):
                return True
    except Exception:
        pass
    return False


def index_vectors():
    """How many vectors the scam_patterns collection holds.

    Read with sqlite3, not chromadb: this server is standard library only and
    has to start whether or not the venv is active. None means the question
    could not be answered - no database file yet, or a Chroma schema this
    query no longer fits.
    """
    if not CHROMA_DB.exists():
        return None
    import sqlite3
    try:
        con = sqlite3.connect("file:%s?mode=ro" % CHROMA_DB.as_posix(), uri=True)
    except Exception:
        return None
    try:
        row = con.execute(
            "SELECT COUNT(e.id) FROM collections c "
            "LEFT JOIN segments s ON s.collection = c.id "
            "LEFT JOIN embeddings e ON e.segment_id = s.id "
            "WHERE c.name = ?", (KB_COLLECTION,)).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None
    finally:
        con.close()


def kb_state():
    """What the Web-RAG knowledge base holds: the JSON, and the index built
    from it. They disagree whenever a harvest has not been followed by a
    rebuild, which is the case the page needs to be able to point at."""
    state = {"patterns": None, "categories": 0, "last_refresh": None,
             "refresh_count": 0, "vectors": index_vectors(),
             "stale": False, "tavily": tavily_key_present()}
    try:
        kb = json.loads(KB_JSON.read_text(encoding="utf-8"))
    except Exception:
        return state
    patterns = kb.get("patterns", [])
    state["patterns"] = len(patterns)
    state["categories"] = len({p.get("category") for p in patterns})
    state["last_refresh"] = kb.get("last_refresh")
    state["refresh_count"] = kb.get("refresh_count", 0)
    state["stale"] = state["vectors"] is not None and state["vectors"] != len(patterns)
    return state


# -------------------------------------------------------------- run records
# A run is two files: <id>.json (what was asked for) and <id>.log (what it
# said). Nothing is held only in memory, so the server can be restarted
# underneath a running benchmark without losing track of it.
# Popen objects for runs this server started. Without one to poll(), a
# finished child stays a zombie - and os.kill(pid, 0) succeeds on a zombie,
# so a killed run would read as "running" forever. Runs inherited from an
# earlier server are not in here and need the plain pid check instead.
LIVE = {}


# A run id is generated here (a timestamp and the baseline keys), never typed,
# so anything outside this shape came from a crafted request. It matters for
# delete: run_path() is string concatenation, and "../../x" would resolve
# outside RUNS_DIR. Reads were already safe by accident - they only ever open
# a file that has to exist - but unlink is not something to leave to luck.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_+.-]{1,120}$")


def run_path(run_id, ext):
    return RUNS_DIR / ("%s.%s" % (run_id, ext))


def checked_run_id(run_id):
    """The id, or a ValueError. Shape first, then the resolved path."""
    run_id = (run_id or "").strip()
    if not RUN_ID_RE.match(run_id):
        raise ValueError("bad run id")
    target = run_path(run_id, "json").resolve()
    if target.parent != RUNS_DIR.resolve():
        raise ValueError("bad run id")
    return run_id


def load_run(run_id):
    try:
        with open(run_path(run_id, "json"), encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    # a file caught mid-write, or one edited by hand into something odd,
    # drops out of the listing rather than failing the whole endpoint
    if not isinstance(meta, dict) or "id" not in meta:
        return None
    try:
        meta["status"] = run_status(meta)
    except Exception:
        meta["status"] = "failed"
    return meta


def all_runs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in RUNS_DIR.glob("*.json"):
        meta = load_run(p.stem)
        if meta:
            out.append(meta)
    out.sort(key=lambda m: m.get("started", 0), reverse=True)
    return out


def run_status(meta):
    """running / done / failed / stopped, worked out from the log and the pid.

    The exit marker in the log is the authority: it is written by the wrapper
    after run_all.sh returns, so it is still correct if this server was not
    running at the time.
    """
    log = run_path(meta["id"], "log")
    tail = ""
    if log.exists():
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 4096))
            tail = f.read().decode("utf-8", "replace")
    m = re.search(re.escape(EXIT_MARK) + r"\s+(\d+)", tail)
    if m:
        code = int(m.group(1))
        if code == 0:
            return "done"
        return "stopped" if code in (130, 143) else "failed"
    proc = LIVE.get(meta["id"])
    if proc is not None:
        code = proc.poll()          # reaps it, so no zombie to mistake for alive
        if code is None:
            return "running"
        # negative is the signal that killed it: -15 is the Stop button
        return "stopped" if code in (130, 143, -2, -15) else "failed"
    if pid_alive(meta.get("pid")):
        return "running"
    return "failed"     # the process is gone but never wrote a marker


def pid_alive(pid):
    """Is that pid still running? Only needed for runs this server did not
    start - after a restart, LIVE is empty and the pid is all there is.
    """
    if not pid:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a probe on Windows: it goes through
        # TerminateProcess and raises on processes it cannot open, so live
        # ones read as dead. Ask the OS for the exit code instead.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def start_run(form):
    """Validate the form, then hand the work to run_all.sh."""
    ds = form.get("dataset", "")
    limit = str(form.get("limit", "0")).strip()
    model = form.get("model", "")
    # the form sends a list; a bare string is still accepted so the older
    # single-baseline shape of this call keeps working
    picked = form.get("baselines") or form.get("baseline") or []
    if isinstance(picked, str):
        picked = [b for b in picked.split(",") if b.strip()]

    valid_ds = {d["path"] for d in datasets()}
    if ds not in valid_ds:
        raise ValueError("unknown dataset")
    want_stripped = bool(form.get("stripped"))
    stripped = ""
    known = {b[0]: b for b in BASELINES}
    bad = [b for b in picked if b not in known]
    if bad:
        raise ValueError("unknown baseline: " + ", ".join(bad))
    if not picked:
        raise ValueError("tick at least one baseline")
    # keep menu order, drop duplicates: run_all.sh reorders anyway, but the
    # run label should read the same way the checkboxes do
    baselines = [b[0] for b in BASELINES if b[0] in set(picked)]
    if not LIMIT_RE.match(limit):
        raise ValueError("limit must be a whole number, id:<value>, or idx:<n>")
    if want_stripped:
        # No twin file to find and none to get wrong: the stripped copy is
        # built from this same dataset, so every dataset can be tested.
        stripped = ds
        if limit.startswith(("id:", "idx:")):
            raise ValueError("the content-deletion test scores every held-out "
                             "call twice; it cannot run on a single transcript")
    num_ctx = numeric(form, {"num_ctx": ("num_ctx", int, 2048, 131072, None)},
                      "num_ctx")
    if any(known[b][3] for b in baselines):
        if model not in MODELS:
            raise ValueError("pick a model")
    else:
        model = ""

    running = [r for r in all_runs() if r["status"] == "running"]
    if running:
        raise ValueError("a run is already going (%s). Stop it first - the GPU "
                         "cannot hold two." % running[0]["id"])

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + (
        "all" if len(baselines) == len(BASELINES) - 1
        else "+".join(baselines))
    log = run_path(run_id, "log")

    flags = ["--dataset", ds, "--baseline", ",".join(baselines),
             "--limit", limit]
    if model:
        flags += ["--model", model]
    if stripped:
        # run_all.sh resolves the twin the same way; the path above is only
        # so this server can refuse a bad pair before starting anything
        flags.append("--stripped")

    # $1 is the script and "${@:2}" the flags, so nothing here is re-parsed as
    # shell syntax. The marker records the exit status in the log itself.
    wrapper = 'bash "$1" "${@:2}"; printf "\\n%s %%s\\n" "$?"' % EXIT_MARK
    argv = [BASH, "-c", wrapper, "run_all", str(RUN_ALL)] + flags

    env = dict(os.environ)
    env["RUN_ALL_DETACHED"] = "1"       # never ask the tmux question
    env["TERM"] = "dumb"
    # The context window the LLM systems size their prompts against. It is an
    # environment variable rather than a run_all.sh flag because every script
    # under it reads it the same way; without this the page could only ever
    # run at whatever the server itself was started with, which on a dataset
    # of long calls is the difference between a verdict and a guess.
    if num_ctx:
        env["SCAM_NUM_CTX"] = str(num_ctx)

    with open(log, "wb") as out:
        out.write(("$ ./run_all.sh " + " ".join(shlex.quote(f) for f in flags)
                   + "\n\n").encode())
        out.flush()
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT_DIR), stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, **DETACHED)

    LIVE[run_id] = proc

    meta = {"id": run_id, "pid": proc.pid, "dataset": ds,
            "baseline": ",".join(baselines), "baselines": baselines,
            "limit": limit, "model": model, "started": time.time(),
            "stripped": stripped}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def start_kb_run(form):
    """Refresh the Web-RAG knowledge base: harvest -> JSON -> vector index.

    Same shape as start_run - a detached child writing into the same runs
    directory - so a knowledge-base update shows up under Recent runs, streams
    into the same Output pane, and can be stopped with the same button.
    """
    mode = str(form.get("mode", "refresh"))
    known = {m[0]: m for m in KB_MODES}
    if mode not in known:
        raise ValueError("unknown knowledge-base mode: " + mode)
    _, label, _, harvest, rebuild, exclusive = known[mode]

    # --dry-run and --stats touch nothing, so they are allowed at any time.
    # The rest either drive the LLM or delete and recreate the collection a
    # webrag run may be reading from.
    if exclusive:
        running = [r for r in all_runs() if r["status"] == "running"]
        if running:
            raise ValueError("a run is already going (%s). Stop it first - the "
                             "index cannot be rebuilt underneath one."
                             % running[0]["id"])
        if harvest is not None and not tavily_key_present():
            raise ValueError("no TAVILY_API_KEY in .env or the environment, so "
                             "there is nothing to harvest with. \"Rebuild the "
                             "index only\" needs no key.")

    steps = []
    if harvest is not None:
        step = ("plan" if "--dry-run" in harvest else
                "stats" if "--stats" in harvest else "harvest")
        steps.append((step, ["scripts/harvest_patterns.py"] + harvest))
    if rebuild:
        steps.append(("index", ["scripts/build_index.py"]))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_kb_" + mode
    log = run_path(run_id, "log")

    # run_all.sh is not involved here, so its venv preamble is repeated: the
    # server itself runs on system python and must not assume the venv one.
    body = [
        "kb_update() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
    ]
    for name, cmd in steps:
        quoted = " ".join(shlex.quote(c) for c in cmd)
        # ==> / ok / fail are the same three shapes run_all.sh prints, which is
        # what the Output pane colours on
        body += [
            '  printf "\\n==> %s\\n" ' + shlex.quote(name),
            '  "$PY" -u ' + quoted + ' || { printf "  fail %s\\n" '
            + shlex.quote(name) + '; return 1; }',
            '  printf "  ok %s\\n" ' + shlex.quote(name),
        ]
    body += ["}", "kb_update",
             'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK]
    argv = [BASH, "-c", "\n".join(body)]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        for _, cmd in steps:
            out.write(("$ python " + " ".join(shlex.quote(c) for c in cmd)
                       + "\n").encode())
        out.write(b"\n")
        out.flush()
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT_DIR), stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, **DETACHED)

    LIVE[run_id] = proc

    meta = {"id": run_id, "pid": proc.pid, "kind": "kb", "mode": mode,
            "label": "knowledge base · " + label[0].lower() + label[1:],
            "dataset": "knowledge/scam_patterns.json",
            "baseline": "kb:" + mode, "limit": "-", "model": "",
            "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def stop_run(run_id):
    meta = load_run(run_id)
    if not meta:
        raise ValueError("no such run")
    pid = meta.get("pid")
    if not pid_alive(pid):
        return {"stopped": False, "note": "not running"}
    # start_new_session put the run in its own process group, so this reaches
    # run_all.sh and every python step it spawned, not just the wrapper.
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        os.kill(pid, signal.SIGTERM)
    return {"stopped": True}


def delete_run(run_id):
    """Drop one run from the history.

    Removes the three things that run owns outright: its metadata, the output
    this server captured, and the results/logs/run_<stamp>/ directory the run
    itself wrote its per-baseline logs into.

    It deliberately does NOT touch results/*.csv. Those are named after the
    dataset and the limit, not after the run - two runs over the same dataset
    write the same combined_results_646.csv - so deleting one run's history
    would silently take another run's numbers with it.
    """
    run_id = checked_run_id(run_id)
    meta = load_run(run_id)
    log = run_path(run_id, "log")
    if meta is None and not log.exists():
        raise ValueError("no such run")
    if meta and meta.get("status") == "running":
        raise ValueError("that run is still going - stop it first")

    # Resolve the step-log directory before the log that names it is deleted.
    logdir = logdir_of(run_id)

    removed = []
    for f in (run_path(run_id, "json"), log):
        try:
            f.unlink()
            removed.append(f.name)
        except FileNotFoundError:
            pass

    # Belt and braces on a path that came out of a log file: it has to sit
    # directly under results/logs/ and be one of run_all.sh's own run_<stamp>
    # directories before anything is removed recursively.
    if logdir is not None:
        d = logdir.resolve()
        parent = (PROJECT_DIR / "results" / "logs").resolve()
        if d.parent == parent and re.fullmatch(r"run_\d{8}_\d{6}", d.name):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)

    LIVE.pop(run_id, None)
    return {"deleted": True, "id": run_id, "removed": removed}


def delete_finished():
    """Every run that is not currently going. Returns what went."""
    gone, kept = [], 0
    for meta in all_runs():
        if meta.get("status") == "running":
            kept += 1
            continue
        try:
            delete_run(meta["id"])
            gone.append(meta["id"])
        except ValueError:
            kept += 1
    return {"deleted": len(gone), "ids": gone, "still_running": kept}


def read_log(run_id, offset):
    log = run_path(run_id, "log")
    if not log.exists():
        return {"offset": 0, "text": ""}
    size = log.stat().st_size
    offset = max(0, min(offset, size))
    with open(log, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    text = ANSI.sub("", chunk.decode("utf-8", "replace"))
    text = re.sub(re.escape(EXIT_MARK) + r"\s+\d+\s*", "", text)
    return {"offset": size, "text": text}


def logdir_of(run_id):
    """The results/logs/run_<stamp> directory this run wrote into.

    run_all.sh announces it on its "ok logs" line, so it is read back out
    of the captured output rather than guessed from timestamps.
    """
    log = run_path(run_id, "log")
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"(results/logs/run_\d{8}_\d{6})", text)
    if not hits:
        return None
    d = PROJECT_DIR / hits[-1]
    return d if d.is_dir() else None


def results_of(run_id):
    """The finished numbers, via collect_results.py --json."""
    d = logdir_of(run_id)
    if d is None:
        return None
    try:
        out = subprocess.run(
            [sys.executable, "scripts/collect_results.py",
             str(d.relative_to(PROJECT_DIR)), "--json"],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)
    except Exception:
        return None


# ---------------------------------------------------------------- the ledger
# Every successful benchmark run leaves its numbers here, one line per system,
# so a table can be built up a baseline at a time over days instead of by one
# run of everything that takes all afternoon. It lives outside results/logs/
# on purpose: deleting a run from Recent runs removes its logs, and the whole
# point of the ledger is that the number it produced is not lost with them.
LEDGER = PROJECT_DIR / "results" / "ledger.jsonl"
LEDGER_LOCK = threading.Lock()

# collect_results.py names two systems differently from the baseline menu
SYSTEM_TO_BASELINE = {"ontology_rag": "ontology", "mcq_ontology": "mcq"}


def ledger_read():
    """Every ledger line, in the order written. A line that will not parse is
    skipped rather than taking the whole tab down with it."""
    out = []
    try:
        with open(LEDGER, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return out


def ledger_sync():
    """Record any finished run the ledger has not seen yet.

    Runs are looked at once: a run that produced nothing still leaves a marker
    line, so it is not re-parsed on every visit. Called when the Results tab
    asks, and every half-minute from the reaper thread, so a run is recorded
    soon after it finishes even if nobody opens the tab before deleting it.
    """
    with LEDGER_LOCK:
        seen = {e.get("run_id") for e in ledger_read()}
        needs_model = {b[0]: b[3] for b in BASELINES}
        new = []
        for meta in all_runs():
            rid = meta.get("id")
            if (not rid or rid in seen or meta.get("kind")
                    or meta.get("status") != "done"):
                continue
            res = results_of(rid) or {}
            systems = [x for x in res.get("systems", []) if x.get("ran")]
            finished = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(
                    run_path(rid, "log").stat().st_mtime
                    if run_path(rid, "log").exists() else time.time()))
            if not systems:
                new.append({"run_id": rid, "empty": True})
                continue
            for x in systems:
                key = SYSTEM_TO_BASELINE.get(x["system"], x["system"])
                entry = {
                    "run_id": rid,
                    "finished": finished,
                    "dataset": meta.get("dataset", ""),
                    "system": key,
                    "model": meta.get("model") if needs_model.get(key) else "",
                    "limit": str(meta.get("limit", "")),
                    "calls": x["tp"] + x["fp"] + x["fn"] + x["tn"],
                    "stripped_test": bool(meta.get("stripped")),
                }
                for k in ("acc", "p", "r", "f1", "tp", "fp", "fn", "tn"):
                    entry[k] = x[k]
                if x.get("stripped"):
                    entry["stripped_acc"] = x["stripped"]["acc"]
                    entry["trusted"] = x.get("trusted")
                new.append(entry)
        if new:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a", encoding="utf-8") as f:
                for e in new:
                    f.write(json.dumps(e) + "\n")
        return len([e for e in new if not e.get("empty")])


def ledger_view(dataset):
    """What the Results tab shows for one dataset: every recorded result for
    it, newest first, plus the datasets that have anything recorded."""
    rows = [e for e in ledger_read()
            if not e.get("empty") and not e.get("hidden")]
    counts = {}
    for e in rows:
        counts[e["dataset"]] = counts.get(e["dataset"], 0) + 1
    picked = [e for e in rows if e["dataset"] == dataset] if dataset else []
    picked.sort(key=lambda e: (e.get("finished", ""), e.get("run_id", "")),
                reverse=True)
    order = [b[0] for b in BASELINES if b[0] != "all"]
    return {"dataset": dataset, "entries": picked,
            "datasets_recorded": counts, "system_order": order,
            "labels": {b[0]: b[1] for b in BASELINES}}


def ledger_hide(run_id, system):
    """Take one result off the Results tab. It is marked hidden rather than
    deleted, so the next sync does not see the run as new and put it back."""
    run_id = checked_run_id(run_id)
    with LEDGER_LOCK:
        entries = ledger_read()
        hit = 0
        for e in entries:
            if e.get("run_id") == run_id and e.get("system") == system:
                e["hidden"] = True
                hit += 1
        if not hit:
            raise ValueError("no such result")
        tmp = LEDGER.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        tmp.replace(LEDGER)
    return {"hidden": hit}


def artifacts_of(run_id):
    """Everything a run left behind: per-step logs, and per-call CSVs.

    The CSV names are not guessable - they carry a tag built from the
    dataset and the limit - but every script prints the path it wrote, so
    the paths are harvested from the logs and then checked against disk.
    Only names collected this way are ever served, which is also what
    keeps a crafted ?name= from reaching outside results/.
    """
    out = {"steps": [], "csvs": []}
    d = logdir_of(run_id)
    text = ""
    weblog = run_path(run_id, "log")
    if weblog.exists():
        text = weblog.read_text(encoding="utf-8", errors="replace")
    if d is not None:
        for f in sorted(d.glob("*.log")):
            out["steps"].append({"name": f.name, "bytes": f.stat().st_size})
            text += "\n" + f.read_text(encoding="utf-8", errors="replace")
    seen = set()
    for name in re.findall(r"results[/\\]([A-Za-z0-9_.\-]+\.csv)", text):
        if name in seen:
            continue
        seen.add(name)
        f = PROJECT_DIR / "results" / name
        if f.exists():
            out["csvs"].append({"name": name, "bytes": f.stat().st_size})
    out["csvs"].sort(key=lambda c: c["name"])
    return out


def step_log(run_id, name):
    """One per-step log, tail-capped so a huge debug run cannot flood the page."""
    art = artifacts_of(run_id)
    if name not in {s["name"] for s in art["steps"]}:
        raise ValueError("no such step log")
    f = logdir_of(run_id) / name
    size = f.stat().st_size
    with open(f, "rb") as fh:
        if size > STEP_LOG_TAIL:
            fh.seek(size - STEP_LOG_TAIL)
        chunk = fh.read()
    text = ANSI.sub("", chunk.decode("utf-8", "replace"))
    return {"name": name, "bytes": size,
            "truncated": size > STEP_LOG_TAIL, "text": text}


def csv_page(run_id, name, offset, limit):
    """One page of a per-call CSV: the actual prediction for each transcript."""
    import csv
    art = artifacts_of(run_id)
    if name not in {c["name"] for c in art["csvs"]}:
        raise ValueError("no such results file")
    csv.field_size_limit(sys.maxsize)
    rows = []
    columns = []
    total = 0
    with open(PROJECT_DIR / "results" / name, newline="",
              encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        for n, row in enumerate(r):
            if n == 0:
                columns = row
                continue
            total += 1
            if offset < total <= offset + limit:
                rows.append(row)
    return {"name": name, "columns": columns, "rows": rows,
            "offset": offset, "total": total}


# ---------------------------------------------------------------------- BERT
# The second page. Fine-tune a BERT on one of the datasets, keep the
# checkpoint, then put a transcript to it and get back the one thing training
# actually fits: the probability that this call is a scam.
#
# bert_classify is imported rather than shelled out to for the cheap questions
# - listing checkpoints. Its module level is standard library only (torch and
# transformers are imported inside the functions that need them), so this
# server still starts on a machine with neither installed.
import bert_classify

BERT_SCRIPT = PROJECT_DIR / "scripts" / "bert_classify.py"
BOW_SCRIPT = PROJECT_DIR / "scripts" / "bow_classify.py"
LENGTH_SCRIPT = PROJECT_DIR / "scripts" / "length_classify.py"
MODELS_DIR = PROJECT_DIR / "models"
BERT_NAME_RE = bert_classify.NAME_RE

# What training can start from. Any model on the Hub works from the command
# line; the menu offers the four worth comparing that fit on one GPU.
BASE_MODELS = [
    ("bert-base-uncased", "the thesis baseline, 110M parameters"),
    ("distilbert-base-uncased", "40% smaller, about twice as fast, ~1 point behind"),
    ("roberta-base", "better pre-training, same size as BERT"),
    ("albert-base-v2", "12M parameters, but slower per epoch than DistilBERT"),
]

# Bounds on the training form. Each is (flag, cast, low, high, default) and the
# defaults are bert_classify.py's own, repeated here only so the form can
# show them.
TRAIN_FIELDS = {
    "epochs":     ("--epochs", int, 1, 20, 4),
    "batch_size": ("--batch-size", int, 1, 64, 8),
    "max_length": ("--max-length", int, 64, 512, 256),
    "lr":         ("--lr", float, 1e-6, 1e-3, 2e-5),
    "seed":       ("--seed", int, 0, 2 ** 31 - 1, 42),
    "limit":      ("--limit", int, 0, 100000, 0),
    "holdout":    ("--holdout", float, 0.0, 0.5, 0.2),
}
# The same idea for the answering side. These change how options are matched,
# so changing one restarts the answerer.
# The same idea for the scoring side. BERT reads 512 tokens at most and these
# checkpoints are trained at 256 - about 180 words - so a long call is cut
# into windows of that size and every one is scored. Changing any of these
# restarts the worker.
ANSWER_FIELDS = {
    "window":     ("--window", int, 20, 400, 180),
    "stride":     ("--stride", int, 10, 400, 90),
    "max_length": ("--max-length", int, 64, 512, 256),
}

# How long an answerer sits in memory with nothing asked of it before it is
# let go. Loading is the slow part, so it is worth holding; half a gigabyte of
# RAM for a page nobody is looking at any more is not.
WORKER_IDLE = 600


def venv_python():
    """The interpreter that has torch.

    run_all.sh activates venv/ if it is there; this server is deliberately
    started with the system python, so it has to find that interpreter itself
    rather than assume its own is the right one.
    """
    for rel in ("venv/bin/python", "venv/Scripts/python.exe",
                ".venv/bin/python", ".venv/Scripts/python.exe"):
        p = PROJECT_DIR / rel
        if p.exists():
            return str(p)
    return sys.executable


def numeric(form, fields, key):
    """One number off the form, cast and range-checked, or None if it was
    left blank. The server checks every one of these because the page is not
    the only thing that can POST to it."""
    raw = form.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    flag, cast, lo, hi, _ = fields[key]
    try:
        val = cast(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % key.replace("_", " "))
    if not lo <= val <= hi:
        raise ValueError("%s must be between %s and %s"
                         % (key.replace("_", " "), lo, hi))
    return val


class Answerer:
    """One model-serving subprocess, kept alive between questions.

    Loading a checkpoint takes seconds and answering it takes a fraction of
    one, so the model stays in memory between clicks rather than being loaded
    per question. It answers on the CPU unless the page asks otherwise: a
    benchmark run wants the whole GPU, and this page is meant to stay usable
    while one is going.

    The script is a parameter because the BERT page and the bag-of-words page
    speak the same one-JSON-line-per-request protocol to different programs.
    """

    def __init__(self, name, opts, script=None, log="worker.log"):
        self.name, self.opts = name, opts
        self.script = str(script or BERT_SCRIPT)
        self.log = log
        self.lock = threading.Lock()
        self.last = time.time()
        self.lines = queue.Queue()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        # transformers writes progress bars and load reports to stderr; they
        # go to a file so they neither fill the pipe nor reach the replies
        self.errlog = open(RUNS_DIR / log, "ab", buffering=0)
        argv = ([venv_python(), "-u", self.script, "serve", "--name", name]
                + opts)
        self.proc = subprocess.Popen(
            argv, cwd=str(PROJECT_DIR), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.errlog, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        # a first load reads several hundred megabytes off disk, and on a cold
        # page cache that is not quick
        ready = self._take(300)
        if not ready.get("ok"):
            self.close()
            raise ValueError(ready.get("error") or "the answerer would not start")
        self.device = ready.get("device", "cpu")

    def _pump(self):
        """Replies are read by a thread of their own, so a worker that dies
        mid-answer shows up as an empty line rather than as a wait with no
        end to it."""
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put("")

    def _take(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise ValueError("the answerer has not replied in %ds - see "
                             "results/logs/web/%s" % (timeout, self.log))
        if not line.strip():
            raise ValueError("the answerer stopped - see "
                             "results/logs/web/" + self.log)
        try:
            return json.loads(line)
        except ValueError:
            raise ValueError("the answerer said something that is not JSON: "
                             + line[:200])

    def ask(self, payload, timeout=300):
        with self.lock:
            # marked busy before the question as well as after the answer, so
            # the idle reaper cannot close a worker that is mid-answer
            self.last = time.time()
            if self.proc.poll() is not None:
                raise ValueError("the answerer has stopped - ask again to "
                                 "start it back up")
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise ValueError("the answerer stopped before it could be asked")
            out = self._take(timeout)
            self.last = time.time()
            return out

    def close(self):
        for shut in (lambda: (self.proc.stdin.write('{"quit":true}\n'),
                              self.proc.stdin.flush()),
                     lambda: self.proc.wait(timeout=5),
                     self.proc.kill,
                     self.errlog.close):
            try:
                shut()
            except Exception:
                pass


# One answerer at a time. Two would be two copies of BERT in memory for no
# gain: the page only ever asks about the model that is selected.
class Slot:
    """One held-open worker, and the lock that guards swapping it.

    There is one of these per page rather than one for the whole server: a
    bag-of-words model is a few hundred kilobytes, and making it evict a
    half-gigabyte BERT checkpoint - or the other way round - every time
    someone switches tab would be a reload nobody asked for.
    """

    def __init__(self, script, log):
        self.script, self.log = script, log
        self.worker = None
        self.lock = threading.Lock()

    def for_model(self, name, opts):
        """The live worker for that model, started - or restarted - if the
        model or the settings it was loaded with have changed."""
        with self.lock:
            w = self.worker
            if w is not None and (w.name != name or w.opts != opts
                                  or w.proc.poll() is not None):
                w.close()
                self.worker = None
            if self.worker is None:
                self.worker = Answerer(name, opts, self.script, self.log)
            return self.worker

    def unload(self):
        with self.lock:
            if self.worker is None:
                return {"unloaded": False, "note": "nothing was loaded"}
            name = self.worker.name
            self.worker.close()
            self.worker = None
            return {"unloaded": True, "model": name}

    def state(self):
        w = self.worker
        if w is None or w.proc.poll() is not None:
            return {"loaded": False}
        return {"loaded": True, "model": w.name, "device": w.device,
                "idle_s": int(time.time() - w.last), "idle_limit": WORKER_IDLE}

    def reap(self):
        if self.worker is not None and time.time() - self.worker.last > WORKER_IDLE:
            self.unload()


BERT_SLOT = Slot(BERT_SCRIPT, "bert_worker.log")
BOW_SLOT = Slot(BOW_SCRIPT, "bow_worker.log")
LENGTH_SLOT = Slot(LENGTH_SCRIPT, "length_worker.log")
SLOTS = (BERT_SLOT, BOW_SLOT, LENGTH_SLOT)


def answerer_for(name, opts):
    return BERT_SLOT.for_model(name, opts)


def unload_answerer():
    return BERT_SLOT.unload()


def worker_state():
    return BERT_SLOT.state()


def worker_reaper():
    while True:
        time.sleep(30)
        for slot in SLOTS:
            slot.reap()
        try:
            ledger_sync()
        except Exception as e:           # never let the ledger kill the reaper
            sys.stderr.write("ledger sync failed: %s\n" % e)


def bert_config():
    """Everything the BERT page needs to draw itself once."""
    return {
        "datasets": datasets(),
        "bases": [{"id": k, "note": n} for k, n in BASE_MODELS],
        "models": bert_classify.list_models(),
        "defaults": {k: v[4] for k, v in
                     list(TRAIN_FIELDS.items()) + list(ANSWER_FIELDS.items())},
        "worker": worker_state(),
    }


def dataset_row(path, idx):
    """One transcript out of a dataset, to drop into the Ask box.

    Read with the csv module rather than pandas: this server is standard
    library only, and has to work whether or not the venv is there.
    """
    if path not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    import csv
    csv.field_size_limit(sys.maxsize)
    with open(PROJECT_DIR / path, newline="", encoding="utf-8-sig",
              errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = [c.lower() for c in next(reader)]
        except StopIteration:
            raise ValueError("that dataset is empty")
        rows = list(reader)

    def col(names):
        for n in names:
            if n in header:
                return header.index(n)
        return -1

    # the same column names bert_baseline.py looks for, so the page shows the
    # text that training would have used
    ti = col(["transcript", "text", "call", "conversation", "dialogue",
              "content", "body"])
    li = col(["label", "is_scam", "scam", "target", "class", "y", "ground_truth"])
    ii = col(["id", "call_id", "conv_id"])
    if ti < 0:
        raise ValueError("no transcript column in that dataset")
    if not rows:
        raise ValueError("that dataset has no rows")
    idx = max(0, min(int(idx), len(rows) - 1))
    row = rows[idx]
    cell = lambda i: row[i] if 0 <= i < len(row) else ""
    return {"idx": idx, "total": len(rows), "text": cell(ti),
            "label": cell(li), "row_id": cell(ii),
            "dataset": path}


def bert_verdict(form):
    """Put one transcript to one checkpoint."""
    name = str(form.get("model", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("pick a trained model first")
    if not (MODELS_DIR / name / "config.json").exists():
        raise ValueError("models/%s is not a trained checkpoint" % name)
    text = (form.get("transcript") or "").strip()
    if not text:
        raise ValueError("paste a transcript, or load one from a dataset")
    if len(text) > 400_000:
        raise ValueError("that is longer than any call in the datasets - "
                         "paste one call, not a whole file")
    threshold = numeric(form, {"threshold": ("threshold", float, 0.0, 1.0,
                                             None)}, "threshold")

    opts = []
    for key in ANSWER_FIELDS:
        val = numeric(form, ANSWER_FIELDS, key)
        if val is not None:
            opts += [ANSWER_FIELDS[key][0], str(val)]
    aggregate = str(form.get("aggregate") or "max")
    if aggregate not in ("max", "mean"):
        raise ValueError("aggregate must be max or mean")
    # The GPU by default. While a benchmark run is going it is using the GPU
    # itself, so a one-call question drops to the CPU for that call rather
    # than being refused - the answer card says which device answered.
    busy = any(r["status"] == "running" for r in all_runs())
    opts.append("--gpu" if form.get("gpu") and not busy else "--cpu")

    # aggregate, threshold and strip_tags ride with the request rather than
    # the worker's argv: they change what is done with the window scores, not
    # how the model reads, and a toggle should not cost a model reload.
    out = answerer_for(name, opts).ask(
        {"transcript": text, "threshold": threshold, "aggregate": aggregate,
         "strip_tags": bool(form.get("strip_tags"))})
    if not out.get("ok"):
        raise ValueError(out.get("error") or "the model could not score that")
    return out


# -------------------------------------------------------------- Bag of words
# The third page, and the control the other two are measured against. Same
# shape as the BERT page on purpose: train a model on a dataset, keep it, put
# a transcript to it, get a probability. TF-IDF over unigrams and bigrams into
# a logistic regression - the same vectoriser and classifier the `bow`
# baseline cross-validates on the Benchmark page.
#
# It trains in about a second and it can be read back exactly, which is the
# reason it earns a page rather than a row in a table: if it scores near a
# fine-tuned BERT, the dataset is separable on vocabulary and neither number
# is about understanding scams.
import bow_classify

BOW_TRAIN_FIELDS = {
    "ngram_max": ("--ngram-max", int, 1, 3, 2),
    "min_df":    ("--min-df", int, 1, 50, 2),
    "holdout":   ("--holdout", float, 0.0, 0.5, 0.2),
    "seed":      ("--seed", int, 0, 2 ** 31 - 1, 42),
    "limit":     ("--limit", int, 0, 100000, 0),
}


def bow_config():
    """Everything the Bag of words page needs to draw itself once."""
    return {
        "datasets": datasets(),
        "models": bow_classify.list_models(),
        "defaults": {k: v[4] for k, v in BOW_TRAIN_FIELDS.items()},
        "worker": BOW_SLOT.state(),
    }


def bow_verdict(form):
    """Put one transcript to one bag-of-words model."""
    name = str(form.get("model", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("pick a trained model first")
    if not (MODELS_DIR / name / bow_classify.MODEL_FILE).exists():
        raise ValueError("models/%s is not a bag-of-words model" % name)
    text = (form.get("transcript") or "").strip()
    if not text:
        raise ValueError("paste a transcript, or load one from a dataset")
    if len(text) > 400_000:
        raise ValueError("that is longer than any call in the datasets - "
                         "paste one call, not a whole file")
    threshold = numeric(form, {"threshold": ("threshold", float, 0.0, 1.0,
                                             None)}, "threshold")
    top = numeric(form, {"top": ("top", int, 3, 50, None)}, "top")

    # Nothing here changes how the model reads, so it all rides with the
    # request and no toggle costs a reload.
    out = BOW_SLOT.for_model(name, []).ask(
        {"transcript": text, "threshold": threshold, "top": top,
         "strip_tags": bool(form.get("strip_tags"))})
    if not out.get("ok"):
        raise ValueError(out.get("error") or "the model could not score that")
    return out


def remove_bow_model(name):
    name = (name or "").strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("bad model name")
    d = MODELS_DIR / name
    if not (d / bow_classify.MODEL_FILE).exists():
        raise ValueError("no such bag-of-words model: %s" % name)
    for slot in SLOTS:
        if slot.worker is not None and slot.worker.name == name:
            slot.unload()
    shutil.rmtree(d)
    return {"deleted": True, "id": name}


def start_bow_train_run(form):
    """Fit a bag-of-words model. Detached and logged like every other run.

    It finishes in about a second, so unlike BERT training it does not take
    the run lock - there is nothing for it to fight a benchmark over. No GPU,
    no VRAM, no epochs.
    """
    name = str(form.get("name", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("a name is letters, digits, dot, dash or underscore")
    ds = form.get("dataset", "")
    if ds not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    overwrite = bool(form.get("overwrite"))
    if (MODELS_DIR / name).exists() and not overwrite:
        raise ValueError("models/%s already exists - pick another name, or "
                         "tick replace" % name)

    flags = ["train", "--csv", ds, "--name", name]
    for key, (flag, _c, _lo, _hi, _d) in BOW_TRAIN_FIELDS.items():
        val = numeric(form, BOW_TRAIN_FIELDS, key)
        if val is None or (key == "limit" and not val):
            continue
        flags += [flag, str(val)]
    if form.get("strip_tags"):
        flags.append("--strip-tags")
    if overwrite:
        flags.append("--overwrite")
        for slot in SLOTS:
            if slot.worker is not None and slot.worker.name == name:
                slot.unload()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_bow_" + name
    log = run_path(run_id, "log")

    # The same venv preamble start_train_run uses, for the same reason:
    # run_all.sh is not involved, and the server's own python is not the one
    # with sklearn on it.
    quoted = " ".join(shlex.quote(f) for f in flags)
    body = [
        "fit() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
        '  printf "\\n==> fit\\n"',
        '  "$PY" -u scripts/bow_classify.py ' + quoted
        + ' || { printf "  fail fit\\n"; return 1; }',
        '  printf "  ok fit\\n"',
        "}", "fit",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        out.write(("$ python scripts/bow_classify.py " + quoted
                   + "\n\n").encode())
        out.flush()
        proc = subprocess.Popen(
            [BASH, "-c", "\n".join(body)], cwd=str(PROJECT_DIR), stdout=out,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            **DETACHED)

    LIVE[run_id] = proc
    meta = {"id": run_id, "pid": proc.pid, "kind": "bow_train",
            "model_name": name, "label": "fit bag of words \u00b7 " + name,
            "dataset": ds, "baseline": "bow:" + name, "limit": "-",
            "model": "tfidf+logreg", "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


# --------------------------------------------------------------- Length only
# The fourth page, and the floor. Count the words, compare the count to one
# number, call it. Nothing in the call is read - not a word of it - so
# whatever BERT or the bag of words beats this by is the whole of what those
# models are worth on that dataset.
#
# It is the `length` baseline on the Benchmark page, given a page of its own
# because the number it produces is not the interesting part: the interesting
# parts are which direction the rule has to point (on two of the datasets
# here the scam calls are the SHORTER ones, the opposite of what
# combined_evaluate's threshold of 45 assumes) and how flat the sweep curve
# is around the chosen threshold.
import length_classify

LENGTH_TRAIN_FIELDS = {
    "threshold": ("--threshold", int, 1, 200000, None),
    "holdout":   ("--holdout", float, 0.0, 0.5, 0.2),
    "seed":      ("--seed", int, 0, 2 ** 31 - 1, 42),
    "limit":     ("--limit", int, 0, 100000, 0),
}


def length_config():
    """Everything the Length only page needs to draw itself once."""
    return {
        "datasets": datasets(),
        "models": length_classify.list_models(),
        "defaults": {k: v[4] for k, v in LENGTH_TRAIN_FIELDS.items()},
        "benchmark_threshold": length_classify.BENCHMARK_THRESHOLD,
        "worker": LENGTH_SLOT.state(),
    }


def length_verdict(form):
    """Put one transcript to one length-only model."""
    name = str(form.get("model", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("pick a fitted model first")
    if not (MODELS_DIR / name / length_classify.MODEL_FILE).exists():
        raise ValueError("models/%s is not a length-only model" % name)
    text = (form.get("transcript") or "").strip()
    if not text:
        raise ValueError("paste a transcript, or load one from a dataset")
    if len(text) > 400_000:
        raise ValueError("that is longer than any call in the datasets - "
                         "paste one call, not a whole file")
    threshold = numeric(form, {"threshold": ("threshold", int, 1, 200000,
                                             None)}, "threshold")
    direction = str(form.get("direction") or "").strip()
    if direction and direction not in (length_classify.LONGER,
                                       length_classify.SHORTER):
        raise ValueError("direction is 'longer' or 'shorter'")

    out = LENGTH_SLOT.for_model(name, []).ask(
        {"transcript": text, "threshold": threshold,
         "direction": direction or None,
         "strip_tags": bool(form.get("strip_tags"))})
    if not out.get("ok"):
        raise ValueError(out.get("error") or "the model could not score that")
    return out


def remove_length_model(name):
    name = (name or "").strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("bad model name")
    d = MODELS_DIR / name
    if not (d / length_classify.MODEL_FILE).exists():
        raise ValueError("no such length-only model: %s" % name)
    for slot in SLOTS:
        if slot.worker is not None and slot.worker.name == name:
            slot.unload()
    shutil.rmtree(d)
    return {"deleted": True, "id": name}


def start_length_train_run(form):
    """Fit a threshold. Detached and logged like every other run.

    It is one sort of the fitting set, so it finishes faster than the page
    can ask about it, and like the bag of words it does not take the run lock
    - there is nothing for it to fight a benchmark over.
    """
    name = str(form.get("name", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("a name is letters, digits, dot, dash or underscore")
    ds = form.get("dataset", "")
    if ds not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    overwrite = bool(form.get("overwrite"))
    if (MODELS_DIR / name).exists() and not overwrite:
        raise ValueError("models/%s already exists - pick another name, or "
                         "tick replace" % name)

    flags = ["fit", "--csv", ds, "--name", name]
    pinned = bool(form.get("pin"))
    for key, (flag, _c, _lo, _hi, _d) in LENGTH_TRAIN_FIELDS.items():
        # the threshold rides only when the form asked for it to be pinned:
        # sending it otherwise would turn every fit into a pinned one
        if key == "threshold" and not pinned:
            continue
        val = numeric(form, LENGTH_TRAIN_FIELDS, key)
        if val is None or (key == "limit" and not val):
            continue
        flags += [flag, str(val)]
    if pinned and "--threshold" not in flags:
        raise ValueError("pinning the threshold needs a number to pin it to")
    if pinned:
        flags += ["--direction", str(form.get("direction")
                                     or length_classify.LONGER)]
    metric = str(form.get("metric") or "f1")
    if metric not in ("f1", "acc"):
        raise ValueError("the sweep maximises f1 or acc")
    flags += ["--metric", metric]
    if form.get("strip_tags"):
        flags.append("--strip-tags")
    if overwrite:
        flags.append("--overwrite")
        for slot in SLOTS:
            if slot.worker is not None and slot.worker.name == name:
                slot.unload()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_length_" + name
    log = run_path(run_id, "log")

    # The same venv preamble the other two fits use: run_all.sh is not
    # involved, and the server's own python is not the one with pandas on it.
    quoted = " ".join(shlex.quote(f) for f in flags)
    body = [
        "fit() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
        '  printf "\\n==> fit\\n"',
        '  "$PY" -u scripts/length_classify.py ' + quoted
        + ' || { printf "  fail fit\\n"; return 1; }',
        '  printf "  ok fit\\n"',
        "}", "fit",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        out.write(("$ python scripts/length_classify.py " + quoted
                   + "\n\n").encode())
        out.flush()
        proc = subprocess.Popen(
            [BASH, "-c", "\n".join(body)], cwd=str(PROJECT_DIR), stdout=out,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            **DETACHED)

    LIVE[run_id] = proc
    meta = {"id": run_id, "pid": proc.pid, "kind": "length_train",
            "model_name": name, "label": "fit length · " + name,
            "dataset": ds, "baseline": "length:" + name, "limit": "-",
            "model": "one threshold", "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


# ---------------------------------------------------------------- LLM judge
# The last page, and the simplest thing in the project: no retrieval, no
# ontology, no fine-tuned anything. The transcript goes to the local model,
# which says Fraud or Normal and gives its reason. It is the llm_only control
# from the Benchmark page asked one call at a time, using the same prompt and
# the same verdict parser, so what it says here is what the benchmark would
# have recorded for that call.
#
# llm_judge's module level is standard library only and it speaks to ollama
# over plain HTTP, so this page works from the system python like the rest of
# the server - no venv, no subprocess, nothing to keep alive between clicks.
import llm_judge
# Also standard library only, and for the same reason: it is the fitting side
# of the same page, and this server starts without the venv.
import llm_fit

# (kwarg, cast, low, high, default) - the same shape numeric() reads for the
# other two pages.
LLM_FIELDS = {
    "max_tokens":  ("max_tokens", int, 32, 4000, llm_judge.DEFAULT_MAX_TOKENS),
    "num_ctx":     ("num_ctx", int, 512, 131072, llm_judge.DEFAULT_NUM_CTX),
    "temperature": ("temperature", float, 0.0, 2.0, 0.0),
}
# The fitting side of the LLM page. Nothing here fine-tunes anything - the
# weights Ollama is holding do not move and cannot be moved from here. What is
# fitted is the prompt: worked examples drawn from a dataset, a rubric the
# model writes from them, and the standing instructions box made durable. The
# page says so in those words, because "trained" next to a page that really
# does train a BERT would be a lie.
#
# The one thing that makes it worth doing rather than guessing is that fitting
# scores the holdout twice, fitted and bare, so the run reports what the
# prompt bought rather than just an accuracy.
LLM_FIT_FIELDS = {
    "shots":         ("--shots", int, 0, llm_fit.MAX_SHOTS, 4),
    "shot_words":    ("--shot-words", int, 20, llm_fit.MAX_SHOT_WORDS, 120),
    "holdout_calls": ("--holdout-calls", int, 0, 400, 20),
    "seed":          ("--seed", int, 0, 2 ** 31 - 1, 42),
    "limit":         ("--limit", int, 0, 100000, 0),
}

LLM_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,120}$")
LLM_TIMEOUT = 600
# One generation at a time. Ollama will queue a second, but a 14B model is
# most of the VRAM and two people clicking at once should be told so rather
# than both sitting on a spinner.
LLM_LOCK = threading.Lock()


def llm_config():
    """Everything the LLM judge page needs to draw itself once."""
    cfg = {
        "datasets": datasets(),
        "host": llm_judge.OLLAMA_HOST,
        "default_model": llm_judge.DEFAULT_MODEL,
        "defaults": {k: v[4] for k, v in LLM_FIELDS.items()},
        "fit_defaults": {k: v[4] for k, v in LLM_FIT_FIELDS.items()},
        "profiles": llm_fit.list_models(),
        "models": [],
    }
    # ollama being down is a normal state for this page to be in - the box may
    # not have it running yet - so it is reported, not raised
    try:
        cfg["models"] = llm_judge.list_models()
    except RuntimeError as e:
        cfg["ollama_error"] = str(e)
    return cfg


def llm_verdict(form):
    """Put one transcript to the local model."""
    text = (form.get("transcript") or "").strip()
    if not text:
        raise ValueError("paste a transcript, or load one from a dataset")
    model = str(form.get("model") or llm_judge.DEFAULT_MODEL).strip()
    if not LLM_MODEL_RE.match(model):
        raise ValueError("%r is not a name ollama would accept" % model[:60])
    kw = {}
    for key in LLM_FIELDS:
        val = numeric(form, LLM_FIELDS, key)
        if val is not None:
            kw[LLM_FIELDS[key][0]] = val
    # Standing instructions from the box under the transcript. Held nowhere:
    # the page sends them with every question and the server forgets them the
    # moment it has answered.
    guidance = (form.get("guidance") or "").strip()
    if len(guidance) > llm_judge.MAX_GUIDANCE:
        raise ValueError("that is a lot of instructions - keep them under %d "
                         "characters" % llm_judge.MAX_GUIDANCE)
    # A fitted prompt, if one is picked. "" means the bare llm_only control,
    # which is the page's default and the only setting whose answer is
    # comparable with a benchmark row.
    profile = None
    pname = str(form.get("profile") or "").strip()
    if pname:
        if not BERT_NAME_RE.match(pname):
            raise ValueError("bad profile name")
        if not (MODELS_DIR / pname / llm_fit.PROFILE_FILE).exists():
            raise ValueError("models/%s is not a fitted prompt" % pname)
        profile = llm_fit.load_profile(pname)
    if not LLM_LOCK.acquire(blocking=False):
        raise ValueError("the model is already answering something - one call "
                         "at a time, or they fight for the VRAM")
    try:
        return llm_judge.judge(text, model=model, timeout=LLM_TIMEOUT,
                               guidance=guidance, profile=profile, **kw)
    except RuntimeError as e:
        raise ValueError(str(e))
    finally:
        LLM_LOCK.release()


def remove_model(name):
    """Delete one checkpoint. Several hundred megabytes each, so the page asks
    first and this says exactly what went."""
    name = (name or "").strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("bad model name")
    d = MODELS_DIR / name
    if not d.is_dir() or d.resolve().parent != MODELS_DIR.resolve():
        raise ValueError("no such model: %s" % name)
    for slot in SLOTS:              # cannot delete a model that is open
        if slot.worker is not None and slot.worker.name == name:
            slot.unload()
    shutil.rmtree(d)
    return {"deleted": True, "id": name}


def start_train_run(form):
    """Fine-tune a checkpoint. Detached and logged like every other run, so it
    appears in Recent runs and can be stopped with the same button."""
    name = str(form.get("name", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("the model needs a name: letters, digits, dot, dash "
                         "or underscore, starting with a letter or digit")
    ds = form.get("dataset", "")
    if ds not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    base = str(form.get("base", ""))
    if base not in {b[0] for b in BASE_MODELS}:
        raise ValueError("pick a model to start from")
    overwrite = bool(form.get("overwrite"))
    if (MODELS_DIR / name).exists() and not overwrite:
        raise ValueError("models/%s already exists - pick another name, or "
                         "tick \"replace it\"" % name)

    running = [r for r in all_runs() if r["status"] == "running"]
    if running:
        raise ValueError("a run is already going (%s). Stop it first - the GPU "
                         "cannot hold two." % running[0]["id"])

    flags = ["train", "--csv", ds, "--name", name, "--model", base]
    for key, (flag, _, _, _, _) in TRAIN_FIELDS.items():
        val = numeric(form, TRAIN_FIELDS, key)
        if val is None or (key == "limit" and val == 0):
            continue            # limit 0 is "the whole dataset", i.e. no flag
        flags += [flag, str(val)]
    if form.get("cpu"):
        flags.append("--cpu")
    if overwrite:
        flags.append("--overwrite")
        # the checkpoint is about to be rewritten underneath anything holding it
        for slot in SLOTS:
            if slot.worker is not None and slot.worker.name == name:
                slot.unload()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_train_" + name
    log = run_path(run_id, "log")

    # run_all.sh is not involved, so its venv preamble is repeated here - the
    # same shape start_kb_run uses, for the same reason.
    quoted = " ".join(shlex.quote(f) for f in flags)
    body = [
        "train() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
        '  printf "\\n==> train\\n"',
        '  "$PY" -u scripts/bert_classify.py ' + quoted
        + ' || { printf "  fail train\\n"; return 1; }',
        '  printf "  ok train\\n"',
        "}", "train",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        out.write(("$ python scripts/bert_classify.py " + quoted + "\n\n").encode())
        out.flush()
        proc = subprocess.Popen(
            [BASH, "-c", "\n".join(body)], cwd=str(PROJECT_DIR), stdout=out,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            **DETACHED)

    LIVE[run_id] = proc
    meta = {"id": run_id, "pid": proc.pid, "kind": "train", "model_name": name,
            "label": "train BERT · " + name, "dataset": ds,
            "baseline": "train:" + name, "limit": "-", "model": base,
            "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # one line per run, not per poll
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/api/config":
                return self._send(200, {
                    "datasets": datasets(),
                    "baselines": [{"key": k, "label": l, "note": n, "needs_model": m}
                                  for k, l, n, m in BASELINES],
                    "models": MODELS,
                    "ollama": ollama_state(),
                    "kb_modes": [{"key": k, "label": l, "note": n,
                                  "harvests": h is not None, "exclusive": x}
                                 for k, l, n, h, _, x in KB_MODES],
                    "kb": kb_state(),
                    "idx_pool": idx_pool(),
                })
            if u.path == "/api/kb":
                return self._send(200, {"kb": kb_state()})
            if u.path == "/api/admin/version":
                return self._send(200, version_info())
            if u.path == "/api/ledger":
                ledger_sync()
                return self._send(200, ledger_view(q.get("dataset", [""])[0]))
            if u.path == "/api/runs":
                return self._send(200, {"runs": all_runs()})
            # "output", not "log": privacy filter lists block paths that
            # look like telemetry, and Edge has tracking prevention on by
            # default - these two endpoints were the only ones carrying the
            # word "log", and the only ones that never arrived there. The old
            # paths stay as aliases so a page left open somewhere still works.
            # Two names for one endpoint. The page must ask for /api/output:
            # ad blockers and Edge tracking prevention block "/api/log" as a
            # telemetry path, and a blocked poll leaves a run looking hung or
            # failed. /api/log stays for anything already pointed at it.
            if u.path in ("/api/output", "/api/log"):
                rid = q.get("id", [""])[0]
                off = int(q.get("offset", ["0"])[0])
                meta = load_run(rid)
                if not meta and not run_path(rid, "log").exists():
                    return self._send(404, {"error": "no such run: %s" % rid})
                out = read_log(rid, off)
                # a run whose .json went missing still has its output, and
                # that is the part worth showing
                out["status"] = meta["status"] if meta else run_status({"id": rid})
                return self._send(200, out)
            if u.path == "/api/results":
                return self._send(200, {"results": results_of(q.get("id", [""])[0])})
            if u.path == "/api/artifacts":
                return self._send(200, artifacts_of(q.get("id", [""])[0]))
            if u.path in ("/api/step", "/api/steplog"):
                return self._send(200, step_log(q.get("id", [""])[0],
                                                q.get("name", [""])[0]))
            if u.path == "/api/csv":
                return self._send(200, csv_page(
                    q.get("id", [""])[0], q.get("name", [""])[0],
                    int(q.get("offset", ["0"])[0]),
                    min(200, int(q.get("limit", ["50"])[0]))))
            # ---- the BERT page
            if u.path == "/api/bert/config":
                return self._send(200, bert_config())
            if u.path == "/api/bert/models":
                return self._send(200, {"models": bert_classify.list_models(),
                                        "worker": worker_state()})
            if u.path == "/api/dataset/sample":
                return self._send(200, dataset_row(
                    q.get("dataset", [""])[0], int(q.get("idx", ["0"])[0])))
            if u.path == "/api/dataset/context":
                return self._send(200,
                                  dataset_context(q.get("dataset", [""])[0]))
            # ---- the Bag of words page
            if u.path == "/api/bow/config":
                return self._send(200, bow_config())
            if u.path == "/api/bow/models":
                return self._send(200, {"models": bow_classify.list_models(),
                                        "worker": BOW_SLOT.state()})
            if u.path == "/api/eval/result":
                return self._send(200, eval_result(q.get("id", [""])[0]))
            if u.path == "/api/llm/profiles":
                return self._send(200, {"profiles": llm_fit.list_models()})
            if u.path == "/api/length/config":
                return self._send(200, length_config())
            if u.path == "/api/length/models":
                return self._send(200,
                                  {"models": length_classify.list_models(),
                                   "worker": LENGTH_SLOT.state()})
            # ---- the LLM judge page
            if u.path == "/api/llm/config":
                return self._send(200, llm_config())
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            form = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        try:
            if u.path == "/api/run":
                meta = start_run(form)
                sys.stderr.write("started %s  %s / %s / limit %s%s\n" % (
                    meta["id"], meta["dataset"], meta["baseline"], meta["limit"],
                    ("  " + meta["model"]) if meta["model"] else ""))
                return self._send(200, meta)
            if u.path == "/api/kb":
                meta = start_kb_run(form)
                sys.stderr.write("started %s  knowledge base / %s\n"
                                 % (meta["id"], meta["mode"]))
                return self._send(200, meta)
            if u.path == "/api/stop":
                return self._send(200, stop_run(form.get("id", "")))
            if u.path == "/api/delete":
                out = (delete_finished() if form.get("scope") == "finished"
                       else delete_run(form.get("id", "")))
                sys.stderr.write("deleted %s\n" % json.dumps(out))
                return self._send(200, out)
            # ---- the BERT page
            if u.path == "/api/bert/train":
                meta = start_train_run(form)
                sys.stderr.write("started %s  train %s on %s\n"
                                 % (meta["id"], meta["model"], meta["dataset"]))
                return self._send(200, meta)
            if u.path == "/api/bert/classify":
                return self._send(200, bert_verdict(form))
            if u.path == "/api/bert/unload":
                return self._send(200, unload_answerer())
            if u.path == "/api/bert/delete_model":
                out = remove_model(form.get("name", ""))
                sys.stderr.write("deleted checkpoint %s\n" % out["id"])
                return self._send(200, out)
            # ---- the Bag of words page
            if u.path == "/api/bow/train":
                meta = start_bow_train_run(form)
                sys.stderr.write("started %s  fit bow %s on %s\n"
                                 % (meta["id"], meta["model_name"],
                                    meta["dataset"]))
                return self._send(200, meta)
            if u.path == "/api/bow/classify":
                return self._send(200, bow_verdict(form))
            if u.path == "/api/bow/unload":
                return self._send(200, BOW_SLOT.unload())
            if u.path == "/api/bow/delete_model":
                out = remove_bow_model(form.get("name", ""))
                sys.stderr.write("deleted bow model %s\n" % out["id"])
                return self._send(200, out)
            # ---- the Length only page
            if u.path == "/api/length/train":
                meta = start_length_train_run(form)
                sys.stderr.write("started %s  fit length %s on %s\n"
                                 % (meta["id"], meta["model_name"],
                                    meta["dataset"]))
                return self._send(200, meta)
            if u.path == "/api/length/classify":
                return self._send(200, length_verdict(form))
            if u.path == "/api/length/unload":
                return self._send(200, LENGTH_SLOT.unload())
            if u.path == "/api/length/delete_model":
                out = remove_length_model(form.get("name", ""))
                sys.stderr.write("deleted length model %s\n" % out["id"])
                return self._send(200, out)
            if u.path == "/api/admin/stop":
                sys.stderr.write("stop requested from the web UI\n")
                return self._send(200, admin_stop())
            if u.path == "/api/admin/update":
                out = admin_update()
                sys.stderr.write("update from the web UI: %s -> %s%s\n"
                                 % (out["before"], out["after"],
                                    ", restarting" if out["restarting"] else ""))
                return self._send(200, out)
            if u.path == "/api/ledger/hide":
                return self._send(200, ledger_hide(form.get("run_id", ""),
                                                   form.get("system", "")))
            # ---- scoring a whole dataset, from any of the four pages
            if u.path.startswith("/api/") and u.path.endswith("/evaluate"):
                page = u.path[len("/api/"):-len("/evaluate")]
                meta = start_eval_run(page, form)
                sys.stderr.write("started %s  score %s %s on %s\n"
                                 % (meta["id"], page, meta["model_name"]
                                    or "(bare)", meta["dataset"]))
                return self._send(200, meta)
            # ---- the LLM judge page
            if u.path == "/api/llm/judge":
                return self._send(200, llm_verdict(form))
            if u.path == "/api/llm/fit":
                meta = start_llm_fit_run(form)
                sys.stderr.write("started %s  fit prompt %s on %s\n"
                                 % (meta["id"], meta["model_name"],
                                    meta["dataset"]))
                return self._send(200, meta)
            if u.path == "/api/llm/delete_profile":
                out = remove_llm_profile(form.get("name", ""))
                sys.stderr.write("deleted fitted prompt %s\n" % out["id"])
                return self._send(200, out)
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)})


# ------------------------------------------------------- scoring a dataset
# The fourth thing each model page can do, and the one that turns four toys
# into four measurements: put a whole dataset to a fitted model and report the
# confusion matrix.
#
# All four go out through the same shape - a detached run with a log, writing
# a metrics JSON beside it - so the page code is one form and one card rather
# than four, and the numbers from the four pages are computed by the same
# function in eval_common and can be read side by side.
#
# What it is NOT is the Benchmark page. That cross-validates: every call is
# predicted by a model that never saw it. This scores calls with one already
# fitted model, so pointing a model at its own training set measures memory.
# Every evaluate run says so when the two datasets match, and the page repeats
# it on the card.
EVAL_PAGES = {
    # page id -> (script, subcommand flags builder)
    "bert":   BERT_SCRIPT,
    "bow":    BOW_SCRIPT,
    "length": LENGTH_SCRIPT,
    "llm":    PROJECT_DIR / "scripts" / "llm_fit.py",
}

EVAL_FIELDS = {
    "limit":     ("--limit", int, 0, 100000, 0),
    "threshold": ("--threshold", float, 0.0, 1.0, None),
}


def eval_result(run_id):
    """The metrics a finished evaluate run wrote, or why there are none.

    "Not ready" is three different things - still going, finished and wrote
    nothing, crashed - and a page that cannot tell them apart can only say
    "no output", which is what it said. So the reason and the tail of the run
    come back with the answer, and the page prints them where the numbers
    would have been rather than leaving someone to go and find a log.
    """
    run_id = checked_run_id(run_id)
    path = run_path(run_id, "metrics.json")
    meta = load_run(run_id)
    status = (meta or {}).get("status")

    if path.exists():
        try:
            out = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ready": False, "status": status,
                    "why": "the results file is there but could not be read "
                           "(%s)" % e, "tail": log_tail(run_id)}
        out["ready"] = True
        out["status"] = status
        return out

    if status == "running":
        return {"ready": False, "status": status, "why": "still going"}
    return {
        "ready": False,
        "status": status,
        "why": ("the run stopped before it scored anything"
                if status == "stopped" else
                "the run ended without writing a score" if status == "done"
                else "the run failed"),
        "tail": log_tail(run_id),
    }


def log_tail(run_id, lines=30):
    """The last few meaningful lines of a run's log.

    Blank lines are dropped from the end: read_log strips the exit marker and
    leaves its newlines behind, so a log shown scrolled to the bottom can be
    all whitespace - which is exactly how a crash comes to look like no
    output at all.
    """
    log = run_path(run_id, "log")
    if not log.exists():
        return ("(the run left no log at all - the process could not be "
                "started, or something removed it)")
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return "(could not read the log: %s)" % e
    text = ANSI.sub("", text)
    text = re.sub(re.escape(EXIT_MARK) + r"\s+\d+\s*", "", text)
    kept = [ln for ln in text.rstrip().split("\n")]
    return "\n".join(kept[-lines:])


def start_eval_run(page, form):
    """Score a whole dataset with one fitted model, on any of the four pages.

    Only the flags differ between pages; everything else - the detached run,
    the log, the metrics file the page reads afterwards - is shared, because
    a score that was computed differently per page could not be compared
    across them, which is the entire point of having four of them.
    """
    script = EVAL_PAGES.get(page)
    if script is None:
        raise ValueError("unknown page")
    ds = form.get("dataset", "")
    if ds not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")

    name = str(form.get("model", "")).strip()
    flags = ["evaluate", "--csv", ds]

    if page == "llm":
        # the LLM page's "model" is what ollama has pulled; the fitted prompt
        # is a separate, optional thing
        base = str(form.get("base_model") or llm_judge.DEFAULT_MODEL).strip()
        if not LLM_MODEL_RE.match(base):
            raise ValueError("%r is not a name ollama would accept" % base[:60])
        flags += ["--model", base]
        if name:
            if not BERT_NAME_RE.match(name):
                raise ValueError("bad profile name")
            if not (MODELS_DIR / name / llm_fit.PROFILE_FILE).exists():
                raise ValueError("models/%s is not a fitted prompt" % name)
            flags += ["--profile", name]
        for key in ("num_ctx", "max_tokens", "temperature"):
            val = numeric(form, LLM_FIELDS, key)
            if val is not None:
                flags += ["--" + key.replace("_", "-"), str(val)]
        running = [r for r in all_runs() if r["status"] == "running"]
        if running:
            raise ValueError("a run is already going (%s). Stop it first - "
                             "this wants the model ollama is holding, and so "
                             "does that." % running[0]["id"])
    else:
        if not BERT_NAME_RE.match(name):
            raise ValueError("pick a fitted model first")
        marker = {"bert": "config.json", "bow": bow_classify.MODEL_FILE,
                  "length": length_classify.MODEL_FILE}[page]
        if not (MODELS_DIR / name / marker).exists():
            raise ValueError("models/%s is not a %s model" % (name, page))
        flags += ["--name", name]
        if form.get("strip_tags"):
            flags.append("--strip-tags")

    if page == "bert":
        # a whole dataset through BERT is the one evaluate that wants the GPU,
        # and the one that fights a benchmark for it
        running = [r for r in all_runs() if r["status"] == "running"]
        if running and form.get("gpu"):
            raise ValueError("a run is already going (%s), and it has the "
                             "GPU. Stop it first, or untick \"use the GPU\" "
                             "to score on the CPU alongside it."
                             % running[0]["id"])
        flags.append("--gpu" if form.get("gpu") else "--cpu")

    limit = numeric(form, EVAL_FIELDS, "limit")
    if limit:
        flags += ["--limit", str(limit)]
    if page in ("bert", "bow"):
        thr = numeric(form, EVAL_FIELDS, "threshold")
        if thr is not None:
            flags += ["--threshold", str(thr)]
    if page == "length":
        thr = numeric(form, {"threshold": ("threshold", int, 1, 200000, None)},
                      "threshold")
        if thr is not None:
            flags += ["--threshold", str(thr)]
        way = str(form.get("direction") or "").strip()
        if way in (length_classify.LONGER, length_classify.SHORTER):
            flags += ["--direction", way]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = (time.strftime("%Y%m%d_%H%M%S") + "_eval_" + page + "_"
              + (name or "bare"))
    flags += ["--out", str(run_path(run_id, "metrics.json"))]

    quoted = " ".join(shlex.quote(f) for f in flags)
    rel = script.relative_to(PROJECT_DIR).as_posix()
    body = [
        "score() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
        '  printf "\\n==> score\\n"',
        '  "$PY" -u %s ' % shlex.quote(rel) + quoted
        + ' || { printf "  fail score\\n"; return 1; }',
        '  printf "  ok score\\n"',
        "}", "score",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    log = run_path(run_id, "log")
    with open(log, "wb") as out:
        out.write(("$ python %s %s\n\n" % (rel, quoted)).encode())
        out.flush()
        proc = subprocess.Popen(
            [BASH, "-c", "\n".join(body)], cwd=str(PROJECT_DIR), stdout=out,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            **DETACHED)

    LIVE[run_id] = proc
    meta = {"id": run_id, "pid": proc.pid, "kind": "eval",
            "page": page, "model_name": name,
            "label": "score %s · %s" % (page, name or "bare"),
            "dataset": ds, "baseline": "%s:%s" % (page, name or "bare"),
            "limit": str(limit or "-"), "model": name or page,
            "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def remove_llm_profile(name):
    name = (name or "").strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("bad profile name")
    d = MODELS_DIR / name
    if not (d / llm_fit.PROFILE_FILE).exists():
        raise ValueError("no such fitted prompt: %s" % name)
    shutil.rmtree(d)
    return {"deleted": True, "id": name}


def start_llm_fit_run(form):
    """Fit a prompt. Detached and logged like every other run.

    This one is not quick: it scores the holdout twice, so it is two LLM calls
    per held-out call plus one for the rubric, and on a 14B model that is
    minutes rather than seconds. It refuses to start alongside another run for
    the same reason a BERT training run does - it wants the model Ollama is
    holding, and a benchmark running at the same time wants the same one.
    """
    name = str(form.get("name", "")).strip()
    if not BERT_NAME_RE.match(name):
        raise ValueError("a name is letters, digits, dot, dash or underscore")
    ds = form.get("dataset", "")
    if ds not in {d["path"] for d in datasets()}:
        raise ValueError("unknown dataset")
    model = str(form.get("model") or llm_judge.DEFAULT_MODEL).strip()
    if not LLM_MODEL_RE.match(model):
        raise ValueError("%r is not a name ollama would accept" % model[:60])
    overwrite = bool(form.get("overwrite"))
    if (MODELS_DIR / name).exists() and not overwrite:
        raise ValueError("models/%s already exists - pick another name, or "
                         "tick replace" % name)

    running = [r for r in all_runs() if r["status"] == "running"]
    if running:
        raise ValueError("a run is already going (%s). Stop it first - this "
                         "wants the model ollama is holding, and so does "
                         "that." % running[0]["id"])

    flags = ["fit", "--csv", ds, "--name", name, "--model", model]
    for key, (flag, _c, _lo, _hi, _d) in LLM_FIT_FIELDS.items():
        val = numeric(form, LLM_FIT_FIELDS, key)
        if val is None or (key == "limit" and not val):
            continue
        flags += [flag, str(val)]
    if form.get("rubric"):
        flags.append("--rubric")
    if overwrite:
        flags.append("--overwrite")

    num_ctx = numeric(form, LLM_FIELDS, "num_ctx")
    if num_ctx:
        flags += ["--num-ctx", str(num_ctx)]

    # The standing instructions go through a file rather than the command
    # line: they are free text a person typed, they can run to pages, and a
    # command line is not where either of those belongs.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_llmfit_" + name
    guidance = (form.get("guidance") or "").strip()
    if len(guidance) > llm_judge.MAX_GUIDANCE:
        raise ValueError("that is a lot of instructions - keep them under %d "
                         "characters" % llm_judge.MAX_GUIDANCE)
    if guidance:
        gfile = run_path(run_id, "guidance.txt")
        with open(gfile, "w", encoding="utf-8") as f:
            f.write(guidance)
        flags += ["--guidance-file", str(gfile)]

    log = run_path(run_id, "log")
    quoted = " ".join(shlex.quote(f) for f in flags)
    body = [
        "fit() {",
        '  if [ -f venv/bin/activate ]; then . venv/bin/activate;',
        '  elif [ -f venv/Scripts/activate ]; then . venv/Scripts/activate;',
        '  else printf "  warn no venv/, using whatever python is on PATH\\n"; fi',
        '  PY=python; command -v python >/dev/null 2>&1 || PY=python3',
        '  printf "\\n==> fit\\n"',
        '  "$PY" -u scripts/llm_fit.py ' + quoted
        + ' || { printf "  fail fit\\n"; return 1; }',
        '  printf "  ok fit\\n"',
        "}", "fit",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        out.write(("$ python scripts/llm_fit.py " + quoted + "\n\n").encode())
        out.flush()
        proc = subprocess.Popen(
            [BASH, "-c", "\n".join(body)], cwd=str(PROJECT_DIR), stdout=out,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            **DETACHED)

    LIVE[run_id] = proc
    meta = {"id": run_id, "pid": proc.pid, "kind": "llm_fit",
            "model_name": name, "label": "fit prompt · " + name,
            "dataset": ds, "baseline": "llm_prompt:" + name, "limit": "-",
            "model": model, "started": time.time()}
    with open(run_path(run_id, "json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scam-detection benchmark</title>
<style>
  :root {
    --bg:#f7f7f5; --panel:#fff; --ink:#1c1b19; --dim:#6b6862; --line:#e2e0db;
    --accent:#2f6f4f; --warn:#8a6d1f; --bad:#a33a2a; --mono:ui-monospace,
    SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --panel:#1e1e23; --ink:#e8e6e1; --dim:#9a968e;
            --line:#32323a; --accent:#6bbf90; --warn:#d8b464; --bad:#e08272; }
  }
  * { box-sizing:border-box; }
  /* .wrap is display:grid, and an explicit display beats the hidden
     attribute - without this the two pages render one under the other */
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:17px; font-weight:600; letter-spacing:-.01em; }
  header .sub { color:var(--dim); font-size:13px; }
  .wrap { display:grid; grid-template-columns:340px 1fr; gap:0; align-items:start; }
  /* The sidebar stays put while the output log scrolls. Reading a long log is
     exactly when you want to reach another run, and before this the history
     scrolled away with everything else. */
  .side { padding:22px; border-right:1px solid var(--line);
          position:sticky; top:0; max-height:100vh; overflow-y:auto; }
  .main { padding:22px; min-width:0; }
  @media (max-width:900px){
    .wrap { grid-template-columns:1fr; }
    /* one column: the sidebar is above the content, so pinning it would put a
       full-height scroll box in front of everything */
    .side { position:static; max-height:none; overflow:visible;
            border-right:none; border-bottom:1px solid var(--line); }
  }

  /* ---- the three blocks the sidebar is made of ---- */
  .sect { padding-bottom:20px; margin-bottom:20px;
          border-bottom:1px solid var(--line); }
  .sect:last-child { padding-bottom:0; margin-bottom:0; border-bottom:none; }
  .secthead { font-size:11px; font-weight:700; letter-spacing:.08em;
              text-transform:uppercase; color:var(--dim); margin-bottom:10px; }
  label { display:block; font-weight:600; margin:16px 0 6px; font-size:13px; }
  label:first-of-type { margin-top:0; }
  select, input[type=text] {
    width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:6px;
    background:var(--panel); color:var(--ink); font:inherit; }
  .hint { color:var(--dim); font-size:12px; margin-top:5px; }
  .checks { margin-top:8px; }
  .checks label { display:flex; gap:9px; align-items:flex-start; margin:0;
                  padding:6px 0; font-weight:400; cursor:pointer;
                  border-bottom:1px solid var(--line); }
  .checks label:last-child { border-bottom:none; }
  .checks input { margin:3px 0 0; flex:none; }
  .checks .name { font-weight:600; }
  .checks .note { color:var(--dim); font-size:12px; display:block; }
  button.link { background:none; border:none; padding:0; color:var(--accent);
                font-size:12px; font-weight:600; text-decoration:underline;
                cursor:pointer; }
  button { font:inherit; font-weight:600; padding:9px 16px; border-radius:6px;
           border:1px solid transparent; cursor:pointer; }
  .go { background:var(--accent); color:#fff; width:100%; margin-top:20px; }
  .go[disabled] { opacity:.5; cursor:not-allowed; }
  .stop { background:transparent; color:var(--bad); border-color:var(--bad); }
  .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:14px 16px; margin-bottom:18px; }

  /* ---- the tab strip that appears once a run is selected ---- */
  .tabs { display:flex; gap:4px; border-bottom:1px solid var(--line);
          margin-bottom:16px; flex-wrap:wrap; }
  .tabs button { background:none; border:none; border-bottom:2px solid transparent;
                 border-radius:0; padding:9px 14px; color:var(--dim); font-weight:600; }
  .tabs button:hover { color:var(--ink); }
  .tabs button.on { color:var(--accent); border-bottom-color:var(--accent); }
  .tabs button[disabled] { opacity:.4; cursor:not-allowed; }
  .tabs .count { font-weight:400; font-size:12px; opacity:.75; }

  pre.log { background:var(--panel); border:1px solid var(--line); border-radius:8px;
            padding:14px 16px; font-family:var(--mono); font-size:12.5px;
            line-height:1.5; max-height:52vh; overflow:auto; white-space:pre-wrap;
            word-break:break-word; margin:0; }
  .l-step { color:var(--accent); font-weight:600; }
  .l-ok   { color:var(--accent); }
  .l-warn { color:var(--warn); }
  .l-fail { color:var(--bad); font-weight:600; }

  .scroll { overflow-x:auto; }

  /* Chart palette: slots 1-4 of the reference categorical order, validated
     against this card's own surface (#fff light, #1e1e23 dark) - every check
     passes; light mode returns a contrast WARN on aqua and yellow, whose
     relief is the value label on every bar plus the Table view beside it. */
  .viz {
    --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#eda100;
    --grid:#e1e0d9; --axis:#c3c2b7;
  }
  @media (prefers-color-scheme: dark) {
    .viz {
      --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
      --grid:#2c2c2a; --axis:#383835;
    }
  }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:7px;
         overflow:hidden; margin-bottom:14px; }
  .seg button { border:none; border-radius:0; background:transparent; color:var(--dim);
                padding:6px 15px; font-size:13px; }
  .seg button + button { border-left:1px solid var(--line); }
  .seg button.on { background:var(--accent); color:#fff; }
  .legend { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
  .legend button { display:flex; align-items:center; gap:7px; font-size:12px;
                   font-weight:500; color:var(--ink); background:transparent;
                   border:1px solid var(--line); border-radius:99px;
                   padding:4px 12px 4px 9px; }
  .legend button:hover { border-color:var(--dim); }
  .legend button.off { color:var(--dim); }
  .legend button.off i { background:transparent !important;
                         box-shadow:inset 0 0 0 1.5px var(--dim); }
  .legend i { width:11px; height:11px; border-radius:3px; display:block;
              flex:none; }
  .viz text { font:12px ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  .viz .tick { fill:var(--dim); }
  .viz .name { fill:var(--ink); font-weight:600; }
  .viz .val  { fill:var(--dim); font-size:11px; }
  .viz .band { fill:transparent; }
  .viz .band:hover { fill:var(--line); opacity:.35; }
  .tip { position:fixed; z-index:9; pointer-events:none; background:var(--panel);
         border:1px solid var(--line); border-radius:7px; padding:9px 11px;
         font-size:12px; box-shadow:0 6px 20px rgba(0,0,0,.16); max-width:280px; }
  .tip b { display:block; margin-bottom:5px; font-family:var(--mono); }
  .tip table { width:auto; }
  .tip td { border:none; padding:1px 0 1px 12px; text-align:right; }
  .tip td:first-child { padding-left:0; text-align:left; color:var(--dim); }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--line);
           white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; font-family:var(--mono); }
  th { color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase;
       letter-spacing:.04em; }
  tr.skipped td { color:var(--dim); }
  table.calls td, table.calls th { font-size:12.5px; }
  table.calls td.text { text-align:left; white-space:normal; min-width:340px;
                        max-width:640px; color:var(--dim); }
  table.calls td.hit  { color:var(--accent); }
  table.calls td.miss { color:var(--bad); font-weight:600; }
  /* The model's own account of the call. Prose, so it wraps and is never
     coloured right/wrong - it is not a prediction to score. */
  table.calls td.why { text-align:left; white-space:normal; color:var(--dim);
                       font-size:12px; min-width:190px; max-width:300px;
                       font-style:italic; }
  table.calls th.why { color:var(--dim); font-style:italic; }

  .pill { font-size:12px; padding:2px 9px; border-radius:99px; border:1px solid var(--line);
          color:var(--dim); }
  .pill.running { color:var(--accent); border-color:var(--accent); }
  .pill.failed  { color:var(--bad); border-color:var(--bad); }
  /* the dataset's own label for a loaded row - the answer, before the model
     has been asked. Same colours the verdict uses, so agreeing and
     disagreeing read at a glance. */
  .pill.truth-scam { color:var(--bad); border-color:var(--bad); font-weight:700; }
  .pill.truth-legit { color:var(--accent); border-color:var(--accent); font-weight:700; }
  /* Capped so fifteen runs cannot push the run form below the fold - the
     whole point of moving the list up here. */
  .hist { font-size:13px; max-height:34vh; overflow-y:auto; }
  .hist a { display:block; position:relative; padding:7px 24px 7px 0;
            border-bottom:1px solid var(--line);
            color:inherit; text-decoration:none; cursor:pointer; }
  .hist a:hover { color:var(--accent); }
  .hist a.on { color:var(--accent); font-weight:600; }
  .hist .meta { color:var(--dim); font-size:12px; font-weight:400; }
  /* Hidden until the row is hovered or the button is tabbed to, so a list of
     fifteen runs is not fifteen delete buttons competing with the names. It
     stays reachable from the keyboard either way. */
  .hist .del { position:absolute; top:4px; right:0; width:20px; height:20px;
               padding:0; line-height:18px; text-align:center; font-size:15px;
               font-weight:400; border:1px solid transparent; border-radius:4px;
               background:none; color:var(--dim); opacity:0; cursor:pointer; }
  .hist a:hover .del, .hist .del:focus { opacity:1; }
  .hist .del:hover { color:var(--bad); border-color:var(--bad); }
  .hist .del[disabled] { opacity:0; cursor:not-allowed; }
  .muted { color:var(--dim); }
  .pager { display:flex; align-items:center; gap:12px; margin-top:12px; }
  .pager button { border-color:var(--line); background:var(--panel); color:var(--ink); }
  .pager button[disabled] { opacity:.4; cursor:not-allowed; }
  .filepick { display:flex; align-items:center; gap:10px; margin-bottom:12px;
              flex-wrap:wrap; }
  .filepick select { width:auto; min-width:260px; }
  label.inline { display:flex; align-items:center; gap:6px; margin:0;
                 font-weight:400; font-size:13px; cursor:pointer; }
  label.inline input { margin:0; }

  /* ---- knowledge base: a maintenance job, not part of configuring a run,
     so it folds away. The state line stays in the summary, because "index
     out of step" is worth seeing without opening anything. ---- */
  details.kb { border:1px solid var(--line); border-radius:8px; padding:0 12px; }
  details.kb > summary { position:relative; cursor:pointer; list-style:none;
                         padding:10px 22px 10px 0; font-size:13px;
                         font-weight:600; }
  details.kb > summary::-webkit-details-marker { display:none; }
  /* absolute, not float: the state line under the title is a block, and a
     float placed after it drops onto a line of its own */
  details.kb > summary::after { content:"\25b8"; position:absolute; right:0;
                                top:10px; color:var(--dim); font-weight:400; }
  details.kb[open] > summary::after { content:"\25be"; }
  details.kb > summary:hover { color:var(--accent); }
  details.kb .kbbody { padding-bottom:14px; }
  details.kb .kbbody .go { margin-top:14px; }
  #kbstate { margin-top:3px; font-weight:400; }

  /* ---- the two pages ---- */
  nav.pages { display:flex; gap:5px; }
  nav.pages button { background:none; border:1px solid var(--line); color:var(--dim);
                     padding:5px 14px; border-radius:99px; font-size:13px; }
  nav.pages button:hover { color:var(--ink); border-color:var(--dim); }
  nav.pages button.on { background:var(--accent); border-color:var(--accent);
                        color:#fff; }

  /* ---- BERT ---- */
  textarea { width:100%; padding:10px 12px; border:1px solid var(--line);
             border-radius:6px; background:var(--panel); color:var(--ink);
             font:12.5px/1.6 var(--mono); resize:vertical; min-height:150px; }
  /* seven hyperparameters one under another is a very long sidebar */
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:0 12px; }
  .grid2 label { margin-top:14px; }
  details.adv { border:1px solid var(--line); border-radius:8px; padding:0 12px;
                margin:12px 0; }
  details.adv > summary { cursor:pointer; padding:9px 0; font-size:13px;
                          font-weight:600; color:var(--dim); list-style:none; }
  details.adv > summary::-webkit-details-marker { display:none; }
  details.adv > summary:hover { color:var(--accent); }
  details.adv > summary::before { content:"\25b8 "; color:var(--dim); }
  details.adv[open] > summary::before { content:"\25be "; }
  details.adv .advbody { padding-bottom:12px; }

  /* the headline: the verdict, the probability behind it, and how it moved
     across the call - side by side, because the interesting case is the one
     where reading only the opening would have said something else */
  .verdict { display:flex; gap:30px; flex-wrap:wrap; align-items:flex-end; }
  .big { font-size:30px; font-weight:700; line-height:1.05; letter-spacing:-.02em;
         font-variant-numeric:tabular-nums; }
  .big.scam { color:var(--bad); }
  .big.legitimate { color:var(--accent); }
  .big.uncertain { color:var(--warn); }
  .cap { font-size:11px; font-weight:700; letter-spacing:.08em; margin-bottom:5px;
         text-transform:uppercase; color:var(--dim); }

  .qp { font-weight:600; margin-bottom:8px; }
  .bar { height:5px; background:var(--line); border-radius:99px; margin-top:9px;
         overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--accent); border-radius:99px; }
  .bar.low i { background:var(--warn); }
  /* the stretch of the call that scored highest - without it a probability
     is a claim with nothing behind it */
  .ev { margin-top:9px; color:var(--dim); font-size:12.5px; font-style:italic;
        border-left:2px solid var(--line); padding-left:11px; }
  table.opts td, table.opts th { font-size:12.5px; }
  table.opts td.optname { text-align:left; white-space:normal; font-family:inherit; }
  table.opts tr.chosen td { color:var(--ink); font-weight:700; }
  /* ---- the length axis ----
     One call against the two fitting distributions, on a log scale because
     the transcripts here run from a dozen words to sixty thousand. The bands
     are p10-p90 with the median ticked; the line is the threshold; the
     diamond is the call being asked about. */
  .lenaxis { position:relative; height:64px; margin-top:4px;
             border-bottom:1px solid var(--line); }
  .lenband { position:absolute; height:14px; border-radius:99px; opacity:.45; }
  .lenband.legit { top:12px; background:var(--accent); }
  .lenband.scam { top:34px; background:var(--bad); }
  .lentick { position:absolute; width:2px; height:14px; }
  .lentick.legit { top:12px; background:var(--accent); }
  .lentick.scam { top:34px; background:var(--bad); }
  .lenline { position:absolute; top:4px; bottom:0; width:0;
             border-left:2px dashed var(--dim); }
  .lenhere { position:absolute; top:22px; width:12px; height:12px;
             margin-left:-6px; transform:rotate(45deg); background:var(--ink);
             border:2px solid var(--panel); }
  .lenhere.scam { background:var(--bad); }
  .lenhere.legitimate { background:var(--accent); }
  .lenkey { display:inline-block; width:11px; height:7px; border-radius:99px;
            vertical-align:middle; margin-right:3px; }
  .lenkey.legit { background:var(--accent); }
  .lenkey.scam { background:var(--bad); }
  .lenkey.line { height:0; border-top:2px dashed var(--dim); border-radius:0; }
  .lenkey.here { background:var(--ink); border-radius:0;
                 transform:rotate(45deg) scale(.8); width:8px; height:8px; }
  /* said out loud, not tucked into a hint: the rule has just contradicted
     the data it was fitted on */
  .card.warn { border-color:var(--warn); color:var(--ink); font-size:12.5px; }

  /* a fitted prompt that beat the control, and one that did not - the
     second is the more useful of the two and must not be hidden */
  /* the four cells every other number on the card comes off, so they can be
     checked rather than taken on trust */
  table.cm { border-collapse:collapse; margin-top:4px; }
  table.cm th, table.cm td { padding:7px 14px; text-align:right;
                             border:1px solid var(--line); font-size:13px; }
  table.cm th { color:var(--dim); font-weight:600; font-size:12px; }
  table.cm tr th:first-child { text-align:left; }
  table.cm td { font-variant-numeric:tabular-nums; }
  table.cm td.good { color:var(--accent); }
  table.cm td.bad { color:var(--bad); }
  /* the Results ledger: a summary row per baseline, its history under it */
  table.ledger { border-collapse:collapse; width:100%; }
  table.ledger th, table.ledger td { padding:7px 10px; text-align:right;
    border-bottom:1px solid var(--line); font-size:13px; white-space:nowrap;
    font-variant-numeric:tabular-nums; }
  table.ledger th { color:var(--dim); font-weight:600; font-size:12px; }
  table.ledger td:last-child { white-space:normal; }
  table.ledger th:first-child, table.ledger td:first-child { text-align:left; }
  table.ledger tr.sub td { font-size:12px; color:var(--dim);
    border-bottom:1px dashed var(--line); }
  table.ledger tr.sub td:first-child { padding-left:24px; }
  #benchview { margin-bottom:14px; }
  .hdrbtn { background:none; border:1px solid var(--line); color:var(--dim);
            border-radius:99px; padding:5px 13px; font:inherit; font-size:13px;
            cursor:pointer; }
  .hdrbtn:hover { color:var(--ink); border-color:var(--dim); }
  .hdrbtn.off:hover { color:var(--bad); border-color:var(--bad); }
  #version { font-family:var(--mono); font-size:12px; }
  /* the Ollama line gives way before the buttons do: it truncates, and its
     full text is in its tooltip */
  #ollama { flex:1 1 0; min-width:0; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; }
  header .admin { display:flex; align-items:center; gap:8px; flex:none;
                  margin-left:auto; }
  #adminveil { position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:50;
               display:flex; align-items:center; justify-content:center;
               padding:16px; }
  #adminbox { max-width:620px; width:100%; }
  #adminbox .log { max-height:260px; margin-top:10px; }
  .gain-up { color:var(--accent); }
  .gain-down { color:var(--bad); }
  .note { color:var(--dim); font-size:12.5px; }
  .note p { margin:0 0 10px; }
  .note p:last-child { margin-bottom:0; }
  .note code { font-family:var(--mono); font-size:12px; }
  #transcript { margin-bottom:12px; }
  .askrow { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  .askrow label.inline { font-weight:600; }
  .askrow select, .askrow input[type=text] { width:auto; padding:6px 9px; }
  /* has to out-rank the "select, input[type=text]" width:100% above it */
  input[type=text].num { width:74px; }
</style>
</head>
<body>
<header>
  <h1>scam-detection</h1>
  <nav class="pages" id="pages">
    <button data-page="bench" class="on">Benchmark</button>
    <button data-page="bert">BERT</button>
    <button data-page="bow">Bag of words</button>
    <button data-page="length">Length only</button>
    <button data-page="llm">LLM judge</button>
  </nav>
  <span class="sub" id="ollama">checking Ollama…</span>
  <span class="admin">
    <span class="sub" id="version" title=""></span>
    <button class="hdrbtn" id="updbtn" title="git pull the latest code and restart this server">Update</button>
    <button class="hdrbtn off" id="offbtn" title="stop this server - runs already going are not affected">Stop server</button>
  </span>
</header>

<div class="wrap" id="page-bench">
  <div class="side">
    <!-- First, because on a return visit the run you want to read is the
         reason the page is open. It used to be below the form and the
         knowledge base, which on a ten-baseline list meant scrolling. -->
    <div class="sect">
      <div class="secthead">Recent runs</div>
      <div class="hint" style="margin-top:0">pick one to read its output and
        results · hover a run to delete it</div>
      <div class="hist" id="hist"></div>
      <div class="row" style="margin-top:8px">
        <button class="link" id="clearhist" hidden>clear finished runs</button>
      </div>
      <div class="hint" id="histerr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">New run</div>
      <label for="dataset">Dataset</label>
      <select id="dataset"></select>

      <label class="inline" style="margin-top:12px">
        <input type="checkbox" id="stripped">
        <span><span class="name">Content-deletion test</span>
        <span class="note">also score every held-out call with its content
          words deleted</span></span>
      </label>
      <div class="hint" id="strippedhint">The folds do not change. A system
        that learns from the data is trained on the original text of the four
        training folds, then the held-out fold is scored twice by those same
        weights &mdash; as written, and with the content words deleted &mdash;
        and that rotates through all five. Nothing is ever trained on stripped
        text. The results gain a stripped-accuracy column and a trusted
        accuracy beside it. A word survives only if it is a determiner,
        pronoun, preposition, conjunction, auxiliary or negation &mdash; the
        closed class in <code>scripts/trusted.py</code>; everything else goes.
        The stripped copy is built from the dataset itself, so this works on
        any dataset in the list &mdash; there is no second file to make or to
        keep in step.</div>

      <label>Baselines</label>
      <div class="hint" style="margin-top:-2px">only the ticked ones run</div>
      <div class="checks" id="baselines"></div>
      <div class="row">
        <button class="link" id="pickall">select all</button>
        <button class="link" id="picknone">clear</button>
      </div>

      <label for="scope">How much to run</label>
      <select id="scope">
        <option value="count">A number of calls</option>
        <option value="idx">One transcript, by index</option>
        <option value="id">One transcript, by id</option>
      </select>

      <label for="limit" id="limitlabel">How many calls</label>
      <input type="text" id="limit" value="0" spellcheck="false">
      <div class="hint" id="limithint"></div>

      <div id="modelbox">
        <label for="model">Model</label>
        <select id="model"></select>
        <label for="numctx">Context window</label>
        <input type="text" id="numctx" placeholder="8192" spellcheck="false">
        <div class="hint" id="ctxhint">How many tokens the model may read.
          Ollama cuts an overlong prompt from the <em>front</em> —
          instructions first — and keeps only about half the window, so a call
          that does not fit comes back as a confident verdict on its last few
          minutes.</div>
      </div>

      <button class="go" id="go">Run</button>
      <div class="hint" id="formerr" style="color:var(--bad)"></div>
    </div>

    <!-- Updating the knowledge base is something you do occasionally, not
         part of setting up a run, so it no longer sits between the Run
         button and the history taking up room. Closed by default; it opens
         itself when the index is missing or out of step with the JSON. -->
    <div class="sect">
      <details class="kb" id="kbpanel">
        <summary>Web-RAG knowledge base
          <div class="hint" id="kbstate">checking…</div>
        </summary>
        <div class="kbbody">
          <label for="kbmode" style="margin-top:6px">What to do</label>
          <select id="kbmode"></select>
          <div class="hint" id="kbnote"></div>
          <button class="go" id="kbgo">Update knowledge base</button>
          <div class="hint" id="kberr" style="color:var(--bad)"></div>
        </div>
      </details>
    </div>
  </div>

  <div class="main">
    <div class="tabs" id="benchview">
      <button data-bv="run" class="on">Run</button>
      <button data-bv="ledger">Results</button>
    </div>

    <!-- every recorded result, built up a baseline at a time -->
    <div id="bench-ledger" hidden>
      <div class="card">
        <div class="row" style="gap:10px; flex-wrap:wrap">
          <label for="ledgerds" style="margin:0">Dataset</label>
          <select id="ledgerds" style="flex:1; min-width:220px"></select>
          <button class="link" id="ledgerrefresh">refresh</button>
        </div>
        <div class="hint" id="ledgersub">Every benchmark run that finishes is
          recorded here, one line per baseline, and kept even if the run is
          later deleted from Recent runs. Run baselines one at a time whenever
          suits you; the table fills in as you go.</div>
      </div>
      <div id="ledgerbody"></div>
    </div>

    <div id="bench-run">
    <div class="card" id="statuscard">
      <div class="row">
        <strong id="runtitle">No run selected</strong>
        <span class="pill" id="runpill" hidden></span>
        <span style="flex:1"></span>
        <button class="stop" id="stopbtn" hidden>Stop</button>
      </div>
      <div class="hint" id="runsub"></div>
    </div>

    <div class="tabs" id="tabs" hidden>
      <button data-tab="output">Output</button>
      <button data-tab="results">Table</button>
      <button data-tab="calls">Per-call <span class="count" id="c-calls"></span></button>
      <button data-tab="steps">Step logs <span class="count" id="c-steps"></span></button>
    </div>

    <!-- output -->
    <div id="t-output">
      <pre class="log" id="log">Pick a dataset and a baseline, then press Run.

The run is detached from this page: closing the browser, or losing the SSH
connection, does not stop it. Come back to this URL and it is still here.

Past runs are top left - selecting one brings back its output, its
results table, and the prediction it made for every single call.</pre>
    </div>

    <!-- results -->
    <div id="t-results" hidden>
      <div class="card">
        <div class="seg">
          <button id="v-table" class="on">Data</button>
          <button id="v-chart">Graph</button>
        </div>
        <div id="r-table"><div class="scroll"><table id="results"></table></div></div>
        <div id="r-chart" hidden>
          <div class="legend viz" id="chartkey"></div>
          <div class="scroll viz" id="chart"></div>
        </div>
        <div class="hint" id="spread"></div>
      </div>
    </div>

    <!-- per-call -->
    <div id="t-calls" hidden>
      <div class="filepick">
        <select id="csvpick"></select>
        <label class="inline" id="whybox" hidden>
          <input type="checkbox" id="showwhy" checked> show reasons
        </label>
        <span class="hint" id="csvinfo"></span>
      </div>
      <div class="scroll"><table class="calls" id="calls"></table></div>
      <div class="pager">
        <button id="prev">‹ Previous</button>
        <button id="next">Next ›</button>
        <span class="hint" id="pageinfo"></span>
      </div>
    </div>

    <!-- step logs -->
    <div id="t-steps" hidden>
      <div class="filepick">
        <select id="steppick"></select>
        <span class="hint" id="stepinfo"></span>
      </div>
      <pre class="log" id="steplog"></pre>
    </div>
    </div>
  </div>
</div>

<!-- ==================== page two: BERT ==================== -->
<div class="wrap" id="page-bert" hidden>
  <div class="side">
    <div class="sect">
      <div class="secthead">Trained models</div>
      <div class="hint" style="margin-top:0">the checkpoint a call is put to ·
        hover to delete one</div>
      <div class="hist" id="bertmodels"></div>
      <div class="row" style="margin-top:8px">
        <span class="hint" id="workerstate" style="flex:1"></span>
        <button class="link" id="unload" hidden>unload it</button>
      </div>
      <div class="hint" id="modelerr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Train a model</div>

      <label for="tname">Name</label>
      <input type="text" id="tname" placeholder="e.g. zhi-bert" spellcheck="false">
      <div class="hint">saved as models/&lt;name&gt;/ — a few hundred MB, and
        gitignored</div>

      <label for="tdataset">Dataset</label>
      <select id="tdataset"></select>

      <label for="tbase">Start from</label>
      <select id="tbase"></select>
      <div class="hint" id="tbasenote"></div>

      <div class="grid2">
        <div><label for="tepochs">Epochs</label>
             <input type="text" id="tepochs" class="num" style="width:100%"></div>
        <div><label for="tbatch">Batch size</label>
             <input type="text" id="tbatch" class="num" style="width:100%"></div>
        <div><label for="tmaxlen">Tokens per call</label>
             <input type="text" id="tmaxlen" class="num" style="width:100%"></div>
        <div><label for="tlr">Learning rate</label>
             <input type="text" id="tlr" class="num" style="width:100%"></div>
        <div><label for="tseed">Seed</label>
             <input type="text" id="tseed" class="num" style="width:100%"></div>
        <div><label for="tlimit">Calls (0 = all)</label>
             <input type="text" id="tlimit" class="num" style="width:100%"></div>
      </div>
      <label for="tholdout">Held back to score it</label>
      <input type="text" id="tholdout" class="num">
      <div class="hint">a fraction, e.g. 0.2 — stratified, and only to put a
        number on the checkpoint. The benchmark's BERT figure is the k-fold one
        on the other page.</div>

      <div class="checks" style="margin-top:14px">
        <label><input type="checkbox" id="tcpu">
          <span><span class="name">Train on the CPU</span>
          <span class="note">much slower; leave off unless the GPU is busy</span></span></label>
        <label><input type="checkbox" id="toverwrite">
          <span><span class="name">Replace a model of that name</span>
          <span class="note">the old checkpoint is overwritten, not kept</span></span></label>
      </div>

      <button class="go" id="traingo">Train</button>
      <div class="hint" id="trainerr" style="color:var(--bad)"></div>
    </div>
  </div>

  <div class="main">
    <div class="card">
      <div class="row">
        <strong id="berttitle">No model selected</strong>
        <span class="pill" id="bertpill" hidden></span>
        <span style="flex:1"></span>
        <button class="stop" id="trainstop" hidden>Stop</button>
      </div>
      <div class="hint" id="bertsub">Train one on the left, then put a
        transcript to it.</div>
    </div>

    <div class="tabs" id="berttabs">
      <button data-mtab="ask" class="on">Classify</button>
      <button data-mtab="eval">Score a dataset</button>
      <button data-mtab="train">Training output</button>
      <button data-mtab="about">How it decides</button>
    </div>

    <!-- ask -->
    <div id="m-ask">
      <div class="card">
        <div class="filepick">
          <select id="sampleds"></select>
          <input type="text" id="sampleidx" class="num" value="0" spellcheck="false">
          <button class="link" id="sampleload">load that row</button>
          <span class="hint" id="sampleinfo"></span>
        </div>
        <textarea id="transcript" placeholder="Paste a call transcript here, or load one from a dataset above."></textarea>
        <div class="askrow">
          <label class="inline" for="aggregate">Combine windows by
            <select id="aggregate">
              <option value="max">the strongest stretch</option>
              <option value="mean">the average</option>
            </select></label>
          <label class="inline" for="threshold">Scam at
            <input type="text" id="threshold" class="num" placeholder="0.5"></label>
          <label class="inline"><input type="checkbox" id="agpu" checked>
            score on the GPU</label>
        </div>
        <details class="adv">
          <summary>How the call is read</summary>
          <div class="advbody">
            <div class="grid2">
              <div><label for="awindow">Window (words)</label>
                   <input type="text" id="awindow" style="width:100%"></div>
              <div><label for="astride">Stride (words)</label>
                   <input type="text" id="astride" style="width:100%"></div>
              <div><label for="amaxlen">Tokens per window</label>
                   <input type="text" id="amaxlen" style="width:100%"></div>
            </div>
            <div class="hint">BERT reads 512 tokens at most and these
              checkpoints are trained at 256 — about 180 words. A call longer
              than that is cut into overlapping windows of the size training
              used, each is scored, and the call takes the strongest (or the
              average). Handing it the whole transcript instead would score
              the first two minutes and ignore the rest. Changing any of these
              reloads the model, so the next answer is slower.</div>
            <div style="margin-top:11px">
              <label class="inline"><input type="checkbox" id="astrip">
                strip the tone tags first</label>
              <div class="hint">Off by default. Takes out
                <code>[curious]</code>, <code>[long pause]</code> and the rest
                — nobody said them out loud, but training read them, so this
                asks the model about text of a kind it never saw. Worth one
                run both ways: in
                <code>scamai_full_1000.csv</code>, <code>[satisfied]</code>
                sits on 54.8% of legitimate calls and 31.0% of scams, so a
                score that moves a lot here was partly reading the annotation
                style rather than the call.</div>
            </div>
          </div>
        </details>
        <button class="go" id="askgo">Classify this call</button>
        <div class="hint" id="askerr" style="color:var(--bad)"></div>
      </div>
      <div id="answer"></div>
    </div>

    <!-- training output -->

    <div id="m-eval" hidden>
      <div class="card">
        <div class="filepick">
          <select id="berteds"></select>
          <input type="text" id="bertevlimit" class="num" placeholder="all"
                 spellcheck="false" title="calls to score, 0 or blank for all">
          <span class="hint">calls (blank = all, a balanced head otherwise)</span>
        </div>
        <div class="askrow">
          <label class="inline" for="bertevthr">Scam at
            <input type="text" id="bertevthr" class="num" placeholder="model's own"></label>
          <label class="inline"><input type="checkbox" id="bertevstrip">
            strip tone tags</label>
          <label class="inline"><input type="checkbox" id="bertevgpu" checked>
            use the GPU</label></div>
        <button class="go" id="bertevgo">Score every call</button>
        <div class="hint" id="berteverr" style="color:var(--bad)"></div>
        <div class="hint">Every call is scored by the one model selected on the left, so pointing it at the dataset it was fitted on measures memory rather than skill — the card says so when the two match. The Benchmark page cross-validates instead, which is the number to quote. A long call is scored in windows, as on the Classify tab.</div>
      </div>
      <div id="bertevout"></div>
      <pre class="log" id="bertevlog" hidden></pre>
    </div>

    <div id="m-train" hidden>
      <pre class="log" id="trainlog">No training run selected.

Fill in the form on the left and press Train. The run is detached from this
page exactly like a benchmark run, so it survives closing the browser, and it
shows up in Recent runs on the Benchmark page too.</pre>
    </div>

    <!-- about -->
    <div id="m-about" hidden>
      <div class="card note">
        <p><strong>One number, and it is the one training fits.</strong> The
        checkpoint is a binary classifier: it was shown labelled calls and
        fitted to separate scam from legitimate. <code>prob_scam</code> is
        that head speaking. Nothing else on this page is inferred, weighted or
        scored — the model was trained to produce this and only this.</p>

        <p><strong>Why the call is cut into windows.</strong> BERT reads 512
        tokens at most, and these checkpoints are trained at 256 — roughly 180
        words. Some calls in <code>datasets/</code> run past ten thousand.
        Handing the whole transcript to the tokenizer scores its opening and
        silently drops the rest, which on a long call means judging it by the
        hellos. So the transcript is cut into overlapping windows the size
        training used, every window is scored, and the call takes either the
        strongest window or the average of them.</p>

        <p><strong>Strongest or average.</strong> <em>Strongest</em> says a
        scam signal anywhere is a scam signal, which suits calls that are
        mostly small talk around one telling exchange. <em>Average</em> is
        steadier but dilutes that exchange in a long friendly call. Both are
        shown whichever you pick, along with <strong>first window</strong> —
        what a single truncated read would have said. When those three
        disagree, the disagreement is the finding.</p>

        <p><strong>What the holdout number is not.</strong> The accuracy
        beside a trained model is a stratified slice of its own training file,
        kept back. It says the checkpoint learnt something; it does not say
        the something is scam detection. On
        <code>scambait_bank_422.csv</code> a checkpoint reaches 100% — the
        scam side is YouTube scam-baiting and the legitimate side is the
        HarperValleyBank corpus, so the two are separable on recording
        pipeline alone. Compare against the bag-of-words baseline on the
        Benchmark page before believing any of it.</p>
      </div>
    </div>
  </div>
</div>

<!-- ==================== page three: Bag of words ==================== -->
<div class="wrap" id="page-bow" hidden>
  <div class="side">
    <div class="sect">
      <div class="secthead">Fitted models</div>
      <div class="hint" style="margin-top:0">the model a call is put to ·
        hover to delete one</div>
      <div class="hist" id="bowmodels"></div>
      <div class="row" style="margin-top:8px">
        <span class="hint" id="bowworkerstate" style="flex:1"></span>
        <button class="link" id="bowunload" hidden>unload it</button>
      </div>
      <div class="hint" id="bowmodelerr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Fit a model</div>

      <label for="bowname">Name</label>
      <input type="text" id="bowname" placeholder="e.g. bank-bow" spellcheck="false">
      <div class="hint">saved as models/&lt;name&gt;/ — a few hundred KB, and
        gitignored. It sits beside the BERT checkpoints without colliding:
        they are told apart by what is in the directory.</div>

      <label for="bowdataset">Dataset</label>
      <select id="bowdataset"></select>

      <div class="grid2">
        <div><label for="bowngram">N-grams up to</label>
             <input type="text" id="bowngram" style="width:100%"></div>
        <div><label for="bowmindf">Least calls per term</label>
             <input type="text" id="bowmindf" style="width:100%"></div>
        <div><label for="bowholdout">Held back to score it</label>
             <input type="text" id="bowholdout" style="width:100%"></div>
        <div><label for="bowseed">Seed</label>
             <input type="text" id="bowseed" style="width:100%"></div>
      </div>
      <div class="hint">2 and 2 are what the <code>bow</code> baseline on the
        Benchmark page cross-validates with, so leave them there if you want
        the two numbers to be about the same model.</div>

      <label class="inline" style="margin-top:14px">
        <input type="checkbox" id="bowstriptrain">
        <span><span class="name">Strip tone tags before fitting</span>
        <span class="note">drops [curious], [long pause] and the rest</span></span>
      </label>
      <label class="inline">
        <input type="checkbox" id="bowoverwrite">
        <span><span class="name">Replace a model of that name</span>
        <span class="note">the old one is overwritten, not kept</span></span>
      </label>

      <button class="go" id="bowtrain">Fit</button>
      <div class="hint" id="bowtrainerr" style="color:var(--bad)"></div>
    </div>
  </div>

  <div class="main">
    <div class="card">
      <div class="row">
        <strong id="bowtitle">No model selected</strong>
        <span style="flex:1"></span>
      </div>
      <div class="hint" id="bowsub">Fit one on the left — it takes about a
        second — then put a transcript to it.</div>
    </div>

    <div class="tabs" id="bowtabs">
      <button data-btab="ask" class="on">Classify</button>
      <button data-btab="eval">Score a dataset</button>
      <button data-btab="train">Fitting output</button>
      <button data-btab="about">How it decides</button>
    </div>

    <div id="b-ask">
      <div class="card">
        <div class="filepick">
          <select id="bowds"></select>
          <input type="text" id="bowidx" class="num" value="0" spellcheck="false">
          <button class="link" id="bowload">load that row</button>
          <span class="hint" id="bowinfo"></span>
        </div>
        <textarea id="bowtranscript" placeholder="Paste a call transcript here, or load one from a dataset above."></textarea>
        <div class="askrow">
          <label class="inline" for="bowthreshold">Scam at
            <input type="text" id="bowthreshold" class="num" placeholder="0.5"></label>
          <label class="inline" for="bowtop">Terms to show
            <input type="text" id="bowtop" class="num" placeholder="12"></label>
          <label class="inline"><input type="checkbox" id="bowstrip">
            strip tone tags</label>
        </div>
        <button class="go" id="bowgo">Classify this call</button>
        <div class="hint" id="bowaskerr" style="color:var(--bad)"></div>
      </div>
      <div id="bowanswer"></div>
    </div>


    <div id="b-eval" hidden>
      <div class="card">
        <div class="filepick">
          <select id="boweds"></select>
          <input type="text" id="bowevlimit" class="num" placeholder="all"
                 spellcheck="false" title="calls to score, 0 or blank for all">
          <span class="hint">calls (blank = all, a balanced head otherwise)</span>
        </div>
        <div class="askrow">
          <label class="inline" for="bowevthr">Scam at
            <input type="text" id="bowevthr" class="num" placeholder="model's own"></label>
          <label class="inline"><input type="checkbox" id="bowevstrip">
            strip tone tags</label></div>
        <button class="go" id="bowevgo">Score every call</button>
        <div class="hint" id="boweverr" style="color:var(--bad)"></div>
        <div class="hint">Every call is scored by the one model selected on the left, so pointing it at the dataset it was fitted on measures memory rather than skill — the card says so when the two match. The Benchmark page cross-validates instead, which is the number to quote.</div>
      </div>
      <div id="bowevout"></div>
      <pre class="log" id="bowevlog" hidden></pre>
    </div>

    <div id="b-train" hidden>
      <pre class="log" id="bowtrainlog">No fitting run selected.

Fill in the form on the left and press Fit. It takes about a second, and the
run appears under Recent runs on the Benchmark page like any other.</pre>
    </div>

    <div id="b-about" hidden>
      <div class="card note">
        <p><strong>What it is.</strong> TF-IDF over word unigrams and bigrams
        into a logistic regression — the same vectoriser and classifier the
        <code>bow</code> baseline on the Benchmark page cross-validates. No
        embeddings, no attention, no GPU. It fits in about a second.</p>

        <p><strong>It reads the whole call.</strong> BERT takes 512 tokens, so
        that page cuts a long transcript into windows. TF-IDF has no length
        limit: every word is counted. So on a long call the two pages are not
        being asked the same question, and this is the one that saw all of
        it.</p>

        <p><strong>Why it can be read back exactly.</strong> A linear model
        over TF-IDF decomposes: the score is the intercept plus, for every
        term in the call, its TF-IDF weight times its coefficient. The terms
        listed under a verdict are those products, largest first, and they sum
        to the score shown. This is arithmetic, not a story told about the
        model afterwards — and it is the thing BERT cannot give you.</p>

        <p><strong>Read the terms, not the accuracy.</strong> If this scores
        near a fine-tuned BERT on a dataset, that dataset is separable on
        vocabulary and neither number is evidence about understanding scams.
        On <code>scambait_bank_422.csv</code> it reaches 100% held-out in a
        tenth of a second — the same as BERT — and the terms doing the work
        are <code>so</code>, <code>me</code>, <code>yes</code>,
        <code>to</code>, <code>is</code>, <code>what</code>. Those are
        function words: the two halves of that dataset come from different
        recording pipelines, and this is what separating on transcription
        style looks like from the inside.</p>
      </div>
    </div>
  </div>
</div>

<!-- ==================== page four: Length only ==================== -->
<div class="wrap" id="page-length" hidden>
  <div class="side">
    <div class="sect">
      <div class="secthead">Fitted thresholds</div>
      <div class="hint" style="margin-top:0">the rule a call is put to &middot;
        hover to delete one</div>
      <div class="hist" id="lenmodels"></div>
      <div class="row" style="margin-top:8px">
        <span class="hint" id="lenworkerstate" style="flex:1"></span>
        <button class="link" id="lenunload" hidden>unload it</button>
      </div>
      <div class="hint" id="lenmodelerr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Fit a threshold</div>

      <label for="lenname">Name</label>
      <input type="text" id="lenname" placeholder="e.g. bank-length" spellcheck="false">
      <div class="hint">saved as models/&lt;name&gt;/ &mdash; a few kilobytes of
        JSON. It sits beside the BERT checkpoints and the bag-of-words models
        without colliding: the three are told apart by what is in the
        directory.</div>

      <label for="lendataset">Dataset</label>
      <select id="lendataset"></select>

      <div class="grid2">
        <div><label for="lenmetric">Sweep maximises</label>
             <select id="lenmetric" style="width:100%">
               <option value="f1">F1</option>
               <option value="acc">accuracy</option>
             </select></div>
        <div><label for="lenholdout">Held back to score it</label>
             <input type="text" id="lenholdout" style="width:100%"></div>
        <div><label for="lenseed">Seed</label>
             <input type="text" id="lenseed" style="width:100%"></div>
        <div><label for="lenlimit">Calls (0 = all)</label>
             <input type="text" id="lenlimit" style="width:100%"></div>
      </div>
      <div class="hint">The sweep tries every threshold the fitting calls
        suggest, in both directions, and keeps the best. That is one number
        fitted to one dataset &mdash; it overfits happily, which is why the
        output lists the runner-up thresholds and says when the curve is
        flat.</div>

      <label class="inline" style="margin-top:14px">
        <input type="checkbox" id="lenpin">
        <span><span class="name">Pin the threshold instead of sweeping</span>
        <span class="note">no fitting at all &mdash; use the benchmark's own
          rule</span></span>
      </label>
      <div class="grid2" id="lenpinrow" hidden>
        <div><label for="lenthreshold">Words</label>
             <input type="text" id="lenthreshold" style="width:100%"></div>
        <div><label for="lendirection">Scam is the</label>
             <select id="lendirection" style="width:100%">
               <option value="longer">longer side</option>
               <option value="shorter">shorter side</option>
             </select></div>
      </div>

      <label class="inline">
        <input type="checkbox" id="lenstriptrain">
        <span><span class="name">Strip tone tags before counting</span>
        <span class="note">drops [curious], [long pause] and the rest</span></span>
      </label>
      <label class="inline">
        <input type="checkbox" id="lenoverwrite">
        <span><span class="name">Replace a model of that name</span>
        <span class="note">the old one is overwritten, not kept</span></span>
      </label>

      <button class="go" id="lenfit">Fit</button>
      <div class="hint" id="lenfiterr" style="color:var(--bad)"></div>
    </div>
  </div>

  <div class="main">
    <div class="card">
      <div class="row">
        <strong id="lentitle">No model selected</strong>
        <span style="flex:1"></span>
      </div>
      <div class="hint" id="lensub">Fit one on the left &mdash; it is one sort
        of the dataset &mdash; then put a transcript to it.</div>
    </div>

    <div class="tabs" id="lentabs">
      <button data-ltab="ask" class="on">Classify</button>
      <button data-ltab="eval">Score a dataset</button>
      <button data-ltab="train">Fitting output</button>
      <button data-ltab="about">How it decides</button>
    </div>

    <div id="l-ask">
      <div class="card">
        <div class="filepick">
          <select id="lends"></select>
          <input type="text" id="lenidx" class="num" value="0" spellcheck="false">
          <button class="link" id="lenload">load that row</button>
          <span class="hint" id="leninfo"></span>
        </div>
        <textarea id="lentranscript" placeholder="Paste a call transcript here, or load one from a dataset above."></textarea>
        <div class="askrow">
          <label class="inline" for="lenaskthreshold">Scam past
            <input type="text" id="lenaskthreshold" class="num" placeholder="fitted"></label>
          <label class="inline" for="lenaskdirection">on the
            <select id="lenaskdirection" style="width:auto">
              <option value="">fitted side</option>
              <option value="longer">longer side</option>
              <option value="shorter">shorter side</option>
            </select></label>
          <label class="inline"><input type="checkbox" id="lenstrip">
            strip tone tags</label>
        </div>
        <button class="go" id="lengo">Classify this call</button>
        <div class="hint" id="lenaskerr" style="color:var(--bad)"></div>
      </div>
      <div id="lenanswer"></div>
    </div>


    <div id="l-eval" hidden>
      <div class="card">
        <div class="filepick">
          <select id="leneds"></select>
          <input type="text" id="lenevlimit" class="num" placeholder="all"
                 spellcheck="false" title="calls to score, 0 or blank for all">
          <span class="hint">calls (blank = all, a balanced head otherwise)</span>
        </div>
        <div class="askrow">
          <label class="inline" for="lenevthr">Scam past
            <input type="text" id="lenevthr" class="num" placeholder="fitted"></label>
          <label class="inline" for="lenevdir">on the
            <select id="lenevdir" style="width:auto">
              <option value="">fitted side</option>
              <option value="longer">longer side</option>
              <option value="shorter">shorter side</option>
            </select></label>
          <label class="inline"><input type="checkbox" id="lenevstrip">
            strip tone tags</label></div>
        <button class="go" id="lenevgo">Score every call</button>
        <div class="hint" id="leneverr" style="color:var(--bad)"></div>
        <div class="hint">Every call is scored by the one model selected on the left, so pointing it at the dataset it was fitted on measures memory rather than skill — the card says so when the two match. The Benchmark page cross-validates instead, which is the number to quote. The card also reports what the same threshold pointing the other way would have scored.</div>
      </div>
      <div id="lenevout"></div>
      <pre class="log" id="lenevlog" hidden></pre>
    </div>

    <div id="l-train" hidden>
      <pre class="log" id="lentrainlog">No fitting run selected.

Fill in the form on the left and press Fit. It is one sort of the dataset, and
the run appears under Recent runs on the Benchmark page like any other.</pre>
    </div>

    <div id="l-about" hidden>
      <div class="card note">
        <p><strong>What it is.</strong> The number of words in the transcript,
        compared to one other number. That is the whole model. It is the
        <code>length</code> baseline on the Benchmark page, which is
        <code>trivial_length</code> in <code>combined_evaluate.py</code>:
        <em>Fraud if the call is longer than 45 words</em>.</p>

        <p><strong>Why it has a page.</strong> Not because the accuracy is
        interesting &mdash; because it is the floor. Nothing in the call is
        read: not a word, not an entity, not a tone tag. Whatever BERT or the
        bag of words beats this by is the whole of what those models are
        worth on that dataset, and on several of the datasets here the gap is
        smaller than the write-up would like.</p>

        <p><strong>&ldquo;Fit&rdquo; is a sweep, not learning.</strong> There
        is one parameter and it is chosen by trying every threshold the
        fitting calls suggest and keeping the best-scoring one. That will
        overfit a single number to a single dataset without complaint, so the
        fitting output lists the runner-up thresholds and says out loud when
        the curve is flat &mdash; when a couple of hundred thresholds come
        within a point of the winner, the exact number means nothing. Tick
        <em>pin the threshold</em> to skip the fitting entirely and use the
        benchmark's own 45.</p>

        <p><strong>The direction is not a given.</strong> &ldquo;Scams are
        longer&rdquo; is an assumption about a corpus, not a fact about
        scams, and the sweep tests both ways round. On
        <code>scambait_bank_422.csv</code> and
        <code>zhi_english_646.csv</code> the scam calls are the longer ones.
        On <code>scamai_full_1000.csv</code> and
        <code>everything_7013.csv</code> they are the <em>shorter</em> ones
        &mdash; so <code>trivial_length</code>'s rule is pointing the wrong
        way on those two, and the number it reports there is worse than the
        same threshold read backwards.</p>

        <p><strong>There is no probability here, so none is invented.</strong>
        A threshold cannot say how confident it is. What is shown in place of
        one is a fact about the fitting set: the share of fitting calls on
        this side of the line that really were scams. When that share
        contradicts the verdict &mdash; the rule calls a call a scam, but most
        fitting calls on that side were not &mdash; the page says so rather
        than dressing the number up.</p>

        <p><strong>Read this next to the other pages.</strong> On
        <code>scambait_bank_422.csv</code> a fitted threshold reaches about
        72% held out. BERT and the bag of words both reach 100% on the same
        split. The distance between 72% and 100% is what reading the words
        bought; the distance between 50% and 72% is what counting them
        bought, from a model that is one integer.</p>
      </div>
    </div>
  </div>
</div>

<div class="wrap" id="page-llm" hidden>
  <div class="side">
    <div class="sect">
      <div class="secthead">Model</div>
      <div class="hint" style="margin-top:0">what ollama has pulled on this
        machine</div>
      <select id="llmmodel" style="width:100%; margin-top:8px"></select>
      <div class="hint" id="llmhost"></div>
      <div class="hint" id="llmerr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Fitted prompts</div>
      <div class="hint" style="margin-top:0">what goes in the prompt before
        the call &middot; hover to delete one</div>
      <div class="hist" id="llmprofiles"></div>
      <div class="hint" id="llmproferr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Fit a prompt</div>
      <div class="hint" style="margin-top:0"><strong>This does not fine-tune
        anything.</strong> The weights Ollama is holding do not move and
        cannot be moved from here. What is fitted is the prompt: worked
        examples out of a dataset, a rubric the model writes from them, and
        your standing instructions made durable. That is in-context learning,
        and it is the only kind of training a frozen local model can be
        given.</div>

      <label for="llmfitname">Name</label>
      <input type="text" id="llmfitname" placeholder="e.g. zhi-prompt" spellcheck="false">

      <label for="llmfitds">Dataset</label>
      <select id="llmfitds"></select>

      <div class="grid2">
        <div><label for="llmshots">Worked examples</label>
             <input type="text" id="llmshots" style="width:100%"></div>
        <div><label for="llmshotwords">Words from each</label>
             <input type="text" id="llmshotwords" style="width:100%"></div>
        <div><label for="llmholdcalls">Calls to score on</label>
             <input type="text" id="llmholdcalls" style="width:100%"></div>
        <div><label for="llmfitseed">Seed</label>
             <input type="text" id="llmfitseed" style="width:100%"></div>
      </div>

      <label class="inline" style="margin-top:14px">
        <input type="checkbox" id="llmrubric" checked>
        <span><span class="name">Have the model write the rubric</span>
        <span class="note">it reads the examples and writes the rules, which
          then ride in every prompt</span></span>
      </label>
      <label class="inline">
        <input type="checkbox" id="llmfitguide" checked>
        <span><span class="name">Carry the standing instructions in</span>
        <span class="note">whatever is in the box on the right, made durable
          instead of lost on reload</span></span>
      </label>
      <label class="inline">
        <input type="checkbox" id="llmfitover">
        <span><span class="name">Replace a prompt of that name</span>
        <span class="note">the old one is overwritten, not kept</span></span>
      </label>

      <div class="hint" id="llmfitcost"></div>
      <button class="go" id="llmfitgo">Fit</button>
      <div class="hint" id="llmfiterr" style="color:var(--bad)"></div>
    </div>

    <div class="sect">
      <div class="secthead">Settings</div>
      <div class="grid2">
        <div><label for="llmmaxtok">Reply tokens</label>
             <input type="text" id="llmmaxtok" style="width:100%"></div>
        <div><label for="llmctx">Context window</label>
             <input type="text" id="llmctx" style="width:100%"></div>
        <div><label for="llmtemp">Temperature</label>
             <input type="text" id="llmtemp" style="width:100%"></div>
      </div>
      <div class="hint">The context window is the one worth watching. Ollama
        drops the <em>front</em> of a prompt that overflows it — the
        instructions first — and what comes back then reads like a bad model
        rather than a bad setting. The page warns you when a transcript is
        close to the edge.</div>
    </div>

    <div class="sect">
      <div class="secthead">What this is</div>
      <div class="hint" style="margin-top:0">The <code>llm_only</code> control
        from the Benchmark page, asked one call at a time. No retrieval, no
        ontology, no fine-tuning — just the transcript and the question. It
        uses the same prompt and the same verdict parser as the benchmark, so
        the answer here is the answer that would have been recorded there.</div>
    </div>
  </div>

  <div class="main">
    <div class="card">
      <div class="row">
        <strong id="llmtitle">Bare llm_only control</strong>
        <span style="flex:1"></span>
      </div>
      <div class="hint" id="llmsub">No fitted prompt — the transcript and the
        question, exactly as the benchmark asks it.</div>
    </div>

    <div class="tabs" id="llmtabs">
      <button data-jtab="ask" class="on">Ask</button>
      <button data-jtab="eval">Score a dataset</button>
      <button data-jtab="fit">Fitting output</button>
      <button data-jtab="about">How it decides</button>
    </div>

    <div id="j-ask">
    <div class="card">
      <div class="filepick">
        <select id="llmds"></select>
        <input type="text" id="llmidx" class="num" value="0" spellcheck="false">
        <button class="link" id="llmload">load that row</button>
        <span class="hint" id="llminfo"></span>
      </div>
      <textarea id="llmtranscript" placeholder="Paste a call transcript here, or load one from a dataset above."></textarea>
      <div class="askrow">
        <span class="hint" id="llmsize"></span>
      </div>

      <label for="llmguidance" style="margin-top:14px">Standing instructions</label>
      <div class="hint" style="margin-top:0">When the model gets one wrong,
        write the correction here and ask again — it is sent with every
        question from now on, fenced off from the transcript so the model
        reads it as a rule rather than as something the caller said. It is
        <strong>not</strong> training: nothing is stored and nothing is
        learned, so this box is the whole of the model's memory and closing
        the page empties it.</div>
      <textarea id="llmguidance" style="min-height:90px" placeholder="e.g. A bank asking the customer to confirm the last four digits of a card is normal here — only treat a full card number, PIN or one-time passcode as a scam signal."></textarea>
      <div class="askrow">
        <span class="hint" id="llmguidesize"></span>
        <span style="flex:1"></span>
        <button class="link" id="llmguideclear">clear them</button>
      </div>

      <button class="go" id="llmgo">Ask the model</button>
      <div class="hint" id="llmasker" style="color:var(--bad)"></div>
    </div>
    <div id="llmanswer"></div>
    </div>


    <div id="j-eval" hidden>
      <div class="card">
        <div class="filepick">
          <select id="llmeds"></select>
          <input type="text" id="llmevlimit" class="num" placeholder="all"
                 spellcheck="false" title="calls to score, 0 or blank for all">
          <span class="hint">calls (blank = all, a balanced head otherwise)</span>
        </div>
        <div class="askrow">
          <span class="hint" id="llmevcost" style="flex:1"></span></div>
        <button class="go" id="llmevgo">Score every call</button>
        <div class="hint" id="llmeverr" style="color:var(--bad)"></div>
        <div class="hint">One generation per call, so this is the slow one: a thousand-call dataset is hours on a 14B model. Set a limit. It scores under whichever fitted prompt is picked on the left, or bare with <em>None</em> — running it both ways on the same calls is how you find out whether the fitting helped on more than its own holdout.</div>
      </div>
      <div id="llmevout"></div>
      <pre class="log" id="llmevlog" hidden></pre>
    </div>

    <div id="j-fit" hidden>
      <pre class="log" id="llmfitlog">No fitting run selected.

Fill in "Fit a prompt" on the left and press Fit. It scores the held-out calls
twice - once with the fitted prompt and once with the bare one - so what comes
back is what the fitting bought, not just an accuracy. That is two LLM calls
per held-out call, so it takes minutes, not seconds.</pre>
    </div>

    <div id="j-about" hidden>
      <div class="card note">
        <p><strong>What this is.</strong> The <code>llm_only</code> control
        from the Benchmark page, asked one call at a time. No retrieval, no
        ontology. With no fitted prompt selected it uses the same prompt and
        the same verdict parser as the benchmark, so the answer here is the
        answer that would have been recorded there.</p>

        <p><strong>Fitting a prompt is not fine-tuning.</strong> The weights
        Ollama is holding do not move, and nothing in this project can move
        them. A fitted prompt is text that gets prepended to every question:
        worked examples drawn from a dataset, a rubric the model wrote after
        reading them, and your standing instructions. The other three pages
        train a model; this one writes a better question. They are not the
        same thing and the page will not call them the same thing.</p>

        <p><strong>Why the fit scores everything twice.</strong> A longer
        prompt always <em>feels</em> like an improvement, and often is not. So
        fitting runs the held-out calls through the fitted prompt and through
        the bare one, in the same order, and reports the difference. If the
        fitted prompt did not beat the control it has cost you context window
        and bought nothing — and the run says so in those words rather than
        quietly reporting a number that looks fine on its own.</p>

        <p><strong>Where the parts sit in the prompt, and why it matters.</strong>
        Ollama truncates an overlong prompt from the <em>front</em>. So the
        order is worst-to-best: worked examples, then the rubric, then the
        transcript, then the rules and the answer format. A fitted prompt that
        overflows loses its examples first and decays into the control, rather
        than into a headless wall of transcript with no question attached. The
        fitting run counts how many held-out prompts this happened to.</p>

        <p><strong>Standing instructions.</strong> The box under the
        transcript is sent with every question, fenced off so the model reads
        it as a rule rather than as something the caller said. On its own it
        is not stored — closing the page empties it. Tick <em>carry the
        standing instructions in</em> when fitting and they become part of the
        saved prompt instead.</p>

        <p><strong>A verdict under either is not the control.</strong> The
        moment a fitted prompt or standing instructions are in play, the
        answer stops being comparable with an <code>llm_only</code> row on the
        Benchmark page. The answer card says which applied.</p>
      </div>
    </div>
  </div>
</div>

<div id="adminveil" hidden>
  <div class="card" id="adminbox">
    <strong id="admintitle"></strong>
    <pre class="log" id="adminlog" hidden></pre>
    <div class="hint" id="adminnote"></div>
    <button class="link" id="adminclose" hidden>close</button>
  </div>
</div>

<script>
let CFG = null, current = null, offset = 0, timer = null;
let ART = {steps: [], csvs: []}, tab = 'output';
// id -> meta for every run the page knows about, so the panes can tell a
// benchmark from a knowledge-base update before the first poll comes back
let RUNS = {};
let page = 0, PAGE_SIZE = 50;

const $ = id => document.getElementById(id);

// A run is over when the output endpoint stops calling it running. The field is
// `status`; there is no `done` key, and a poller that waits for one polls
// until the tab is closed while its pane sits on "scoring…" forever.
const runOver = r => !!(r.status && r.status !== 'running');

// poller name -> how many polls in a row have failed, so a dropped request
// does not end a watch that is still worth keeping
const POLL_FAILS = {};

// Every poller reads a run's output through here, and it asks for
// /api/output rather than /api/log on purpose.
//
// The two are the same endpoint. But ad blockers and privacy extensions ship
// rules against paths that look like telemetry, and "/api/log" looks exactly
// like one: uBlock and Edge tracking prevention both refuse it, the fetch
// fails with a bare "Failed to fetch", and the page is left polling something
// that will never answer. That is not hypothetical - it is what happened on
// the Bag of words page, where a blocked poll meant a finished run reported
// no score at all.
//
// Do not "tidy" this back to /api/log.
const runLog = (id, offset) =>
  api(`/api/output?id=${encodeURIComponent(id)}&offset=${offset || 0}`);

// Always resolves to an object. A fetch that fails, or a reply that is not
// JSON - a proxy's error page, say - used to reject and take the whole poll
// down with it, leaving a blank panel and no clue why.
async function api(path, body) {
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                      body: JSON.stringify(body)} : {};
  let r;
  try {
    r = await fetch(path, opt);
  } catch (e) {
    // The server being down and an extension refusing the request look the
    // same from here, and the second is far more likely for a page that was
    // loading a moment ago.
    return {error: 'could not fetch ' + path + ' (' + e.message + ') - if the ' +
                   'rest of the page works, an ad blocker or Edge tracking ' +
                   'prevention is probably blocking this request'};
  }
  const text = await r.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return {error: 'HTTP ' + r.status + ' from ' + path + ' - ' +
                   (text.slice(0, 200) || 'empty reply')};
  }
}

// Put a problem where it can be seen, rather than returning quietly. A poll
// that keeps failing would otherwise paper the log with the same line every
// tick, so an immediate repeat is dropped.
let lastFail = '';
function fail(msg) {
  $('runsub').textContent = msg;
  if (msg === lastFail) return;
  lastFail = msg;
  const log = $('log');
  const span = document.createElement('span');
  span.className = 'l-fail';
  span.textContent = msg + '\n';
  log.appendChild(span);
  log.scrollTop = log.scrollHeight;
}

// /api/runs as an array, or null if the call did not come back with one.
// api() never rejects, so a dropped fetch or a proxy's error page arrives as
// {error} - and reading .runs off that gave "Cannot read properties of
// undefined (reading 'find')", a TypeError naming neither the request nor the
// reason, which took down whatever the caller was in the middle of.
async function runList() {
  const r = await api('/api/runs');
  if (r.error || !Array.isArray(r.runs)) {
    fail(r.error || 'unexpected reply from /api/runs');
    return null;
  }
  lastFail = '';
  return r.runs;
}

addEventListener('unhandledrejection', e => fail('script error: ' + (e.reason && e.reason.message || e.reason)));
addEventListener('error', e => fail('script error: ' + e.message));

// ------------------------------------------------------------------ setup
async function boot() {
  CFG = await api('/api/config');
  if (CFG.error || !CFG.baselines) {
    fail(CFG.error || 'unexpected reply from /api/config');
    return;
  }

  $('dataset').innerHTML = CFG.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  $('dataset').addEventListener('change', sizeContext);
  sizeContext();

  // "all" is not offered as a box of its own - ticking every box is "all",
  // and the select-all link is a clearer way to say it
  $('baselines').innerHTML = CFG.baselines.filter(b => b.key !== 'all').map(b => `
    <label>
      <input type="checkbox" value="${b.key}">
      <span><span class="name">${b.label}</span>
            <span class="note">${b.note}</span></span>
    </label>`).join('');

  const o = CFG.ollama;
  $('model').innerHTML = CFG.models.map(m =>
    `<option value="${m}">${m}${o.up ? (o.pulled.includes(m) ? '' : ' (not pulled)') : ''}</option>`
  ).join('');

  if (!o.up) {
    $('ollama').textContent = 'Ollama is not answering — only the length, bag-of-words and BERT baselines can run';
    $('ollama').style.color = 'var(--bad)';
  } else if (o.loaded.length) {
    $('ollama').textContent = 'Ollama up · holding ' + o.loaded.join(', ') + ' in VRAM';
  } else {
    $('ollama').textContent = 'Ollama up · nothing loaded';
  }
  // the header truncates this line when space is short; the whole of it
  // stays readable on hover
  $('ollama').title = $('ollama').textContent;

  for (const c of checks()) c.onchange = onBaselines;
  $('clearhist').onclick = clearFinished;
  $('pickall').onclick = () => setAll(true);
  $('picknone').onclick = () => setAll(false);
  onBaselines();
  $('kbmode').innerHTML = CFG.kb_modes.map(m =>
    `<option value="${m.key}">${m.label}</option>`).join('');
  $('kbmode').onchange = onKbMode;
  $('kbgo').onclick = updateKb;
  onKbMode();
  paintKb(CFG.kb);

  if (CFG.idx_pool) IDX_POOL = CFG.idx_pool;
  $('scope').onchange = onScope;
  $('limit').onkeydown = e => { if (e.key === 'Enter' && !$('go').disabled) go(); };
  onScope();
  $('go').onclick = go;
  $('stopbtn').onclick = stop;
  $('updbtn').onclick = doUpdate;
  $('offbtn').onclick = doStop;
  $('adminclose').onclick = () => { $('adminveil').hidden = true; };
  showVersion();
  for (const b of $('benchview').querySelectorAll('button'))
    b.onclick = () => benchView(b.dataset.bv);
  $('ledgerds').onchange = loadLedger;
  $('ledgerrefresh').onclick = loadLedger;
  for (const b of $('tabs').querySelectorAll('button')) b.onclick = () => showTab(b.dataset.tab);
  $('v-table').onclick = () => showView('table');
  $('v-chart').onclick = () => showView('chart');
  $('csvpick').onchange = () => { page = 0; loadCalls(); };
  $('showwhy').onchange = loadCalls;
  $('steppick').onchange = loadStep;
  $('prev').onclick = () => { if (page > 0) { page--; loadCalls(); } };
  $('next').onclick = () => { page++; loadCalls(); };
  for (const b of $('pages').querySelectorAll('button'))
    b.onclick = () => showPage(b.dataset.page);
  addEventListener('hashchange', () => showPage(pageInUrl()));
  showPage(pageInUrl());

  await refreshHistory();
  const running = (await runList() || []).find(r => r.status === 'running');
  if (running) select(running.id);
}

const checks = () => Array.from($('baselines').querySelectorAll('input'));
const chosen = () => checks().filter(c => c.checked).map(c => c.value);

function setAll(on) {
  for (const c of checks()) c.checked = on;
  onBaselines();
}

// The model question only matters if something that calls an LLM is ticked,
// and there is nothing to run until at least one box is.
// Ask how long this dataset's calls are and fill the window in before the
// run, rather than letting it start at 8192 and warn afterwards. Missing the
// window is not a near miss - Ollama keeps about half of it and drops the
// front - so the default being wrong costs a whole run.
async function sizeContext() {
  const ds = $('dataset').value;
  if (!ds) return;
  $('ctxhint').textContent = 'measuring this dataset…';
  const r = await api('/api/dataset/context?dataset=' + encodeURIComponent(ds));
  if (r.error) { $('ctxhint').textContent = r.error; return; }
  $('numctx').value = r.recommended;
  const n = x => x.toLocaleString();
  $('ctxhint').innerHTML = !r.over_default
    ? `Every call here fits the 8192 default — longest is about ${n(r.max)} `
      + `tokens. Nothing to change.`
    : `<strong>${n(r.over_default)} of ${n(r.rows)} calls here do not fit the `
      + `8192 default.</strong> Longest is about ${n(r.max)} tokens, 90% are `
      + `under ${n(r.p90)}. <strong>${n(r.recommended)}</strong> is filled in `
      + `above — ` + (r.over_recommended
          ? `it still leaves ${n(r.over_recommended)} call`
            + `${r.over_recommended === 1 ? '' : 's'} too long for any window `
            + `a local model will give, and those need a decision (drop, `
            + `split, or summarise) rather than a bigger number.`
          : `enough for every call in the set.`)
      + ` A call that overflows loses about half the window, from the front, `
      + `and comes back as a confident verdict on its goodbyes.`;
}

function onBaselines() {
  const sel = chosen();
  $('modelbox').hidden = !sel.some(k =>
    CFG.baselines.find(b => b.key === k).needs_model);
  $('go').disabled = sel.length === 0;
  $('go').textContent = sel.length > 1 ? `Run ${sel.length} baselines` : 'Run';
  if (!sel.length) $('formerr').textContent = '';
}

// The three shapes run_all.sh --limit takes. Picking one here rather than
// typing "idx:19" into a free text box is the whole point: the common case -
// re-running the single transcript a previous run went wrong on - is now a
// menu choice and a number.
let IDX_POOL = 40;      // replaced at boot by run_all.sh's IDX_LIMIT_USED

const SCOPES = {
  count: {label: 'How many calls', prefix: '', preset: '0', ph: '0', ok: /^\d+$/,
          hint: '0 = the whole dataset · N = the first N calls',
          bad: 'enter a whole number, or 0 for the whole dataset'},
  idx:   {label: 'Transcript index', prefix: 'idx:', preset: '', ph: 'e.g. 19',
          ok: /^\d+$/,
          hint: () => 'the n-th call of a --limit ' + IDX_POOL + ' run, '
              + 'counting from 0 - the same call check_one.py --idx n '
              + 'reproduces, so an index read off an earlier run names the '
              + 'same transcript here',
          bad: 'enter the index as a whole number, e.g. 19'},
  id:    {label: 'Transcript id', prefix: 'id:', preset: '', ph: 'e.g. CONV_0421',
          ok: /.+/,
          hint: "matched against the dataset's id column - the one row that "
              + 'equals it is the whole run',
          bad: 'enter the id exactly as the dataset spells it'},
};

// Relabel the box under the menu and clear what was typed for the old shape:
// a 0 left over from "a number of calls" is not a transcript anyone means.
function onScope() {
  const s = SCOPES[$('scope').value];
  $('limitlabel').textContent = s.label;
  $('limithint').textContent = typeof s.hint === 'function' ? s.hint() : s.hint;
  $('limit').value = s.preset;
  $('limit').placeholder = s.ph;
  $('formerr').textContent = '';
}

// What goes on the wire as --limit. The server checks this too; the point of
// checking here is to say which box is wrong while it is still on screen.
function limitValue() {
  const key = $('scope').value, s = SCOPES[key], v = $('limit').value.trim();
  if (!s.ok.test(v)) { $('formerr').textContent = s.bad; return null; }
  if (key === 'idx' && Number(v) >= IDX_POOL) {
    $('formerr').textContent = 'that order only has ' + IDX_POOL
      + ' calls in it, so the last index is ' + (IDX_POOL - 1);
    return null;
  }
  return s.prefix + v;
}

// "limit idx:19" is not what anyone calls that run - say it the way the form
// asked the question.
function limitText(limit) {
  const l = String(limit == null ? '' : limit);
  if (l.startsWith('idx:')) return 'transcript idx ' + l.slice(4);
  if (l.startsWith('id:'))  return 'transcript id ' + l.slice(3);
  if (l === '0') return 'all rows';
  return l + ' calls';
}

// -------------------------------------------------------------- run control
async function go() {
  $('formerr').textContent = '';
  const limit = limitValue();
  if (limit === null) return;
  $('go').disabled = true;
  const res = await api('/api/run', {
    dataset: $('dataset').value,
    baselines: chosen(),
    limit: limit,
    model: $('model').value,
    num_ctx: $('numctx').value,
    stripped: $('stripped').checked,
  });
  onBaselines();
  if (res.error) { $('formerr').textContent = res.error; return; }
  await refreshHistory();
  select(res.id);
}

async function stop() {
  if (!current) return;
  await api('/api/stop', {id: current});
}

// ------------------------------------------------- Web-RAG knowledge base
// harvest_patterns.py writes knowledge/scam_patterns.json, build_index.py
// re-embeds it into chroma_db. The webrag baseline reads the second one, so
// the two counts disagreeing is worth saying out loud.
let KB = null, kbNudged = false;

function paintKb(kb) {
  if (!kb) return;
  KB = kb;
  const bits = [kb.patterns === null ? 'no scam_patterns.json yet'
                : kb.patterns + ' patterns · ' + kb.categories + ' categories'];
  bits.push(kb.vectors === null ? 'index unreadable'
            : kb.vectors + ' indexed' + (kb.stale ? ' — index out of step, rebuild it' : ''));
  if (kb.last_refresh)
    bits.push('harvested ' + new Date(kb.last_refresh).toLocaleDateString());
  $('kbstate').textContent = bits.join(' · ');
  $('kbstate').style.color = kb.stale ? 'var(--warn)' : '';
  // A stale or missing index is the one case where the panel has something to
  // say, so it opens itself - once, so it never fights a user who closed it.
  if (!kbNudged && (kb.stale || kb.patterns === null || kb.vectors === null)) {
    kbNudged = true;
    $('kbpanel').open = true;
  }
  onKbMode();
}

function onKbMode() {
  const m = CFG.kb_modes.find(x => x.key === $('kbmode').value);
  if (!m) return;
  $('kbnote').textContent = m.note;
  // the two read-only modes never call Tavily, so the missing key is only a
  // problem for the ones that would harvest
  $('kberr').textContent = (m.harvests && m.exclusive && KB && !KB.tavily)
    ? 'no TAVILY_API_KEY in .env — harvesting needs one, rebuilding the index does not'
    : '';
}

async function updateKb() {
  $('kbpanel').open = true;
  $('kberr').textContent = '';
  $('kbgo').disabled = true;
  const res = await api('/api/kb', {mode: $('kbmode').value});
  $('kbgo').disabled = false;
  if (res.error) { $('kberr').textContent = res.error; return; }
  await refreshHistory();
  select(res.id);
}

async function refreshKb() {
  const r = await api('/api/kb');
  if (!r.error) paintKb(r.kb);
}

// ================================================== the Results ledger
// One run of every baseline takes an afternoon. This lets the table be built
// a baseline at a time instead: every finished run is recorded server-side
// in results/ledger.jsonl, and this view reads it back for one dataset.
let LEDGER = null, ledgerOpen = {};

function benchView(name) {
  for (const b of $('benchview').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.bv === name);
  $('bench-run').hidden = name !== 'run';
  $('bench-ledger').hidden = name !== 'ledger';
  if (name === 'ledger') loadLedger();
}

async function loadLedger() {
  const ds = $('ledgerds').value || '';
  $('ledgerbody').innerHTML = '<div class="card muted">reading the ledger…</div>';
  const r = await api('/api/ledger?dataset=' + encodeURIComponent(ds));
  if (r.error) {
    $('ledgerbody').innerHTML = '<div class="card hint" style="color:var(--bad)">'
      + esc(r.error) + '</div>';
    return;
  }
  LEDGER = r;
  // the dropdown lists every dataset, with how many results each already has,
  // and opens on one that has some rather than on an empty table
  const have = r.datasets_recorded || {};
  const opts = (CFG && CFG.datasets || []).map(d => d.path);
  for (const k of Object.keys(have)) if (!opts.includes(k)) opts.push(k);
  // first visit: the dataset the run form is on, since that is the one being
  // worked on - then any dataset that has something recorded
  const formDs = $('dataset') && $('dataset').value;
  const keep = ds || (have[formDs] ? formDs : '') || opts.find(o => have[o])
    || formDs || opts[0] || '';
  $('ledgerds').innerHTML = opts.map(o =>
    `<option value="${esc(o)}">${esc(o.replace(/^datasets\//, ''))}`
    + `${have[o] ? ' — ' + have[o] + ' result' + (have[o] === 1 ? '' : 's') : ''}</option>`
  ).join('');
  $('ledgerds').value = keep;
  if (keep !== ds) return loadLedger();
  paintLedger();
}

function paintLedger() {
  const r = LEDGER;
  if (!r.entries.length) {
    $('ledgerbody').innerHTML = '<div class="card muted">Nothing recorded for '
      + 'this dataset yet. Run any baseline on it from the form on the left; '
      + 'when the run finishes its numbers land here.</div>';
    return;
  }
  const by = {};
  for (const e of r.entries) (by[e.system] = by[e.system] || []).push(e);
  const systems = r.system_order.filter(k => by[k])
    .concat(Object.keys(by).filter(k => !r.system_order.includes(k)));
  const anyStripped = r.entries.some(e => e.stripped_acc !== undefined);
  const pct = x => x === undefined || x === null ? '—' : x.toFixed(1) + '%';
  const f3 = x => x === undefined || x === null ? '—' : x.toFixed(3);
  // the Model column only earns its width when an LLM baseline is in the table
  const anyModel = r.entries.some(e => e.model);
  const when = t => (t || '').slice(5);           // 2026-09-23 03:06 -> 09-23 03:06

  // The summary row for a baseline is its most recent run over the whole
  // dataset, if it has one - a 40-call pilot is recorded, but it should not
  // stand in for the real number just because it happened to be later.
  const full = e => !e.limit || e.limit === '0' || e.limit === '-';
  const pick = list => list.find(full) || list[0];

  const head = `<tr><th>Baseline</th><th>Acc</th><th>P</th><th>R</th><th>F1</th>`
    + (anyStripped ? '<th>Stripped acc</th><th>Trusted A</th>' : '')
    + '<th>Calls</th>' + (anyModel ? '<th>Model</th>' : '') + '<th>When</th><th>Runs</th></tr>';

  const line = (e, sub) => `<tr class="${sub ? 'sub' : 'top'}"
      ${sub ? '' : `data-sys="${esc(e.system)}"`}>
    <td class="optname">${sub ? '' : `<strong>${esc(r.labels[e.system] || e.system)}</strong>`}
      ${full(e) ? '' : '<span class="pill">pilot</span>'}</td>
    <td>${pct(e.acc)}</td><td>${f3(e.p)}</td><td>${f3(e.r)}</td><td>${f3(e.f1)}</td>
    ${anyStripped ? `<td>${pct(e.stripped_acc)}</td><td><strong>${pct(e.trusted)}</strong></td>` : ''}
    <td>${e.calls}</td>${anyModel ? `<td>${esc(e.model || '—')}</td>` : ''}
    <td class="muted" title="${esc(e.run_id)}">${esc(when(e.finished))}</td>`;

  let rows = '';
  for (const k of systems) {
    const list = by[k], top = pick(list);
    const accs = list.filter(full).map(e => e.acc);
    const lo = Math.min(...accs), hi = Math.max(...accs);
    // the spread is only worth a line when the runs actually disagree
    const spread = accs.length > 1 && hi > lo
      ? `<div class="hint" style="margin:0">${lo.toFixed(1)}–${hi.toFixed(1)}%</div>` : '';
    const toggle = list.length > 1
      ? ` <button class="link" data-toggle="${esc(k)}">${ledgerOpen[k] ? 'hide' : 'all runs'}</button>` : '';
    rows += line(top, false) + `<td>${list.length}${toggle}${spread}</td></tr>`;
    if (ledgerOpen[k]) for (const e of list)
      rows += line(e, true) + `<td><button class="link" title="${esc(e.run_id)}"`
        + ` data-hide="${esc(e.run_id)}" data-hsys="${esc(k)}">remove</button></td></tr>`;
  }
  $('ledgerbody').innerHTML = `<div class="card"><div class="scroll">
      <table class="ledger">${head}${rows}</table></div>
    <div class="hint" style="margin-top:10px">One row per baseline: its latest
      run over the whole dataset (a <span class="pill">pilot</span> run on a
      --limit only stands in when there is no full one). "all runs" lists every
      result it has, newest first, and the range under the run count is the
      spread across full runs. "remove" hides one result from here without
      touching the run itself.${anyStripped ? ' Trusted A = acc − max(0, stripped acc − 50).' : ''}</div>
    </div>`;

  for (const b of $('ledgerbody').querySelectorAll('[data-toggle]'))
    b.onclick = () => { ledgerOpen[b.dataset.toggle] = !ledgerOpen[b.dataset.toggle]; paintLedger(); };
  for (const b of $('ledgerbody').querySelectorAll('[data-hide]'))
    b.onclick = async () => {
      if (!confirm('Remove this ' + b.dataset.hsys + ' result from the Results tab?')) return;
      const res = await api('/api/ledger/hide', {run_id: b.dataset.hide, system: b.dataset.hsys});
      if (res.error) { alert(res.error); return; }
      loadLedger();
    };
}

// ======================================================= stop and update
function veil(title, note, log, closable) {
  $('adminveil').hidden = false;
  $('admintitle').textContent = title;
  $('adminnote').innerHTML = note || '';
  $('adminlog').hidden = !log;
  $('adminlog').textContent = log || '';
  $('adminclose').hidden = !closable;
}

async function showVersion() {
  const v = await api('/api/admin/version');
  if (v.error) return;
  const [hash, ...rest] = (v.commit || '').split(' ');
  $('version').textContent = hash + (v.branch && v.branch !== 'main' ? ' · ' + v.branch : '');
  $('version').title = 'running ' + v.commit + ' on ' + v.branch;
}

async function doUpdate() {
  if (!confirm('Pull the latest code from GitHub and restart this server?\n\n'
      + 'Runs already finished are kept. The public link stays the same.')) return;
  veil('Updating…', 'running git pull');
  const r = await api('/api/admin/update', {});
  if (r.error) {
    veil('Update not done', 'Nothing was changed.', r.error, true);
    return;
  }
  if (!r.updated) {
    veil('Already up to date', 'Running ' + esc(r.after) + ' — nothing to restart.',
         r.output, true);
    return;
  }
  veil('Updated — restarting',
       esc(r.before) + ' → <strong>' + esc(r.after) + '</strong><br>'
       + (r.web_ui_sh_changed ? '<span style="color:var(--warn)">web_ui.sh itself '
          + 'changed too; the server restarts now, but that script only takes '
          + 'effect the next time you start it from the terminal.</span><br>' : '')
       + 'The page reloads by itself when the new server answers.',
       r.commits || r.output, false);
  // wait for the new process to answer with the new commit, then reload
  const want = (r.after || '').split(' ')[0];
  for (let i = 0; i < 90; i++) {
    await new Promise(res => setTimeout(res, 1000));
    const v = await api('/api/admin/version');
    if (!v.error && (v.commit || '').split(' ')[0] === want) {
      location.reload();
      return;
    }
  }
  veil('The server has not come back', 'It was restarting into ' + esc(want)
       + '. Check the terminal (or the tmux session) it was started from.', '', true);
}

async function doStop() {
  if (!confirm('Stop this server?\n\nRuns already going keep going. The public '
      + 'link closes, and the page stops working until the server is started '
      + 'again from the terminal.')) return;
  const r = await api('/api/admin/stop', {});
  if (r.error) { veil('Could not stop', '', r.error, true); return; }
  const going = r.runs_still_going || [];
  veil('Server stopped',
       (going.length ? going.length + ' run' + (going.length === 1 ? ' is' : 's are')
          + ' still going and will finish on their own.<br>' : '')
       + 'Start it again on the machine with <code>./web_ui.sh --public</code> '
       + '(or <code>--tmux</code>). Finished runs and the Results tab are kept.');
}

function select(id) {
  benchView('run');
  current = id; offset = 0; page = 0;
  ART = {steps: [], csvs: []};
  $('log').textContent = '';
  $('tabs').hidden = false;
  $('results').innerHTML = '';
  $('calls').innerHTML = '';
  $('steplog').textContent = '';
  // a knowledge-base update and a BERT training run both produce no results
  // table and no per-call CSV, so they get the Output pane on their own
  showOnlyOutput(sideJob(id));
  showTab('output');
  markHistory();
  // load them now as well as on completion, so a run selected while it is
  // still going already offers the step logs of whatever has finished
  if (!sideJob(id)) loadArtifacts();
  if (timer) clearInterval(timer);
  poll();
  timer = setInterval(poll, 900);
}

// A run with a "kind" is a side job - a knowledge-base update, or training a
// checkpoint. Neither is scored against a dataset, so neither has a results
// table, per-call CSVs, or per-baseline step logs to show.
const sideJob = id => !!(RUNS[id] && RUNS[id].kind);

function showOnlyOutput(only) {
  for (const b of $('tabs').querySelectorAll('button'))
    b.hidden = only && b.dataset.tab !== 'output';
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  tab = name;
  for (const b of $('tabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.tab === name);
  for (const t of ['output','results','calls','steps'])
    $('t-' + t).hidden = t !== name;
  // a chart drawn while its tab was hidden measured a zero-width box and fell
  // back to the minimum, so redraw it now that it has a real width
  // Each tab fetches what it needs rather than relying on the poll loop
  // having got there first - a poll that failed must not leave Results blank.
  if (name === 'results' && !RESULTS) showResults();
  if (name === 'results' && view === 'chart' && RESULTS) paintChart();
  if (name === 'output' && current && !$('log').textContent) { offset = 0; poll(); }
  if (name === 'calls' && !$('calls').rows.length) loadCalls();
  if (name === 'steps' && !$('steplog').textContent) loadStep();
}

// ------------------------------------------------------------------ polling
async function poll() {
  if (!current) return;
  const r = await api(`/api/output?id=${encodeURIComponent(current)}&offset=${offset}`);
  if (r.error) {
    // looping on a broken request just hides it behind another one
    clearInterval(timer); timer = null;
    fail(r.error);
    return;
  }
  offset = r.offset;
  if (r.text) append(r.text);

  // missing this costs one header repaint - the run's own status came from
  // /api/output above, so the finish handling below still happens
  const runs = await runList();
  const meta = runs && runs.find(x => x.id === current);
  if (meta) { RUNS[meta.id] = meta; paintHeader(meta, r.status); }

  if (r.status !== 'running') {
    clearInterval(timer); timer = null;
    await refreshHistory();
    if (RUNS[current] && RUNS[current].kind === 'kb') {
      // the counts on the left are what just changed
      await refreshKb();
    } else if (RUNS[current] && RUNS[current].kind === 'train') {
      // a new checkpoint is what just changed, and it is on the other page
      await refreshModels();
    } else {
      await loadArtifacts();
      await showResults();
    }
  }
}

function append(text, into) {
  const log = $(into || 'log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  for (const line of text.split('\n')) {
    const span = document.createElement('span');
    let cls = '';
    if (line.startsWith('==>')) cls = 'l-step';
    else if (/^\s*ok\b/.test(line)) cls = 'l-ok';
    else if (/^\s*warn\b/.test(line)) cls = 'l-warn';
    else if (/^\s*fail\b/.test(line)) cls = 'l-fail';
    if (cls) span.className = cls;
    span.textContent = line + '\n';
    log.appendChild(span);
  }
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function paintHeader(meta, status) {
  const side = !!meta.kind;
  showOnlyOutput(side);
  if (side && tab !== 'output') showTab('output');
  $('runtitle').textContent = meta.label ||
    (meta.baseline + ' · ' + meta.dataset.replace('datasets/', ''));
  const bits = [side ? meta.dataset.replace('datasets/', '')
                     : limitText(meta.limit)];
  if (meta.model) bits.push(meta.model);
  bits.push(new Date(meta.started * 1000).toLocaleString());
  $('runsub').textContent = bits.join(' · ');
  const pill = $('runpill');
  pill.hidden = false;
  pill.textContent = status;
  pill.className = 'pill ' + status;
  $('stopbtn').hidden = status !== 'running';
}

// ------------------------------------------------------------------ results
let RESULTS = null, view = 'table';
const METRICS = [
  {key: 'acc', label: 'Accuracy', slot: 1, pct: true},
  {key: 'p',   label: 'Precision', slot: 2},
  {key: 'r',   label: 'Recall',   slot: 3},
  {key: 'f1',  label: 'F1',       slot: 4},
];

async function showResults() {
  if (!current) return;
  const got = await api(`/api/results?id=${encodeURIComponent(current)}`);
  if (got.error) {
    $('results').innerHTML = `<tr><td class="muted">${esc(got.error)}</td></tr>`;
    return;
  }
  RESULTS = got.results;
  if (!RESULTS) {
    $('results').innerHTML = '<tr><td class="muted">no numbers in this run\'s ' +
                             'logs yet - see the Output tab</td></tr>';
    return;
  }
  paintTable();
  showView(view);
  $('spread').textContent = (RESULTS.bert_spread || []).join(' · ');
}

function showView(which) {
  view = which;
  $('v-table').classList.toggle('on', which === 'table');
  $('v-chart').classList.toggle('on', which === 'chart');
  $('r-table').hidden = which !== 'table';
  $('r-chart').hidden = which === 'table';
  if (which === 'chart') paintChart();
}

function paintTable() {
  // A run given a stripped twin adds two columns: accuracy on the stripped
  // copy, from the same trained model, and trusted accuracy
  // A = a_full - max(0, a_stripped - 0.5). collect_results.py computes A.
  const paired = !!RESULTS.paired;
  const head = ['system','acc','P','R','F1','TP','FP','FN','TN']
    .concat(paired ? ['stripped acc', 'trusted A'] : []);
  let h = '<tr>' + head.map(x => `<th>${x}</th>`).join('') + '</tr>';
  for (const s of RESULTS.systems) {
    if (!s.ran) {
      h += `<tr class="skipped"><td>${s.system}</td>` +
           `<td colspan="${head.length - 1}">not run</td></tr>`;
      continue;
    }
    h += `<tr><td>${s.system}</td><td>${s.acc.toFixed(1)}%</td>` +
         `<td>${s.p.toFixed(3)}</td><td>${s.r.toFixed(3)}</td>` +
         `<td>${s.f1.toFixed(3)}</td><td>${s.tp}</td><td>${s.fp}</td>` +
         `<td>${s.fn}</td><td>${s.tn}</td>`;
    if (paired) {
      h += s.stripped
        ? `<td title="stripped copy: TP${s.stripped.tp} FP${s.stripped.fp} ` +
          `FN${s.stripped.fn} TN${s.stripped.tn}">${s.stripped.acc.toFixed(1)}%</td>` +
          `<td>${s.trusted.toFixed(1)}%</td>`
        : '<td class="muted">—</td><td class="muted">—</td>';
    }
    h += '</tr>';
  }
  $('results').innerHTML = h;
}

// --------------------------------------------------------------- the graph
// Grouped columns: one band per system that ran, one column per metric still
// switched on. All of them share a single 0-100% axis - accuracy is already a
// percentage and P/R/F1 are scaled to match, so there is never a second scale
// to reconcile. Plain SVG; nothing is fetched from anywhere.
//
// Turning a metric off removes its column and nothing else: the axis stays
// 0-100 and every surviving metric keeps the colour it had, so a chart read
// before the click still reads the same way after it.
const VISIBLE = new Set(METRICS.map(m => m.key));
const COLMAX = 24, COLGAP = 2, BANDPAD = 26, PLOTH = 300;
const PADL = 46, PADR = 14, PADT = 22;

function paintKey() {
  $('chartkey').innerHTML = METRICS.map(m => `
    <button data-k="${m.key}" class="${VISIBLE.has(m.key) ? '' : 'off'}"
            aria-pressed="${VISIBLE.has(m.key)}">
      <i style="background:var(--series-${m.slot})"></i>${m.label}
    </button>`).join('');
  for (const b of $('chartkey').querySelectorAll('button')) {
    b.onclick = () => {
      const k = b.dataset.k;
      if (VISIBLE.has(k)) VISIBLE.delete(k); else VISIBLE.add(k);
      paintChart();
    };
  }
}

function paintChart() {
  paintKey();
  const rows = (RESULTS ? RESULTS.systems : []).filter(s => s.ran);
  const ms = METRICS.filter(m => VISIBLE.has(m.key));
  const box = $('chart');

  if (!rows.length) {
    box.innerHTML = '<div class="muted">nothing ran in this run, so there is ' +
                    'nothing to plot</div>';
    return;
  }
  if (!ms.length) {
    box.innerHTML = '<div class="muted">every metric is switched off - turn ' +
                    'one back on above</div>';
    return;
  }

  const W = Math.max(box.clientWidth || 640, 420);
  const plotW = W - PADL - PADR;
  const bandW = plotW / rows.length;

  let colW = (bandW - BANDPAD - (ms.length - 1) * COLGAP) / ms.length;
  colW = Math.max(4, Math.min(COLMAX, colW));
  const groupW = ms.length * colW + (ms.length - 1) * COLGAP;

  // Measure before deciding: a value on every cap is right when the columns
  // are wide enough to hold one, and unreadable overlap when they are not.
  // The gridline ticks and the Data view carry the numbers either way.
  // Only when a single metric is on: four columns 24px apart cannot each
  // carry a 33px label, but one column per system has the whole band to itself.
  const capLabels = ms.length === 1 && bandW >= 40;
  const nameW = Math.max(...rows.map(r => r.system.length)) * 6.6;
  const tilt = bandW < nameW + 10;
  const PADB = tilt ? Math.min(96, Math.round(nameW * 0.62) + 26) : 42;

  const H = PADT + PLOTH + PADB;
  const base = PADT + PLOTH;
  const y = v => PADT + PLOTH * (1 - v / 100);

  let g = '';

  // recessive hairline grid; its ticks carry whatever the caps do not
  for (const t of [0, 25, 50, 75, 100]) {
    g += `<line x1="${PADL}" y1="${y(t)}" x2="${PADL + plotW}" y2="${y(t)}"
           stroke="var(--${t === 0 ? 'axis' : 'grid'})" stroke-width="1"/>`;
    g += `<text class="tick" x="${PADL - 9}" y="${y(t)}" text-anchor="end"
           dominant-baseline="middle">${t}%</text>`;
  }

  rows.forEach((sy, gi) => {
    const bandX = PADL + gi * bandW;
    const x0 = bandX + (bandW - groupW) / 2;

    // One hover target per system, the full height of the band, so there is
    // never a thin column to chase with the pointer.
    g += `<rect class="band" data-i="${gi}" x="${bandX}" y="${PADT}"
           width="${bandW}" height="${PLOTH}"/>`;

    ms.forEach((m, si) => {
      const v = m.pct ? sy[m.key] : sy[m.key] * 100;
      const x = x0 + si * (colW + COLGAP);
      g += col(x, y(v), colW, base - y(v), `var(--series-${m.slot})`);
      if (capLabels) {
        g += `<text class="val" x="${x + colW / 2}" y="${y(v) - 6}"
               text-anchor="middle">${m.pct ? v.toFixed(1) + '%'
                                            : (v / 100).toFixed(3)}</text>`;
      }
    });

    const cx = bandX + bandW / 2;
    g += tilt
      ? `<text class="name" x="${cx}" y="${base + 14}" text-anchor="end"
          transform="rotate(-35 ${cx} ${base + 14})">${esc(sy.system)}</text>`
      : `<text class="name" x="${cx}" y="${base + 18}" text-anchor="middle"
          dominant-baseline="hanging">${esc(sy.system)}</text>`;
  });

  box.innerHTML = `<svg class="viz" width="${W}" height="${H}" role="img"
    aria-label="${ms.map(m => m.label).join(', ')} for each system that ran"
    >${g}</svg>`;

  for (const b of box.querySelectorAll('.band')) {
    b.onmousemove = e => tip(e, rows[+b.dataset.i], ms);
    b.onmouseleave = hideTip;
  }
}

// A column with a 4px rounded cap and a square foot on the baseline.
function col(x, yTop, w, h, fill) {
  if (h <= 0.5) return '';
  const r = Math.min(4, w / 2, h);
  return `<path d="M${x},${yTop + h} V${yTop + r} a${r},${r} 0 0 1 ${r},${-r}
           H${x + w - r} a${r},${r} 0 0 1 ${r},${r} V${yTop + h} Z" fill="${fill}"/>`;
}

let TIP = null;
function tip(e, sy, ms) {
  if (!TIP) { TIP = document.createElement('div'); TIP.className = 'tip';
              document.body.appendChild(TIP); }
  TIP.innerHTML = `<b>${esc(sy.system)}</b><table>` +
    ms.map(m => `<tr><td>${m.label}</td><td>${m.pct ? sy[m.key].toFixed(1) + '%'
                    : sy[m.key].toFixed(3)}</td></tr>`).join('') +
    `<tr><td>TP / FP</td><td>${sy.tp} / ${sy.fp}</td></tr>` +
    `<tr><td>FN / TN</td><td>${sy.fn} / ${sy.tn}</td></tr></table>`;
  TIP.style.left = Math.min(e.clientX + 16, innerWidth - 300) + 'px';
  TIP.style.top = Math.min(e.clientY + 16, innerHeight - 190) + 'px';
  TIP.hidden = false;
}
function hideTip() { if (TIP) TIP.hidden = true; }

addEventListener('resize', () => { if (view === 'chart' && RESULTS) paintChart(); });

// ---------------------------------------------------------------- artifacts
async function loadArtifacts() {
  ART = await api(`/api/artifacts?id=${encodeURIComponent(current)}`);
  $('c-calls').textContent = ART.csvs.length ? '(' + ART.csvs.length + ')' : '';
  $('c-steps').textContent = ART.steps.length ? '(' + ART.steps.length + ')' : '';
  for (const b of $('tabs').querySelectorAll('button')) {
    if (b.dataset.tab === 'calls') b.disabled = !ART.csvs.length;
    if (b.dataset.tab === 'steps') b.disabled = !ART.steps.length;
  }
  $('csvpick').innerHTML = ART.csvs.map(c =>
    `<option value="${c.name}">${c.name} — ${kb(c.bytes)}</option>`).join('');
  $('steppick').innerHTML = ART.steps.map(s =>
    `<option value="${s.name}">${s.name} — ${kb(s.bytes)}</option>`).join('');
}

function kb(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  return (n / 1024 / 1024).toFixed(1) + ' MB';
}

// ------------------------------------------------------------ per-call rows
async function loadCalls() {
  const name = $('csvpick').value;
  if (!name) return;
  const r = await api(`/api/csv?id=${encodeURIComponent(current)}` +
                      `&name=${encodeURIComponent(name)}` +
                      `&offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}`);
  if (r.error) { $('calls').innerHTML = `<tr><td class="muted">${r.error}</td></tr>`; return; }

  // "true" is the gold label; every other non-text column is a system's call,
  // so it can be marked as agreeing with the label or not. A <system>_why
  // column is the exception: it is the model's reasoning, not a verdict, so it
  // is never scored against the label.
  const cols = r.columns;
  const truthAt = cols.indexOf('true');
  const why = cols.map(c => c.endsWith('_why'));
  // hiding them is worth having: seven systems means seven extra prose
  // columns, and the table is already wide
  $('whybox').hidden = !why.some(Boolean);
  const show = $('showwhy').checked;
  const keep = i => show || !why[i];

  let h = '<tr>' + cols.map((c, i) => keep(i)
      ? `<th class="${why[i] ? 'why' : ''}">${esc(why[i]
          ? c.slice(0, -4) + ' · why' : c)}</th>`
      : '').join('') + '</tr>';
  for (const row of r.rows) {
    h += '<tr>' + row.map((v, i) => {
      if (!keep(i)) return '';
      const c = cols[i];
      if (c === 'text' || c === 'transcript' || c === 'text_stripped')
        return `<td class="text">${esc(v.length > 260 ? v.slice(0, 260) + '…' : v)}</td>`;
      // the cell is clipped by CSS, so the full sentence goes in the tooltip
      if (why[i])
        return `<td class="why" title="${esc(v)}">${esc(v)}</td>`;
      if (truthAt >= 0 && i > truthAt && v)
        return `<td class="${v === row[truthAt] ? 'hit' : 'miss'}">${esc(v)}</td>`;
      return `<td>${esc(v)}</td>`;
    }).join('') + '</tr>';
  }
  $('calls').innerHTML = h;

  const from = r.total ? r.offset + 1 : 0;
  const to = Math.min(r.offset + PAGE_SIZE, r.total);
  $('pageinfo').textContent = `${from}–${to} of ${r.total}`;
  $('csvinfo').textContent = truthAt >= 0
    ? 'green = agrees with the true label, red = got it wrong' : '';
  $('prev').disabled = page === 0;
  $('next').disabled = to >= r.total;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

// ------------------------------------------------------------- step logs
async function loadStep() {
  const name = $('steppick').value;
  if (!name) return;
  const r = await api(`/api/step?id=${encodeURIComponent(current)}` +
                      `&name=${encodeURIComponent(name)}`);
  if (r.error) { $('steplog').textContent = r.error; return; }
  $('steplog').textContent = r.text;
  $('stepinfo').textContent = r.truncated
    ? 'showing the last part of ' + kb(r.bytes) : kb(r.bytes);
}

// ------------------------------------------------------------------ history
async function refreshHistory() {
  const all = await runList();
  if (!all) return;                 // leave the list showing what it had
  const runs = all.slice(0, 15);
  for (const r of runs) RUNS[r.id] = r;
  $('clearhist').hidden = !runs.some(r => r.status !== 'running');
  if (!runs.length) {
    $('hist').innerHTML = '<div class="muted">nothing yet</div>';
    $('clearhist').hidden = true;
    return;
  }
  $('hist').innerHTML = runs.map(r => `
    <a data-id="${r.id}">
      <button class="del" data-del="${r.id}" ${r.status === 'running' ? 'disabled' : ''}
              title="${r.status === 'running'
                        ? 'still running - stop it first'
                        : 'delete this run from the history'}"
              aria-label="delete this run">×</button>
      ${r.label || (r.baseline + ' · ' + r.dataset.replace('datasets/',''))}
      <span class="pill ${r.status}">${r.status}</span>
      <div class="meta">${r.kind ? '' : limitText(r.limit) + ' · '}${
        r.model ? r.model + ' · ' : ''}${new Date(r.started * 1000).toLocaleString()}</div>
    </a>`).join('');
  for (const a of $('hist').querySelectorAll('a')) a.onclick = () => select(a.dataset.id);
  // the button sits inside the row, so its click must not also select the run
  for (const b of $('hist').querySelectorAll('[data-del]'))
    b.onclick = e => { e.stopPropagation(); delRun(b.dataset.del); };
  markHistory();
}

// Deleting takes files off disk, so both paths say exactly what goes and both
// ask first. Neither touches results/*.csv - those are named after the dataset
// and the limit, so they are shared between runs.
async function delRun(id) {
  const r = RUNS[id];
  const name = (r && (r.label || r.baseline)) || id;
  if (!confirm(`Delete "${name}" from the history?\n\nIts captured output and `
             + `its per-baseline step logs are removed. The per-call CSVs in `
             + `results/ are left alone - other runs share them.`)) return;
  await sendDelete({id});
}

async function clearFinished() {
  const n = Object.values(RUNS).filter(r => r.status !== 'running').length;
  if (!confirm(`Delete every finished run from the history?\n\n`
             + `${n} run${n === 1 ? '' : 's'} on the list, plus anything older `
             + `than the fifteen shown. A run that is still going is kept. `
             + `The per-call CSVs in results/ are left alone.`)) return;
  await sendDelete({scope: 'finished'});
}

async function sendDelete(body) {
  $('histerr').textContent = '';
  const res = await api('/api/delete', body);
  if (res.error) { $('histerr').textContent = res.error; return; }
  // whatever went, it must not stay selected or the poll loop keeps asking
  // for a run that is no longer there
  const gone = body.scope === 'finished' ? (res.ids || []) : [body.id];
  for (const id of gone) delete RUNS[id];
  if (gone.includes(current)) deselect();
  await refreshHistory();
}

// Back to the state the page boots in, with no run selected.
function deselect() {
  if (timer) { clearInterval(timer); timer = null; }
  current = null; offset = 0; page = 0;
  ART = {steps: [], csvs: []};
  RESULTS = null;
  $('log').textContent = '';
  $('results').innerHTML = '';
  $('calls').innerHTML = '';
  $('steplog').textContent = '';
  $('tabs').hidden = true;
  $('runtitle').textContent = 'No run selected';
  $('runsub').textContent = '';
  $('runpill').hidden = true;
  $('stopbtn').hidden = true;
  markHistory();
}

function markHistory() {
  for (const a of $('hist').querySelectorAll('a'))
    a.classList.toggle('on', a.dataset.id === current);
}

// ================================================================ BERT
// The second page keeps its own state throughout - its own selected run, its
// own poller - so switching pages never disturbs a benchmark streaming into
// the first one. The only thing the two share is the run machinery on the
// server: a training run is a detached run like any other, which is why it
// also turns up under Recent runs.
const PAGES = ['bench', 'bert', 'bow', 'length', 'llm'];
// The BERT page was #mcq until the ontology came off it. Someone's bookmark
// should not quietly land on the benchmark form.
const PAGE_WAS = {mcq: 'bert'};
const pageInUrl = () => {
  const h = PAGE_WAS[location.hash.slice(1)] || location.hash.slice(1);
  return PAGES.includes(h) ? h : 'bench';
};

let BERT = null, model = null, MODELS = [];
let trainRun = null, mtimer = null, moffset = 0;

// form field -> the key /api/bert/config sends its default under
const MFIELDS = {tepochs: 'epochs', tbatch: 'batch_size', tmaxlen: 'max_length',
                 tlr: 'lr', tseed: 'seed', tlimit: 'limit', tholdout: 'holdout',
                 awindow: 'window', astride: 'stride', amaxlen: 'max_length'};

// The page is in the URL, so #bert can be bookmarked, reloaded, and sent to
// someone - and reloading while reading an answer comes back to the answer
// pane rather than to the benchmark form.
function showPage(name) {
  if (location.hash.slice(1) !== name)
    history.replaceState(null, '', name === 'bench' ? location.pathname : '#' + name);
  for (const b of $('pages').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.page === name);
  $('page-bench').hidden = name !== 'bench';
  $('page-bert').hidden = name !== 'bert';
  $('page-bow').hidden = name !== 'bow';
  $('page-length').hidden = name !== 'length';
  $('page-llm').hidden = name !== 'llm';
  // drawn on first visit rather than at boot: someone who only ever runs
  // benchmarks should not be made to wait for a directory scan of models/,
  // nor for ollama to be asked what it has pulled
  if (name === 'bert' && !BERT) bertBoot();
  if (name === 'bow' && !BOW) bowBoot();
  if (name === 'length' && !LEN) lenBoot();
  if (name === 'llm' && !LLM) llmBoot();
}

async function bertBoot() {
  const cfg = await api('/api/bert/config');
  if (cfg.error || !cfg.bases) {
    $('modelerr').textContent = cfg.error || 'unexpected reply from /api/bert/config';
    return;
  }
  BERT = cfg;

  $('tdataset').innerHTML = $('sampleds').innerHTML = BERT.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  $('tbase').innerHTML = BERT.bases.map(b =>
    `<option value="${b.id}">${b.id}</option>`).join('');
  for (const [id, key] of Object.entries(MFIELDS))
    if ($(id) && BERT.defaults[key] !== undefined) $(id).value = BERT.defaults[key];

  $('tbase').onchange = onBase;
  onBase();
  $('traingo').onclick = train;
  $('trainstop').onclick = () => trainRun && api('/api/stop', {id: trainRun});
  $('askgo').onclick = ask;
  $('sampleload').onclick = loadSample;
  forgetRowOnEdit('transcript', 'sampleinfo');
  $('sampleidx').onkeydown = e => { if (e.key === 'Enter') loadSample(); };
  $('unload').onclick = unloadModel;
  $('berteds').innerHTML = BERT.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  $('bertevgo').onclick = () => evalRun('bert', evalIds('bert', 'm-eval'), {
    model: model, dataset: $('berteds').value, limit: $('bertevlimit').value,
    threshold: $('bertevthr').value, strip_tags: $('bertevstrip').checked,
    gpu: $('bertevgpu').checked,
  });
  for (const b of $('berttabs').querySelectorAll('button'))
    b.onclick = () => showMtab(b.dataset.mtab);

  await refreshModels();
  // a training run already going when the page opens - a reload mid-train, or
  // one started before this server was restarted - is picked back up
  const going = (await runList() || []).find(
    r => r.kind === 'train' && r.status === 'running');
  if (going) { RUNS[going.id] = going; watchTraining(going.id); showMtab('train'); }
}

function onBase() {
  const b = BERT.bases.find(x => x.id === $('tbase').value);
  $('tbasenote').textContent = b ? b.note : '';
}

function showMtab(name) {
  for (const b of $('berttabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.mtab === name);
  for (const t of ['ask', 'eval', 'train', 'about'])
    $('m-' + t).hidden = t !== name;
}

// ------------------------------------------------------- trained models
async function refreshModels() {
  const r = await api('/api/bert/models');
  if (r.error) { $('modelerr').textContent = r.error; return; }
  $('modelerr').textContent = '';
  paintModels(r.models || [], r.worker || {loaded: false});
}

function paintModels(models, worker) {
  MODELS = models;
  paintWorker(worker);
  if (!models.length) {
    $('bertmodels').innerHTML = '<div class="muted">nothing trained yet</div>';
    model = null;
    paintBertHeader();
    return;
  }
  if (!models.some(m => m.name === model)) model = models[0].name;
  $('bertmodels').innerHTML = models.map(m => {
    const acc = m.holdout && m.holdout.acc != null
      ? 'holdout acc ' + (100 * m.holdout.acc).toFixed(1) + '%'
      : 'not scored';
    const live = worker.loaded && worker.model === m.name
      ? ' <span class="pill running">in memory</span>' : '';
    return `
    <a data-model="${esc(m.name)}">
      <button class="del" data-delmodel="${esc(m.name)}"
              title="delete this checkpoint" aria-label="delete this checkpoint">×</button>
      ${esc(m.name)}${live}
      <div class="meta">${esc(m.base)} · ${acc}<br>
        ${esc((m.dataset || 'dataset unrecorded').replace('datasets/', ''))} · ${kb(m.bytes)}</div>
    </a>`;
  }).join('');
  for (const a of $('bertmodels').querySelectorAll('a'))
    a.onclick = () => { model = a.dataset.model; markModels(); paintBertHeader(); };
  for (const b of $('bertmodels').querySelectorAll('[data-delmodel]'))
    b.onclick = e => { e.stopPropagation(); delModel(b.dataset.delmodel); };
  markModels();
  paintBertHeader();
}

function markModels() {
  for (const a of $('bertmodels').querySelectorAll('a'))
    a.classList.toggle('on', a.dataset.model === model);
}

function paintWorker(w) {
  $('unload').hidden = !w.loaded;
  $('workerstate').textContent = w.loaded
    ? `models/${w.model} is in memory on ${w.device}`
      + (w.idle_s > 90 ? ` · idle ${Math.round(w.idle_s / 60)} min` : '')
    : 'nothing in memory — the first question loads the checkpoint, which '
      + 'takes a few seconds';
}

function paintBertHeader(status) {
  const m = MODELS.find(x => x.name === model);
  if (m) {
    $('berttitle').textContent = m.name;
    const bits = [m.base];
    if (m.dataset) bits.push(m.dataset.replace('datasets/', '')
                             + (m.rows ? ' · ' + m.rows + ' calls' : ''));
    if (m.holdout && m.holdout.acc != null)
      bits.push('holdout acc ' + (100 * m.holdout.acc).toFixed(1) + '%'
                + ' · F1 ' + m.holdout.f1.toFixed(3));
    if (m.trained_at) bits.push(new Date(m.trained_at).toLocaleString());
    $('bertsub').textContent = bits.join(' · ');
  } else {
    $('berttitle').textContent = 'No model selected';
    $('bertsub').textContent = 'Train one on the left, then put a transcript '
                            + 'to it.';
  }
  const st = status || (trainRun && RUNS[trainRun] && RUNS[trainRun].status);
  const pill = $('bertpill');
  pill.hidden = !st;
  if (st) { pill.textContent = 'training · ' + st; pill.className = 'pill ' + st; }
  $('trainstop').hidden = st !== 'running';
}

async function delModel(name) {
  const m = MODELS.find(x => x.name === name);
  if (!confirm(`Delete the checkpoint models/${name}?\n\n`
             + `${m ? kb(m.bytes) + ' on disk. ' : ''}It cannot be recovered - `
             + `it would have to be trained again. Nothing else is touched.`)) return;
  $('modelerr').textContent = '';
  const r = await api('/api/bert/delete_model', {name});
  if (r.error) { $('modelerr').textContent = r.error; return; }
  if (model === name) { model = null; $('answer').innerHTML = ''; }
  await refreshModels();
}

async function unloadModel() {
  const r = await api('/api/bert/unload', {});
  if (r.error) { $('modelerr').textContent = r.error; return; }
  await refreshModels();
}

// ------------------------------------------------------------- training
async function train() {
  $('trainerr').textContent = '';
  $('traingo').disabled = true;
  const res = await api('/api/bert/train', {
    name: $('tname').value, dataset: $('tdataset').value, base: $('tbase').value,
    epochs: $('tepochs').value, batch_size: $('tbatch').value,
    max_length: $('tmaxlen').value, lr: $('tlr').value, seed: $('tseed').value,
    limit: $('tlimit').value, holdout: $('tholdout').value,
    cpu: $('tcpu').checked, overwrite: $('toverwrite').checked,
  });
  $('traingo').disabled = false;
  if (res.error) { $('trainerr').textContent = res.error; return; }
  RUNS[res.id] = res;
  watchTraining(res.id);
  showMtab('train');
  await refreshHistory();
}

function watchTraining(id) {
  trainRun = id;
  moffset = 0;
  $('trainlog').textContent = '';
  if (mtimer) clearInterval(mtimer);
  mpoll();
  mtimer = setInterval(mpoll, 900);
}

async function mpoll() {
  if (!trainRun) return;
  const r = await api(`/api/output?id=${encodeURIComponent(trainRun)}`
                    + `&offset=${moffset}`);
  if (r.error) {
    clearInterval(mtimer); mtimer = null;
    append('\n' + r.error + '\n', 'trainlog');
    return;
  }
  moffset = r.offset;
  if (r.text) append(r.text, 'trainlog');
  if (RUNS[trainRun]) RUNS[trainRun].status = r.status;
  paintBertHeader(r.status);
  if (r.status !== 'running') {
    clearInterval(mtimer); mtimer = null;
    // the checkpoint that just appeared is the point of the whole run
    await refreshModels();
    await refreshHistory();
  }
}

// ----------------------------------------------------------- asking it
// The dataset's own label for the row just loaded. It is the ground truth,
// not a prediction, so it is worth showing before the model is asked - and
// worth taking away the moment the text stops being that row.
function truthPill(label) {
  const v = String(label || '').trim().toLowerCase();
  if (!v) return '';
  const scam = ['scam', 'fraud', 'fraudulent', '1', 'true', 'yes'].includes(v);
  return ` <span class="pill ${scam ? 'truth-scam' : 'truth-legit'}">`
       + `labelled ${scam ? 'SCAM' : 'LEGITIMATE'}</span>`;
}

// Wires a transcript box so that editing it clears the row's label: once the
// text is not that row any more, the label is about nothing.
function forgetRowOnEdit(boxId, infoId) {
  $(boxId).addEventListener('input', () => { $(infoId).innerHTML = ''; });
}

async function loadSample() {
  $('sampleinfo').textContent = 'loading…';
  const r = await api(`/api/dataset/sample?dataset=${encodeURIComponent($('sampleds').value)}`
                    + `&idx=${encodeURIComponent($('sampleidx').value || 0)}`);
  if (r.error) { $('sampleinfo').textContent = r.error; return; }
  $('transcript').value = r.text;
  $('sampleidx').value = r.idx;
  $('sampleinfo').innerHTML = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + esc(r.row_id) : '') + truthPill(r.label);
}

async function ask() {
  $('askerr').textContent = '';
  if (!model) { $('askerr').textContent = 'train a model first - there is '
                                        + 'nothing to ask'; return; }
  const text = $('transcript').value.trim();
  if (!text) { $('askerr').textContent = 'paste a transcript, or load one from '
                                       + 'a dataset above'; return; }
  $('askgo').disabled = true;
  $('askgo').textContent = 'Classifying…';
  $('answer').innerHTML = '<div class="card muted">putting the call to '
    + esc(model) + '… the first one after a model or a setting changes also '
    + 'loads the checkpoint, which takes a few seconds</div>';
  const res = await api('/api/bert/classify', {
    model: model, transcript: text, gpu: $('agpu').checked,
    aggregate: $('aggregate').value, threshold: $('threshold').value,
    strip_tags: $('astrip').checked,
    window: $('awindow').value, stride: $('astride').value,
    max_length: $('amaxlen').value,
  });
  $('askgo').disabled = false;
  $('askgo').textContent = 'Classify this call';
  if (res.error) {
    $('askerr').textContent = res.error;
    $('answer').innerHTML = '';
    return;
  }
  paintAnswer(res);
  await refreshModels();        // the checkpoint is in memory now
}

function paintAnswer(a) {
  const pct = x => (100 * x).toFixed(1) + '%';
  const cls = a.verdict;

  // The three numbers side by side, because the interesting case is when they
  // disagree: "first window" is what a single truncated read would have said,
  // and on a long call that is a verdict on the hellos.
  const head = `
  <div class="card">
    <div class="verdict">
      <div>
        <div class="cap">Verdict</div>
        <div class="big ${cls}">${a.verdict.toUpperCase()}</div>
        <div class="hint">scam at ${a.threshold}</div>
      </div>
      <div>
        <div class="cap">prob_scam</div>
        <div class="big ${cls}">${pct(a.prob_scam)}</div>
        <div class="hint">${a.aggregate === 'max' ? 'strongest of'
          : 'average over'} ${a.windows} window${a.windows === 1 ? '' : 's'}</div>
      </div>
      <div style="flex:1; min-width:210px">
        <div class="cap">Across the call</div>
        <div style="font-weight:600">max ${pct(a.max)} · mean ${pct(a.mean)}
          · first ${pct(a.first_window)}</div>
        <div class="hint">${a.words} words · ${a.window_words}-word windows,
          stride ${a.stride}</div>
      </div>
    </div>
  </div>`;

  const notes = [];
  if (a.windows > 1 && Math.abs(a.max - a.first_window) >= 0.2)
    notes.push(`The opening of this call scores ${pct(a.first_window)} and its `
      + `strongest stretch ${pct(a.max)}. Reading only the first `
      + `${a.window_words} words — which is what a single pass through the `
      + `tokenizer does — would have said something else.`);
  if (a.stripped_tags)
    notes.push('Tone tags were stripped before scoring. Training read them, so '
      + 'this is the model being asked about text of a kind it never saw — a '
      + 'useful experiment, not a like-for-like number.');
  if (a.windows === 1)
    notes.push('This call fits in one window, so there was nothing to combine '
      + 'and all three numbers are the same read.');

  const bars = a.profile.map((s, i) => `
    <tr class="${i === a.hottest.index ? 'chosen' : ''}">
      <td class="optname">window ${i + 1}</td>
      <td>${pct(s)}</td>
      <td style="width:60%"><div class="bar ${s < 0.5 ? 'low' : ''}"
        style="margin:0"><i style="width:${Math.max(2, 100 * s)}%"></i></div></td>
    </tr>`).join('');

  $('answer').innerHTML = head
    + notes.map(n => `<div class="card hint">${n}</div>`).join('')
    + `
  <div class="card">
    <div class="qp">Strongest stretch — window ${a.hottest.index + 1} of
      ${a.windows}, ${pct(a.hottest.prob)}</div>
    <div class="ev">…${esc(a.hottest.text.slice(0, 400))}${
      a.hottest.text.length > 400 ? '…' : ''}</div>
    <details class="adv" style="margin-bottom:0; margin-top:14px">
      <summary>every window</summary>
      <div class="advbody"><div class="scroll"><table class="opts">
        <tr><th>window</th><th>prob_scam</th><th></th></tr>
        ${bars}
      </table></div></div>
    </details>
  </div>
  <div class="card hint">models/${esc(a.model.name)} · ${esc(a.model.base || '?')}
    · ${a.max_length} tokens per window · ${a.elapsed_ms} ms on
    ${esc(a.device)}</div>`;
}

// ========================================================== Bag of words
// The control page. Same shape as BERT - fit, then classify - but the model
// is a few hundred kilobytes and can be read back exactly, so the answer is
// the terms that decided it rather than a window profile.
let BOW = null, bowModel = null, bowModels = [], bowTimer = null, bowRun = null;
const BFIELDS = {bowngram: 'ngram_max', bowmindf: 'min_df',
                 bowholdout: 'holdout', bowseed: 'seed'};

async function bowBoot() {
  const cfg = await api('/api/bow/config');
  if (cfg.error || !cfg.datasets) {
    $('bowmodelerr').textContent = cfg.error || 'unexpected reply from /api/bow/config';
    return;
  }
  BOW = cfg;
  $('bowdataset').innerHTML = $('bowds').innerHTML = cfg.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  for (const [id, key] of Object.entries(BFIELDS))
    if (cfg.defaults[key] !== undefined) $(id).value = cfg.defaults[key];

  $('bowload').onclick = bowLoadRow;
  forgetRowOnEdit('bowtranscript', 'bowinfo');
  $('bowgo').onclick = bowAsk;
  $('bowtrain').onclick = bowFit;
  $('bowunload').onclick = async () => {
    await api('/api/bow/unload', {}); await bowRefresh();
  };
  $('boweds').innerHTML = $('bowdataset').innerHTML;
  $('bowevgo').onclick = () => evalRun('bow', evalIds('bow', 'b-eval'), {
    model: bowModel, dataset: $('boweds').value, limit: $('bowevlimit').value,
    threshold: $('bowevthr').value, strip_tags: $('bowevstrip').checked,
  });
  for (const b of $('bowtabs').querySelectorAll('button'))
    b.onclick = () => bowTab(b.dataset.btab);
  await bowRefresh();
}

function bowTab(name) {
  for (const b of $('bowtabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.btab === name);
  for (const t of ['ask', 'eval', 'train', 'about'])
    $('b-' + t).hidden = t !== name;
}

async function bowRefresh() {
  const r = await api('/api/bow/models');
  if (r.error) { $('bowmodelerr').textContent = r.error; return; }
  bowModels = r.models || [];
  const w = r.worker || {};
  $('bowworkerstate').textContent = w.loaded
    ? `models/${w.model} is in memory` : '';
  $('bowunload').hidden = !w.loaded;

  if (!bowModels.length) {
    $('bowmodels').innerHTML = '<div class="muted">nothing fitted yet</div>';
    bowModel = null;
  } else {
    if (!bowModels.some(m => m.name === bowModel)) bowModel = bowModels[0].name;
    $('bowmodels').innerHTML = bowModels.map(m => {
      const acc = m.holdout && m.holdout.acc;
      return `<div class="row">
        <a href="#bow" data-bow="${esc(m.name)}"
           class="${m.name === bowModel ? 'on' : ''}">
          <strong>${esc(m.name)}</strong>
          <span class="muted">${esc(m.dataset || '')}
            · ${m.features || '?'} features${acc
              ? ' · holdout acc ' + (100 * acc).toFixed(1) + '%' : ''}</span>
        </a>
        <button class="link" data-bowdel="${esc(m.name)}">delete</button>
      </div>`;
    }).join('');
    for (const a of $('bowmodels').querySelectorAll('a'))
      a.onclick = e => { e.preventDefault(); bowModel = a.dataset.bow; bowRefresh(); };
    for (const b of $('bowmodels').querySelectorAll('[data-bowdel]'))
      b.onclick = async () => {
        if (!confirm('Delete models/' + b.dataset.bowdel + '?')) return;
        const r = await api('/api/bow/delete_model', {name: b.dataset.bowdel});
        if (r.error) { $('bowmodelerr').textContent = r.error; return; }
        await bowRefresh();
      };
  }
  const m = bowModels.find(x => x.name === bowModel);
  $('bowtitle').textContent = m ? m.name : 'No model selected';
  $('bowsub').textContent = m
    ? [m.dataset, (m.rows || '?') + ' calls', (m.features || '?') + ' features',
       'n-grams to ' + (m.ngram_max || '?'),
       m.holdout && m.holdout.acc
         ? 'holdout acc ' + (100 * m.holdout.acc).toFixed(1) + '%' : null,
       m.elapsed_s !== undefined ? 'fitted in ' + m.elapsed_s + 's' : null,
      ].filter(Boolean).join(' · ')
    : 'Fit one on the left — it takes about a second — then put a transcript to it.';
}

async function bowLoadRow() {
  $('bowinfo').textContent = 'loading…';
  const r = await api(`/api/dataset/sample?dataset=${encodeURIComponent($('bowds').value)}`
                    + `&idx=${encodeURIComponent($('bowidx').value || 0)}`);
  if (r.error) { $('bowinfo').textContent = r.error; return; }
  $('bowtranscript').value = r.text;
  $('bowidx').value = r.idx;
  $('bowinfo').innerHTML = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + esc(r.row_id) : '') + truthPill(r.label);
}

async function bowFit() {
  $('bowtrainerr').textContent = '';
  const body = {name: $('bowname').value, dataset: $('bowdataset').value,
                strip_tags: $('bowstriptrain').checked,
                overwrite: $('bowoverwrite').checked};
  for (const [id, key] of Object.entries(BFIELDS)) body[key] = $(id).value;
  const res = await api('/api/bow/train', body);
  if (res.error) { $('bowtrainerr').textContent = res.error; return; }
  bowRun = res.id;
  bowTab('train');
  bowPoll();
}

// It finishes in about a second, so this polls briefly rather than streaming.
async function bowPoll() {
  if (!bowRun) return;
  const r = await runLog(bowRun);
  if (!r.error) $('bowtrainlog').textContent = r.text || '(no output yet)';
  clearTimeout(bowTimer);
  // A failed poll is not a finished run. Retry a few times before giving up,
  // rather than concluding the run ended because one request did not land.
  if (r.error) {
    if ((POLL_FAILS['bowPoll'] = (POLL_FAILS['bowPoll'] || 0) + 1) < 5) {
      bowTimer = setTimeout(bowPoll, 2000);
    } else {
      $('bowtrainlog').textContent = 'lost contact with the run: ' + r.error;
    }
    return;
  }
  POLL_FAILS['bowPoll'] = 0;
  if (runOver(r)) { await bowRefresh(); return; }
  bowTimer = setTimeout(bowPoll, 900);
}

async function bowAsk() {
  $('bowaskerr').textContent = '';
  if (!bowModel) { $('bowaskerr').textContent = 'fit a model first - there is '
                                              + 'nothing to ask'; return; }
  const text = $('bowtranscript').value.trim();
  if (!text) { $('bowaskerr').textContent = 'paste a transcript, or load one '
                                          + 'from a dataset above'; return; }
  $('bowgo').disabled = true;
  $('bowgo').textContent = 'Classifying…';
  const res = await api('/api/bow/classify', {
    model: bowModel, transcript: text, threshold: $('bowthreshold').value,
    top: $('bowtop').value, strip_tags: $('bowstrip').checked,
  });
  $('bowgo').disabled = false;
  $('bowgo').textContent = 'Classify this call';
  if (res.error) {
    $('bowaskerr').textContent = res.error;
    $('bowanswer').innerHTML = '';
    return;
  }
  bowPaint(res);
  await bowRefresh();
}

function bowPaint(a) {
  const pct = x => (100 * x).toFixed(1) + '%';
  const terms = (list, cls) => list.length ? list.map(t => `
    <tr><td class="optname">${esc(t.term)}</td>
        <td class="${cls}">${t.contribution > 0 ? '+' : ''}${t.contribution.toFixed(4)}</td>
        <td style="width:55%"><div class="bar" style="margin:0"><i
          style="width:${Math.min(100, Math.abs(t.contribution) * 100 / Math.max(
            0.0001, Math.abs((list[0] || {}).contribution || 1)))}%;
          background:var(${cls === 'up' ? '--bad' : '--accent'})"></i></div></td>
    </tr>`).join('') : '<tr><td class="optname muted">nothing</td><td></td><td></td></tr>';

  $('bowanswer').innerHTML = `
  <div class="card">
    <div class="verdict">
      <div>
        <div class="cap">Verdict</div>
        <div class="big ${a.verdict}">${a.verdict.toUpperCase()}</div>
        <div class="hint">scam at ${a.threshold}</div>
      </div>
      <div>
        <div class="cap">prob_scam</div>
        <div class="big ${a.verdict}">${pct(a.prob_scam)}</div>
        <div class="hint">score ${a.score > 0 ? '+' : ''}${a.score.toFixed(3)}</div>
      </div>
      <div style="flex:1; min-width:230px">
        <div class="cap">What it could see</div>
        <div style="font-weight:600">${a.vocab_known} of ${a.vocab_distinct}
          distinct words known · ${a.matched} features matched</div>
        <div class="hint">${a.words} words${a.stripped_tags
          ? ' · tone tags stripped' : ''}</div>
      </div>
    </div>
  </div>
  <div class="card hint">The score is the intercept
    (${a.intercept > 0 ? '+' : ''}${a.intercept.toFixed(3)}) plus every term's
    TF-IDF weight times its coefficient. The terms below are those products,
    largest first — they sum to
    ${(a.score - a.intercept) > 0 ? '+' : ''}${(a.score - a.intercept).toFixed(3)},
    which with the intercept is the score. This is the arithmetic, not a story
    about it.</div>
  <div class="card">
    <div class="qp">Towards SCAM</div>
    <div class="scroll"><table class="opts">${terms(a.toward_scam, 'up')}</table></div>
    <div class="qp" style="margin-top:18px">Towards LEGITIMATE</div>
    <div class="scroll"><table class="opts">${terms(a.toward_legit, 'down')}</table></div>
  </div>
  <div class="card hint">models/${esc(a.model.name)} · ${esc(a.model.dataset || '?')}
    · ${a.model.features || '?'} features · ${a.elapsed_ms} ms</div>`;
}

// ====================================================== scoring a dataset
// One implementation for all four pages. The four models are different
// enough that they each get their own page, but a confusion matrix is a
// confusion matrix, and four copies of this would be four chances for the
// numbers to stop meaning the same thing.
const EVAL = {};     // page -> {run, ids, timer}

function evalIds(pre, pane) {
  return {ds: pre + 'eds', limit: pre + 'evlimit', go: pre + 'evgo',
          err: pre + 'everr', out: pre + 'evout', log: pre + 'evlog',
          pane: pane};
}

// Each page says what it is scoring and with what; everything after the POST
// is shared.
async function evalRun(page, ids, body) {
  $(ids.err).textContent = '';
  if (!body.dataset) { $(ids.err).textContent = 'pick a dataset'; return; }
  $(ids.go).disabled = true;
  const res = await api('/api/' + page + '/evaluate', body);
  $(ids.go).disabled = false;
  if (res.error) { $(ids.err).textContent = res.error; return; }
  clearTimeout((EVAL[page] || {}).timer);
  EVAL[page] = {run: res.id, ids: ids};
  $(ids.out).innerHTML = '<div class="card muted">scoring…</div>';
  $(ids.log).hidden = false;
  evalPoll(page);
}

async function evalPoll(page) {
  const st = EVAL[page];
  if (!st || !st.run) return;
  const r = await runLog(st.run);
  if (r.error) {
    // A failed poll is not a finished run. Treating it as one is what turned
    // a blocked or dropped request into "the run finished without writing a
    // score", with an empty log pane under it and no way to tell that the
    // run was in fact still going. So: say so, keep trying, and only give up
    // after several in a row.
    st.fails = (st.fails || 0) + 1;
    $(st.ids.log).textContent = 'could not read the run log (attempt '
      + st.fails + '): ' + r.error;
    if (st.fails < 5) {
      st.timer = setTimeout(() => evalPoll(page), 2000);
      return;
    }
  } else {
    st.fails = 0;
    // the tail is the interesting part while it runs - a 7,000-call log is
    // 7,000 lines and the browser should not be asked to lay all of them out.
    // Trailing blank lines go: read_log strips the exit marker and leaves its
    // newlines, and a pane scrolled to the bottom of those is a blank box.
    const text = (r.text || '').replace(/\s+$/, '');
    const lines = text.split('\n');
    $(st.ids.log).textContent = lines.length > 400
      ? '… ' + (lines.length - 400) + ' earlier lines\n'
        + lines.slice(-400).join('\n')
      : (text || '(no output yet)');
    $(st.ids.log).scrollTop = $(st.ids.log).scrollHeight;
  }
  clearTimeout(st.timer);
  if (r.error || runOver(r)) {
    const m = await api(`/api/eval/result?id=${encodeURIComponent(st.run)}`);
    // the result endpoint knows whether the run is still going even when the
    // log could not be read, so a run that is merely unreachable keeps being
    // waited on rather than being declared over
    if (!m.error && !m.ready && m.status === 'running') {
      st.timer = setTimeout(() => evalPoll(page), 2000);
      return;
    }
    if (m.error || !m.ready) {
      // Say what went wrong here, not "see the log": the log pane is below
      // the fold and is scrolled to its end, where a stripped exit marker
      // leaves blank lines - so a crash used to look like no output at all.
      $(st.ids.out).innerHTML = `<div class="card">
        <div class="qp">No score — ${esc(m.error || m.why || 'the run did not finish')}</div>
        <div class="hint">The run is <code>${esc(st.run)}</code>${
          m.status ? ' and its status is <code>' + esc(m.status) + '</code>' : ''}.
          ${m.error ? 'The server could not be asked for the result.'
                    : 'This is the end of what it printed:'}</div>
        ${m.tail ? `<pre class="log" style="margin-top:12px">${esc(m.tail)}</pre>`
                 : ''}</div>`;
      return;
    }
    $(st.ids.out).innerHTML = evalCard(m);
    return;
  }
  st.timer = setTimeout(() => evalPoll(page), 1200);
}

const pc = x => (100 * x).toFixed(1) + '%';

// The confusion matrix as a table, because the four cells are what every
// other number on the card is derived from and a reader should be able to
// check the arithmetic.
function evalMatrix(m) {
  const cell = (n, tot, cls) => `<td class="${cls}"><strong>${n}</strong>`
    + `<span class="muted"> ${tot ? pc(n / tot) : '—'}</span></td>`;
  const scam = m.tp + m.fn, legit = m.fp + m.tn;
  return `<table class="cm">
    <tr><th></th><th>said scam</th><th>said legitimate</th></tr>
    <tr><th>really scam</th>${cell(m.tp, scam, 'good')}${cell(m.fn, scam, 'bad')}</tr>
    <tr><th>really legitimate</th>${cell(m.fp, legit, 'bad')}${cell(m.tn, legit, 'good')}</tr>
  </table>`;
}

function missList(title, list, note) {
  if (!list || !list.length) return '';
  return `<div class="qp" style="margin-top:16px">${title}
      <span class="muted" style="font-weight:400">— ${note}</span></div>`
    + list.map(x => `<div class="ev">
        <span class="muted">id ${esc(String(x.id))} · ${x.words} words${
          x.prob_scam ? ' · p(scam) ' + x.prob_scam : ''}${
          x.words_counted !== undefined ? ' · counted ' + x.words_counted : ''}${
          x.verdict ? ' · said ' + esc(x.verdict) : ''}</span><br>${esc(x.excerpt)}${
          x.reason ? '<br><em>' + esc(x.reason) + '</em>' : ''}</div>`).join('');
}

function evalCard(d) {
  const m = d.metrics, b = d.baselines;
  const floor = Math.max(b.always_scam.acc, b.never_scam.acc);
  const over = m.acc - floor;
  const notes = [];

  // Which of three experiments this is. They are not comparable with each
  // other, and none of them is the Benchmark page's cross-validated figure -
  // a 100% there next to a 50% here is two experiments, not a disagreement,
  // and that is exactly the reading this note exists to stop.
  // The LLM was never fitted on anything, so none of the three applies to
  // it; what matters there is which prompt was used, which is below.
  if (d.kind === 'llm') {
    notes.push(d.profile
      ? 'Scored under the fitted prompt <code>models/' + esc(d.profile)
        + '</code>. Run it again with <em>None</em> picked to see what the '
        + 'bare <code>llm_only</code> prompt gets on the same calls — that '
        + 'difference is what the fitting bought outside its own holdout.'
      : 'Scored with the bare <code>llm_only</code> prompt, so this is the '
        + 'control. It is the same prompt the Benchmark page uses, over the '
        + 'same calls.');
  } else if (d.same_dataset) notes.push('<strong>This model was fitted on this '
    + 'dataset.</strong> Unless rows were held back, it has read these calls '
    + 'before, so the score is a memory test rather than a measurement. Point '
    + 'it at a dataset it has never seen for the number worth quoting.');
  else if (d.trained_on) notes.push('<strong>This is a transfer test.</strong> '
    + 'The model was fitted on <code>' + esc(d.trained_on) + '</code> and '
    + 'scored on <code>' + esc(d.dataset) + '</code> — how far what it learned '
    + 'on one corpus carries to another. That is a harder question than the '
    + 'Benchmark page asks: its figure is k-fold cross-validation <em>within</em> '
    + 'one dataset, so it trains and scores on the same kind of text. A low '
    + 'number here beside a high one there is a finding, not a contradiction.');
  else notes.push('This model does not record what it was fitted on, so '
    + 'whether it has already read these calls cannot be told from here.');
  if (m.acc <= floor) notes.push('<strong>This does not beat answering the '
    + 'same thing every time</strong> (' + pc(floor) + ' by always saying '
    + (b.always_scam.acc >= b.never_scam.acc ? 'scam' : 'legitimate')
    + '). On this dataset the model is adding nothing to the class balance.');
  if (m.unreadable) notes.push(m.unreadable + ' call(s) got no readable '
    + 'answer. They are left out of the scores above rather than counted as '
    + 'legitimate — which is what would quietly turn every one of them into a '
    + 'false negative.');
  if (d.truncated) notes.push(d.truncated + ' prompt(s) were truncated by '
    + 'ollama, so those verdicts are about part of the call. Raise the '
    + 'context window and score again before quoting this.');
  if (d.mirror && d.mirror.acc > m.acc) notes.push('The same threshold '
    + 'pointing the other way would get <strong>' + pc(d.mirror.acc)
    + '</strong>. The rule is the wrong way round for this dataset.');

  const big = (label, val, sub) => `<div>
      <div class="cap">${label}</div>
      <div class="big">${val}</div>
      <div class="hint">${sub}</div></div>`;

  return `
  <div class="card">
    <div class="verdict">
      ${big('Accuracy', pc(m.acc),
            (over > 0 ? '+' + (100 * over).toFixed(1) + ' over ' : 'under ')
            + 'the best constant answer')}
      ${big('Precision', m.precision.toFixed(3), 'of the calls it called scam')}
      ${big('Recall', m.recall.toFixed(3), 'of the scams it found')}
      ${big('F1', m.f1.toFixed(3), 'balanced acc ' + pc(m.balanced_acc))}
    </div>
  </div>
  <div class="card">
    <div class="qp">Where the ${m.scored} scored calls went</div>
    ${evalMatrix(m)}
    <div class="hint" style="margin-top:12px">
      always scam would get ${pc(b.always_scam.acc)} ·
      never scam ${pc(b.never_scam.acc)} ·
      specificity ${m.specificity.toFixed(3)} ·
      ${d.calls} calls in ${d.elapsed_s}s</div>
  </div>
  ${notes.map(n => `<div class="card hint">${n}</div>`).join('')}
  <div class="card">
    ${missList('Called scam, was not', d.misses && d.misses.false_scam,
               'false positives')}
    ${missList('Called legitimate, was a scam', d.misses && d.misses.missed_scam,
               'false negatives — the expensive kind')}
    ${missList('No readable answer', d.misses && d.misses.unreadable,
               'scored as neither')}
    ${(d.misses && (d.misses.false_scam.length || d.misses.missed_scam.length
      || d.misses.unreadable.length)) ? '' :
      '<div class="qp">Nothing to show — it got every call right.</div>'}
    <div class="hint" style="margin-top:14px">Up to ten of each are sampled
      here. Every call is in <code>${esc(d.per_call_csv)}</code>, with the
      prediction, the truth and the transcript.</div>
  </div>`;
}

// =========================================================== Length only
// The floor. Same shape as the other two - fit, then classify - but the
// model is one integer, so the answer is the comparison itself: where the
// call falls against the line, against the two fitting distributions, and
// what share of fitting calls on that side of the line really were scams.
let LEN = null, lenModel = null, lenModels = [], lenTimer = null, lenRun = null;
const LFIELDS2 = {lenholdout: 'holdout', lenseed: 'seed', lenlimit: 'limit',
                  lenthreshold: 'threshold'};

async function lenBoot() {
  const cfg = await api('/api/length/config');
  if (cfg.error || !cfg.datasets) {
    $('lenmodelerr').textContent = cfg.error
      || 'unexpected reply from /api/length/config';
    return;
  }
  LEN = cfg;
  $('lendataset').innerHTML = $('lends').innerHTML = cfg.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  for (const [id, key] of Object.entries(LFIELDS2))
    if (cfg.defaults[key] !== undefined && cfg.defaults[key] !== null)
      $(id).value = cfg.defaults[key];
  // the pinned box opens on the benchmark's own rule, since that is the only
  // reason to pin it rather than sweep
  $('lenthreshold').value = cfg.benchmark_threshold;

  $('lenpin').onchange = () => { $('lenpinrow').hidden = !$('lenpin').checked; };
  $('lenload').onclick = lenLoadRow;
  forgetRowOnEdit('lentranscript', 'leninfo');
  $('lengo').onclick = lenAsk;
  $('lenfit').onclick = lenFit;
  $('lenunload').onclick = async () => {
    await api('/api/length/unload', {}); await lenRefresh();
  };
  $('leneds').innerHTML = $('lendataset').innerHTML;
  $('lenevgo').onclick = () => evalRun('length', evalIds('len', 'l-eval'), {
    model: lenModel, dataset: $('leneds').value, limit: $('lenevlimit').value,
    threshold: $('lenevthr').value, direction: $('lenevdir').value,
    strip_tags: $('lenevstrip').checked,
  });
  for (const b of $('lentabs').querySelectorAll('button'))
    b.onclick = () => lenTab(b.dataset.ltab);
  await lenRefresh();
}

function lenTab(name) {
  for (const b of $('lentabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.ltab === name);
  for (const t of ['ask', 'eval', 'train', 'about'])
    $('l-' + t).hidden = t !== name;
}

const lenRule = m => `${m.direction || '?'} than ${m.threshold} words`;

async function lenRefresh() {
  const r = await api('/api/length/models');
  if (r.error) { $('lenmodelerr').textContent = r.error; return; }
  lenModels = r.models || [];
  const w = r.worker || {};
  $('lenworkerstate').textContent = w.loaded
    ? `models/${w.model} is loaded` : '';
  $('lenunload').hidden = !w.loaded;

  if (!lenModels.length) {
    $('lenmodels').innerHTML = '<div class="muted">nothing fitted yet</div>';
    lenModel = null;
  } else {
    if (!lenModels.some(m => m.name === lenModel)) lenModel = lenModels[0].name;
    $('lenmodels').innerHTML = lenModels.map(m => {
      const acc = m.holdout && m.holdout.acc;
      return `<div class="row">
        <a href="#length" data-len="${esc(m.name)}"
           class="${m.name === lenModel ? 'on' : ''}">
          <strong>${esc(m.name)}</strong>
          <span class="muted">${esc(m.dataset || '')}
            · ${esc(lenRule(m))}${m.swept === false ? ' · pinned' : ''}${acc
              ? ' · holdout acc ' + (100 * acc).toFixed(1) + '%' : ''}</span>
        </a>
        <button class="link" data-lendel="${esc(m.name)}">delete</button>
      </div>`;
    }).join('');
    for (const a of $('lenmodels').querySelectorAll('a'))
      a.onclick = e => { e.preventDefault(); lenModel = a.dataset.len; lenRefresh(); };
    for (const b of $('lenmodels').querySelectorAll('[data-lendel]'))
      b.onclick = async () => {
        if (!confirm('Delete models/' + b.dataset.lendel + '?')) return;
        const r = await api('/api/length/delete_model', {name: b.dataset.lendel});
        if (r.error) { $('lenmodelerr').textContent = r.error; return; }
        await lenRefresh();
      };
  }
  const m = lenModels.find(x => x.name === lenModel);
  $('lentitle').textContent = m ? m.name : 'No model selected';
  $('lensub').textContent = m
    ? [m.dataset, (m.rows || '?') + ' calls', lenRule(m),
       m.swept === false ? 'pinned, not swept' : 'swept',
       m.holdout && m.holdout.acc
         ? 'holdout acc ' + (100 * m.holdout.acc).toFixed(1) + '%' : null,
      ].filter(Boolean).join(' · ')
    : 'Fit one on the left — it is one sort of the dataset — then put a '
      + 'transcript to it.';
}

async function lenLoadRow() {
  $('leninfo').textContent = 'loading…';
  const r = await api(`/api/dataset/sample?dataset=${encodeURIComponent($('lends').value)}`
                    + `&idx=${encodeURIComponent($('lenidx').value || 0)}`);
  if (r.error) { $('leninfo').textContent = r.error; return; }
  $('lentranscript').value = r.text;
  $('lenidx').value = r.idx;
  $('leninfo').innerHTML = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + esc(r.row_id) : '') + truthPill(r.label);
}

async function lenFit() {
  $('lenfiterr').textContent = '';
  const body = {name: $('lenname').value, dataset: $('lendataset').value,
                metric: $('lenmetric').value,
                pin: $('lenpin').checked,
                direction: $('lendirection').value,
                strip_tags: $('lenstriptrain').checked,
                overwrite: $('lenoverwrite').checked};
  for (const [id, key] of Object.entries(LFIELDS2)) body[key] = $(id).value;
  const res = await api('/api/length/train', body);
  if (res.error) { $('lenfiterr').textContent = res.error; return; }
  lenRun = res.id;
  lenTab('train');
  lenPoll();
}

async function lenPoll() {
  if (!lenRun) return;
  const r = await runLog(lenRun);
  if (!r.error) $('lentrainlog').textContent = r.text || '(no output yet)';
  clearTimeout(lenTimer);
  // A failed poll is not a finished run. Retry a few times before giving up,
  // rather than concluding the run ended because one request did not land.
  if (r.error) {
    if ((POLL_FAILS['lenPoll'] = (POLL_FAILS['lenPoll'] || 0) + 1) < 5) {
      lenTimer = setTimeout(lenPoll, 2000);
    } else {
      $('lentrainlog').textContent = 'lost contact with the run: ' + r.error;
    }
    return;
  }
  POLL_FAILS['lenPoll'] = 0;
  if (runOver(r)) { await lenRefresh(); return; }
  lenTimer = setTimeout(lenPoll, 900);
}

async function lenAsk() {
  $('lenaskerr').textContent = '';
  if (!lenModel) { $('lenaskerr').textContent = 'fit a threshold first - '
                                              + 'there is nothing to ask'; return; }
  const text = $('lentranscript').value.trim();
  if (!text) { $('lenaskerr').textContent = 'paste a transcript, or load one '
                                          + 'from a dataset above'; return; }
  $('lengo').disabled = true;
  $('lengo').textContent = 'Classifying…';
  const res = await api('/api/length/classify', {
    model: lenModel, transcript: text,
    threshold: $('lenaskthreshold').value,
    direction: $('lenaskdirection').value,
    strip_tags: $('lenstrip').checked,
  });
  $('lengo').disabled = false;
  $('lengo').textContent = 'Classify this call';
  if (res.error) {
    $('lenaskerr').textContent = res.error;
    $('lenanswer').innerHTML = '';
    return;
  }
  lenPaint(res);
  await lenRefresh();
}

// Where this call sits on a log scale between the two fitting distributions,
// with the line drawn through it. Lengths here run from a dozen words to
// sixty thousand, so a linear axis would put every short call on the same
// pixel.
function lenScale(a) {
  const d = a.dist || {};
  const pts = [a.words, a.threshold];
  for (const k of ['scam', 'legit'])
    if (d[k]) for (const p of ['p10', 'p50', 'p90']) pts.push(d[k][p]);
  const lo = Math.max(1, Math.min(...pts.filter(x => x > 0)) * 0.7);
  const hi = Math.max(...pts) * 1.3;
  const L = Math.log(lo), H = Math.log(hi);
  return x => 100 * (Math.log(Math.max(1, x)) - L) / Math.max(0.0001, H - L);
}

function lenPaint(a) {
  const d = a.dist || {}, at = lenScale(a);
  const span = (k, cls) => d[k] ? `
    <div class="lenband ${cls}" style="left:${at(d[k].p10)}%;
      width:${Math.max(0.6, at(d[k].p90) - at(d[k].p10))}%"></div>
    <div class="lentick ${cls}" style="left:${at(d[k].p50)}%"></div>` : '';

  const rate = a.prob_scam === null || a.prob_scam === undefined
    ? '<span class="muted">not known</span>'
    : `${(100 * a.prob_scam).toFixed(1)}%`;

  $('lenanswer').innerHTML = `
  <div class="card">
    <div class="verdict">
      <div>
        <div class="cap">Verdict</div>
        <div class="big ${a.verdict}">${a.verdict.toUpperCase()}</div>
        <div class="hint">${esc(a.direction)} than ${a.threshold} words${
          a.moved ? ' · moved from the fitted rule' : ''}</div>
      </div>
      <div>
        <div class="cap">This call</div>
        <div class="big">${a.words}</div>
        <div class="hint">words · ${a.margin > 0 ? '+' : ''}${a.margin} past
          the line</div>
      </div>
      <div style="flex:1; min-width:230px">
        <div class="cap">Of the fitting calls on this side</div>
        <div style="font-weight:600">${rate} were scams</div>
        <div class="hint">${a.side_n} calls ${a.side === 'above'
          ? 'longer than' : 'at or under'} ${a.threshold} words${
          a.stripped_tags ? ' · tone tags stripped' : ''}</div>
      </div>
    </div>
  </div>

  ${a.rate_disagrees ? `<div class="card warn">Most fitting calls on this side
    of the line were <strong>not</strong> what the rule just called this one.
    The threshold is past the point where it carries anything — the verdict is
    the rule being applied, not evidence.</div>` : ''}

  <div class="card">
    <div class="qp">Where it falls</div>
    <div class="lenaxis">
      ${span('legit', 'legit')}
      ${span('scam', 'scam')}
      <div class="lenline" style="left:${at(a.threshold)}%"></div>
      <div class="lenhere ${a.verdict}" style="left:${at(a.words)}%"></div>
    </div>
    <div class="hint" style="margin-top:10px">
      <span class="lenkey legit"></span> legitimate calls in the fitting set
        (p10–p90, median marked)${d.legit
          ? ` — median ${d.legit.p50} words` : ''}
      &nbsp;&nbsp;<span class="lenkey scam"></span> scam calls${d.scam
          ? ` — median ${d.scam.p50} words` : ''}
      &nbsp;&nbsp;<span class="lenkey line"></span> the threshold
      &nbsp;&nbsp;<span class="lenkey here"></span> this call.
      Log scale: calls here run from a dozen words to tens of thousands.</div>
  </div>

  <div class="card hint">Nothing in the call was read — not a word of it, only
    how much of it there was. models/${esc(a.model.name)} ·
    ${esc(a.model.dataset || '?')} ·
    ${a.model.swept === false ? 'threshold pinned' : 'threshold swept'} ·
    ${a.elapsed_ms} ms</div>`;
}

// ============================================================== LLM judge
// The third page. One transcript, one question, one verdict with a reason.
// It holds nothing between clicks - there is no worker to keep alive, since
// ollama is the thing holding the model.
let LLM = null, llmProf = '', llmProfs = [];
let llmFitRun = null, llmFitTimer = null;
const LFIELDS = {llmmaxtok: 'max_tokens', llmctx: 'num_ctx', llmtemp: 'temperature'};
const JFIELDS = {llmshots: 'shots', llmshotwords: 'shot_words',
                 llmholdcalls: 'holdout_calls', llmfitseed: 'seed'};

async function llmBoot() {
  const cfg = await api('/api/llm/config');
  if (cfg.error) { $('llmerr').textContent = cfg.error; return; }
  LLM = cfg;

  $('llmhost').textContent = 'ollama at ' + cfg.host;
  for (const [id, key] of Object.entries(LFIELDS))
    if (cfg.defaults[key] !== undefined) $(id).value = cfg.defaults[key];

  if (cfg.ollama_error) {
    $('llmerr').textContent = cfg.ollama_error;
    $('llmmodel').innerHTML = '<option value="">nothing to choose from</option>';
  } else if (!cfg.models.length) {
    $('llmerr').textContent = 'ollama is running but has no models pulled — '
      + 'try: ollama pull qwen2.5:14b';
    $('llmmodel').innerHTML = '<option value="">nothing pulled</option>';
  } else {
    // the configured model first if it is there, so the page opens on the one
    // the rest of the project uses
    const names = cfg.models.map(m => m.name);
    if (names.includes(cfg.default_model))
      names.splice(names.indexOf(cfg.default_model), 1),
      names.unshift(cfg.default_model);
    $('llmmodel').innerHTML = names.map(n =>
      `<option value="${esc(n)}">${esc(n)}</option>`).join('');
  }

  $('llmds').innerHTML = cfg.datasets.map(d =>
    `<option value="${esc(d.path)}">${esc(d.name)}</option>`).join('');
  $('llmload').onclick = llmLoadRow;
  forgetRowOnEdit('llmtranscript', 'llminfo');
  $('llmgo').onclick = llmAsk;
  $('llmguideclear').onclick = () => { $('llmguidance').value = ''; llmSize(); };
  for (const id of ['llmtranscript', 'llmguidance', 'llmctx'])
    $(id).addEventListener('input', llmSize);

  // the fitting side
  $('llmfitds').innerHTML = cfg.datasets.map(d =>
    `<option value="${esc(d.path)}">${esc(d.name)} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  for (const [id, key] of Object.entries(JFIELDS))
    if (cfg.fit_defaults[key] !== undefined) $(id).value = cfg.fit_defaults[key];
  $('llmfitgo').onclick = llmFit;
  for (const id of ['llmshots', 'llmholdcalls', 'llmrubric'])
    $(id).addEventListener('input', llmFitCost);
  $('llmeds').innerHTML = cfg.datasets.map(d =>
    `<option value="${esc(d.path)}">${esc(d.name)} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  $('llmevlimit').addEventListener('input', llmEvalCost);
  $('llmeds').addEventListener('change', llmEvalCost);
  $('llmevgo').onclick = () => evalRun('llm', evalIds('llm', 'j-eval'), {
    model: llmProf, base_model: $('llmmodel').value,
    dataset: $('llmeds').value, limit: $('llmevlimit').value,
    num_ctx: $('llmctx').value, max_tokens: $('llmmaxtok').value,
    temperature: $('llmtemp').value,
  });
  llmEvalCost();
  for (const b of $('llmtabs').querySelectorAll('button'))
    b.onclick = () => llmTab(b.dataset.jtab);
  llmFitCost();
  await llmProfiles();
  llmSize();
}

function llmTab(name) {
  for (const b of $('llmtabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.jtab === name);
  for (const t of ['ask', 'eval', 'fit', 'about'])
    $('j-' + t).hidden = t !== name;
}

// What a fit is going to cost, before it is started rather than after. Two
// calls per held-out call is the honest price of knowing whether the fitted
// prompt beat the control, and it is not obvious from the form.
function llmFitCost() {
  const hold = parseInt($('llmholdcalls').value, 10) || 0;
  const n = hold * 2 + ($('llmrubric').checked ? 1 : 0);
  $('llmfitcost').innerHTML = n
    ? `about <strong>${n}</strong> calls to the model: ${hold} held-out `
      + `call${hold === 1 ? '' : 's'} scored twice, fitted and bare`
      + ($('llmrubric').checked ? ', plus one to write the rubric' : '')
      + '. On a 14B model that is minutes, not seconds.'
    : 'no holdout — the prompt will be saved unmeasured, which means you will '
      + 'not know whether it beat the bare one.';
}

// The one evaluate that has to be budgeted rather than just started. One
// generation per call, and the datasets here run to seven thousand.
function llmEvalCost() {
  const opt = $('llmeds').selectedOptions[0];
  const rows = opt ? parseInt((opt.textContent.match(/([0-9]+) rows/) || [])[1], 10) : 0;
  const asked = parseInt($('llmevlimit').value, 10) || 0;
  const n = asked ? Math.min(asked, rows || asked) : rows;
  if (!n) { $('llmevcost').textContent = ''; return; }
  // 6s/call is a 14B model on a GPU with a short transcript; long calls and
  // CPU are both far worse, so this is the optimistic end and is labelled so
  const secs = n * 6;
  const h = secs >= 3600 ? (secs / 3600).toFixed(1) + ' hours'
          : secs >= 60 ? Math.round(secs / 60) + ' minutes'
          : secs + ' seconds';
  $('llmevcost').innerHTML = `<strong>${n}</strong> generation`
    + (n === 1 ? '' : 's') + ` — at best about ${h} on a 14B model with the `
    + `GPU, and a good deal longer on CPU or with long calls`;
}

async function llmProfiles() {
  const r = await api('/api/llm/profiles');
  if (r.error) { $('llmproferr').textContent = r.error; return; }
  llmProfs = r.profiles || [];
  if (!llmProfs.some(p => p.name === llmProf)) llmProf = '';
  const row = (name, label, sub, on) => `<div class="row">
      <a href="#llm" data-prof="${esc(name)}" class="${on ? 'on' : ''}">
        <strong>${esc(label)}</strong>
        <span class="muted">${sub}</span>
      </a>${name ? `<button class="link" data-profdel="${esc(name)}">delete</button>`
                 : ''}</div>`;

  let html = row('', 'None — the bare control', 'the transcript and the '
    + 'question, nothing else. The only setting comparable with a benchmark '
    + 'row.', !llmProf);
  html += llmProfs.map(p => {
    const acc = p.holdout && p.holdout.acc;
    const gain = p.gain;
    const bits = [esc(p.dataset || ''), (p.shots || 0) + ' example'
      + (p.shots === 1 ? '' : 's')];
    if (p.rubric) bits.push('rubric');
    if (p.guided) bits.push('instructions');
    if (acc !== undefined && acc !== null)
      bits.push(`holdout ${(100 * acc).toFixed(1)}%`
        + (gain === null || gain === undefined ? ''
           : `, <strong class="${gain > 0 ? 'gain-up' : 'gain-down'}">`
             + `${gain > 0 ? '+' : ''}${(100 * gain).toFixed(1)} vs control</strong>`));
    return row(p.name, p.name, bits.join(' · '), p.name === llmProf);
  }).join('');
  $('llmprofiles').innerHTML = html;

  for (const a of $('llmprofiles').querySelectorAll('a'))
    a.onclick = e => { e.preventDefault(); llmProf = a.dataset.prof; llmProfiles(); };
  for (const b of $('llmprofiles').querySelectorAll('[data-profdel]'))
    b.onclick = async () => {
      if (!confirm('Delete models/' + b.dataset.profdel + '?')) return;
      const r = await api('/api/llm/delete_profile', {name: b.dataset.profdel});
      if (r.error) { $('llmproferr').textContent = r.error; return; }
      await llmProfiles();
    };

  const p = llmProfs.find(x => x.name === llmProf);
  $('llmtitle').textContent = p ? p.name : 'Bare llm_only control';
  $('llmsub').innerHTML = p
    ? [esc(p.dataset || '?'), (p.shots || 0) + ' worked example'
        + (p.shots === 1 ? '' : 's'),
       p.rubric ? 'a rubric' : null,
       p.guided ? 'standing instructions' : null,
       p.prompt_tokens ? '~' + p.prompt_tokens + ' extra tokens per call' : null,
       (p.gain === null || p.gain === undefined) ? null
         : `${p.gain > 0 ? '+' : ''}${(100 * p.gain).toFixed(1)} points against `
           + `the control on ${p.holdout_rows} held-out calls`,
      ].filter(Boolean).join(' · ')
    : 'No fitted prompt — the transcript and the question, exactly as the '
      + 'benchmark asks it.';
}

async function llmFit() {
  $('llmfiterr').textContent = '';
  const model = $('llmmodel').value;
  if (!model) { $('llmfiterr').textContent = 'no model to fit against - pull '
                                           + 'one with ollama first'; return; }
  const body = {name: $('llmfitname').value, dataset: $('llmfitds').value,
                model: model, rubric: $('llmrubric').checked,
                overwrite: $('llmfitover').checked,
                num_ctx: $('llmctx').value,
                guidance: $('llmfitguide').checked ? $('llmguidance').value : ''};
  for (const [id, key] of Object.entries(JFIELDS)) body[key] = $(id).value;
  const res = await api('/api/llm/fit', body);
  if (res.error) { $('llmfiterr').textContent = res.error; return; }
  llmFitRun = res.id;
  llmTab('fit');
  llmFitPoll();
}

async function llmFitPoll() {
  if (!llmFitRun) return;
  const r = await runLog(llmFitRun);
  if (!r.error) $('llmfitlog').textContent = r.text || '(no output yet)';
  clearTimeout(llmFitTimer);
  // A failed poll is not a finished run. Retry a few times before giving up,
  // rather than concluding the run ended because one request did not land.
  if (r.error) {
    if ((POLL_FAILS['llmFitPoll'] = (POLL_FAILS['llmFitPoll'] || 0) + 1) < 5) {
      llmFitTimer = setTimeout(llmFitPoll, 2000);
    } else {
      $('llmfitlog').textContent = 'lost contact with the run: ' + r.error;
    }
    return;
  }
  POLL_FAILS['llmFitPoll'] = 0;
  if (runOver(r)) { await llmProfiles(); return; }
  llmFitTimer = setTimeout(llmFitPoll, 1500);
}

const wordsIn = id => $(id).value.trim().split(/\s+/).filter(Boolean).length;

// The context window is the failure people cannot see: over it, ollama cuts
// the front of the prompt off and the instructions go with it. So the size is
// on screen before the model is asked, not explained afterwards. The standing
// instructions count towards it too - they are part of every prompt.
// Mirrors ollama_ctx.estimate_tokens: the larger of 1.4 per word and one per
// four characters, since these transcripts are full of short words, digits
// and names that split into several tokens each. The 50 and 81 are the
// prompt's own boilerplate - the question, the answer format, and the fence
// the instructions go in - so the number here is the number the reply comes
// back with.
function llmSize() {
  const words = wordsIn('llmtranscript'), guide = wordsIn('llmguidance');
  const chars = $('llmtranscript').value.length + $('llmguidance').value.length;
  const boiler = guide ? 81 : 50;
  const est = Math.max(Math.floor((words + guide + boiler) * 1.4),
                       Math.floor((chars + boiler * 6) / 4)) + 1;
  const ctx = parseInt($('llmctx').value, 10) || 0;
  const over = ctx && est > ctx;
  $('llmsize').innerHTML = words
    ? `${words} words${guide ? ' + ' + guide + ' of instructions' : ''}`
      + ` · about ${est} tokens${ctx ? ' of ' + ctx : ''}`
      + (over ? ' — <strong style="color:var(--bad)">over the window, raise it'
              + ' or the instructions get cut off</strong>' : '')
    : '';
  $('llmguidesize').textContent = guide
    ? `${guide} words, sent with every question from now on`
    : 'none — the model judges on the call alone';
}

async function llmLoadRow() {
  $('llminfo').textContent = 'loading…';
  const r = await api(`/api/dataset/sample?dataset=${encodeURIComponent($('llmds').value)}`
                    + `&idx=${encodeURIComponent($('llmidx').value || 0)}`);
  if (r.error) { $('llminfo').textContent = r.error; return; }
  $('llmtranscript').value = r.text;
  $('llmidx').value = r.idx;
  $('llminfo').innerHTML = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + esc(r.row_id) : '') + truthPill(r.label);
  llmSize();
}

async function llmAsk() {
  $('llmasker').textContent = '';
  const text = $('llmtranscript').value.trim();
  if (!text) { $('llmasker').textContent = 'paste a transcript, or load one '
                                         + 'from a dataset above'; return; }
  const model = $('llmmodel').value;
  if (!model) { $('llmasker').textContent = 'no model to ask - pull one with '
                                          + 'ollama first'; return; }
  $('llmgo').disabled = true;
  $('llmgo').textContent = 'Asking…';
  $('llmanswer').innerHTML = '<div class="card muted">' + esc(model)
    + ' is reading the call… a 14B model takes a few seconds on a GPU and '
    + 'rather longer on a CPU</div>';
  const body = {model: model, transcript: text, profile: llmProf,
                guidance: $('llmguidance').value};
  for (const [id, key] of Object.entries(LFIELDS)) body[key] = $(id).value;
  const res = await api('/api/llm/judge', body);
  $('llmgo').disabled = false;
  $('llmgo').textContent = 'Ask the model';
  if (res.error) {
    $('llmasker').textContent = res.error;
    $('llmanswer').innerHTML = '';
    return;
  }
  paintVerdict(res);
}

function paintVerdict(r) {
  const cls = r.unreadable ? 'uncertain' : (r.scam ? 'scam' : 'legitimate');
  const word = r.unreadable ? 'UNREADABLE' : (r.scam ? 'SCAM' : 'LEGITIMATE');
  const notes = [];
  if (r.profile) notes.push('Judged under the fitted prompt <strong>models/'
    + esc(r.profile) + '</strong> — ' + r.shots + ' worked example'
    + (r.shots === 1 ? '' : 's') + (r.rubric ? ' and a rubric' : '')
    + ' went in ahead of the call. Nothing was fine-tuned: that is text in the '
    + 'prompt, not a change to the model. This is not the <code>llm_only</code> '
    + 'control — pick <em>None</em> on the left for that.');
  if (r.guided) notes.push('Judged under your standing instructions, so this is '
    + 'not the <code>llm_only</code> control any more — it is the model doing '
    + 'what you told it. Clear the box to get the unguided verdict back.');
  if (r.over_context) notes.push('The prompt was about ' + r.prompt_tokens_estimated
    + ' tokens and the window ' + r.num_ctx + '. Ollama drops the front of an '
    + 'overlong prompt — the instructions with it — so this verdict may be an '
    + 'answer to a headless transcript. Raise the context window and ask again.');
  if (r.prompt_truncated) notes.push('<strong>Ollama read only '
    + r.prompt_tokens_read + ' of the ~' + r.prompt_tokens_estimated
    + ' tokens sent</strong> — ' + Math.round(100 - (r.prompt_kept_pct || 0))
    + '% of this call was thrown away, from the beginning, so the verdict is '
    + 'about whatever was left. On a long call that is the goodbyes. This is '
    + "Ollama's own count, not an estimate. Note it read about half the "
    + r.num_ctx + '-token window rather than all of it: a window merely close '
    + 'to the prompt size is no use, it has to exceed it.');
  if (r.prompt_truncated && r.profile) notes.push('Because the prompt was '
    + 'truncated, the worked examples were the first thing to go — they sit at '
    + 'the front for exactly that reason. This verdict is closer to the bare '
    + 'control than to the fitted prompt you picked.');
  if (r.truncated) notes.push('The reply looks cut off. Raise the reply tokens.');
  if (r.retried) notes.push('The first reply could not be read, so the model was '
    + 'asked again for a single word. The reason below is from the first reply.');
  if (r.unreadable) notes.push('Neither reply gave a verdict this page can read. '
    + 'The benchmark would have scored this call Normal; it is shown as '
    + 'unreadable here rather than counted as legitimate.');

  $('llmanswer').innerHTML = `
  <div class="card">
    <div class="verdict">
      <div>
        <div class="cap">Verdict</div>
        <div class="big ${cls}">${word}</div>
        <div class="hint">${esc(r.model)}</div>
      </div>
      <div style="flex:1; min-width:240px">
        <div class="cap">Reason the model gave</div>
        <div style="font-weight:600">${esc(r.reason)}</div>
      </div>
    </div>
  </div>
  ${notes.map(n => `<div class="card hint">${n}</div>`).join('')}
  <div class="card">
    <details class="adv" style="margin-bottom:0">
      <summary>what the model actually replied</summary>
      <div class="advbody"><div class="ev" style="font-style:normal; white-space:pre-wrap">${
        esc(r.raw || '(nothing)')}</div></div>
    </details>
  </div>
  <div class="card hint">${r.words} words · about ${r.prompt_tokens_estimated}
    prompt tokens of ${r.num_ctx}${r.prompt_tokens_read
      ? ` · ollama read ${r.prompt_tokens_read} (${r.prompt_kept_pct}%)` : ''}
    · ${r.elapsed_ms} ms</div>`;
}

boot();
</script>
</body>
</html>
"""


def lan_address():
    """The address other machines can reach this box on.

    Asks the routing table which local interface would be used to get out,
    which is the one a colleague on the same network will come in on. The
    UDP socket is never actually sent anything.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return None
    finally:
        s.close()


# ------------------------------------------------------ stop and update
# Two buttons in the header. Both act on this server process only: runs are
# detached into their own sessions and outlive it, as they always have.
#
# Restarting is done by re-executing this same process (os.execv) rather than
# by exiting and letting web_ui.sh start a new one. The PID stays the same, so
# web_ui.sh keeps waiting on it and never runs its cleanup - the public tunnel
# stays up on the same address, and is only missing a server for the second
# or two the new one takes to bind. The listening socket is not inherited
# across exec (Python sockets are non-inheritable), so the new image binds the
# port afresh; the BERT worker pipes are not inherited either, and the workers
# are unloaded first so they do not linger.
ADMIN_LOCK = threading.Lock()


def git(*args, timeout=60):
    out = subprocess.run(["git"] + list(args), cwd=str(PROJECT_DIR),
                         capture_output=True, text=True, timeout=timeout)
    return out.returncode, (out.stdout + out.stderr).strip()


def version_info():
    rc, head = git("log", "-1", "--format=%h %s", timeout=10)
    rc2, branch = git("rev-parse", "--abbrev-ref", "HEAD", timeout=10)
    return {"commit": head if rc == 0 else "unknown",
            "branch": branch if rc2 == 0 else "unknown"}


def _after_reply(fn, delay=0.8):
    """Run fn a moment from now, so the reply that asked for it gets out."""
    def go():
        time.sleep(delay)
        fn()
    threading.Thread(target=go, daemon=True).start()


def _restart_in_place():
    for slot in SLOTS:
        try:
            slot.unload()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    # -u because web_ui.sh starts it unbuffered and sys.argv does not carry
    # interpreter flags - without it the restarted server's output sits in a
    # buffer instead of reaching the terminal or the tmux log
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)


def _stop_now():
    for slot in SLOTS:
        try:
            slot.unload()
        except Exception:
            pass
    sys.stderr.write("stopped from the web UI\n")
    sys.stderr.flush()
    # os._exit rather than sys.exit: this runs on a helper thread, and the
    # server's own loop would otherwise keep the process alive. web_ui.sh sees
    # its server go, closes the tunnel and exits - and with it the tmux session.
    os._exit(0)


def admin_stop():
    running = [r["id"] for r in all_runs() if r["status"] == "running"]
    _after_reply(_stop_now)
    return {"stopping": True, "runs_still_going": running}


def admin_update():
    """git pull --ff-only, and restart into the new code if anything came in.

    Refused while a run is going: run_all.sh is read by bash as it executes,
    so rewriting it under a live run can break that run part way through.
    """
    if not ADMIN_LOCK.acquire(blocking=False):
        raise ValueError("an update is already in progress")
    try:
        running = [r["id"] for r in all_runs() if r["status"] == "running"]
        if running:
            raise ValueError(
                "a run is going (%s). Updating now would change run_all.sh "
                "under it, and bash reads that file as it goes - wait for it "
                "to finish, or stop it, then update." % running[0])
        before = version_info()["commit"]
        rc, fetched = git("pull", "--ff-only", timeout=120)
        if rc != 0:
            raise ValueError("git pull failed, nothing was changed:\n" + fetched)
        after = version_info()["commit"]
        changed = before.split()[0] != after.split()[0]
        log = ""
        if changed:
            _, log = git("log", "--format=%h %s",
                         "%s..%s" % (before.split()[0], after.split()[0]))
            _, files = git("diff", "--name-only", before.split()[0],
                           after.split()[0])
            shell_changed = "web_ui.sh" in files.split()
            _after_reply(_restart_in_place)
        else:
            shell_changed = False
        return {"updated": changed, "before": before, "after": after,
                "output": fetched, "commits": log,
                "restarting": changed,
                # web_ui.sh itself is the one file a restart does not reload
                "web_ui_sh_changed": shell_changed}
    finally:
        ADMIN_LOCK.release()


def main():
    ap = argparse.ArgumentParser(description="browser front end for run_all.sh")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0",
                    help="interface to bind (default 0.0.0.0: every one)")
    ap.add_argument("--local", action="store_true",
                    help="bind 127.0.0.1 only - this machine, or an SSH tunnel")
    args = ap.parse_args()
    if args.local:
        args.host = "127.0.0.1"

    if not RUN_ALL.exists():
        sys.exit("run_all.sh not found next to scripts/ - run this from the repo")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    # lets go of a BERT nobody is asking questions of any more
    threading.Thread(target=worker_reaper, daemon=True).start()

    if os.environ.get("SCAM_UI_RESTARTED"):
        print("\nrestarted into %s" % version_info()["commit"])
    os.environ["SCAM_UI_RESTARTED"] = "1"
    print("scam-detection UI")
    print("  here:           http://localhost:%d" % args.port)
    if args.host == "127.0.0.1":
        print("  local only - other machines cannot reach it")
        print("  over SSH:      ssh -L %d:localhost:%d %s@<this host>"
              % (args.port, args.port, os.environ.get("USER", "you")))
    else:
        ip = lan_address()
        if ip:
            print("  other machines: http://%s:%d" % (ip, args.port))
        # anyone who can reach the port can start and stop runs on this box
        print("  open to the network - run with --local to keep it to this machine")
    print("  ctrl-c stops the server (runs already going are not affected)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped")


if __name__ == "__main__":
    main()
