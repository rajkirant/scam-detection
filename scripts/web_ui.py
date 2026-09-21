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

Two pages:

  Benchmark   the form above, the output of a run, its results table and its
              prediction for every call.
  BERT + MCQ  fine-tune a BERT on one of the datasets and keep the
              checkpoint, then put knowledge/mcq_ontology.json to it one
              transcript at a time - the same question set the mcq baseline
              puts to the LLM, answered instead by a model trained on this
              data. scripts/bert_mcq.py does the work; this is its front end.

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
    ("trivial",  "length + bag-of-words", "trivial references, no LLM",           False),
    ("llm_only", "LLM-only",              "the model decides alone, no retrieval", True),
    ("singh",    "Singh",                 "policy-compliance baseline",           True),
    ("webrag",   "Web-RAG",               "KB-only retrieval",                    True),
    ("qwen_kb",  "Qwen-KB",               "learns a KB from a held-out split, k-fold", True),
    ("hybrid",   "Hybrid",                "Web-RAG + Qwen-KB over one shared KB",  True),
    ("ontology", "Ontology RAG",          "scam_ontology.json",                   True),
    ("mcq",      "MCQ ontology",          "mcq_ontology.json, 2 calls per transcript", True),
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
            "limit": limit, "model": model, "started": time.time()}
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


# --------------------------------------------------------------- BERT + MCQ
# The second page. Fine-tune a BERT on one of the datasets, keep the
# checkpoint, then make that checkpoint answer knowledge/mcq_ontology.json for
# a single transcript - the same question set the mcq baseline puts to the LLM,
# put instead to a model trained on this data.
#
# bert_mcq is imported rather than shelled out to for the cheap questions -
# listing checkpoints, reading the ontology. Its module level is standard
# library only (torch and transformers are imported inside the functions that
# need them), so this server still starts on a machine with neither installed.
import bert_mcq

MCQ_SCRIPT = PROJECT_DIR / "scripts" / "bert_mcq.py"
MODELS_DIR = PROJECT_DIR / "models"
MCQ_NAME_RE = bert_mcq.NAME_RE

# What training can start from. Any model on the Hub works from the command
# line; the menu offers the four worth comparing that fit on one GPU.
BASE_MODELS = [
    ("bert-base-uncased", "the thesis baseline, 110M parameters"),
    ("distilbert-base-uncased", "40% smaller, about twice as fast, ~1 point behind"),
    ("roberta-base", "better pre-training, same size as BERT"),
    ("albert-base-v2", "12M parameters, but slower per epoch than DistilBERT"),
]

# Bounds on the training form. Each is (flag, cast, low, high, default) and the
# defaults are bert_mcq.py's own, repeated here only so the form can show them.
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
ANSWER_FIELDS = {
    "window":         ("--window", int, 10, 400, 45),
    "stride":         ("--stride", int, 5, 400, 15),
    "max_length":     ("--max-length", int, 64, 512, 256),
    "min_confidence": ("--min-confidence", float, 0.0, 0.95, 0.30),
    "min_margin":     ("--min-margin", float, 0.0, 0.95, 0.04),
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
    """One `bert_mcq.py serve` process, kept alive between questions.

    Loading a checkpoint takes seconds and answering it takes a fraction of
    one, so the model stays in memory between clicks rather than being loaded
    per question. It answers on the CPU unless the page asks otherwise: a
    benchmark run wants the whole GPU, and this page is meant to stay usable
    while one is going.
    """

    def __init__(self, name, opts):
        self.name, self.opts = name, opts
        self.lock = threading.Lock()
        self.last = time.time()
        self.lines = queue.Queue()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        # transformers writes progress bars and load reports to stderr; they
        # go to a file so they neither fill the pipe nor reach the replies
        self.errlog = open(RUNS_DIR / "mcq_worker.log", "ab", buffering=0)
        argv = ([venv_python(), "-u", str(MCQ_SCRIPT), "serve", "--name", name]
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
                             "results/logs/web/mcq_worker.log" % timeout)
        if not line.strip():
            raise ValueError("the answerer stopped - see "
                             "results/logs/web/mcq_worker.log")
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
WORKER = None
WORKER_LOCK = threading.Lock()


def answerer_for(name, opts):
    """The live answerer for that checkpoint, started - or restarted - if the
    model or the settings it was loaded with have changed."""
    global WORKER
    with WORKER_LOCK:
        if WORKER is not None and (WORKER.name != name or WORKER.opts != opts
                                   or WORKER.proc.poll() is not None):
            WORKER.close()
            WORKER = None
        if WORKER is None:
            WORKER = Answerer(name, opts)
        return WORKER


def unload_answerer():
    global WORKER
    with WORKER_LOCK:
        if WORKER is None:
            return {"unloaded": False, "note": "nothing was loaded"}
        name = WORKER.name
        WORKER.close()
        WORKER = None
        return {"unloaded": True, "model": name}


def worker_state():
    w = WORKER
    if w is None or w.proc.poll() is not None:
        return {"loaded": False}
    return {"loaded": True, "model": w.name, "device": w.device,
            "idle_s": int(time.time() - w.last), "idle_limit": WORKER_IDLE}


def worker_reaper():
    while True:
        time.sleep(30)
        w = WORKER
        if w is not None and time.time() - w.last > WORKER_IDLE:
            unload_answerer()


def mcq_config():
    """Everything the BERT + MCQ page needs to draw itself once."""
    onto = bert_mcq.load_ontology()
    return {
        "datasets": datasets(),
        "bases": [{"id": k, "note": n} for k, n in BASE_MODELS],
        "models": bert_mcq.list_models(),
        "branches": [{"id": o["id"], "text": o.get("text", ""),
                      "questions": len(o.get("questions", []))}
                     for o in onto["options"]],
        "ontology": {"prompt": onto.get("prompt", ""),
                     "bands": onto.get("bands"),
                     "scoring": onto.get("scoring")},
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
    with open(PROJECT_DIR / path, newline="", encoding="utf-8",
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


def mcq_answer(form):
    """Put the ontology to one checkpoint for one transcript."""
    name = str(form.get("model", "")).strip()
    if not MCQ_NAME_RE.match(name):
        raise ValueError("pick a trained model first")
    if not (MODELS_DIR / name / "config.json").exists():
        raise ValueError("models/%s is not a trained checkpoint" % name)
    text = (form.get("transcript") or "").strip()
    if not text:
        raise ValueError("paste a transcript, or load one from a dataset")
    if len(text) > 400_000:
        raise ValueError("that is longer than any call in the datasets - "
                         "paste one call, not a whole file")
    branch = str(form.get("branch") or "auto")
    try:
        cutoff = float(form.get("cutoff") or 0.0)
    except (TypeError, ValueError):
        raise ValueError("the scam cut-off must be a number")

    opts = []
    for key in ANSWER_FIELDS:
        val = numeric(form, ANSWER_FIELDS, key)
        if val is not None:
            opts += [ANSWER_FIELDS[key][0], str(val)]
    if form.get("gpu"):
        # the one thing that would actually fight a benchmark for VRAM
        if any(r["status"] == "running" for r in all_runs()):
            raise ValueError("a run is going, and answering on the GPU would "
                             "fight it for VRAM. Untick \"answer on the GPU\", "
                             "or stop the run first.")
        opts.append("--gpu")
    # The comparison the thesis wants: the same questions matched in the
    # checkpoint's own hidden states rather than in a sentence-similarity
    # space. Worth running once to see the difference; not the default,
    # because a binary classification objective never built a space that can
    # tell one option from another.
    if form.get("self_encoder"):
        opts += ["--encoder", "self"]
    if form.get("raw_text"):
        opts.append("--raw")

    out = answerer_for(name, opts).ask(
        {"transcript": text, "branch": branch, "cutoff": cutoff})
    if not out.get("ok"):
        raise ValueError(out.get("error") or "the answerer could not answer that")
    return out


# ---------------------------------------------------------------- LLM judge
# The third page, and the simplest thing in the project: no retrieval, no
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

# (kwarg, cast, low, high, default) - the same shape numeric() reads for the
# other two pages.
LLM_FIELDS = {
    "max_tokens":  ("max_tokens", int, 32, 4000, llm_judge.DEFAULT_MAX_TOKENS),
    "num_ctx":     ("num_ctx", int, 512, 131072, llm_judge.DEFAULT_NUM_CTX),
    "temperature": ("temperature", float, 0.0, 2.0, 0.0),
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
    if not LLM_LOCK.acquire(blocking=False):
        raise ValueError("the model is already answering something - one call "
                         "at a time, or they fight for the VRAM")
    try:
        return llm_judge.judge(text, model=model, timeout=LLM_TIMEOUT,
                               guidance=guidance, **kw)
    except RuntimeError as e:
        raise ValueError(str(e))
    finally:
        LLM_LOCK.release()


def remove_model(name):
    """Delete one checkpoint. Several hundred megabytes each, so the page asks
    first and this says exactly what went."""
    name = (name or "").strip()
    if not MCQ_NAME_RE.match(name):
        raise ValueError("bad model name")
    d = MODELS_DIR / name
    if not d.is_dir() or d.resolve().parent != MODELS_DIR.resolve():
        raise ValueError("no such model: %s" % name)
    if WORKER is not None and WORKER.name == name:
        unload_answerer()           # cannot delete a checkpoint that is open
    shutil.rmtree(d)
    return {"deleted": True, "id": name}


def start_train_run(form):
    """Fine-tune a checkpoint. Detached and logged like every other run, so it
    appears in Recent runs and can be stopped with the same button."""
    name = str(form.get("name", "")).strip()
    if not MCQ_NAME_RE.match(name):
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
        if WORKER is not None and WORKER.name == name:
            unload_answerer()

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
        '  "$PY" -u scripts/bert_mcq.py ' + quoted
        + ' || { printf "  fail train\\n"; return 1; }',
        '  printf "  ok train\\n"',
        "}", "train",
        'printf "\\n%s %%s\\n" "$?"' % EXIT_MARK,
    ]

    env = dict(os.environ)
    env["TERM"] = "dumb"

    with open(log, "wb") as out:
        out.write(("$ python scripts/bert_mcq.py " + quoted + "\n\n").encode())
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
            if u.path == "/api/runs":
                return self._send(200, {"runs": all_runs()})
            # "output", not "log": privacy filter lists block paths that
            # look like telemetry, and Edge has tracking prevention on by
            # default - these two endpoints were the only ones carrying the
            # word "log", and the only ones that never arrived there. The old
            # paths stay as aliases so a page left open somewhere still works.
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
            # ---- the BERT + MCQ page
            if u.path == "/api/mcq/config":
                return self._send(200, mcq_config())
            if u.path == "/api/mcq/models":
                return self._send(200, {"models": bert_mcq.list_models(),
                                        "worker": worker_state()})
            if u.path == "/api/mcq/sample":
                return self._send(200, dataset_row(
                    q.get("dataset", [""])[0], int(q.get("idx", ["0"])[0])))
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
            # ---- the BERT + MCQ page
            if u.path == "/api/mcq/train":
                meta = start_train_run(form)
                sys.stderr.write("started %s  train %s on %s\n"
                                 % (meta["id"], meta["model"], meta["dataset"]))
                return self._send(200, meta)
            if u.path == "/api/mcq/answer":
                return self._send(200, mcq_answer(form))
            if u.path == "/api/mcq/unload":
                return self._send(200, unload_answerer())
            if u.path == "/api/mcq/delete_model":
                out = remove_model(form.get("name", ""))
                sys.stderr.write("deleted checkpoint %s\n" % out["id"])
                return self._send(200, out)
            # ---- the LLM judge page
            if u.path == "/api/llm/judge":
                return self._send(200, llm_verdict(form))
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": str(e)})


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

  /* ---- BERT + MCQ ---- */
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

  /* the headline: what the ontology made of the call, and what the trained
     head made of it, side by side - they are different claims and the page
     should never let them be read as one number */
  .verdict { display:flex; gap:30px; flex-wrap:wrap; align-items:flex-end; }
  .big { font-size:30px; font-weight:700; line-height:1.05; letter-spacing:-.02em;
         font-variant-numeric:tabular-nums; }
  .big.scam { color:var(--bad); }
  .big.legitimate { color:var(--accent); }
  .big.uncertain { color:var(--warn); }
  .cap { font-size:11px; font-weight:700; letter-spacing:.08em; margin-bottom:5px;
         text-transform:uppercase; color:var(--dim); }

  /* one answered question */
  .q { border-top:1px solid var(--line); padding:14px 0; }
  .q:first-child { border-top:none; padding-top:0; }
  .q:last-child { padding-bottom:0; }
  .qp { font-weight:600; margin-bottom:8px; }
  .qa { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .qa .pick { flex:1; min-width:220px; }
  .chip { font-size:12px; font-weight:700; padding:2px 9px; border-radius:99px;
          border:1px solid var(--line); font-variant-numeric:tabular-nums;
          white-space:nowrap; }
  .chip.pos  { color:var(--bad); border-color:var(--bad); }
  .chip.neg  { color:var(--accent); border-color:var(--accent); }
  .chip.zero { color:var(--dim); }
  .bar { height:5px; background:var(--line); border-radius:99px; margin-top:9px;
         overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--accent); border-radius:99px; }
  .bar.low i { background:var(--warn); }
  /* the stretch of transcript the chosen option actually matched against -
     without it an answer is a claim with nothing behind it */
  .ev { margin-top:9px; color:var(--dim); font-size:12.5px; font-style:italic;
        border-left:2px solid var(--line); padding-left:11px; }
  table.opts td, table.opts th { font-size:12.5px; }
  table.opts td.optname { text-align:left; white-space:normal; font-family:inherit; }
  table.opts tr.chosen td { color:var(--ink); font-weight:700; }
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
    <button data-page="mcq">BERT + MCQ</button>
    <button data-page="llm">LLM judge</button>
  </nav>
  <span class="sub" id="ollama">checking Ollama…</span>
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
        <div class="hint">How many tokens the model may read. Ollama cuts an
          overlong prompt from the <em>front</em> — instructions first — and
          keeps only about half the window, so a call that does not fit comes
          back as a confident verdict on its last few minutes. Leave it blank
          for 8192. <code>scamai_hard_subset.csv</code> needs 16384 to cover
          97% of its calls and 32768 for all but one; both cost VRAM on top of
          the model.</div>
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
      <button data-tab="results">Results</button>
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

<!-- ==================== page two: BERT + MCQ ==================== -->
<div class="wrap" id="page-mcq" hidden>
  <div class="side">
    <div class="sect">
      <div class="secthead">Trained models</div>
      <div class="hint" style="margin-top:0">the checkpoint the questions are
        put to · hover to delete one</div>
      <div class="hist" id="mcqmodels"></div>
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
        <strong id="mcqtitle">No model selected</strong>
        <span class="pill" id="mcqpill" hidden></span>
        <span style="flex:1"></span>
        <button class="stop" id="trainstop" hidden>Stop</button>
      </div>
      <div class="hint" id="mcqsub">Train one on the left, then put the
        ontology's questions to it.</div>
    </div>

    <div class="tabs" id="mcqtabs">
      <button data-mtab="ask" class="on">Ask</button>
      <button data-mtab="train">Training output</button>
      <button data-mtab="about">How it answers</button>
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
          <label class="inline" for="branch">Branch
            <select id="branch"></select></label>
          <label class="inline" for="cutoff">Scam cut-off
            <input type="text" id="cutoff" class="num" value="0"></label>
          <label class="inline"><input type="checkbox" id="agpu">
            answer on the GPU</label>
        </div>
        <details class="adv">
          <summary>How the options are matched</summary>
          <div class="advbody">
            <div class="grid2">
              <div><label for="awindow">Window (words)</label>
                   <input type="text" id="awindow" style="width:100%"></div>
              <div><label for="astride">Stride (words)</label>
                   <input type="text" id="astride" style="width:100%"></div>
              <div><label for="amaxlen">Tokens per window</label>
                   <input type="text" id="amaxlen" style="width:100%"></div>
              <div><label for="aminconf">Abstain below</label>
                   <input type="text" id="aminconf" style="width:100%"></div>
              <div><label for="aminmargin">Least margin</label>
                   <input type="text" id="aminmargin" style="width:100%"></div>
            </div>
            <div class="hint">The transcript is cut into overlapping windows and
              an option scores its best match against any one of them. The
              margin is how far the best option is clear of the runner-up, in
              raw cosine; under it the question abstains rather than picking
              between scores that are the same number twice. Changing any of
              these reloads the model, so the next answer is slower.</div>
            <div style="margin-top:11px">
              <label class="inline"><input type="checkbox" id="aself">
                match in the checkpoint's own hidden states</label>
              <div class="hint">Off by default. Fine-tuning fits a binary
                scam/legitimate head and never asks the encoder to tell "a
                courier" from "a customs agency", so matching there gives
                every option nearly the same score and the winner is decided
                by noise. Tick it to see that happen.</div>
            </div>
            <div style="margin-top:9px">
              <label class="inline"><input type="checkbox" id="araw">
                match against the raw transcript</label>
              <div class="hint">Off by default. Normally the tone tags
                ([curious], [long pause]) come out and the apostrophes ASR
                dropped go back in, because "i m" and "don t" are not words
                any encoder was trained on.</div>
            </div>
          </div>
        </details>
        <button class="go" id="askgo">Answer the questions</button>
        <div class="hint" id="askerr" style="color:var(--bad)"></div>
      </div>
      <div id="answer"></div>
    </div>

    <!-- training output -->
    <div id="m-train" hidden>
      <pre class="log" id="trainlog">No training run selected.

Fill in the form on the left and press Train. The run is detached from this
page exactly like a benchmark run, so it survives closing the browser, and it
shows up in Recent runs on the Benchmark page too.</pre>
    </div>

    <!-- about -->
    <div id="m-about" hidden>
      <div class="card note">
        <p><strong>Two different claims, kept apart.</strong> Training fits a
        binary scam/legitimate classifier, and <code>prob_scam</code> is that
        head speaking — it is the only number the model was directly trained to
        produce. The MCQ score beside it is the ontology's: the sum of the
        values of the options chosen below, banded by the cut-offs in
        <code>knowledge/mcq_ontology.json</code>. They can disagree, and when
        they do that is worth reading, not averaging.</p>

        <p><strong>How a question gets answered without MCQ labels.</strong>
        By similarity. The transcript is cut into overlapping word windows;
        each window and each option text is mean-pooled into a vector; an
        option scores the best cosine similarity it reaches against any
        window. The mean direction of the whole option corpus is subtracted
        from both sides first — sentence vectors out of any BERT sit in a
        narrow cone, so two unrelated phrases still score .85 against each
        other, and taking that shared direction out is what gives the options
        room to differ.</p>

        <p><strong>Why the options are not matched in the checkpoint.</strong>
        They were, and it was the reason the answers looked arbitrary.
        Fine-tuning fits a binary scam/legitimate head; nothing in that
        objective asks the encoder to tell "a courier" from "a customs
        agency", which is the distinction every question here turns on. Every
        option came back within a few hundredths of every other, and a softmax
        over noise still has to hand its probability to somebody. So the match
        runs in <code>all-MiniLM-L6-v2</code> — the model that already indexes
        the policy KB — and the checkpoint keeps the job it was trained for,
        which is <code>prob_scam</code>. The tickbox under <em>How the options
        are matched</em> puts it back the old way if you want to see the
        difference.</p>

        <p><strong>The margin is the number to read.</strong> A question is
        only answered when its best option is clear of the runner-up by the
        margin you set, in raw cosine. Confidence cannot carry that on its
        own: four scores that are the same number twice still produce a
        confident-looking softmax. Under the margin the question abstains to
        its "not stated" answer and contributes nothing to the score —
        not knowing whether the caller asked for anything is not evidence that
        they asked for nothing.</p>

        <p><strong>Where it is weak.</strong> Similarity reads subject matter,
        not negation — "I will <em>not</em> ask for your PIN" sits close to the
        option about asking for a PIN. That is the honest limit of matching
        rather than reasoning, and it is the gap the LLM-driven
        <code>mcq</code> baseline on the other page exists to close.</p>

        <p><strong>The evidence line</strong> under each answer is the window
        that scored highest for the chosen option — the stretch of the call the
        answer actually came from. Open <em>all options</em> to see what every
        other option scored, and what it would have contributed.</p>
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
</div>

<script>
let CFG = null, current = null, offset = 0, timer = null;
let ART = {steps: [], csvs: []}, tab = 'output';
// id -> meta for every run the page knows about, so the panes can tell a
// benchmark from a knowledge-base update before the first poll comes back
let RUNS = {};
let page = 0, PAGE_SIZE = 50;

const $ = id => document.getElementById(id);

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
    $('ollama').textContent = 'Ollama is not answering — only the trivial and BERT baselines can run';
    $('ollama').style.color = 'var(--bad)';
  } else if (o.loaded.length) {
    $('ollama').textContent = 'Ollama up · holding ' + o.loaded.join(', ') + ' in VRAM';
  } else {
    $('ollama').textContent = 'Ollama up · nothing loaded';
  }

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

function select(id) {
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
  const head = ['system','acc','P','R','F1','TP','FP','FN','TN'];
  let h = '<tr>' + head.map(x => `<th>${x}</th>`).join('') + '</tr>';
  for (const s of RESULTS.systems) {
    if (!s.ran) {
      h += `<tr class="skipped"><td>${s.system}</td><td colspan="8">not run</td></tr>`;
      continue;
    }
    h += `<tr><td>${s.system}</td><td>${s.acc.toFixed(1)}%</td>` +
         `<td>${s.p.toFixed(3)}</td><td>${s.r.toFixed(3)}</td>` +
         `<td>${s.f1.toFixed(3)}</td><td>${s.tp}</td><td>${s.fp}</td>` +
         `<td>${s.fn}</td><td>${s.tn}</td></tr>`;
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
      if (c === 'text' || c === 'transcript')
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

// ============================================================ BERT + MCQ
// The second page keeps its own state throughout - its own selected run, its
// own poller - so switching pages never disturbs a benchmark streaming into
// the first one. The only thing the two share is the run machinery on the
// server: a training run is a detached run like any other, which is why it
// also turns up under Recent runs.
const PAGES = ['bench', 'mcq', 'llm'];
const pageInUrl = () => PAGES.includes(location.hash.slice(1))
  ? location.hash.slice(1) : 'bench';

let MCQ = null, model = null, MODELS = [];
let trainRun = null, mtimer = null, moffset = 0;

// form field -> the key /api/mcq/config sends its default under
const MFIELDS = {tepochs: 'epochs', tbatch: 'batch_size', tmaxlen: 'max_length',
                 tlr: 'lr', tseed: 'seed', tlimit: 'limit', tholdout: 'holdout',
                 awindow: 'window', astride: 'stride', amaxlen: 'max_length',
                 aminconf: 'min_confidence', aminmargin: 'min_margin'};

// The page is in the URL, so #mcq can be bookmarked, reloaded, and sent to
// someone - and reloading while reading an answer comes back to the answer
// pane rather than to the benchmark form.
function showPage(name) {
  if (location.hash.slice(1) !== name)
    history.replaceState(null, '', name === 'bench' ? location.pathname : '#' + name);
  for (const b of $('pages').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.page === name);
  $('page-bench').hidden = name !== 'bench';
  $('page-mcq').hidden = name !== 'mcq';
  $('page-llm').hidden = name !== 'llm';
  // drawn on first visit rather than at boot: someone who only ever runs
  // benchmarks should not be made to wait for a directory scan of models/,
  // nor for ollama to be asked what it has pulled
  if (name === 'mcq' && !MCQ) mcqBoot();
  if (name === 'llm' && !LLM) llmBoot();
}

async function mcqBoot() {
  const cfg = await api('/api/mcq/config');
  if (cfg.error || !cfg.branches) {
    $('modelerr').textContent = cfg.error || 'unexpected reply from /api/mcq/config';
    return;
  }
  MCQ = cfg;

  $('tdataset').innerHTML = $('sampleds').innerHTML = MCQ.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');
  $('tbase').innerHTML = MCQ.bases.map(b =>
    `<option value="${b.id}">${b.id}</option>`).join('');
  // "auto" is the interesting setting - forcing a branch is for checking what
  // the questions of another branch would have made of the same call
  $('branch').innerHTML =
    '<option value="auto">let the model route it</option>' +
    MCQ.branches.map(b => `<option value="${b.id}">${esc(b.text)}` +
      ` (${b.questions} question${b.questions === 1 ? '' : 's'})</option>`).join('');
  for (const [id, key] of Object.entries(MFIELDS)) $(id).value = MCQ.defaults[key];

  $('tbase').onchange = onBase;
  onBase();
  $('traingo').onclick = train;
  $('trainstop').onclick = () => trainRun && api('/api/stop', {id: trainRun});
  $('askgo').onclick = ask;
  $('sampleload').onclick = loadSample;
  $('sampleidx').onkeydown = e => { if (e.key === 'Enter') loadSample(); };
  $('unload').onclick = unloadModel;
  for (const b of $('mcqtabs').querySelectorAll('button'))
    b.onclick = () => showMtab(b.dataset.mtab);

  await refreshModels();
  // a training run already going when the page opens - a reload mid-train, or
  // one started before this server was restarted - is picked back up
  const going = (await runList() || []).find(
    r => r.kind === 'train' && r.status === 'running');
  if (going) { RUNS[going.id] = going; watchTraining(going.id); showMtab('train'); }
}

function onBase() {
  const b = MCQ.bases.find(x => x.id === $('tbase').value);
  $('tbasenote').textContent = b ? b.note : '';
}

function showMtab(name) {
  for (const b of $('mcqtabs').querySelectorAll('button'))
    b.classList.toggle('on', b.dataset.mtab === name);
  for (const t of ['ask', 'train', 'about']) $('m-' + t).hidden = t !== name;
}

// ------------------------------------------------------- trained models
async function refreshModels() {
  const r = await api('/api/mcq/models');
  if (r.error) { $('modelerr').textContent = r.error; return; }
  $('modelerr').textContent = '';
  paintModels(r.models || [], r.worker || {loaded: false});
}

function paintModels(models, worker) {
  MODELS = models;
  paintWorker(worker);
  if (!models.length) {
    $('mcqmodels').innerHTML = '<div class="muted">nothing trained yet</div>';
    model = null;
    paintMcqHeader();
    return;
  }
  if (!models.some(m => m.name === model)) model = models[0].name;
  $('mcqmodels').innerHTML = models.map(m => {
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
  for (const a of $('mcqmodels').querySelectorAll('a'))
    a.onclick = () => { model = a.dataset.model; markModels(); paintMcqHeader(); };
  for (const b of $('mcqmodels').querySelectorAll('[data-delmodel]'))
    b.onclick = e => { e.stopPropagation(); delModel(b.dataset.delmodel); };
  markModels();
  paintMcqHeader();
}

function markModels() {
  for (const a of $('mcqmodels').querySelectorAll('a'))
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

function paintMcqHeader(status) {
  const m = MODELS.find(x => x.name === model);
  if (m) {
    $('mcqtitle').textContent = m.name;
    const bits = [m.base];
    if (m.dataset) bits.push(m.dataset.replace('datasets/', '')
                             + (m.rows ? ' · ' + m.rows + ' calls' : ''));
    if (m.holdout && m.holdout.acc != null)
      bits.push('holdout acc ' + (100 * m.holdout.acc).toFixed(1) + '%'
                + ' · F1 ' + m.holdout.f1.toFixed(3));
    if (m.trained_at) bits.push(new Date(m.trained_at).toLocaleString());
    $('mcqsub').textContent = bits.join(' · ');
  } else {
    $('mcqtitle').textContent = 'No model selected';
    $('mcqsub').textContent = "Train one on the left, then put the ontology's "
                            + 'questions to it.';
  }
  const st = status || (trainRun && RUNS[trainRun] && RUNS[trainRun].status);
  const pill = $('mcqpill');
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
  const r = await api('/api/mcq/delete_model', {name});
  if (r.error) { $('modelerr').textContent = r.error; return; }
  if (model === name) { model = null; $('answer').innerHTML = ''; }
  await refreshModels();
}

async function unloadModel() {
  const r = await api('/api/mcq/unload', {});
  if (r.error) { $('modelerr').textContent = r.error; return; }
  await refreshModels();
}

// ------------------------------------------------------------- training
async function train() {
  $('trainerr').textContent = '';
  $('traingo').disabled = true;
  const res = await api('/api/mcq/train', {
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
  paintMcqHeader(r.status);
  if (r.status !== 'running') {
    clearInterval(mtimer); mtimer = null;
    // the checkpoint that just appeared is the point of the whole run
    await refreshModels();
    await refreshHistory();
  }
}

// ----------------------------------------------------------- asking it
async function loadSample() {
  $('sampleinfo').textContent = 'loading…';
  const r = await api(`/api/mcq/sample?dataset=${encodeURIComponent($('sampleds').value)}`
                    + `&idx=${encodeURIComponent($('sampleidx').value || 0)}`);
  if (r.error) { $('sampleinfo').textContent = r.error; return; }
  $('transcript').value = r.text;
  $('sampleidx').value = r.idx;
  $('sampleinfo').textContent = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + r.row_id : '')
    + (r.label ? ' · labelled ' + r.label : '');
}

async function ask() {
  $('askerr').textContent = '';
  if (!model) { $('askerr').textContent = 'train a model first - there is '
                                        + 'nothing to put the questions to'; return; }
  const text = $('transcript').value.trim();
  if (!text) { $('askerr').textContent = 'paste a transcript, or load one from '
                                       + 'a dataset above'; return; }
  $('askgo').disabled = true;
  $('askgo').textContent = 'Answering…';
  $('answer').innerHTML = '<div class="card muted">putting the questions to '
    + esc(model) + '… the first one after a model or a setting changes also '
    + 'loads the checkpoint, which takes a few seconds</div>';
  const res = await api('/api/mcq/answer', {
    model: model, transcript: text, branch: $('branch').value,
    cutoff: $('cutoff').value, gpu: $('agpu').checked,
    window: $('awindow').value, stride: $('astride').value,
    max_length: $('amaxlen').value, min_confidence: $('aminconf').value,
    min_margin: $('aminmargin').value, self_encoder: $('aself').checked,
    raw_text: $('araw').checked,
  });
  $('askgo').disabled = false;
  $('askgo').textContent = 'Answer the questions';
  if (res.error) {
    $('askerr').textContent = res.error;
    $('answer').innerHTML = '';
    return;
  }
  paintAnswer(res);
  await refreshModels();        // the checkpoint is in memory now
}

function paintAnswer(a) {
  const v = a.score.verdict;
  const p = a.classifier.prob_scam;
  const cls = p >= a.classifier.threshold ? 'scam' : 'legitimate';
  const sum = (a.score.sum > 0 ? '+' : '') + a.score.sum.toFixed(2);

  // The two numbers side by side and never combined: one is the sum of the
  // options chosen below, the other is the head that was actually trained.
  const head = `
  <div class="card">
    <div class="verdict">
      <div>
        <div class="cap">MCQ score</div>
        <div class="big ${v}">${sum}</div>
        <div class="hint">${esc(v)}${a.score.cutoff
            ? ' · cut-off ' + a.score.cutoff : ''}</div>
      </div>
      <div>
        <div class="cap">Trained head</div>
        <div class="big ${cls}">${(100 * p).toFixed(1)}%</div>
        <div class="hint">prob_scam · ${cls} at ${a.classifier.threshold}</div>
      </div>
      <div style="flex:1; min-width:210px">
        <div class="cap">Routed to</div>
        <div style="font-weight:600">${esc(a.route.chosen_text)}</div>
        <div class="hint">${a.route.forced ? 'the branch you chose'
          : (100 * a.route.confidence).toFixed(0) + '% confident'} · ${
          a.questions.length} question${a.questions.length === 1 ? '' : 's'}</div>
      </div>
    </div>
    ${a.route.legit_contrast ? `<div class="ev" style="font-style:normal; margin-top:15px">
      <strong>A real call of this kind:</strong> ${esc(a.route.legit_contrast)}</div>` : ''}
  </div>`;

  const body = a.questions.length
    ? '<div class="card">' + a.questions.map(qBlock).join('') + '</div>'
    : '<div class="card muted">that branch asks no questions, so there is '
      + 'nothing to score - the call did not look like any of the kinds the '
      + 'ontology covers</div>';

  const m = a.matching || {};
  $('answer').innerHTML = head + body + `
    <div class="card hint">models/${esc(a.model.name)} · ${esc(a.model.base || '?')}
      · ${a.windows} window${a.windows === 1 ? '' : 's'} · ${a.elapsed_ms} ms
      on ${esc(a.device)}${m.encoder ? `<br>options matched in
      ${esc(m.encoder === 'self' ? "the checkpoint's own hidden states"
        : m.encoder)}${m.centred ? ', centred' : ', uncentred'}${
        m.normalised ? '' : ', raw transcript'} · answered ${m.answered} of
      ${m.asked} question${m.asked === 1 ? '' : 's'}` : ''}</div>`;
}

function qBlock(q) {
  const pct = Math.round(100 * q.confidence);
  const chip = q.recorded
    ? '<span class="chip zero">recorded</span>'
    : `<span class="chip ${q.contributes > 0 ? 'pos'
        : q.contributes < 0 ? 'neg' : 'zero'}">${
        q.contributes > 0 ? '+' : ''}${q.contributes.toFixed(1)}</span>`;
  const val = o => (o.value === null || o.value === undefined) ? '—'
    : (o.value > 0 ? '+' : '') + o.value.toFixed(1);
  const rows = q.options.map(o => `
    <tr class="${o.id === q.chosen ? 'chosen' : ''}">
      <td class="optname">${esc(o.text)}</td>
      <td>${val(o)}</td>
      <td>${(100 * o.confidence).toFixed(0)}%</td>
      <td>${o.similarity.toFixed(3)}</td>
    </tr>`).join('');
  const ev = q.evidence
    ? `<div class="ev">…${esc(q.evidence.slice(0, 320))}${
        q.evidence.length > 320 ? '…' : ''}</div>` : '';
  return `
  <div class="q">
    <div class="qp">${esc(q.prompt)}</div>
    <div class="qa">
      <span class="pick">${esc(q.chosen_text)}${q.abstained
        ? ' <span class="hint">— abstained</span>' : ''}</span>
      ${chip}<span class="hint">${pct}%${q.margin === undefined ? ''
        : ' · margin ' + q.margin.toFixed(3)}</span>
    </div>
    ${q.abstained && q.why_abstained ? `<div class="hint">${esc(q.why_abstained)}${
      q.best_text ? ' — the best option was “' + esc(q.best_text) + '”' : ''}</div>` : ''}
    <div class="bar ${q.abstained || q.confidence < 0.4 ? 'low' : ''}">
      <i style="width:${Math.max(2, pct)}%"></i></div>
    ${ev}
    <details class="adv" style="margin-bottom:0">
      <summary>all ${q.options.length} options${q.recorded
        ? ' · recorded for the explanation, scores nothing' : ''}</summary>
      <div class="advbody">
        <div class="scroll"><table class="opts">
          <tr><th>option</th><th>value</th><th>confidence</th><th>similarity</th></tr>
          ${rows}
        </table></div>
        ${q.note ? `<div class="hint" style="margin-top:11px">${esc(q.note)}</div>` : ''}
      </div>
    </details>
  </div>`;
}

// ============================================================== LLM judge
// The third page. One transcript, one question, one verdict with a reason.
// It holds nothing between clicks - there is no worker to keep alive, since
// ollama is the thing holding the model.
let LLM = null;
const LFIELDS = {llmmaxtok: 'max_tokens', llmctx: 'num_ctx', llmtemp: 'temperature'};

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
  $('llmgo').onclick = llmAsk;
  $('llmguideclear').onclick = () => { $('llmguidance').value = ''; llmSize(); };
  for (const id of ['llmtranscript', 'llmguidance', 'llmctx'])
    $(id).addEventListener('input', llmSize);
  llmSize();
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
  const r = await api(`/api/mcq/sample?dataset=${encodeURIComponent($('llmds').value)}`
                    + `&idx=${encodeURIComponent($('llmidx').value || 0)}`);
  if (r.error) { $('llminfo').textContent = r.error; return; }
  $('llmtranscript').value = r.text;
  $('llmidx').value = r.idx;
  $('llminfo').textContent = `row ${r.idx} of ${r.total}`
    + (r.row_id ? ' · id ' + r.row_id : '')
    + (r.label ? ' · labelled ' + r.label : '');
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
  const body = {model: model, transcript: text,
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
