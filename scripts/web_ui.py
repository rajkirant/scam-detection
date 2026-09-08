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

Over SSH, forward the port rather than binding to 0.0.0.0:

    ssh -L 8000:localhost:8000 rkt29@cs25003ay

Runs are detached from this server (their own session, output straight to a
file), so a run survives closing the browser, and it survives restarting or
killing this server too. Reopen the page and the run is still there.

Standard library only - nothing to install.
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
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


def run_path(run_id, ext):
    return RUNS_DIR / ("%s.%s" % (run_id, ext))


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
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:17px; font-weight:600; letter-spacing:-.01em; }
  header .sub { color:var(--dim); font-size:13px; }
  .wrap { display:grid; grid-template-columns:340px 1fr; gap:0; align-items:start; }
  @media (max-width:900px){ .wrap { grid-template-columns:1fr; } }
  .side { padding:22px; border-right:1px solid var(--line); }
  .main { padding:22px; min-width:0; }
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

  .pill { font-size:12px; padding:2px 9px; border-radius:99px; border:1px solid var(--line);
          color:var(--dim); }
  .pill.running { color:var(--accent); border-color:var(--accent); }
  .pill.failed  { color:var(--bad); border-color:var(--bad); }
  .hist { font-size:13px; }
  .hist a { display:block; padding:7px 0; border-bottom:1px solid var(--line);
            color:inherit; text-decoration:none; cursor:pointer; }
  .hist a:hover { color:var(--accent); }
  .hist a.on { color:var(--accent); font-weight:600; }
  .hist .meta { color:var(--dim); font-size:12px; font-weight:400; }
  .muted { color:var(--dim); }
  .pager { display:flex; align-items:center; gap:12px; margin-top:12px; }
  .pager button { border-color:var(--line); background:var(--panel); color:var(--ink); }
  .pager button[disabled] { opacity:.4; cursor:not-allowed; }
  .filepick { display:flex; align-items:center; gap:10px; margin-bottom:12px;
              flex-wrap:wrap; }
  .filepick select { width:auto; min-width:260px; }
</style>
</head>
<body>
<header>
  <h1>scam-detection benchmark</h1>
  <span class="sub" id="ollama">checking Ollama…</span>
</header>

<div class="wrap">
  <div class="side">
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
    </div>

    <button class="go" id="go">Run</button>
    <div class="hint" id="formerr" style="color:var(--bad)"></div>

    <label style="margin-top:26px" for="kbmode">Web-RAG knowledge base</label>
    <div class="hint" style="margin-top:-2px" id="kbstate">checking…</div>
    <select id="kbmode" style="margin-top:8px"></select>
    <div class="hint" id="kbnote"></div>
    <button class="go" id="kbgo">Update knowledge base</button>
    <div class="hint" id="kberr" style="color:var(--bad)"></div>

    <label style="margin-top:26px">Recent runs</label>
    <div class="hint" style="margin-top:-2px">pick one to read its output and results</div>
    <div class="hist" id="hist"></div>
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

Past runs are on the left - selecting one brings back its output, its
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
  $('steppick').onchange = loadStep;
  $('prev').onclick = () => { if (page > 0) { page--; loadCalls(); } };
  $('next').onclick = () => { page++; loadCalls(); };

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
let KB = null;

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
  // a knowledge-base update produces no results table and no per-call CSV,
  // so it gets the Output pane on its own
  showKbTabs(isKb(id));
  showTab('output');
  markHistory();
  // load them now as well as on completion, so a run selected while it is
  // still going already offers the step logs of whatever has finished
  if (!isKb(id)) loadArtifacts();
  if (timer) clearInterval(timer);
  poll();
  timer = setInterval(poll, 900);
}

const isKb = id => !!(RUNS[id] && RUNS[id].kind === 'kb');

function showKbTabs(kb) {
  for (const b of $('tabs').querySelectorAll('button'))
    b.hidden = kb && b.dataset.tab !== 'output';
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
    if (isKb(current)) {
      // the counts on the left are what just changed
      await refreshKb();
    } else {
      await loadArtifacts();
      await showResults();
    }
  }
}

function append(text) {
  const log = $('log');
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
  const kb = meta.kind === 'kb';
  showKbTabs(kb);
  if (kb && tab !== 'output') showTab('output');
  $('runtitle').textContent = meta.label ||
    (meta.baseline + ' · ' + meta.dataset.replace('datasets/', ''));
  const bits = [kb ? meta.dataset : limitText(meta.limit)];
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
  // so it can be marked as agreeing with the label or not.
  const cols = r.columns;
  const truthAt = cols.indexOf('true');
  let h = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
  for (const row of r.rows) {
    h += '<tr>' + row.map((v, i) => {
      const c = cols[i];
      if (c === 'text' || c === 'transcript')
        return `<td class="text">${esc(v.length > 260 ? v.slice(0, 260) + '…' : v)}</td>`;
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
  if (!runs.length) { $('hist').innerHTML = '<div class="muted">nothing yet</div>'; return; }
  $('hist').innerHTML = runs.map(r => `
    <a data-id="${r.id}">
      ${r.label || (r.baseline + ' · ' + r.dataset.replace('datasets/',''))}
      <span class="pill ${r.status}">${r.status}</span>
      <div class="meta">${r.kind === 'kb' ? '' : limitText(r.limit) + ' · '}${
        r.model ? r.model + ' · ' : ''}${new Date(r.started * 1000).toLocaleString()}</div>
    </a>`).join('');
  for (const a of $('hist').querySelectorAll('a')) a.onclick = () => select(a.dataset.id);
  markHistory();
}

function markHistory() {
  for (const a of $('hist').querySelectorAll('a'))
    a.classList.toggle('on', a.dataset.id === current);
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
