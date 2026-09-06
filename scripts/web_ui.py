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
    ("ontology", "Ontology RAG",          "scam_ontology.json",                   True),
    ("mcq",      "MCQ ontology",          "mcq_ontology.json, 2 calls per transcript", True),
    ("bert",     "BERT",                  "fine-tuned classifier, no LLM",        False),
]
MODELS = ["qwen2.5:14b", "llama3.1:8b"]
LIMIT_RE = re.compile(r"^(?:\d+|id:.+|idx:\d+)$")
EXIT_MARK = "__RUN_EXIT__"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

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
    meta["status"] = run_status(meta)
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
    baseline = form.get("baseline", "")
    limit = str(form.get("limit", "0")).strip()
    model = form.get("model", "")

    valid_ds = {d["path"] for d in datasets()}
    if ds not in valid_ds:
        raise ValueError("unknown dataset")
    entry = next((b for b in BASELINES if b[0] == baseline), None)
    if entry is None:
        raise ValueError("unknown baseline")
    if not LIMIT_RE.match(limit):
        raise ValueError("limit must be a whole number, id:<value>, or idx:<n>")
    needs_model = entry[3]
    if needs_model:
        if model not in MODELS:
            raise ValueError("pick a model")
    else:
        model = ""

    running = [r for r in all_runs() if r["status"] == "running"]
    if running:
        raise ValueError("a run is already going (%s). Stop it first - the GPU "
                         "cannot hold two." % running[0]["id"])

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + baseline
    log = run_path(run_id, "log")

    flags = ["--dataset", ds, "--baseline", baseline, "--limit", limit]
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

    meta = {"id": run_id, "pid": proc.pid, "dataset": ds, "baseline": baseline,
            "limit": limit, "model": model, "started": time.time()}
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


def results_of(run_id):
    """The finished numbers, via collect_results.py --json."""
    meta = load_run(run_id)
    if not meta:
        return None
    log = run_path(run_id, "log")
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"(results/logs/run_\d{8}_\d{6})", text)
    if not m:
        return None
    try:
        out = subprocess.run(
            [sys.executable, "scripts/collect_results.py", m[-1], "--json"],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)
    except Exception:
        return None


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
                })
            if u.path == "/api/runs":
                return self._send(200, {"runs": all_runs()})
            if u.path == "/api/log":
                rid = q.get("id", [""])[0]
                off = int(q.get("offset", ["0"])[0])
                meta = load_run(rid)
                if not meta:
                    return self._send(404, {"error": "no such run"})
                out = read_log(rid, off)
                out["status"] = meta["status"]
                return self._send(200, out)
            if u.path == "/api/results":
                return self._send(200, {"results": results_of(q.get("id", [""])[0])})
            return self._send(404, {"error": "not found"})
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
  button { font:inherit; font-weight:600; padding:9px 16px; border-radius:6px;
           border:1px solid transparent; cursor:pointer; }
  .go { background:var(--accent); color:#fff; width:100%; margin-top:20px; }
  .go[disabled] { opacity:.5; cursor:not-allowed; }
  .stop { background:transparent; color:var(--bad); border-color:var(--bad); }
  .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:14px 16px; margin-bottom:18px; }
  .err { border-color:var(--bad); color:var(--bad); }
  pre.log { background:var(--panel); border:1px solid var(--line); border-radius:8px;
            padding:14px 16px; font-family:var(--mono); font-size:12.5px;
            line-height:1.5; max-height:52vh; overflow:auto; white-space:pre-wrap;
            word-break:break-word; margin:0; }
  .l-step { color:var(--accent); font-weight:600; }
  .l-ok   { color:var(--accent); }
  .l-warn { color:var(--warn); }
  .l-fail { color:var(--bad); font-weight:600; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; font-family:var(--mono); }
  th { color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase;
       letter-spacing:.04em; }
  tr.skipped td { color:var(--dim); }
  .pill { font-size:12px; padding:2px 9px; border-radius:99px; border:1px solid var(--line);
          color:var(--dim); }
  .pill.running { color:var(--accent); border-color:var(--accent); }
  .pill.failed  { color:var(--bad); border-color:var(--bad); }
  .hist { font-size:13px; }
  .hist a { display:block; padding:7px 0; border-bottom:1px solid var(--line);
            color:inherit; text-decoration:none; cursor:pointer; }
  .hist a:hover { color:var(--accent); }
  .hist .meta { color:var(--dim); font-size:12px; }
  .muted { color:var(--dim); }
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

    <label for="baseline">Baseline</label>
    <select id="baseline"></select>
    <div class="hint" id="bnote"></div>

    <label for="limit">How many calls</label>
    <input type="text" id="limit" value="0" spellcheck="false">
    <div class="hint">0 = whole dataset · N = first N · id:&lt;value&gt; = one row
      · idx:&lt;n&gt; = the n-th call of a --limit 40 run</div>

    <div id="modelbox">
      <label for="model">Model</label>
      <select id="model"></select>
    </div>

    <button class="go" id="go">Run</button>
    <div class="hint" id="formerr" style="color:var(--bad)"></div>

    <label style="margin-top:26px">Recent runs</label>
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

    <div class="card" id="resultscard" hidden>
      <table id="results"></table>
      <div class="hint" id="spread"></div>
    </div>

    <pre class="log" id="log">Pick a dataset and a baseline, then press Run.

The run is detached from this page: closing the browser, or losing the SSH
connection, does not stop it. Come back to this URL and it is still here.</pre>
  </div>
</div>

<script>
let CFG = null, current = null, offset = 0, timer = null;

const $ = id => document.getElementById(id);

async function api(path, body) {
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                      body: JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  return r.json();
}

// ------------------------------------------------------------------ setup
async function boot() {
  CFG = await api('/api/config');

  $('dataset').innerHTML = CFG.datasets.map(d =>
    `<option value="${d.path}">${d.name} — ${d.rows === null ? '?' : d.rows} rows</option>`
  ).join('');

  $('baseline').innerHTML = CFG.baselines.map(b =>
    `<option value="${b.key}">${b.label}</option>`).join('');

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

  $('baseline').onchange = onBaseline;
  onBaseline();
  $('go').onclick = go;
  $('stopbtn').onclick = stop;
  await refreshHistory();
  const running = (await api('/api/runs')).runs.find(r => r.status === 'running');
  if (running) select(running.id);
}

function onBaseline() {
  const b = CFG.baselines.find(x => x.key === $('baseline').value);
  $('bnote').textContent = b.note;
  $('modelbox').hidden = !b.needs_model;
}

// -------------------------------------------------------------- run control
async function go() {
  $('formerr').textContent = '';
  $('go').disabled = true;
  const res = await api('/api/run', {
    dataset: $('dataset').value,
    baseline: $('baseline').value,
    limit: $('limit').value,
    model: $('model').value,
  });
  $('go').disabled = false;
  if (res.error) { $('formerr').textContent = res.error; return; }
  await refreshHistory();
  select(res.id);
}

async function stop() {
  if (!current) return;
  await api('/api/stop', {id: current});
}

function select(id) {
  current = id; offset = 0;
  $('log').textContent = '';
  $('resultscard').hidden = true;
  if (timer) clearInterval(timer);
  poll();
  timer = setInterval(poll, 900);
}

// ------------------------------------------------------------------ polling
async function poll() {
  if (!current) return;
  const r = await api(`/api/log?id=${encodeURIComponent(current)}&offset=${offset}`);
  if (r.error) return;
  offset = r.offset;
  if (r.text) append(r.text);

  const meta = (await api('/api/runs')).runs.find(x => x.id === current);
  if (meta) paintHeader(meta, r.status);

  if (r.status !== 'running') {
    clearInterval(timer); timer = null;
    await refreshHistory();
    await showResults();
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
  $('runtitle').textContent = meta.baseline + ' · ' + meta.dataset.replace('datasets/', '');
  const bits = ['limit ' + meta.limit];
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
async function showResults() {
  const r = (await api(`/api/results?id=${encodeURIComponent(current)}`)).results;
  if (!r) return;
  const head = ['system','acc','P','R','F1','TP','FP','FN','TN'];
  let h = '<tr>' + head.map(x => `<th>${x}</th>`).join('') + '</tr>';
  for (const s of r.systems) {
    if (!s.ran) {
      h += `<tr class="skipped"><td>${s.system}</td>` +
           `<td colspan="8">not run</td></tr>`;
      continue;
    }
    h += `<tr><td>${s.system}</td><td>${s.acc.toFixed(1)}%</td>` +
         `<td>${s.p.toFixed(3)}</td><td>${s.r.toFixed(3)}</td>` +
         `<td>${s.f1.toFixed(3)}</td><td>${s.tp}</td><td>${s.fp}</td>` +
         `<td>${s.fn}</td><td>${s.tn}</td></tr>`;
  }
  $('results').innerHTML = h;
  $('spread').textContent = (r.bert_spread || []).join(' · ');
  $('resultscard').hidden = false;
}

// ------------------------------------------------------------------ history
async function refreshHistory() {
  const runs = (await api('/api/runs')).runs.slice(0, 15);
  if (!runs.length) { $('hist').innerHTML = '<div class="muted">nothing yet</div>'; return; }
  $('hist').innerHTML = runs.map(r => `
    <a data-id="${r.id}">
      ${r.baseline} · ${r.dataset.replace('datasets/','')}
      <span class="pill ${r.status}">${r.status}</span>
      <div class="meta">limit ${r.limit}${r.model ? ' · ' + r.model : ''} ·
        ${new Date(r.started * 1000).toLocaleString()}</div>
    </a>`).join('');
  for (const a of $('hist').querySelectorAll('a')) a.onclick = () => select(a.dataset.id);
}

boot();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="browser front end for run_all.sh")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) or 0.0.0.0 to expose it")
    args = ap.parse_args()

    if not RUN_ALL.exists():
        sys.exit("run_all.sh not found next to scripts/ - run this from the repo")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    where = "http://%s:%d" % ("localhost" if args.host == "127.0.0.1" else args.host,
                              args.port)
    print("scam-detection UI on %s" % where)
    if args.host == "127.0.0.1":
        print("over SSH:  ssh -L %d:localhost:%d %s@<host>"
              % (args.port, args.port, os.environ.get("USER", "you")))
    print("ctrl-c to stop the server (runs already going are not affected)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped")


if __name__ == "__main__":
    main()
