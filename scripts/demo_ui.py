#!/usr/bin/env python3
"""
demo_ui.py - a small, read-only demo of the detectors for readers of the paper.

It is a separate server from web_ui.py, so the full UI (runs, training, the
file browser, the MCQ editor) stays private while this one is public. It has
three pages and nothing else:

  Try it yourself   paste a call transcript, or pick an example, and get the
                    verdict and the reason from LLM-only (and from ScamRAG,
                    when its pattern store is available on this machine)
  Results           the tables of the paper, and per-call answers from the
                    result files listed in demo/demo_config.json
  Reproduce         the steps to reproduce the results, and the code link

Protection for a public link:
  - no endpoint writes anything, starts a run or reads a file that is not
    listed in demo/demo_config.json
  - each visitor may submit DEMO_PER_CLIENT checks per DEMO_WINDOW seconds,
    and the whole demo DEMO_PER_HOUR checks per hour
  - one check runs at a time; a check that cannot start within
    DEMO_QUEUE_WAIT seconds is told the demo is busy
  - transcripts longer than DEMO_MAX_CHARS characters are refused
  - nothing a visitor submits is stored

Run it with web_ui.sh --demo (port 8081, this machine only by default), and
put it behind a tunnel. Standard library only, like web_ui.py; ScamRAG needs
the venv (chromadb), and is left out with a note when it cannot be loaded.

    ./web_ui.sh --demo
    python3 scripts/demo_ui.py --port 8081 --host 127.0.0.1
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

CONFIG_PATH = Path(os.environ.get("DEMO_CONFIG", ROOT / "demo" / "demo_config.json"))

PER_CLIENT = int(os.environ.get("DEMO_PER_CLIENT", "5"))      # checks per visitor
WINDOW = int(os.environ.get("DEMO_WINDOW", "600"))            # ... per 10 minutes
PER_HOUR = int(os.environ.get("DEMO_PER_HOUR", "60"))         # checks for everyone
QUEUE_WAIT = int(os.environ.get("DEMO_QUEUE_WAIT", "90"))     # seconds to wait for the GPU
MAX_CHARS = int(os.environ.get("DEMO_MAX_CHARS", "6000"))
MAX_BODY = MAX_CHARS * 4 + 1024
PAGE_ROWS = 20

csv.field_size_limit(10 ** 9)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("tables", [])
    cfg.setdefault("examples", [])
    cfg.setdefault("percall", [])
    cfg.setdefault("reproduce", [])
    return cfg


def percall_file(cfg, key):
    """The result file listed under `key` in the config, and only that: a
    visitor cannot name a path, so nothing else on disk can be read."""
    for entry in cfg["percall"]:
        if entry.get("key") == key:
            path = (ROOT / entry["file"]).resolve()
            if ROOT not in path.parents:
                raise ValueError("listed file is outside the project")
            if not path.is_file():
                raise ValueError("%s is not on this machine" % entry["file"])
            return entry, path
    raise ValueError("no such result file")


def percall_page(cfg, key, offset):
    entry, path = percall_file(cfg, key)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    offset = max(0, min(offset, max(0, len(rows) - 1)))
    page = rows[offset:offset + PAGE_ROWS]
    shown = entry.get("columns") or [c for c in (rows[0].keys() if rows else [])]
    return {
        "label": entry.get("label", key),
        "note": entry.get("note", ""),
        "total": len(rows),
        "offset": offset,
        "columns": shown,
        "rows": [{c: (r.get(c) or "") for c in shown} for r in page],
    }


# ---------------------------------------------------------------------------
# rate limiting and the single GPU slot
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_by_client = defaultdict(deque)
_all = deque()
_gpu = threading.Semaphore(1)


def allow(client):
    """True if this visitor, and the demo as a whole, are under their limits.
    Counted when a check is accepted, so a refused one costs nothing."""
    now = time.time()
    with _lock:
        mine = _by_client[client]
        while mine and now - mine[0] > WINDOW:
            mine.popleft()
        while _all and now - _all[0] > 3600:
            _all.popleft()
        if len(mine) >= PER_CLIENT:
            wait = int(WINDOW - (now - mine[0])) + 1
            return False, "You have used your %d checks for now. Please try again in about %d minutes." % (PER_CLIENT, max(1, wait // 60))
        if len(_all) >= PER_HOUR:
            return False, "The demo has reached its hourly limit. Please try again later."
        mine.append(now)
        _all.append(now)
        if len(_by_client) > 10000:              # forget idle visitors
            for k in [k for k, v in _by_client.items() if not v]:
                del _by_client[k]
        return True, ""


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
_scamrag = {"loaded": False, "collection": None, "module": None, "error": ""}


def scamrag_available():
    """Load ScamRAG once, if this machine has its pattern store. Optional:
    the demo still works with LLM-only when it is missing."""
    if os.environ.get("DEMO_SCAMRAG", "1") == "0":
        _scamrag["error"] = "turned off"
        return False
    if not _scamrag["loaded"]:
        _scamrag["loaded"] = True
        try:
            import webrag_system as W
            _scamrag["collection"] = W.get_kb_collection()
            _scamrag["module"] = W
        except Exception as e:                   # no venv, no store, no GPU
            _scamrag["error"] = "%s: %s" % (type(e).__name__, e)
    return _scamrag["module"] is not None


def check_llm_only(text):
    import llm_judge
    r = llm_judge.judge(text)
    return {
        "detector": "LLM-only",
        "label": None if r["verdict"] is None else ("Scam" if r["scam"] else "Legitimate"),
        "reason": r["reason"],
        "seconds": round(r["elapsed_ms"] / 1000.0, 1),
    }


def check_scamrag(text):
    W = _scamrag["module"]
    t0 = time.time()
    out = W.detect(text, _scamrag["collection"], use_web=False)
    top = out.get("kb_top_type") or ""
    patterns = [top.replace("_", " ")] if top else []
    return {
        "detector": "ScamRAG",
        "label": "Scam" if out.get("predicted") == "Fraud" else "Legitimate",
        "confidence": out.get("confidence"),
        "reason": out.get("reason", ""),
        "patterns_used": out.get("n_kb_kept"),
        "patterns": patterns,
        "seconds": round(time.time() - t0, 1),
    }


def run_check(text, detectors):
    if not _gpu.acquire(timeout=QUEUE_WAIT):
        raise BusyError("The demo is busy with other checks. Please try again in a minute.")
    try:
        results = []
        for d in detectors:
            try:
                if d == "llm_only":
                    results.append(check_llm_only(text))
                elif d == "scamrag" and scamrag_available():
                    results.append(check_scamrag(text))
            except Exception as e:
                results.append({"detector": d, "error": "this detector failed: %s" % type(e).__name__})
        return results
    finally:
        _gpu.release()


class BusyError(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "demo"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except ConnectionError:
            pass

    def client(self):
        # behind the tunnel every request comes from 127.0.0.1, so the visitor
        # is taken from the header the Worker sets, then Cloudflare's own
        for h in ("X-Demo-Client", "CF-Connecting-IP", "X-Forwarded-For"):
            v = self.headers.get(h)
            if v:
                return v.split(",")[0].strip()[:64]
        return self.client_address[0]

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            cfg = load_config()
            if u.path == "/":
                return self._send(200, PAGE, "text/html")
            if u.path == "/api/config":
                return self._send(200, {
                    "title": cfg.get("title", "Telephone scam detection demo"),
                    "intro": cfg.get("intro", ""),
                    "tables": cfg["tables"],
                    "examples": cfg["examples"],
                    "percall": [{"key": e["key"], "label": e.get("label", e["key"])} for e in cfg["percall"]],
                    "reproduce": cfg["reproduce"],
                    "code_url": cfg.get("code_url", ""),
                    "scamrag": scamrag_available(),
                    "limits": {"per_client": PER_CLIENT, "window_min": WINDOW // 60, "max_chars": MAX_CHARS},
                })
            if u.path == "/api/percall":
                offset = int((q.get("offset") or ["0"])[0] or 0)
                return self._send(200, percall_page(cfg, (q.get("key") or [""])[0], offset))
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": "server error: %s" % type(e).__name__})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/check":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "Please paste one call of up to %d characters." % MAX_CHARS})
            form = json.loads(self.rfile.read(n).decode("utf-8"))
            text = str(form.get("text") or "").strip()
            if not text:
                return self._send(400, {"error": "Please paste a call transcript first."})
            if len(text) > MAX_CHARS:
                return self._send(413, {"error": "Please paste one call of up to %d characters." % MAX_CHARS})
            ok, why = allow(self.client())
            if not ok:
                return self._send(429, {"error": why})
            detectors = [d for d in (form.get("detectors") or ["llm_only"])
                         if d in ("llm_only", "scamrag")][:2] or ["llm_only"]
            return self._send(200, {"results": run_check(text, detectors)})
        except BusyError as e:
            return self._send(503, {"error": str(e)})
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "That request could not be read."})
        except Exception as e:
            return self._send(500, {"error": "server error: %s" % type(e).__name__})

    def do_PUT(self):
        self._send(405, {"error": "not allowed"})

    do_DELETE = do_PATCH = do_PUT


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scam detection demo</title>
<style>
:root{--bg:#fafafa;--card:#fff;--ink:#1d1d1f;--mute:#5f6368;--line:#e2e2e2;--acc:#1a56db;--scam:#b42318;--ok:#067647}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--ink:#ececec;--mute:#a0a0a0;--line:#333;--acc:#6b9bff;--scam:#ff7b72;--ok:#56d364}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}p.lead{color:var(--mute);margin:0 0 20px}
nav{display:flex;gap:8px;border-bottom:1px solid var(--line);margin-bottom:20px;flex-wrap:wrap}
nav button{background:none;border:0;border-bottom:3px solid transparent;padding:10px 6px;font:inherit;color:var(--mute);cursor:pointer}
nav button.on{color:var(--ink);border-color:var(--acc);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:200px;font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button.main{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer}
button.main:disabled{opacity:.5;cursor:wait}
button.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:6px 12px;font:inherit;font-size:.9rem;color:var(--ink);cursor:pointer}
.verdict{font-size:1.3rem;font-weight:700}.scam{color:var(--scam)}.legit{color:var(--ok)}
.mute{color:var(--mute);font-size:.9rem}.err{color:var(--scam)}
table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}th{font-weight:600;border-bottom:2px solid var(--line)}
.scroll{overflow-x:auto}td.text{text-align:left;max-width:420px;white-space:pre-wrap}
ol li{margin-bottom:8px}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:.9em}
[hidden]{display:none!important}
</style></head><body><main>
<h1 id="title">Scam detection demo</h1>
<p class="lead" id="intro"></p>
<nav>
 <button data-tab="try" class="on">Try it yourself</button>
 <button data-tab="results">Results in the paper</button>
 <button data-tab="repro">Reproduce</button>
</nav>

<section id="try">
 <div class="card">
  <label for="text"><b>1. Paste a phone call transcript</b> <span class="mute">(or pick an example)</span></label>
  <div class="row" id="examples"></div>
  <textarea id="text" placeholder="Caller: Hello, this is ... &#10;Customer: ..."></textarea>
  <div class="row"><b>2. Choose detectors</b>
   <label><input type="checkbox" id="d_llm" checked> LLM-only</label>
   <label id="l_rag"><input type="checkbox" id="d_rag" checked> ScamRAG</label></div>
  <div class="row"><button class="main" id="go">3. Check call</button>
   <span class="mute" id="limits"></span></div>
 </div>
 <div id="out"></div>
 <p class="mute">Nothing you paste is stored. Each check runs on one desktop GPU and may take up to a minute.</p>
</section>

<section id="results" hidden>
 <div id="tables"></div>
 <div class="card" id="pc_card" hidden>
  <h3>Per-call answers</h3>
  <div class="row"><select id="pc_key"></select><span class="mute" id="pc_info"></span></div>
  <div class="scroll"><table id="pc"></table></div>
  <div class="row"><button class="chip" id="pc_prev">Previous</button><button class="chip" id="pc_next">Next</button></div>
 </div>
</section>

<section id="repro" hidden><div class="card"><ol id="steps"></ol><p id="code"></p></div></section>
</main>
<script>
const $=s=>document.querySelector(s);const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let CFG=null,pcOff=0;
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
 document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x===b));
 for(const id of ["try","results","repro"])$("#"+id).hidden=id!==b.dataset.tab;});
async function init(){
 CFG=await (await fetch("/api/config")).json();
 $("#title").textContent=CFG.title;document.title=CFG.title;$("#intro").textContent=CFG.intro;
 $("#limits").textContent=`Up to ${CFG.limits.per_client} checks per ${CFG.limits.window_min} minutes.`;
 if(!CFG.scamrag){$("#l_rag").hidden=true;$("#d_rag").checked=false;}
 CFG.examples.forEach(e=>{const b=document.createElement("button");b.className="chip";b.textContent=e.label;
  b.onclick=()=>{$("#text").value=e.text;};$("#examples").appendChild(b);});
 $("#tables").innerHTML=CFG.tables.map(t=>`<div class="card"><h3>${esc(t.title)}</h3><div class="scroll"><table>
  <tr>${t.header.map(h=>`<th>${esc(h)}</th>`).join("")}</tr>
  ${t.rows.map(r=>`<tr>${r.map(c=>`<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table></div>
  ${t.note?`<p class="mute">${esc(t.note)}</p>`:""}</div>`).join("");
 if(CFG.percall.length){$("#pc_card").hidden=false;
  $("#pc_key").innerHTML=CFG.percall.map(p=>`<option value="${esc(p.key)}">${esc(p.label)}</option>`).join("");
  $("#pc_key").onchange=()=>{pcOff=0;loadPc();};loadPc();}
 $("#steps").innerHTML=CFG.reproduce.map(s=>`<li>${esc(s)}</li>`).join("");
 if(CFG.code_url)$("#code").innerHTML=`Code and data: <a href="${esc(CFG.code_url)}" target="_blank" rel="noopener">${esc(CFG.code_url)}</a>`;
}
async function loadPc(){
 const r=await (await fetch(`/api/percall?key=${encodeURIComponent($("#pc_key").value)}&offset=${pcOff}`)).json();
 if(r.error){$("#pc").innerHTML=`<tr><td class="err">${esc(r.error)}</td></tr>`;return;}
 $("#pc_info").textContent=`Calls ${r.offset+1} to ${Math.min(r.offset+r.rows.length,r.total)} of ${r.total}. ${r.note||""}`;
 $("#pc").innerHTML=`<tr>${r.columns.map(c=>`<th>${esc(c)}</th>`).join("")}</tr>`+
  r.rows.map(row=>`<tr>${r.columns.map(c=>`<td class="${/text|why/.test(c)?"text":""}">${esc(row[c])}</td>`).join("")}</tr>`).join("");
 $("#pc_prev").onclick=()=>{pcOff=Math.max(0,r.offset-20);loadPc();};
 $("#pc_next").onclick=()=>{if(r.offset+20<r.total){pcOff=r.offset+20;loadPc();}};
}
$("#go").onclick=async()=>{
 const text=$("#text").value.trim();if(!text){$("#out").innerHTML=`<p class="err">Please paste a call first.</p>`;return;}
 const detectors=[];if($("#d_llm").checked)detectors.push("llm_only");if($("#d_rag").checked&&!$("#l_rag").hidden)detectors.push("scamrag");
 $("#go").disabled=true;$("#out").innerHTML=`<p class="mute">Checking the call...</p>`;
 try{const res=await fetch("/api/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,detectors})});
  const r=await res.json();
  if(r.error){$("#out").innerHTML=`<p class="err">${esc(r.error)}</p>`;return;}
  $("#out").innerHTML=r.results.map(x=>x.error?`<div class="card"><b>${esc(x.detector)}</b><p class="err">${esc(x.error)}</p></div>`:
   `<div class="card"><div class="mute">${esc(x.detector)}</div>
    <div class="verdict ${x.label==="Scam"?"scam":"legit"}">${esc(x.label??"No clear answer")}${x.confidence!=null?` <span class="mute">(confidence ${esc(x.confidence)} of 100)</span>`:""}</div>
    <p>${esc(x.reason)}</p>
    ${x.patterns&&x.patterns.length?`<p class="mute">Scam patterns used: ${x.patterns.map(esc).join("; ")}</p>`:""}
    <p class="mute">${esc(x.seconds)} seconds</p></div>`).join("");
 }catch(e){$("#out").innerHTML=`<p class="err">The demo could not be reached. Please try again.</p>`;}
 finally{$("#go").disabled=false;}
};
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="public, read-only demo of the detectors")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load_config()                                  # fail now, not on the first visit
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("demo on http://%s:%d  (config %s)" % (args.host, args.port, CONFIG_PATH), flush=True)
    print("limits: %d checks per visitor per %d min, %d per hour in total"
          % (PER_CLIENT, WINDOW // 60, PER_HOUR), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
