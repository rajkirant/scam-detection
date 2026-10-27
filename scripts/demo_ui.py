#!/usr/bin/env python3
"""
demo_ui.py - a small, read-only demo of the detectors for readers of the paper.

It is a separate server from web_ui.py, so the full UI (runs, training, the
file browser, the MCQ editor) stays private while this one is public. It has
four pages and nothing else:

  Try it yourself   paste a call transcript, or pick an example, and get the
                    verdict and the reason from LLM-only (and from ScamRAG,
                    when its pattern store is available on this machine)
  Run the tests     run the content-deletion test and the held-out test on a
                    small random sample, with one detector, and see every
                    call it got wrong: false alarms and missed scams
  Results           the tables of the paper, and per-call answers from the
                    result files listed in demo/demo_config.json
  Reproduce         the steps to reproduce the results, and the code link

Protection for a public link:
  - no endpoint writes to disk, starts a benchmark run, trains a saved model
    or reads a file that is not named in demo/demo_config.json
  - each visitor may submit DEMO_PER_CLIENT checks per DEMO_WINDOW seconds,
    and the whole demo DEMO_PER_HOUR checks per hour
  - each visitor may start DEMO_TESTS_PER_CLIENT test runs per DEMO_WINDOW
    seconds, of at most DEMO_MAX_SAMPLE calls; one test runs at a time and at
    most DEMO_TEST_QUEUE wait behind it
  - one request uses the GPU at a time; a check that cannot start within
    DEMO_QUEUE_WAIT seconds is told the demo is busy
  - transcripts longer than DEMO_MAX_CHARS characters are refused
  - nothing a visitor submits is stored, and test results are kept in memory
    for two hours only

Run it with web_ui.sh --demo (port 8081, this machine only by default), and
put it behind a tunnel. LLM-only and Length-only need only the standard
library; BoW needs scikit-learn and ScamRAG needs chromadb (both in the venv),
and each is left out when it cannot be loaded.

    ./web_ui.sh --demo
    python3 scripts/demo_ui.py --port 8081 --host 127.0.0.1

DEMO_FAKE_LLM=1 replaces the LLM with a keyword rule, to try the page on a
machine without Ollama. Never use it for the public demo.
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import secrets
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


def _env(name, default):
    return int(os.environ.get(name, str(default)))


PER_CLIENT = _env("DEMO_PER_CLIENT", 100)          # checks per visitor ...
WINDOW = _env("DEMO_WINDOW", 600)                  # ... per 10 minutes
PER_HOUR = _env("DEMO_PER_HOUR", 1000)             # checks for everyone, per hour
TESTS_PER_CLIENT = _env("DEMO_TESTS_PER_CLIENT", 100)   # test runs per visitor per window
TEST_QUEUE = _env("DEMO_TEST_QUEUE", 20)            # test runs that may wait
MAX_SAMPLE = _env("DEMO_MAX_SAMPLE", 40)           # calls in one test run
QUEUE_WAIT = _env("DEMO_QUEUE_WAIT", 90)           # seconds a check waits for the GPU
MAX_CHARS = _env("DEMO_MAX_CHARS", 6000)
FAKE_LLM = os.environ.get("DEMO_FAKE_LLM") == "1"
MAX_BODY = MAX_CHARS * 4 + 1024
PAGE_ROWS = 20
JOB_TTL = 2 * 3600
LENGTH_THRESHOLD = 45                              # the paper's length-only rule

csv.field_size_limit(10 ** 9)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DEFAULT_TESTS = {
    "deletion": {
        "label": "Content-deletion test",
        "dataset": "datasets/huggingface_1600.csv",
        "dataset_name": "Dialogue-1600",
        "train": "rest",                 # learners train on the calls not sampled
    },
    "heldout": {
        "label": "Held-out test",
        "dataset": "datasets/scambait_huggingface_real_tech_support_only_200.csv",
        "dataset_name": "Support-200",
        "train": "datasets/zhi_english_tech_support_removed_682.csv",
        "train_name": "Zhi-628",
    },
}


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("tables", [])
    cfg.setdefault("examples", [])
    cfg.setdefault("percall", [])
    cfg.setdefault("reproduce", [])
    tests = {k: dict(v) for k, v in DEFAULT_TESTS.items()}
    for k, v in (cfg.get("tests") or {}).items():
        if k in tests and isinstance(v, dict):
            tests[k].update(v)
    cfg["tests"] = tests
    cfg.setdefault("paper_values", {})
    return cfg


def project_file(rel):
    """A path from the config, inside the project; visitors never name one."""
    path = (ROOT / rel).resolve()
    if ROOT not in path.parents:
        raise ValueError("listed file is outside the project")
    if not path.is_file():
        raise ValueError("%s is not on this machine" % rel)
    return path


def percall_page(cfg, key, offset):
    for entry in cfg["percall"]:
        if entry.get("key") == key:
            break
    else:
        raise ValueError("no such result file")
    with open(project_file(entry["file"]), encoding="utf-8", newline="") as f:
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
_events = {"check": defaultdict(deque), "test": defaultdict(deque)}
_all_checks = deque()
_gpu = threading.Semaphore(1)


class BusyError(Exception):
    pass


def allow(kind, client):
    """True if this visitor, and the demo as a whole, are under their limits.
    Counted when a request is accepted, so a refused one costs nothing."""
    now = time.time()
    limit = PER_CLIENT if kind == "check" else TESTS_PER_CLIENT
    noun = "checks" if kind == "check" else "test runs"
    with _lock:
        mine = _events[kind][client]
        while mine and now - mine[0] > WINDOW:
            mine.popleft()
        while _all_checks and now - _all_checks[0] > 3600:
            _all_checks.popleft()
        if len(mine) >= limit:
            wait = int(WINDOW - (now - mine[0])) + 1
            return False, ("You have used your %d %s for now. Please try again in about %d minutes."
                           % (limit, noun, max(1, wait // 60)))
        if kind == "check":
            if len(_all_checks) >= PER_HOUR:
                return False, "The demo has reached its hourly limit. Please try again later."
            _all_checks.append(now)
        mine.append(now)
        table = _events[kind]
        if len(table) > 10000:                   # forget idle visitors
            for k in [k for k, v in table.items() if not v]:
                del table[k]
        return True, ""


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
# Loading ScamRAG (chromadb and the embedding model) and scikit-learn takes
# several seconds, so both are loaded once in the background when the server
# starts. The page never waits for them: it is told what is ready so far.
_scamrag = {"done": threading.Event(), "collection": None, "module": None, "error": ""}
_bow = {"done": threading.Event(), "ok": False}


def preload():
    try:
        import sklearn  # noqa: F401
        _bow["ok"] = True
    except ImportError:
        pass
    _bow["done"].set()
    if os.environ.get("DEMO_SCAMRAG", "1") == "0" or FAKE_LLM:
        _scamrag["error"] = "turned off"
    else:
        try:
            import webrag_system as W
            _scamrag["collection"] = W.get_kb_collection()
            _scamrag["module"] = W
        except Exception as e:                   # no venv, no store, no GPU
            _scamrag["error"] = "%s: %s" % (type(e).__name__, e)
    _scamrag["done"].set()


def scamrag_state():
    """"loading", "ready" or "off", without waiting."""
    if not _scamrag["done"].is_set():
        return "loading"
    return "ready" if _scamrag["module"] is not None else "off"


def scamrag_available():
    """For a check: waits for the background load if it is still going."""
    _scamrag["done"].wait(120)
    return _scamrag["module"] is not None


def bow_available(wait=True):
    if wait:
        _bow["done"].wait(60)
    elif not _bow["done"].is_set():
        return True                              # offer it while sklearn loads
    return _bow["ok"]


def _fake_judge(text):
    """For trying the page without Ollama only (DEMO_FAKE_LLM=1)."""
    t = text.lower()
    hit = [w for w in ("gift card", "anydesk", "remote", "virus", "pin", "password",
                       "arrest", "refund", "bitcoin", "verify") if w in t]
    return {"verdict": "Fraud" if hit else "Normal", "scam": bool(hit),
            "reason": "(fake LLM) " + (", ".join(hit) if hit else "no keyword found"),
            "elapsed_ms": 5}


def llm_judge_once(text):
    if FAKE_LLM:
        time.sleep(0.05)
        return _fake_judge(text)
    import llm_judge
    return llm_judge.judge(text)


def check_llm_only(text):
    r = llm_judge_once(text)
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
    return {
        "detector": "ScamRAG",
        "label": "Scam" if out.get("predicted") == "Fraud" else "Legitimate",
        "confidence": out.get("confidence"),
        "reason": out.get("reason", ""),
        "patterns": [top.replace("_", " ")] if top else [],
        "seconds": round(time.time() - t0, 1),
    }


def run_check(text, detectors):
    if not _gpu.acquire(timeout=QUEUE_WAIT):
        raise BusyError("The demo is busy. Please try again in a minute.")
    try:
        results = []
        for d in detectors:
            try:
                if d == "llm_only":
                    results.append(check_llm_only(text))
                elif d == "scamrag" and scamrag_available():
                    results.append(check_scamrag(text))
            except Exception as e:
                results.append({"detector": d, "error": friendly_error(e)})
        return results
    finally:
        _gpu.release()


def friendly_error(e):
    msg = str(e).lower()
    if "connection" in msg or "refused" in msg or "unreachable" in msg or "ollama" in msg:
        return "The language model is not available right now. Please try again later."
    return "This detector failed (%s)." % type(e).__name__


# ---------------------------------------------------------------------------
# the two tests on a small sample
# ---------------------------------------------------------------------------
TEST_DETECTORS = {"length": "Length-only", "bow": "BoW", "llm_only": "LLM-only"}
_data_cache = {}
_bow_cache = {}


def load_calls(rel):
    """[(id, text, is_scam)] from a dataset named in the config."""
    if rel not in _data_cache:
        import dataset_io as D
        path = project_file(rel)
        rows = D.read_rows(path)
        tcol, lcol, icol = D.columns(rows, str(path))
        calls = []
        for n, r in enumerate(rows):
            text = (r.get(tcol) or "").strip()
            if text:
                cid = (r.get(icol) if icol else None) or str(n)
                calls.append((str(cid), text, bool(D.is_scam(r[lcol]))))
        _data_cache[rel] = calls
    return _data_cache[rel]


def balanced_sample(calls, size, seed):
    rng = random.Random(seed)
    scam = [c for c in calls if c[2]]
    legit = [c for c in calls if not c[2]]
    k = min(size // 2, len(scam), len(legit))
    pick = rng.sample(scam, k) + rng.sample(legit, k)
    rng.shuffle(pick)
    return pick


def train_bow(calls):
    """TF-IDF on words and word pairs into logistic regression: the BoW
    baseline of the paper, with min_df=2 and the same fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    texts = [c[1] for c in calls]
    y = [1 if c[2] else 0 for c in calls]
    for min_df in (2, 1):
        try:
            return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=min_df),
                                 LogisticRegression(max_iter=2000)).fit(texts, y)
        except ValueError:
            continue
    raise ValueError("BoW could not be trained")


class Predictor:
    """One detector, trained if it needs training, answering one text at a time."""

    def __init__(self, detector, train_calls=None, cache_key=None):
        self.detector = detector
        self.model = None
        if detector == "bow":
            if cache_key and cache_key in _bow_cache:
                self.model = _bow_cache[cache_key]
            else:
                self.model = train_bow(train_calls)
                if cache_key:
                    _bow_cache[cache_key] = self.model

    def __call__(self, text):
        """(is_scam, reason)"""
        if self.detector == "length":
            n = len(text.split())
            return n > LENGTH_THRESHOLD, "%d words (scam if more than %d)" % (n, LENGTH_THRESHOLD)
        if self.detector == "bow":
            p = float(self.model.predict_proba([text])[0][1])
            return p >= 0.5, "scam probability %.2f" % p
        if not _gpu.acquire(timeout=900):
            raise BusyError("The GPU stayed busy for too long.")
        try:
            r = llm_judge_once(text)
        finally:
            _gpu.release()
        if r["verdict"] is None:                 # scored Normal, as in the benchmark
            return False, "(answer could not be read, counted as legitimate) " + (r.get("reason") or "")
        return bool(r["scam"]), r.get("reason") or ""


def outcome(true_scam, pred_scam):
    if true_scam == pred_scam:
        return "correct"
    return "missed scam" if true_scam else "false alarm"


def counts(rows, key):
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in rows:
        t, p = r["true_scam"], r[key]
        c["tp" if t and p else "fn" if t else "fp" if p else "tn"] += 1
    return c


def accuracy(c):
    n = sum(c.values())
    return 100.0 * (c["tp"] + c["tn"]) / n if n else 0.0


class Job:
    def __init__(self, client, params):
        self.id = secrets.token_urlsafe(12)
        self.client = client
        self.params = params
        self.status = "queued"
        self.done = 0
        self.total = 0
        self.created = time.time()
        self.error = ""
        self.results = []

    def view(self):
        ahead = 0
        if self.status == "queued":
            with _jobs_lock:
                ahead = sum(1 for j in _jobs.values()
                            if j.status in ("queued", "running") and j.created < self.created)
        return {"id": self.id, "status": self.status, "done": self.done, "total": self.total,
                "ahead": ahead, "error": self.error, "params": self.params, "results": self.results}


_jobs = {}
_jobs_lock = threading.Lock()
_job_queue = queue.Queue()


def start_job(client, form, cfg):
    test = form.get("test")
    detector = form.get("detector")
    if test not in ("deletion", "heldout", "both"):
        raise ValueError("unknown test")
    if detector not in TEST_DETECTORS:
        raise ValueError("unknown detector")
    if detector == "bow" and not bow_available():
        raise ValueError("BoW is not available on this machine")
    size = int(form.get("size") or 20)
    size = max(4, min(MAX_SAMPLE, size - size % 2))
    seed = int(form.get("seed") or 1) % 1000000
    with _jobs_lock:
        now = time.time()
        for k in [k for k, j in _jobs.items() if now - j.created > JOB_TTL]:
            del _jobs[k]
        waiting = sum(1 for j in _jobs.values() if j.status in ("queued", "running"))
        if waiting > TEST_QUEUE:
            raise BusyError("Several tests are already waiting. Please try again in a few minutes.")
    ok, why = allow("test", client)
    if not ok:
        raise PermissionError(why)
    job = Job(client, {"test": test, "detector": detector, "size": size, "seed": seed})
    with _jobs_lock:
        _jobs[job.id] = job
    _job_queue.put(job)
    return job


def run_one_test(job, name, spec, predictor_for, cfg):
    p = job.params
    calls = load_calls(spec["dataset"])
    sample = balanced_sample(calls, p["size"], p["seed"])
    predict = predictor_for(name, sample)
    strip = None
    if name == "deletion":
        from trusted import strip_content_words as strip
    rows = []
    for cid, text, true_scam in sample:
        pred, why = predict(text)
        job.done += 1
        row = {"id": cid, "true_scam": true_scam, "pred": pred, "why": why,
               "outcome": outcome(true_scam, pred), "text": text[:4000]}
        if strip:
            stripped = strip(text)
            pred_s, why_s = predict(stripped)
            job.done += 1
            row.update({"text_stripped": stripped[:4000], "pred_stripped": pred_s,
                        "why_stripped": why_s, "outcome_stripped": outcome(true_scam, pred_s)})
        rows.append(row)
    c = counts(rows, "pred")
    summary = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
               "accuracy": round(accuracy(c), 1)}
    if name == "deletion":
        cs = counts(rows, "pred_stripped")
        a_full, a_str = accuracy(c), accuracy(cs)
        summary.update({"accuracy_stripped": round(a_str, 1),
                        "trusted": round(a_full - max(0.0, a_str - 50.0), 1)})
    else:
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        spec_ = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0
        summary.update({"recall": round(recall, 2), "specificity": round(spec_, 2),
                        "nsdi": round(100 * math.sqrt(recall * spec_), 1)})
    paper = ((cfg.get("paper_values") or {}).get(name) or {}).get(p["detector"])
    return {"test": name, "label": spec["label"], "dataset_name": spec.get("dataset_name", ""),
            "train_name": spec.get("train_name", ""), "summary": summary, "paper": paper,
            "rows": rows}


def run_job(job):
    cfg = load_config()
    p = job.params
    names = ["deletion", "heldout"] if p["test"] == "both" else [p["test"]]
    # answers to give: each call once, and once more stripped for the deletion test
    job.total = sum(min(p["size"], len(load_calls(cfg["tests"][n]["dataset"])))
                    * (2 if n == "deletion" else 1) for n in names)

    def predictor_for(name, sample):
        spec = cfg["tests"][name]
        if p["detector"] != "bow":
            return Predictor(p["detector"])
        if spec.get("train") == "rest":          # every call of the set that was not sampled
            ids = {c[0] for c in sample}
            train = [c for c in load_calls(spec["dataset"]) if c[0] not in ids]
            return Predictor("bow", train)
        return Predictor("bow", load_calls(spec["train"]), cache_key=spec["train"])

    for name in names:
        job.results.append(run_one_test(job, name, cfg["tests"][name], predictor_for, cfg))


def job_worker():
    while True:
        job = _job_queue.get()
        job.status = "running"
        try:
            run_job(job)
            job.status = "done"
        except (Exception, SystemExit) as e:      # dataset_io exits on a bad file
            job.status = "error"
            job.error = (str(e) if isinstance(e, (ValueError, BusyError, SystemExit))
                         else friendly_error(e))


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

    def _form(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise OverflowError
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            cfg = load_config()
            if u.path == "/":
                return self._send(200, PAGE, "text/html")
            if u.path == "/api/config":
                detectors = [{"key": k, "label": v} for k, v in TEST_DETECTORS.items()
                             if k != "bow" or bow_available(wait=False)]
                return self._send(200, {
                    "title": cfg.get("title", "Telephone scam detection demo"),
                    "intro": cfg.get("intro", ""),
                    "tables": cfg["tables"],
                    "examples": cfg["examples"],
                    "percall": [{"key": e["key"], "label": e.get("label", e["key"])} for e in cfg["percall"]],
                    "reproduce": cfg["reproduce"],
                    "code_url": cfg.get("code_url", ""),
                    "scamrag": scamrag_state(),
                    "tests": {k: {"label": v["label"], "dataset_name": v.get("dataset_name", ""),
                                  "train_name": v.get("train_name", "")} for k, v in cfg["tests"].items()},
                    "test_detectors": detectors,
                    "fake_llm": FAKE_LLM,
                    "limits": {"per_client": PER_CLIENT, "window_min": WINDOW // 60,
                               "max_chars": MAX_CHARS, "tests_per_client": TESTS_PER_CLIENT,
                               "max_sample": MAX_SAMPLE},
                })
            if u.path == "/api/percall":
                offset = int((q.get("offset") or ["0"])[0] or 0)
                return self._send(200, percall_page(cfg, (q.get("key") or [""])[0], offset))
            if u.path == "/api/test":
                with _jobs_lock:
                    job = _jobs.get((q.get("id") or [""])[0])
                if not job:
                    return self._send(404, {"error": "This test run has expired. Please run it again."})
                return self._send(200, job.view())
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": "server error: %s" % type(e).__name__})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/check":
                form = self._form()
                text = str(form.get("text") or "").strip()
                if not text:
                    return self._send(400, {"error": "Please paste a call transcript first."})
                if len(text) > MAX_CHARS:
                    raise OverflowError
                ok, why = allow("check", self.client())
                if not ok:
                    return self._send(429, {"error": why})
                detectors = [d for d in (form.get("detectors") or ["llm_only"])
                             if d in ("llm_only", "scamrag")][:2] or ["llm_only"]
                return self._send(200, {"results": run_check(text, detectors)})
            if u.path == "/api/test":
                job = start_job(self.client(), self._form(), load_config())
                return self._send(200, job.view())
            return self._send(404, {"error": "not found"})
        except OverflowError:
            return self._send(413, {"error": "Please paste one call of up to %d characters." % MAX_CHARS})
        except PermissionError as e:
            return self._send(429, {"error": str(e)})
        except BusyError as e:
            return self._send(503, {"error": str(e)})
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e) or "That request could not be read."})
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
:root{--bg:#fafafa;--card:#fff;--ink:#1d1d1f;--mute:#5f6368;--line:#e2e2e2;--acc:#1a56db;--scam:#b42318;--ok:#067647;--warn:#b54708;--soft:#f2f4f7}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--ink:#ececec;--mute:#a0a0a0;--line:#333;--acc:#6b9bff;--scam:#ff7b72;--ok:#56d364;--warn:#f0b35a;--soft:#262626}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h3{margin:0 0 10px}p.lead{color:var(--mute);margin:0 0 20px}
nav{display:flex;gap:8px;border-bottom:1px solid var(--line);margin-bottom:20px;flex-wrap:wrap}
nav button{background:none;border:0;border-bottom:3px solid transparent;padding:10px 6px;font:inherit;color:var(--mute);cursor:pointer}
nav button.on{color:var(--ink);border-color:var(--acc);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:200px;font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
select,input[type=number]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.grid label{display:flex;flex-direction:column;gap:4px;font-weight:600;font-size:.92rem}
button.main{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer}
button.main:disabled{opacity:.5;cursor:wait}
button.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:6px 12px;font:inherit;font-size:.9rem;color:var(--ink);cursor:pointer}
button.chip.on{border-color:var(--acc);color:var(--acc);font-weight:600}
.verdict{font-size:1.3rem;font-weight:700}.scam{color:var(--scam)}.legit{color:var(--ok)}
.mute{color:var(--mute);font-size:.9rem}.err{color:var(--scam)}
table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}th{font-weight:600;border-bottom:2px solid var(--line)}
td.l,th.l{text-align:left}.scroll{overflow-x:auto}td.text{text-align:left;max-width:420px;white-space:pre-wrap}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.82rem;font-weight:600;white-space:nowrap}
.b-correct{background:#e7f6ec;color:#067647}.b-false{background:#fef0c7;color:#93370d}.b-missed{background:#fee4e2;color:#b42318}
@media (prefers-color-scheme:dark){.b-correct{background:#12391f;color:#7ee2a0}.b-false{background:#4a3410;color:#f7c873}.b-missed{background:#4c1714;color:#ff9c94}}
.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 12px}
.metric{background:var(--soft);border-radius:8px;padding:10px 14px;min-width:150px}
.metric b{display:block;font-size:1.4rem}.metric span{color:var(--mute);font-size:.85rem}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc);width:0}
tr.call{cursor:pointer}tr.detail td{background:var(--soft);text-align:left;white-space:pre-wrap;font-size:.88rem}
ol li{margin-bottom:8px}[hidden]{display:none!important}
</style></head><body><main>
<h1 id="title">Scam detection demo</h1>
<p class="lead" id="intro"></p>
<p class="err" id="fake" hidden>Test mode: the language model is replaced by a keyword rule. Do not use these answers.</p>
<nav>
 <button data-tab="try" class="on">Try it yourself</button>
 <button data-tab="tests">Run the two tests</button>
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
   <label id="l_rag" hidden><input type="checkbox" id="d_rag" checked> ScamRAG
    <span class="mute" id="rag_note"></span></label></div>
  <div class="row"><button class="main" id="go">3. Check call</button>
   <span class="mute" id="limits"></span></div>
 </div>
 <div id="out"></div>
 <p class="mute">Nothing you paste is stored. Each check runs on one desktop GPU and may take up to a minute.</p>
</section>

<section id="tests" hidden>
 <div class="card">
  <h3>Run the two tests on a small sample</h3>
  <p><b>Content-deletion test.</b> Every content word (nouns, verbs, adjectives, adverbs) is deleted from each call, and the detector answers again. Accuracy that survives above chance (50%) cannot come from what the caller said, and is subtracted: <i>trusted accuracy</i> = accuracy minus (accuracy after deletion minus 50), when the latter is above 50.</p>
  <p><b>Held-out test.</b> The detector is tested on technical-support scams, a scam type left out of everything it learned from. <i>NSDI</i> = 100 &times; &radic;(recall on the scams &times; specificity on the legitimate calls).</p>
  <div class="grid">
   <label>Test<select id="t_test"><option value="both">Both tests</option>
    <option value="deletion">Content-deletion test (Dialogue-1600)</option>
    <option value="heldout">Held-out test (Support-200)</option></select></label>
   <label>Detector<select id="t_det"><option value="length">Length-only</option>
    <option value="bow">BoW</option><option value="llm_only" selected>LLM-only</option></select></label>
   <label>Calls in the sample<select id="t_size"><option>10</option><option selected>20</option>
    <option>40</option></select></label>
   <label>Random seed<input type="number" id="t_seed" value="1" min="0" max="999999"></label>
  </div>
  <div class="row"><button class="main" id="t_go">Run test</button><span class="mute" id="t_hint"></span></div>
  <div id="t_prog" hidden><div class="row"><span id="t_state"></span></div><div class="bar"><i id="t_bar"></i></div></div>
  <p class="err" id="t_err"></p>
 </div>
 <div id="t_out"></div>
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
let CFG=null,pcOff=0,JOB=null,TIMER=null;
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
 document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x===b));
 for(const id of ["try","tests","results","repro"])$("#"+id).hidden=id!==b.dataset.tab;});
function hint(){const n=+$("#t_size").value,t=$("#t_test").value,d=$("#t_det").value;
 const calls=(t==="both"?3:t==="deletion"?2:1)*n;
 $("#t_hint").textContent=d==="llm_only"?`${calls} model answers, about ${Math.max(1,Math.round(calls*3/60))} minute(s). Half the sample is scams, half legitimate.`:"Takes a few seconds. Half the sample is scams, half legitimate.";}
["#t_size","#t_test","#t_det"].forEach(s=>$(s).onchange=hint);
function ragState(st){
 $("#l_rag").hidden=st==="off";$("#d_rag").disabled=st!=="ready";
 $("#rag_note").textContent=st==="loading"?"(loading...)":"";
 if(st==="loading")setTimeout(async()=>{try{const c=await (await fetch("/api/config")).json();ragState(c.scamrag);}catch(e){}},3000);
}
async function init(){
 hint();
 CFG=await (await fetch("/api/config")).json();
 $("#title").textContent=CFG.title;document.title=CFG.title;$("#intro").textContent=CFG.intro;$("#fake").hidden=!CFG.fake_llm;
 $("#limits").textContent=`Up to ${CFG.limits.per_client} checks per ${CFG.limits.window_min} minutes.`;
 ragState(CFG.scamrag);
 CFG.examples.forEach(e=>{const b=document.createElement("button");b.className="chip";b.textContent=e.label;
  b.onclick=()=>{$("#text").value=e.text;};$("#examples").appendChild(b);});
 const keep=$("#t_det").value;   /* drop BoW if this machine cannot run it, keep the choice */
 $("#t_det").innerHTML=CFG.test_detectors.map(d=>`<option value="${d.key}">${esc(d.label)}</option>`).join("");
 $("#t_det").value=CFG.test_detectors.some(d=>d.key===keep)?keep:"llm_only";hint();
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
 const detectors=[];if($("#d_llm").checked)detectors.push("llm_only");if($("#d_rag").checked&&!$("#l_rag").hidden&&!$("#d_rag").disabled)detectors.push("scamrag");
 $("#go").disabled=true;$("#out").innerHTML=`<p class="mute">Checking the call...</p>`;
 try{const res=await fetch("/api/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,detectors})});
  const r=await res.json();
  if(r.error){$("#out").innerHTML=`<p class="err">${esc(r.error)}</p>`;return;}
  $("#out").innerHTML=r.results.map(x=>x.error?`<div class="card"><b>${esc(x.detector)}</b><p class="err">${esc(x.error)}</p></div>`:
   `<div class="card"><div class="mute">${esc(x.detector)}</div>
    <div class="verdict ${x.label==="Scam"?"scam":"legit"}">${esc(x.label??"No clear answer")}${x.confidence!=null?` <span class="mute">(confidence ${esc(x.confidence)} of 100)</span>`:""}</div>
    <p>${esc(x.reason)}</p>
    ${x.patterns&&x.patterns.length?`<p class="mute">Closest scam pattern: ${x.patterns.map(esc).join("; ")}</p>`:""}
    <p class="mute">${esc(x.seconds)} seconds</p></div>`).join("");
 }catch(e){$("#out").innerHTML=`<p class="err">The demo could not be reached. Please try again.</p>`;}
 finally{$("#go").disabled=false;}
};

/* ---------------- the two tests ---------------- */
const lab=b=>b?"Scam":"Legitimate";
const badge=o=>o==="correct"?`<span class="badge b-correct">Correct</span>`:o==="false alarm"?`<span class="badge b-false">False alarm</span>`:`<span class="badge b-missed">Missed scam</span>`;
$("#t_go").onclick=async()=>{
 $("#t_err").textContent="";$("#t_out").innerHTML="";
 const body={test:$("#t_test").value,detector:$("#t_det").value,size:+$("#t_size").value,seed:+$("#t_seed").value||0};
 $("#t_go").disabled=true;
 try{const res=await fetch("/api/test",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const r=await res.json();if(r.error){$("#t_err").textContent=r.error;$("#t_go").disabled=false;return;}
  JOB=r.id;$("#t_prog").hidden=false;poll();
 }catch(e){$("#t_err").textContent="The demo could not be reached. Please try again.";$("#t_go").disabled=false;}
};
async function poll(){
 clearTimeout(TIMER);
 let r;try{r=await (await fetch("/api/test?id="+encodeURIComponent(JOB))).json();}catch(e){TIMER=setTimeout(poll,3000);return;}
 if(r.error&&!r.status){$("#t_err").textContent=r.error;$("#t_go").disabled=false;$("#t_prog").hidden=true;return;}
 const pct=r.total?Math.round(100*r.done/r.total):0;$("#t_bar").style.width=pct+"%";
 $("#t_state").textContent=r.status==="queued"?`Waiting for the GPU: ${r.ahead} test run(s) ahead.`:
  r.status==="running"?`Running: ${r.done} of ${r.total} answers.`:r.status==="done"?"Finished.":"Stopped.";
 if(r.status==="queued"||r.status==="running"){TIMER=setTimeout(poll,1500);return;}
 $("#t_go").disabled=false;
 if(r.status==="error"){$("#t_err").textContent=r.error;return;}
 $("#t_out").innerHTML=r.results.map((t,i)=>testCard(t,r.params,i)).join("");
 r.results.forEach((t,i)=>wire(t,i));
}
function metric(v,label){return `<div class="metric"><b>${esc(v)}</b><span>${esc(label)}</span></div>`;}
function testCard(t,p,i){
 const s=t.summary,det=CFG.test_detectors.find(d=>d.key===p.detector)?.label||p.detector,del=t.test==="deletion";
 const errs=t.rows.filter(x=>x.outcome!=="correct").length;
 let m=del?metric(s.accuracy+"%","accuracy on the calls as given")+metric(s.accuracy_stripped+"%","accuracy after content deletion")+metric(s.trusted,"trusted accuracy")
          :metric(s.recall.toFixed(2),"recall on the held-out scams")+metric(s.specificity.toFixed(2),"specificity on legitimate calls")+metric(s.nsdi,"NSDI");
 m+=metric(s.fp,"false alarms")+metric(s.fn,"missed scams");
 const paper=t.paper?`<p class="mute">In the paper, on all calls: ${esc(Object.entries(t.paper).map(([k,v])=>`${k} ${v}`).join(", "))}.</p>`:"";
 const read=del?(s.accuracy_stripped>55?`After deletion, ${det} still labels ${s.accuracy_stripped}% of the calls correctly, so part of its accuracy does not come from what is said in the calls.`
                 :`After deletion, ${det} falls to about chance, so this test finds no shortcut in this sample.`)
              :`${det} catches ${s.tp} of ${s.tp+s.fn} technical-support scams and leaves ${s.tn} of ${s.tn+s.fp} legitimate calls alone.`;
 return `<div class="card"><h3>${esc(t.label)}: ${esc(det)} on ${p.size} calls of ${esc(t.dataset_name)}</h3>
  ${t.train_name&&p.detector==="bow"?`<p class="mute">BoW was trained on ${esc(t.train_name)}, which has no technical-support calls.</p>`:""}
  ${del&&p.detector==="bow"?`<p class="mute">BoW was trained on the other ${esc(t.dataset_name)} calls, as given (never on deleted text).</p>`:""}
  <div class="metrics">${m}</div><p>${esc(read)}</p>${paper}
  <p class="mute">A sample of ${p.size} calls varies a lot from one seed to another; the paper uses every call.</p>
  <div class="row" id="f${i}"><b>Show</b>
   <button class="chip on" data-f="all">All calls (${t.rows.length})</button>
   <button class="chip" data-f="errors">Wrong answers (${errs})</button>
   <button class="chip" data-f="false alarm">False alarms (${s.fp})</button>
   <button class="chip" data-f="missed scam">Missed scams (${s.fn})</button></div>
  <div class="scroll"><table id="tt${i}"></table></div><p class="mute">Click "Show call" (or anywhere on a row) to see the call${del?" and its text after content deletion":""}.</p></div>`;
}
function wire(t,i){
 const del=t.test==="deletion";
 const draw=f=>{
  const rows=t.rows.filter(x=>f==="all"||(f==="errors"?x.outcome!=="correct":x.outcome===f));
  const head=`<tr><th class="l">Call</th><th class="l">True label</th><th class="l">Answer</th><th class="l">Result</th>${del?`<th class="l">After deletion</th>`:""}<th class="l">Reason</th><th class="l"></th></tr>`;
  $("#tt"+i).innerHTML=head+(rows.length?rows.map((x,k)=>`<tr class="call" data-k="${k}"><td>${esc(x.id)}</td><td class="l">${lab(x.true_scam)}</td>
   <td class="l ${x.pred?"scam":"legit"}">${lab(x.pred)}</td><td class="l">${badge(x.outcome)}</td>
   ${del?`<td class="l"><span class="${x.pred_stripped?"scam":"legit"}">${lab(x.pred_stripped)}</span> ${x.outcome_stripped==="correct"?"(right)":"(wrong)"}</td>`:""}
   <td class="text">${esc(x.why)}</td><td class="l"><button class="chip show">Show call</button></td></tr><tr class="detail" hidden><td colspan="${del?7:6}"><b>Call as given</b>\n${esc(x.text)}${del?`\n\n<b>After content deletion</b>\n${esc(x.text_stripped)}\n\n<b>Reason after deletion</b>\n${esc(x.why_stripped)}`:""}</td></tr>`).join("")
   :`<tr><td colspan="7" class="mute">No calls of this kind in the sample.</td></tr>`);
  $("#tt"+i).querySelectorAll("tr.call").forEach(tr=>tr.onclick=()=>{const d=tr.nextElementSibling;d.hidden=!d.hidden;
   tr.querySelector("button.show").textContent=d.hidden?"Show call":"Hide call";});
 };
 $("#f"+i).querySelectorAll("button").forEach(b=>b.onclick=()=>{$("#f"+i).querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));draw(b.dataset.f);});
 draw("all");
}
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="public, read-only demo of the detectors")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load_config()                                  # fail now, not on the first visit
    threading.Thread(target=job_worker, daemon=True).start()
    threading.Thread(target=preload, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("demo on http://%s:%d  (config %s)" % (args.host, args.port, CONFIG_PATH), flush=True)
    print("limits: %d checks per visitor per %d min, %d per hour in total; %d test runs per visitor per %d min, up to %d calls each"
          % (PER_CLIENT, WINDOW // 60, PER_HOUR, TESTS_PER_CLIENT, WINDOW // 60, MAX_SAMPLE), flush=True)
    if FAKE_LLM:
        print("DEMO_FAKE_LLM=1: the LLM is a keyword rule. Do not use this for the public demo.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
