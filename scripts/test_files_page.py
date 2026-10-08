#!/usr/bin/env python3
"""
Offline tests for the web UI's Files page: browsing and reading the project's
files, read only.

The page can be public, so most of this is about what it must never hand
out: anything outside the project, hidden files (.env holds the API keys),
the virtualenv, files named like keys, and anything reached through a
symlink that leads to one of those. A temporary folder stands in for the
project, and the real request handler serves it on a local port.

    python scripts/test_files_page.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import web_ui                                                  # noqa: E402

fails = 0


def check(name, got, want=True):
    global fails
    ok = got == want
    fails += not ok
    print("  %-66s %s" % (name, "ok" if ok else "FAIL (got %r, want %r)"
                          % (got, want)))


def refused(fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        return str(e)
    return None


tmp = Path(tempfile.mkdtemp())
outside = Path(tempfile.mkdtemp())
(outside / "elsewhere.txt").write_text("not the project's\n")
root = tmp / "project"
(root / "knowledge" / "countries").mkdir(parents=True)
(root / "knowledge" / "countries" / "ireland.json").write_text('{"name": "Ireland"}\n')
(root / "datasets").mkdir()
(root / "datasets" / "calls.csv").write_text(
    "id,label,text\n" + "".join('c%d,scam,"line one\nline two %d"\n' % (i, i)
                                for i in range(120)))
(root / "README.md").write_text("# hello\n" + "x\n" * 10)
(root / "big.log").write_text("y" * (web_ui.FILES_TEXT_MAX + 10))
(root / "model.bin").write_bytes(b"\0\1\2" * 100)
(root / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 20)
(root / "page.html").write_text("<script>alert(1)</script>")
(root / ".env").write_text("TAVILY_API_KEY=secret\n")
(root / ".git").mkdir()
(root / ".git" / "config").write_text("[core]\n")
(root / "venv" / "lib").mkdir(parents=True)
(root / "venv" / "lib" / "site.py").write_text("")
(root / "id_rsa").write_text("PRIVATE KEY\n")
(root / "server.pem").write_text("PRIVATE KEY\n")
links = True
try:
    (root / "out").symlink_to(outside)
    (root / "env_link.txt").symlink_to(root / ".env")
except (OSError, NotImplementedError):
    links = False                 # Windows without the right to make them
web_ui.FILES_ROOT = root.resolve()

print("listing")
top = web_ui.files_list("")
names = [e["name"] for e in top["entries"]]
check("folders first, then files, by name whatever the case",
      names, ["datasets", "knowledge", "big.log", "chart.png", "model.bin",
              "page.html", "README.md"])
check("hidden files, the virtualenv and key files are left out",
      [n for n in (".env", ".git", "venv", "id_rsa", "server.pem") if n in names],
      [])
if links:
    check("so are symlinks to outside the project, or to .env",
          [n for n in ("out", "env_link.txt") if n in names], [])
    check("and a search does not find them",
          [m["path"] for m in web_ui.files_find("o")["matches"]
           if m["path"] in ("out", "env_link.txt")], [])
check("a folder deep down", [e["name"] for e in
                             web_ui.files_list("knowledge/countries")["entries"]],
      ["ireland.json"])
r = web_ui.files_list("knowledge/countries/ireland.json")
check("a file lists its folder and says which file is open",
      (r["path"], r["open"]), ("knowledge/countries", "knowledge/countries/ireland.json"))
check("a Windows-style path works too",
      web_ui.files_list("knowledge\\countries")["path"], "knowledge/countries")

print("\nwhat is never handed out")
for bad in ("..", "../", "knowledge/../../", "/etc/passwd", "C:/Windows",
            ".env", ".git/config", "venv/lib/site.py", "id_rsa", "server.pem",
            "knowledge/./countries"):
    check("refused: %s" % bad, refused(web_ui.files_read, bad) is not None)
if links:
    check("refused: a symlink that leads outside the project",
          refused(web_ui.files_read, "out/elsewhere.txt") is not None)
    check("refused: a symlink that leads to .env",
          refused(web_ui.files_read, "env_link.txt") is not None)
check("refused: a file that is not there",
      "no such" in (refused(web_ui.files_read, "nope.txt") or ""))
found = [m["path"] for m in web_ui.files_find("e")["matches"]]
check("a search never turns up hidden files or the virtualenv",
      [p for p in found if p.startswith((".", "venv")) or "/." in p
       or p in ("id_rsa", "server.pem")], [])

print("\nreading")
r = web_ui.files_read("README.md")
check("a text file", (r["kind"], r["text"].splitlines()[0], r["truncated"]),
      ("text", "# hello", False))
r = web_ui.files_read("big.log")
check("a big text file: the first 1 MB, marked as cut short",
      (len(r["text"]), r["truncated"]), (web_ui.FILES_TEXT_MAX, True))
r = web_ui.files_read("datasets/calls.csv")
check("a CSV: a page of rows, the header, and how many rows in all",
      (r["kind"], r["header"], len(r["rows"]), r["total"], r["rows"][0][2]),
      ("csv", ["id", "label", "text"], 50, 120, "line one\nline two 0"))
r = web_ui.files_read("datasets/calls.csv", 100)
check("a CSV's last page", (r["offset"], len(r["rows"]), r["rows"][0][0]),
      (100, 20, "c100"))
check("a CSV as plain text",
      web_ui.files_read("datasets/calls.csv", 0, True)["kind"], "text")
check("a binary file and an image are not sent as text",
      [web_ui.files_read(p)["kind"] for p in ("model.bin", "chart.png")],
      ["binary", "image"])
check("a folder is not read as a file",
      "folder" in (refused(web_ui.files_read, "datasets") or ""))
found = [m["path"] for m in web_ui.files_find("IRE")["matches"]]
check("a search finds a file by part of its name, any case",
      found, ["knowledge/countries/ireland.json"])

print("\nover HTTP")
srv = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = "http://127.0.0.1:%d" % srv.server_address[1]


def get(path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


code, _, body = get("/api/files/list?path=knowledge")
check("the listing", (code, json.loads(body)["path"]), (200, "knowledge"))
code, h, body = get("/api/files/raw?path=page.html")
check("an .html file downloads, sandboxed, never shown as a page",
      (code, h["Content-Type"], h["Content-Disposition"].split(";")[0],
       h["Content-Security-Policy"], h["X-Content-Type-Options"]),
      (200, "application/octet-stream", "attachment", "sandbox", "nosniff"))
code, h, body = get("/api/files/raw?path=chart.png")
check("an image is shown inline, as itself",
      (code, h["Content-Type"], h["Content-Disposition"].split(";")[0],
       body[:4]), (200, "image/png", "inline", b"\x89PNG"))
for bad in (".env", "..%2F..%2Fetc%2Fpasswd", "venv%2Flib%2Fsite.py"):
    code, _, body = get("/api/files/raw?path=" + bad)
    check("raw refuses %s" % bad.replace("%2F", "/"),
          (code, b"secret" in body, b"root:" in body), (400, False, False))
code, _, body = get("/api/files/read?path=.env")
check("read refuses .env", (code, b"secret" in body), (400, False))
code, _, body = get("/api/files/find?q=env")
check("find never names .env",
      (code, [m["path"] for m in json.loads(body)["matches"]
              if m["path"].endswith(".env")]), (200, []))
srv.shutdown()

print()
if fails:
    print("%d FAILED" % fails)
    sys.exit(1)
print("all good - the Files page reads the project and nothing hidden")
