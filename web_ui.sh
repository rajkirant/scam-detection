#!/usr/bin/env bash
#
# web_ui.sh - start the browser front end for run_all.sh.
#
#   ./web_ui.sh                 # foreground, http://localhost:8000
#   ./web_ui.sh --port 8080
#   ./web_ui.sh --tmux          # detached, survives an SSH disconnect
#   ./web_ui.sh --local         # this machine only
#
# It binds every interface, so another machine on the same network can open
# it directly - the startup banner prints the address to use. Anyone who can
# reach the port can start and stop runs on this box, so on an untrusted
# network use --local and forward the port instead:
#
#   ssh -L 8000:localhost:8000 user@your-gpu-host
#
# then open http://localhost:8000
#
# --local is also the one to use behind a tunnel that runs as a service of its
# own (cloudflared pointed at http://localhost:8000, say): the tunnel reaches
# the server on this machine, and nothing else on the network can. This script
# neither starts nor stops a tunnel.
#
# Benchmark runs started from the page are detached from this server, so
# stopping it does not stop a run that is already going.
#
set -uo pipefail

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
say()  { echo -e "\n${CYN}==>${NC} $*"; }
ok()   { echo -e "${GRN}  ok${NC} $*"; }
warn() { echo -e "${YLW}  warn${NC} $*"; }
die()  { echo -e "${RED}  fail${NC} $*"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || die "cannot cd to $PROJECT_DIR"

PORT=8000
HOST=0.0.0.0        # every interface: other machines can reach it
DETACH=0
SESSION="scam_ui"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)    PORT="${2:-}";    shift 2 ;;
    --host)       HOST="${2:-}";    shift 2 ;;
    --local)      HOST=127.0.0.1;   shift   ;;
    -t|--tmux)    DETACH=1;         shift   ;;
    -s|--session) SESSION="${2:-}"; DETACH=1; shift 2 ;;
    -h|--help)    awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
    *)            die "unknown argument: $1  (try --help)" ;;
  esac
done

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate || die "venv failed"
elif [[ -f venv/Scripts/activate ]]; then
  # shellcheck disable=SC1091
  source venv/Scripts/activate || die "venv failed"
else
  warn "no venv/ found, using whatever python is on PATH"
fi

# web_ui.py is standard library only, so the venv is a convenience, not a
# requirement - but the runs it launches do need it, and they inherit it.
command -v python3 >/dev/null || die "python3 not found"

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p results/logs
  UILOG="$PROJECT_DIR/results/logs/web_ui.log"
  # re-run this same script inside tmux, without --tmux, so the detached copy
  # starts the server exactly the way the foreground one does
  CMD="cd $(printf %q "$PROJECT_DIR") && exec bash web_ui.sh --port $PORT --host $HOST"
  say "Starting the UI detached"
  if command -v tmux >/dev/null; then
    tmux has-session -t "$SESSION" 2>/dev/null \
      && die "session $SESSION already exists - tmux kill-session -t $SESSION"
    tmux new-session -d -s "$SESSION" "bash -lc $(printf %q "$CMD 2>&1 | tee -a $(printf %q "$UILOG")")" \
      || die "tmux could not start session $SESSION"
    ok "session   $SESSION"
  else
    warn "tmux is not installed, falling back to setsid + nohup"
    setsid nohup bash -c "$CMD" >> "$UILOG" 2>&1 &
    ok "pid       $!"
  fi
  ok "url       http://localhost:$PORT"
  if [[ "$HOST" == "127.0.0.1" ]]; then
    ok "reach     this machine only"
  else
    LAN="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "$LAN" ]] && ok "elsewhere http://$LAN:$PORT"
  fi
  ok "log       $UILOG"
  echo
  if [[ "$HOST" == "127.0.0.1" ]]; then
    echo "  from your laptop:  ssh -L $PORT:localhost:$PORT \$USER@\$(hostname)"
  fi
  echo "  stop the server:   tmux kill-session -t $SESSION   (runs keep going)"
  echo
  exit 0
fi

# Is something already listening on $PORT? Asked before the server starts, so
# the answer is what holds the port and how to stop it, rather than a
# traceback about an address already in use.
port_in_use() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# Whatever is holding the port, if this machine will say.
port_owner() {
  local pid=""
  command -v ss >/dev/null \
    && pid="$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
  [[ -z "$pid" ]] && command -v lsof >/dev/null \
    && pid="$(lsof -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -1)"
  echo "$pid"
}

if port_in_use "$PORT"; then
  owner="$(port_owner "$PORT")"
  what=""
  [[ -n "$owner" ]] && what="$(ps -o args= -p "$owner" 2>/dev/null | cut -c1-80)"
  echo
  warn "port $PORT is already in use${what:+ by: $what}"
  if [[ "$what" == *web_ui.py* ]]; then
    echo "  That is an earlier copy of this UI, still running."
    echo "  Use that one, or stop it and start again:"
  else
    echo "  Stop whatever it is, or pick another port with --port."
  fi
  tmux has-session -t "$SESSION" 2>/dev/null \
    && echo "      tmux kill-session -t $SESSION"
  [[ -n "$owner" ]] && echo "      kill $owner"
  [[ -z "$owner" ]] && echo "      ss -ltnp | grep :$PORT     (to find it)"
  echo
  die "not starting a second server on port $PORT"
fi

# exec, so the server is this process: ctrl-c and tmux kill-session reach it
# directly, and once it is running bash is no longer reading this file - the
# Update button can git pull a new copy of it at any time.
exec python3 -u scripts/web_ui.py --port "$PORT" --host "$HOST"
