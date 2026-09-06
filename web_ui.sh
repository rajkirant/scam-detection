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
#   ssh -L 8000:localhost:8000 rkt29@cs25003ay
#
# then open http://localhost:8000
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
  CMD="cd $(printf %q "$PROJECT_DIR") && exec python3 -u scripts/web_ui.py --port $PORT --host $HOST"
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

exec python3 -u scripts/web_ui.py --port "$PORT" --host "$HOST"
