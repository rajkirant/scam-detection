#!/usr/bin/env bash
#
# web_ui.sh - start the browser front end for run_all.sh.
#
#   ./web_ui.sh                 # foreground, http://localhost:8000
#   ./web_ui.sh --port 8080
#   ./web_ui.sh --tmux          # detached, survives an SSH disconnect
#   ./web_ui.sh --local         # this machine only
#   ./web_ui.sh --public        # plus a public https link, via localhost.run
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
# --public also opens an SSH reverse tunnel to localhost.run, which needs no
# account and hands back an https URL that works from anywhere - useful for
# showing a run to someone off this network. The tunnel runs alongside the
# server and is closed when the server stops. Note what that URL exposes:
# anyone who opens it can start and stop runs on this machine and read every
# transcript, with no password in front of it.
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
PUBLIC=0
SESSION="scam_ui"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)    PORT="${2:-}";    shift 2 ;;
    --host)       HOST="${2:-}";    shift 2 ;;
    --local)      HOST=127.0.0.1;   shift   ;;
    --public)     PUBLIC=1;         shift   ;;
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

# localhost.run wants no account: it accepts any key for the "nokey" user and
# prints the public URL over the session. That output goes to a file so the
# URL can be picked out of it without tangling with the server's own output.
TUNNEL_PID=""
TUNNEL_LOG="$PROJECT_DIR/results/logs/tunnel.log"
start_tunnel() {
  command -v ssh >/dev/null || die "ssh not found, needed for --public"
  mkdir -p results/logs
  : > "$TUNNEL_LOG"
  # accept-new answers the first-connection host key prompt without turning
  # checking off; ExitOnForwardFailure makes a refused forward an error rather
  # than a tunnel that silently forwards nothing.
  ssh -T -o StrictHostKeyChecking=accept-new \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
         -o ExitOnForwardFailure=yes \
         -R 80:localhost:"$PORT" nokey@localhost.run \
      > "$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!

  say "Opening a public link"
  local url="" i
  for (( i = 0; i < 40; i++ )); do
    url="$(grep -oEm1 "https://[A-Za-z0-9._-]+\\.lhr\\.life" "$TUNNEL_LOG" 2>/dev/null)"
    [[ -z "$url" ]] \
      && url="$(grep -oEm1 "https://[A-Za-z0-9._-]{4,}" "$TUNNEL_LOG" 2>/dev/null)"
    [[ -n "$url" ]] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    sleep 1
  done

  if [[ -n "$url" ]]; then
    ok "public    $url"
    warn "no password in front of it - anyone with the link can start runs here"
  else
    warn "localhost.run did not hand back a URL, see $TUNNEL_LOG"
    warn "the server still starts, just without the public link"
  fi
}

stop_tunnel() {
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null
  return 0
}

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p results/logs
  UILOG="$PROJECT_DIR/results/logs/web_ui.log"
  # re-run this same script inside tmux, without --tmux, so the detached copy
  # sets up the tunnel exactly the way the foreground one does
  CMD="cd $(printf %q "$PROJECT_DIR") && exec bash web_ui.sh --port $PORT --host $HOST"
  [[ "$PUBLIC" -eq 1 ]] && CMD="$CMD --public"
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
  [[ "$PUBLIC" -eq 1 ]] && ok "public    the https link is printed in $UILOG"
  echo
  if [[ "$HOST" == "127.0.0.1" ]]; then
    echo "  from your laptop:  ssh -L $PORT:localhost:$PORT \$USER@\$(hostname)"
  fi
  echo "  stop the server:   tmux kill-session -t $SESSION   (runs keep going)"
  echo
  exit 0
fi

# No exec: the trap has to survive the server so the tunnel is closed with it.
trap stop_tunnel EXIT INT TERM
[[ "$PUBLIC" -eq 1 ]] && start_tunnel
python3 -u scripts/web_ui.py --port "$PORT" --host "$HOST"
