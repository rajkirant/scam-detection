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
#   ssh -L 8000:localhost:8000 user@your-gpu-host
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
# localhost.run does not hold a connection forever, so the tunnel is
# supervised: when it drops, a new one is opened and the link is printed here
# again. An anonymous tunnel can also expire without the connection closing -
# the address stops serving while ssh sits there looking healthy - so the
# supervisor asks the public URL itself once a minute and reconnects when the
# answer stops coming, which prints the new link too.
# If this machine has an SSH key the address stays the same across
# those reconnects; without one every reconnect gets a fresh address, and the
# previous link stops working - so on a box you will share a link from,
#
#   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_lhr      (once)
#
# is worth doing - ~/.ssh/id_lhr is used for the tunnel in preference to the
# box's own key, which on a shared machine is often passphrase-protected or
# otherwise not one localhost.run will take. Any default key is still used if
# that file is absent. The whole history is in results/logs/tunnel.log.
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

# localhost.run needs no account: it accepts any key for the "nokey" user and
# prints the public URL over the session. That output goes to a file so the URL
# can be picked out of it without tangling with the server's own output.
#
# The connection does not last forever - localhost.run drops it, or the network
# blips - so ssh is run under a supervisor that reconnects instead of leaving a
# dead link behind. Two things make that work: -o ServerAliveInterval makes ssh
# notice a connection that has stopped answering rather than hanging on it, and
# -n takes the terminal away from ssh's stdin. Without -n a backgrounded ssh is
# stopped by SIGTTIN the moment it reads, which looks exactly like a drop.
TUNNEL_PID=""
TUNNEL_LOG="$PROJECT_DIR/results/logs/tunnel.log"
TUNNEL_PIDFILE="$PROJECT_DIR/results/logs/.tunnel.ssh.pid"
TUNNEL_TARGETFILE="$PROJECT_DIR/results/logs/.tunnel.target"
# An anonymous tunnel gets a new address every reconnect, so the link
# printed at startup goes stale. This file always holds the live one.
PUBLIC_URL_FILE="$PROJECT_DIR/results/logs/public_url.txt"
TUNNEL_TARGET="nokey@localhost.run"
# How often the live link is asked whether it still answers, and how many
# misses in a row mean it is gone rather than a blip.
TUNNEL_CHECK_EVERY=60
TUNNEL_CHECK_FAILS=2
LAST_CODE=""        # what the last url_answers call got back
# A key kept for this tunnel alone. Optional, and the reason it exists is that
# the default key on a shared box is often the one localhost.run will not take
# - a passphrase with no agent looks identical to a refusal from here - and
# regenerating that key is not a thing to do to fix a demo link.
TUNNEL_KEY="$HOME/.ssh/id_lhr"

# localhost.run gives a keyed connection the same subdomain every time, and the
# "nokey" user a fresh random one on every reconnect. A stable link matters
# more here than anonymity - a link handed to someone must survive a reconnect -
# so use a key when there is one. If that turns out not to connect, start_tunnel
# falls back to nokey.
pick_target() {
  local k
  # a key set aside for this tunnel, or an entry for localhost.run in ssh's own
  # config, each mean a keyed connection is wanted even with no default key
  [[ -f "$TUNNEL_KEY" ]] && { echo "${USER:-ui}@localhost.run"; return; }
  if grep -qiE "^[[:space:]]*Host[[:space:]].*localhost[.]run" ~/.ssh/config 2>/dev/null; then
    echo "${USER:-ui}@localhost.run"; return
  fi
  for k in ~/.ssh/id_ed25519 ~/.ssh/id_ecdsa ~/.ssh/id_rsa; do
    [[ -f "$k" ]] && { echo "${USER:-ui}@localhost.run"; return; }
  done
  echo "nokey@localhost.run"
}

# The URL of the most recent connection in the log. The announcement line
# ("<host> tunneled with tls termination, https://<host>") is the only line
# that names the tunnel - the welcome banner is full of localhost.run's own
# links, so matching any https:// would latch one of those instead.
latest_url() {
  local u
  u="$(sed -n "s@.*tunneled with[^,]*, *\(https://[A-Za-z0-9._-]*\).*@\1@p" \
       "$TUNNEL_LOG" 2>/dev/null | tail -1)"
  [[ -z "$u" ]] \
    && u="$(grep -oE "https://[A-Za-z0-9-]+\.lhr\.life" "$TUNNEL_LOG" 2>/dev/null | tail -1)"
  echo "$u"
}

# Wait for a URL that is not the one we already had. $1 = the previous URL,
# $2 = how many seconds to wait, $3 = the ssh pid to give up on if it dies.
await_url() {
  local prev="$1" secs="$2" pid="${3:-}" url i
  for (( i = 0; i < secs; i++ )); do
    url="$(latest_url)"
    [[ -n "$url" && "$url" != "$prev" ]] && { echo "$url"; return 0; }
    [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && break
    sleep 1
  done
  echo ""
  return 1
}

# Sits on one connection until it is no longer serving the link. ssh exiting
# is the easy case; the one worth this code is an anonymous tunnel expiring
# while the ssh session stays up, where nothing local notices and the link
# simply stops working. Killing the ssh here is what makes tunnel_loop open the
# next one and print its address.
supervise() {
  local sshpid="$1" prev="$2" url="${3:-}" miss=0 i
  if [[ -z "$url" ]]; then
    url="$(await_url "$prev" 45 "$sshpid")"
    # a keyed tunnel comes back on the address it had, so nothing new is
    # announced and the one already in the log is the one to watch
    [[ -z "$url" ]] && url="$(latest_url)"
  fi
  while kill -0 "$sshpid" 2>/dev/null; do
    for (( i = 0; i < TUNNEL_CHECK_EVERY; i++ )); do
      kill -0 "$sshpid" 2>/dev/null || return 0
      sleep 1
    done
    [[ -z "$url" ]] && { url="$(latest_url)"; continue; }
    if url_answers "$url"; then
      miss=0
      continue
    fi
    miss=$(( miss + 1 ))
    echo "[$(date "+%F %T")] $url answered ${LAST_CODE:-nothing} ($miss of $TUNNEL_CHECK_FAILS)" >> "$TUNNEL_LOG"
    (( miss < TUNNEL_CHECK_FAILS )) && continue
    echo
    warn "the public link stopped answering (HTTP ${LAST_CODE:-no reply}) - opening a new one"
    kill "$sshpid" 2>/dev/null
    return 0
  done
}

# Keeps one ssh alive. A connection that dies without ever announcing a URL is
# a failure rather than a drop, so those back off instead of hammering
# localhost.run, and two of them in a row mean the key is not welcome - at
# which point it switches to the anonymous user rather than retrying forever.
tunnel_loop() {
  local sshpid prev url t0 el fails=0 pause had=0 keyopt=()
  # IdentitiesOnly, or ssh offers the box's other keys first and localhost.run
  # answers whichever one it is handed - which is how you end up on a different
  # subdomain than the one you shared
  [[ -f "$TUNNEL_KEY" ]] && keyopt=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes)
  while true; do
    prev="$(latest_url)"
    url=""
    t0=$(date +%s)
    ssh -n -T -o StrictHostKeyChecking=accept-new \
           -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
           -o ExitOnForwardFailure=yes \
           ${keyopt[@]+"${keyopt[@]}"} -R 80:localhost:"$PORT" "$TUNNEL_TARGET" \
        >> "$TUNNEL_LOG" 2>&1 &
    sshpid=$!
    echo "$sshpid" > "$TUNNEL_PIDFILE"

    # The first working URL is announced by start_tunnel, which is still
    # waiting for it. Only once a link has actually existed is a new one a
    # reconnect worth reporting - and with the anonymous user it is a
    # different address from the one before, so it has to be reported.
    if [[ "$had" -eq 1 ]]; then
      url="$(await_url "$prev" 40 "$sshpid")"
      if [[ -n "$url" ]]; then
        echo "$url" > "$PUBLIC_URL_FILE"
        echo -e "\n${YLW}  warn${NC} the tunnel dropped and reconnected"
        echo -e "${GRN}  ok${NC} public    $url"
        [[ "$TUNNEL_TARGET" == nokey@* ]] \
          && echo -e "${YLW}  warn${NC} that is a new address - the previous link is dead"
      fi
    fi

    # ssh does not always end when the tunnel does, so watch the link itself
    supervise "$sshpid" "$prev" "$url"
    wait "$sshpid" 2>/dev/null
    [[ -n "$(latest_url)" ]] && had=1
    el=$(( $(date +%s) - t0 ))
    if [[ "$(latest_url)" != "$prev" ]]; then
      fails=0                     # it did connect, however briefly
    else
      fails=$(( fails + 1 ))
    fi
    echo "" >> "$TUNNEL_LOG"
    echo "[$(date "+%F %T")] ssh exited after ${el}s (consecutive failures: $fails)" \
      >> "$TUNNEL_LOG"

    if [[ "$fails" -ge 2 && "$TUNNEL_TARGET" != nokey@* ]]; then
      TUNNEL_TARGET="nokey@localhost.run"
      echo "$TUNNEL_TARGET" > "$TUNNEL_TARGETFILE"
      echo -e "${YLW}  warn${NC} localhost.run would not take your SSH key, using an anonymous tunnel"
      # the reason matters: a passphrase-protected key with no agent looks
      # exactly like a rejected one from here, and is worth knowing about
      local why
      why="$(grep -iE "permission denied|passphrase|no such identity|Too many auth" \
             "$TUNNEL_LOG" 2>/dev/null | tail -1)"
      [[ -n "$why" ]] && echo -e "${YLW}  warn${NC} ssh said: ${why# }"
      fails=0
    fi

    pause=$(( fails <= 1 ? 3 : fails * 10 ))
    (( pause > 60 )) && pause=60
    sleep "$pause"
  done
}

start_tunnel() {
  command -v ssh >/dev/null || die "ssh not found, needed for --public"
  mkdir -p results/logs
  : > "$TUNNEL_LOG"
  TUNNEL_TARGET="$(pick_target)"
  echo "$TUNNEL_TARGET" > "$TUNNEL_TARGETFILE"

  say "Opening a public link"
  tunnel_loop &
  TUNNEL_PID=$!
  # long enough to cover a refused key, the switch, and the retry after it
  local url; url="$(await_url "" 60)"
  local target; target="$(cat "$TUNNEL_TARGETFILE" 2>/dev/null)"

  if [[ -n "$url" ]]; then
    echo "$url" > "$PUBLIC_URL_FILE"
    ok "public    $url"
    check_public "$url"
    if [[ "$target" == nokey@* ]]; then
      warn "anonymous tunnel, so a reconnect gets a different address"
      warn "(an SSH key on this machine would keep the address stable)"
    fi
    warn "no password in front of it - anyone with the link can start runs here"
    ok "current   $PUBLIC_URL_FILE always holds the live address"
  else
    warn "localhost.run did not hand back a URL, see $TUNNEL_LOG"
    warn "the server still starts, just without the public link"
  fi
}

# Does the public URL still reach this machine? With no curl there is nothing
# to ask with, so say yes rather than tearing down a tunnel that is probably
# fine. Both the startup check and the watchdog ask through here.
url_answers() {
  LAST_CODE=""
  command -v curl >/dev/null || return 0
  LAST_CODE="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 25 "$1" 2>/dev/null)"
  [[ "$LAST_CODE" == "200" ]]
}

# The same question at startup, said out loud. Printing a link and leaving you
# to discover in a browser that it does not answer is the one thing worth
# spending a few seconds to avoid.
check_public() {
  command -v curl >/dev/null || return 0
  if url_answers "$1"; then
    ok "checked   that link reaches this server"
    return 0
  fi
  warn "that link did NOT answer (HTTP ${LAST_CODE:-no reply})"
  warn "the tunnel log is $TUNNEL_LOG"
  return 1
}

stop_tunnel() {
  # the supervisor first, so it cannot start another ssh, then the ssh itself
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null
  [[ -s "$TUNNEL_PIDFILE" ]] && kill "$(cat "$TUNNEL_PIDFILE")" 2>/dev/null
  rm -f "$TUNNEL_PIDFILE" "$TUNNEL_TARGETFILE" "$PUBLIC_URL_FILE"
  TUNNEL_PID=""
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

SERVER_PID=""
cleanup() {
  stop_tunnel
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  return 0
}

# Wait until the server actually answers, so the tunnel is never published
# pointing at a port with nothing behind it.
wait_for_server() {
  command -v curl >/dev/null || { sleep 2; return 0; }
  local i
  for (( i = 0; i < 30; i++ )); do
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/" && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

# No exec: the trap has to outlive the server so the tunnel is closed with it.
trap cleanup EXIT INT TERM

python3 -u scripts/web_ui.py --port "$PORT" --host "$HOST" &
SERVER_PID=$!

# The tunnel goes up second. The other way round publishes a URL that answers
# with an error for however long the server takes to bind, and leaves nothing
# for check_public to test against.
if [[ "$PUBLIC" -eq 1 ]]; then
  if wait_for_server; then
    start_tunnel
  else
    warn "the server never came up, so no tunnel was opened"
  fi
fi

wait "$SERVER_PID"
