#!/usr/bin/env bash
#
# web_ui.sh - start the browser front end for run_all.sh.
#
#   ./web_ui.sh                 # foreground, http://localhost:8000
#   ./web_ui.sh --port 8080
#   ./web_ui.sh --tmux          # detached, survives an SSH disconnect
#   ./web_ui.sh --local         # this machine only
#   ./web_ui.sh --public        # plus a public https link
#   ./web_ui.sh --public --domain your-name.ngrok-free.app   # a permanent one
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
# A PERMANENT ADDRESS needs ngrok. localhost.run's free addresses are not
# permanent: an SSH key keeps the same one for a while, but localhost.run
# retires free addresses from time to time (the old one answers 503) and the
# next connection gets a new one. ngrok gives every free account one static
# domain that never changes. Once:
#
#   1. sign up at https://ngrok.com (free) and download the ngrok program
#      for Linux; put it on PATH, or in ~/bin, ~/.local/bin or ./bin
#   2. ngrok config add-authtoken <the token on your ngrok dashboard>
#   3. copy your free static domain from the dashboard (Domains), and put
#      NGROK_DOMAIN=your-name.ngrok-free.app in this project's .env
#
# after which ./web_ui.sh --public always comes up on that address (or pass
# --domain on the command line). Without a domain it falls back to
# localhost.run as before.
#
# Either way the tunnel is supervised: when it drops it is reopened, and the
# public URL itself is asked once a minute, because a tunnel can stop serving
# while the process behind it looks healthy. With ngrok the address stays the
# same through all of that. With localhost.run, ~/.ssh/id_lhr (made with
# ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_lhr) is used in preference to the
# box's own key and keeps the address for longer, but not forever. The whole
# history is in results/logs/tunnel.log.
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
DOMAIN="${NGROK_DOMAIN:-}"      # a permanent ngrok address, if there is one
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)    PORT="${2:-}";    shift 2 ;;
    --host)       HOST="${2:-}";    shift 2 ;;
    --local)      HOST=127.0.0.1;   shift   ;;
    --public)     PUBLIC=1;         shift   ;;
    --domain)     DOMAIN="${2:-}";  PUBLIC=1; shift 2 ;;
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

# The permanent address, if none was given: NGROK_DOMAIN in .env, where the
# Tavily key already lives. Only that one line is read.
if [[ -z "$DOMAIN" && -f .env ]]; then
  DOMAIN="$(sed -n 's/^[[:space:]]*NGROK_DOMAIN[[:space:]]*=[[:space:]]*//p' .env \
            | tail -1 | tr -d "\"'" | tr -d '[:space:]')"
fi
# people paste the whole URL; ngrok wants the host
DOMAIN="${DOMAIN#https://}"; DOMAIN="${DOMAIN#http://}"; DOMAIN="${DOMAIN%%/*}"

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
TUNNEL_CHECK_EVERY="${SCAM_TUNNEL_CHECK_EVERY:-60}"
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

# How many tunnels have been announced in the log so far. A session connected
# if this went up while it ran - which is true whether localhost.run handed
# back a new address or, for a keyed tunnel, the same one as last time.
announced() {
  # grep -c prints "0" AND exits 1 when nothing matches, so "|| echo 0" would
  # print a second 0 and hand arithmetic "0<newline>0". Take what it printed,
  # and fall back to 0 only when it printed nothing (no log file yet).
  local n
  n="$(grep -c "tunneled with" "$TUNNEL_LOG" 2>/dev/null)"
  echo "${n:-0}"
}

# Wait for the NEXT announcement after $1 lines, and give its URL.
# $2 = seconds to wait, $3 = the ssh pid to give up on if it dies.
await_announce() {
  local n0="$1" secs="$2" pid="${3:-}" i
  for (( i = 0; i < secs; i++ )); do
    (( $(announced) > n0 )) && { latest_url; return 0; }
    [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && break
    sleep 1
  done
  echo ""
  return 1
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
    if [[ "$PROVIDER" == ngrok ]]; then
      warn "the public link stopped answering (HTTP ${LAST_CODE:-no reply}) - reconnecting, same address"
    else
      warn "the public link stopped answering (HTTP ${LAST_CODE:-no reply}) - opening a new one"
    fi
    kill "$sshpid" 2>/dev/null
    return 0
  done
}

# Keeps one ssh alive. A connection that dies without ever announcing a URL is
# a failure rather than a drop, so those back off instead of hammering
# localhost.run, and two of them in a row mean the key is not welcome - at
# which point it switches to the anonymous user rather than retrying forever.
tunnel_loop() {
  local sshpid prev n0 url t0 el fails=0 pause had=0 keyopt=()
  # IdentitiesOnly, or ssh offers the box's other keys first and localhost.run
  # answers whichever one it is handed - which is how you end up on a different
  # subdomain than the one you shared
  [[ -f "$TUNNEL_KEY" ]] && keyopt=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes)
  while true; do
    prev="$(latest_url)"
    n0="$(announced)"
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
    #
    # Waiting for "a URL different from before" used to be how a reconnect was
    # recognised. A keyed tunnel comes back on the SAME address, so that wait
    # never ended, and worse, every ordinary drop was then counted as a
    # failure below - two of those and the key was abandoned for an anonymous
    # tunnel with a random address. So wait for a new announcement instead.
    url="$(await_announce "$n0" 45 "$sshpid")"
    if [[ "$had" -eq 1 && -n "$url" ]]; then
      echo "$url" > "$PUBLIC_URL_FILE"
      echo -e "\n${YLW}  warn${NC} the tunnel dropped and reconnected"
      if [[ "$url" == "$prev" ]]; then
        echo -e "${GRN}  ok${NC} public    $url   (same address as before)"
      else
        echo -e "${GRN}  ok${NC} public    $url"
        echo -e "${YLW}  warn${NC} that is a new address - the previous link is dead"
      fi
    fi

    # ssh does not always end when the tunnel does, so watch the link itself
    supervise "$sshpid" "$prev" "$url"
    wait "$sshpid" 2>/dev/null
    [[ -n "$(latest_url)" ]] && had=1
    el=$(( $(date +%s) - t0 ))
    if (( $(announced) > n0 )); then
      fails=0                     # it did connect, however briefly
    else
      fails=$(( fails + 1 ))      # it never got as far as a tunnel
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

# ------------------------------------------------------------------ ngrok
# The permanent address. ngrok's free plan includes one static domain per
# account, so every connection - the first, and every reconnect after a drop -
# comes up on the same https://<domain>, and a link handed to someone keeps
# working for as long as this script is running.
PROVIDER=localhost.run
NGROK=""

find_ngrok() {
  local c
  for c in "$(command -v ngrok 2>/dev/null)" "$PROJECT_DIR/bin/ngrok" \
           "$HOME/bin/ngrok" "$HOME/.local/bin/ngrok"; do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

# How many times ngrok has said it is up, so far this run.
ngrok_started() {
  local n
  n="$(grep -c 'started tunnel' "$TUNNEL_LOG" 2>/dev/null)"
  echo "${n:-0}"
}

# Wait for this ngrok to be serving: its own "started tunnel" line, or the
# address answering. $1 = the count before it started, $2 = seconds, $3 = pid.
await_ngrok() {
  local n0="$1" secs="$2" pid="$3" i
  for (( i = 0; i < secs; i++ )); do
    (( $(ngrok_started) > n0 )) && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    (( i % 5 == 4 )) && url_answers "https://$DOMAIN" && return 0
    sleep 1
  done
  return 1
}

# ngrok's own last complaint, which is what to show when it will not start -
# no authtoken, the domain belonging to another account, or this domain
# already online from another copy of ngrok.
ngrok_error() {
  grep -E 'lvl=(eror|crit)|ERR_NGROK|ERROR:' "$TUNNEL_LOG" 2>/dev/null | tail -1 \
    | sed -e 's/.*err="\{0,1\}//' -e 's/"$//' | cut -c1-240
}

ngrok_loop() {
  local pid n0 t0 el fails=0 pause had=0 said="" url="https://$DOMAIN" where=()
  # ngrok 3.16+ takes --url, older 3.x --domain; ask which this one knows
  if "$NGROK" http --help 2>&1 | grep -q -- "--url"; then
    where=(--url "$url")
  else
    where=(--domain "$DOMAIN")
  fi
  while true; do
    n0="$(ngrok_started)"
    t0=$(date +%s)
    "$NGROK" http "$PORT" "${where[@]}" --log stdout --log-format logfmt \
      < /dev/null >> "$TUNNEL_LOG" 2>&1 &
    pid=$!
    echo "$pid" > "$TUNNEL_PIDFILE"
    if await_ngrok "$n0" 45 "$pid"; then
      fails=0
      echo "$url" > "$PUBLIC_URL_FILE"
      if [[ "$had" -eq 1 ]]; then
        echo -e "\n${YLW}  warn${NC} the tunnel dropped and reconnected"
        echo -e "${GRN}  ok${NC} public    $url   (same address as before)"
      fi
      had=1
      supervise "$pid" "" "$url"
    else
      fails=$(( fails + 1 ))
    fi
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    el=$(( $(date +%s) - t0 ))
    echo "[$(date "+%F %T")] ngrok exited after ${el}s (consecutive failures: $fails)" \
      >> "$TUNNEL_LOG"
    # say why once, not on every retry
    if (( fails > 0 )); then
      local why; why="$(ngrok_error)"
      if [[ -n "$why" && "$why" != "$said" ]]; then
        echo -e "${YLW}  warn${NC} ngrok: $why"
        said="$why"
      fi
    fi
    pause=$(( fails <= 1 ? 3 : fails * 10 ))
    (( pause > 60 )) && pause=60
    sleep "$pause"
  done
}

start_ngrok() {
  say "Opening the public link (ngrok, permanent address)"
  ngrok_loop &
  TUNNEL_PID=$!
  local url="https://$DOMAIN" i
  for (( i = 0; i < 60; i++ )); do
    [[ -s "$PUBLIC_URL_FILE" ]] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ -s "$PUBLIC_URL_FILE" ]]; then
    ok "public    $url"
    check_public "$url"
    ok "permanent this address stays the same across reconnects and restarts"
    warn "no password in front of it - anyone with the link can start runs here"
    warn "ngrok's free plan shows a one-time warning page to each new browser;"
    warn "press Visit Site once and it does not come back"
  else
    # the retry loop has already printed ngrok's own reason, if it gave one
    warn "ngrok did not come up on $DOMAIN"
    warn "it keeps retrying in the background; the log is $TUNNEL_LOG"
  fi
}

start_tunnel() {
  mkdir -p results/logs
  : > "$TUNNEL_LOG"
  rm -f "$PUBLIC_URL_FILE"
  if [[ -n "$DOMAIN" ]]; then
    if NGROK="$(find_ngrok)"; then
      PROVIDER=ngrok
      start_ngrok
      return
    fi
    warn "NGROK_DOMAIN is $DOMAIN but the ngrok program was not found"
    warn "(looked on PATH, ./bin, ~/bin and ~/.local/bin) - using localhost.run,"
    warn "whose address is not permanent"
  fi
  command -v ssh >/dev/null || die "ssh not found, needed for --public"
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
    else
      warn "localhost.run keeps this address for a while, but not for good"
    fi
    warn "for a permanent address, set up ngrok: ./web_ui.sh --help"
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
  LAST_CODE="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 25 \
               -H "ngrok-skip-browser-warning: 1" "$1" 2>/dev/null)"
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
  [[ -n "$DOMAIN" ]] && CMD="$CMD --domain $(printf %q "$DOMAIN")"
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

# Is something already listening on $PORT? Asked before anything starts,
# because the failure it prevents is quiet: the new server dies on "Address
# already in use", but wait_for_server then finds the OLD one answering, opens
# a second tunnel in front of it - which localhost.run gives a different
# address from the tunnel the old instance still holds - checks that link
# "reaches this server" (it reaches the old one), and exits.
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
    [[ -s "$PUBLIC_URL_FILE" ]] \
      && echo "  Its public link is probably still up: $(cat "$PUBLIC_URL_FILE")"
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
    # a reply only counts while our own server is alive - otherwise it is
    # something else on the port answering, and the tunnel would front that
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/" \
      && sleep 1 && kill -0 "$SERVER_PID" 2>/dev/null && return 0
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

# One line, read whole before it runs: the Update button can git pull a new
# copy of this file while bash is still sitting here, and bash reads a script
# as it goes - so once the server is gone, exit without reading any further.
wait "$SERVER_PID"; exit $?
