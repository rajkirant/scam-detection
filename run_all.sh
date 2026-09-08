#!/usr/bin/env bash
#
# run_all.sh - interactive launcher for the scam-detection benchmark.
#
#   ./run_all.sh
#
# Asks up to five questions, then runs just what you picked:
#
#   1. which dataset    (numbered menu of every CSV in datasets/)
#   2. which baselines  (one, several as a comma list, or all of them)
#   3. how many calls   (0 = the whole dataset, a number = the first N,
#                         id:<value> = the one row whose id column matches,
#                         idx:<n> = the same call as index n in a --limit 40
#                         style shuffled run, matching check_one.py --idx)
#   4. which model      (only asked when the baseline actually calls an LLM)
#   5. tmux or not      (always asked, whatever the other four answers were)
#
# Everything can also be given up front, which skips the questions:
#
#   ./run_all.sh --dataset datasets/zhi_balanced_333x333.csv --baseline singh \
#                --limit 40 --model qwen2.5:14b
#   ./run_all.sh --dataset datasets/paired_scam_legit_198.csv --baseline mcq \
#                --limit idx:19 --model qwen2.5:14b  # the SAME call as --idx 19
#                                                     # in check_one.py / a --limit 40 run
#   ./run_all.sh -d 2 -b 4 -l 0 -m 1          # menu numbers work too
#   ./run_all.sh -b llm_only,mcq,bert -l 0    # just those three, one run
#   ./run_all.sh -b 3,7,8 -l 0                # the same three, by number
#
# A long run can outlive the SSH session, so question 5 offers to hand the
# actual work to a detached tmux session: closing the terminal, or losing
# the link, then does not kill it. It is asked on every run, including a
# fully flagged one, and defaults to no. --tmux answers it up front:
#
#   ./run_all.sh --tmux                       # menus, then detach
#   ./run_all.sh -d 3 -b 1 -l 0 -m 1 --tmux   # no questions at all
#   ./run_all.sh --tmux --session nightly     # name the session yourself
#
# Everything the run prints is teed to a log, so the output survives even
# if the tmux session is later killed. With no tmux installed it falls
# back to setsid+nohup, which also survives a disconnect.
#
# When the baseline is "all", BERT runs last because qwen2.5:14b holds ~9.5 GB
# of 11.4 GB VRAM and BERT fine-tuning needs 3-4 GB. They cannot both be
# resident.
#
set -uo pipefail

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'
say()  { echo -e "\n${CYN}==>${NC} $*"; }
ok()   { echo -e "${GRN}  ok${NC} $*"; }
warn() { echo -e "${YLW}  warn${NC} $*"; }
die()  { echo -e "${RED}  fail${NC} $*"; exit 1; }

# run from the project root, wherever this script happens to live
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || die "cannot cd to $PROJECT_DIR"

ONTOLOGY="knowledge/scam_ontology.json"
MCQ_ONTOLOGY="knowledge/mcq_ontology.json"
MODELS=("qwen2.5:14b" "llama3.1:8b")

BL_KEYS=(all trivial llm_only singh webrag qwen_kb hybrid ontology mcq bert)
BL_LABELS=(
  "all                       every system below, in one run"
  "length + bag-of-words     trivial references, no LLM"
  "LLM-only                  the model decides alone, no retrieval"
  "Singh                     policy-compliance baseline"
  "Web-RAG                   KB-only retrieval"
  "Qwen-KB                   learns a KB from a held-out split, k-fold"
  "Hybrid                    Web-RAG + Qwen-KB over one shared KB"
  "Ontology RAG              scam_ontology.json"
  "MCQ ontology              mcq_ontology.json, 2 calls per transcript"
  "BERT                      fine-tuned classifier, no LLM"
)

prompt() {            # prompt <text> <varname>   - read only echoes its own
  printf "%s" "$1"    # prompt on a terminal, so print it here instead
  read -r "$2" || die "no input (answers can be piped in, or use the flags)"
}

# ---------------------------------------------------------------- arguments
ARG_DATASET=""; ARG_BASELINE=""; ARG_LIMIT=""; ARG_MODEL=""
DETACH=0; SESSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)  ARG_DATASET="${2:-}";  shift 2 ;;
    -b|--baseline) ARG_BASELINE="${2:-}"; shift 2 ;;
    -l|--limit)    ARG_LIMIT="${2:-}";    shift 2 ;;
    -m|--model)    ARG_MODEL="${2:-}";    shift 2 ;;
    -t|--tmux)     DETACH=1;              shift   ;;
    -s|--session)  SESSION="${2:-}"; DETACH=1; shift 2 ;;
    -h|--help)     awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "$0"; exit 0 ;;
    *)             die "unknown argument: $1  (try --help)" ;;
  esac
done

# ------------------------------------------------------------- dataset menu
shopt -s nullglob
DATASETS=(datasets/*.csv)
shopt -u nullglob
[[ ${#DATASETS[@]} -gt 0 ]] || die "no CSV files in datasets/"
NWIDTH=${#DATASETS[@]}; NWIDTH=${#NWIDTH}   # digits in the highest menu number

# wc -l counts newline characters, not CSV rows. A transcript field that
# contains an embedded newline (a multi-turn call quoted across several
# physical lines) inflates the count. Parse the file as CSV instead, and
# fall back to the old line-count method only if that parse fails, so a
# malformed file still shows a number rather than crashing the menu.
rows_of() {
  python3 -c "
import csv, sys
csv.field_size_limit(sys.maxsize)
try:
    with open('$1', newline='', encoding='utf-8') as f:
        print(sum(1 for _ in csv.reader(f)) - 1)
except Exception:
    sys.exit(1)
" 2>/dev/null || echo $(( $(wc -l < "$1") - 1 ))
}

ask_dataset() {
  echo
  echo -e "${BLD}  Which dataset?${NC}"
  local i
  for i in "${!DATASETS[@]}"; do
    printf "    %${NWIDTH}d) %-34s %6s rows\n" $((i + 1)) "$(basename "${DATASETS[$i]}")" "$(rows_of "${DATASETS[$i]}")"
  done
  local pick=""
  while true; do
    prompt "  choose 1-${#DATASETS[@]}: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#DATASETS[@]} )); then
      DATASET="${DATASETS[$((pick - 1))]}"
      return
    fi
    warn "enter a number between 1 and ${#DATASETS[@]}"
  done
}

# ------------------------------------------------------------ baseline menu
# Turns an answer into SEL, the list of baselines to run. Accepts menu
# numbers, key names, and any comma-separated mix of the two, so "3", "mcq"
# and "3,7,8" are all valid. "all" expands to every real baseline.
# Whatever order they arrive in, SEL comes back in menu order, which is why
# BERT still runs last: it is last in BL_KEYS, and it cannot share the GPU
# with a loaded qwen.
SEL=()
parse_baselines() {
  local raw="$1" tok key k picked=() ordered=()
  local IFS=,
  for tok in $raw; do
    tok="${tok// /}"
    [[ -z "$tok" ]] && continue
    if [[ "$tok" =~ ^[0-9]+$ ]]; then
      (( tok >= 1 && tok <= ${#BL_KEYS[@]} )) || return 1
      key="${BL_KEYS[$((tok - 1))]}"
    else
      printf "%s\n" "${BL_KEYS[@]}" | grep -qx "$tok" || return 1
      key="$tok"
    fi
    if [[ "$key" == "all" ]]; then
      picked=("${BL_KEYS[@]:1}")
    else
      picked+=("$key")
    fi
  done
  for k in "${BL_KEYS[@]:1}"; do
    printf "%s\n" "${picked[@]:-}" | grep -qx "$k" && ordered+=("$k")
  done
  [[ ${#ordered[@]} -gt 0 ]] || return 1
  SEL=("${ordered[@]}")
  return 0
}

ask_baseline() {
  echo
  echo -e "${BLD}  Which baselines?${NC}  (one, or several separated by commas)"
  local i
  for i in "${!BL_KEYS[@]}"; do
    printf "    %d) %s\n" $((i + 1)) "${BL_LABELS[$i]}"
  done
  local pick=""
  while true; do
    prompt "  choose 1-${#BL_KEYS[@]}, e.g. 3 or 3,7,8: " pick
    parse_baselines "$pick" && return
    warn "enter numbers or names between 1 and ${#BL_KEYS[@]}, comma-separated"
  done
}

# --------------------------------------------------------------- limit menu
# Accepts three shapes of answer:
#   0            -> the whole dataset
#   a number N   -> the first N calls (same as before)
#   id:<value>   -> just the one row whose "id" column equals <value>
# LIMIT is set to 0 in the id: case; ONE_ID carries the id to look up. The
# actual row extraction happens later, once $DATASET is known to exist.
ONE_ID=""
ONE_IDX=""
ask_limit() {
  echo
  echo -e "${BLD}  How many calls?${NC}  (0 = whole dataset, id:<value> = one row by its id column,\n  idx:<n> = the same call as index n in a --limit 40 style run)"
  local pick=""
  while true; do
    prompt "  limit: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]]; then
      LIMIT="$pick"; ONE_ID=""; ONE_IDX=""
      return
    fi
    if [[ "$pick" =~ ^idx:([0-9]+)$ ]]; then
      LIMIT=0; ONE_ID=""; ONE_IDX="${BASH_REMATCH[1]}"
      return
    fi
    if [[ "$pick" =~ ^id:(.+)$ ]]; then
      LIMIT=0; ONE_ID="${BASH_REMATCH[1]}"; ONE_IDX=""
      return
    fi
    warn "enter a whole number, 0 for all, id:<value> e.g. id:19, or idx:<n> e.g. idx:19"
  done
}

# Pull the single row whose id column equals $1 out of $2 (the dataset CSV)
# into a temporary CSV with the same header, and print the temp file's path.
# Uses Python's csv module rather than grep, since a transcript field can
# contain commas and embedded newlines that break naive text matching.
extract_single_row() {
  local id="$1" csv_in="$2"
  python3 -c "
import csv, sys, tempfile
csv.field_size_limit(sys.maxsize)
target = '''$id'''
with open('$csv_in', newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    rows = [row for row in r if row.get('id') == target]
    fieldnames = r.fieldnames
if not rows:
    sys.stderr.write('no row with id %r in $csv_in\n' % target)
    sys.exit(1)
if len(rows) > 1:
    sys.stderr.write('warning: %d rows share id %r, using the first\n' % (len(rows), target))
fd, path = tempfile.mkstemp(prefix='run_all_row_', suffix='.csv')
with open(path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(rows[0])
print(path)
"
}

# Pull the row at position $1 in the SAME shuffled order that
# evaluate_mcq_ontology.py's load() produces for a --limit 40 style run
# (scam/nonscam split, seed 42, sampled to --limit, then shuffled). This is
# what check_one.py --idx also reproduces, so idx:N here, --idx N there, and
# a call's position inside a real --limit N run all agree on the same call.
extract_by_idx() {
  local idx="$1" csv_in="$2" idx_limit="$3"
  python3 -c "
import csv, sys, random, tempfile
csv.field_size_limit(sys.maxsize)
SCAM_WORDS = {'scam', 'fraud', 'fraudulent', '1', 'true', 'yes'}
with open('$csv_in', newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    rows = list(r)
    fieldnames = r.fieldnames
data = []
for row in rows:
    lab = (row.get('label') or '').strip().lower()
    txt = (row.get('text') or '').strip()
    if txt:
        data.append(row)
limit = $idx_limit
if limit:
    scam = [row for row in data if (row.get('label') or '').strip().lower() in SCAM_WORDS]
    norm = [row for row in data if (row.get('label') or '').strip().lower() not in SCAM_WORDS]
    k = limit // 2
    random.seed(42)
    data = random.sample(scam, min(k, len(scam))) + random.sample(norm, min(limit - k, len(norm)))
random.seed(42)
random.shuffle(data)
idx = $idx
if not (0 <= idx < len(data)):
    sys.stderr.write('idx %d out of range for %d rows loaded with limit=%d\n' % (idx, len(data), limit))
    sys.exit(1)
fd, path = tempfile.mkstemp(prefix='run_all_row_', suffix='.csv')
with open(path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(data[idx])
print(path)
"
}

# ------------------------------------------------------------- detach menu
# Deliberately not prompt(), which dies on EOF. A fully flagged run gets
# asked this too, and such a run may have no terminal on stdin at all, so
# EOF - like an empty answer - has to mean "no, run in the foreground".
ask_detach() {
  echo
  echo -e "${BLD}  Run detached in tmux?${NC}  (survives an SSH disconnect)"
  local pick=""
  printf "  [y/N]: "
  read -r pick || pick=""
  case "$pick" in
    [yY]|[yY][eE][sS]) DETACH=1 ;;
    *)                 DETACH=0 ;;
  esac
}

# --------------------------------------------------------------- model menu
OLLAMA_URL="http://localhost:11434"
PULLED=""                       # raw /api/tags JSON, empty if Ollama is down
OLLAMA_UP=0
PULLED="$(curl -s -m 5 "$OLLAMA_URL/api/tags" 2>/dev/null)" || PULLED=""
[[ -n "$PULLED" ]] && OLLAMA_UP=1
is_pulled() { [[ "$PULLED" == *"\"name\":\"$1\""* ]]; }

ask_model() {
  echo
  echo -e "${BLD}  Which model?${NC}"
  local i note
  for i in "${!MODELS[@]}"; do
    note=""
    if [[ "$OLLAMA_UP" -eq 1 ]]; then
      is_pulled "${MODELS[$i]}" && note="  (installed)" || note="  (not pulled)"
    fi
    printf "    %d) %s%s\n" $((i + 1)) "${MODELS[$i]}" "$note"
  done
  local pick=""
  while true; do
    prompt "  choose 1-${#MODELS[@]}: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#MODELS[@]} )); then
      MODEL="${MODELS[$((pick - 1))]}"
      return
    fi
    warn "enter a number between 1 and ${#MODELS[@]}"
  done
}

# -------------------------------------------------------------- the answers
# a flag value may be a menu number or the thing itself
if [[ -n "$ARG_DATASET" ]]; then
  if [[ "$ARG_DATASET" =~ ^[0-9]+$ ]] && (( ARG_DATASET >= 1 && ARG_DATASET <= ${#DATASETS[@]} )); then
    DATASET="${DATASETS[$((ARG_DATASET - 1))]}"
  else
    DATASET="$ARG_DATASET"
  fi
else
  ask_dataset
fi

if [[ -n "$ARG_BASELINE" ]]; then
  parse_baselines "$ARG_BASELINE" \
    || die "unknown baseline in: $ARG_BASELINE  (one of: ${BL_KEYS[*]})"
else
  ask_baseline
fi

has_bl() { printf "%s\n" "${SEL[@]}" | grep -qx "$1"; }

# what the chosen baselines actually run
RUN_COMBINED=0; RUN_ONTOLOGY=0; RUN_MCQ=0; RUN_BERT=0; COMBINED_EXTRA=""
has_bl ontology && RUN_ONTOLOGY=1
has_bl mcq      && RUN_MCQ=1
has_bl bert     && RUN_BERT=1

# combined_evaluate.py owns four of the seven systems, so it runs whenever
# any of them was picked, and is told to skip the ones that were not.
COMBINED_SKIP=()
has_bl trivial  || COMBINED_SKIP+=(length bow)
has_bl llm_only || COMBINED_SKIP+=(llm_only)
has_bl singh    || COMBINED_SKIP+=(singh)
has_bl webrag   || COMBINED_SKIP+=(webrag)
has_bl qwen_kb   || COMBINED_SKIP+=(qwen_kb)
has_bl hybrid   || COMBINED_SKIP+=(hybrid)
if [[ ${#COMBINED_SKIP[@]} -lt 7 ]]; then     # fewer than all seven skipped
  RUN_COMBINED=1
  [[ ${#COMBINED_SKIP[@]} -gt 0 ]] \
    && COMBINED_EXTRA="--skip $(IFS=,; echo "${COMBINED_SKIP[*]}")"
fi

# only the systems that actually call an LLM make the model question worth
# asking - a trivial+bert selection needs no model at all
NEEDS_MODEL=0
for _k in llm_only singh webrag qwen_kb hybrid ontology mcq; do
  has_bl "$_k" && NEEDS_MODEL=1
done

BASELINE="$(IFS=,; echo "${SEL[*]}")"
# a short name for logs and tmux sessions: no commas, and "all" when it is
BASELINE_TAG="$BASELINE"
[[ ${#SEL[@]} -eq $(( ${#BL_KEYS[@]} - 1 )) ]] && BASELINE_TAG="all"
BASELINE_TAG="${BASELINE_TAG//,/+}"

if [[ -n "$ARG_LIMIT" ]]; then
  if [[ "$ARG_LIMIT" =~ ^[0-9]+$ ]]; then
    LIMIT="$ARG_LIMIT"; ONE_ID=""; ONE_IDX=""
  elif [[ "$ARG_LIMIT" =~ ^id:(.+)$ ]]; then
    LIMIT=0; ONE_ID="${BASH_REMATCH[1]}"; ONE_IDX=""
  elif [[ "$ARG_LIMIT" =~ ^idx:([0-9]+)$ ]]; then
    LIMIT=0; ONE_ID=""; ONE_IDX="${BASH_REMATCH[1]}"
  else
    die "--limit needs a whole number, id:<value>, or idx:<n>, got '$ARG_LIMIT'"
  fi
else
  ask_limit
fi

MODEL=""
if [[ "$NEEDS_MODEL" -eq 1 ]]; then
  if [[ -n "$ARG_MODEL" ]]; then
    if [[ "$ARG_MODEL" =~ ^[0-9]+$ ]] && (( ARG_MODEL >= 1 && ARG_MODEL <= ${#MODELS[@]} )); then
      MODEL="${MODELS[$((ARG_MODEL - 1))]}"
    else
      MODEL="$ARG_MODEL"
    fi
  else
    ask_model
  fi
fi

# Asked on every run, no matter how the other four answers arrived, since
# whether a run should outlive the SSH session is independent of what it
# is running. --tmux/--session on the command line have already answered
# it, and the copy inside tmux must not be asked at all.
if [[ "$DETACH" -eq 0 && -z "${RUN_ALL_DETACHED:-}" ]]; then
  ask_detach
fi

# --------------------------------------------------------------- preflight
# only the LLM baselines need Ollama, so only they are blocked by it
if [[ "$NEEDS_MODEL" -eq 1 ]]; then
  if [[ "$OLLAMA_UP" -ne 1 ]]; then
    echo
    die "Ollama is not answering at $OLLAMA_URL
       start it with:  ollama serve
       (or pick a baseline that needs no LLM: trivial, or bert)"
  fi
  is_pulled "$MODEL" || warn "$MODEL is not pulled here.  Get it with:  ollama pull $MODEL"
fi

# ------------------------------------------------------------------- setup
[[ -f "$DATASET" ]] || die "missing $DATASET"
[[ "$RUN_ONTOLOGY" -eq 1 && ! -f "$ONTOLOGY" ]] && die "missing $ONTOLOGY"
[[ "$RUN_MCQ" -eq 1 && ! -f "$MCQ_ONTOLOGY" ]] && die "missing $MCQ_ONTOLOGY"

# ------------------------------------------------------- detached re-exec
# Everything above is cheap and interactive: the menus, the Ollama probe,
# the file checks. Everything below is the long part. So the split happens
# here - the answers are turned back into flags and handed to a copy of
# this script running under tmux, which asks nothing and outlives the SSH
# session. RUN_ALL_DETACHED stops that copy from detaching again.
SELF="$PROJECT_DIR/$(basename "${BASH_SOURCE[0]}")"

relaunch_cmd() {           # the exact command line the detached copy runs
  local lim
  if   [[ -n "$ONE_ID"  ]]; then lim="id:$ONE_ID"
  elif [[ -n "$ONE_IDX" ]]; then lim="idx:$ONE_IDX"
  else                           lim="$LIMIT"
  fi
  printf 'bash %q --dataset %q --baseline %q --limit %q' \
    "$SELF" "$DATASET" "$BASELINE" "$lim"
  [[ -n "$MODEL" ]] && printf ' --model %q' "$MODEL"
  printf '\n'
}

if [[ "$DETACH" -eq 1 && -z "${RUN_ALL_DETACHED:-}" ]]; then
  DSTAMP="$(date +%Y%m%d_%H%M%S)"
  [[ -n "$SESSION" ]] || SESSION="scam_${BASELINE_TAG}_${DSTAMP}"
  mkdir -p results/logs
  DLOG="$PROJECT_DIR/results/logs/detached_${SESSION}.log"

  # A launcher script rather than a nested -c string: no quoting to get
  # wrong, and it is left on disk as a record of what was actually run.
  LAUNCH="$PROJECT_DIR/results/logs/launch_${SESSION}.sh"
  {
    echo "#!/usr/bin/env bash"
    printf 'cd %q || exit 1\n' "$PROJECT_DIR"
    echo "export RUN_ALL_DETACHED=1"
    printf '%s 2>&1 | tee -a %q\n' "$(relaunch_cmd)" "$DLOG"
  } > "$LAUNCH"
  chmod +x "$LAUNCH"

  say "Detaching"
  ok "dataset   $DATASET"
  ok "baseline  ${SEL[*]}"
  ok "model     ${MODEL:-none}"
  if command -v tmux >/dev/null; then
    tmux new-session -d -s "$SESSION" "bash $(printf %q "$LAUNCH")" \
      || die "tmux could not start session $SESSION"
    # keep the finished pane around so the results table can be read back
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null 2>&1 || true
    ok "session   $SESSION"
    ok "log       $DLOG"
    echo
    echo "  watch it:   tmux attach -t $SESSION      (detach again: ctrl-b d)"
    echo "  or tail:    tail -f $DLOG"
    echo "  stop it:    tmux kill-session -t $SESSION"
  else
    warn "falling back to setsid + nohup, which also survives a disconnect"
    setsid nohup bash "$LAUNCH" >/dev/null 2>&1 &
    ok "pid       $!"
    ok "log       $DLOG"
    echo
    echo "  watch it:   tail -f $DLOG"
    echo "  stop it:    kill $!"
  fi
  echo
  exit 0
fi

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate || die "venv failed"
elif [[ -f venv/Scripts/activate ]]; then
  # shellcheck disable=SC1091
  source venv/Scripts/activate || die "venv failed"
else
  warn "no venv/ found, using whatever python is on PATH"
fi

# If a single transcript was requested, carve it out into a temp 1-row CSV
# now and point every downstream step at that file instead of the original
# dataset. This is the only place the id: mode touches the rest of the
# script - everything after this block behaves exactly as it always did,
# just against a dataset that happens to have one row in it.
TMP_ROW_CSV=""
# The full dataset, kept before DATASET is swapped for the 1-row temp file
# below. Qwen-KB learns its patterns from a training split, and one row
# cannot be split - so in single-transcript mode it is pointed at the
# dataset the row came from instead, with that row removed from it.
FULL_DATASET="$DATASET"
IDX_LIMIT_USED=40   # must match the --limit you used when you read the idx off a prior run
if [[ -n "$ONE_ID" ]]; then
  say "Extracting transcript id=$ONE_ID"
  TMP_ROW_CSV="$(extract_single_row "$ONE_ID" "$DATASET")" \
    || die "could not find id '$ONE_ID' in $DATASET"
  ok "wrote $TMP_ROW_CSV"
  DATASET="$TMP_ROW_CSV"
  trap '[[ -n "$TMP_ROW_CSV" ]] && rm -f "$TMP_ROW_CSV"' EXIT
elif [[ -n "$ONE_IDX" ]]; then
  say "Extracting transcript idx=$ONE_IDX (same order as --limit $IDX_LIMIT_USED, seed 42)"
  TMP_ROW_CSV="$(extract_by_idx "$ONE_IDX" "$DATASET" "$IDX_LIMIT_USED")" \
    || die "idx $ONE_IDX not available - check IDX_LIMIT_USED near the top of the script matches the --limit you used elsewhere"
  ok "wrote $TMP_ROW_CSV"
  DATASET="$TMP_ROW_CSV"
  trap '[[ -n "$TMP_ROW_CSV" ]] && rm -f "$TMP_ROW_CSV"' EXIT
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/logs/run_${STAMP}"
mkdir -p "$LOGDIR"

if [[ -n "$ONE_ID" || -n "$ONE_IDX" ]]; then
  BASE="$(basename "${ARG_DATASET:-$DATASET}" .csv)"
  if [[ -n "$ONE_ID" ]]; then
    SAFE_ID="$(echo "$ONE_ID" | tr -c '[:alnum:]_-' '_')"
    TAG="${BASE}_id${SAFE_ID}"
  else
    TAG="${BASE}_idx${ONE_IDX}"
  fi
  LIMIT_ARG=""
  BERT_ARGS="--folds 2 --epochs 1"   # a 1-row dataset cannot support 5-fold CV
else
  BASE="$(basename "$DATASET" .csv)"
  if [[ "$LIMIT" -gt 0 ]]; then
    LIMIT_ARG="--limit $LIMIT"
    BERT_ARGS="--limit $LIMIT --folds 3 --epochs 2"
    TAG="${BASE}_pilot${LIMIT}"
  else
    LIMIT_ARG=""
    BERT_ARGS="--folds 5 --epochs 4"
    TAG="$BASE"
  fi
fi

ONTO_OUT="results/ontology_results_${TAG}.csv"
MCQ_OUT="results/mcq_ontology_results_${TAG}.csv"
BERT_OUT="results/bert_results_${TAG}.csv"

say "Setup"
if [[ -n "$ONE_ID" ]]; then
  ok "dataset   $DATASET  (1 row, id=$ONE_ID)"
elif [[ -n "$ONE_IDX" ]]; then
  ok "dataset   $DATASET  (1 row, idx=$ONE_IDX of a --limit $IDX_LIMIT_USED order)"
else
  ok "dataset   $DATASET  ($(rows_of "$DATASET") rows)"
fi
ok "baseline  ${SEL[*]}"
if [[ -n "$ONE_ID" ]]; then
  ok "limit     single transcript (id=$ONE_ID)"
elif [[ -n "$ONE_IDX" ]]; then
  ok "limit     single transcript (idx=$ONE_IDX)"
else
  ok "limit     $( [[ "$LIMIT" -gt 0 ]] && echo "$LIMIT calls" || echo "all rows" )"
fi
ok "model     ${MODEL:-none needed for this baseline}"
ok "logs      $LOGDIR"
# a single-transcript run is for reading the model reasoning, so it turns
# on --debug where the baseline supports it (every step now streams its output
# either way)
SINGLE_MODE=0
[[ -n "$ONE_ID" || -n "$ONE_IDX" ]] && SINGLE_MODE=1
[[ -n "$LIMIT_ARG" ]] && warn "pilot mode: $LIMIT_ARG"
if [[ ( -n "$ONE_ID" || -n "$ONE_IDX" ) && "$RUN_BERT" -eq 1 ]]; then
  warn "BERT needs several rows per fold to train on; a 1-row run will fail or be meaningless"
fi

[[ -n "$MODEL" ]] && export SCAM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL"

declare -A STATUS DURATION
STEPS_RUN=()
START_ALL=$(date +%s)

# mm:ss for anything under an hour, h:mm:ss past that
hms() {
  local s="$1"
  if (( s >= 3600 )); then
    printf "%dh%02dm%02ds" $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 ))
  else
    printf "%dm%02ds" $(( s / 60 )) $(( s % 60 ))
  fi
}

# Printed alongside a running step. Every baseline reports its own progress
# ("LLM-only: 40/198", "fold 2/3"), but a single LLM call can take a minute,
# and a terminal that has gone quiet for a minute looks exactly like one that
# has hung. Comparing the log size against last time says which it is.
HEARTBEAT_SECS="${HEARTBEAT_SECS:-30}"
heartbeat() {
  local log="$1" t0="$2" last=0 size now el quiet last_out
  last_out=$(date +%s)
  while true; do
    sleep "$HEARTBEAT_SECS"
    size=$(wc -c < "$log" 2>/dev/null || echo 0)
    now=$(date +%s); el=$(( now - t0 ))

    # Output arriving is itself proof of life, so say nothing on top of it.
    # The heartbeat is only there to fill a silence.
    if (( size > last )); then
      last="$size"; last_out="$now"
      continue
    fi

    # A gap of one interval is normal between two progress lines - only a
    # gap long enough to look like a stall is worth a line.
    quiet=$(( now - last_out ))
    (( quiet < 2 * HEARTBEAT_SECS )) && continue

    if (( quiet > 600 )); then
      # ten minutes without a word, measured from the last output rather than
      # from the start of the step, which is the difference between "stalled"
      # and "has simply been running a long time"
      echo -e "    ${YLW}...${NC} $(hms "$el") into this step, nothing printed for" \
              "$(hms "$quiet") - check nvidia-smi / ollama ps"
    else
      echo -e "    ${CYN}...${NC} $(hms "$el") into this step, still working" \
              "(last output $(hms "$quiet") ago)"
    fi
  done
}

run_step() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  STEPS_RUN+=("$name")
  say "$name"
  echo "  $*"
  echo "  started $(date +%H:%M:%S)"
  local t0; t0=$(date +%s)

  heartbeat "$log" "$t0" &
  local hb=$!

  # tee first, indent second: the log keeps the exact output the script
  # produced (collect_results.py and the web UI both read it), while the
  # copy on screen is indented so it sits under this step, live.
  # A read loop rather than sed: sed block-buffers when its stdout is a file
  # rather than a terminal, which is exactly the case under the web UI, and
  # the whole run would then land in one lump at the end. read is always one
  # line at a time.
  local status line
  "$@" 2>&1 | tee "$log" | while IFS= read -r line || [[ -n "$line" ]]; do
    printf "    %s\n" "$line"
  done
  status=${PIPESTATUS[0]}

  kill "$hb" 2>/dev/null
  wait "$hb" 2>/dev/null

  if [[ "$status" -eq 0 ]]; then
    STATUS[$name]="ok"
  else
    STATUS[$name]="FAILED"
  fi
  DURATION[$name]=$(( $(date +%s) - t0 ))
  if [[ "${STATUS[$name]}" == "ok" ]]; then
    ok "$name finished in $(hms "${DURATION[$name]}")"
  else
    warn "$name FAILED after $(hms "${DURATION[$name]}") - full log: $log"
  fi
}

# ------------------------------------------------- 1. combined_evaluate.py
# length and bag-of-words are instant, so they come along with any of the
# three LLM baselines that live in this script
# Single-transcript mode has no training split to hold out, so the two
# learning baselines - Qwen-KB and the hybrid - are told to learn from the
# dataset the row was carved out of instead. They share the flag because they
# share the patterns.
QWEN_ARGS=""
if [[ "$SINGLE_MODE" -eq 1 ]] && { has_bl qwen_kb || has_bl hybrid; }; then
  QWEN_ARGS="--qwen-train-csv $FULL_DATASET"
fi

if [[ "$RUN_COMBINED" -eq 1 ]]; then
  # shellcheck disable=SC2086
  run_step "combined" python -u scripts/combined_evaluate.py \
    --csv "$DATASET" $LIMIT_ARG $COMBINED_EXTRA $QWEN_ARGS
fi

# ---------------------------------------------------------- 2. ontology RAG
if [[ "$RUN_ONTOLOGY" -eq 1 ]]; then
  DEBUG_FLAG=""; [[ "$SINGLE_MODE" -eq 1 ]] && DEBUG_FLAG="--debug"
  if [[ "$SINGLE_MODE" -eq 1 ]]; then
    warn "evaluate_ontology.py (term-list) has no per-question explain output;"
    warn "choose baseline 'mcq' instead to see the Agent/Caller breakdown"
  fi
  # shellcheck disable=SC2086
  run_step "ontology" python -u scripts/evaluate_ontology.py \
    --csv "$DATASET" --model "$MODEL" --ontology "$ONTOLOGY" \
    --out "$ONTO_OUT" $LIMIT_ARG $DEBUG_FLAG
fi

# ---------------------------------------------------------- 3. MCQ ontology
if [[ "$RUN_MCQ" -eq 1 ]]; then
  DEBUG_FLAG=""; [[ "$SINGLE_MODE" -eq 1 ]] && DEBUG_FLAG="--debug"
  # shellcheck disable=SC2086
  run_step "mcq" python -u scripts/evaluate_mcq_ontology.py \
    --csv "$DATASET" --model "$MODEL" --ontology "$MCQ_ONTOLOGY" \
    --out "$MCQ_OUT" $LIMIT_ARG $DEBUG_FLAG
fi

# --------------------------------------------------------- 4. free GPU, BERT
# Ask Ollama what it is currently holding in VRAM. Prints one model name per
# line, and nothing at all if Ollama is down or idle.
ollama_loaded() {
  curl -s -m 5 "$OLLAMA_URL/api/ps" 2>/dev/null | python3 -c "
import json, sys
try:
    for m in json.load(sys.stdin).get('models', []):
        name = m.get('name') or m.get('model')
        if name:
            print(name)
except Exception:
    pass
" 2>/dev/null
}

BERT_MIN_FREE_MB="${BERT_MIN_FREE_MB:-4000}"

if [[ "$RUN_BERT" -eq 1 ]]; then
  # Every resident model has to go, not just $MODEL. When the baseline is
  # "bert" on its own $MODEL is empty, but a qwen2.5:14b left loaded by an
  # earlier run still owns ~9 GB, and BERT then OOMs part way into fold 1.
  if [[ "$OLLAMA_UP" -eq 1 ]]; then
    LOADED="$(ollama_loaded)"
    if [[ -n "$LOADED" ]]; then
      while read -r m; do
        [[ -z "$m" ]] && continue
        say "Unloading $m"
        curl -s "$OLLAMA_URL/api/generate" \
          -d "{\"model\":\"$m\",\"prompt\":\"\",\"keep_alive\":0}" >/dev/null
      done <<< "$LOADED"
      sleep 5
    fi
  fi
  if command -v nvidia-smi >/dev/null && [[ "$BERT_MIN_FREE_MB" -gt 0 ]]; then
    FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    ok "${FREE} MB VRAM free"
    if [[ "$FREE" -lt "$BERT_MIN_FREE_MB" ]]; then
      # Stopping here costs a second. Going ahead costs minutes and still fails.
      warn "BERT needs about ${BERT_MIN_FREE_MB} MB, only ${FREE} MB is free"
      [[ "$OLLAMA_UP" -eq 1 ]] && warn "still resident: $(ollama_loaded | tr '\n' ' ')"
      die "not enough VRAM for BERT
       see what holds the GPU with:  nvidia-smi
       if it is Ollama:              ollama stop <model>
       to try anyway:                BERT_MIN_FREE_MB=0 ./run_all.sh ..."
    fi
  fi
  # shellcheck disable=SC2086
  run_step "bert" python -u scripts/bert_baseline.py cv \
    --csv "$DATASET" --out "$BERT_OUT" $BERT_ARGS
fi

# ------------------------------------------------------------------ results
ELAPSED=$(( $(date +%s) - START_ALL ))

echo
echo "  dataset: $DATASET   baseline: $BASELINE   model: ${MODEL:-none}"
python scripts/collect_results.py "$LOGDIR"

echo "=========================================================================="
printf " timing:"
for n in "${STEPS_RUN[@]}"; do
  printf "  %s %ss" "$n" "${DURATION[$n]}"
done
printf "\n total: %dh %dm\n" $((ELAPSED/3600)) $((ELAPSED%3600/60))
echo
echo " per-call CSVs:"
[[ "$RUN_COMBINED" -eq 1 ]] && echo "   results/combined_results_*.csv"
[[ "$RUN_ONTOLOGY" -eq 1 ]] && echo "   $ONTO_OUT"
[[ "$RUN_MCQ" -eq 1 ]] && echo "   $MCQ_OUT"
[[ "$RUN_BERT" -eq 1 ]] && echo "   $BERT_OUT"
echo " logs: $LOGDIR"
echo "=========================================================================="

FAILED=0
for n in "${!STATUS[@]}"; do
  [[ "${STATUS[$n]}" == "FAILED" ]] && FAILED=1
done
exit "$FAILED"