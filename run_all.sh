#!/usr/bin/env bash
#
# run_all.sh - interactive launcher for the scam-detection benchmark.
#
#   ./run_all.sh
#
# Asks up to four questions, then runs just what you picked:
#
#   1. which dataset    (numbered menu of every CSV in datasets/)
#   2. which baseline   (one system, or all of them)
#   3. how many calls   (0 = the whole dataset, a number = the first N,
#                         id:<value> = the one row whose id column matches,
#                         idx:<n> = the same call as index n in a --limit 40
#                         style shuffled run, matching check_one.py --idx)
#   4. which model      (only asked when the baseline actually calls an LLM)
#
# Everything can also be given up front, which skips the questions:
#
#   ./run_all.sh --dataset datasets/zhi_balanced_333x333.csv --baseline singh \
#                --limit 40 --model qwen2.5:14b
#   ./run_all.sh --dataset datasets/paired_scam_legit_198.csv --baseline mcq \
#                --limit idx:19 --model qwen2.5:14b  # the SAME call as --idx 19
#                                                     # in check_one.py / a --limit 40 run
#   ./run_all.sh -d 2 -b 4 -l 0 -m 1          # menu numbers work too
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

BL_KEYS=(all trivial llm_only singh webrag ontology mcq bert)
BL_LABELS=(
  "all                       every system below, in one run"
  "length + bag-of-words     trivial references, no LLM"
  "LLM-only                  the model decides alone, no retrieval"
  "Singh                     policy-compliance baseline"
  "Web-RAG                   KB-only retrieval"
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
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)  ARG_DATASET="${2:-}";  shift 2 ;;
    -b|--baseline) ARG_BASELINE="${2:-}"; shift 2 ;;
    -l|--limit)    ARG_LIMIT="${2:-}";    shift 2 ;;
    -m|--model)    ARG_MODEL="${2:-}";    shift 2 ;;
    -h|--help)     sed -n '2,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
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
ask_baseline() {
  echo
  echo -e "${BLD}  Which baseline?${NC}"
  local i
  for i in "${!BL_KEYS[@]}"; do
    printf "    %d) %s\n" $((i + 1)) "${BL_LABELS[$i]}"
  done
  local pick=""
  while true; do
    prompt "  choose 1-${#BL_KEYS[@]}: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#BL_KEYS[@]} )); then
      BASELINE="${BL_KEYS[$((pick - 1))]}"
      return
    fi
    warn "enter a number between 1 and ${#BL_KEYS[@]}"
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
ask_limit() {
  echo
  echo -e "${BLD}  How many calls?${NC}  (0 = whole dataset, id:<value> = one row by its id column,\n  idx:<n> = the same call as index n in a --limit 40 style run)"
  local pick=""
  while true; do
    prompt "  limit: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]]; then
      LIMIT="$pick"; ONE_ID=""
      return
    fi
    if [[ "$pick" =~ ^id:(.+)$ ]]; then
      LIMIT=0; ONE_ID="${BASH_REMATCH[1]}"
      return
    fi
    warn "enter a whole number, 0 for all, or id:<value> e.g. id:19"
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
  if [[ "$ARG_BASELINE" =~ ^[0-9]+$ ]] && (( ARG_BASELINE >= 1 && ARG_BASELINE <= ${#BL_KEYS[@]} )); then
    BASELINE="${BL_KEYS[$((ARG_BASELINE - 1))]}"
  else
    BASELINE="$ARG_BASELINE"
    printf "%s\n" "${BL_KEYS[@]}" | grep -qx "$BASELINE" \
      || die "unknown baseline: $BASELINE  (one of: ${BL_KEYS[*]})"
  fi
else
  ask_baseline
fi

# what the chosen baseline actually runs
RUN_COMBINED=0; RUN_ONTOLOGY=0; RUN_MCQ=0; RUN_BERT=0; COMBINED_EXTRA=""; NEEDS_MODEL=1
case "$BASELINE" in
  all)      RUN_COMBINED=1; RUN_ONTOLOGY=1; RUN_MCQ=1; RUN_BERT=1 ;;
  trivial)  RUN_COMBINED=1; COMBINED_EXTRA="--trivial-only"; NEEDS_MODEL=0 ;;
  llm_only) RUN_COMBINED=1; COMBINED_EXTRA="--skip singh,webrag" ;;
  singh)    RUN_COMBINED=1; COMBINED_EXTRA="--skip llm_only,webrag" ;;
  webrag)   RUN_COMBINED=1; COMBINED_EXTRA="--skip llm_only,singh" ;;
  ontology) RUN_ONTOLOGY=1 ;;
  mcq)      RUN_MCQ=1 ;;
  bert)     RUN_BERT=1; NEEDS_MODEL=0 ;;
esac

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
ok "baseline  $BASELINE"
if [[ -n "$ONE_ID" ]]; then
  ok "limit     single transcript (id=$ONE_ID)"
elif [[ -n "$ONE_IDX" ]]; then
  ok "limit     single transcript (idx=$ONE_IDX)"
else
  ok "limit     $( [[ "$LIMIT" -gt 0 ]] && echo "$LIMIT calls" || echo "all rows" )"
fi
ok "model     ${MODEL:-none needed for this baseline}"
ok "logs      $LOGDIR"
[[ -n "$LIMIT_ARG" ]] && warn "pilot mode: $LIMIT_ARG"
if [[ ( -n "$ONE_ID" || -n "$ONE_IDX" ) && "$RUN_BERT" -eq 1 ]]; then
  warn "BERT needs several rows per fold to train on; a 1-row run will fail or be meaningless"
fi

[[ -n "$MODEL" ]] && export SCAM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL"

declare -A STATUS DURATION
STEPS_RUN=()
START_ALL=$(date +%s)

run_step() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  STEPS_RUN+=("$name")
  say "$name"
  echo "  $*"
  local t0; t0=$(date +%s)
  if "$@" > "$log" 2>&1; then
    STATUS[$name]="ok"
  else
    STATUS[$name]="FAILED"
  fi
  DURATION[$name]=$(( $(date +%s) - t0 ))
  if [[ "${STATUS[$name]}" == "ok" ]]; then
    ok "${DURATION[$name]}s"
  else
    warn "failed after ${DURATION[$name]}s"
    tail -12 "$log" | sed 's/^/     /'
  fi
}

# ------------------------------------------------- 1. combined_evaluate.py
# length and bag-of-words are instant, so they come along with any of the
# three LLM baselines that live in this script
if [[ "$RUN_COMBINED" -eq 1 ]]; then
  # shellcheck disable=SC2086
  run_step "combined" python -u scripts/combined_evaluate.py \
    --csv "$DATASET" $LIMIT_ARG $COMBINED_EXTRA
fi

# ---------------------------------------------------------- 2. ontology RAG
if [[ "$RUN_ONTOLOGY" -eq 1 ]]; then
  # shellcheck disable=SC2086
  run_step "ontology" python -u scripts/evaluate_ontology.py \
    --csv "$DATASET" --model "$MODEL" --ontology "$ONTOLOGY" \
    --out "$ONTO_OUT" $LIMIT_ARG
fi

# ---------------------------------------------------------- 3. MCQ ontology
if [[ "$RUN_MCQ" -eq 1 ]]; then
  # shellcheck disable=SC2086
  run_step "mcq" python -u scripts/evaluate_mcq_ontology.py \
    --csv "$DATASET" --model "$MODEL" --ontology "$MCQ_ONTOLOGY" \
    --out "$MCQ_OUT" $LIMIT_ARG
fi

# --------------------------------------------------------- 4. free GPU, BERT
if [[ "$RUN_BERT" -eq 1 ]]; then
  if [[ -n "$MODEL" && "$OLLAMA_UP" -eq 1 ]]; then
    say "Unloading $MODEL"
    curl -s "$OLLAMA_URL/api/generate" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"\",\"keep_alive\":0}" >/dev/null
    sleep 5
  fi
  if command -v nvidia-smi >/dev/null; then
    FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    ok "${FREE} MB VRAM free"
    [[ "$FREE" -lt 4000 ]] && warn "low, BERT may hit CUDA OOM"
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