#!/usr/bin/env bash
#
# run_all.sh - interactive launcher for the scam-detection benchmark.
#
#   ./run_all.sh
#
# Asks three questions, then runs the whole pipeline on that one dataset:
#
#   1. which dataset  (numbered menu of every CSV in datasets/)
#   2. how many calls (0 = the whole dataset)
#   3. which model    (qwen2.5:14b or llama3.1:8b, both served by Ollama)
#
# Everything can also be given up front, which skips the questions:
#
#   ./run_all.sh --dataset datasets/zhi_balanced_333x333.csv --limit 40 --model qwen2.5:14b
#   ./run_all.sh -d 2 -l 0 -m 1          # menu numbers work too
#
# BERT runs last because qwen2.5:14b holds ~9.5 GB of 11.4 GB VRAM and BERT
# fine-tuning needs 3-4 GB. They cannot both be resident.
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
MODELS=("qwen2.5:14b" "llama3.1:8b")

# ---------------------------------------------------------------- arguments
ARG_DATASET=""; ARG_LIMIT=""; ARG_MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset) ARG_DATASET="${2:-}"; shift 2 ;;
    -l|--limit)   ARG_LIMIT="${2:-}";   shift 2 ;;
    -m|--model)   ARG_MODEL="${2:-}";   shift 2 ;;
    -h|--help)    sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)            die "unknown argument: $1  (try --help)" ;;
  esac
done

prompt() {            # prompt <text> <varname>   - read only echoes its own
  printf "%s" "$1"    # prompt on a terminal, so print it here instead
  read -r "$2" || die "no input (answers can also be piped in, or use --dataset/--limit/--model)"
}

# ------------------------------------------------------------- dataset menu
shopt -s nullglob
DATASETS=(datasets/*.csv)
shopt -u nullglob
[[ ${#DATASETS[@]} -gt 0 ]] || die "no CSV files in datasets/"
NWIDTH=${#DATASETS[@]}; NWIDTH=${#NWIDTH}   # digits in the highest menu number

rows_of() { echo $(( $(wc -l < "$1") - 1 )); }

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

ask_limit() {
  echo
  echo -e "${BLD}  How many calls?${NC}  (0 = the whole dataset)"
  local pick=""
  while true; do
    prompt "  limit: " pick
    if [[ "$pick" =~ ^[0-9]+$ ]]; then
      LIMIT="$pick"
      return
    fi
    warn "enter a whole number, 0 for all"
  done
}

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

if [[ -n "$ARG_LIMIT" ]]; then
  [[ "$ARG_LIMIT" =~ ^[0-9]+$ ]] || die "--limit needs a whole number, got '$ARG_LIMIT'"
  LIMIT="$ARG_LIMIT"
else
  ask_limit
fi

if [[ -n "$ARG_MODEL" ]]; then
  if [[ "$ARG_MODEL" =~ ^[0-9]+$ ]] && (( ARG_MODEL >= 1 && ARG_MODEL <= ${#MODELS[@]} )); then
    MODEL="${MODELS[$((ARG_MODEL - 1))]}"
  else
    MODEL="$ARG_MODEL"
  fi
else
  ask_model
fi

# --------------------------------------------------------------- preflight
# every LLM step needs Ollama, so check once here rather than failing three
# times over in a row
if [[ "$OLLAMA_UP" -ne 1 ]]; then
  echo
  die "Ollama is not answering at $OLLAMA_URL
       start it with:  ollama serve
       (this project's LLM steps run on the box that has Ollama and venv/)"
fi
if ! is_pulled "$MODEL"; then
  warn "$MODEL is not pulled on this machine.  Get it with:  ollama pull $MODEL"
fi

# ------------------------------------------------------------------- setup
[[ -f "$DATASET" ]]  || die "missing $DATASET"
[[ -f "$ONTOLOGY" ]] || die "missing $ONTOLOGY"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate || die "venv failed"
elif [[ -f venv/Scripts/activate ]]; then
  # shellcheck disable=SC1091
  source venv/Scripts/activate || die "venv failed"
else
  warn "no venv/ found, using whatever python is on PATH"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/logs/run_${STAMP}"
mkdir -p "$LOGDIR"

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

ONTO_OUT="results/ontology_results_${TAG}.csv"
BERT_OUT="results/bert_results_${TAG}.csv"

say "Setup"
ok "dataset  $DATASET  ($(rows_of "$DATASET") rows)"
ok "model    $MODEL"
ok "limit    $( [[ "$LIMIT" -gt 0 ]] && echo "$LIMIT calls" || echo "all rows" )"
ok "logs     $LOGDIR"
[[ -n "$LIMIT_ARG" ]] && warn "pilot mode: $LIMIT_ARG"

export SCAM_MODEL="$MODEL" OLLAMA_MODEL="$MODEL"

declare -A STATUS DURATION
START_ALL=$(date +%s)

run_step() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
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

# ------------------------------------------------------- 1. five baselines
# shellcheck disable=SC2086
run_step "combined" python -u scripts/combined_evaluate.py \
  --csv "$DATASET" $LIMIT_ARG

# ------------------------------------------------------- 2. ontology RAG
# shellcheck disable=SC2086
run_step "ontology" python -u scripts/evaluate_ontology.py \
  --csv "$DATASET" --model "$MODEL" --ontology "$ONTOLOGY" \
  --out "$ONTO_OUT" $LIMIT_ARG

# ------------------------------------------------------- 3. free GPU, BERT
say "Unloading $MODEL"
curl -s http://localhost:11434/api/generate \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"\",\"keep_alive\":0}" >/dev/null
sleep 5
if command -v nvidia-smi >/dev/null; then
  FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  ok "${FREE} MB VRAM free"
  [[ "$FREE" -lt 4000 ]] && warn "low, BERT may hit CUDA OOM"
fi

# shellcheck disable=SC2086
run_step "bert" python -u scripts/bert_baseline.py cv \
  --csv "$DATASET" --out "$BERT_OUT" $BERT_ARGS

# ------------------------------------------------------------------ results
ELAPSED=$(( $(date +%s) - START_ALL ))

echo
echo "  dataset: $DATASET   model: $MODEL"
python scripts/collect_results.py "$LOGDIR"

echo "=========================================================================="
printf " timing:"
for n in combined ontology bert; do
  [[ -v STATUS[$n] ]] && printf "  %s %ss" "$n" "${DURATION[$n]}"
done
printf "\n total: %dh %dm\n" $((ELAPSED/3600)) $((ELAPSED%3600/60))
echo
echo " per-call CSVs:"
echo "   results/combined_results_*.csv"
echo "   $ONTO_OUT"
echo "   $BERT_OUT"
echo " logs: $LOGDIR"
echo "=========================================================================="

FAILED=0
for n in "${!STATUS[@]}"; do
  [[ "${STATUS[$n]}" == "FAILED" ]] && FAILED=1
done
exit "$FAILED"
