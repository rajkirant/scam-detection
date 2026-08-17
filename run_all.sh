#!/usr/bin/env bash
#
# run_all.sh - run every system on the Zhi 794 dataset and print one table
#
#   ./run_all.sh              full run (2.5-4 hrs, use tmux)
#   ./run_all.sh --limit 40   quick pilot (~10 min)
#
# BERT runs last because qwen2.5:14b holds ~9.5 GB of 11.4 GB VRAM and BERT
# fine-tuning needs 3-4 GB. They cannot both be resident.
#
set -uo pipefail

MODEL="qwen2.5:14b"
DATASET="datasets/zhi_scam_vs_legit_794.csv"
ONTOLOGY="scripts/scam_ontology.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/logs/run_${STAMP}"

# pass through --limit N if given
LIMIT_ARG=""
BERT_ARGS="--folds 5 --epochs 4"
TAG="794"
if [[ "${1:-}" == "--limit" && -n "${2:-}" ]]; then
  LIMIT_ARG="--limit $2"
  BERT_ARGS="--limit $2 --folds 3 --epochs 2"
  TAG="pilot$2"
fi

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
say()  { echo -e "\n${CYN}==>${NC} $*"; }
ok()   { echo -e "${GRN}  ok${NC} $*"; }
warn() { echo -e "${YLW}  warn${NC} $*"; }
die()  { echo -e "${RED}  fail${NC} $*"; exit 1; }

cd "$HOME/scam-detection" || die "cannot cd to project"
# shellcheck disable=SC1091
source venv/bin/activate || die "venv failed"
[[ -f "$DATASET" ]]  || die "missing $DATASET"
[[ -f "$ONTOLOGY" ]] || die "missing $ONTOLOGY"
mkdir -p "$LOGDIR"

ONTO_OUT="results/ontology_results_${TAG}.csv"
BERT_OUT="results/bert_results_${TAG}.csv"

say "Setup"
ok "dataset  $DATASET  ($(( $(wc -l < "$DATASET") - 1 )) rows)"
ok "model    $MODEL"
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
