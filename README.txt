cd ~/scam-detection && source venv/bin/activate

# 1. Five baselines: length, BoW, LLM-only, Singh, Web-RAG
export SCAM_MODEL=qwen2.5:14b
python scripts/combined_evaluate.py --csv datasets/zhi_scam_vs_legit_794.csv --limit 20

# 2. Ontology RAG
python scripts/evaluate_ontology.py \
  --csv datasets/zhi_scam_vs_legit_794.csv \
  --model qwen2.5:14b \
  --ontology knowledge/scam_ontology.json
  
# 3. Unload qwen, then BERT
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"","keep_alive":0}' > /dev/null

python scripts/bert_baseline.py cv --csv datasets/zhi_scam_vs_legit_794.csv \
  --out results/bert_results_794.csv

python scripts/evaluate_mcq_ontology.py \
  --csv datasets/zhi_scam_vs_legit_794.csv \
  --model qwen2.5:14b \
  --limit 20 --debug




  datasets
  scam_vs_bank_243x243.csv — 486 real calls: 243 scam calls transcribed from YouTube scam-baiting videos, paired with 243 legitimate calls from the HarperValleyBank corpus. Both sides are real recorded speech. This is the dataset in your current proposal tables (referred to there as the 486)
  zhi_balanced_333x333.csv — 666 rows from the Zhi et al. corpus, template-generated text, after filtered out non-conversational entries (things with [Link] tokens etc.) and downsampled the legitimate side to match the scam side 1:1. This is a cleaned descendant of the original Zhi data.
  paired_scam_legit_198.csv — 198 rows, 99 topic-matched pairs. Real YouTube scam transcripts paired with LLM hand-written legitimate counterparts on the same topic, then laundered through blind rewriting so style doesn't leak. This is your best-controlled dataset — the one where BoW still sat at 99% even after every confound was removed, which is strongest evidence for the "detection on clean transcripts is a lexical task" argument.