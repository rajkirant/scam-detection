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