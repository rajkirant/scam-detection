cd ~/scam-detection && source venv/bin/activate


# =====================================================================
#  A. Browser UI - everything run_all.sh does, as a form
# =====================================================================

./web_ui.sh                   # http://localhost:8000, ctrl-c to stop
./web_ui.sh --tmux            # detached: survives an SSH disconnect
./web_ui.sh --port 8080       # somewhere else
./web_ui.sh --local           # this machine only (then forward the port)
./web_ui.sh --public          # plus an https link that works from anywhere

# It binds every interface, so another machine on the same network can open
# it directly - the startup banner prints the address to use:
#
#   scam-detection UI
#     here:           http://localhost:8000
#     other machines: http://10.196.217.243:8000
#
# --public also opens an SSH reverse tunnel to localhost.run (no account
# needed) and prints the public URL:
#
#   ==> Opening a public link
#     ok public    https://fa58e6c3b454ab.lhr.life
#
# The tunnel is supervised - localhost.run drops it eventually, and a new
# one is opened and announced in the terminal. With an SSH key on this
# machine the address survives a reconnect; without one it changes each
# time (ssh-keygen -t ed25519, once, fixes that). See
# results/logs/tunnel.log for the connection history.
#
# That link has no password in front of it: anyone who opens it can start
# and stop runs on this box and read every transcript. Fine for showing a
# result to someone for ten minutes, not something to leave up. On an
# untrusted network use --local and tunnel in yourself instead:
#
#   ssh -L 8000:localhost:8000 rkt29@cs25003ay
#
# Runs started from the page are detached from the server, so closing the
# browser, dropping the SSH link, or restarting the server does not stop
# them - reopen the URL and the run is still there. Pick a run under
# "Recent runs" to get its Output, Results, Per-call predictions and the
# per-baseline Step logs.
#
# Under the Run button, "Web-RAG knowledge base" does what section C.6 does
# by hand: harvest_patterns.py refreshes knowledge/scam_patterns.json, then
# build_index.py re-embeds it into chroma_db, which is what the webrag
# baseline retrieves from. The line above the picker is the current state -
# patterns in the JSON, vectors in the index, and the date of the last
# harvest - and it turns amber when the two counts disagree, i.e. the index
# needs rebuilding. Five options:
#
#   Harvest the web, then rebuild the index    the weekly refresh
#   Harvest ignoring the cache, then rebuild   re-fetches every seed query
#   Rebuild the index only                     no web calls, no LLM
#   Dry run - show the plan                    stops before any network call
#   Stats only                                 what the KB already holds
#
# Harvesting needs TAVILY_API_KEY in .env and Ollama up; the last two options
# need neither and can be used while a benchmark is running. The others take
# the same lock a benchmark does - the index cannot be rebuilt underneath a
# run that is reading it. The update streams into the same Output pane and
# lands in "Recent runs" like any other run.


# =====================================================================
#  B. The same thing in the terminal
# =====================================================================

./run_all.sh                  # asks five questions, then runs
./run_all.sh --tmux           # ... and detaches into tmux

# or answer up front and it asks nothing:
./run_all.sh -d 3 -b all -l 0 -m 1
./run_all.sh --dataset datasets/paired_scam_legit_198.csv \
             --baseline llm_only,mcq,bert --limit 40 --model qwen2.5:14b

# baselines: all trivial llm_only singh webrag qwen_kb hybrid ontology mcq bert
#   comma-separate for several - "-b 3,7,8" works too (menu numbers)
# limit:     0 = whole dataset, N = first N calls,
#            id:<value> = one row by its id column,
#            idx:<n> = the n-th call of a --limit 40 style run
#
# Each step streams its progress as it goes. While that output is flowing
# nothing else is printed - it is its own proof of life. Only a silence of
# a minute or more draws a heartbeat line, and only ten minutes of actual
# silence turns it into a warning. HEARTBEAT_SECS=10 ./run_all.sh checks
# more often.
#
# Results land in results/logs/run_<stamp>/ (one log per baseline) and
# results/*.csv (one row per call). Every LLM system also records the
# reason it gave for each call, in a <system>_why column next to its
# verdict - the Per-call tab shows them, and the checkbox above the
# table hides them again when the width gets in the way.


# =====================================================================
#  C. The individual scripts, if you want one on its own
# =====================================================================

# 1. Seven baselines: length, BoW, LLM-only, Singh, Web-RAG, Qwen-KB, Hybrid
export SCAM_MODEL=qwen2.5:14b
python scripts/combined_evaluate.py --csv datasets/paired_scam_legit_198.csv --limit 20
#   --skip length,bow,singh,webrag,qwen_kb,hybrid   run just one of the seven
#
# Qwen-KB is the learning baseline. Like BERT and bag-of-words it is
# cross-validated: on each fold the model generalises that fold's TRAINING
# scams into patterns, those patterns are indexed as a knowledge base, and
# the held-out calls are judged against what is retrieved from it. Every
# call is scored exactly once, while it was held out, so its row in the
# results table is comparable with the rest.
#   --qwen-folds 3            fewer folds = fewer KB-building calls
#   --qwen-patterns 8         size of the learned KB per fold
#   --qwen-max-examples 40    training scams sampled per fold
#   --qwen-train-csv FILE     learn from a separate file instead of folds
#                             (what a single-transcript run uses)
#
# Hybrid is Web-RAG and Qwen-KB sharing ONE knowledge base: the web-harvested
# patterns in chroma_db and the patterns learned from the training split go
# into the same collection, and Web-RAG's pipeline - signal extraction, the
# two-stage relevance gate, graded 0-100 confidence, threshold - runs over the
# mix. Retrieval decides per call which kind of knowledge is worth showing,
# and the prompt labels which is which. It runs on the same folds as Qwen-KB
# and reuses the patterns Qwen-KB already learned (the fold KB is built once
# and shared), so hybrid vs qwen_kb isolates the web KB and hybrid vs webrag
# isolates the learned patterns. Watch the "evidence mix" line it prints: all
# web means the learned patterns are not earning their place, all learned
# means it is Qwen-KB with a slower pipeline.
#   --skip hybrid             the merged run costs 3 LLM calls per transcript,
#                             the same as webrag

# 2. Ontology RAG
python scripts/evaluate_ontology.py \
  --csv datasets/paired_scam_legit_198.csv \
  --model qwen2.5:14b \
  --ontology knowledge/scam_ontology.json

# 3. Unload qwen, then BERT - qwen holds ~9.5 GB of 11.4 GB and BERT
#    fine-tuning needs 3-4 GB, so they cannot both be resident
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"","keep_alive":0}' > /dev/null

python scripts/bert_baseline.py cv --csv datasets/paired_scam_legit_198.csv \
  --out results/bert_results_198.csv

# 4. MCQ ontology (two LLM calls per transcript)
python scripts/evaluate_mcq_ontology.py \
  --csv datasets/paired_scam_legit_198.csv \
  --model qwen2.5:14b \
  --limit 20 --debug

# 5. Re-print the table for any past run
python scripts/collect_results.py results/logs/run_<stamp>

# 6. Refresh the Web-RAG knowledge base (needs TAVILY_API_KEY in .env).
#    The JSON is the source of truth; the vector index is derived from it,
#    so the rebuild follows the harvest.
python scripts/harvest_patterns.py --stats      # what is in there now
python scripts/harvest_patterns.py --dry-run    # the plan, no network
python scripts/harvest_patterns.py              # harvest (--no-cache to refetch)
python scripts/build_index.py                   # re-embed into chroma_db
python scripts/build_index.py --test            # + show which hits pass the gate

# 7. Web-RAG relevance gate.
#    The KB holds scam patterns only, so a vector search hands back three of
#    them for EVERY call, legitimate ones included. Retrieved chunks now have
#    to prove they match the transcript before reaching the prompt: a cheap
#    similarity floor, then a yes/no LLM relevance check. When nothing
#    survives, the model is told so and judges the transcript alone.
#      WEBRAG_MIN_SIMILARITY=0.35   cosine floor (pre-filter only - measured
#                                   Fraud/Normal distributions overlap almost
#                                   exactly, so this cannot separate on its own)
#      WEBRAG_LLM_GATE=1            the check that actually decides; =0 to
#                                   ablate it, at one fewer Ollama call each
python scripts/test_relevance_gate.py           # gate logic, offline, no Ollama




  datasets
  scam_vs_bank_243x243.csv — 486 real calls: 243 scam calls transcribed from YouTube scam-baiting videos, paired with 243 legitimate calls from the HarperValleyBank corpus. Both sides are real recorded speech. This is the dataset in your current proposal tables (referred to there as the 486)
  zhi_balanced_333x333.csv — 666 rows from the Zhi et al. corpus, template-generated text, after filtered out non-conversational entries (things with [Link] tokens etc.) and downsampled the legitimate side to match the scam side 1:1. This is a cleaned descendant of the original Zhi data.
  paired_scam_legit_198.csv — 198 rows, 99 topic-matched pairs. Real YouTube scam transcripts paired with LLM hand-written legitimate counterparts on the same topic, then laundered through blind rewriting so style doesn't leak. This is your best-controlled dataset — the one where BoW still sat at 99% even after every confound was removed, which is strongest evidence for the "detection on clean transcripts is a lexical task" argument.