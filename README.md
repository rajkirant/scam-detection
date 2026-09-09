# scam-detection

Benchmark harness for scam-call detection over call transcripts. It runs nine
systems — from a word-count threshold up to a retrieval-augmented local LLM —
over the same datasets and reports accuracy, precision, recall and F1 for each,
with one row per call and the model's stated reason next to every verdict.

Everything runs locally: the LLM is served by [Ollama](https://ollama.com) on
your own machine, retrieval is a local Chroma index, and the only optional
network dependency is the Tavily search API used to refresh the web knowledge
base.

- [What is in here](#what-is-in-here)
- [Setup from scratch](#setup-from-scratch)
- [First run](#first-run)
- [A. Browser UI](#a-browser-ui)
- [B. The same thing in the terminal](#b-the-same-thing-in-the-terminal)
- [C. The individual scripts](#c-the-individual-scripts)
- [Datasets](#datasets)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## What is in here

| Path | What it holds |
| --- | --- |
| `run_all.sh` | Interactive terminal launcher — asks five questions, then runs |
| `web_ui.sh` | Browser front end for the same thing |
| `scripts/` | The evaluation systems and the supporting tools |
| `datasets/` | The transcript CSVs (`id,label,…,text`) |
| `knowledge/` | `scam_ontology.json`, `mcq_ontology.json`, `scam_patterns.json` |
| `policies/` | Three bank policy documents — the corpus the Singh baseline retrieves from |
| `chroma_db/` | Local vector index — **not in git, you build it** (setup step 6) |
| `results/` | `*.csv` per-call predictions, `logs/run_<stamp>/` one log per baseline |

The nine baselines, in escalating order:

| Key | System | Needs an LLM |
| --- | --- | --- |
| `trivial` | length threshold + TF-IDF bag-of-words, 5-fold CV | no |
| `llm_only` | the model decides alone, no retrieval — the control | yes |
| `singh` | policy-compliance check against the `bank_policies` collection | yes |
| `webrag` | retrieval over the web-harvested `scam_patterns` KB, with a relevance gate | yes |
| `qwen_kb` | the LLM generalises each fold's *training* scams into patterns, then judges held-out calls against them | yes |
| `hybrid` | Web-RAG and Qwen-KB over one shared KB | yes |
| `ontology` | ontology-guided RAG over `knowledge/scam_ontology.json` | yes |
| `mcq` | MCQ ontology, two LLM calls per transcript | yes |
| `bert` | fine-tuned BERT classifier, k-fold CV | no (but wants the GPU) |

---

## Setup from scratch

Written for a fresh machine. On Windows, run every shell command below from
**Git Bash** — `run_all.sh` and `web_ui.sh` are bash scripts and will not run
under `cmd.exe` or PowerShell.

### 1. System prerequisites

| Thing | Why | Check |
| --- | --- | --- |
| Python 3.10–3.12 | everything | `python3 --version` |
| git | cloning | `git --version` |
| bash | `run_all.sh`, `web_ui.sh` | `bash --version` |
| curl | Ollama probes | `curl --version` |
| Ollama | serves the local LLM | `ollama --version` |
| tmux *(optional)* | detached runs that survive an SSH drop | `tmux -V` |
| NVIDIA GPU + driver *(optional)* | `qwen2.5:14b` wants ~9.5 GB VRAM; BERT wants 3–4 GB | `nvidia-smi` |

Python 3.13+ is not recommended: the pinned `torch==2.5.1` has no wheel for it.
Everything except the `bert` baseline works on CPU, just slowly.

Install Ollama:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows: download the installer from https://ollama.com/download
```

### 2. Clone the repository

```bash
git clone <your-remote-url> scam-detection
cd scam-detection
```

### 3. Create the virtual environment

The launcher scripts look for a venv at `venv/` in the project root and
activate it themselves — `venv/bin/activate` on Linux/macOS,
`venv/Scripts/activate` on Windows. Name it `venv`, not `.venv`, or they will
fall back to whatever Python is on `PATH`.

```bash
python3 -m venv venv

source venv/bin/activate        # Linux / macOS
source venv/Scripts/activate    # Windows, Git Bash

python -m pip install --upgrade pip
```

### 4. Install the Python libraries

`requirements.txt` pins the CUDA 12.1 build of PyTorch (`torch==2.5.1+cu121`),
which lives on PyTorch's own index rather than PyPI.

**With an NVIDIA GPU:**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (or Apple Silicon)** — install the plain wheels first, then the rest
without the CUDA-pinned lines:

```bash
pip install torch==2.5.1 torchvision==0.20.1
grep -v -E "^(torch|torchvision|triton|nvidia-)" requirements.txt > req-cpu.txt
pip install -r req-cpu.txt
rm req-cpu.txt
```

Verify:

```bash
python -c "import torch, chromadb, sentence_transformers, transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### 5. Pull the LLM

Start the Ollama server (it listens on `localhost:11434`; the macOS and Windows
apps start it for you):

```bash
ollama serve          # leave this running in its own terminal
```

Then pull the models. `qwen2.5:14b` is the primary model behind the reported
results; `llama3.1:8b` is the smaller alternative and is what the scripts
default to when `SCAM_MODEL` is unset.

```bash
ollama pull qwen2.5:14b     # ~9 GB download, ~9.5 GB VRAM when resident
ollama pull llama3.1:8b     # ~4.7 GB, the lighter option
ollama list                 # confirm both are there
curl -s localhost:11434/api/tags   # confirm the server answers
```

On a GPU with less than ~10 GB of VRAM, use `llama3.1:8b` and skip
`qwen2.5:14b`.

### 6. Build the vector index

`chroma_db/` is regenerable and therefore gitignored, so a fresh clone has no
index at all. Two collections have to be built, and both need the
`all-MiniLM-L6-v2` sentence-transformer, which downloads from Hugging Face on
first use (~90 MB, needs internet once):

```bash
python scripts/setup_vector_db.py     # -> bank_policies   (the singh baseline)
python scripts/build_index.py         # -> scam_patterns   (webrag + hybrid)
```

`build_index.py` reads `knowledge/scam_patterns.json`, which ships with the
repo (20 harvested patterns), so this step needs no API key. Check it:

```bash
python scripts/build_index.py --test   # rebuild, then show which hits pass the gate
```

Skip this step only if you are running `trivial`, `llm_only` or `bert`, which
retrieve nothing.

### 7. Optional — the Tavily API key

Only `harvest_patterns.py` (refreshing the web knowledge base from live search)
needs this. Everything else, including running the `webrag` baseline against
the KB that ships with the repo, works without it.

```bash
cp .env.example .env
# then edit .env:
#   TAVILY_API_KEY=tvly-...
```

Get a free key at [tavily.com](https://tavily.com). `.env` is gitignored.

### 8. Smoke test

```bash
./run_all.sh --dataset datasets/scambait_synthetic_196.csv \
             --baseline llm_only --limit 5 --model qwen2.5:14b
```

Five calls, one system. If that prints a results table, the install is good.

---

## First run

```bash
./web_ui.sh          # browser at http://localhost:8000
# or
./run_all.sh         # the same thing as five terminal questions
```

Results land in `results/logs/run_<stamp>/` (one log per baseline) and
`results/*.csv` (one row per call). Every LLM system also records the reason it
gave for each call, in a `<system>_why` column next to its verdict — the
Per-call tab shows them, and the checkbox above the table hides them again when
the width gets in the way.

---

## A. Browser UI

Everything `run_all.sh` does, as a form.

```bash
./web_ui.sh                   # http://localhost:8000, ctrl-c to stop
./web_ui.sh --tmux            # detached: survives an SSH disconnect
./web_ui.sh --port 8080       # somewhere else
./web_ui.sh --local           # this machine only (then forward the port)
./web_ui.sh --public          # plus an https link that works from anywhere
```

It binds every interface, so another machine on the same network can open it
directly — the startup banner prints the address to use:

```
scam-detection UI
  here:           http://localhost:8000
  other machines: http://192.168.1.42:8000
```

`--public` also opens an SSH reverse tunnel to localhost.run (no account
needed) and prints the public URL:

```
==> Opening a public link
  ok public    https://fa58e6c3b454ab.lhr.life
```

The tunnel is supervised — localhost.run drops it eventually, and a new one is
opened and announced in the terminal. With an SSH key on this machine the
address survives a reconnect; without one it changes each time
(`ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_lhr`, once, fixes that). See
`results/logs/tunnel.log` for the connection history.

> **That link has no password in front of it.** Anyone who opens it can start
> and stop runs on this box and read every transcript. Fine for showing a
> result to someone for ten minutes, not something to leave up. On an untrusted
> network use `--local` and tunnel in yourself instead:
>
> ```bash
> ssh -L 8000:localhost:8000 user@host
> ```

Runs started from the page are detached from the server, so closing the
browser, dropping the SSH link, or restarting the server does not stop them —
reopen the URL and the run is still there. Pick a run under **Recent runs** to
get its Output, Results, Per-call predictions and the per-baseline Step logs.

### Web-RAG knowledge base panel

Under the Run button, **Web-RAG knowledge base** does what
[section C.6](#6-refresh-the-web-rag-knowledge-base) does by hand:
`harvest_patterns.py` refreshes `knowledge/scam_patterns.json`, then
`build_index.py` re-embeds it into `chroma_db`, which is what the `webrag`
baseline retrieves from. The line above the picker is the current state —
patterns in the JSON, vectors in the index, and the date of the last harvest —
and it turns amber when the two counts disagree, i.e. the index needs
rebuilding. Five options:

| Option | What it does |
| --- | --- |
| Harvest the web, then rebuild the index | the weekly refresh |
| Harvest ignoring the cache, then rebuild | re-fetches every seed query |
| Rebuild the index only | no web calls, no LLM |
| Dry run — show the plan | stops before any network call |
| Stats only | what the KB already holds |

Harvesting needs `TAVILY_API_KEY` in `.env` and Ollama up; the last two options
need neither and can be used while a benchmark is running. The others take the
same lock a benchmark does — the index cannot be rebuilt underneath a run that
is reading it. The update streams into the same Output pane and lands in
"Recent runs" like any other run.

---

## B. The same thing in the terminal

```bash
./run_all.sh                  # asks five questions, then runs
./run_all.sh --tmux           # ... and detaches into tmux
```

Or answer up front and it asks nothing:

```bash
./run_all.sh -d 3 -b all -l 0 -m 1
./run_all.sh --dataset datasets/scambait_synthetic_196.csv \
             --baseline llm_only,mcq,bert --limit 40 --model qwen2.5:14b
```

| Flag | Values |
| --- | --- |
| `-d, --dataset` | a path, or a menu number |
| `-b, --baseline` | `all trivial llm_only singh webrag qwen_kb hybrid ontology mcq bert` — comma-separate for several; menu numbers work too (`-b 3,7,8`) |
| `-l, --limit` | `0` = whole dataset, `N` = first N calls, `id:<value>` = one row by its id column, `idx:<n>` = the n-th call of a `--limit 40` style run |
| `-m, --model` | `qwen2.5:14b` or `llama3.1:8b`, or the menu number |
| `-t, --tmux` | detach into tmux |
| `-s, --session` | name the tmux session yourself |

Each step streams its progress as it goes. While that output is flowing nothing
else is printed — it is its own proof of life. Only a silence of a minute or
more draws a heartbeat line, and only ten minutes of actual silence turns it
into a warning. `HEARTBEAT_SECS=10 ./run_all.sh` checks more often.

A long run can outlive the SSH session, so the last question offers to hand the
work to a detached tmux session. With no tmux installed it falls back to
`setsid`+`nohup`, which also survives a disconnect. Everything the run prints is
teed to a log either way.

When the baseline is `all`, BERT runs last: `qwen2.5:14b` holds ~9.5 GB of
11.4 GB VRAM and BERT fine-tuning needs 3–4 GB, so they cannot both be
resident.

---

## C. The individual scripts

Run these with the venv active, from the project root.

### 1. Seven baselines in one script

```bash
export SCAM_MODEL=qwen2.5:14b
python scripts/combined_evaluate.py --csv datasets/scambait_synthetic_196.csv --limit 20
#   --skip length,bow,singh,webrag,qwen_kb,hybrid   run just one of the seven
```

**Qwen-KB** is the learning baseline. Like BERT and bag-of-words it is
cross-validated: on each fold the model generalises that fold's *training*
scams into patterns, those patterns are indexed as a knowledge base, and the
held-out calls are judged against what is retrieved from it. Every call is
scored exactly once, while it was held out, so its row in the results table is
comparable with the rest.

```
--qwen-folds 3            fewer folds = fewer KB-building calls
--qwen-patterns 8         size of the learned KB per fold
--qwen-max-examples 40    training scams sampled per fold
--qwen-train-csv FILE     learn from a separate file instead of folds
                          (what a single-transcript run uses)
```

**Hybrid** is Web-RAG and Qwen-KB sharing ONE knowledge base: the web-harvested
patterns in `chroma_db` and the patterns learned from the training split go
into the same collection, and Web-RAG's pipeline — signal extraction, the
two-stage relevance gate, graded 0–100 confidence, threshold — runs over the
mix. Retrieval decides per call which kind of knowledge is worth showing, and
the prompt labels which is which. It runs on the same folds as Qwen-KB and
reuses the patterns Qwen-KB already learned (the fold KB is built once and
shared), so hybrid vs qwen_kb isolates the web KB and hybrid vs webrag isolates
the learned patterns. Watch the "evidence mix" line it prints: all web means
the learned patterns are not earning their place, all learned means it is
Qwen-KB with a slower pipeline. `--skip hybrid` drops it; the merged run costs
3 LLM calls per transcript, the same as `webrag`.

### 2. Ontology RAG

```bash
python scripts/evaluate_ontology.py \
  --csv datasets/scambait_synthetic_196.csv \
  --model qwen2.5:14b \
  --ontology knowledge/scam_ontology.json
```

### 3. BERT — unload qwen first

qwen holds ~9.5 GB of 11.4 GB and BERT fine-tuning needs 3–4 GB, so they cannot
both be resident.

```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"","keep_alive":0}' > /dev/null

python scripts/bert_baseline.py cv --csv datasets/scambait_synthetic_196.csv \
  --out results/bert_results_196.csv
```

`bert_baseline.py` also has a `transfer` mode (`--train-csv` / `--test-csv`)
for training on one dataset and evaluating on another, and `--cpu` to force the
CPU.

### 4. MCQ ontology (two LLM calls per transcript)

```bash
python scripts/evaluate_mcq_ontology.py \
  --csv datasets/scambait_synthetic_196.csv \
  --model qwen2.5:14b \
  --limit 20 --debug
```

### 5. Re-print the table for any past run

```bash
python scripts/collect_results.py results/logs/run_<stamp>
```

### 6. Refresh the Web-RAG knowledge base

Needs `TAVILY_API_KEY` in `.env`. The JSON is the source of truth; the vector
index is derived from it, so the rebuild follows the harvest.

```bash
python scripts/harvest_patterns.py --stats      # what is in there now
python scripts/harvest_patterns.py --dry-run    # the plan, no network
python scripts/harvest_patterns.py              # harvest (--no-cache to refetch)
python scripts/build_index.py                   # re-embed into chroma_db
python scripts/build_index.py --test            # + show which hits pass the gate
```

### 7. Web-RAG relevance gate

The KB holds scam patterns only, so a vector search hands back three of them
for EVERY call, legitimate ones included. Retrieved chunks have to prove they
match the transcript before reaching the prompt: a cheap similarity floor, then
a yes/no LLM relevance check. When nothing survives, the model is told so and
judges the transcript alone.

```
WEBRAG_MIN_SIMILARITY=0.35   cosine floor (pre-filter only — measured
                             Fraud/Normal distributions overlap almost
                             exactly, so this cannot separate on its own)
WEBRAG_LLM_GATE=1            the check that actually decides; =0 to
                             ablate it, at one fewer Ollama call each
```

```bash
python scripts/test_relevance_gate.py           # gate logic, offline, no Ollama
```

### 8. One transcript at a time

```bash
python scripts/check_one.py --csv datasets/scambait_synthetic_196.csv --idx 19
python scripts/check_one.py --csv datasets/scambait_synthetic_196.csv --idx 19 --runs 5
```

`--idx` uses the same shuffled order (seed 42) as `evaluate_mcq_ontology.py`,
so idx 19 here is the same call as idx 19 in a prior `--limit 40` run.
`--runs N` repeats the same transcript to show how much the verdict moves
between calls. Use `--raw-row` for a row by its position in the CSV as it sits
on disk.

---

## Datasets

Every CSV in `datasets/` shows up in the launcher menus automatically. All of
them carry at least `id`, `label` (`scam` / `nonscam`), `source` and `text`.

| File | Rows | scam / nonscam | What it is |
| --- | --- | --- | --- |
| `scambait_bank_422.csv` | 422 | 211 / 211 | Real calls: scam transcripts from YouTube scam-baiting videos paired with legitimate calls from the HarperValleyBank corpus. Both sides are real recorded speech. |
| `scambait_synthetic_196.csv` | 196 | 98 / 98 | Topic-matched pairs (`pair_id`, `topic`) — 97 complete pairs plus 2 unpaired rows across 99 pair ids. Real YouTube scam transcripts paired with hand-written legitimate counterparts on the same topic, then laundered through blind rewriting so style does not leak. The best-controlled set — the one where bag-of-words still scores very high after every confound is removed, which is the strongest evidence for the "detection on clean transcripts is a lexical task" argument. |
| `zhi_english_646.csv` | 646 | 323 / 323 | The Zhi et al. corpus, template-generated text, filtered of non-conversational entries and downsampled so the legitimate side matches the scam side 1:1. |
| `scamai_full_1000.csv` | 1000 | 500 / 500 | Honeypot-captured calls with rich metadata (`opening_type`, `ending_type`, `asks`, `signals`, `n_turns`, `duration_s`). |
| `everything_7013.csv` | 7013 | 2756 / 4257 | Everything pooled, multilingual and **not balanced**, with `language`, `script`, `country`, `call_type` and `scam_type` columns for slicing. |

Some transcript fields contain embedded newlines, so `wc -l` will not give you
the row count — the launcher parses them as CSV instead.

---

## Environment variables

| Variable | Default | What it does |
| --- | --- | --- |
| `SCAM_MODEL` | `llama3.1:8b` | Ollama model the LLM systems call. `run_all.sh -m` sets it for you. |
| `OLLAMA_URL` | `http://localhost:11434` | where the web UI probes for Ollama |
| `TAVILY_API_KEY` | — | read from `.env`; only `harvest_patterns.py` needs it |
| `WEBRAG_MIN_SIMILARITY` | `0.35` | cosine floor before the LLM relevance gate |
| `WEBRAG_LLM_GATE` | `1` | `0` ablates the LLM relevance check |
| `QWEN_NUM_CTX`, `QWEN_BATCH_SIZE`, `QWEN_N_RETRIEVE`, `QWEN_MIN_SIMILARITY`, `QWEN_EXAMPLE_CHARS` | see `scripts/combined_evaluate.py` | Qwen-KB tuning |
| `HEARTBEAT_SECS` | `60` | how often `run_all.sh` checks for silence |
| `SCAM_BASH` | `bash` | the bash the web UI shells out to |

---

## Troubleshooting

**`Ollama is not answering at http://localhost:11434`** — start it with
`ollama serve`, or pick a baseline that needs no LLM (`trivial`, `bert`).

**`<model> is not pulled here`** — `ollama pull qwen2.5:14b`.

**`Collection bank_policies does not exist` / `scam_patterns does not exist`** —
you skipped [setup step 6](#6-build-the-vector-index). Run
`python scripts/setup_vector_db.py` and `python scripts/build_index.py`.

**CUDA out of memory during the BERT baseline** — Ollama is still holding the
model. Unload it (`ollama stop qwen2.5:14b`, or the `keep_alive:0` curl in
[C.3](#3-bert--unload-qwen-first)), check with `nvidia-smi`, then re-run. Or
pass `--cpu`.

**`no venv/ found, using whatever python is on PATH`** — the venv is not at
`venv/` in the project root, or it is named `.venv`. See setup step 3.

**`pip install` cannot find `torch==2.5.1+cu121`** — add
`--extra-index-url https://download.pytorch.org/whl/cu121`, or use the CPU
route in setup step 4.

**A run reports many unreadable verdicts** — the model is being verbose and
running out of tokens before it reaches its answer. Try the other model, or
lower `--limit` and inspect the `<system>_why` column in the per-call CSV.

**The public link stopped working** — localhost.run rotates anonymous
subdomains on every reconnect. `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_lhr`,
once, keeps the address stable. History is in `results/logs/tunnel.log`.

---

## License

See [LICENSE](LICENSE).
