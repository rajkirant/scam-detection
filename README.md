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
| `models/` | Fine-tuned BERT checkpoints from the BERT + MCQ page — **not in git** |
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

Everything `run_all.sh` does, as a form — plus a second page that trains a BERT
and puts the MCQ ontology to it.

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

### BERT + MCQ page

The second tab in the header. It does two things `run_all.sh` has no mode for:
fine-tune a BERT and *keep* the checkpoint, then put every question in
`knowledge/mcq_ontology.json` to that checkpoint, one transcript at a time.
`scripts/bert_mcq.py` does the work and can be used on its own
([section C.9](#9-train-a-bert-and-answer-the-mcq-ontology-with-it)).

**Train a model** (left) is `bert_baseline.py`'s training loop pointed at the
whole dataset rather than at k folds, saving to `models/<name>/` with a
`mcq_meta.json` recording the dataset, the hyperparameters and a stratified
holdout score. `models/` is gitignored — each checkpoint is a few hundred MB.
The run is detached and logged exactly like a benchmark run, streams into the
**Training output** tab, and appears under *Recent runs* on the other page. It
takes the same lock a benchmark does, so it cannot start while one is going.

**Ask** (right) puts the ontology to the selected checkpoint: it routes the
call to a branch, answers that branch's questions, and sums the values of the
chosen options into the ontology's verdict. Paste a transcript, or pull one out
of a dataset by row. Each answer shows its confidence, what it contributed to
the score, the stretch of the call it was matched against, and — under *all N
options* — what every other option scored.

Two numbers sit side by side at the top and are deliberately never combined:

| | Where it comes from |
| --- | --- |
| **MCQ score** | the sum of the option values chosen below, banded by the cut-offs in `mcq_ontology.json` |
| **prob_scam** | the binary classification head — the only thing training directly fits |

The questions themselves have no labels to train on, so they are answered by
similarity: the transcript is cut into overlapping windows, each window and
each option text is mean-pooled into a vector, and an option scores the best
cosine similarity it reaches against any window.

Four things about that matching decide whether the answers mean anything, and
each of them was, at one point, the reason they did not:

- **The encoder.** Options are matched in `all-MiniLM-L6-v2` — already this
  project's embedding model — not in the fine-tuned checkpoint's own hidden
  states. A binary scam/legitimate objective never asks an encoder to tell "a
  courier" from "a customs agency", which is the distinction every question
  here turns on. Matched in the checkpoint, every option of a question came
  back within a few hundredths of every other and the winner was decided by
  noise. `--encoder self` puts it back for comparison.
- **The key.** An option is embedded as its own text. Prefixing the question
  prompt, which is identical across that question's options, made the strings
  being compared about 90% the same characters.
- **Centring.** The mean direction of the whole option corpus is subtracted
  from both sides before the cosine. BERT sentence vectors sit in a narrow
  cone, so two unrelated phrases still score .85 against each other; removing
  that shared direction is what gives the options room to differ.
  `--no-center` disables it.
- **The gate.** A question is answered only when its best option is clear of
  the runner-up by `--min-margin` in raw cosine. A softmax over four
  near-identical scores still has to hand its probability to somebody, so
  confidence alone cannot tell a real answer from a coin flip. Below the
  margin the question abstains and contributes **nothing** — not knowing
  whether the caller asked for anything is not evidence that they asked for
  nothing.

Transcripts are repaired first: the tone tags (`[curious]`, `[long pause]`)
come out, redacted entities keep their word (`[CARD]` → "card number"), and
the apostrophes ASR dropped go back in, because "i m" and "don t" are not
words any encoder was trained on. `--raw` disables it. Note that the tone tags
also reach `bert_baseline.py`, which is worth knowing about: in
`scamai_full_1000.csv`, `[satisfied]` appears on 54.8% of legitimate calls and
31.0% of scams, so a classifier reading them can score off the labelling
convention rather than off the call.

What training changes is therefore `prob_scam`, and the MCQ answers are
zero-shot. The **How it answers** tab says this on the page, including where
the method is weak (similarity reads subject matter, not negation).

Answering happens on the CPU unless *answer on the GPU* is ticked, so the page
stays usable while a benchmark run has the VRAM. The checkpoint is held in
memory between questions and let go after ten idle minutes, or when you press
*unload it*.

### LLM judge page

The third tab, and the simplest thing in the project. Paste a transcript (or
load a row from a dataset with the picker at the top), press **Ask the
model**, and the local LLM says whether it is a scam and gives its reason. No
retrieval, no ontology, no fine-tuned anything.

It is the `llm_only` control from the Benchmark page asked one call at a time,
and it shares the benchmark's prompt and verdict parser — so the answer on
this page is the answer that would have been recorded there for that call.
`scripts/llm_judge.py` does the work
([section C.10](#10-ask-the-llm-about-one-call)).

The model menu lists whatever `ollama list` would: the model in `SCAM_MODEL`
is put first, so the page opens on the one the rest of the project uses.
Ollama holds the model, not the server, so there is nothing to load or unload
here.

**Standing instructions.** The second box under the transcript is where a
correction goes when the model gets one wrong — *"a bank asking for the last
four digits of a card is normal here"* — and it is sent with every question
from then on, fenced off from the transcript so the model reads it as a rule
rather than as something the caller said. The fencing is the point: a
transcript is full of people telling each other what to do, and a model that
cannot tell an instruction from the call it is reading will start taking
orders from the caller.

It is not training. Nothing is stored and nothing is learned: the text goes
into the prompt, every time, so the box *is* the model's whole memory and
closing the page empties it. A verdict reached under instructions comes back
flagged `guided` and is called out on the page, because at that point it is no
longer the `llm_only` control — with the box empty the prompt is byte for byte
what `combined_evaluate.py` sends, and with it filled it is not.

Two things the page shows that the benchmark does not:

- **The prompt size, before you ask.** Ollama drops the *front* of a prompt
  that overflows the context window — the instructions go first — and what
  comes back then reads like a bad model rather than a bad setting. The word
  and token count sits under the box and turns red when a transcript is over
  the window.
- **Unreadable answers, as unreadable.** When neither the first reply nor the
  one-word retry can be parsed, the benchmark has to score something and
  settles for Normal. Here it says so, because on a single call "Normal"
  should not sometimes mean "the model did not answer".

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

### 9. Train a BERT and answer the MCQ ontology with it

The command line behind the **BERT + MCQ** page
([section A](#bert--mcq-page)). Unlike `bert_baseline.py`, which trains per
fold and throws each model away, this keeps the checkpoint.

```bash
# fine-tune on a whole dataset and keep it in models/zhi-bert/
python scripts/bert_mcq.py train --csv datasets/zhi_english_646.csv \
    --name zhi-bert --epochs 4 --holdout 0.2

# put every question in knowledge/mcq_ontology.json to that checkpoint
python scripts/bert_mcq.py answer --name zhi-bert --text "Hello, this is..."
python scripts/bert_mcq.py answer --name zhi-bert \
    --csv datasets/zhi_english_646.csv --idx 3 --json

# the numbers behind every pick, when the answers look arbitrary
python scripts/bert_mcq.py diagnose --name zhi-bert \
    --csv datasets/zhi_english_646.csv --idx 3 --control

python scripts/bert_mcq.py models              # what is in models/
python scripts/bert_mcq.py delete --name zhi-bert
```

`diagnose` prints the raw cosine behind every option, the margin between the
best two, and the spread across each question — read those before trusting a
verdict. `--control` answers a word-shuffled copy of the same call as well.
Shuffling keeps every word and destroys every phrase, so any answer that
survives it was reading the vocabulary rather than the call; a low agreement
count is the good result. The knobs worth moving are `--min-margin` (raise it
until only the questions the call really answers come through), `--encoder`
(`self` matches in the checkpoint, which is the old behaviour) and
`--no-center`.

`answer` prints the branch it routed to, the option it chose for each question
with a confidence, the summed score and the verdict, and `prob_scam` from the
trained classification head beside it for contrast. `--json` gives the whole
thing including every option's score and the transcript window each answer was
matched against — the shape the web UI renders.

Answering is on the CPU unless `--gpu` is passed, so it does not compete with a
benchmark for VRAM. `--branch <id>` forces a branch instead of routing to one,
`--cutoff` moves the scam threshold (RQ2), and `--window` / `--stride` /
`--min-confidence` control how options are matched. There is also a `serve`
mode — one JSON request per line on stdin — which is how the web UI keeps a
checkpoint loaded between questions.

### 10. Ask the LLM about one call

The command line behind the **LLM judge** page
([section A](#llm-judge-page)): the `llm_only` baseline pointed at a single
transcript instead of at a dataset.

```bash
python scripts/llm_judge.py --text "Hello, this is your bank's fraud team..."
python scripts/llm_judge.py --csv datasets/scambait_bank_422.csv --idx 3
python scripts/llm_judge.py --text "..." --model qwen2.5:14b --json

# standing instructions, the same box the page has
python scripts/llm_judge.py --text "..." \
    --guidance "Confirming the last four digits of a card is normal here."
python scripts/llm_judge.py --text "..." --guidance-file house_rules.txt

python scripts/llm_judge.py models        # what ollama has pulled
```

It prints the verdict, the model's own reason, and a warning when the prompt
is bigger than `--num-ctx` — Ollama truncates an overlong prompt from the
front, so the instructions are the first thing lost and the reply that comes
back is answering a headless transcript. `--max-tokens` raises the reply
budget when an answer comes back cut off. Unlike the benchmark, a reply that
cannot be parsed is reported as unreadable rather than scored Normal.

The prompt and the verdict parser are imported from `combined_evaluate.py`
rather than copied, so a verdict here is the verdict the benchmark would have
recorded for that call. It talks to Ollama over plain HTTP with nothing but
the standard library, so it runs outside the venv as well as in it.
`OLLAMA_URL` (or `OLLAMA_HOST`) points it at another machine and `SCAM_MODEL`
sets the default model.

---

## The context window

Ollama's context window defaults to **2048 tokens** unless the model's
Modelfile raises it, and a prompt longer than the window is truncated **from
the front** — where the instructions are. It does not error, does not warn and
does not come back empty. It comes back as a fluent, plausible verdict on the
first ~1,500 words of the call, with the question that was asked about it cut
off.

How much that matters depends entirely on the dataset:

| Dataset | Median tokens | p90 | Max | Over 2048 |
| --- | --- | --- | --- | --- |
| `scamai_hard_subset.csv` | ~5,290 | ~11,400 | ~56,500 | **96%** |
| `scamai_full_1000.csv` | ~4,350 | ~8,000 | ~8,500 | **86%** |
| `everything_7013.csv` | ~435 | ~3,200 | ~8,500 | **27%** |
| `scambait_bank_422.csv` | ~180 | ~300 | ~430 | 0% |
| `scambait_synthetic_196.csv` | ~140 | ~190 | ~250 | 0% |
| `zhi_english_646.csv` | ~57 | ~96 | ~360 | 0% |

So a number measured on the bank, synthetic or Zhi sets is unaffected. A
number measured on the two `scamai` sets before this was fixed was mostly
measured on truncated calls — on `scamai_hard_subset.csv`, only **3.6%** of
the calls fit the 2048 default at all.

### Checking it

```bash
# 1. offline, no Ollama and no GPU: every client sends a big enough window
python scripts/test_ollama_ctx.py

# 2. against your real Ollama - compare the estimate to what it actually read
python scripts/llm_judge.py --csv datasets/scamai_hard_subset.csv --idx 0
python scripts/llm_judge.py --csv datasets/scamai_hard_subset.csv --idx 0 \
    --num-ctx 2048          # what every baseline used to do
```

The last line of each run is the evidence, because Ollama reports
`prompt_eval_count` — how many prompt tokens it actually read, which is the
only number here that is not an estimate. Measured on `qwen2.5:14b` against
row 0 of the hard subset (10,873 words, ~15,293 tokens):

```
window  8192 → ollama read  4098  (27%)     estimate ~15,293
window  2048 → ollama read  1026  ( 7%)     estimate ~15,293
window 32768 → ollama read 15293  (100%)    nothing lost
```

### Missing the window costs half of it, not the overflow

Look at those first two numbers: `4098` is `8192/2 + 2`, and `1026` is
`2048/2 + 2`. When a prompt overflows, Ollama does not trim it to fit — it
keeps **about half the window** and discards the rest, from the front. A
window merely *close* to the prompt size is therefore no use at all; it has to
exceed it.

What that looks like in practice, on row 0 of `scamai_hard_subset.csv`, which
is labelled **scam**:

```
kept (the last 27%):  "No bother, love. We got there in the end, didn't we?
                       ... You have a good day now. ... Bye bye."
discarded (the front): "responding to the health insurance application
                       ... submitted for mister [NAME] ..."
```

At both 8192 and 2048 the model answered **Normal** — "a legitimate benefits
coordinator helping to finalize the application" — which is a fair reading of
the goodbyes it was given, and a false negative on the call. That is the
failure this whole section exists to make impossible to miss: fluent,
confident, and wrong, with no error anywhere.

A third run is the one that matters for the thesis: the same rows through the
benchmark at both windows. If the verdicts move, every earlier number on that
dataset was measured on truncated calls.

```bash
SCAM_NUM_CTX=2048  ./run_all.sh -d datasets/scamai_hard_subset.csv -b llm_only -l 20
SCAM_NUM_CTX=16384 ./run_all.sh -d datasets/scamai_hard_subset.csv -b llm_only -l 20
```

### Picking `SCAM_NUM_CTX`

For `scamai_hard_subset.csv`, the share of calls that fit the window:

| `SCAM_NUM_CTX` | Calls that fit | |
| --- | --- | --- |
| 2048 | 3.6% | Ollama's default — what it used to be |
| 8192 | 76.2% | the default here |
| 16384 | 97.4% | |
| 32768 | 99.7% | `qwen2.5`'s own maximum |

The single longest call in that set is ~56,500 tokens and fits no `qwen2.5`
window at all. Those calls will warn on every run, which is the point — they
need a decision (drop them, split them, or summarise them first), not a
silent truncation.

Because a prompt that misses the window loses half of it rather than the
overflow, "fits" above means *strictly* fits — there is no partial credit. The
p90 of that set is ~11,400 tokens, so 16384 is the smallest window that covers
most of it, and `qwen2.5`'s 32768 covers all but one call. Both cost VRAM on
top of the ~9.5 GB the weights already take on an 11.4 GB card, so check
`nvidia-smi` and expect to unload between a long LLM run and a BERT run.

Every Ollama client in the project now routes its window through
`scripts/ollama_ctx.py`, which sizes it to the prompt, floors it at Ollama's
own 2048, caps it at `SCAM_NUM_CTX` (default 8192) and **says so on stderr**
when even the cap is not enough. `combined_evaluate.py` prints a summary of
any overflows directly under the results table, because an overflow
invalidates the rows above it.

It is a ceiling rather than a fixed size because the window costs VRAM:
`qwen2.5:14b` is already ~9.5 GB of an 11.4 GB card, so a short call asks for
a small window and only a long one pays for a big one. Check `nvidia-smi` has
the headroom before raising it:

```bash
SCAM_NUM_CTX=16384 ./run_all.sh -d datasets/scamai_hard_subset.csv -b llm_only
```

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
| `OLLAMA_URL` | `http://localhost:11434` | where the web UI probes for Ollama, and where the LLM judge page sends its transcripts (`OLLAMA_HOST` is read as a fallback, so ollama's own variable works too) |
| `TAVILY_API_KEY` | — | read from `.env`; only `harvest_patterns.py` needs it |
| `WEBRAG_MIN_SIMILARITY` | `0.35` | cosine floor before the LLM relevance gate |
| `WEBRAG_LLM_GATE` | `1` | `0` ablates the LLM relevance check |
| `SCAM_NUM_CTX` | `8192` | ceiling on the Ollama context window every LLM system asks for — see [the context window](#the-context-window) |
| `QWEN_NUM_CTX`, `QWEN_BATCH_SIZE`, `QWEN_N_RETRIEVE`, `QWEN_MIN_SIMILARITY`, `QWEN_EXAMPLE_CHARS` | see `scripts/combined_evaluate.py` | Qwen-KB tuning |
| `HEARTBEAT_SECS` | `60` | how often `run_all.sh` checks for silence |
| `SCAM_BASH` | `bash` | the bash the web UI shells out to |

---

## Troubleshooting

**`Ollama is not answering at http://localhost:11434`** — start it with
`ollama serve`, or pick a baseline that needs no LLM (`trivial`, `bert`).

**`<model> is not pulled here`** — `ollama pull qwen2.5:14b`.

**`WARNING [llm_only] prompt is about N tokens, window is M`** — the call did
not fit the context window, so Ollama cut the *front* off it and the model
answered without its instructions. Raise `SCAM_NUM_CTX` and re-run; any result
printed under that warning was measured on truncated calls. See
[the context window](#the-context-window) for why this is a ceiling rather
than a fixed size.

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
