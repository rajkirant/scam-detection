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
| `models/` | Fine-tuned BERT checkpoints, bag-of-words models, length thresholds and fitted LLM prompts, from the four model pages — **not in git** |
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

Everything `run_all.sh` does, as a form — plus four pages that put one call to
one model: a fine-tuned BERT, a bag of words, a length threshold, or the local
LLM. Each of the four fits on a dataset, keeps what it fitted in `models/`,
and scores itself on held-out calls.

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

### BERT page

The second tab in the header. It does two things `run_all.sh` has no mode for:
fine-tune a BERT and *keep* the checkpoint, then put a transcript to that
checkpoint and get back the probability that the call is a scam.
`scripts/bert_classify.py` does the work and can be used on its own
([section C.9](#9-train-a-bert-and-classify-one-call-with-it)).

**Train a model** (left) is `bert_baseline.py`'s training loop pointed at the
whole dataset rather than at k folds, saving to `models/<name>/` with a
`meta.json` recording the dataset, the hyperparameters and a stratified
holdout score. `models/` is gitignored — each checkpoint is a few hundred MB.
The run is detached and logged exactly like a benchmark run, streams into the
**Training output** tab, and appears under *Recent runs* on the other page. It
takes the same lock a benchmark does, so it cannot start while one is going.

**Classify** (right) puts one call to the selected checkpoint. Paste a
transcript, or pull one out of a dataset by row.

The number it gives is `prob_scam` — the binary head, which is the one thing
training directly fits. Nothing on this page is inferred, weighted or scored
on top of it.

**Why the call is read in windows.** BERT takes 512 tokens at most and these
checkpoints are trained at 256 — about 180 words. Calls in
`scamai_hard_307_ordered.csv` run past ten thousand. Handing the whole
transcript to the tokenizer scores its opening and silently drops the rest,
which on a long call means judging it by the hellos — the same failure the
Ollama context window had ([the context window](#the-context-window)). So the
transcript is cut into overlapping windows of the size training used, every
window is scored, and the call takes either the strongest window or the
average:

| | |
| --- | --- |
| **strongest** | a scam signal anywhere is a scam signal — suits calls that are mostly small talk around one telling exchange |
| **average** | steadier, but dilutes that exchange in a long friendly call |

Both are shown whichever you pick, next to **first window** — what a single
truncated read would have said. When the three disagree, the disagreement is
the finding, and the page says so. Under *every window* is the score of each
one, so you can see where in the call the signal is.

**Strip the tone tags** (under *How the call is read*) takes `[curious]`,
`[long pause]` and the rest out before scoring. It is off by default and it is
an experiment, not a correction: training read those tags, so a checkpoint
asked about text without them is being asked about text of a kind it never
saw. Worth one run each way — in `scamai_full_1000.csv`, `[satisfied]` sits on
54.8% of legitimate calls and 31.0% of scams, so a score that moves a lot here
was partly reading the annotation style rather than the call.

**The holdout number beside a trained model is not a result.** It is a
stratified slice of that model's own training file. On
`scambait_bank_422.csv` a checkpoint reaches 100%, because the scam side is
YouTube scam-baiting and the legitimate side is the HarperValleyBank corpus —
separable on recording pipeline alone. Compare against the bag-of-words
baseline on the Benchmark page before believing any of it.

Scoring happens on the CPU unless *score on the GPU* is ticked, so the page
stays usable while a benchmark run has the VRAM. The checkpoint is held in
memory between calls and let go after ten idle minutes, or when you press
*unload it*.

### Bag of words page

The third tab, and the control the other two are measured against. Same shape
as the BERT page on purpose: fit a model on a dataset, keep it, put a
transcript to it, get a probability. `scripts/bow_classify.py` does the work
([section C.11](#11-fit-a-bag-of-words-model-and-classify-one-call)).

Under it is TF-IDF over word unigrams and bigrams into a logistic regression —
the same vectoriser and classifier the `bow` baseline cross-validates on the
Benchmark page. No embeddings, no attention, no GPU. It fits in about a
second.

Two differences from the BERT page matter when reading them side by side:

- **It reads the whole call.** BERT takes 512 tokens, so that page scores a
  long transcript in windows. TF-IDF has no length limit — every word is
  counted. On a long call the two pages are not being asked the same
  question, and this is the one that saw all of it.
- **It can be read back exactly.** A linear model over TF-IDF decomposes: the
  score is the intercept plus, for every term in the call, its TF-IDF weight
  times its coefficient. The terms under a verdict are those products, largest
  first, and they sum to the score shown. That is arithmetic, not a story told
  about the model afterwards — and it is what BERT cannot give you.

**Read the terms, not the accuracy.** On `scambait_bank_422.csv` this reaches
**100% held out in 0.06 seconds** — the same score a fine-tuned BERT takes 63
seconds on a GPU to reach. The terms doing the work on a real scam call from
that set are:

| Towards scam | Towards legitimate |
| --- | --- |
| `computer` +0.110 | `is` −0.047 |
| `so` +0.060 | `what` −0.028 |
| `tell` +0.060 | `can` −0.028 |
| `me` +0.059 | `hello` −0.023 |
| `yes` +0.047 | `company` −0.019 |

Those are function words and discourse markers. The two halves of that dataset
come from different recording pipelines — YouTube scam-baiting and the
HarperValleyBank corpus — and this is what separating on transcription style
looks like from the inside. When this page scores near BERT, neither number is
evidence about understanding scams, and this page shows you why in a way the
other cannot.

### Length only page

The fourth tab, and the floor. Count the words in the transcript, compare the
count to one number, call it. Nothing in the call is read — not a word, not an
entity, not a tone tag. `scripts/length_classify.py` does the work
([section C.12](#12-fit-a-length-threshold-and-classify-one-call)).

It is the `length` baseline from the Benchmark page, which is
`trivial_length` in `combined_evaluate.py`: *Fraud if the call is longer than
45 words*. It has a page of its own because **whatever BERT or the bag of
words beats this by is the whole of what those models are worth on that
dataset**, and on several of the datasets here the gap is smaller than the
write-up would like.

Three things the page is built to show:

- **"Fit" is a sweep, not learning.** There is one parameter, chosen by
  trying every threshold the fitting calls suggest and keeping the
  best-scoring one. That overfits a single number to a single dataset without
  complaint, so the fitting output lists the runner-up thresholds and says out
  loud when the curve is flat — on `everything_7013.csv`, 209 of 3,896
  thresholds come within a point of the winner, which means the exact number
  is meaningless. Tick **pin the threshold** to skip fitting entirely and use
  the benchmark's own 45.
- **The direction is not a given.** "Scams are longer" is an assumption about
  a corpus, not a fact about scams, and the sweep tests both ways round. See
  the table below — on two of these datasets it points the other way.
- **No probability is invented.** A threshold cannot say how confident it is.
  In place of one the page shows a fact about the fitting set: the share of
  fitting calls on this side of the line that really were scams. Move the
  threshold on the classify form and that share moves with it. When the share
  contradicts the verdict, the page says so instead of dressing the number up.

What a fitted threshold actually gets, holdout, alongside what always
answering the same thing gets:

| Dataset | Fitted rule | Holdout acc | Best constant answer |
| --- | --- | --- | --- |
| `zhi_english_646.csv` | longer than 34 words | **85.4%** | 50.0% |
| `scambait_bank_422.csv` | longer than 130 words | **71.8%** | 50.6% |
| `everything_7013.csv` | **shorter** than 726 words | **69.8%** | 60.7% |
| `scamai_full_1000.csv` | **shorter** than 3,574 words | **64.5%** | 50.0% |

Two results in that table are worth carrying into the write-up:

1. On `scamai_full_1000.csv` and `everything_7013.csv` the scam calls are the
   **shorter** ones. `trivial_length`'s rule points the wrong way on both, so
   the number the Benchmark page reports for `length` there is worse than the
   same threshold read backwards.
2. On `scambait_bank_422.csv`, `trivial_length`'s threshold of 45 calls
   **every single call a scam** — the shortest call in that set is longer than
   45 words — so its 49.4% there is the always-scam rate and nothing else. The
   fitting output prints a note when this happens.

And for the comparison the page exists to make: on `scambait_bank_422.csv`,
counting the words gets 71.8% from a model that is one integer. BERT and the
bag of words both get 100% on the same split. The distance between 50% and 72%
is what counting bought; the distance between 72% and 100% is what reading
bought.

### LLM judge page

The last tab, and the simplest thing in the project. Paste a transcript (or
load a row from a dataset with the picker at the top), press **Ask the
model**, and the local LLM says whether it is a scam and gives its reason. No
retrieval, no ontology, no fine-tuned anything.

With **None** picked under *Fitted prompts* it is exactly that and nothing
else; with a fitted prompt picked it is that plus worked examples out of a
dataset — see *Fitting a prompt* below.

It is the `llm_only` control from the Benchmark page asked one call at a time,
and it shares the benchmark's prompt and verdict parser — so the answer on
this page is the answer that would have been recorded there for that call.
`scripts/llm_judge.py` does the work
([section C.10](#10-ask-the-llm-about-one-call)).

The model menu lists whatever `ollama list` would: the model in `SCAM_MODEL`
is put first, so the page opens on the one the rest of the project uses.
Ollama holds the model, not the server, so there is nothing to load or unload
here.

**Fitting a prompt.** The left-hand panel fits a prompt on a dataset, keeps
it in `models/<name>/`, and lets you pick it from a list — the same shape as
the other three pages. `scripts/llm_fit.py` does the work
([section C.13](#13-fit-a-prompt-for-the-llm)).

**It does not fine-tune anything.** The weights Ollama is holding do not move,
and nothing in this project can move them. A fitted prompt is text prepended
to every question, made of up to three parts:

| Part | What it is |
| --- | --- |
| worked examples | *k* calls from the fitting split with their real answers attached, balanced between the classes and excerpted — a median call in some of these sets is 5,000 tokens and four whole ones would crowd out the call being judged |
| rubric | what the model itself writes when shown those examples and asked what separates the two classes. One extra call at fit time, then a fixed piece of text |
| standing instructions | the box below, carried into the profile so a correction survives a page reload |

That is in-context learning, and it is the only kind of training a frozen
local model can be given from a web page. The other three pages train a model;
this one writes a better question. The page will not call those the same
thing.

**The fit scores everything twice.** A longer prompt always *feels* like an
improvement and often is not, so fitting runs the held-out calls through the
fitted prompt *and* through the bare one, in the same order, and reports the
difference. If the fitted prompt did not beat the control it has cost context
window and bought nothing, and the run says so in those words rather than
reporting a number that looks fine on its own. That costs two LLM calls per
held-out call, so the form shows the bill before you press Fit.

**Where the parts sit, and why it matters.** Ollama truncates an overlong
prompt from the *front*, so the order is worst-to-best: worked examples,
rubric, transcript, then the rules and answer format. A fitted prompt that
overflows loses its examples first and decays into the control, rather than
into a headless wall of transcript with no question attached. The fit counts
how many held-out prompts this happened to, and the answer card says when it
happened to the call on screen.

**Standing instructions.** The second box under the transcript is where a
correction goes when the model gets one wrong — *"a bank asking for the last
four digits of a card is normal here"* — and it is sent with every question
from then on, fenced off from the transcript so the model reads it as a rule
rather than as something the caller said. The fencing is the point: a
transcript is full of people telling each other what to do, and a model that
cannot tell an instruction from the call it is reading will start taking
orders from the caller.

On its own it is not stored: the text goes into the prompt, every time, so the
box *is* the model's whole memory and closing the page empties it. Tick *carry
the standing instructions in* when fitting to make them part of a saved prompt
instead. A verdict reached under instructions or a fitted prompt comes back
flagged and is called out on the page, because at that point it is no
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

### 9. Train a BERT and classify one call with it

The command line behind the **BERT** page ([section A](#bert-page)). Unlike
`bert_baseline.py`, which trains per fold and throws each model away, this
keeps the checkpoint.

```bash
# fine-tune on a whole dataset and keep it in models/zhi-bert/
python scripts/bert_classify.py train --csv datasets/zhi_english_646.csv \
    --name zhi-bert --epochs 4 --holdout 0.2

# scam or not, for one call
python scripts/bert_classify.py classify --name zhi-bert --text "Hello, this is..."
python scripts/bert_classify.py classify --name zhi-bert \
    --csv datasets/scambait_bank_422.csv --idx 3 --json

python scripts/bert_classify.py models              # what is in models/
python scripts/bert_classify.py delete --name zhi-bert
```

`classify` prints the verdict and `prob_scam`, then the three numbers that
matter on a long call: the strongest window, the average, and what the first
window alone would have said. When the first disagrees with the strongest by
a wide margin it says so — that gap is the difference between reading the call
and reading its opening.

`--aggregate mean` averages the windows instead of taking the strongest,
`--threshold` overrides the checkpoint's own cut-off, `--window` / `--stride`
change how the call is cut up, and `--strip-tags` removes the `[curious]`
annotations first (an experiment — training read them). There is also a
`serve` mode — one JSON request per line on stdin — which is how the web UI
keeps a checkpoint loaded between calls.

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

### 11. Fit a bag-of-words model and classify one call

The command line behind the **Bag of words** page
([section A](#bag-of-words-page)).

```bash
# fit on a whole dataset and keep it in models/bank-bow/
python scripts/bow_classify.py train --csv datasets/scambait_bank_422.csv \
    --name bank-bow

# scam or not, and the terms that decided it
python scripts/bow_classify.py classify --name bank-bow \
    --csv datasets/scambait_bank_422.csv --idx 0

python scripts/bow_classify.py models               # what is fitted
python scripts/bow_classify.py delete --name bank-bow
```

`classify` prints the verdict, `prob_scam`, and the decomposition: the score,
the intercept, and the terms pushing each way with what each contributed.
Those contributions are exact — TF-IDF weight times coefficient — and they sum
to the score.

`--ngram-max 1` drops bigrams, `--min-df` changes how rare a term may be,
`--top` sets how many terms are listed each way, `--threshold` overrides the
model's cut-off, and `--strip-tags` removes the `[curious]` annotations (at
fit time, at classify time, or both — they are separate flags on the two
subcommands). There is also a `serve` mode on the same one-JSON-line protocol
the BERT page uses.

### 12. Fit a length threshold and classify one call

The command line behind the **Length only** page
([section A](#length-only-page)).

```bash
# sweep for a threshold and keep it in models/bank-length/
python scripts/length_classify.py fit --csv datasets/scambait_bank_422.csv \
    --name bank-length

# no sweep at all - combined_evaluate.py's own rule
python scripts/length_classify.py fit --csv datasets/scambait_bank_422.csv \
    --name bank-45 --threshold 45 --direction longer

# scam or not, and the whole of the reasoning
python scripts/length_classify.py classify --name bank-length \
    --csv datasets/scambait_bank_422.csv --idx 0

python scripts/length_classify.py models            # what is fitted
python scripts/length_classify.py delete --name bank-length
```

`fit` prints the two classes' length distributions, the chosen rule, the five
runner-up thresholds, a warning when the curve is flat, and the holdout score
next to what always-scam and never-scam get on the same split — so a rule that
beats nothing is visible as one.

`classify` prints the verdict, the word count, how far past the line it is,
and the share of fitting calls on that side of the line that were scams. It
never prints a probability, because a threshold does not have one.

`--metric acc` makes the sweep maximise accuracy rather than F1,
`--threshold` pins the line instead of fitting it, `--direction` says which
side is the scam when it is pinned (a sweep decides this for itself),
`--holdout 0` uses every call to fit, and `--strip-tags` removes the
`[curious]` annotations before counting. There is also a `serve` mode on the
same one-JSON-line protocol the other two pages use, and `train` works as an
alias for `fit`.

The offline checks for it need no dataset and no venv:

```bash
python3 scripts/test_length_classify.py
```

### 13. Fit a prompt for the LLM

The command line behind the **Fit a prompt** panel on the LLM judge page
([section A](#llm-judge-page)). **It does not fine-tune anything** — see that
section for what it does instead.

```bash
# worked examples + a rubric the model writes, measured against the control
python scripts/llm_fit.py fit --csv datasets/zhi_english_646.csv \
    --name zhi-prompt --shots 4 --rubric --holdout-calls 20

# what ended up in it
python scripts/llm_fit.py show --name zhi-prompt

# use it
python scripts/llm_judge.py --profile zhi-prompt --text "Hello, this is..."

python scripts/llm_fit.py models                    # what is fitted
python scripts/llm_fit.py delete --name zhi-prompt
```

`fit` prints which calls it picked as examples, the rubric the model wrote,
how many tokens the fitted prompt adds to every call, then the holdout scored
twice — once fitted, once bare — with the difference between them. **The
difference is the number that matters.** A fitted prompt that does not beat
the control has cost you context window and bought nothing, and this is the
only way to find that out.

Each held-out call costs two LLM calls, plus one for the rubric, so
`--holdout-calls 20 --rubric` is 41 calls to the model. Budget minutes, not
seconds.

`--shots 0` fits nothing but a rubric and your instructions, `--shot-words`
trades window for detail in each example, `--guidance-file` carries standing
instructions into the profile, `--holdout-calls 0` skips the measurement
entirely (and then you will not know whether it helped), and `--seed` changes
which calls are drawn as examples. `train` works as an alias for `fit`.

Standard library only, like `llm_judge.py` — no venv needed.

The offline checks run the whole fitting path against a fake Ollama on a spare
port, so they need neither a GPU nor a model pulled. What they mostly guard is
the order of the prompt — get that backwards and an overlong fitted prompt
stops being a prompt at all, silently, while still returning a fluent verdict:

```bash
python3 scripts/test_llm_fit.py
```

Models land in `models/` beside the BERT checkpoints without colliding: a
bag-of-words model is a directory with `bow.joblib` in it, a BERT checkpoint
is one with `config.json`, and each listing skips the other kind.

---

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
the goodbyes it was given, and a false negative on the call. At 32768, with
the whole call in front of it, the same model on the same transcript answered
**Fraud**: *"repeatedly asked for sensitive information such as credit card
details and social security numbers."*

Same model, same prompt, same row. Only the window changed. That is the
failure this section exists to make impossible to miss: fluent, confident,
and wrong, with no error anywhere.

One detail from that run worth keeping: Ollama read **15,933** tokens where
this project estimated 15,293 — the heuristic runs a few percent low against a
real tokenizer. Since missing the window costs half of it, `fit_num_ctx` sizes
windows with 15% headroom over the estimate rather than trusting it.

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
