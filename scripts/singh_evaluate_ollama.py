"""
Singh et al. (2025) baseline evaluation using a local Ollama model.
Run from project root:  python scripts/singh_evaluate_ollama.py
"""

import re
import csv
import time
from pathlib import Path

import requests
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "bank_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TRANSCRIPTS_DIR = Path("./transcripts")
LABELS_FILE = Path("./labels.csv")
RESULTS_FILE = Path("./results/singh_results_ollama.csv")

OLLAMA_MODEL = os.environ.get("SCAM_MODEL", "llama3.1:8b")
OLLAMA_URL = "http://localhost:11434/api/generate"
N_CHUNKS = 3


def call_ollama(prompt, max_tokens=300):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama at localhost:11434. Start it with: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out (>180s).")


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def extract_bank_name(transcript):
    prompt = f"""You are analysing a phone call transcript. Identify which bank \
the caller claims to represent. The bank will be one of: Bank1, Bank2, Bank3.

Transcript:
{transcript}

Respond with ONLY the bank name (Bank1, Bank2, or Bank3). If no bank is clearly \
identifiable, respond with Unknown. Do not add any other text."""
    answer = call_ollama(prompt, max_tokens=10)
    for bank in ["Bank1", "Bank2", "Bank3"]:
        if bank.lower() in answer.lower():
            return bank
    return "Unknown"


def retrieve_policy(collection, transcript, bank):
    if bank == "Unknown":
        res = collection.query(query_texts=[transcript], n_results=N_CHUNKS)
    else:
        res = collection.query(query_texts=[transcript], n_results=N_CHUNKS,
                               where={"bank": bank})
    return "\n\n---\n\n".join(res["documents"][0])


def policy_check(policy_context, transcript):
    prompt = f"""You are a policy inspector. Your role is to ensure that the \
conversation complies with the policies provided.
Don't use any external source of information other than the policies provided to \
you as context.

Policies: {policy_context}

Conversation: {transcript}

Q: Does the conversation break any policies? If yes, return 'Fraud'. If no, return \
'Normal'.
Please provide only the label in the first line and start answer with "Answer:".
Provide justification in the next line starting with the word "Justification:"."""

    text = call_ollama(prompt, max_tokens=300)

    label = "Unknown"
    if re.search(r"answer:\s*fraud", text, re.IGNORECASE):
        label = "Fraud"
    elif re.search(r"answer:\s*normal", text, re.IGNORECASE):
        label = "Normal"
    elif "fraud" in text.lower():
        label = "Fraud"
    elif "normal" in text.lower():
        label = "Normal"

    m = re.search(r"justification:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return label, (m.group(1).strip() if m else text)


def load_labels():
    with open(LABELS_FILE, encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def compute_metrics(results):
    tp = fp = fn = tn = 0
    for r in results:
        p, t = r["predicted"], r["true_label"]
        if p == "Fraud" and t == "Fraud":    tp += 1
        elif p == "Fraud" and t == "Normal": fp += 1
        elif p == "Normal" and t == "Fraud": fn += 1
        elif p == "Normal" and t == "Normal": tn += 1
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    print("=" * 68)
    print(f"Singh baseline - local Ollama ({OLLAMA_MODEL})")
    print("=" * 68)

    print("\nChecking Ollama...")
    try:
        print(f"  responded: '{call_ollama('Reply with one word: ready', 5)[:40]}'")
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    collection = get_collection()
    labels = load_labels()

    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        print(f"ERROR: no transcripts in {TRANSCRIPTS_DIR}")
        return

    print(f"\nEvaluating {len(files)} calls...\n")
    results = []
    t0 = time.time()

    for i, tf in enumerate(files, 1):
        name = tf.name
        if name not in labels:
            print(f"  [{i:>3}] SKIP {name} - not in labels.csv")
            continue

        transcript = tf.read_text(encoding="utf-8")
        transcript = re.sub(r"GROUND TRUTH LABEL:.*", "", transcript,
                            flags=re.IGNORECASE).strip()

        true_label = labels[name]["label"]
        true_bank = labels[name].get("bank", "")

        try:
            bank = extract_bank_name(transcript)
            ctx = retrieve_policy(collection, transcript, bank)
            pred, just = policy_check(ctx, transcript)

            mark = "OK" if pred == true_label else "WRONG"
            bmark = "ok" if bank == true_bank else f"got {bank}/{true_bank}"
            print(f"  [{i:>3}] {name:<36} {pred:<7} (true {true_label:<6}) {mark:<5} bank {bmark}")

            results.append({"filename": name, "true_bank": true_bank,
                            "detected_bank": bank, "true_label": true_label,
                            "predicted": pred, "correct": mark,
                            "justification": just[:250]})
        except Exception as e:
            print(f"  [{i:>3}] {name}: ERROR - {e}")
            results.append({"filename": name, "true_bank": true_bank,
                            "detected_bank": "ERROR", "true_label": true_label,
                            "predicted": "ERROR", "correct": "ERROR",
                            "justification": str(e)[:250]})

    elapsed = time.time() - t0

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_bank", "detected_bank",
                                          "true_label", "predicted", "correct",
                                          "justification"])
        w.writeheader()
        w.writerows(results)

    valid = [r for r in results if r["predicted"] in ("Fraud", "Normal")]
    m = compute_metrics(valid)

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"  model:      {OLLAMA_MODEL}")
    print(f"  evaluated:  {len(valid)} of {len(results)}")
    print(f"  time:       {elapsed:.1f}s  ({elapsed/max(len(results),1):.1f}s per call)")
    print(f"\n  TP {m['TP']}   FP {m['FP']}   FN {m['FN']}   TN {m['TN']}")
    print(f"\n  Accuracy   {m['accuracy']:.4f}   ({m['accuracy']*100:.2f}%)")
    print(f"  Precision  {m['precision']:.4f}")
    print(f"  Recall     {m['recall']:.4f}")
    print(f"  F1         {m['f1']:.4f}")
    print(f"\n  Singh reported (100 synthetic calls): Accuracy 97.98%, F1 97.44%")
    print(f"  per-call results: {RESULTS_FILE}")
    print("=" * 68)


if __name__ == "__main__":
    main()
