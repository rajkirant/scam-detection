"""
Build the vector database from the three bank policy documents.
Run from project root:  python scripts/setup_vector_db.py
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

POLICIES_DIR = Path("./policies")
CHROMA_DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "bank_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_policy_document(text, bank_name):
    """Split a policy into chunks, one per '## N. Section' heading."""
    chunks = []
    sections = re.split(r'(?=^## \d+\.)', text, flags=re.MULTILINE)
    for section in sections:
        section = section.strip()
        if not section or len(section) < 100:
            continue
        header = re.match(r'## (\d+\.\s*[^\n]+)', section)
        title = header.group(1) if header else "Unknown Section"
        chunks.append({"text": section, "bank": bank_name, "section": title})
    return chunks


def load_all_policies():
    all_chunks = []
    for policy_file in sorted(POLICIES_DIR.glob("*.md")):
        bank_name = policy_file.stem.replace("_policy", "").capitalize()
        text = policy_file.read_text(encoding="utf-8")
        chunks = chunk_policy_document(text, bank_name)
        all_chunks.extend(chunks)
        print(f"  {policy_file.name}: {len(chunks)} chunks  (bank='{bank_name}')")
    return all_chunks


def setup_vector_database(chunks):
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"  deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"bank": c["bank"], "section": c["section"]} for c in chunks],
        ids=[f"{c['bank']}_{i}" for i, c in enumerate(chunks)],
    )
    print(f"\n  added {len(chunks)} chunks to '{COLLECTION_NAME}'")
    print(f"  saved to {CHROMA_DB_DIR.absolute()}")


def test_query():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    q = "employee asked for full credit card number and CVV code"
    results = collection.query(query_texts=[q], n_results=3, where={"bank": "Bank1"})

    print(f"\n  test query: '{q}'  (filtered to Bank1)")
    for i in range(len(results["ids"][0])):
        section = results["metadatas"][0][i]["section"]
        dist = results["distances"][0][i]
        print(f"    {i+1}. {section}  (distance {dist:.4f})")


def main():
    print("=" * 60)
    print("Building vector database")
    print("=" * 60)

    if not POLICIES_DIR.exists():
        print(f"ERROR: {POLICIES_DIR} not found. Run from ~/scam-detection")
        return

    print("\nChunking policies...")
    chunks = load_all_policies()
    if not chunks:
        print("ERROR: no chunks produced. Check the .md files in policies/")
        return
    print(f"  total: {len(chunks)} chunks")

    print("\nEmbedding and storing (first run downloads ~90MB model)...")
    setup_vector_database(chunks)

    print("\nTesting retrieval...")
    test_query()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
