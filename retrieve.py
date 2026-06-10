"""
Stage 3 + 4: embedding + vector store + retrieval.

Loads the chunks produced by build_chunks.py (chunks.json), embeds them with
all-MiniLM-L6-v2, and stores them in a persistent ChromaDB collection along with
the source metadata needed for attribution. Exposes retrieve() for the
generation stage (Milestone 5).

Distance space is cosine (hnsw:space=cosine), so scores run 0 (identical) to 2
(opposite); the Milestone 4 checkpoint wants top results below ~0.5.

Usage:
    python retrieve.py --rebuild      # (re)embed chunks.json into the store
    python retrieve.py "your question" # ad-hoc query against the store
    python retrieve.py                 # run the evaluation-plan smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

CHUNKS_PATH = "chunks.json"
PERSIST_DIR = "chroma_db"
COLLECTION = "guides"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# A few evaluation-plan questions for the smoke test (see Planning.md).
EVAL_QUESTIONS = [
    "What order should I list my work experience in on a resume?",
    "How should I describe my accomplishments in resume bullet points?",
    "What are the main parts of a cover letter?",
]

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=PERSIST_DIR)


def build_index(chunks_path: str = CHUNKS_PATH) -> int:
    """Embed every chunk in chunks.json and (re)load it into ChromaDB."""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    client = get_client()

    # Drop any existing collection so a rebuild is clean, not additive.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "filename": c["filename"],
            "doc_type": c["doc_type"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = get_model().encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    print(f"Stored {collection.count()} chunks in '{COLLECTION}' ({PERSIST_DIR})")
    return collection.count()


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks for a query, each with source + distance."""
    collection = get_client().get_collection(COLLECTION)
    q_emb = get_model().encode([query], normalize_embeddings=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k)

    out: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out


def _print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        preview = r["text"][:220].replace("\n", " ")
        print(f"\n[{i}] dist={r['distance']:.3f}  "
              f"{m['source']}/{m['doc_type']}  (#{m['chunk_index']})")
        print(f"    {preview}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed + retrieve over chunks.")
    parser.add_argument("query", nargs="?", help="ad-hoc query string")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-embed chunks.json into the vector store")
    parser.add_argument("-k", type=int, default=TOP_K, help="top-k (default 5)")
    args = parser.parse_args()

    if args.rebuild:
        build_index()

    if args.query:
        _print_results(args.query, retrieve(args.query, args.k))
    elif not args.rebuild:
        # Smoke test on the evaluation-plan questions.
        for q in EVAL_QUESTIONS:
            _print_results(q, retrieve(q, args.k))
