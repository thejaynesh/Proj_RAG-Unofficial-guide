"""
Stage 5: grounded generation.

Wires retrieval (retrieve.py) to Groq's llama-3.3-70b-versatile. Grounding is
enforced two ways:

  1. Structurally - chunks whose cosine distance exceeds MAX_DISTANCE are
     dropped before the prompt is built. If nothing relevant survives, we return
     the refusal WITHOUT calling the LLM, so an out-of-corpus question can't be
     answered from the model's training knowledge.
  2. By instruction - the system prompt restricts the model to the supplied
     context and tells it to refuse when the context is insufficient.

Source attribution is computed programmatically from the retrieved chunks'
metadata, not left to the model to invent.

Usage:
    python generate.py "How long should a student resume be?"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve, TOP_K

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_DISTANCE = 0.65          # drop chunks looser than this (cosine distance)
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about writing resumes and "
    "cover letters, using ONLY the numbered context excerpts provided, which "
    "come from university career guides.\n\n"
    "Rules:\n"
    "1. Use only information found in the context. Do not use outside knowledge.\n"
    f"2. If the context does not contain enough information to answer, reply "
    f"exactly: \"{REFUSAL}\" and nothing else.\n"
    "3. When excerpts from different schools give differing advice, attribute "
    "it (e.g., \"Harvard's guide recommends... while Yale's suggests...\") "
    "rather than presenting one school's opinion as universal fact.\n"
    "4. Be concise and practical."
)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set (copy .env.example to .env)")
        _client = Groq(api_key=key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        label = f"{m['source']} ({m['doc_type']})"
        blocks.append(f"[{i}] Source: {label}\n{c['text']}")
    return "\n\n".join(blocks)


def _sources(chunks: list[dict]) -> list[str]:
    """Unique source labels from the retrieved chunks, in first-seen order."""
    seen: list[str] = []
    for c in chunks:
        m = c["metadata"]
        label = f"{m['source']} — {m['filename']}"
        if label not in seen:
            seen.append(label)
    return seen


def ask(question: str, k: int = TOP_K) -> dict:
    """Retrieve, generate a grounded answer, and attach source attribution."""
    results = retrieve(question, k)
    relevant = [r for r in results if r["distance"] <= MAX_DISTANCE]

    # Structural grounding: nothing relevant retrieved -> refuse, no LLM call.
    if not relevant:
        return {"answer": REFUSAL, "sources": [], "chunks": results}

    context = _format_context(relevant)
    user_msg = (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    completion = get_client().chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # Don't attribute sources to a refusal.
    sources = [] if answer.startswith(REFUSAL[:20]) else _sources(relevant)
    return {"answer": answer, "sources": sources, "chunks": relevant}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python generate.py "your question"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = ask(q)
    print(f"\nQ: {q}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s}")
