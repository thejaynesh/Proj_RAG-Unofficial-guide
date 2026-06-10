"""
Stage 1 + 2 driver for the resume/cover-letter RAG pipeline.

Ingests every PDF in documents/, chunks them per the plan (1000 chars / 150
overlap, recursive splitting), validates the output the way Milestone 3's
checkpoint requires, and writes the chunks to chunks.json so the embedding
stage (Milestone 4) can load them without re-parsing PDFs.

Usage:
    python build_chunks.py [documents_dir] [-o chunks.json] [-n 5]

The checkpoint this satisfies:
  - 5 random chunks printed for manual inspection (readable, self-contained)
  - total chunk count reported and sanity-checked against the 50-2,000 range
  - empty chunks and duplicate chunk_ids flagged as errors
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Chunk text contains non-cp1252 glyphs (smart quotes, box-drawing chars from
# PDF tables); force UTF-8 so printing samples doesn't crash on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8")

from chunk import chunk_documents
from ingest import ingest_dir

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build(documents_dir: str, chunk_size: int = CHUNK_SIZE,
          chunk_overlap: int = CHUNK_OVERLAP):
    docs = ingest_dir(documents_dir)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    return docs, chunks


def report(docs, chunks, sample_n: int = 5) -> bool:
    """Print Milestone 3 stats + samples. Return True if checks pass."""
    print(f"\nIngested {len(docs)} document(s) -> {len(chunks)} chunk(s)\n")

    # Per-document chunk counts so it's obvious if one PDF dominates or vanished.
    print("Chunks per document:")
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.filename] = per_doc.get(c.filename, 0) + 1
    for fname, n in sorted(per_doc.items()):
        print(f"  {n:>4}  {fname}")

    sizes = [c.metadata["char_count"] for c in chunks]
    print(f"\nChar size  min={min(sizes)}  max={max(sizes)}  "
          f"avg={sum(sizes) // len(sizes)}")

    # --- checkpoint validations -------------------------------------------- #
    ok = True

    empties = [c.chunk_id for c in chunks if not c.text.strip()]
    if empties:
        ok = False
        print(f"\n[FAIL] {len(empties)} empty chunk(s): {empties[:5]}")
    else:
        print("\n[ok] no empty chunks")

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        ok = False
        dupes = {i for i in ids if ids.count(i) > 1}
        print(f"[FAIL] duplicate chunk_ids: {sorted(dupes)[:5]}")
    else:
        print("[ok] all chunk_ids unique")

    if not (50 <= len(chunks) <= 2000):
        ok = False
        print(f"[WARN] chunk count {len(chunks)} outside the 50-2000 range")
    else:
        print(f"[ok] chunk count {len(chunks)} within 50-2000 range")

    # --- random sample for manual reading ---------------------------------- #
    print(f"\n{'=' * 70}\n{sample_n} RANDOM CHUNKS (read each: self-contained?)\n{'=' * 70}")
    for c in random.sample(chunks, min(sample_n, len(chunks))):
        print(f"\n--- {c.chunk_id}  [{c.source}/{c.doc_type}]  "
              f"{c.metadata['char_count']} chars ---")
        print(c.text)

    return ok


def save(chunks, out_path: str) -> None:
    payload = [asdict(c) for c in chunks]
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nWrote {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build chunks from PDFs.")
    parser.add_argument("documents_dir", nargs="?", default="documents",
                        help="folder of PDF guides (default: documents)")
    parser.add_argument("-o", "--out", default="chunks.json",
                        help="output JSON path (default: chunks.json)")
    parser.add_argument("-n", "--sample", type=int, default=5,
                        help="how many random chunks to print (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the random sample for reproducible output")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    docs, chunks = build(args.documents_dir)
    passed = report(docs, chunks, args.sample)
    save(chunks, args.out)
    print("\nCheckpoint:", "PASS" if passed else "REVIEW NEEDED")
