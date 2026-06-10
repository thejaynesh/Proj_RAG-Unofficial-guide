# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A coursework RAG system (CodePath AI201, Project 1) answering resume/cover-letter questions grounded in 10 university career-guide PDFs. The full spec lives in `Planning.md` — read it before changing pipeline behavior; chunk sizes, top-k, and model choices there are deliberate and justified. `README.md` is the graded submission template, filled in *after* each part works.

## Pipeline (5 stages)

```
documents/*.pdf → ingest.py → chunk.py → [embed+store] → [retrieve] → [generate]
                   stage 1      stage 2      stage 3        stage 4      stage 5
```

- **`ingest.py`** (done) — `ingest_dir(dir)` / `ingest_pdf(path)` → `IngestedDoc`. Extracts PDF text with `pdfplumber` (`layout=True`), strips page numbers + repeated header/footer boilerplate (detected by fingerprinting edge lines across pages), repairs hyphenated line-wraps. Derives `source` label (e.g. `harvard`) and `doc_type` (`resume`/`cover_letter`) from filename.
- **`chunk.py`** (done) — `chunk_documents(docs)` → `list[Chunk]`. `RecursiveCharacterTextSplitter`, 1000 chars / 150 overlap, separators `["\n\n","\n",". "," ",""]`. Each `Chunk` carries `source`/`filename`/`doc_type`/`chunk_index` so retrieval can attribute and contrast advice across schools.
- **`build_chunks.py`** — empty; intended driver wiring ingest → chunk.
- **Stages 3–5** (embed/store, retrieve, generate) — not built yet. Plan: `all-MiniLM-L6-v2` embeddings, ChromaDB store, top-k=5, Groq `llama-3.3-70b-versatile` generation.

Metadata threading (`source`/`doc_type`) is the backbone — it exists so generation can say "Harvard recommends X while Yale suggests Y" and flag disagreement. Preserve it through every new stage.

## Commands

```powershell
.venv\Scripts\Activate.ps1                 # venv already created
pip install -r requirements.txt
python ingest.py documents                 # ingest CLI: prints per-doc summary
python chunk.py                             # self-test on synthetic text, no PDFs needed
```

No test framework — each module verifies itself via its `__main__` block.

## Dependency gap (important)

`requirements.txt` does **not** install everything the existing code imports:
- `ingest.py` needs `pdfplumber` (commented out in requirements.txt)
- `chunk.py` needs `langchain-text-splitters` (absent entirely)

Install these before running, and add them to `requirements.txt` when touching those modules.

## Conventions

- Constraints are spec-driven: chunk size matches MiniLM's 256-token truncation limit; changing it can silently break embedding. If you change chunking or retrieval, update the corresponding `Planning.md` section too (the spec requires this).
- `GROQ_API_KEY` comes from `.env` (copy `.env.example`); free key from console.groq.com.
- PDFs go in `documents/` (gitignored content, kept via `.gitkeep`). `chroma_db/` is gitignored.
