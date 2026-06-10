"""
Document ingestion for the resume/cover-letter RAG system.

Stage 1 of the pipeline:
  1. Extract raw text from each university PDF guide.
  2. Strip boilerplate that would otherwise pollute downstream chunks/embeddings:
     repeated page headers/footers and standalone page numbers.
  3. Attach source metadata (school label + doc type) so later stages can
     attribute and contrast advice across schools.

"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class IngestedDoc:
    """One cleaned document, ready to be chunked in the next stage."""
    source: str            # short school/source label, e.g. "harvard"
    filename: str          # original file name
    path: str              # absolute path on disk
    text: str              # cleaned full text
    n_pages: int           # page count
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Source labeling
# --------------------------------------------------------------------------- #
def source_label_from_filename(filename: str) -> str:
    """
    Derive a clean source label from a filename so retrieved chunks can be
    attributed, e.g. 'Resume-Guide-harvard.pdf' -> 'harvard',
    'coverletter-guide-gatech.pdf' -> 'gatech'.
    """
    stem = Path(filename).stem.lower()
    # Drop the descriptive prefix, keep the trailing school token.
    stem = re.sub(r"(resume|cover[\s_-]?letter|guide)[\s_-]*", "", stem)
    token = re.split(r"[\s_-]+", stem.strip("-_ "))[-1] if stem else "unknown"
    return token or "unknown"


def doc_type_from_filename(filename: str) -> str:
    """Tag whether a guide is about resumes or cover letters (for filtering)."""
    name = filename.lower()
    if "cover" in name:
        return "cover_letter"
    return "resume"


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
# Lines that are nothing but a page number, e.g. "3", "- 4 -", "Page 5", "5/12".
_PAGE_NUMBER_RE = re.compile(
    r"""^\s*
        (?:page\s*)?            # optional 'Page'
        [-–—|\s]*               # optional surrounding dashes/pipes
        \d+                     # the number
        (?:\s*/\s*\d+)?         # optional '/ 12'
        [-–—|\s]*
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_line(line: str) -> str:
    """Collapse internal whitespace so header/footer matching is robust."""
    return re.sub(r"\s+", " ", line).strip()


def _fingerprint(line: str) -> str:
    """
    Normalize a line for repeated-boilerplate detection. In addition to
    collapsing whitespace, mask digit runs so footers/headers that embed a
    changing page number (e.g. '... RESUME GUIDE PAGE 2', 'PAGE 3') collapse to
    one fingerprint and are recognized as the same recurring line.
    """
    return re.sub(r"\d+", "#", _normalize_line(line))


def _detect_repeated_lines(pages: list[str], band: int = 3,
                           min_fraction: float = 0.5) -> set[str]:
    """
    Find header/footer boilerplate by looking at the first `band` and last
    `band` lines of every page. Any fingerprint that appears on at least
    `min_fraction` of pages is treated as boilerplate.

    Intentionally conservative: a line must recur across most pages to be
    removed, so legitimate body text is not stripped.
    """
    if len(pages) < 3:
        # Too few pages to reliably distinguish boilerplate from content.
        return set()

    counter: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.splitlines() if ln.strip()]
        edge_lines = lines[:band] + lines[-band:]
        # Use a set per page so a line repeated within one page isn't over-counted.
        for fp in {_fingerprint(l) for l in edge_lines if _fingerprint(l)}:
            counter[fp] += 1

    threshold = max(2, int(len(pages) * min_fraction))
    return {fp for fp, count in counter.items()
            if count >= threshold and len(fp) <= 120}


def _clean_text(pages: list[str]) -> str:
    """Remove page numbers and detected header/footer boilerplate, then tidy."""
    boilerplate = _detect_repeated_lines(pages)
    cleaned_pages: list[str] = []

    for page in pages:
        kept: list[str] = []
        for raw_line in page.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                kept.append("")  # preserve paragraph breaks for the chunker
                continue
            if _PAGE_NUMBER_RE.match(line):
                continue
            if _fingerprint(line) in boilerplate:
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))

    text = "\n\n".join(cleaned_pages)

    # Repair hyphenated line-wrap breaks: "experi-\nence" -> "experience".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of 3+ newlines down to a clean paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def ingest_pdf(path: str | Path) -> IngestedDoc:
    """Extract and clean a single PDF into an IngestedDoc."""
    path = Path(path)
    raw_pages: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # layout=True keeps reading order closer to the visual layout,
            # which helps with the multi-column guides.
            raw_pages.append(page.extract_text(layout=True) or "")

    cleaned = _clean_text(raw_pages)
    filename = path.name

    return IngestedDoc(
        source=source_label_from_filename(filename),
        filename=filename,
        path=str(path.resolve()),
        text=cleaned,
        n_pages=len(raw_pages),
        metadata={
            "doc_type": doc_type_from_filename(filename),
            "char_count": len(cleaned),
        },
    )


def ingest_dir(directory: str | Path, pattern: str = "*.pdf") -> list[IngestedDoc]:
    """Ingest every PDF in a directory, sorted for reproducible ordering."""
    directory = Path(directory)
    pdf_paths = sorted(directory.glob(pattern))
    if not pdf_paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    docs: list[IngestedDoc] = []
    for p in pdf_paths:
        try:
            docs.append(ingest_pdf(p))
        except Exception as exc:  # keep going if one PDF is malformed
            print(f"[warn] failed to ingest {p.name}: {exc}")
    return docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest resume/cover-letter PDFs.")
    parser.add_argument("directory", help="Folder containing the PDF guides")
    args = parser.parse_args()

    docs = ingest_dir(args.directory)
    print(f"Ingested {len(docs)} document(s):\n")
    for doc in docs:
        preview = doc.text[:200].replace("\n", " ")
        print(f"{doc.source:>12} | {doc.n_pages}p | "
              f"{doc.metadata['char_count']:>6} chars | {doc.metadata['doc_type']}")
        print(f"             | {preview}...\n")