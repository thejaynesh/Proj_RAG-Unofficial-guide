"""
Recursive character chunking for the resume/cover-letter RAG system.

Stage 2 of the pipeline. Uses LangChain's RecursiveCharacterTextSplitter to
split each cleaned document on the cleanest available boundary first
(paragraph -> line -> sentence -> space -> char), then wraps each chunk with
the source metadata needed to attribute advice across schools.

Defaults (from the plan):
  chunk_size    = 1000 characters  (~250 words, fits MiniLM's 256-token limit)
  chunk_overlap = 150 characters   (~15%, standard 10-20% range)

"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Boundary preference: paragraph, then line, then sentence, then word, then char.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single retrievable unit plus the metadata needed for attribution."""
    chunk_id: str
    text: str
    source: str
    filename: str
    doc_type: str
    chunk_index: int          # position of this chunk within its source doc
    metadata: dict = field(default_factory=dict)


def make_splitter(chunk_size: int = 1000,
                  chunk_overlap: int = 150) -> RecursiveCharacterTextSplitter:
    """Build the LangChain splitter configured to the plan's settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,        # measure in characters, matching the plan
        keep_separator=True,
    )


def chunk_documents(docs, chunk_size: int = 1000,
                    chunk_overlap: int = 150) -> list[Chunk]:
    """
    Turn a list of IngestedDoc (from ingest.py) into a flat list of Chunk
    objects, each tagged with its source so retrieval results can be attributed.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)
    all_chunks: list[Chunk] = []

    for doc in docs:
        doc_type = doc.metadata.get("doc_type", "resume")
        for idx, piece in enumerate(splitter.split_text(doc.text)):
            all_chunks.append(
                Chunk(
                    # Include doc_type so two guides sharing a source label
                    # (e.g. gatech resume + gatech cover letter) don't collide.
                    chunk_id=f"{doc.source}-{doc_type}-{idx:03d}",
                    text=piece,
                    source=doc.source,
                    filename=doc.filename,
                    doc_type=doc_type,
                    chunk_index=idx,
                    metadata={"char_count": len(piece)},
                )
            )
    return all_chunks


if __name__ == "__main__":
    # Quick self-test on synthetic text so the splitter can be verified without PDFs.
    sample = (
        "Education\n\n"
        "List your degree, institution, and graduation date in reverse "
        "chronological order. Include GPA if it is 3.0 or higher.\n\n"
        "Work Experience\n\n"
        "Begin every bullet with a strong action verb such as developed, led, "
        "or analyzed. Quantify results wherever possible. Avoid personal "
        "pronouns and full sentences. " * 8
    )
    sp = make_splitter(chunk_size=300, chunk_overlap=50)
    for i, c in enumerate(sp.split_text(sample)):
        print(f"--- chunk {i} ({len(c)} chars) ---\n{c}\n")