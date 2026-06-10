# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system for resume and cover-letter
advice, grounded in 10 university career guides.

## Quickstart

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python build_chunks.py            # ingest PDFs -> chunks.json (693 chunks)
python retrieve.py --rebuild      # embed chunks -> ChromaDB
python app.py                     # launch UI at http://localhost:7860
```

---

## Domain

Resume and cover letter information. Although there are many resources available,
the information is scattered across university career guides and websites, making
it difficult to navigate and easy to get conflicting advice. This RAG system
combines ten university guides into one searchable source that gives users clear,
attributed answers — and surfaces where schools disagree — while they create or
improve their resumes and cover letters. The hard part to find elsewhere is the
*cross-source* view: any one guide is easy to read, but knowing whether "one page"
or "list references" is universal advice or just one school's opinion normally
requires reading all ten.

---

## Document Sources

Ten career guides (PDF) from university career-development offices. Nine are
resume guides and one is a dedicated cover-letter guide; together they cover
formatting, sections, bullet/accomplishment writing, length, references, and
cover-letter structure across a range of schools and student populations
(undergrad, graduate, recent-grad, career-changer).

| #  | Source label | Type         | File                          |
|----|--------------|--------------|-------------------------------|
| 1  | harvard      | resume       | Resume-Guide-harvard.pdf      |
| 2  | cmu          | resume       | Resume-Guide-cmu.pdf          |
| 3  | gatech       | resume       | Resume-Guide-gatech.pdf       |
| 4  | ucdavis      | resume       | Resume-Guide-ucdavis.pdf      |
| 5  | baruch       | resume       | Resume-Guide-baruch.pdf       |
| 6  | manhattan    | resume       | Resume-Guide-manhattan.pdf    |
| 7  | uconn        | resume       | Resume-Guide-uconn.pdf        |
| 8  | umichigan    | resume       | Resume-Guide-umichigan.pdf    |
| 9  | yale         | resume       | Resume-Guide-yale.pdf         |
| 10 | gatech       | cover_letter | coverletter-guide-gatech.pdf  |

---

## Chunking Strategy

**Chunk size:** 1000 characters (~250 words)

**Overlap:** 150 characters (~15%)

**Preprocessing (before chunking):** Each PDF is extracted with `pdfplumber`
using `layout=True` to keep reading order close to the visual layout (important
for the multi-column guides). Cleaning then removes the noise that would pollute
embeddings: standalone page numbers, and repeated header/footer boilerplate
detected by fingerprinting the first and last few lines of every page (a line is
treated as boilerplate only if it recurs on ≥50% of pages, with digit runs masked
so "PAGE 2"/"PAGE 3" collapse to one fingerprint). Hyphenated line-wraps
(`experi-\nence`) are repaired, and runs of blank lines/spaces are collapsed so
paragraph boundaries survive for the splitter.

**Method:** `RecursiveCharacterTextSplitter` (LangChain) with separator priority
`["\n\n", "\n", ". ", " ", ""]` — it targets the 1000-char size but splits on the
cleanest available boundary first (paragraph, then line, then sentence, then
space), so chunks rarely cut through the middle of a sentence or detach a heading
from its text.

**Why these choices fit the documents:** The 1000-character size is set to match
the embedding model (`all-MiniLM-L6-v2`), which silently truncates input beyond
256 tokens (~1000 characters). Larger chunks would be cut off at embed time with
no error; 1000 keeps each chunk fully embedded while still holding a complete tip
or sub-section. The 150-character (~15%) overlap is insurance against boundary
splits — if a definition or tip lands near a cut, it appears in both neighboring
chunks and can be matched from either side. I considered pure heading/section
chunking (the guides are organized into Education, Experience, Formatting, etc.),
but PDF extraction mangles headings and multi-column layouts, so detecting true
section boundaries is fragile; recursive chunking captures most of that benefit
without depending on extraction quality.

**Final chunk count:** **693 chunks** across the 10 documents
(baruch 172, yale 142, uconn 119, manhattan 64, ucdavis 50, harvard 38, cmu 33,
gatech-resume 31, gatech-cover_letter 30, umichigan 14). This sits comfortably in
the sensible 50–2,000 range. Each chunk carries `source`, `filename`, `doc_type`,
and `chunk_index` metadata so retrieval results can be attributed and contrasted
across schools.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim
embeddings). It is free, runs locally with no API cost or rate limits, embeds the
whole corpus in ~20 seconds, and its 256-token (~1000-char) input limit is exactly
what the 1000-character chunk size is designed around, so no chunk is silently
truncated at embedding time. Embeddings are L2-normalized and stored in a
**ChromaDB** collection configured for cosine distance (`hnsw:space=cosine`), so
distance scores run 0 (identical) to 2 (opposite) and the project's "below ~0.5"
relevance bar is meaningful. Retrieval uses **top-k = 5**.

**Production tradeoff reflection:** If deploying for real users with no cost
constraint, MiniLM's main weakness is that it is a small, general-purpose model
that truncates at 256 tokens, which forces small chunks and can hurt accuracy on
longer or more nuanced passages. I would weigh:

- **Context length:** a 512-token model such as `bge-base-en-v1.5` would allow
  larger chunks that keep a full section intact, reducing the chance that related
  advice is split across chunks (the exact failure that hurt the resume-length
  question below).
- **Accuracy / retrieval quality:** larger, higher-ranked models (BGE/E5 families,
  or a hosted option like OpenAI's `text-embedding-3-large`) score meaningfully
  higher on retrieval benchmarks and would likely surface the right guidance more
  often — which matters most for a system whose value is trustworthy answers.
- **Latency and cost:** bigger local models are slower and need more memory; hosted
  API models add per-call cost and network latency. For an interactive tool,
  response speed is part of the experience, so accuracy gains must be balanced
  against it.
- **Multilingual support:** the corpus is English-only and US-focused, so a
  multilingual model adds little here — but if expanded to international students
  or non-English guides, `multilingual-e5` would become worth the tradeoff.

For this project's scope, MiniLM's speed and zero cost outweigh the accuracy
ceiling; in production I would most likely move to a 512-token BGE or E5 model to
gain context length and retrieval accuracy while keeping inference local.

---

## Grounded Generation

The generator is Groq's `llama-3.3-70b-versatile` (free-tier, OpenAI-compatible,
key from `.env`). Grounding is enforced in **two layers**:

**1. Structural (before the LLM is called).** Retrieval returns the top-5 chunks
with cosine distances. Any chunk with distance > 0.65 is dropped. If *no* chunk
survives, the system returns the refusal string immediately **without calling the
LLM at all** — so a question the corpus doesn't cover cannot be answered from the
model's training knowledge. The retrieved context is formatted as numbered,
source-labeled excerpts (`[1] Source: yale (resume) …`).

**2. System-prompt instruction.** The model is told to use *only* the numbered
context, never outside knowledge; to reply with the exact phrase *"I don't have
enough information on that."* when the context is insufficient; and, when excerpts
from different schools disagree, to attribute advice per school
("Harvard's guide recommends… while Yale's suggests…") rather than present one
opinion as universal fact. Temperature is 0.2 to keep answers close to the source.

**System prompt grounding instruction (verbatim core):**
> "Use only information found in the context. Do not use outside knowledge. If the
> context does not contain enough information to answer, reply exactly: 'I don't
> have enough information on that.' … When excerpts from different schools give
> differing advice, attribute it rather than presenting one school's opinion as
> universal fact."

**How source attribution is surfaced:** Sources are computed **programmatically**
from the retrieved chunks' metadata (`source — filename`), deduplicated in
first-seen order, and returned alongside the answer — they are not left to the
model to invent. The Gradio UI shows them in a separate "Retrieved from" box. A
refusal returns an empty source list (nothing is attributed to "I don't know").

---

## Evaluation Report

All 5 questions were run end-to-end (`python generate.py "<question>"`, top-k=5).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What order should I list my work experience in? | Reverse chronological, most recent first | Reverse chronological, most recent first; attributed to Baruch, UCDavis, UMichigan | Relevant (top dist ~0.32) | **Accurate** |
| 2 | How should I describe accomplishments in bullet points? | Strong past-tense action verbs; quantify results; no pronouns/full sentences | Action verbs + quantify; synthesized and attributed four frameworks (gatech SAIL, ucdavis Action-Verb-Context-Results, yale SAR, uconn SCO) | Relevant (top dist 0.215) | **Accurate** |
| 3 | Should I include "References available upon request"? | No — outdated, wastes space; references provided separately when asked | "You do not need to include it"; references go on a separate document. Added a hedge ("but you can say it") reflecting gatech's softer wording | Relevant (cited gatech, ucdavis) | **Partially accurate** |
| 4 | What are the main parts of a cover letter? | Opening, body, closing paragraphs; greeting and formal sign-off | Intro/body/conclusion with heading and closing (Baruch); contrasted gatech's "4 paragraphs" and ucdavis's component list | Relevant (top dist 0.287) | **Accurate** |
| 5 | How long should a resume be for a student/recent grad? | Generally one page for students/recent grads | "I don't have enough information on that." (refused) | **Off-target** (correct one-page chunk ranked #8, outside top-5) | **Inaccurate** |

**Retrieval quality:** Relevant for 4/5; off-target for Q5.
**Response accuracy:** 3 Accurate, 1 Partially accurate, 1 Inaccurate.

The system is strongest where it was designed to be: Q1, Q2, and Q4 don't just
answer correctly, they pull chunks from multiple schools and attribute the advice
— exactly the cross-source value the domain was chosen for.

---

## Failure Case Analysis

**Question that failed:** "How long should a resume be for a student or recent
graduate?" (Q5)

**What the system returned:** "I don't have enough information on that." — a
refusal, despite the corpus clearly containing the one-page rule.

**Root cause (tied to the retrieval/embedding stage):** This is a **retrieval
failure caused by vocabulary mismatch in the embedding model**, not a generation
failure. The query is phrased around *length* ("how long", "be"), while the
relevant guidance is phrased around *page count* ("Must be one-page, unless you
have significant experience"). `all-MiniLM-L6-v2` embeds those as only loosely
related, so the chunk that actually answers the question (baruch chunk #3) ranked
**8th** at cosine distance 0.508, *outside* the top-5 window. The five chunks that
were retrieved (0.41–0.48) were about the Education, Experience, and Contact
sections — topically adjacent but silent on length. Generation then did exactly
what grounding requires: with no length information in its context, it refused
rather than inventing the (correct) answer from training knowledge. So the
*grounding worked correctly*; the *retrieval under-fetched*.

**What I would change to fix it:** Two options, confirmed by inspection. (a)
Raise top-k from 5 to 8 — the answering chunk is at rank 8 with distance 0.508,
well inside the 0.65 relevance gate, so k=8 recovers it immediately (kept at k=5
here to stay faithful to the spec and to preserve this as an honest failure case).
(b) The more robust production fix is a longer-context embedding model
(`bge-base-en-v1.5`) with larger chunks, so length guidance isn't isolated in a
short chunk that competes poorly on phrasing; or hybrid retrieval that adds a
keyword/BM25 signal to catch the literal "one page" match the dense embedding
missed.

---

## Spec Reflection

**One way the spec helped during implementation:** The Chunking Strategy section
tied the 1000-character chunk size directly to MiniLM's 256-token truncation
limit. That single justified number prevented a silent, hard-to-debug bug:
without it I might have used a "nicer" 2000-character chunk, and the back half of
every chunk would have been dropped at embedding time with no error and degraded
retrieval for no visible reason. Because the spec explained *why* 1000, the
embedding and chunking stages were consistent by construction, and the
distance-based relevance bar ("below ~0.5") gave a concrete target to verify
retrieval against.

**One way the implementation diverged from the spec, and why:** The spec described
grounding only as a *prompt instruction* ("instruct the generator to answer from
context only"). In implementation I added a **structural** grounding layer the
spec didn't mention: chunks past a 0.65 cosine-distance gate are dropped, and if
none survive the system refuses *without calling the LLM at all*. I added this
after realizing that prompt-only grounding still hands an out-of-corpus question's
loosely-related chunks to a 70B model that is perfectly capable of "helpfully"
answering from training knowledge anyway. The deterministic pre-filter makes
out-of-corpus refusal a guarantee rather than a hope. (A second, smaller
divergence: the spec fixed top-k=5 and noted it might be raised during tuning;
evaluation showed k=8 would fix Q5, but I kept k=5 to document the failure
honestly rather than tune it away.)

---

## AI Usage

**Instance 1 — Document pipeline (Milestone 3)**

- *What I gave the AI:* My `Planning.md` Documents and Chunking Strategy sections
  (10 PDFs, recursive splitting, 1000 chars / 150 overlap, separator priority) and
  the architecture diagram, asking it to wire `ingest.py` → `chunk.py` into a
  driver and verify the output.
- *What it produced:* A `build_chunks.py` driver running ingest → chunk with
  validation (no empty chunks, count in the 50–2,000 range, random sample print)
  and a `chunks.json` export for the embedding stage.
- *What I changed or overrode:* I caught a real bug it would otherwise have shipped
  — the chunk IDs were `{source}-{index}`, and because two of my guides share the
  `gatech` source label (one resume, one cover letter), their IDs collided, which
  would have silently overwritten chunks in ChromaDB. I directed the fix to include
  `doc_type` in the ID (`gatech-resume-000` vs `gatech-cover_letter-000`). I also
  had it force UTF-8 stdout after a box-drawing character from a PDF table crashed
  the sample print on the Windows console.

**Instance 2 — Embedding, retrieval, and grounded generation (Milestones 4–5)**

- *What I gave the AI:* My Retrieval Approach section (MiniLM, top-k=5) and my
  grounding requirement (answer from retrieved context only, decline when
  insufficient, attribute sources), asking it to implement embedding + ChromaDB
  storage + a retrieval function, then connect retrieval to Groq.
- *What it produced:* `retrieve.py` (embed + cosine ChromaDB store + `retrieve()`)
  and `generate.py` + `app.py` (grounded answer + Gradio UI).
- *What I changed or overrode:* I directed the vector store to use cosine space
  (`hnsw:space=cosine`) so distances map onto the spec's "below ~0.5" bar instead
  of ChromaDB's default squared-L2, which would have made the threshold
  meaningless. I also insisted source attribution be built *programmatically* from
  chunk metadata rather than trusting the model to cite correctly, and added the
  structural pre-LLM refusal filter described in the Spec Reflection. When the
  resume-length question failed, I directed a diagnosis (printing per-chunk
  distances) rather than accepting the refusal, which is how the rank-8 retrieval
  miss was found.
