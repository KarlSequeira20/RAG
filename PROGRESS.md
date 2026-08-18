# Progress Log

Running log of what I built, what broke, and what I learned. Update this at the end of each work session — few minutes, not an essay. This is also for Claude Code to read at the start of a session to know where I left off and why I made certain calls.

Format per entry: date, day number from ROADMAP.md, what I did, what broke, what I learned/still don't get.

---

## Template (copy this for each new entry)

### [Date] — Day X
**Did:**
-

**Broke / got stuck on:**
-

**Learned:**
-

**Still don't fully get:**
-

---

## Entries

### 2026-08-15 — Day 1

**Did:**
- Set up `.venv` (Python 3.14.6), installed langchain, langchain-community, chromadb, pypdf.
- `git init`, wrote `.gitignore` (excludes `.venv/`, `chroma_db/`, `.DS_Store`).
- Picked the document set: 4 floor-plan analysis papers in `data/`, 60 pages total.
- Wrote `ingest.py` — loads every PDF, reports page counts, counts blank pages,
  and prints the metadata of one page.

**Broke / got stuck on:**
- Nothing broke. All 4 PDFs loaded, 0 blank pages out of 60.
- One deprecation warning: `langchain-community` is being sunset; `PyPDFLoader` is
  moving to a standalone package. Works fine for now, revisit before deploying.

**Learned:**
- `PyPDFLoader.load()` returns one `Document` per page, each with `.page_content`
  (text) and `.metadata` (dict). `Document` is the container that flows through the
  whole LangChain pipeline.
- A blank `page_content` means a scanned page — an image, with no text to extract.
  It fails silently rather than raising, so ingestion has to check for it explicitly.
  Our papers are all born-digital, so this isn't an issue for this doc set.
- Metadata has both `page` (0-indexed) and `page_label` (printed number, string).
  Have to pick one deliberately for citations or they'll be off by one.

**Still don't fully get:**
- Whether the two-column layout of these papers is scrambling the extracted text
  on body pages. Only checked page 1, which looked clean.

---

### 2026-08-15 — Day 2

**Did:**
- Wrote `chunk_test.py` to compare chunk sizes 200/500/1000/2000 on one paper and
  read the actual chunks produced. Settled on 1000 chars / 100 overlap.
- Pulled `nomic-embed-text` via Ollama (768-dimensional embeddings).
- Wrote `build_index.py`: loads all PDFs, chunks them, embeds in batches of 32,
  stores in a persistent Chroma collection at `chroma_db/`. 392 chunks total.
- Sanity-checked retrieval with the query "What is the CubiCasa5K dataset?" —
  top result was the actual dataset description on page 4. Pipeline works end to end.

**Broke / got stuck on:**
- Nothing broke, but retrieval surfaced a real quality problem (below).

**Learned:**
- Chunk size is a real tradeoff, not a default to copy. Too small and a chunk starts
  mid-sentence with no subject; too big and one embedding has to represent several
  topics at once, so it matches everything weakly.
- `chunk_size` is a maximum, not a target — the recursive splitter prefers to break at
  paragraph/line/sentence boundaries, so average chunk size came out at 834.
- Query and documents must be embedded with the *same* model. Changing the embedding
  model means rebuilding the whole index.
- Chroma distances are cosine distance: lower is more similar, and ~0.6 is a normal
  score for a good hit. Not a percentage.

**Still don't fully get:**
- Figure/diagram text is being extracted into chunks as unreadable label soup, and it
  ranked 2nd on a test query (0.649 vs 0.609 for the correct chunk). Don't yet know
  how to filter it out without also throwing away real content.
- Why `2211.15658v2.pdf` produces 165 chunks from 15 pages when a similar-length paper
  produces 43 from 12.

---

### 2026-08-15 — Day 3

**Did:**
- Pulled `llama3.1:8b` via Ollama.
- Wrote `ask.py` — the complete RAG loop: embed question, retrieve top-4 chunks from
  Chroma, format them as numbered sources, send to the LLM with a grounding system
  prompt, print the answer plus the sources it drew on.
- Tested with an in-scope question (correct answer, cited [1] and [2] with page
  numbers) and an out-of-scope one ("What is the capital of France?" — correctly
  refused).

**Broke / got stuck on:**
- Nothing broke. Two things look suspicious, noted below.

**Learned:**
- The system prompt is doing the real work. Three parts: use only the given sources,
  never use outside knowledge, and an explicit sentence to use when the answer isn't
  there. That last part is what stops the model inventing — with no sanctioned way to
  decline, it will produce something rather than nothing.
- The refusal test is the meaningful one. Llama 3.1 obviously knows the capital of
  France; that it declined proves the grounding holds rather than just looking like it
  does on questions the model could have answered anyway.
- Chroma always returns exactly k results. It cannot say "no match" — it returns the
  least-bad options regardless. Relevance has to be judged from the distances, not
  from the fact that results came back.

**Still don't fully get:**
- Sources [3] and [4] on the out-of-scope query were both page 10 at an identical
  distance of 1.173. Probably near-duplicate chunks produced by the 100-char overlap,
  which would mean duplicates are quietly eating retrieval slots. Not verified.
- Whether top-k=4 is right, or a distance threshold would be better than a fixed k.
