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

---

### 2026-08-15 — Day 5

**Did:**
- Investigated the two things flagged as suspicious on Day 3, before starting manual
  answer verification, so the verification wouldn't be tripping over known bugs.
- Found the cause of the chunk-count anomaly and fixed it: added `clean_text()` to
  `ingest.py`, stripping `<latexit>...</latexit>` blocks at load time.
- Rebuilt the index and re-tested.

**Broke / got stuck on:**
- Not broken so much as quietly wrong the whole time. `2211.15658v2.pdf` embeds the
  LaTeX source of every equation as hidden text — invisible when reading the PDF, but
  pypdf extracts it as long base64 blobs. 94 such blocks, 44% of that document's
  extracted characters, 57% of its chunks. Roughly a quarter of the entire index was
  garbage and nothing about the output looked wrong.

**Learned:**
- Extracted PDF text is not what you see on the page. Tools embed things a reader never
  sees, and extraction pulls them out along with the real content.
- The way to find this was measuring rather than reading. Counting duplicate chunk
  texts and comparing per-file chunk statistics exposed it immediately; spot-checking
  answers never would have, because answers still looked fine.
- Cleaning is worth doing at load time, in one place, so every downstream stage
  benefits and no script can accidentally skip it.
- Result was measurable: index 392 -> 305 chunks, and retrieval distances on that paper
  improved from ~0.60 to ~0.42-0.54. Less noise competing for the top-k slots.

**Still don't fully get:**
- Figure labels (axis text, diagram annotations) are still extracted into chunks. Less
  severe than the LaTeX problem, but no filter for it yet, and it's harder because it
  isn't wrapped in a tag that marks it as junk.
- 14 of 305 chunks are still exact duplicates from the overlap. Left alone for now.

**Next:** write 10-15 real questions about these papers and check, for each one, whether
the cited source actually contains the answer.

---

### 2026-08-15 — Day 6

**Did:**
- Wrote `EVAL_QUESTIONS.md` — 15 questions in five categories (specific numbers, correct
  source, multi-chunk, paraphrased, and not-in-documents refusal tests).
- Ran four of them and found two genuine failures.
- Fixed a chain of three separate root causes, measuring after each one.
- Built `retriever.py`: hybrid BM25 + vector retrieval merged with Reciprocal Rank
  Fusion, and rewired `ask.py` to use it.
- Added `requirements.txt`.

**Broke / got stuck on:**
- Questions naming a specific paper retrieved chunks from the wrong paper. Three causes,
  found one at a time by measuring where the correct chunk actually ranked:
  1. Vector search cannot distinguish proper nouns. All four papers are about
     floorplans, so "FloorPlanCAD" and "CubiCasa5K" sit in nearly the same place in
     meaning space. Fix: prefix each chunk with its paper title before embedding.
     Result: correct paper now dominates the results, but the right *chunk* still
     ranked #11 and #250.
  2. PDF hyphenation. Justified text breaks words across lines — 568 times in these
     four papers. "FloorPlan-\nCAD" means the token "floorplancad" does not exist, so
     keyword search could never match it. Fix: rejoin in `clean_text()`, and strip
     hyphens in the tokenizer.
  3. RRF rewards agreement. The chunk listing FloorPlanCAD's contributions ranked #1 by
     keyword and #250 by vector, and lost to chunks both methods ranked mediocrely.
     Fix: reserve 2 slots for each method's own top hits before RRF fills the rest.

**Learned:**
- Vector search and keyword search fail in opposite directions. Embeddings handle
  rephrasing but not exact tokens; BM25 handles exact tokens but not rephrasing. Hybrid
  retrieval is not a refinement — for this document set it was the difference between
  answering and not answering.
- Debugging retrieval means asking "where does the correct chunk actually rank?", not
  "did the answer look right?". Rank #13 vs #250 vs not-in-corpus are three different
  bugs with three different fixes, and they are indistinguishable from the answer alone.
- What gets embedded matters as much as how. A chunk from the middle of a paper says
  "our dataset" and never names it, so nothing ties it to its source. Adding the title
  fixed that in one line.
- Fixing each cause one at a time, and re-measuring, is what made the third cause
  visible. Changing everything at once would have hidden it.

**Still don't fully get:**
- Generation is now the bottleneck. On the FloorPlanCAD count question the correct chunk
  *is* retrieved and llama3.1:8b still misreads it — quoting half the sentence, claiming
  no total is given, then reporting a wrong number from another page. Unclear whether
  that is prompt design, too many chunks at once, or the model being 8B.
- All these settings were tuned against 5 questions. Almost certainly overfitted.

**Next:** run the full 15-question set, record PASS/FAIL, and investigate the generation
failure.

---

### 2026-08-15 — Day 6 (later)

**Did:**
- Wrote `run_eval.py` and ran the full 15-question set.
- Result: 12 PASS, 2 PARTIAL, 1 FAIL. Recorded in `EVAL_QUESTIONS.md`.
- Diagnosed the one failure.

**Broke / got stuck on:**
- Q13 "Who made FloorPlanCAD?" fails. The chunk with the author affiliations ranks
  keyword #11 and vector #57, so it never enters the top 6.

**Learned:**
- The settings tuned on 5 questions held up across all 15. Worth having checked rather
  than assumed, and it doesn't prove they're optimal — only that they're not obviously
  overfitted.
- Some questions are structurally unanswerable by chunk retrieval. "Who wrote this",
  "when was it published", "how many pages" are about a document, not about anything
  said inside it. Similarity search over content cannot find a byline, because a byline
  isn't semantically similar to a question about authorship. The answer is to serve
  those from stored metadata, not to keep tuning retrieval.
- Partial answers are their own failure mode. Q3 states the right number and then adds
  two unrelated figures from other pages; Q12 answers a narrower question than the one
  asked. Both would score as passes on a keyword check and are wrong to a reader — which
  is why an automated `expect` string can't replace reading the answers.
- Two of my own `expect` keywords were badly chosen and produced false alarms. The
  grading script narrows what needs attention; it doesn't decide.

**Still don't fully get:**
- Whether the Q3 noise is a prompt problem (model over-including) or a retrieval one
  (too many marginal chunks at k=6).
- The model narrating its own indecision before answering. Probably an 8B artifact,
  possibly fixable in the prompt.

**Next:** serve metadata-type questions from metadata, and tighten the prompt so answers
stop dragging in unrelated figures.

---

### 2026-08-15 — Day 7

**Did:**
- Iterated the system prompt three times, re-running all 15 questions after each change.
- Kept the version that removed pasted source extracts, narrated reasoning, and invented
  arithmetic.
- Updated ARCHITECTURE.md with eval results, and filled in the eval and bug questions in
  CONCEPTS.md.

**Broke / got stuck on:**
- Second prompt iteration made the model fabricate a calculation: given "2,500 files" and
  "over 100 projects" from one page and "over 15,000 floor plans" from another, it
  multiplied them and presented the result as an answer. Nothing in the paper does that.
  Fixed with an explicit rule against calculating or inferring unstated numbers.

**Learned:**
- Prompt changes trade regressions. Telling the model to answer only what was asked made
  it paste source extracts instead; forbidding extracts made it fabricate arithmetic to
  reconcile figures. Every rule in the prompt now exists because of a specific observed
  failure.
- The pass rate stayed at 12/15 across all three iterations while answer quality improved
  a lot. A single headline number hides that, which is an argument for keeping the actual
  answers alongside the score.
- Knowing when to stop matters. After the third iteration each change fixed one question
  and broke another, which is the signal that the limit is the model rather than the
  wording.

**Still don't fully get:**
- Whether a larger model would fix A3 and D12, or whether they need retrieval changes.
  Have not tested one.

**Next:** Week 2, Day 8-10 — wrap the pipeline in a FastAPI endpoint. Remaining known
issues (metadata questions, A3's competing figures) are recorded as limitations rather
than blockers.

---

### 2026-08-15 — Day 8

**Did:**
- Wrote `api.py`: FastAPI wrapper over the pipeline. `GET /health` and `POST /ask`.
- Built the retriever once at startup via a `lifespan` handler instead of per request.
- Tested seven error cases with curl; all behave as intended.
- Added fastapi and uvicorn to `requirements.txt`.

**Broke / got stuck on:**
- Nothing broke. One gap found while testing: a whitespace-only question `"   "` passes
  Pydantic's `min_length=1`, so it needed an explicit check after stripping.

**Learned:**
- Pydantic handled four of the seven error cases without any code in the endpoint —
  empty string, missing field, malformed JSON, oversized input all rejected before the
  function runs, each with a 422 and a message naming the offending field. Validation
  written by describing the data rather than by writing checks.
- Declarative validation still has edges. `min_length=1` is about length, not content,
  so whitespace slips through. Worth knowing where the automatic help stops.
- Where expensive setup lives matters. `HybridRetriever()` loads the corpus and builds
  BM25 statistics; in a CLI that runs once per process it can sit anywhere, but in a
  server it has to move to startup or every request pays for it.
- "No answer in the documents" is a 200, not a 4xx. The request succeeded and the
  system correctly reported that the corpus doesn't cover it. Making that an error would
  be wrong, and would make the refusal behaviour look like a malfunction to a client.

**Still don't fully get:**
- The 503 path for Ollama being down is written but untested — didn't want to stop the
  Ollama service to exercise it.

**Next:** Day 11-12, a Streamlit UI on top of this API.
