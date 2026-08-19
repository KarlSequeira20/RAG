# Architecture Notes

Running notes on how the system is built and *why* — not a polished doc, just capturing decisions as I make them so I'm not trying to remember my own reasoning in Week 3 when I write the real README.

Update a section whenever you make or change a real design decision. Leave "why" blank until you actually understand it — don't fill it in with a guess.

---

## Document Set
- **What:** 4 research papers on floor plan / building layout analysis, in `data/`:
  - `1904.01920v1.pdf` — CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis (12 pages)
  - `2105.07147v2.pdf` (18 pages)
  - `2211.15658v2.pdf` (15 pages)
  - `ijgi-10-00828-v2.pdf` — ISPRS Int. J. Geo-Information (15 pages)
  - 60 pages total.
- **Why this choice:** Same domain as the floor-plan/CV project planned next, so the
  reading isn't throwaway. All four are born-digital (LaTeX/publisher-generated), not
  scans, so text extraction works without OCR — verified, 0 blank pages out of 60.

## Web Page Ingestion (added Day 13)
- **What:** any URL can be indexed alongside the PDFs (`web.py`, `build_index.index_url`).
  The vector store does not care where text came from, so web chunks sit in the same
  collection and are retrieved by the same hybrid search and reranker.
- **Extraction:** `trafilatura`, not raw HTML parsing. A typical page is mostly
  navigation, cookie banners, footers, "related articles" and ads; indexing those
  produces chunks that compete with real content during retrieval — the same failure the
  hidden LaTeX source caused in the PDFs. trafilatura identifies the main article body
  and discards the furniture.
- **Citation:** web pages have no page numbers, so their chunks are cited as *parts*
  (part 1, part 2...). A `kind` field on every chunk (`"pdf"` or `"web"`) drives this, so
  no display surface has to guess from the filename. `ask.describe_location()` is the one
  place that formats it, shared by the CLI, API and UI.
- **Snapshot, not live.** A page is fetched once at index time and not re-checked when a
  question is asked. Re-fetching per query would mean re-extracting and re-embedding on
  every question. The consequence — an index can go stale — is stated in the UI.
- **Failure cases handled:** non-http URLs, unreachable or blocking hosts, and pages with
  no extractable text. The last is common and worth naming: a page built entirely in
  JavaScript returns nothing, because only the initial HTML is fetched and no browser
  runs to build the page.
- **Verified:** indexed the Wikipedia article on retrieval-augmented generation (20
  chunks). Asking "What is retrieval-augmented generation?" retrieves only web chunks;
  asking about CubiCasa5K retrieves only PDF chunks. Retrieval metrics were unchanged
  after adding it, so the new content did not crowd out the papers.

## Chunking
- **Chunk size / overlap used:** 1000 characters, 100 overlap, via
  `RecursiveCharacterTextSplitter`. Produces 305 chunks across the 4 papers
  (392 before hidden-LaTeX cleaning was added — see Open Questions).
- **Why:** Compared 200/500/1000/2000 on one paper (`chunk_test.py`) and read the
  actual output rather than copying a number from a tutorial.
  - Too small (200) fragments mid-sentence — sample chunk began "of outputs in the
    very latest block", with no subject. Retrievable but useless to the LLM.
  - Too big dilutes the embedding: one vector covering several topics matches
    everything vaguely and nothing precisely, and wastes LLM context.
  - 1000 is large enough to hold one complete idea, small enough to stay on topic.
    100 overlap (10%) means a thought split across a boundary still appears whole
    in one of the two chunks.
  - Note: the splitter treats `chunk_size` as a ceiling, not a target — it breaks on
    paragraphs, then lines, then sentences. Average chunk came out at 834 chars.
    It also never merges across pages, which is what keeps page metadata accurate.
- **Not yet tuned.** Day 13 evaluation is where these numbers should be tested with
  real retrieval scores instead of judgement.

## Embeddings
- **Model used:** `nomic-embed-text` via Ollama, 768 dimensions.
- **Why:** Runs locally, no API key, no per-call cost. Ollama is already the plan for
  the LLM on Day 3, so the whole stack stays on one runtime. Better quality than
  Chroma's bundled default (`all-MiniLM-L6-v2`), at the cost of a 274MB pull.
  Swapping the embedding model means re-running `build_index.py` from scratch —
  query embeddings and stored embeddings must come from the same model, or
  similarity search is meaningless.

## Vector Store
- **Using:** Chroma (`PersistentClient`, on-disk at `chroma_db/`)
- **Why Chroma (vs alternatives):** Runs in-process with no server to operate, and
  persists to a local folder, so there's nothing to deploy alongside the app. Fine at
  this scale (hundreds of chunks). Pinecone/Weaviate solve problems — millions of
  vectors, multi-user scale — that this project doesn't have.
- `chroma_db/` is gitignored: it's fully rebuildable from the PDFs by re-running
  `build_index.py`.

## Retrieval
- **Method:** hybrid — BM25 keyword search and vector search, merged with Reciprocal
  Rank Fusion (`retriever.py`).
- **Why hybrid:** pure vector search failed on questions naming a specific paper. All
  four documents are about floorplans, so they are near-identical in meaning space, and
  the only distinguishing token is a proper noun — exactly what embeddings compress
  away. Measured: the chunk answering "how many floor plans are in FloorPlanCAD" ranked
  #13; the chunk listing that paper's three contributions ranked #250 by vector and #1
  by keyword. Keyword search matches exact tokens, so it covers precisely that gap.
- **Reserved slots:** plain RRF rewards agreement between methods, which buries a chunk
  only one method can find. The contributions chunk (keyword #1, vector #250) lost to
  chunks both methods ranked moderately. Each method now gets 2 guaranteed slots before
  RRF fills the rest.
- **Also needed, to make keyword search work at all:** title prefixing at embed time and
  hyphenation repair — see Chunking and Open Questions.
- **Reranking (added Day 13):** hybrid search now returns 20 candidates, and a
  cross-encoder rescores them down to the final 6.
  - **Why a cross-encoder:** the embedding model is a *bi-encoder* — question and chunk
    are encoded separately into fixed vectors and then compared, so the model never sees
    the two together. A cross-encoder reads question and chunk jointly and scores how
    well that chunk answers that question. Much more accurate, and far too slow to run
    over the whole corpus, which is why it only reorders candidates retrieval has already
    narrowed down.
  - **Why it was the right next step:** measured, not guessed. hit@10 was 100% while
    hit@6 was 94% and MRR 0.677 — the correct chunk was nearly always being found but not
    ranked highly. That gap is exactly what reranking addresses.
  - **Model:** `ms-marco-MiniLM-L-12-v2` via flashrank — 21.6 MB, runs on onnxruntime,
    no PyTorch. Chosen to keep the stack local and light.
  - **Measured effect:**

    | metric | hybrid only | + reranker |
    |---|---|---|
    | hit@1 | 50% | 62% |
    | hit@3 | 81% | 94% |
    | **hit@6** | 94% | **100%** |
    | MRR | 0.677 | **0.783** |

  - **End to end:** two previously-broken questions now pass. D13 ("Who made
    FloorPlanCAD?") returns the correct authors and affiliations — the failure previously
    recorded as structural and needing a metadata lookup. D12 now gives the full answer
    (density map of the input point cloud) rather than just "density image".
- **Relevance threshold (added Day 13):** chunks scoring below 0.05 after reranking are
  dropped instead of being sent to the model, and if nothing survives the model is not
  called at all — the refusal is returned directly.
  - **Why a threshold is now possible:** reranker scores are comparable across questions
    in a way RRF scores and cosine distances are not. Measured across a deliberately
    mixed corpus (floorplan papers, a Wikipedia article, a travel company page): relevant
    chunks score 0.85-1.00, irrelevant chunks score **exactly 0.0000**. The cross-encoder
    is decisive about irrelevance, so the gap is enormous and the exact cutoff barely
    matters.
  - **Why it matters:** retrieval always returns k results whether or not anything is
    relevant. On a single-topic corpus that is harmless; on a mixed one it means a
    question about one subject arrives at the model padded with confidently-irrelevant
    chunks from another.
  - **Effect:** "What is the best pizza topping?" now returns 0 chunks and answers in
    1.0s without calling the model, versus ~6s. No model call means no possibility of
    hallucination on out-of-scope questions.
  - **No cost:** retrieval metrics were identical before and after (hit@6 100%, MRR 0.783).
- **Source filtering:** `search(..., sources=[...])` restricts which documents are
  searched, exposed as a "Search in" multiselect in the UI. The threshold handles clear-cut
  irrelevance automatically; this covers ambiguous questions the reranker cannot resolve —
  a vague query on a mixed corpus that legitimately matches the wrong topic.
- **Top-k value used:** 6 (was 4 with pure vector search)
- **Why:** Enough that an answer needing facts from two different pages can still be
  assembled, without padding the prompt with mostly-irrelevant text. Not tuned yet —
  Day 13 evaluation should test this properly.
- **Observed (pre-hybrid, vector distances):** for a well-covered question ~0.60-0.68. For a question
  the papers don't address at all ("What is the capital of France?") they jumped to
  ~1.12-1.17. Chroma always returns k results regardless — it has no notion of "no
  match" — but that distance gap is a usable signal. A threshold around 1.0 could skip
  the LLM call entirely for out-of-scope questions.

## LLM
- **Model used (Ollama model name, or swapped to API):** `llama3.1:8b` via Ollama (4.9 GB)
- **Why:** Runs locally alongside the embedding model, so no API key and no per-query
  cost while iterating. 8B is small enough to respond quickly on this machine and has
  proved able to follow the grounding instructions reliably.
- **Prompt rules (each added in response to an observed failure):** answer only from the
  numbered sources; no outside knowledge; an exact refusal string; cite by number;
  answer only what was asked; own words, no pasted extracts; state a shared fact once;
  never calculate or infer an unstated number; when sources give conflicting figures,
  pick the one that answers the question rather than reconciling them.
  The last two exist because the model invented arithmetic ("2,500 per project times
  over 100 projects") to reconcile two figures it had been given.
- **Prompt tuning has hit its limit.** Iterating the prompt improved answer quality
  (shorter, no pasted quotes, no narrated reasoning, no fabricated arithmetic) but did
  not change the 12/15 pass rate. Each further change traded one regression for another.
  The remaining failures look like model capability, not wording.
- **Grounding approach:** the system prompt restricts the model to the numbered sources
  supplied in the user message, forbids outside knowledge, requires citing source
  numbers, and gives an explicit refusal string for questions the sources don't cover.
  The escape hatch matters — without a sanctioned way to decline, a model will strain
  to answer from training data instead.
- **Verified:** asked "What is the capital of France?" — a fact the model certainly
  knows — and it correctly answered "That isn't covered in these documents." Grounding
  is holding, not just appearing to.

## Source Citation
- **How citations are tracked (page/chunk metadata approach):** Built. `PyPDFLoader`
  attaches per-page metadata; chunking never merges across pages, so every chunk keeps
  the page it came from. `build_index.clean_metadata()` stores `source` (bare filename),
  `title`, `page` (0-indexed) and `page_label` (printed number) on each chunk.
  Retrieved chunks are numbered `[1]`, `[2]`... in the prompt with their title and page,
  the model is told to cite by number, and every surface — CLI, API, UI — shows the same
  numbering alongside the chunk text so a reader can check the citation directly.
- **Displayed page number:** `page_label`, the number printed on the page, because that
  is what a reader looking at the PDF will see.
- **Known issues (e.g. hallucinated citations, mismatches):**
  - `page` and `page_label` differ by one. Page one of a PDF is `page: 0` /
    `page_label: '1'`. Mixing them up gives citations that are silently off by one.
  - `source` is an absolute local path, which will be meaningless once deployed.
    Needs to become just the filename before Week 3.

## API Layer
- **Run with:** `./.venv/bin/uvicorn api:app --reload`
- **Endpoints:**
  - `GET /health` — liveness, plus the number of chunks currently indexed and the model
    in use. Returns 503 if the retriever failed to load.
  - `POST /ask` — `{"question": "..."}` in; question, answer, model, and a list of
    sources out. Each source carries its number (matching the `[1]` citations in the
    answer), paper title, filename, printed page number, and a 300-character snippet so
    a caller can verify the citation without opening the PDF.
- **Retriever is built once at startup**, in a FastAPI `lifespan` handler, not per
  request. Constructing a `HybridRetriever` reads all 303 chunks out of Chroma and
  computes BM25 statistics over the whole corpus — acceptable once, wasteful per call.
- **Error cases handled:**
  | Case | Status | Handled by |
  |---|---|---|
  | Empty question `""` | 422 | Pydantic `min_length=1` |
  | Whitespace-only question | 422 | Explicit check — `min_length` does not catch it |
  | Missing `question` field | 422 | Pydantic |
  | Malformed JSON body | 422 | Pydantic |
  | Question over 1000 chars | 422 | Pydantic `max_length` |
  | GET on `/ask` | 405 | FastAPI routing |
  | Question the papers don't answer | 200 | Model returns the refusal string — not an error |
  | Ollama down / model missing | 503 | `try/except` around generation |
- **Untested:** the 503 path for Ollama being unreachable. The handler exists but has not
  been exercised against a stopped Ollama.
- **Note:** an unanswerable question is deliberately a 200, not an error. "The documents
  don't cover this" is a successful, correct response.

## Evaluation
- **Eval set size:** 15 questions (`EVAL_QUESTIONS.md`), across five categories: specific
  numbers, correct-source attribution, multi-chunk synthesis, paraphrased wording, and
  refusal of questions the documents don't answer.
- **Metric(s) used:** manual PASS / PARTIAL / FAIL per question, plus retrieval rank of
  the known-correct chunk when a question fails. `run_eval.py` runs the set; the flags it
  prints narrow down what to read, they do not decide.
- **Results (Day 6):** 12 PASS, 2 PARTIAL, 1 FAIL.
  - PARTIAL — A3: gives the correct "over 15,000" but leads with an unrelated 2,500 from
    another page. The paper contains both figures in different contexts.
  - PARTIAL — D12: answers "density image", which is the network's input; the paper's
    pipeline starts from a 3D point cloud projected into that density map.
  - FAIL — D13 "Who made FloorPlanCAD?": a metadata question. The affiliation chunk
    ranks keyword #11 / vector #57, so neither retriever reaches it.
- **Caveat:** the expected answers were drawn from the extracted text, which is the same
  text the system reads. Not independent ground truth. Several should be checked against
  the original PDFs.
- **Retrieval measured separately (`eval_retrieval.py`).** The generation stage can only
  be as good as what reaches it, so retrieval is scored on its own against 16 questions
  labelled with the document and page(s) that contain the answer. E14/E15 are excluded —
  they have no correct chunk to retrieve, so retrieval metrics are meaningless for them.

  | k | hit@k | doc@k | MRR |
  |---|---|---|---|
  | 1 | 50% | 100% | 0.500 |
  | 3 | 81% | 100% | 0.646 |
  | **6 (current)** | **94%** | **100%** | **0.677** |
  | 10 | 100% | 100% | 0.685 |

  - `hit@k` — a chunk from the right file and right page appears in the top k.
  - `doc@k` — the right paper appears at all, ignoring which page.
  - `MRR` — mean reciprocal rank of the first correct chunk; 1.0 would mean always first.

  **doc@k is 100% at every k, including k=1** — the correct paper is always retrieved.
  That is the title-prefixing fix validated: the original failure was questions about
  FloorPlanCAD returning CubiCasa chunks.

  **hit@10 is 100%**, so every answer is reachable and the remaining problem is ranking,
  not coverage. The single miss at k=6 is F16, the "why" question.

  **Caveat — this is slightly optimistic.** Ground truth is labelled per page, and a page
  holds several chunks. D13 counts as a hit whenever any page-1 chunk is retrieved, even
  though the chunk with the author affiliations is not among them (keyword #11, vector
  #57). Chunk-level labels would be more accurate and much more laborious.

- **RAGAS not used.** The metrics here are computed against hand-labelled ground truth,
  which is more informative for this project than a framework score, and avoids adding an
  LLM-judge dependency to a deliberately local-only stack.

## Frontend
- **Run with:** `./.venv/bin/streamlit run app.py`
- **Built with:** Streamlit (`app.py`) — question box, answer, and every source in an
  expander showing the paper title and the exact chunk text the model was given.
  Sidebar lists the indexed documents and allows uploading a new PDF.
- **Imports the pipeline directly rather than calling the FastAPI service.** One process,
  no network hop, which is what Hugging Face Spaces expects. `api.py` stays as the
  programmatic interface; it is not what the UI talks to.
- **Retriever is cached with `st.cache_resource`.** Streamlit re-runs the whole script on
  every interaction, so without caching the corpus and BM25 statistics would be rebuilt
  on each rerun. The cache is explicitly cleared after an upload, since the corpus changed.
- **Upload handling:** `build_index.index_pdf()` is shared between the batch build and the
  upload handler, so a runtime upload is processed identically to a pre-indexed file.
  Re-indexing the same filename deletes its existing chunks first — verified: re-indexing
  a file left the count at 303 with 43 unique ids, no duplicates.
- **Untested:** uploading an actual scanned PDF. The handler reports 0 chunks and explains
  that OCR would be needed, but has not been exercised against a real scan.

## Deployment
- **Platform:**
- **Known limitations of the deployed version:**

---

## Open Questions / Things to Revisit
(running list of "I did this but I'm not sure it's the right way" items)
- **"Why" questions retrieve the wrong section, and it is structural.** Asking why a
  mechanism exists retrieves the results and ablation text rather than the motivation.
  The motivation chunk describes the *problem* in the problem's vocabulary and never
  names the mechanism, while the ablation chunks are full of the mechanism's name — so a
  question naming the mechanism matches the results section. Measured on "Why does
  RoomFormer use two-level queries?": the chunk stating the real reason (a floorplan has
  a variable number of rooms, each with a variable number of corners) ranks keyword #9
  and vector #27, outside the top 6 for both, so hybrid retrieval does not rescue it.
  **Update (Day 13):** reranking fixed the retrieval half — the page-2 chunk stating the
  real reason now reaches the prompt at position 5, and hit@6 is 100%. Generation still
  answers from the page-8 ablation text instead. So this is no longer a retrieval failure
  but a generation one: the model prefers text that repeats the question's vocabulary over
  text that answers it.
- **C9 regressed when reranking was added.** It now lists the third FloorPlanCAD
  contribution as "fine-grained annotations for 35 object categories" instead of the
  CNN-GCN baseline. Better retrieval, worse answer — the reranker changed which chunks
  reach the prompt, and the model picked wrongly among them.
- **Generation is now the weak link, not retrieval.** On "How many floor plans are in
  the FloorPlanCAD dataset?", the correct chunk *is* retrieved as source [4] and reads
  "containing over 15,000 floor plans, ranging from residential to commercial
  buildings". llama3.1:8b quoted the second half of that sentence, stated the sources
  gave no total, and reported 2,500 from a different page. It had the answer and
  misread it. Worth testing with fewer chunks in the prompt, a sharper prompt, or a
  larger model.
- **Figure and table text pollutes chunks.** (Still open — the `<latexit>` fix below
  removed the worst of it, but figure *labels* remain.)
- **PDF hyphenation was silently breaking keyword search.** Justified text splits words
  across lines ("FloorPlan-\nCAD", "con-\ntaining") — 568 occurrences across the four
  papers. Every one was a token no search could match. Now repaired in
  `ingest.clean_text()`, and `retriever.tokenize()` additionally strips hyphens so
  "FloorPlan-CAD" and "FloorPlanCAD" match.
- **Retrieval settings are tuned against 5 questions.** That is not enough to trust.
  Day 13's full eval set should re-check top-k, reserved slots, and RRF_K. Diagram labels extracted from figures
  (e.g. `[D6]ConvTranspose(512,512,2,2)` strings from an architecture diagram) end up
  as chunk text. On a test query this noise chunk ranked 2nd, at distance 0.649 vs
  0.609 for the genuinely relevant chunk — close enough to displace a real result.
  No filtering in place yet.
- ~~`2211.15658v2.pdf` produces 165 chunks from 15 pages~~ **SOLVED (Day 5).** The
  paper embeds the LaTeX source of every equation as hidden text
  (`<latexit sha1_base64="...">...</latexit>`), invisible to a reader but extracted by
  pypdf. It was 44% of that document's characters and 57% of its chunks. Now stripped
  in `ingest.clean_text()`; the file dropped from 165 to 78 chunks, the index from 392
  to 305, and retrieval distances on that paper improved from ~0.60 to ~0.42-0.54.
- **14 of 305 chunks are still exact duplicates** (~5%), a side effect of the 100-char
  overlap. Small enough to leave alone for now.
- **Two-column layout** — only verified that page 1 of one paper extracts in a sensible
  reading order. Body pages not checked.
- **`langchain-community` is deprecated** (`PyPDFLoader` is moving to a standalone
  package). Works now; should be migrated before deployment.
