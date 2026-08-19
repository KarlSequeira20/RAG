# Concepts — plain-English reference

What every piece of this project actually is, in terms I'd use to explain it to
someone who's never seen it. Written as I go, so that by the time the project is
finished I can revise from this instead of re-reading code.

The last section is interview-oriented: the questions someone would actually ask
about this project, and honest answers.

---

## The one-paragraph version

RAG (Retrieval-Augmented Generation) means: before asking an LLM a question, go
find the relevant text yourself and hand it to the model along with the question.
The model answers from what you gave it instead of from its training data. That's
the whole idea. Everything else is plumbing to make "find the relevant text" fast
and accurate.

**Why it matters:** an LLM on its own can't answer questions about documents it has
never seen (your PDFs, your company's docs, anything written after its training
cutoff), and when it doesn't know something it tends to invent a confident answer.
RAG fixes both — the model is working from real text you supplied, and because you
know which text you supplied, you can cite it.

---

## The pipeline, stage by stage

Two separate phases. **Indexing** happens once, ahead of time. **Querying** happens
every time someone asks a question.

```
INDEXING (once):
  PDFs  ->  text  ->  chunks  ->  embeddings  ->  vector database

QUERYING (every question):
  question  ->  embedding  ->  find nearest chunks  ->  chunks + question -> LLM  ->  answer + citation
```

### Document loading
Extracting the text out of a PDF file. A PDF is a layout format, not a text format,
so this is messier than it sounds — text comes out in reading order *usually*, and
figure labels/headers/footers come out mixed in with the real content.

**Trap:** a scanned PDF is just photographs of pages. There is no text inside it to
extract. The loader returns empty strings and raises no error, so the failure is
silent — the pipeline "succeeds" and produces nothing. Getting text out of scans
needs OCR (optical character recognition, e.g. Tesseract), which is a separate
problem.

### Chunking
Cutting the text into smaller pieces to be searched individually.

**Why not just embed a whole page or a whole document?** Because an embedding is one
fixed-size summary of meaning. If a chunk covers five topics, its embedding is the
average of five topics — it sits in the middle of nowhere on the meaning-map and
matches every question weakly rather than any question strongly.

**The tradeoff:**
- Too small — a chunk gets separated from the context that makes it meaningful.
  "This improved accuracy by 12%" is useless without knowing what *this* is.
- Too big — the embedding gets diluted (above), and you waste LLM context sending
  mostly-irrelevant text, which also gives the model more room to go off-track.

**Overlap** repeats the last N characters of each chunk at the start of the next one.
It's insurance: if an idea gets split across a boundary, it still appears complete in
one of the two chunks.

**`RecursiveCharacterTextSplitter`** doesn't cut blindly every N characters. It tries
to break at paragraph breaks first, then line breaks, then sentences, then words —
falling back only when it has to. So `chunk_size` is a ceiling, not a target, and
real chunks come out shorter than the limit.

### Embeddings
A model that converts text into a list of numbers (a **vector**) representing its
meaning. Ours produces 768 numbers per chunk.

The useful property: **text with similar meaning produces similar numbers.** "How
many rooms does the model detect?" and "room count prediction" land near each other
even with no words in common. That's why this beats keyword search — it matches
meaning, not spelling.

Think of it as a coordinate on a giant map of meaning. Similar ideas are neighbours.

**Rule that will bite you if you forget it:** the query and the stored documents must
be embedded by the *same model*. Different models produce coordinates on completely
different maps, so comparing across them gives nonsense. Changing the embedding model
means rebuilding the entire index.

### Vector database
A database built for one specific question: *"here is a vector — give me the stored
vectors closest to it."* Doing that by brute force means comparing against every
chunk you have; vector DBs use indexes to make it fast at scale.

It also stores, alongside each vector, the original text and any **metadata** you
attach (filename, page number). The metadata is what makes citation possible.

### Similarity / distance
How "close" two vectors are. We use **cosine distance**: lower means more similar,
roughly 0 = near-identical, roughly 1 = unrelated.

It is **not** a confidence score or a percentage. A distance of 0.6 can be an
excellent match. The numbers are only meaningful compared against each other, within
the same model.

### Top-k retrieval
"Give me the k closest chunks." k=3 means the three best matches get sent to the LLM.

Small k risks missing the chunk that held the answer. Large k pads the prompt with
irrelevant text, which costs money/time and can distract the model. Something to tune
with real evaluation numbers rather than guesswork.

### Reranking
Retrieval has to be fast, because it compares against every chunk you have. That speed
comes from a **bi-encoder**: the question and each chunk are turned into vectors
*separately*, then compared. The model never sees a question and a chunk side by side, so
it cannot notice that one specifically answers the other.

A **cross-encoder** does see them together — it takes the question and one chunk as a
single input and outputs a relevance score. Much more accurate, and far too slow to run
over a whole corpus.

So you use both: fast retrieval narrows thousands of chunks to ~20 candidates, then the
cross-encoder rescores just those and keeps the best handful. This is standard in
production RAG, and the measurable sign you need it is a gap between hit@10 and hit@6 —
the right chunk is being found, just not ranked highly enough to reach the prompt.

### Generation
Putting the retrieved chunks and the question into a prompt and asking the LLM to
answer. The key instruction is *answer only from the provided text, and say you don't
know if it isn't there* — otherwise the model quietly falls back on its training data
and you've defeated the point of RAG while everything still looks fine.

### Citation
Because each chunk carries metadata saying which file and page it came from, an
answer can be shown with its sources. This is the difference between "trust me" and
"check page 4 yourself" — and it's the honest way to handle the fact that LLMs
sometimes make things up.

---

## Tools used, and what each one is for

| Thing | What it is | Why it's here |
|---|---|---|
| **LangChain** | A library of ready-made building blocks for LLM apps (loaders, splitters, chains) | Saves writing PDF parsing and text splitting from scratch |
| **`Document`** | LangChain's container: `.page_content` (text) + `.metadata` (dict) | The same object flows through every stage, which is what keeps page numbers attached from PDF all the way to citation |
| **pypdf / `PyPDFLoader`** | Extracts text from PDFs, one `Document` per page | Step one of everything |
| **Ollama** | Runs LLMs locally on your own machine | No API key, no per-token cost, data never leaves the laptop |
| **`nomic-embed-text`** | The embedding model, via Ollama, 768 dimensions | Turns text into meaning-vectors |
| **Chroma** | Vector database, runs in-process, saves to a local folder | Stores and searches the vectors; nothing to deploy or operate |
| **BM25** (`rank_bm25`) | Classic keyword search — scores exact word overlap | Finds proper nouns and numbers, which embeddings compress away |
| **flashrank** | Cross-encoder reranker, 21.6 MB, runs on onnxruntime | Rescores the top candidates jointly against the question; no PyTorch needed |
| **FastAPI** *(Week 2)* | Turns Python functions into a web API | Lets a frontend — or anyone — call the pipeline over HTTP |
| **Streamlit** *(Week 2)* | Builds a simple web UI from a Python script | A usable interface without writing frontend code |

---

## Things that go wrong (and how to recognise them)

| Symptom | Likely cause |
|---|---|
| Index builds, but every answer is empty or nonsense | PDFs are scans — no text was ever extracted. Check for blank pages at load time |
| Retrieval returns unrelated chunks | Query and documents embedded with different models, or the index wasn't rebuilt after a settings change |
| Answers are right but not actually in your documents | The LLM is falling back on training data. The prompt isn't constraining it hard enough |
| Answer is correct, cited page is wrong | Mixing up `page` (0-indexed) and `page_label` (printed number). Off by one |
| Retrieved chunk is on-topic but the answer is incomplete | Chunk too small — the supporting context was cut into a different chunk |
| Junk chunks outranking real content | Figure labels, tables, headers extracted as text. Needs filtering before chunking |
| Duplicate results | Index built more than once without clearing the old collection first |

---

## Questions I should be able to answer about this project

**Why RAG instead of just fine-tuning a model on the documents?**
RAG is cheaper, updates instantly when documents change (just re-index), and can cite
sources. Fine-tuning teaches a model *style and behaviour*, not reliable recall of
facts — and a fine-tuned model still can't tell you where an answer came from.

**Why not just paste the whole document into the prompt?**
Works for one short document. Breaks on 60 pages: context limits, cost scaling with
every query, and models genuinely getting worse at finding details buried in very long
inputs. Retrieval sends only the relevant few thousand characters.

**How did you pick your chunk size?**
Tested 200/500/1000/2000 on real documents and read the output. 200 produced fragments
starting mid-sentence with no subject. Settled on 1000/100 as a starting point, with
evaluation planned to test it properly rather than trusting the judgement call.

**Why this embedding model?**
Runs locally through Ollama, no API key or per-call cost, and keeps the whole stack on
one runtime alongside the LLM. Better quality than the bundled default, at the cost of
a 274MB download.

**Why Chroma over Pinecone/Weaviate/FAISS?**
Runs in-process and persists to a local folder — nothing to operate or deploy
alongside the app. The hosted options solve scale problems (millions of vectors,
multiple users) this project doesn't have. Wrong tool for the size.

**How do you know it actually works?**
Two levels of measurement. End to end: an 18-question eval set covering six failure
categories — specific numbers, correct-source
attribution, multi-chunk synthesis, paraphrased wording, and refusing questions the
documents don't answer. Current score: 12 PASS, 2 PARTIAL, 1 FAIL. When a question fails
I check where the known-correct chunk actually ranked, because rank #11 and rank #250 are
different bugs with different fixes.

Retrieval is also scored on its own, against questions labelled with the document and page
holding the answer: hit@6 100%, doc@k 100% at every k, MRR 0.783. Before adding a
cross-encoder reranker those were 94% and 0.677 — the reranker was chosen because hit@10
was already 100% while hit@6 was not, which says the right chunk was being found but not
ranked highly.
Measuring retrieval separately matters because generation can only be as good as what
reaches it — hit@10 being 100% says every answer is reachable and the remaining problem is
ranking, not coverage.

**Tell me about a bug you found.**
Roughly a quarter of the index was garbage and nothing in the output looked wrong. One
paper embedded the LaTeX source of every equation as hidden text — invisible to a reader,
extracted by pypdf — making up 44% of that document's characters. Found it by comparing
chunk counts per file, not by reading answers. Separately, 568 words were split across
line breaks by PDF hyphenation, so keyword search could never match them: the token
"floorplancad" did not exist anywhere in the text.

**Why hybrid retrieval instead of just vector search?**
All four documents are about floorplans, so they sit almost on top of each other in
meaning space, and the only thing distinguishing them is a proper noun — exactly what
embeddings compress away. Measured it: the chunk listing one paper's contributions ranked
#250 by vector search and #1 by keyword search. Adding BM25 alongside the vector search
turned two failures into passes.

**What are the weaknesses of what you built?**
The honest list lives in the Open Questions section of `ARCHITECTURE.md`. Being able
to name real limitations is worth more than claiming there aren't any.
