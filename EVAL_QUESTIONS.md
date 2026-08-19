# Eval questions — Day 5-7 verification

15 questions to run through `ask.py`. For each one: run it, then **open the cited page
in the actual PDF** and confirm the answer is really there.

The "expected" column is drawn from the extracted text, mostly the first two pages of
each paper. That's the same text the RAG reads, so it is *not* independent ground
truth — if extraction mangled something, both the system and this file would be wrong
in the same way. Spot-check several against the real PDF before trusting the set.

Mark each row PASS / FAIL / PARTIAL as you go.

## Results — 2026-08-15 (Day 6), run via `run_eval.py`

**12 PASS, 2 PARTIAL, 1 FAIL** on questions 1-15; **2 PASS, 1 FAIL** on the "why"
questions added later (section F). Config: hybrid BM25 + vector retrieval, RRF with 2
reserved slots per method, top-k 6, llama3.1:8b.

| # | Verdict | Note |
|---|---|---|
| 1, 2, 4 | PASS | Correct figure, correct paper |
| 3 | PARTIAL | Gets "over 15,000" right, then muddies it with "2,500 files" and "100 projects" pulled from another page |
| 5, 6 | PASS | |
| 7 | PASS | Correct; only flagged by the script because the `expect` keyword was poorly chosen |
| 8, 9, 10 | PASS | 9 was a total failure before hybrid retrieval |
| 11 | PASS | |
| 12 | PARTIAL | Answers "a density image" — the network's input, but the paper's pipeline starts from a 3D point cloud projected into that density map |
| 13 | FAIL | Refused. See below |
| 14, 15 | PASS | Both refusals correct |

### Q13 — the one outright failure

"Who made FloorPlanCAD?" The affiliation chunk ranks **keyword #11, vector #57**, so it
never reaches the top 6. Neither method can find it: "Alibaba" does not appear in the
question, and "who made" has no semantic overlap with an author byline.

This is a **metadata question**, not a content question. Who wrote it, when it was
published, how long it is — the answers live in headers, not prose, so similarity search
over chunks is the wrong tool by construction. The fix is to answer from stored metadata
(the `title` field already works this way) rather than to tune retrieval further.

Also visible in Q13: the model narrated its own indecision ("However, it seems more
likely to look at [2]...") before refusing. Reasoning leaking into user-facing output.

### What this run says about overfitting

Retrieval settings had been tuned against 5 questions. On the full 15 they held up —
no regressions, and the two previously-failing questions now pass. Not proof the
settings are optimal, but they are not obviously overfitted either.

## The papers

| Short name | File | Topic |
|---|---|---|
| CubiCasa5K | `1904.01920v1.pdf` | Floorplan image dataset, 5000 samples, multi-task CNN |
| FloorPlanCAD | `2105.07147v2.pdf` | CAD drawing dataset, panoptic symbol spotting, CNN-GCN |
| RoomFormer | `2211.15658v2.pdf` | Floorplan reconstruction from 3D scans, Transformer |
| Patch-based | `ijgi-10-00828-v2.pdf` | CNN on patches, large-scale complex buildings |

---

## A. Specific numbers — easiest place to catch a fabrication

| # | Question | Expected | Expected source | Result |
|---|---|---|---|---|
| 1 | How many samples does the CubiCasa5K dataset contain? | 5000 | CubiCasa5K, p1 | |
| 2 | How many floorplan object categories is CubiCasa5K annotated into? | Over 80 | CubiCasa5K, p1 | |
| 3 | How many floor plans are in the FloorPlanCAD dataset? | Over 15,000 | FloorPlanCAD, p1 | |
| 4 | How many object categories does FloorPlanCAD provide annotations for? | 35 | FloorPlanCAD, p1 | |

**Watch for:** the number 15,000 appears in *both* papers for different reasons —
FloorPlanCAD has 15,000+ floor plans, and CubiCasa5K was collected from a larger set of
15,000 mostly Finnish images. Q3 is a deliberate trap. If the answer cites CubiCasa5K,
retrieval grabbed the wrong document.

## B. Right source, not just right answer

| # | Question | Expected | Expected source | Result |
|---|---|---|---|---|
| 5 | Which datasets is RoomFormer evaluated on? | Structured3D and SceneCAD | RoomFormer, p1 | |
| 6 | What is panoptic symbol spotting? | Spotting instances of countable "things" and the semantics of uncountable "stuff" as one recognition task | FloorPlanCAD, p1-2 | |
| 7 | Why does a CNN struggle with large-scale building floor plans? | CNNs need uniform input size; large plans vary in scale and resolution — hence splitting into regular patches | Patch-based, p2 | |

**Watch for:** all four papers are about floorplans. These test whether retrieval picks
the *right* paper, not just an on-topic one. Check the cited filename, not only the page.

## C. Needs more than one chunk

| # | Question | Expected | Expected source | Result |
|---|---|---|---|---|
| 8 | What are the two levels in RoomFormer's two-level queries? | One level for room polygons, one for their corners | RoomFormer, p1-2 | |
| 9 | What are the three contributions claimed by the FloorPlanCAD paper? | The dataset; the panoptic symbol spotting task; the CNN-GCN baseline | FloorPlanCAD, p1 | |
| 10 | Why does RoomFormer avoid multi-stage pipelines, and what does it use instead? | Prior work uses heuristic multi-stage pipelines; RoomFormer is single-stage structured prediction with a Transformer, end-to-end trainable | RoomFormer, p1-2 | |

**Watch for:** partial answers. If it gets 2 of the 3 contributions in Q9, top-k may be
too small, or one chunk boundary split the list.

## D. Phrased differently than the paper phrases it

| # | Question | Expected | Expected source | Result |
|---|---|---|---|---|
| 11 | Is the CubiCasa5K dataset bigger than what came before it? | Yes — over five times larger than the previously largest dataset | CubiCasa5K, p2 | |
| 12 | What kind of input does RoomFormer start from? | A 3D point cloud, projected along the gravity axis into a 2D density map | RoomFormer, p1 | |
| 13 | Who made FloorPlanCAD? | Alibaba Group and Simon Fraser University | FloorPlanCAD, p1 | |

**Watch for:** Q11 uses "bigger" where the paper says "five times larger"; Q12 says
"input" where the paper says "we start from". This is what embeddings are supposed to
handle. If these fail, the failure is in retrieval, not the LLM.

## E. Not in the documents — refusal tests

| # | Question | Expected | Result |
|---|---|---|---|
| 14 | How much does it cost to license the CubiCasa5K dataset commercially? | "That isn't covered in these documents." | |
| 15 | What is the architecture of the BERT language model? | "That isn't covered in these documents." | |

**Watch for:** Q15 is the important one — the model definitely knows BERT from training.
Answering it means grounding has failed. Q14 is harder: it's on-topic enough that
retrieval will return plausible-looking CubiCasa5K chunks, so the LLM has to notice the
chunks don't actually answer the question.

---

## F. "Why" questions — added Day 11, from independent ground truth

These came from feedback on my own reading of the papers, checked against the original
PDFs rather than the extracted text. That makes them the only questions here with
genuinely independent expected answers.

| # | Question | Expected | Source | Result |
|---|---|---|---|---|
| 16 | Why does RoomFormer use two-level queries instead of a single level? | A floorplan has a variable number of rooms, and each room a variable number of corners. Predicting doors/windows is an added capability, not the reason | RoomFormer, p1-2 | **FAIL** |
| 17 | What is the trade-off when choosing the patch size? | Smaller patch = better resolution but less context/field of view; larger patch = more context but worse resolution | Patch-based, p6-7 | PASS |
| 18 | What does the multi-task uncertainty loss do? | Automatically learns the weights for each task's loss, removing manual tuning | CubiCasa5K, p3, p8 | PASS |

### Q16 — why "why" questions are their own failure category

The chunk holding the answer (`2211.15658v2.pdf:5`, page 2) reads:

> "each floorplan has a different number of rooms (with no natural order), and each
> room polygon has a different number of (ordered) corners"

It ranks **keyword #9, vector #27** — outside the top 6 for both methods, so hybrid
retrieval cannot rescue it either.

The reason is a vocabulary mismatch, and it is structural rather than a tuning problem.
That sentence never contains the phrase "two-level queries": it describes the *problem*,
in the problem's own words. The ablation-study chunks on pages 5 and 8 are full of "two-
level queries" and "improves", so they match a question built from those words. The
system retrieves text that shares vocabulary with the question rather than text that
answers it.

This is a general property of papers: motivation appears in the abstract and
introduction, phrased as a difficulty, while the term being motivated appears later
beside results. A question naming the mechanism will always look more like the results
section than like the motivation.

Worth noting the same mistake was made reading the papers by hand — the method and
results text crowds out the plain statement of why.

**Not fixed.** Possible directions: query expansion (also search for the problem the
mechanism solves), or weighting chunks from the abstract and introduction more heavily
for "why" questions.

---

## What to record when something fails

Note which stage broke, because the fix is different for each:

- **Wrong chunks retrieved** → retrieval problem (chunk size, top-k, embeddings)
- **Right chunks, wrong answer** → generation problem (the prompt)
- **Right answer, wrong page cited** → metadata problem (`page` vs `page_label`)
- **Made something up** → grounding problem (the prompt's escape hatch isn't working)
