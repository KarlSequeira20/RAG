"""Measure retrieval quality as numbers, not per-question verdicts.

`run_eval.py` answers whether the whole system got a question right. This measures the
retrieval stage on its own, which is the part worth tuning: if the correct chunk never
reaches the prompt, no amount of prompt work can fix the answer.

Ground truth here is the document and page(s) that actually contain the answer, taken
from reading the papers. A question is counted as retrieved correctly when a chunk from
the right file and one of the right pages appears in the top k.

Metrics:
  hit@k   -- fraction of questions where a correct chunk appears in the top k.
             The number that matters most: it is the ceiling on how well the system
             can possibly answer.
  MRR     -- mean reciprocal rank of the first correct chunk (1.0 = always ranked
             first, 0.5 = typically second). Rewards ranking the right chunk higher,
             which hit@k alone cannot see.
  doc@k   -- fraction where at least the right *paper* appears, ignoring the page.
             Separates "retrieved the wrong document" from "right document, wrong part".

Known limitation: ground truth is labelled per page, and a page holds several chunks.
D13 ("Who made FloorPlanCAD?") counts as a hit whenever any page-1 chunk is retrieved,
even though the chunk actually containing the author affiliations is not among them --
it ranks keyword #11 / vector #57. So hit@k is slightly optimistic. Labelling the exact
chunk per question would be more accurate and considerably more work.
"""

from retriever import HybridRetriever

CUBI, FPCAD, ROOM, PATCH = (
    "1904.01920v1.pdf",
    "2105.07147v2.pdf",
    "2211.15658v2.pdf",
    "ijgi-10-00828-v2.pdf",
)

# (id, question, expected file, expected pages). Pages are the printed page numbers.
# Several are listed where a fact appears more than once; any of them counts.
GROUND_TRUTH = [
    ("A1", "How many samples does the CubiCasa5K dataset contain?", CUBI, {"1", "2", "4"}),
    ("A2", "How many floorplan object categories is CubiCasa5K annotated into?", CUBI, {"1", "2"}),
    ("A3", "How many floor plans are in the FloorPlanCAD dataset?", FPCAD, {"1", "2"}),
    ("A4", "How many object categories does FloorPlanCAD provide annotations for?", FPCAD, {"1", "2", "4"}),
    ("B5", "Which datasets is RoomFormer evaluated on?", ROOM, {"1", "2", "5", "6"}),
    ("B6", "What is panoptic symbol spotting?", FPCAD, {"1", "2"}),
    ("B7", "Why does a CNN struggle with large-scale building floor plans?", PATCH, {"1", "2", "6"}),
    ("C8", "What are the two levels in RoomFormer's two-level queries?", ROOM, {"1", "2", "3", "4"}),
    ("C9", "What are the three contributions claimed by the FloorPlanCAD paper?", FPCAD, {"1", "2"}),
    ("C10", "Why does RoomFormer avoid multi-stage pipelines, and what does it use instead?", ROOM, {"1", "2"}),
    ("D11", "Is the CubiCasa5K dataset bigger than what came before it?", CUBI, {"1", "2", "4", "5"}),
    ("D12", "What kind of input does RoomFormer start from?", ROOM, {"1", "2"}),
    ("D13", "Who made FloorPlanCAD?", FPCAD, {"1"}),
    ("F16", "Why does RoomFormer use two-level queries instead of a single level?", ROOM, {"1", "2"}),
    ("F17", "What is the trade-off when choosing the patch size in the patch-based floor plan method?", PATCH, {"6", "7"}),
    ("F18", "What does the multi-task uncertainty loss do in CubiCasa5K?", CUBI, {"3", "8"}),
]

# E14 and E15 are excluded on purpose: they have no correct chunk to retrieve, so
# retrieval metrics are meaningless for them. They are refusal tests for the generator.

K_VALUES = (1, 3, 6, 10)


def evaluate(retriever, k, rerank=True):
    hits = docs = 0
    reciprocal_ranks = []
    misses = []

    for qid, question, want_file, want_pages in GROUND_TRUTH:
        results = retriever.search(question, k=k, rerank=rerank)

        rank_of_hit = None
        found_doc = False
        for rank, (_chunk, meta, _score) in enumerate(results, start=1):
            if meta["source"] == want_file:
                found_doc = True
                if meta["page_label"] in want_pages and rank_of_hit is None:
                    rank_of_hit = rank

        hits += rank_of_hit is not None
        docs += found_doc
        reciprocal_ranks.append(1 / rank_of_hit if rank_of_hit else 0.0)
        if rank_of_hit is None:
            misses.append(qid)

    n = len(GROUND_TRUTH)
    return {
        "k": k,
        "hit": hits / n,
        "doc": docs / n,
        "mrr": sum(reciprocal_ranks) / n,
        "misses": misses,
    }


def main():
    retriever = HybridRetriever()
    print(f"{len(GROUND_TRUTH)} questions with labelled ground truth "
          f"(E14/E15 excluded -- nothing to retrieve)\n")
    for rerank in (False, True):
        print(f"--- {'hybrid + cross-encoder rerank' if rerank else 'hybrid only'} ---")
        print(f"{'k':>3}  {'hit@k':>7}  {'doc@k':>7}  {'MRR':>6}   misses")
        print("-" * 58)
        for k in K_VALUES:
            r = evaluate(retriever, k, rerank=rerank)
            print(f"{r['k']:>3}  {r['hit']:>6.0%}  {r['doc']:>7.0%}  {r['mrr']:>6.3f}   "
                  f"{', '.join(r['misses']) or '-'}")
        print()


if __name__ == "__main__":
    main()
