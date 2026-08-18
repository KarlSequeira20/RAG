"""Run the questions in EVAL_QUESTIONS.md and print each answer with its sources.

The `expect` field is a hint for grading, not an automatic pass. Whether an answer is
correct still needs reading -- an answer can contain the right number and still be
wrong about what the number means.
"""

from ask import TOP_K, answer
from retriever import HybridRetriever

REFUSAL = "isn't covered in these documents"

QUESTIONS = [
    ("A1", "How many samples does the CubiCasa5K dataset contain?", "5000", "1904"),
    ("A2", "How many floorplan object categories is CubiCasa5K annotated into?", "80", "1904"),
    ("A3", "How many floor plans are in the FloorPlanCAD dataset?", "15,000", "2105"),
    ("A4", "How many object categories does FloorPlanCAD provide annotations for?", "35", "2105"),
    ("B5", "Which datasets is RoomFormer evaluated on?", "SceneCAD", "2211"),
    ("B6", "What is panoptic symbol spotting?", "stuff", "2105"),
    ("B7", "Why does a CNN struggle with large-scale building floor plans?", "uniform", "ijgi"),
    ("C8", "What are the two levels in RoomFormer's two-level queries?", "corner", "2211"),
    ("C9", "What are the three contributions claimed by the FloorPlanCAD paper?", "CNN-GCN", "2105"),
    ("C10", "Why does RoomFormer avoid multi-stage pipelines, and what does it use instead?", "single-stage", "2211"),
    ("D11", "Is the CubiCasa5K dataset bigger than what came before it?", "larg", "1904"),
    ("D12", "What kind of input does RoomFormer start from?", "point cloud", "2211"),
    ("D13", "Who made FloorPlanCAD?", "Alibaba", "2105"),
    ("E14", "How much does it cost to license the CubiCasa5K dataset commercially?", REFUSAL, None),
    ("E15", "What is the architecture of the BERT language model?", REFUSAL, None),
]


def main():
    retriever = HybridRetriever()
    for qid, question, expect, paper in QUESTIONS:
        text, hits = answer(question, retriever=retriever)
        sources = sorted({f"{m['source'][:8]} p{m['page_label']}" for _t, m, _s in hits})

        refused = REFUSAL in text
        should_refuse = expect == REFUSAL
        hit_expected = expect.lower() in text.lower()
        right_paper = paper is None or any(paper in s for s in sources)

        if should_refuse:
            flag = "PASS" if refused else "FAIL"
        elif refused:
            flag = "FAIL"          # refused a question the documents do answer
        elif hit_expected and right_paper:
            flag = "CHECK"         # looks right, still needs reading
        else:
            flag = "SUSPECT"

        print(f"\n{'='*78}\n[{qid}] {flag}  (expect: {expect!r})\nQ: {question}\n")
        print(text.strip())
        print(f"\nsources (top {TOP_K}): {', '.join(sources)}")


if __name__ == "__main__":
    main()
