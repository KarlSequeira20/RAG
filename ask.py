"""Ask a question and get an answer grounded in the indexed PDFs.

This is the full RAG loop: embed the question, retrieve the nearest chunks,
put those chunks in a prompt, and let the LLM answer from them.
"""

import sys
from pathlib import Path

import ollama

from retriever import HybridRetriever

LLM_MODEL = "llama3.1:8b"
# Wider than the 4 used with pure vector search. Hybrid merges two rankings, so the
# correct chunk is often present but a place or two lower than a neighbouring one.
TOP_K = 6

# Returned without calling the model when nothing relevant was retrieved. Kept identical
# to the refusal string in the prompt so callers see one consistent answer either way.
NO_ANSWER = "That isn't covered in these documents."

# The instruction that keeps answers grounded. Two things matter here:
# the model is told to use only the supplied context, and it is given an
# explicit way out. Without that escape hatch a model will strain to produce
# an answer from its own training data rather than admit the text is missing.
SYSTEM_PROMPT = """You answer questions about a set of research papers.

Rules:
- Answer using ONLY the numbered sources provided in the user's message.
- Do not use any outside knowledge, even if you are confident it is correct.
- If the sources do not contain the answer, reply with exactly: "That isn't covered in these documents." Nothing else.
- Cite the sources you used by their number, like [1] or [2].
- Answer only what was asked. Do not add related figures, statistics or background that
  the question did not ask for, even when a source mentions them.
- Write the answer in your own words as running prose. Never paste or list source
  extracts, and never quote a source at length.
- If several sources agree, answer once and cite them together. Do not restate the same
  fact per source.
- Report only figures stated verbatim in a source. Never calculate, combine, estimate or
  infer a number that is not written down, and never multiply or add figures together.
- If sources give different figures, state the one that answers the question and ignore
  the rest rather than trying to reconcile them.
- Give the answer directly. Do not describe your reasoning, weigh the sources out loud,
  or explain which source you chose and why.
- Different sources may discuss different papers. Use only the ones about the paper the
  question names.
- Be concise: one or two sentences unless the question asks for a list."""


def describe_location(meta):
    """Human-readable pointer to where a chunk came from.

    PDFs cite a page number; web pages have none, so their chunks are cited as parts.
    Chunks indexed before web support existed have no `kind`, hence the default.
    """
    if meta.get("kind") == "web":
        return f"{meta['source']}, part {meta['page_label']}"
    return f"{meta['source']}, page {meta['page_label']}"


def build_prompt(question, hits):
    """Lay the retrieved chunks out as numbered sources, then ask the question."""
    blocks = []
    for i, (text, meta, _score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({meta['title']} -- {describe_location(meta)})\n{text}")
    context = "\n\n".join(blocks)
    return f"Sources:\n\n{context}\n\nQuestion: {question}"


def answer(question, retriever=None, sources=None):
    """Answer a question. `sources` optionally restricts which documents to search."""
    retriever = retriever or HybridRetriever()
    hits = retriever.search(question, k=TOP_K, sources=sources)

    # Nothing cleared the relevance floor, so there is nothing to answer from. Skipping
    # the model here is both faster and safer -- with no relevant text in the prompt,
    # asking it to answer only invites it to fall back on training data.
    if not hits:
        return NO_ANSWER, []

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
    )
    return response["message"]["content"], hits


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python ask.py "your question here"')
        return

    text, hits = answer(question)

    print(f"\nQ: {question}\n")
    print(text)
    if not hits:
        print("\n(No chunk cleared the relevance threshold, so the model was not called.)")
        return
    print("\nSources retrieved:")
    for i, (_text, meta, score) in enumerate(hits, start=1):
        print(f"  [{i}] {describe_location(meta)}  (score {score:.4f})")


if __name__ == "__main__":
    main()
