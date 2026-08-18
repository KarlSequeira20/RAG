"""Ask a question and get an answer grounded in the indexed PDFs.

This is the full RAG loop: embed the question, retrieve the nearest chunks,
put those chunks in a prompt, and let the LLM answer from them.
"""

import sys
from pathlib import Path

import chromadb
import ollama

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "papers"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
TOP_K = 4

# The instruction that keeps answers grounded. Two things matter here:
# the model is told to use only the supplied context, and it is given an
# explicit way out. Without that escape hatch a model will strain to produce
# an answer from its own training data rather than admit the text is missing.
SYSTEM_PROMPT = """You answer questions about a set of research papers.

Rules:
- Answer using ONLY the numbered sources provided in the user's message.
- Do not use any outside knowledge, even if you are confident it is correct.
- If the sources do not contain the answer, say exactly: "That isn't covered in these documents."
- Cite the sources you used by their number, like [1] or [2].
- Be concise."""


def retrieve(collection, question, k=TOP_K):
    """Find the k chunks most similar in meaning to the question."""
    # The question must be embedded by the same model used to build the index,
    # otherwise the vectors describe different spaces and the comparison is noise.
    query_vector = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))


def build_prompt(question, hits):
    """Lay the retrieved chunks out as numbered sources, then ask the question."""
    blocks = []
    for i, (text, meta, _dist) in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({meta['source']}, page {meta['page_label']})\n{text}")
    context = "\n\n".join(blocks)
    return f"Sources:\n\n{context}\n\nQuestion: {question}"


def answer(question):
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    hits = retrieve(collection, question)

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
    print("\nSources retrieved:")
    for i, (_text, meta, dist) in enumerate(hits, start=1):
        print(f"  [{i}] {meta['source']} page {meta['page_label']}  (distance {dist:.3f})")


if __name__ == "__main__":
    main()
