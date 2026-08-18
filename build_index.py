"""Build the vector index: PDFs -> chunks -> embeddings -> Chroma.

Run this once (and again whenever the documents or chunking settings change).
It rebuilds the collection from scratch so re-running never leaves duplicate
chunks behind.
"""

from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import DATA_DIR, get_title, load_pdf

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "papers"

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# How many chunks to send to Ollama per embed call. Batching is much faster
# than one call per chunk, but a whole run at once would be a large request.
BATCH_SIZE = 32


def chunk_pages(pages):
    """Split page-level Documents into smaller, retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def embed_texts(texts):
    """Turn a list of strings into a list of embedding vectors."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(response["embeddings"])
    return vectors


def contextualise(text, title):
    """Prefix a chunk with its paper title, for embedding only.

    Vector search is weak at proper nouns: "FloorPlanCAD" and "CubiCasa5K" both
    encode as roughly "floorplan dataset", so a question naming one paper does not
    reliably retrieve from that paper. Worse, a chunk from the middle of a paper
    says "our dataset" and never repeats the paper's name, so nothing in its
    embedding ties it to the paper it came from.

    Putting the title into the text we embed gives every chunk that signal. The
    stored document stays clean -- the title travels in metadata instead.
    """
    return f"{title}\n\n{text}"


def clean_metadata(metadata, filename, title):
    """Keep only the fields we need for citations.

    Chroma accepts str/int/float/bool values, and the raw pypdf metadata is
    mostly PDF-producer noise. We also store the bare filename rather than the
    absolute path, since the local path is meaningless once this is deployed.
    """
    return {
        "source": filename,
        "title": title,
        "page": metadata["page"],              # 0-indexed
        "page_label": metadata["page_label"],  # printed page number, as shown to a reader
    }


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start clean so re-running doesn't stack duplicate copies of every chunk.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0

    for path in sorted(DATA_DIR.glob("*.pdf")):
        pages = load_pdf(path)
        title = get_title(path, pages)
        chunks = chunk_pages(pages)

        texts = [c.page_content for c in chunks]
        metadatas = [clean_metadata(c.metadata, path.name, title) for c in chunks]
        ids = [f"{path.name}:{i}" for i in range(len(chunks))]

        print(f"{path.name}: {len(pages)} pages -> {len(chunks)} chunks, embedding...")
        # Embed the title-prefixed text, but store the original text unchanged.
        embeddings = embed_texts([contextualise(t, title) for t in texts])

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\nStored {total_chunks} chunks in {CHROMA_DIR}")
    print(f"Collection '{COLLECTION_NAME}' now holds {collection.count()} items.")


if __name__ == "__main__":
    main()
