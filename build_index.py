"""Build the vector index: PDFs -> chunks -> embeddings -> Chroma.

Run this once (and again whenever the documents or chunking settings change).
It rebuilds the collection from scratch so re-running never leaves duplicate
chunks behind.
"""

from pathlib import Path

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

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
        "kind": "pdf",
        "page": metadata["page"],              # 0-indexed
        "page_label": metadata["page_label"],  # printed page number, as shown to a reader
    }


def index_pdf(collection, path):
    """Chunk, embed and store one PDF. Returns (chunks added, blank pages).

    Shared by the batch build below and by the Streamlit upload handler, so a file
    uploaded at runtime is processed exactly like one indexed ahead of time.
    Re-indexing the same filename replaces its chunks rather than duplicating them.
    """
    pages = load_pdf(path)
    title = get_title(path, pages)
    chunks = chunk_pages(pages)
    blank_pages = sum(1 for p in pages if not p.page_content.strip())

    if not chunks:
        return 0, blank_pages

    existing = collection.get(where={"source": path.name}, include=[])["ids"]
    if existing:
        collection.delete(ids=existing)

    texts = [c.page_content for c in chunks]
    collection.add(
        ids=[f"{path.name}:{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embed_texts([contextualise(x, title) for x in texts]),
        metadatas=[clean_metadata(c.metadata, path.name, title) for c in chunks],
    )
    return len(chunks), blank_pages


def index_url(collection, url):
    """Fetch a web page, chunk it and store it. Returns (chunks added, title).

    Web pages have no pages to cite, so chunks are numbered as parts instead. The
    `kind` field lets every display surface label a source correctly without guessing
    from the filename.
    """
    from web import fetch_page  # imported here so PDF-only use needs no web dependency

    title, text = fetch_page(url)
    chunks = chunk_pages([Document(page_content=text, metadata={})])

    existing = collection.get(where={"source": url}, include=[])["ids"]
    if existing:
        collection.delete(ids=existing)

    texts = [c.page_content for c in chunks]
    collection.add(
        ids=[f"{url}:{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embed_texts([contextualise(x, title) for x in texts]),
        metadatas=[
            {
                "source": url,
                "title": title,
                "kind": "web",
                "page": i,
                "page_label": str(i + 1),  # "part N" -- a web page has no page numbers
            }
            for i in range(len(chunks))
        ],
    )
    return len(chunks), title


def get_collection():
    """Open the persistent collection, creating it if this is a first run."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start clean so re-running doesn't stack duplicate copies of every chunk.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0

    for path in sorted(DATA_DIR.glob("*.pdf")):
        print(f"{path.name}: embedding...")
        added, blanks = index_pdf(collection, path)
        note = f", {blanks} blank pages" if blanks else ""
        print(f"{path.name}: {added} chunks{note}")
        total_chunks += added

    print(f"\nStored {total_chunks} chunks in {CHROMA_DIR}")
    print(f"Collection '{COLLECTION_NAME}' now holds {collection.count()} items.")


if __name__ == "__main__":
    main()
