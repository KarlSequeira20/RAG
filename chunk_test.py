"""Compare chunk sizes before committing to one.

Throwaway experiment, not part of the pipeline. The point is to see what
different chunk sizes actually produce on our own documents, instead of
copying a chunk size from a blog post.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import DATA_DIR, load_pdf

SIZES = [(200, 20), (500, 50), (1000, 100), (2000, 200)]


def main():
    path = sorted(DATA_DIR.glob("*.pdf"))[0]
    pages = load_pdf(path)
    total_chars = sum(len(p.page_content) for p in pages)

    print(f"{path.name}: {len(pages)} pages, {total_chars:,} characters\n")

    for size, overlap in SIZES:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(pages)
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"size={size:<5} overlap={overlap:<4} -> {len(chunks):>4} chunks, avg {avg:.0f} chars")

    # Print one chunk from a body page at two different sizes, to see whether
    # each one reads as a self-contained piece of information.
    for size, overlap in [(200, 20), (1000, 100)]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_documents(pages)
        middle = chunks[len(chunks) // 2]
        print(f"\n--- sample chunk at size={size} (page {middle.metadata['page_label']}) ---")
        print(middle.page_content)


if __name__ == "__main__":
    main()
