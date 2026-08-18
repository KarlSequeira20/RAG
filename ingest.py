"""Step 1 of the RAG pipeline: load PDFs and check what we can actually read.

No chunking or embedding yet. The only goal here is to answer one question:
which of these PDFs give us usable text, and what metadata comes with it?
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path(__file__).parent / "data"


def load_pdf(path: Path):
    """Load one PDF into a list of Documents, one per page."""
    loader = PyPDFLoader(str(path))
    return loader.load()


def count_blank_pages(pages) -> int:
    """How many pages came back with no usable text?

    A page whose text is empty (or only whitespace) means pypdf found nothing
    to extract -- usually because the page is a scan, i.e. one big image.
    """
    return sum(1 for page in pages if not page.page_content.strip())


def main():
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}")
        return

    print(f"Found {len(pdf_paths)} PDF(s) in {DATA_DIR}\n")

    for path in pdf_paths:
        try:
            pages = load_pdf(path)
        except Exception as exc:
            # Encrypted, corrupt, or otherwise unreadable. Report and move on
            # rather than killing the whole run.
            print(f"{path.name}: FAILED to load -- {type(exc).__name__}: {exc}")
            continue

        blanks = count_blank_pages(pages)
        note = ""
        if blanks == len(pages):
            note = "  <-- no text at all, likely a scanned PDF"
        elif blanks > len(pages) / 2:
            note = "  <-- mostly blank, check this one"

        print(f"{path.name}: {len(pages)} pages, {blanks} blank{note}")

    # Look closely at a single page. The metadata dict here is what every
    # source citation later in the project will be built from, so it is worth
    # knowing exactly which fields exist before relying on them.
    first_pages = load_pdf(pdf_paths[0])
    print(f"\n--- Inspecting first page of {pdf_paths[0].name} ---")

    print("metadata:", first_pages[0].metadata)
    print("\ntext (first 300 chars):")
    print(first_pages[0].page_content[:300])


if __name__ == "__main__":
    main()
