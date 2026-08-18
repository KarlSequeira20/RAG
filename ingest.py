"""Step 1 of the RAG pipeline: load PDFs and check what we can actually read.

No chunking or embedding yet. The only goal here is to answer one question:
which of these PDFs give us usable text, and what metadata comes with it?
"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path(__file__).parent / "data"


# Some PDFs built with LaTeX figure tools embed the original LaTeX source of every
# equation as hidden text, wrapped in <latexit ...>...</latexit>. A reader never sees
# it, but text extraction pulls it out as long base64 blobs. In 2211.15658v2.pdf this
# accounted for 44% of the extracted characters, and it was ending up in chunks and
# competing with real content during retrieval.
LATEXIT_RE = re.compile(r"<latexit.*?</latexit>", re.DOTALL)

# Justified text hyphenates words across line breaks: "con-\ntaining", "spot-\nting".
# Extraction preserves the break, so the corpus contains 568 words that no search will
# ever match because the token simply is not there. Rejoining them repairs both the
# text a reader sees and the tokens a keyword search can find.
#
# Two cases, distinguished by what follows the break:
#   con-\ntaining  -> containing    (lowercase: a genuine mid-word split, drop the hyphen)
#   FloorPlan-\nCAD -> FloorPlan-CAD (uppercase: might be a real compound, keep the hyphen)
# The second case is still not the literal token "FloorPlanCAD", which is why
# `retriever.tokenize` also normalises hyphens away when matching.
HYPHEN_BREAK_LOWER = re.compile(r"(\w)-\s*\n\s*([a-z])")
HYPHEN_BREAK_OTHER = re.compile(r"(\w)-\s*\n\s*([A-Z0-9])")


def clean_text(text: str) -> str:
    """Strip extraction artefacts that a human reader would never see."""
    text = LATEXIT_RE.sub("", text)
    text = HYPHEN_BREAK_LOWER.sub(r"\1\2", text)
    text = HYPHEN_BREAK_OTHER.sub(r"\1-\2", text)
    return text


def load_pdf(path: Path):
    """Load one PDF into a list of Documents, one per page, with text cleaned."""
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)
    return pages


# Titles are what tell one paper apart from another, and the papers do not repeat
# their own name on every page. pypdf's built-in title metadata is empty for all four
# of these files, so we keep an explicit map and fall back to the top of page 1.
TITLES = {
    "1904.01920v1.pdf": "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis",
    "2105.07147v2.pdf": "FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting",
    "2211.15658v2.pdf": "Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries (RoomFormer)",
    "ijgi-10-00828-v2.pdf": "Automatic Extraction of Indoor Spatial Information from Floor Plan Image: A Patch-Based Deep Learning Methodology Application on Large-Scale Complex Buildings",
}


def get_title(path: Path, pages) -> str:
    """Best-effort title for a PDF.

    The hand-written map is reliable for the current document set. The fallback
    (first couple of lines of page 1) is there so an unknown PDF still gets
    something rather than nothing -- it will need revisiting when arbitrary
    uploads are supported.
    """
    if path.name in TITLES:
        return TITLES[path.name]
    lines = [ln.strip() for ln in pages[0].page_content.splitlines() if ln.strip()]
    return " ".join(lines[:2])[:200] if lines else path.stem


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
