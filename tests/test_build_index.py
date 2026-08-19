"""Tests for chunking, metadata and the re-indexing path."""

import build_index
from build_index import chunk_pages, clean_metadata, contextualise, index_pdf
from langchain_core.documents import Document


class TestChunkPages:
    def test_no_chunk_exceeds_the_configured_size(self):
        page = Document(page_content="word " * 2000, metadata={"page": 0, "page_label": "1"})
        for chunk in chunk_pages([page]):
            assert len(chunk.page_content) <= build_index.CHUNK_SIZE

    def test_page_metadata_survives_chunking(self):
        # Citations depend on this: a chunk must still know which page it came from.
        page = Document(page_content="word " * 2000, metadata={"page": 6, "page_label": "7"})
        for chunk in chunk_pages([page]):
            assert chunk.metadata["page_label"] == "7"

    def test_chunks_are_never_merged_across_pages(self):
        # Merging would produce a chunk spanning two pages, making its citation wrong.
        pages = [
            Document(page_content="alpha", metadata={"page": 0, "page_label": "1"}),
            Document(page_content="beta", metadata={"page": 1, "page_label": "2"}),
        ]
        chunks = chunk_pages(pages)
        for chunk in chunks:
            assert "alpha" not in chunk.page_content or "beta" not in chunk.page_content

    def test_short_text_stays_as_one_chunk(self):
        page = Document(page_content="A short page.", metadata={"page": 0, "page_label": "1"})
        assert len(chunk_pages([page])) == 1


class TestContextualise:
    def test_prefixes_the_title(self):
        # Without this a chunk saying "our dataset" has nothing tying it to its paper.
        assert contextualise("body text", "Paper Title").startswith("Paper Title")

    def test_keeps_the_original_text(self):
        assert "body text" in contextualise("body text", "Paper Title")


class TestCleanMetadata:
    def test_keeps_only_the_fields_needed_for_citation(self):
        raw = {"page": 3, "page_label": "4", "producer": "pdfTeX", "creationdate": "2019"}
        cleaned = clean_metadata(raw, "paper.pdf", "A Title")
        assert cleaned == {
            "source": "paper.pdf",
            "title": "A Title",
            "kind": "pdf",
            "page": 3,
            "page_label": "4",
        }

    def test_stores_the_bare_filename_not_a_path(self):
        # An absolute local path is meaningless to anyone reading the citation.
        cleaned = clean_metadata({"page": 0, "page_label": "1"}, "paper.pdf", "T")
        assert "/" not in cleaned["source"]


class FakeCollection:
    """Records what index_pdf does, so the dedup path can be tested without Chroma."""

    def __init__(self, existing_ids=()):
        self.existing_ids = list(existing_ids)
        self.deleted = []
        self.added = None

    def get(self, where=None, include=None):
        return {"ids": list(self.existing_ids)}

    def delete(self, ids):
        self.deleted.extend(ids)

    def add(self, ids, documents, embeddings, metadatas):
        self.added = {"ids": ids, "documents": documents, "metadatas": metadatas}


class TestIndexPdfReplacesRatherThanDuplicates:
    """Re-uploading a file already indexed used to duplicate every one of its chunks."""

    def _patch(self, monkeypatch, pages):
        monkeypatch.setattr(build_index, "load_pdf", lambda path: pages)
        monkeypatch.setattr(build_index, "get_title", lambda path, pages: "A Title")
        # Embedding would call Ollama; the dedup logic under test does not need it.
        monkeypatch.setattr(build_index, "embed_texts", lambda texts: [[0.0]] * len(texts))

    def test_deletes_existing_chunks_for_the_same_file(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, [Document(page_content="text", metadata={"page": 0, "page_label": "1"})])
        collection = FakeCollection(existing_ids=["paper.pdf:0", "paper.pdf:1"])
        index_pdf(collection, tmp_path / "paper.pdf")
        assert collection.deleted == ["paper.pdf:0", "paper.pdf:1"]

    def test_deletes_nothing_when_the_file_is_new(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, [Document(page_content="text", metadata={"page": 0, "page_label": "1"})])
        collection = FakeCollection()
        index_pdf(collection, tmp_path / "new.pdf")
        assert collection.deleted == []

    def test_chunk_ids_are_unique(self, monkeypatch, tmp_path):
        pages = [Document(page_content="word " * 900, metadata={"page": 0, "page_label": "1"})]
        self._patch(monkeypatch, pages)
        collection = FakeCollection()
        added, _blanks = index_pdf(collection, tmp_path / "paper.pdf")
        ids = collection.added["ids"]
        assert added == len(ids) == len(set(ids))

    def test_reports_blank_pages(self, monkeypatch, tmp_path):
        pages = [
            Document(page_content="real text", metadata={"page": 0, "page_label": "1"}),
            Document(page_content="   ", metadata={"page": 1, "page_label": "2"}),
        ]
        self._patch(monkeypatch, pages)
        _added, blanks = index_pdf(FakeCollection(), tmp_path / "paper.pdf")
        assert blanks == 1

    def test_a_scanned_pdf_adds_nothing(self, monkeypatch, tmp_path):
        # Every page blank means an image-only PDF: nothing to index, and the caller
        # needs to be told rather than seeing a silent success.
        pages = [Document(page_content="  ", metadata={"page": 0, "page_label": "1"})]
        self._patch(monkeypatch, pages)
        collection = FakeCollection()
        added, blanks = index_pdf(collection, tmp_path / "scan.pdf")
        assert added == 0 and blanks == 1
        assert collection.added is None
