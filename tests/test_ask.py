"""Tests for how a chunk's origin is described in a citation."""

from ask import describe_location


class TestDescribeLocation:
    def test_a_pdf_is_cited_by_page(self):
        meta = {"kind": "pdf", "source": "paper.pdf", "page_label": "4"}
        assert describe_location(meta) == "paper.pdf, page 4"

    def test_a_web_page_is_cited_by_part(self):
        # A web page has no pages, so its chunks are numbered as parts instead.
        meta = {"kind": "web", "source": "https://example.com/a", "page_label": "3"}
        assert describe_location(meta) == "https://example.com/a, part 3"

    def test_a_chunk_with_no_kind_is_treated_as_a_pdf(self):
        # Chunks indexed before web support existed carry no `kind` field.
        meta = {"source": "old.pdf", "page_label": "1"}
        assert describe_location(meta) == "old.pdf, page 1"
