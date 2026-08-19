"""Tests for text extraction cleanup.

Each test here corresponds to a bug that actually occurred and was invisible in the
output at the time -- answers looked plausible while the index was full of junk.
"""

from pathlib import Path

from ingest import clean_text, count_blank_pages, get_title


class FakePage:
    """Stands in for a LangChain Document, which is just text plus metadata."""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class TestCleanTextLatex:
    """One paper embedded the LaTeX source of every equation as hidden text.

    Invisible to a reader, extracted by pypdf as long base64 blobs: 44% of that
    document's characters and 57% of its chunks.
    """

    def test_strips_a_latexit_block(self):
        text = 'Before <latexit sha1_base64="abc123">AAAB8nicbVDLSg</latexit> after'
        assert clean_text(text) == "Before  after"

    def test_strips_a_block_spanning_multiple_lines(self):
        text = 'Start <latexit sha1_base64="x">AAAA\nBBBB\nCCCC</latexit> end'
        assert "latexit" not in clean_text(text)
        assert "AAAA" not in clean_text(text)

    def test_strips_several_blocks_in_one_page(self):
        text = "A <latexit>one</latexit> B <latexit>two</latexit> C"
        cleaned = clean_text(text)
        assert "one" not in cleaned and "two" not in cleaned
        assert "A" in cleaned and "B" in cleaned and "C" in cleaned

    def test_leaves_ordinary_text_untouched(self):
        text = "The dataset contains 5000 samples annotated into 80 categories."
        assert clean_text(text) == text


class TestCleanTextHyphenation:
    """Justified text hyphenates across line breaks -- 568 times in these four papers.

    Every one was a token no search could match: "floorplancad" did not exist anywhere
    in the corpus because the text said "FloorPlan-\\nCAD".
    """

    def test_rejoins_a_word_split_across_lines(self):
        assert clean_text("con-\ntaining") == "containing"

    def test_rejoins_when_the_break_has_surrounding_whitespace(self):
        assert clean_text("spot-\n  ting") == "spotting"

    def test_keeps_the_hyphen_when_the_next_word_is_capitalised(self):
        # "FloorPlan-CAD" might be a real compound, so the hyphen stays and the
        # tokenizer normalises it away instead. Only the line break is removed.
        assert clean_text("FloorPlan-\nCAD") == "FloorPlan-CAD"

    def test_keeps_the_hyphen_before_a_digit(self):
        assert clean_text("Figure-\n3") == "Figure-3"

    def test_does_not_touch_a_hyphen_in_the_middle_of_a_line(self):
        assert clean_text("line-grained annotations") == "line-grained annotations"

    def test_does_not_join_across_a_plain_line_break(self):
        # No hyphen means no word was split; the newline must survive.
        assert clean_text("first line\nsecond line") == "first line\nsecond line"


class TestCountBlankPages:
    """A blank page means a scan -- an image with no text. It fails silently."""

    def test_counts_pages_with_no_text(self):
        pages = [FakePage("real text"), FakePage(""), FakePage("more text")]
        assert count_blank_pages(pages) == 1

    def test_treats_whitespace_only_as_blank(self):
        # A page can return "\n  \n", which is not empty to Python but is to a reader.
        assert count_blank_pages([FakePage("\n  \t\n")]) == 1

    def test_returns_zero_when_every_page_has_text(self):
        assert count_blank_pages([FakePage("a"), FakePage("b")]) == 0


class TestGetTitle:
    """Titles are what distinguish one paper from another during retrieval."""

    def test_uses_the_known_title_for_an_indexed_paper(self):
        title = get_title(Path("1904.01920v1.pdf"), [FakePage("ignored")])
        assert title.startswith("CubiCasa5K")

    def test_falls_back_to_the_top_of_page_one_for_an_unknown_file(self):
        pages = [FakePage("Some Paper Title\nA. Author, B. Author\nrest of page")]
        assert get_title(Path("unknown.pdf"), pages) == "Some Paper Title A. Author, B. Author"

    def test_falls_back_to_the_filename_when_page_one_is_empty(self):
        assert get_title(Path("mystery.pdf"), [FakePage("   ")]) == "mystery"
