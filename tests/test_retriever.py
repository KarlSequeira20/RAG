"""Tests for keyword tokenisation.

Tokenisation is half of the hyphenation fix. `clean_text` repairs words split across
line breaks; the tokenizer handles the remaining case, where a compound keeps its hyphen
and the question does not (or vice versa).
"""

from retriever import tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("FloorPlan Dataset") == ["floorplan", "dataset"]

    def test_drops_punctuation(self):
        assert tokenize("dataset, samples. count!") == ["dataset", "samples", "count"]

    def test_a_hyphenated_compound_matches_the_joined_form(self):
        # The papers write "FloorPlan-CAD" in places and "FloorPlanCAD" in others.
        # Both must produce the same token or keyword search cannot connect them.
        assert tokenize("FloorPlan-CAD") == tokenize("FloorPlanCAD") == ["floorplancad"]

    def test_question_and_document_normalise_identically(self):
        question = tokenize("What is in FloorPlanCAD?")
        document = tokenize("The FloorPlan-CAD dataset contains drawings.")
        assert "floorplancad" in question
        assert "floorplancad" in document

    def test_keeps_digits(self):
        assert tokenize("5000 samples") == ["5000", "samples"]

    def test_empty_string_gives_no_tokens(self):
        assert tokenize("") == []
