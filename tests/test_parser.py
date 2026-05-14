"""Tests for src/generation/parser.py — extract_label()."""

import pytest

from src.generation.parser import extract_label


# ---------------------------------------------------------------------------
# Primary path: Final Label: marker
# ---------------------------------------------------------------------------

class TestFinalLabelMarker:
    def test_supports(self):
        assert extract_label("Reasoning: ...\nFinal Label: SUPPORTS") == "SUPPORTS"

    def test_refutes(self):
        assert extract_label("Some reasoning.\nFinal Label: REFUTES") == "REFUTES"

    def test_not_enough_info(self):
        assert extract_label("Final Label: NOT ENOUGH INFO") == "NOT ENOUGH INFO"

    def test_case_insensitive(self):
        assert extract_label("final label: supports") == "SUPPORTS"

    def test_extra_whitespace(self):
        assert extract_label("Final Label:   REFUTES   ") == "REFUTES"

    def test_not_enough_info_extra_spaces(self):
        # Extra internal whitespace inside the label itself
        assert extract_label("Final Label: NOT  ENOUGH  INFO") == "NOT ENOUGH INFO"

    def test_dash_separator(self):
        assert extract_label("Final Label - SUPPORTS") == "SUPPORTS"


# ---------------------------------------------------------------------------
# Fallback path: keyword scan
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def test_supports_keyword(self):
        assert extract_label("The evidence supports the claim.") == "SUPPORTS"

    def test_refutes_keyword(self):
        assert extract_label("This passage refutes what was stated.") == "REFUTES"

    def test_not_enough_info_keyword(self):
        assert extract_label("There is not enough info to decide.") == "NOT ENOUGH INFO"

    def test_not_enough_info_takes_priority_over_supports(self):
        # Both keywords present — NOT ENOUGH INFO must win (checked first).
        assert extract_label("supports the claim but not enough info") == "NOT ENOUGH INFO"


# ---------------------------------------------------------------------------
# Default path: no match
# ---------------------------------------------------------------------------

class TestDefault:
    def test_garbage_text(self):
        assert extract_label("xkcd 1337 lorem ipsum") == "NOT ENOUGH INFO"

    def test_empty_string(self):
        assert extract_label("") == "NOT ENOUGH INFO"


# ---------------------------------------------------------------------------
# extract_answer — HotpotQA QA output
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_final_answer_marker(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Reasoning: ...\nFinal Answer: Switzerland") == "Switzerland"

    def test_answer_marker(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Answer: Marie Curie") == "Marie Curie"

    def test_case_insensitive_marker(self):
        from src.generation.parser import extract_answer
        assert extract_answer("answer: paris") == "paris"

    def test_strips_surrounding_whitespace(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Final Answer:   Arthur C. Clarke   ") == "Arthur C. Clarke"

    def test_strips_trailing_period(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Answer: Switzerland.") == "Switzerland"

    def test_stops_at_newline(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Answer: Paris\nExtra commentary.") == "Paris"

    def test_no_marker_returns_full_text_stripped(self):
        from src.generation.parser import extract_answer
        assert extract_answer("  Switzerland  ") == "Switzerland"

    def test_empty_string(self):
        from src.generation.parser import extract_answer
        assert extract_answer("") == ""

    def test_dash_separator(self):
        from src.generation.parser import extract_answer
        assert extract_answer("Final Answer - Berlin") == "Berlin"
