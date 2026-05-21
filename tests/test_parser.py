"""Tests for src/generation/parser.py - extract_label()."""

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
        # Both keywords present - NOT ENOUGH INFO must win (checked first).
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
# extract_answer - HotpotQA QA output
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

    def test_bold_markdown_final_answer(self):
        from src.generation.parser import extract_answer
        assert extract_answer("**Final Answer:** Switzerland") == "Switzerland"

    def test_bold_markdown_with_verbose_answer(self):
        from src.generation.parser import extract_answer
        result = extract_answer("Passages inconsistent.\n\n**Final Answer:** No.")
        assert result == "No"

    def test_bold_markdown_answer_colon(self):
        from src.generation.parser import extract_answer
        assert extract_answer("**Answer:** Paris") == "Paris"

    def test_bold_with_extra_spaces(self):
        from src.generation.parser import extract_answer
        assert extract_answer("**Final Answer:**  October 1922") == "October 1922"

    def test_inline_answer_is_fallback(self):
        from src.generation.parser import extract_answer
        text = "Passage 1 mentions X. The answer is Firth of Forth."
        assert extract_answer(text) == "Firth of Forth"

    def test_inline_the_correct_answer_is(self):
        from src.generation.parser import extract_answer
        text = "Based on the evidence, the correct answer is yes."
        assert extract_answer(text) == "yes"

    def test_last_marker_wins_with_bold(self):
        from src.generation.parser import extract_answer
        text = "Answer: first guess\n**Final Answer:** Berlin"
        assert extract_answer(text) == "Berlin"


# ---------------------------------------------------------------------------
# extract_contradiction_flag - vigilant prompt consistency check
# ---------------------------------------------------------------------------

class TestExtractContradictionFlag:
    def test_explicit_contradict_returns_true(self):
        from src.generation.parser import extract_contradiction_flag
        text = (
            "Consistency check: The passages contradict each other.\n"
            "Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO): REFUTES"
        )
        assert extract_contradiction_flag(text) is True

    def test_consistent_passages_returns_false(self):
        from src.generation.parser import extract_contradiction_flag
        text = (
            "Consistency check: The passages are consistent with each other.\n"
            "Final Label (SUPPORTS / REFUTES / NOT ENOUGH INFO): SUPPORTS"
        )
        assert extract_contradiction_flag(text) is False

    def test_no_consistency_section_returns_false(self):
        from src.generation.parser import extract_contradiction_flag
        assert extract_contradiction_flag("Final Label: SUPPORTS") is False

    def test_case_insensitive_match(self):
        from src.generation.parser import extract_contradiction_flag
        text = "Consistency Check: passages CONFLICT with each other.\nFinal Label: REFUTES"
        assert extract_contradiction_flag(text) is True

    def test_empty_string_returns_false(self):
        from src.generation.parser import extract_contradiction_flag
        assert extract_contradiction_flag("") is False
