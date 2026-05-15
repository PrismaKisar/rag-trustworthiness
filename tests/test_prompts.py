"""Tests for src/generation/prompts.py (step 14).

Assertions:
- format_prompt() returns a string containing the claim and all passages.
- Each template includes the correct structural markers.
- Passages are numbered in order.
- Invalid prompt_type raises ValueError.
- Empty passages list is handled (no crash, claim still present).
"""

from __future__ import annotations

import pytest

from src.generation.prompts import format_prompt

CLAIM = "Albert Einstein was born in Germany."
PASSAGES = [
    "Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg.",
    "He later moved to Switzerland and then to the United States.",
]


# ---------------------------------------------------------------------------
# Standard template
# ---------------------------------------------------------------------------


class TestStandardTemplate:
    def test_contains_claim(self):
        prompt = format_prompt(CLAIM, PASSAGES, "standard")
        assert CLAIM in prompt

    def test_contains_all_passages(self):
        prompt = format_prompt(CLAIM, PASSAGES, "standard")
        for passage in PASSAGES:
            assert passage in prompt

    def test_passages_numbered(self):
        prompt = format_prompt(CLAIM, PASSAGES, "standard")
        assert "1." in prompt
        assert "2." in prompt

    def test_has_label_marker(self):
        prompt = format_prompt(CLAIM, PASSAGES, "standard")
        assert "Label:" in prompt

    def test_no_cot_markers(self):
        prompt = format_prompt(CLAIM, PASSAGES, "standard")
        assert "Reasoning:" not in prompt
        assert "Consistency check:" not in prompt


# ---------------------------------------------------------------------------
# Chain-of-thought template
# ---------------------------------------------------------------------------


class TestChainOfThoughtTemplate:
    def test_contains_claim(self):
        prompt = format_prompt(CLAIM, PASSAGES, "chain_of_thought")
        assert CLAIM in prompt

    def test_has_reasoning_marker(self):
        prompt = format_prompt(CLAIM, PASSAGES, "chain_of_thought")
        assert "Reasoning:" in prompt

    def test_has_final_label_marker(self):
        prompt = format_prompt(CLAIM, PASSAGES, "chain_of_thought")
        assert "Final Label" in prompt

    def test_contains_all_passages(self):
        prompt = format_prompt(CLAIM, PASSAGES, "chain_of_thought")
        for passage in PASSAGES:
            assert passage in prompt


# ---------------------------------------------------------------------------
# Vigilant template
# ---------------------------------------------------------------------------


class TestVigilantTemplate:
    def test_contains_claim(self):
        prompt = format_prompt(CLAIM, PASSAGES, "vigilant")
        assert CLAIM in prompt

    def test_has_consistency_check_marker(self):
        prompt = format_prompt(CLAIM, PASSAGES, "vigilant")
        assert "Consistency check:" in prompt

    def test_has_final_label_marker(self):
        prompt = format_prompt(CLAIM, PASSAGES, "vigilant")
        assert "Final Label" in prompt

    def test_contains_all_passages(self):
        prompt = format_prompt(CLAIM, PASSAGES, "vigilant")
        for passage in PASSAGES:
            assert passage in prompt


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_prompt_type_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt_type"):
            format_prompt(CLAIM, PASSAGES, "nonexistent")  # type: ignore[arg-type]

    def test_empty_passages_no_crash(self):
        prompt = format_prompt(CLAIM, [], "standard")
        assert CLAIM in prompt

    def test_single_passage(self):
        prompt = format_prompt(CLAIM, ["Only one passage."], "standard")
        assert "1." in prompt
        assert "Only one passage." in prompt

    def test_passage_order_preserved(self):
        passages = ["first", "second", "third"]
        prompt = format_prompt(CLAIM, passages, "standard")
        idx_first = prompt.index("first")
        idx_second = prompt.index("second")
        idx_third = prompt.index("third")
        assert idx_first < idx_second < idx_third

    def test_default_prompt_type_is_standard(self):
        prompt_default = format_prompt(CLAIM, PASSAGES)
        prompt_standard = format_prompt(CLAIM, PASSAGES, "standard")
        assert prompt_default == prompt_standard


# ---------------------------------------------------------------------------
# QA prompts (HotpotQA)
# ---------------------------------------------------------------------------

QUESTION = "Where was Marie Curie born?"
QA_PASSAGES = [
    "Marie Curie was born in Warsaw, Poland on 7 November 1867.",
    "She moved to Paris to study at the Sorbonne in 1891.",
]


class TestStandardQAPrompt:
    def test_contains_question(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "standard_qa")
        assert QUESTION in prompt

    def test_contains_all_passages(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "standard_qa")
        for p in QA_PASSAGES:
            assert p in prompt

    def test_has_answer_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "standard_qa")
        assert "Answer:" in prompt


class TestCotQAPrompt:
    def test_has_reasoning_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "cot_qa")
        assert "Reasoning:" in prompt

    def test_has_final_answer_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "cot_qa")
        assert "Final Answer" in prompt


class TestVigilantQAPrompt:
    def test_has_consistency_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "vigilant_qa")
        assert "Consistency check:" in prompt

    def test_has_final_answer_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "vigilant_qa")
        assert "Final Answer" in prompt


class TestQAPromptEdgeCases:
    def test_invalid_qa_prompt_type_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt_type"):
            format_prompt(QUESTION, QA_PASSAGES, "bogus")

    def test_passages_numbered(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "standard_qa")
        assert "1." in prompt
        assert "2." in prompt


# ---------------------------------------------------------------------------
# Unified formatter: format_prompt handles all 6 prompt types
# ---------------------------------------------------------------------------


class TestUnifiedFormatter:
    def test_qa_type_via_format_prompt_contains_question(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "standard_qa")
        assert QUESTION in prompt

    def test_cot_qa_via_format_prompt_has_reasoning_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "cot_qa")
        assert "Reasoning:" in prompt

    def test_vigilant_qa_via_format_prompt_has_consistency_marker(self):
        prompt = format_prompt(QUESTION, QA_PASSAGES, "vigilant_qa")
        assert "Consistency check:" in prompt

    def test_query_injected_regardless_of_internal_kwarg_name(self):
        claim_prompt = format_prompt("The sky is blue.", [], "standard")
        qa_prompt = format_prompt("The sky is blue.", [], "standard_qa")
        assert "The sky is blue." in claim_prompt
        assert "The sky is blue." in qa_prompt

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown prompt_type"):
            format_prompt(QUESTION, QA_PASSAGES, "bogus_type")
