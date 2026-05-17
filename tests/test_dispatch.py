"""Tests for src/evaluation/dispatch.py - shared parallel LLM dispatch."""

from unittest.mock import MagicMock

from src.evaluation import dispatch


def _case(prompts, prompt_type="standard"):
    c = MagicMock()
    c.prompts = prompts
    c.prompt_type = prompt_type
    return c


def _llm(response="raw response"):
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# Cycle 1 - single case, single prompt
# ---------------------------------------------------------------------------



class TestSingleCaseSinglePrompt:
    def test_returns_list_of_one_inner_list(self):
        cases = [_case(["prompt A"])]
        result = dispatch.resolve_raw(cases, _llm())
        assert len(result) == 1

    def test_inner_list_contains_llm_response(self):
        cases = [_case(["prompt A"])]
        result = dispatch.resolve_raw(cases, _llm("hello"))
        assert result[0] == ["hello"]


# ---------------------------------------------------------------------------
# Cycle 2 - multiple cases, responses in case order
# ---------------------------------------------------------------------------


class TestMultipleCases:
    def test_returns_one_inner_list_per_case(self):
        cases = [_case(["p1"]), _case(["p2"]), _case(["p3"])]
        result = dispatch.resolve_raw(cases, _llm())
        assert len(result) == 3

    def test_case_order_preserved(self):
        responses = {"p1": "resp-1", "p2": "resp-2", "p3": "resp-3"}
        llm = MagicMock()
        llm.complete.side_effect = lambda prompt, _max=None: responses[prompt]
        cases = [_case(["p1"]), _case(["p2"]), _case(["p3"])]
        result = dispatch.resolve_raw(cases, llm)
        assert result[0] == ["resp-1"]
        assert result[1] == ["resp-2"]
        assert result[2] == ["resp-3"]


# ---------------------------------------------------------------------------
# Cycle 3 - multiple prompts per case (self-consistency), prompt order
# ---------------------------------------------------------------------------


class TestMultiplePromptsPerCase:
    def test_inner_list_length_matches_prompt_count(self):
        cases = [_case(["p1", "p2", "p3"])]
        result = dispatch.resolve_raw(cases, _llm())
        assert len(result[0]) == 3

    def test_prompt_order_preserved_within_case(self):
        responses = {"run-0": "r0", "run-1": "r1", "run-2": "r2"}
        llm = MagicMock()
        llm.complete.side_effect = lambda prompt, _max=None: responses[prompt]
        cases = [_case(["run-0", "run-1", "run-2"])]
        result = dispatch.resolve_raw(cases, llm)
        assert result[0] == ["r0", "r1", "r2"]

    def test_multiple_cases_multiple_prompts(self):
        responses = {"a0": "A0", "a1": "A1", "b0": "B0", "b1": "B1"}
        llm = MagicMock()
        llm.complete.side_effect = lambda prompt, _max=None: responses[prompt]
        cases = [_case(["a0", "a1"]), _case(["b0", "b1"])]
        result = dispatch.resolve_raw(cases, llm)
        assert result[0] == ["A0", "A1"]
        assert result[1] == ["B0", "B1"]


# ---------------------------------------------------------------------------
# Cycle 4 - empty cases
# ---------------------------------------------------------------------------


class TestEmptyCases:
    def test_empty_input_returns_empty_list(self):
        result = dispatch.resolve_raw([], _llm())
        assert result == []

    def test_llm_not_called_on_empty_input(self):
        llm = _llm()
        dispatch.resolve_raw([], llm)
        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Cycle 5 - LLM call count
# ---------------------------------------------------------------------------


class TestLLMCallCount:
    def test_called_once_per_prompt(self):
        cases = [_case(["p"]), _case(["q", "r"])]
        llm = _llm()
        dispatch.resolve_raw(cases, llm)
        assert llm.complete.call_count == 3

    def test_prompt_type_passed_to_llm(self):
        cases = [_case(["p"], prompt_type="chain_of_thought")]
        llm = _llm()
        dispatch.resolve_raw(cases, llm)
        llm.complete.assert_called_once_with("p", "chain_of_thought")
