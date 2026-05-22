"""Tests for src/evaluation/qa_scorer.py - three-phase QA pipeline."""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures: minimal HotpotQA-style examples
# ---------------------------------------------------------------------------

HOTPOT_EXAMPLES = [
    {
        "question": "Where was Marie Curie born?",
        "answer": "Warsaw",
        "supporting_facts": [["Marie Curie", 0], ["Warsaw", 0]],
        "context": [
            ["Marie Curie", ["Marie Curie was born in Warsaw.", "She won Nobel prizes."]],
            ["Warsaw", ["Warsaw is the capital of Poland.", "It is in central Poland."]],
            ["Distractor A", ["A totally unrelated paragraph.", "More filler."]],
        ],
    },
    {
        "question": "Who wrote 2001: A Space Odyssey?",
        "answer": "Arthur C. Clarke",
        "supporting_facts": [["2001 film", 0], ["Clarke", 0]],
        "context": [
            ["2001 film", ["The novel was written by Arthur C. Clarke.", "It became a 1968 film."]],
            ["Clarke", ["Arthur C. Clarke was a British author.", "He co-wrote the screenplay."]],
            ["Distractor B", ["Totally unrelated content.", "More filler text."]],
        ],
    },
]


def _retriever(passages=("p1", "p2")):
    r = MagicMock()
    r.retrieve.return_value = list(passages)
    r.k = 10
    return r


def _llm(response="Warsaw"):
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# prepare_cases
# ---------------------------------------------------------------------------


class TestPrepareCases:
    def test_returns_one_case_per_example(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        assert len(cases) == len(HOTPOT_EXAMPLES)

    def test_gold_answer_matches(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        for case, ex in zip(cases, HOTPOT_EXAMPLES):
            assert case.gold_answer == ex["answer"]

    def test_question_matches(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        for case, ex in zip(cases, HOTPOT_EXAMPLES):
            assert case.question == ex["question"]

    def test_prompt_contains_question(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        for case in cases:
            assert case.question in case.prompts[0]

    def test_one_prompt_per_case(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        for case in cases:
            assert len(case.prompts) == 1

    def test_passages_include_retrieved_and_supporting(self):
        """passages = retrieved distractors + injected supporting paragraphs."""
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(("x", "y")))
        case = cases[0]
        # retrieved passages must be present
        assert "x" in case.passages
        assert "y" in case.passages
        # supporting paragraph for "Marie Curie" must be injected
        assert "Marie Curie was born in Warsaw. She won Nobel prizes." in case.passages

    def test_prompt_contains_supporting_passage(self):
        """The injected supporting paragraph must appear in the formatted prompt."""
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        case = cases[0]
        assert "Marie Curie was born in Warsaw." in case.prompts[0]

    def test_retriever_built_per_example(self):
        from src.evaluation import qa_scorer
        r = _retriever()
        qa_scorer.prepare_cases(HOTPOT_EXAMPLES, r)
        assert r.build.call_count == len(HOTPOT_EXAMPLES)

    def test_prompt_type_stored_on_case(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(), prompt_type="cot_qa")
        for case in cases:
            assert case.prompt_type == "cot_qa"

    def test_returns_qa_case_instances(self):
        from src.evaluation import qa_scorer
        from src.evaluation.cases import QACase
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        for case in cases:
            assert isinstance(case, QACase)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def _make_cases(self):
        from src.evaluation import qa_scorer
        return qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())

    def test_empty_cases_returns_empty_list(self):
        from src.evaluation import qa_scorer
        assert qa_scorer.resolve([], _llm()) == []

    def test_returns_one_result_per_case(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm())
        assert len(results) == len(cases)

    def test_runs_length_is_one(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm())
        for result in results:
            assert len(result.runs) == 1

    def test_predicted_answer_extracted_from_llm_output(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm("Final Answer: Warsaw"))
        for result in results:
            assert result.predicted_answer == "Warsaw"

    def test_case_index_matches_position(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm())
        for i, result in enumerate(results):
            assert result.case_index == i


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _make_pair(self, predictions, gold_answers, retrieved=None):
        from src.evaluation.cases import QACase, QAResult
        retrieved = retrieved or [["g"]] * len(predictions)
        cases = [
            QACase(
                question=f"q-{i}",
                gold_answer=gold,
                passages=retrieved[i],
                prompts=["p"],
                prompt_type="standard_qa",
            )
            for i, gold in enumerate(gold_answers)
        ]
        results = [
            QAResult(case_index=i, runs=[pred], predicted_answer=pred)
            for i, pred in enumerate(predictions)
        ]
        return cases, results

    def test_returns_required_keys(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw"], ["Warsaw"])
        out = qa_scorer.aggregate(cases, results)
        assert {"exact_match", "token_f1", "hallucination_rate"} <= out.keys()

    def test_no_recall_at_k_key(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw"], ["Warsaw"])
        out = qa_scorer.aggregate(cases, results)
        assert "recall_at_k" not in out

    def test_no_self_consistency_key(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw"], ["Warsaw"])
        out = qa_scorer.aggregate(cases, results)
        assert "self_consistency" not in out

    def test_perfect_em(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw", "Clarke"], ["Warsaw", "Clarke"])
        out = qa_scorer.aggregate(cases, results)
        assert out["exact_match"] == pytest.approx(1.0)

    def test_zero_em(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Berlin", "Asimov"], ["Warsaw", "Clarke"])
        out = qa_scorer.aggregate(cases, results)
        assert out["exact_match"] == pytest.approx(0.0)

    def test_perfect_f1(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Marie Curie"], ["Marie Curie"])
        out = qa_scorer.aggregate(cases, results)
        assert out["token_f1"] == pytest.approx(1.0)

    def test_hallucination_rate_zero_when_grounded(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(
            ["Warsaw"], ["Warsaw"],
            retrieved=[["Warsaw is the capital of Poland."]],
        )
        out = qa_scorer.aggregate(cases, results)
        assert out["hallucination_rate"] == pytest.approx(0.0)

    def test_hallucination_rate_one_when_ungrounded(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(
            ["xyzzyquux"], ["Warsaw"],
            retrieved=[["Marie Curie was born in Poland."]],
        )
        out = qa_scorer.aggregate(cases, results)
        assert out["hallucination_rate"] == pytest.approx(1.0)

    def test_empty_cases_returns_zero_metrics(self):
        from src.evaluation import qa_scorer
        out = qa_scorer.aggregate([], [])
        assert out == {"exact_match": 0.0, "token_f1": 0.0}


# ---------------------------------------------------------------------------
# run - composer
# ---------------------------------------------------------------------------


class TestRunComposer:
    def test_returns_metric_dict(self):
        from src.evaluation import qa_scorer
        out = qa_scorer.run(
            examples=HOTPOT_EXAMPLES,
            retriever=_retriever(),
            llm=_llm("Final Answer: Warsaw"),
            prompt_type="standard_qa",
        )
        assert {"exact_match", "token_f1", "hallucination_rate"} <= out.keys()
        assert "recall_at_k" not in out
        for v in out.values():
            assert 0.0 <= v <= 1.0
