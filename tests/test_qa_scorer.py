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



def _retriever(passages=("p1", "p2", "p3")):
    r = MagicMock()
    r.retrieve.return_value = list(passages)
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

    def test_single_run_one_prompt(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(), sc_runs=1)
        for case in cases:
            assert len(case.prompts) == 1

    def test_multiple_runs_match_prompt_count(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(), sc_runs=4)
        for case in cases:
            assert len(case.prompts) == 4

    def test_passages_come_from_retriever(self):
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(("x", "y")))
        for case in cases:
            assert case.passages == ["x", "y"]

    def test_gold_passages_include_supporting_paragraphs(self):
        """The gold paragraphs are those whose title appears in supporting_facts."""
        from src.evaluation import qa_scorer
        cases = qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever())
        # For first example: supporting titles are "Marie Curie" and "Warsaw"
        case = cases[0]
        joined_marie = " ".join(HOTPOT_EXAMPLES[0]["context"][0][1])
        joined_warsaw = " ".join(HOTPOT_EXAMPLES[0]["context"][1][1])
        assert joined_marie in case.gold_passages
        assert joined_warsaw in case.gold_passages

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
    def _make_cases(self, sc_runs=1):
        from src.evaluation import qa_scorer
        return qa_scorer.prepare_cases(HOTPOT_EXAMPLES, _retriever(), sc_runs=sc_runs)

    def test_returns_one_result_per_case(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm())
        assert len(results) == len(cases)

    def test_runs_length_matches_prompt_count(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases(sc_runs=3)
        results = qa_scorer.resolve(cases, _llm())
        for case, result in zip(cases, results):
            assert len(result.runs) == len(case.prompts)

    def test_predicted_answer_extracted_from_llm_output(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases()
        results = qa_scorer.resolve(cases, _llm("Final Answer: Warsaw"))
        for result in results:
            assert result.predicted_answer == "Warsaw"

    def test_predicted_answer_majority_across_runs(self):
        from src.evaluation import qa_scorer
        cases = self._make_cases(sc_runs=3)
        llm = MagicMock()
        llm.complete.side_effect = [
            "Answer: Warsaw", "Answer: Warsaw", "Answer: Krakow",
            "Answer: Clarke", "Answer: Clarke", "Answer: Asimov",
        ]
        results = qa_scorer.resolve(cases, llm)
        assert results[0].predicted_answer == "Warsaw"
        assert results[1].predicted_answer == "Clarke"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _make_pair(self, predictions, gold_answers, gold_passages=None, retrieved=None):
        from src.evaluation.cases import QACase, QAResult
        gold_passages = gold_passages or [["g"]] * len(predictions)
        retrieved = retrieved or [["g"]] * len(predictions)
        cases = [
            QACase(
                question=f"q-{i}",
                gold_answer=gold,
                passages=retrieved[i],
                gold_passages=gold_passages[i],
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
        assert {"exact_match", "token_f1", "recall_at_k"} <= out.keys()

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

    def test_recall_at_k_perfect(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(
            ["Warsaw"], ["Warsaw"],
            gold_passages=[["g1", "g2"]],
            retrieved=[["g1", "g2"]],
        )
        out = qa_scorer.aggregate(cases, results)
        assert out["recall_at_k"] == pytest.approx(1.0)

    def test_no_self_consistency_for_single_run(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw"], ["Warsaw"])
        out = qa_scorer.aggregate(cases, results)
        assert "self_consistency" not in out

    def test_includes_hallucination_rate(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(["Warsaw"], ["Warsaw"])
        out = qa_scorer.aggregate(cases, results)
        assert "hallucination_rate" in out
        assert 0.0 <= out["hallucination_rate"] <= 1.0

    def test_hallucination_rate_zero_when_grounded(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(
            ["Warsaw"],
            ["Warsaw"],
            retrieved=[["Warsaw is the capital of Poland."]],
        )
        out = qa_scorer.aggregate(cases, results)
        assert out["hallucination_rate"] == pytest.approx(0.0)

    def test_hallucination_rate_one_when_ungrounded(self):
        from src.evaluation import qa_scorer
        cases, results = self._make_pair(
            ["xyzzyquux"],
            ["Warsaw"],
            retrieved=[["Marie Curie was born in Poland."]],
        )
        out = qa_scorer.aggregate(cases, results)
        assert out["hallucination_rate"] == pytest.approx(1.0)


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
        assert {"exact_match", "token_f1", "recall_at_k"} <= out.keys()
        for v in out.values():
            assert 0.0 <= v <= 1.0
