"""Verify context composition symmetry for r=0 (clean) and r=1 (poisoned).

Design contract:
    passages = top-(k - n_gold) retrieved distractors
             + n_gold injected evidence  ← original gold at r=0, at r=1

The structure (lengths, retriever k argument, passage order) must be identical
at both conditions; only the injected content differs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.evaluation.scorer import prepare_cases as fever_prepare_cases
from src.evaluation.qa_scorer import prepare_cases as hotpot_prepare_cases

K = 10  # matches new config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retriever(returned_passages: list[str], k: int = K) -> MagicMock:
    r = MagicMock()
    r.k = k
    r.retrieve.return_value = returned_passages
    return r


# ---------------------------------------------------------------------------
# FEVER — symmetric structure
# ---------------------------------------------------------------------------

_DISTRACTORS = [f"distractor_{i}" for i in range(K)]

_CLEAN_FEVER = [
    {"claim": "Paris is in France.", "evidence": ["Paris is located in France."], "label": "SUPPORTS"},
    {"claim": "Rome is in Italy.",   "evidence": ["Rome is the capital of Italy.", "It is an ancient city."], "label": "SUPPORTS"},
]
_POISONED_FEVER = [
    {"claim": "Paris is in France.", "evidence": ["Paris is NOT in France."],          "label": "SUPPORTS",
     "poisoned_positions": {0}},
    {"claim": "Rome is in Italy.",   "evidence": ["Rome is NOT in Italy.", "Unrelated."], "label": "SUPPORTS",
     "poisoned_positions": {0, 1}},
]


class TestFeverContextComposition:

    def test_same_total_passage_count(self):
        """Both conditions produce the same total number of passages per case."""
        retriever = _retriever(_DISTRACTORS[:K - 1])  # generous supply
        cases_r0 = fever_prepare_cases(_CLEAN_FEVER, retriever)
        cases_r1 = fever_prepare_cases(_POISONED_FEVER, retriever)
        for c0, c1 in zip(cases_r0, cases_r1):
            assert len(c0.passages) == len(c1.passages)

    def test_retrieved_count_is_k_minus_n_gold(self):
        """Retriever is called with k - n_gold, not with full k."""
        for ex in _CLEAN_FEVER:
            r = _retriever(_DISTRACTORS)
            fever_prepare_cases([ex], r)
            _, call_kwargs = r.retrieve.call_args
            assert call_kwargs["k"] == K - len(ex["evidence"])

    def test_retrieved_distractors_same_in_both_conditions(self):
        """Same distractors are retrieved at r=0 and r=1 (same mock retriever)."""
        fixed = ["d_A", "d_B", "d_C", "d_D", "d_E", "d_F", "d_G", "d_H", "d_I"]
        r0 = _retriever(fixed)
        r1 = _retriever(fixed)
        cases_r0 = fever_prepare_cases(_CLEAN_FEVER, r0)
        cases_r1 = fever_prepare_cases(_POISONED_FEVER, r1)
        for c0, c1, ex in zip(cases_r0, cases_r1, _CLEAN_FEVER):
            n_gold = len(ex["evidence"])
            k_ret = K - n_gold
            # retrieved portion is equal
            assert c0.passages[:k_ret] == c1.passages[:k_ret] == fixed[:k_ret]

    def test_injected_evidence_differs_between_conditions(self):
        """Injected passages differ between r=0 and r=1."""
        r = _retriever(_DISTRACTORS)
        cases_r0 = fever_prepare_cases(_CLEAN_FEVER, r)
        cases_r1 = fever_prepare_cases(_POISONED_FEVER, _retriever(_DISTRACTORS))
        for c0, c1 in zip(cases_r0, cases_r1):
            # injected portion (at the end) must differ
            assert c0.passages != c1.passages

    def test_injected_content_matches_evidence_field(self):
        """Injected passages are exactly example['evidence'] in order."""
        r0 = _retriever(_DISTRACTORS)
        r1 = _retriever(_DISTRACTORS)
        cases_r0 = fever_prepare_cases(_CLEAN_FEVER, r0)
        cases_r1 = fever_prepare_cases(_POISONED_FEVER, r1)
        for c0, ex_clean in zip(cases_r0, _CLEAN_FEVER):
            n_gold = len(ex_clean["evidence"])
            assert c0.passages[-n_gold:] == ex_clean["evidence"]
        for c1, ex_poisoned in zip(cases_r1, _POISONED_FEVER):
            n_gold = len(ex_poisoned["evidence"])
            assert c1.passages[-n_gold:] == ex_poisoned["evidence"]

    def test_prompt_contains_all_passages_in_both_conditions(self):
        """All passages (retrieved + injected) appear in the formatted prompt."""
        r = _retriever(["ret_1", "ret_2", "ret_3", "ret_4",
                         "ret_5", "ret_6", "ret_7", "ret_8", "ret_9"])
        cases_r0 = fever_prepare_cases(_CLEAN_FEVER, r)
        for case in cases_r0:
            for p in case.passages:
                assert p in case.prompts[0]

    def test_total_passages_equals_k_when_enough_distractors(self):
        """Total passages == k when the retriever provides k - n_gold passages."""
        for ex in _CLEAN_FEVER:
            n_gold = len(ex["evidence"])
            k_ret = K - n_gold
            r = _retriever([f"d{i}" for i in range(k_ret)])
            cases = fever_prepare_cases([ex], r)
            assert len(cases[0].passages) == K


# ---------------------------------------------------------------------------
# HotpotQA — symmetric structure
# ---------------------------------------------------------------------------

_CLEAN_HOTPOT = [
    {
        "question": "Where was Curie born?",
        "answer": "Warsaw",
        "supporting_facts": [["Curie", 0], ["Warsaw", 0]],
        "context": [
            ["Curie", ["Curie was born in Warsaw.", "She won two Nobels."]],
            ["Warsaw", ["Warsaw is in Poland.", "It is the capital."]],
            ["Distractor", ["Unrelated text."]],
        ],
    },
    {
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "supporting_facts": [["Shakespeare", 0]],
        "context": [
            ["Shakespeare", ["Shakespeare wrote Hamlet.", "He was English."]],
            ["Distractor", ["Something else."]],
        ],
    },
]

_POISONED_HOTPOT = [
    {
        "question": "Where was Curie born?",
        "answer": "Warsaw",
        "supporting_facts": [["Curie", 0], ["Warsaw", 0]],
        "poisoned_positions": [["Curie", 0]],
        "context": [
            ["Curie", ["Curie was NOT born in Warsaw.", "She won two Nobels."]],  # poisoned
            ["Warsaw", ["Warsaw is in Poland.", "It is the capital."]],
            ["Distractor", ["Unrelated text."]],
        ],
    },
    {
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "supporting_facts": [["Shakespeare", 0]],
        "poisoned_positions": [["Shakespeare", 0]],
        "context": [
            ["Shakespeare", ["Shakespeare did NOT write Hamlet.", "He was English."]],  # poisoned
            ["Distractor", ["Something else."]],
        ],
    },
]


class TestHotpotQAContextComposition:

    def test_same_total_passage_count(self):
        r0 = _retriever(_DISTRACTORS[:K])
        r1 = _retriever(_DISTRACTORS[:K])
        cases_r0 = hotpot_prepare_cases(_CLEAN_HOTPOT, r0)
        cases_r1 = hotpot_prepare_cases(_POISONED_HOTPOT, r1)
        for c0, c1 in zip(cases_r0, cases_r1):
            assert len(c0.passages) == len(c1.passages)

    def test_injected_supporting_passages_differ_between_conditions(self):
        """Poisoned example injects different passage content than clean."""
        r0 = _retriever(["ret_1", "ret_2", "ret_3", "ret_4",
                          "ret_5", "ret_6", "ret_7", "ret_8"])
        r1 = _retriever(["ret_1", "ret_2", "ret_3", "ret_4",
                          "ret_5", "ret_6", "ret_7", "ret_8"])
        cases_r0 = hotpot_prepare_cases(_CLEAN_HOTPOT, r0)
        cases_r1 = hotpot_prepare_cases(_POISONED_HOTPOT, r1)
        for c0, c1 in zip(cases_r0, cases_r1):
            assert c0.passages != c1.passages

    def test_clean_supporting_passages_in_context(self):
        """At r=0 the injected passage is the original supporting text."""
        r = _retriever(_DISTRACTORS)
        cases = hotpot_prepare_cases(_CLEAN_HOTPOT, r)
        # First example: "Curie was born in Warsaw." must appear
        assert "Curie was born in Warsaw. She won two Nobels." in cases[0].passages

    def test_poisoned_supporting_passage_in_context(self):
        """At r=1 the poisoned supporting passage replaces the original."""
        r = _retriever(_DISTRACTORS)
        cases = hotpot_prepare_cases(_POISONED_HOTPOT, r)
        assert "Curie was NOT born in Warsaw. She won two Nobels." in cases[0].passages
        assert "Curie was born in Warsaw. She won two Nobels." not in cases[0].passages
