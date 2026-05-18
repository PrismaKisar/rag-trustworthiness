"""Integration test for src/pipeline.py.

Uses 5 FEVER-format examples with a mocked LLM client so no real API calls
are made. Verifies that the pipeline runs end-to-end and returns valid metrics.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EXAMPLES = [
    {
        "claim": "Marie Curie was born in Poland.",
        "evidence": ["Marie Curie was born in Warsaw, Poland."],
        "label": "SUPPORTS",
    },
    {
        "claim": "The Eiffel Tower is in Berlin.",
        "evidence": ["The Eiffel Tower is located in Paris, France."],
        "label": "REFUTES",
    },
    {
        "claim": "Water boils at 100 degrees Celsius at sea level.",
        "evidence": ["Water boils at 100 °C (212 °F) at standard atmospheric pressure."],
        "label": "SUPPORTS",
    },
    {
        "claim": "Shakespeare invented the telescope.",
        "evidence": ["The telescope was invented by Hans Lippershey in 1608."],
        "label": "REFUTES",
    },
    {
        "claim": "The moon is made of cheese.",
        "evidence": [],
        "label": "NOT ENOUGH INFO",
    },
]

_LABEL_CYCLE = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]


def _make_embedder_mock():
    import numpy as np
    embedder_instance = MagicMock()
    embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
        (len(texts), 384)
    ).astype("float32")
    embedder_instance.embedding_dim = 384
    return embedder_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm():
    """LLM that cycles through labels deterministically."""
    client = MagicMock()
    client.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]
    return client


def test_pipeline_returns_valid_metrics(mock_llm, tmp_path):
    """Pipeline runs end-to-end with 5 mocked examples and returns metric keys."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES),
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        MockEmbedder.return_value = _make_embedder_mock()

        metrics = main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
            "--seed", "42",
        ])

    required_keys = {"accuracy", "macro_f1", "hallucination_rate"}
    assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"
    assert "recall_at_k" not in metrics

    for key in required_keys:
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} not in [0, 1]"


def test_pipeline_poisoned_loader_used_when_poison_rate_positive(mock_llm):
    """When poison_rate > 0, the pre-computed poisoned file is loaded (not the poisoner)."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES) as mock_load,
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        MockEmbedder.return_value = _make_embedder_mock()
        mock_llm.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]

        main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "1.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
        ])

    mock_load.assert_called_once()
    loaded_path = mock_load.call_args[0][0]
    assert "poisoned" in loaded_path, f"Expected poisoned path, got: {loaded_path}"


def test_pipeline_clean_loader_used_when_poison_rate_zero(mock_llm):
    """When poison_rate == 0.0, the clean dev file is loaded."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES) as mock_load,
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        MockEmbedder.return_value = _make_embedder_mock()
        mock_llm.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]

        main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
        ])

    mock_load.assert_called_once()
    loaded_path = mock_load.call_args[0][0]
    assert "poisoned" not in loaded_path, f"Expected clean path, got: {loaded_path}"


# ---------------------------------------------------------------------------
# HotpotQA routing
# ---------------------------------------------------------------------------

FAKE_HOTPOT_EXAMPLES = [
    {
        "question": "Where was Marie Curie born?",
        "answer": "Warsaw",
        "supporting_facts": [["Marie Curie", 0]],
        "context": [
            ["Marie Curie", ["Marie Curie was born in Warsaw.", "She won Nobel prizes."]],
            ["Distractor", ["Unrelated content here.", "More filler."]],
        ],
    },
    {
        "question": "Who wrote 2001: A Space Odyssey?",
        "answer": "Arthur C. Clarke",
        "supporting_facts": [["Clarke", 0]],
        "context": [
            ["Clarke", ["Arthur C. Clarke was a British author.", "He co-wrote the screenplay."]],
            ["Distractor", ["Unrelated content.", "More filler text."]],
        ],
    },
]


def test_pipeline_routes_to_hotpotqa(tmp_path):
    """--dataset hotpotqa loads HotpotQA, runs qa_scorer, returns QA metrics."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = ["Answer: Warsaw", "Answer: Arthur C. Clarke"]

    with (
        patch("src.pipeline.load_hotpotqa", return_value=FAKE_HOTPOT_EXAMPLES) as mock_load,
        patch("src.pipeline.load_fever") as mock_fever_load,
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        MockEmbedder.return_value = _make_embedder_mock()

        metrics = main([
            "--config", "configs/config.yaml",
            "--dataset", "hotpotqa",
            "--n", "2",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard_qa",
            "--seed", "42",
        ])

    mock_load.assert_called_once()
    mock_fever_load.assert_not_called()
    assert {"exact_match", "token_f1"} <= metrics.keys()
    assert "recall_at_k" not in metrics
    for key in ("exact_match", "token_f1"):
        assert 0.0 <= metrics[key] <= 1.0


def test_pipeline_hotpotqa_poisoned_loader_used(tmp_path):
    """When dataset=hotpotqa and poison_rate>0, the pre-computed poisoned file is loaded."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = ["Answer: x"] * 4

    with (
        patch("src.pipeline.load_hotpotqa", return_value=FAKE_HOTPOT_EXAMPLES) as mock_load,
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        MockEmbedder.return_value = _make_embedder_mock()

        main([
            "--config", "configs/config.yaml",
            "--dataset", "hotpotqa",
            "--n", "2",
            "--poison_rate", "1.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard_qa",
        ])

    mock_load.assert_called_once()
    loaded_path = mock_load.call_args[0][0]
    assert "poisoned" in loaded_path, f"Expected poisoned path, got: {loaded_path}"


# ---------------------------------------------------------------------------
# DatasetRunner tests
# ---------------------------------------------------------------------------


class _TraceTask:
    def build_cases(self, examples, retriever, prompt_type, seed, **kwargs):
        return []

    def parse_result(self, case_index, raw_runs):
        return None

    def compute_metrics(self, cases, results, prompt_type):
        return {"tracer_metric": 0.5}


class TestDatasetRunnerNewInterface:
    def test_loader_field_called_and_metrics_returned(self):
        import numpy as np
        from src.pipeline import DatasetRunner, _DATASET_REGISTRY

        mock_loader = MagicMock(return_value=[{"q": "x"}])

        _DATASET_REGISTRY["_trace"] = DatasetRunner(
            loader=mock_loader,
            task=_TraceTask(),
            default_prompt_fn=lambda cfg: "standard",
        )
        try:
            with (
                patch("src.pipeline.Embedder") as MockEmbedder,
                patch("src.pipeline._build_llm", return_value=MagicMock()),
            ):
                MockEmbedder.return_value = _make_embedder_mock()

                result = main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_trace",
                    "--n", "1",
                    "--poison_rate", "0.0",
                    "--seed", "42",
                ])

            mock_loader.assert_called_once()
            assert result == {"tracer_metric": 0.5}
        finally:
            del _DATASET_REGISTRY["_trace"]

    def test_poisoned_loader_called_when_poison_rate_positive(self):
        import numpy as np
        from src.pipeline import DatasetRunner, _DATASET_REGISTRY

        stub_examples = [{"q": "x"}]
        mock_loader = MagicMock(return_value=stub_examples)
        mock_poisoned_loader = MagicMock(return_value=stub_examples)

        _DATASET_REGISTRY["_poison_test"] = DatasetRunner(
            loader=mock_loader,
            task=_TraceTask(),
            default_prompt_fn=lambda cfg: "standard",
            poisoned_loader=mock_poisoned_loader,
        )
        try:
            with (
                patch("src.pipeline.Embedder") as MockEmbedder,
                patch("src.pipeline._build_llm", return_value=MagicMock()),
            ):
                MockEmbedder.return_value = _make_embedder_mock()

                main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_poison_test",
                    "--n", "1",
                    "--poison_rate", "1.0",
                    "--seed", "42",
                ])

            mock_poisoned_loader.assert_called_once()
            mock_loader.assert_not_called()
        finally:
            del _DATASET_REGISTRY["_poison_test"]

    def test_loader_called_not_poisoned_loader_at_rate_zero(self):
        import numpy as np
        from src.pipeline import DatasetRunner, _DATASET_REGISTRY

        stub_examples = [{"q": "x"}]
        mock_loader = MagicMock(return_value=stub_examples)
        mock_poisoned_loader = MagicMock(return_value=stub_examples)

        _DATASET_REGISTRY["_clean_test"] = DatasetRunner(
            loader=mock_loader,
            task=_TraceTask(),
            default_prompt_fn=lambda cfg: "standard",
            poisoned_loader=mock_poisoned_loader,
        )
        try:
            with (
                patch("src.pipeline.Embedder") as MockEmbedder,
                patch("src.pipeline._build_llm", return_value=MagicMock()),
            ):
                MockEmbedder.return_value = _make_embedder_mock()

                main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_clean_test",
                    "--n", "1",
                    "--poison_rate", "0.0",
                    "--seed", "42",
                ])

            mock_loader.assert_called_once()
            mock_poisoned_loader.assert_not_called()
        finally:
            del _DATASET_REGISTRY["_clean_test"]

    def test_no_poisoned_loader_raises_on_poison_rate_positive(self):
        import numpy as np
        from src.pipeline import DatasetRunner, _DATASET_REGISTRY

        _DATASET_REGISTRY["_no_poison"] = DatasetRunner(
            loader=MagicMock(return_value=[{"q": "x"}]),
            task=_TraceTask(),
            default_prompt_fn=lambda cfg: "standard",
            poisoned_loader=None,
        )
        try:
            with (
                patch("src.pipeline.Embedder") as MockEmbedder,
                patch("src.pipeline._build_llm", return_value=MagicMock()),
                pytest.raises(ValueError, match="poisoned"),
            ):
                MockEmbedder.return_value = _make_embedder_mock()

                main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_no_poison",
                    "--n", "1",
                    "--poison_rate", "1.0",
                    "--seed", "42",
                ])
        finally:
            del _DATASET_REGISTRY["_no_poison"]


def test_custom_dataset_registered_and_dispatched():
    import numpy as np
    from src.pipeline import DatasetRunner, _DATASET_REGISTRY

    mock_loader = MagicMock(return_value=[{"q": "x"}])

    custom_runner = DatasetRunner(
        loader=mock_loader,
        task=_TraceTask(),
        default_prompt_fn=lambda cfg: "standard",
    )
    _DATASET_REGISTRY["custom"] = custom_runner
    try:
        with (
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline._build_llm", return_value=MagicMock()),
        ):
            MockEmbedder.return_value = _make_embedder_mock()

            result = main([
                "--config", "configs/config.yaml",
                "--dataset", "custom",
                "--n", "1",
                "--poison_rate", "0.0",
                "--seed", "42",
            ])

        mock_loader.assert_called_once()
        assert result == {"tracer_metric": 0.5}
    finally:
        del _DATASET_REGISTRY["custom"]
