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
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES) as mock_load,
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
    ):
        # Make embedder produce deterministic dummy vectors
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        metrics = main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
            "--seed", "42",
            "--self_consistency_runs", "1",
        ])

    required_keys = {"accuracy", "macro_f1", "hallucination_rate", "precision_at_k"}
    assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"

    for key in required_keys:
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} not in [0, 1]"

    assert mock_load.called


def test_pipeline_poison_rate_override(mock_llm, tmp_path):
    """Poisoner is invoked when poison_rate > 0."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES),
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
        patch("src.pipeline.poison_dataset", wraps=lambda ex, **kw: ex) as mock_poison,
    ):
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        mock_llm.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]

        main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.5",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
            "--self_consistency_runs", "1",
        ])

    mock_poison.assert_called_once()
    _, kwargs = mock_poison.call_args
    assert kwargs["poison_rate"] == 0.5


def test_pipeline_passes_strategy_to_poisoner(mock_llm):
    """Pipeline must forward poisoning.strategy and llm to poison_dataset."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES),
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
        patch("src.pipeline.poison_dataset", wraps=lambda ex, **kw: ex) as mock_poison,
    ):
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        mock_llm.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]

        main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.5",
            "--strategy", "llm_negation",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
            "--self_consistency_runs", "1",
        ])

    mock_poison.assert_called_once()
    _, kwargs = mock_poison.call_args
    assert kwargs["strategy"] == "llm_negation"
    assert kwargs.get("llm") is mock_llm


def test_pipeline_no_poison_skips_poisoner(mock_llm):
    """Poisoner is NOT called when poison_rate == 0.0."""
    with (
        patch("src.pipeline.load_fever", return_value=FAKE_EXAMPLES),
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
        patch("src.pipeline.poison_dataset") as mock_poison,
    ):
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        mock_llm.complete.side_effect = [f"Final Label: {lbl}" for lbl in _LABEL_CYCLE]

        main([
            "--config", "configs/config.yaml",
            "--n", "5",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard",
            "--self_consistency_runs", "1",
        ])

    mock_poison.assert_not_called()


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
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        metrics = main([
            "--config", "configs/config.yaml",
            "--dataset", "hotpotqa",
            "--n", "2",
            "--poison_rate", "0.0",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard_qa",
            "--seed", "42",
            "--self_consistency_runs", "1",
        ])

    mock_load.assert_called_once()
    mock_fever_load.assert_not_called()
    assert {"exact_match", "token_f1", "precision_at_k"} <= metrics.keys()
    for key in ("exact_match", "token_f1", "precision_at_k"):
        assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# Candidate #3: DatasetRunner deepened - loader + task + poisoner fields
# ---------------------------------------------------------------------------


class _FakeCase:
    prompts = ["prompt_0"]
    max_tokens = 64


class _TraceTask:
    def build_cases(self, examples, retriever, prompt_type, sc_runs, seed, **kwargs):
        return []

    def parse_result(self, case_index, raw_runs):
        return None

    def compute_metrics(self, cases, results, prompt_type):
        return {"tracer_metric": 0.5}


class TestDatasetRunnerNewInterface:
    def test_loader_field_called_and_metrics_returned(self):
        """DatasetRunner dispatches via loader + task; loader provides examples."""
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
                embedder_instance = MagicMock()
                embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
                    (len(texts), 384)
                ).astype("float32")
                embedder_instance.embedding_dim = 384
                MockEmbedder.return_value = embedder_instance

                result = main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_trace",
                    "--n", "1",
                    "--poison_rate", "0.0",
                    "--seed", "42",
                    "--self_consistency_runs", "1",
                ])

            mock_loader.assert_called_once()
            assert result == {"tracer_metric": 0.5}
        finally:
            del _DATASET_REGISTRY["_trace"]

    def test_poisoner_field_called_when_poison_rate_positive(self):
        """DatasetRunner.poisoner is invoked and receives poison_rate when rate > 0."""
        import numpy as np
        from src.pipeline import DatasetRunner, _DATASET_REGISTRY

        stub_examples = [{"q": "x"}]
        mock_loader = MagicMock(return_value=stub_examples)
        mock_poisoner = MagicMock(return_value=stub_examples)

        _DATASET_REGISTRY["_poison_test"] = DatasetRunner(
            loader=mock_loader,
            task=_TraceTask(),
            default_prompt_fn=lambda cfg: "standard",
            poisoner=mock_poisoner,
        )
        try:
            with (
                patch("src.pipeline.Embedder") as MockEmbedder,
                patch("src.pipeline._build_llm", return_value=MagicMock()),
            ):
                embedder_instance = MagicMock()
                embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
                    (len(texts), 384)
                ).astype("float32")
                embedder_instance.embedding_dim = 384
                MockEmbedder.return_value = embedder_instance

                main([
                    "--config", "configs/config.yaml",
                    "--dataset", "_poison_test",
                    "--n", "1",
                    "--poison_rate", "0.5",
                    "--seed", "42",
                    "--self_consistency_runs", "1",
                ])

            mock_poisoner.assert_called_once()
            _, kwargs = mock_poisoner.call_args
            assert kwargs.get("poison_rate") == 0.5
        finally:
            del _DATASET_REGISTRY["_poison_test"]


def test_custom_dataset_registered_and_dispatched():
    """A DatasetRunner added to _DATASET_REGISTRY is dispatched through main()."""
    import numpy as np
    from src.pipeline import DatasetRunner, _DATASET_REGISTRY

    metrics_stub = {"custom_metric": 0.75}
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
            embedder_instance = MagicMock()
            embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
                (len(texts), 384)
            ).astype("float32")
            embedder_instance.embedding_dim = 384
            MockEmbedder.return_value = embedder_instance

            result = main([
                "--config", "configs/config.yaml",
                "--dataset", "custom",
                "--n", "1",
                "--poison_rate", "0.0",
                "--seed", "42",
                "--self_consistency_runs", "1",
            ])

        mock_loader.assert_called_once()
        assert result == {"tracer_metric": 0.5}
    finally:
        del _DATASET_REGISTRY["custom"]


def test_pipeline_hotpotqa_poisoner_invoked(tmp_path):
    """When dataset=hotpotqa and poison_rate>0, the HotpotQA poisoner runs."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = ["Answer: x"] * 4

    with (
        patch("src.pipeline.load_hotpotqa", return_value=FAKE_HOTPOT_EXAMPLES),
        patch("src.pipeline.Embedder") as MockEmbedder,
        patch("src.pipeline._build_llm", return_value=mock_llm),
        patch("src.pipeline.poison_hotpotqa", wraps=lambda ex, **kw: ex) as mock_poison,
        patch("src.pipeline.poison_dataset") as mock_fever_poison,
    ):
        import numpy as np

        embedder_instance = MagicMock()
        embedder_instance.encode.side_effect = lambda texts: np.random.default_rng(0).random(
            (len(texts), 384)
        ).astype("float32")
        embedder_instance.embedding_dim = 384
        MockEmbedder.return_value = embedder_instance

        main([
            "--config", "configs/config.yaml",
            "--dataset", "hotpotqa",
            "--n", "2",
            "--poison_rate", "0.5",
            "--model", "claude-haiku-4-5-20251001",
            "--prompt_type", "standard_qa",
            "--self_consistency_runs", "1",
        ])

    mock_poison.assert_called_once()
    mock_fever_poison.assert_not_called()
    _, kwargs = mock_poison.call_args
    assert kwargs["poison_rate"] == 0.5
