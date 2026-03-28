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
