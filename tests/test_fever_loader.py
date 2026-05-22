"""Tests for src/data/fever_loader.py - no real FEVER data required."""

import json
from pathlib import Path

import pytest

from src.data.fever_loader import VALID_LABELS, load_fever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEVER_ROWS = [
    {
        "id": 1,
        "verifiable": "VERIFIABLE",
        "label": "SUPPORTS",
        "claim": "Alice appeared in Wonderland.",
        "evidence": ["Alice was a fictional character.", "She visited Wonderland."],
    },
    {
        "id": 2,
        "verifiable": "VERIFIABLE",
        "label": "REFUTES",
        "claim": "Bob never visited Paris.",
        "evidence": ["He visited Paris many times."],
    },
    {
        "id": 3,
        "verifiable": "NOT VERIFIABLE",
        "label": "NOT ENOUGH INFO",
        "claim": "Carol might exist.",
        "evidence": [],
    },
]

@pytest.fixture()
def fever_file(tmp_path: Path) -> Path:
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in FEVER_ROWS), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_list(fever_file):
    result = load_fever(str(fever_file))
    assert isinstance(result, list)
    assert len(result) == len(FEVER_ROWS)


def test_required_keys(fever_file):
    result = load_fever(str(fever_file))
    for item in result:
        assert set(item.keys()) == {"claim", "evidence", "label"}


def test_claim_is_str(fever_file):
    for item in load_fever(str(fever_file)):
        assert isinstance(item["claim"], str) and item["claim"]


def test_evidence_preserved_as_list_of_str(fever_file):
    """Pre-resolved evidence (list[str]) must round-trip unchanged."""
    result = load_fever(str(fever_file))
    by_label = {r["label"]: r["evidence"] for r in result}
    assert by_label["SUPPORTS"] == [
        "Alice was a fictional character.",
        "She visited Wonderland.",
    ]
    assert by_label["REFUTES"] == ["He visited Paris many times."]
    assert by_label["NOT ENOUGH INFO"] == []
    for ev in result:
        assert isinstance(ev["evidence"], list)
        assert all(isinstance(s, str) for s in ev["evidence"])


def test_label_valid(fever_file):
    for item in load_fever(str(fever_file)):
        assert item["label"] in VALID_LABELS


def test_max_examples(fever_file):
    result = load_fever(str(fever_file), max_examples=2)
    assert len(result) == 2


def test_wiki_pages_dir_kwarg_rejected(fever_file, tmp_path: Path):
    """Regression guard: load_fever no longer accepts wiki_pages_dir."""
    with pytest.raises(TypeError, match="wiki_pages_dir"):
        load_fever(str(fever_file), wiki_pages_dir=str(tmp_path))


def test_old_fever_format_ignored(tmp_path: Path):
    """Legacy nested evidence (list[list[list]]) is not dereferenced - falls back to []."""
    row = {
        "id": 99,
        "label": "SUPPORTS",
        "claim": "Legacy evidence row.",
        "evidence": [[[None, None, "Alice", 0]]],
    }
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")
    result = load_fever(str(p))
    assert result[0]["evidence"] == []


def test_unknown_label_normalized(tmp_path: Path):
    """Labels outside VALID_LABELS should fall back to NOT ENOUGH INFO."""
    bad_row = {"id": 99, "label": "UNKNOWN", "claim": "Test.", "evidence": []}
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(bad_row), encoding="utf-8")
    result = load_fever(str(p))
    assert result[0]["label"] == "NOT ENOUGH INFO"
