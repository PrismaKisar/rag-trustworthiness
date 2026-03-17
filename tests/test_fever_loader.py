"""Tests for src/data/fever_loader.py — no real FEVER data required."""

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
        "evidence": [[[None, None, "Alice", 0]]],
    },
    {
        "id": 2,
        "verifiable": "VERIFIABLE",
        "label": "REFUTES",
        "claim": "Bob never visited Paris.",
        "evidence": [[[None, None, "Bob", 1]]],
    },
    {
        "id": 3,
        "verifiable": "NOT VERIFIABLE",
        "label": "NOT ENOUGH INFO",
        "claim": "Carol might exist.",
        "evidence": [[[None, None, None, None]]],
    },
]

WIKI_PAGES = {
    "Alice": {"id": "Alice", "lines": "0\tAlice was a fictional character.\n1\tShe visited Wonderland."},
    "Bob": {"id": "Bob", "lines": "0\tBob is a person.\n1\tHe visited Paris many times."},
}


@pytest.fixture()
def fever_file(tmp_path: Path) -> Path:
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in FEVER_ROWS), encoding="utf-8")
    return p


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    d = tmp_path / "wiki-pages"
    d.mkdir()
    page_file = d / "pages.jsonl"
    page_file.write_text(
        "\n".join(json.dumps(v) for v in WIKI_PAGES.values()), encoding="utf-8"
    )
    return d


# ---------------------------------------------------------------------------
# Tests – no wiki_pages_dir
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


def test_evidence_is_list_of_str_without_wiki(fever_file):
    for item in load_fever(str(fever_file)):
        assert isinstance(item["evidence"], list)
        assert item["evidence"] == []


def test_label_valid(fever_file):
    for item in load_fever(str(fever_file)):
        assert item["label"] in VALID_LABELS


def test_max_examples(fever_file):
    result = load_fever(str(fever_file), max_examples=2)
    assert len(result) == 2
