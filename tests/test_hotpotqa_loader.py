"""Tests for src/data/hotpotqa_loader.py - no real HotpotQA data required."""

import json
from pathlib import Path

import pytest


HOTPOT_ROWS = [
    {
        "_id": "a1",
        "question": "Which country borders both France and Italy?",
        "answer": "Switzerland",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["Switzerland", 0], ["France", 1]],
        "context": [
            ["Switzerland", ["Switzerland is a landlocked country in Europe.", "It borders France, Italy, Germany and Austria."]],
            ["France", ["France is a country in Western Europe.", "It shares borders with Switzerland and Italy."]],
        ],
    },
    {
        "_id": "a2",
        "question": "Who wrote the novel that inspired the 1968 film 2001: A Space Odyssey?",
        "answer": "Arthur C. Clarke",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [["2001: A Space Odyssey", 0], ["Arthur C. Clarke", 0]],
        "context": [
            ["2001: A Space Odyssey", ["The film was based on a short story by Arthur C. Clarke.", "It premiered in 1968."]],
            ["Arthur C. Clarke", ["Arthur C. Clarke was a British science-fiction writer.", "He co-wrote the screenplay for 2001."]],
        ],
    },
]


@pytest.fixture()
def hotpot_file(tmp_path: Path) -> Path:
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in HOTPOT_ROWS), encoding="utf-8")
    return p


def test_returns_list_of_examples(hotpot_file):
    from src.data.hotpotqa_loader import load_hotpotqa

    result = load_hotpotqa(str(hotpot_file))
    assert isinstance(result, list)
    assert len(result) == len(HOTPOT_ROWS)


def test_required_keys(hotpot_file):
    from src.data.hotpotqa_loader import load_hotpotqa

    for item in load_hotpotqa(str(hotpot_file)):
        assert {"question", "answer", "supporting_facts", "context"} <= set(item.keys())


def test_max_examples(hotpot_file):
    from src.data.hotpotqa_loader import load_hotpotqa

    result = load_hotpotqa(str(hotpot_file), max_examples=1)
    assert len(result) == 1
    assert result[0]["question"] == HOTPOT_ROWS[0]["question"]
