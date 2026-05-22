"""Tests for src/data/download_hotpotqa.py - no network access."""

import json
from pathlib import Path
from unittest.mock import patch


SAMPLE_JSON = [
    {
        "_id": "x1",
        "question": "Q1?",
        "answer": "A1",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [["T", 0]],
        "context": [["T", ["s0", "s1"]]],
    },
    {
        "_id": "x2",
        "question": "Q2?",
        "answer": "A2",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": [["U", 0]],
        "context": [["U", ["s0"]]],
    },
]


def test_convert_writes_one_line_per_example(tmp_path: Path):
    from src.data.download_hotpotqa import convert_json_to_jsonl

    src = tmp_path / "raw.json"
    dst = tmp_path / "dev.jsonl"
    src.write_text(json.dumps(SAMPLE_JSON), encoding="utf-8")

    convert_json_to_jsonl(src, dst)

    lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert [p["question"] for p in parsed] == ["Q1?", "Q2?"]


def test_download_skips_when_target_exists(tmp_path: Path):
    """If the target jsonl already exists, no network call and no overwrite."""
    from src.data import download_hotpotqa

    target = tmp_path / "dev.jsonl"
    target.write_text("already there", encoding="utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        download_hotpotqa.download(target=target, url="http://example/dev.json")

    mock_urlopen.assert_not_called()
    assert target.read_text(encoding="utf-8") == "already there"


def test_download_fetches_and_converts_when_missing(tmp_path: Path):
    """When target does not exist, download() calls urlopen and produces a jsonl file."""
    import json
    from io import BytesIO
    from unittest.mock import MagicMock, patch
    from src.data import download_hotpotqa

    target = tmp_path / "hotpotqa" / "dev.jsonl"
    raw_json = json.dumps([{"_id": "a", "question": "Q?", "answer": "A",
                            "supporting_facts": [], "context": []}])

    mock_response = MagicMock()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read.return_value = raw_json.encode()

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = download_hotpotqa.download(target=target, url="http://example/dev.json")

    assert result == target
    assert target.exists()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["question"] == "Q?"
