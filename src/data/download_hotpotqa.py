"""Idempotent download + jsonl conversion of the HotpotQA dev distractor split.

Usage (from project root):

    .venv/bin/python -m src.data.download_hotpotqa

Skips the network call when the target jsonl already exists. The upstream URL
serves a single JSON array - we stream it once and rewrite each example as a
separate jsonl line so the rest of the pipeline can read it like FEVER.

Attribution: Yang et al. 2018 - HotpotQA dataset; URL from the official
release at https://hotpotqa.github.io/.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DEFAULT_TARGET = Path("data/hotpotqa/dev.jsonl")


def convert_json_to_jsonl(src: Path, dst: Path) -> None:
    """Convert a HotpotQA JSON-array file at *src* into a jsonl file at *dst*."""
    examples = json.loads(Path(src).read_text(encoding="utf-8"))
    with open(dst, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False))
            fh.write("\n")
    logger.info("Wrote %d examples to %s", len(examples), dst)


def download(target: Path = DEFAULT_TARGET, url: str = DEFAULT_URL) -> Path:
    """Download HotpotQA dev distractor split and convert to jsonl at *target*.

    Idempotent: returns immediately if *target* already exists.
    """
    target = Path(target)
    if target.exists():
        logger.info("HotpotQA dev already present at %s - skipping.", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_suffix(".raw.json")

    logger.info("Downloading HotpotQA dev from %s …", url)
    with urllib.request.urlopen(url) as response, open(raw, "wb") as fh:
        fh.write(response.read())

    convert_json_to_jsonl(raw, target)
    raw.unlink(missing_ok=True)
    return target


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    download()
