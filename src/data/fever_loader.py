"""Load FEVER dev/train splits from pre-resolved jsonl files.

Both dev.jsonl and train.jsonl already contain evidence as lists of strings
(pre-resolved from the original Wikipedia dump at dataset preparation time).
No wiki-pages dereference is needed.

Evidence format per claim:
    evidence: list[str]  - sentence strings, empty for NOT ENOUGH INFO

Attribution: Thorne et al., 2018 - FEVER dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def load_fever(
    path: str,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """Load FEVER examples from *path*.

    Args:
        path: Path to a FEVER *.jsonl* split file.
        max_examples: If set, return at most this many examples.

    Returns:
        List of dicts with keys ``claim`` (str), ``evidence`` (list[str]),
        ``label`` (str in VALID_LABELS).
    """
    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if max_examples is not None and len(examples) >= max_examples:
                break
            item = json.loads(raw)
            label = item.get("label", "NOT ENOUGH INFO")
            if label not in VALID_LABELS:
                label = "NOT ENOUGH INFO"

            raw_ev = item.get("evidence", [])
            evidence: list[str] = raw_ev if raw_ev and isinstance(raw_ev[0], str) else []

            examples.append(
                {"claim": item["claim"], "evidence": evidence, "label": label}
            )

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO)
    dev_path = sys.argv[1] if len(sys.argv) > 1 else "data/fever/dev.jsonl"
    samples = load_fever(dev_path, max_examples=5)
    for s in samples:
        print(s)
