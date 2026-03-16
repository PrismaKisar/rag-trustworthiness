"""Load FEVER dev/train splits and dereference evidence sentence IDs to text.

Evidence format per claim (Thorne et al., 2018 – FEVER dataset):
    evidence: list[annotator_group]
    annotator_group: list[[annotator_id, ev_id, wiki_page, sent_id]]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def _build_wiki_index(wiki_pages_dir: str) -> dict[str, dict[int, str]]:
    """Scan all *.jsonl files in wiki_pages_dir and build page -> {sent_id: text}."""
    index: dict[str, dict[int, str]] = {}
    for fpath in Path(wiki_pages_dir).glob("*.jsonl"):
        with open(fpath, encoding="utf-8") as fh:
            for raw in fh:
                doc = json.loads(raw)
                sentences: dict[int, str] = {}
                for line in doc.get("lines", "").split("\n"):
                    parts = line.split("\t", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        sentences[int(parts[0])] = parts[1]
                if sentences:
                    index[doc["id"]] = sentences
    logger.info("Wiki index built: %d pages", len(index))
    return index


def _deref_evidence(
    raw_evidence: list,
    index: dict[str, dict[int, str]],
) -> list[str]:
    """Return deduplicated sentence strings for all annotator groups."""
    seen: set[tuple] = set()
    texts: list[str] = []
    for group in raw_evidence:
        for _, _, wiki_page, sent_id in group:
            if wiki_page is None or sent_id is None:
                continue
            key = (wiki_page, sent_id)
            if key in seen:
                continue
            seen.add(key)
            text = index.get(wiki_page, {}).get(int(sent_id))
            if text:
                texts.append(text)
    return texts


def load_fever(
    path: str,
    wiki_pages_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """Load FEVER examples from *path*.

    Args:
        path: Path to a FEVER *.jsonl* split file.
        wiki_pages_dir: Directory containing Wikipedia dump *.jsonl* files.
                        When None, evidence is returned as an empty list.
        max_examples: If set, return at most this many examples.

    Returns:
        List of dicts with keys ``claim`` (str), ``evidence`` (list[str]),
        ``label`` (str in VALID_LABELS).
    """
    index: dict[str, dict[int, str]] = {}
    if wiki_pages_dir is not None:
        index = _build_wiki_index(wiki_pages_dir)

    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if max_examples is not None and len(examples) >= max_examples:
                break
            item = json.loads(raw)
            label = item.get("label", "NOT ENOUGH INFO")
            if label not in VALID_LABELS:
                label = "NOT ENOUGH INFO"

            evidence: list[str] = []
            if index:
                try:
                    evidence = _deref_evidence(item.get("evidence", []), index)
                except Exception:
                    evidence = []

            examples.append(
                {"claim": item["claim"], "evidence": evidence, "label": label}
            )

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    dev_path = sys.argv[1] if len(sys.argv) > 1 else "data/fever/dev.jsonl"
    wiki_dir = sys.argv[2] if len(sys.argv) > 2 else None
    samples = load_fever(dev_path, wiki_pages_dir=wiki_dir, max_examples=5)
    for s in samples:
        print(s)
