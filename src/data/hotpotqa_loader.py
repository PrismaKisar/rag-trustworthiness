"""Load HotpotQA examples from a jsonl split.

HotpotQA is a multi-hop QA benchmark. Each example carries a question whose
answer requires reasoning across two Wikipedia paragraphs. The schema is kept
faithful to the upstream dataset (Yang et al., 2018) - we do NOT normalise it
to the FEVER claim/evidence/label shape.

Schema per example:
    question:         str
    answer:           str
    supporting_facts: list[[title, sentence_idx]]
    context:          list[[title, [sentence, ...]]]

Attribution: Yang et al., 2018 - HotpotQA dataset.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_hotpotqa(
    path: str,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """Load HotpotQA examples from a jsonl file at *path*."""
    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if max_examples is not None and len(examples) >= max_examples:
                break
            item = json.loads(raw)
            examples.append({
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
            })

    logger.info("Loaded %d HotpotQA examples from %s", len(examples), path)
    return examples
