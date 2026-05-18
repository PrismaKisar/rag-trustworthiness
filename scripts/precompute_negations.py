"""Pre-compute LLM negations for FEVER and HotpotQA.

Run once before experiments (from the project root):

    python scripts/precompute_negations.py

Outputs:
    data/fever/dev_poisoned.jsonl      — all FEVER dev examples with evidence negated
    data/hotpotqa/dev_poisoned.jsonl   — all HotpotQA dev examples with one hop negated

Using a single fixed negation model ensures all evaluated models face the same
attack data, enabling fair cross-model comparison (see Pan et al. 2023, §4.3).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml

# Allow running from project root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fever_loader import load_fever
from src.data.hotpotqa_loader import load_hotpotqa
from src.data.hotpotqa_poisoner import poison_hotpotqa
from src.data.poisoner import poison_dataset
from src.generation.llm_client import HuggingFaceClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _save_jsonl(examples: list[dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples -> {path}")


def _process_fever(cfg: dict, llm, seed: int, n: int | None) -> None:
    src_path = cfg["dataset"]["fever_dev"]
    dst_path = cfg["dataset"]["fever_dev_poisoned"]

    print(f"\n[FEVER] Loading from {src_path} ...")
    examples = load_fever(src_path, max_examples=n)
    print(f"[FEVER] Negating {len(examples)} examples ...")
    poisoned = poison_dataset(examples, poison_rate=1.0, seed=seed, llm=llm)
    _save_jsonl(poisoned, dst_path)


def _process_hotpotqa(cfg: dict, llm, seed: int, n: int | None) -> None:
    src_path = cfg["dataset"]["hotpotqa_dev"]
    dst_path = cfg["dataset"]["hotpotqa_dev_poisoned"]

    print(f"\n[HotpotQA] Loading from {src_path} ...")
    examples = load_hotpotqa(src_path, max_examples=n)
    print(f"[HotpotQA] Negating {len(examples)} examples ...")
    poisoned = poison_hotpotqa(examples, poison_rate=1.0, seed=seed, llm=llm)
    _save_jsonl(poisoned, dst_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute LLM negations for FEVER and HotpotQA dev sets."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--dataset",
        choices=["fever", "hotpotqa", "all"],
        default="all",
        help="Which dataset(s) to process (default: all).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Negation model ID (overrides config negation.model).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Max examples per dataset (omit for full dev set).",
    )
    args = parser.parse_args(argv)

    # Resolve the project root from the config path so the script works
    # regardless of the caller's working directory (e.g. from a notebook).
    config_abs = os.path.abspath(args.config)
    project_root = os.path.dirname(os.path.dirname(config_abs))
    os.chdir(project_root)

    cfg = _load_config(config_abs)
    seed = args.seed if args.seed is not None else cfg["seed"]
    model = args.model if args.model is not None else cfg["negation"]["model"]
    cache_dir = os.path.join(cfg["cache"]["dir"], cfg["cache"]["llm_subdir"])
    temperature = cfg["models"]["temperature"]

    print(f"Negation model : {model}")
    print(f"Seed           : {seed}")
    print(f"Dataset(s)     : {args.dataset}")
    if args.n is not None:
        print(f"Max examples   : {args.n} (dev mode)")

    with HuggingFaceClient(model=model, temperature=temperature, cache_dir=cache_dir) as llm:
        if args.dataset in ("fever", "all"):
            _process_fever(cfg, llm, seed=seed, n=args.n)
        if args.dataset in ("hotpotqa", "all"):
            _process_hotpotqa(cfg, llm, seed=seed, n=args.n)

    print("\nDone. Pre-computed negations are ready for experiments.")


if __name__ == "__main__":
    main()
