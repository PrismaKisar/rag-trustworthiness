"""End-to-end RAG robustness pipeline — CLI entry point.

Two dataset paths are supported and selected via ``--dataset``:

- ``fever``     → load_fever / poison_dataset / scorer (claim verification)
- ``hotpotqa``  → load_hotpotqa / poison_hotpotqa / qa_scorer (multi-hop QA)

Each path uses its native schema and metrics; no normalisation is forced.
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable

import yaml

from src.data.fever_loader import load_fever
from src.data.hotpotqa_loader import load_hotpotqa
from src.data.hotpotqa_poisoner import poison_hotpotqa
from src.data.poisoner import poison_dataset
from src.evaluation import qa_scorer
from src.evaluation.scorer import run as run_scorer
from src.generation.llm_client import HuggingFaceClient
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


@dataclass(frozen=True)
class DatasetRunner:
    """Single registration point for one dataset.

    runner: (cfg, n, poison_rate, seed, strategy, retriever, llm, prompt_type, sc_runs) -> metrics
    default_prompt_fn: resolves the default prompt-type string from the config dict.
    """
    runner: Callable[..., dict[str, float]]
    default_prompt_fn: Callable[[dict], str]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_llm(model: str, cfg: dict):
    cache_dir = os.path.join(cfg["cache"]["dir"], cfg["cache"]["llm_subdir"])
    temperature = cfg["models"]["temperature"]
    return HuggingFaceClient(model=model, temperature=temperature, cache_dir=cache_dir)


def _run_fever(examples, retriever, llm, prompt_type, cfg, seed, sc_runs):
    return run_scorer(
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        distractor_pool_size=cfg["retrieval"]["distractor_pool_size"],
        max_tokens_by_prompt=cfg["prompts"]["max_tokens"],
        seed=seed,
        self_consistency_runs=sc_runs,
    )


def _run_hotpotqa(examples, retriever, llm, prompt_type, cfg, seed, sc_runs):
    return qa_scorer.run(
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        max_tokens_by_prompt=cfg["prompts"]["max_tokens"],
        seed=seed,
        self_consistency_runs=sc_runs,
    )


def _fever_runner(cfg, n, poison_rate, seed, strategy, retriever, llm, prompt_type, sc_runs):
    examples = load_fever(cfg["dataset"]["fever_dev"], max_examples=n)
    if poison_rate > 0.0:
        examples = poison_dataset(
            examples, poison_rate=poison_rate, seed=seed, strategy=strategy, llm=llm,
        )
    return _run_fever(examples, retriever, llm, prompt_type, cfg, seed, sc_runs)


def _hotpotqa_runner(cfg, n, poison_rate, seed, strategy, retriever, llm, prompt_type, sc_runs):
    examples = load_hotpotqa(cfg["dataset"]["hotpotqa_dev"], max_examples=n)
    if poison_rate > 0.0:
        examples = poison_hotpotqa(examples, poison_rate=poison_rate, seed=seed)
    return _run_hotpotqa(examples, retriever, llm, prompt_type, cfg, seed, sc_runs)


_DATASET_REGISTRY: dict[str, DatasetRunner] = {
    "fever": DatasetRunner(
        runner=_fever_runner,
        default_prompt_fn=lambda cfg: cfg["prompts"]["default"],
    ),
    "hotpotqa": DatasetRunner(
        runner=_hotpotqa_runner,
        default_prompt_fn=lambda cfg: cfg["prompts"].get("default_qa", "standard_qa"),
    ),
}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="RAG robustness pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", default=None, choices=list(_DATASET_REGISTRY))
    parser.add_argument("--n", type=int, default=None, help="Number of examples (overrides config)")
    parser.add_argument("--poison_rate", type=float, default=None, help="Fraction of evidence replaced")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["opposite_label", "llm_negation"],
        help="Poisoning strategy (overrides poisoning.strategy in config)",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument(
        "--prompt_type",
        default=None,
        choices=["standard", "chain_of_thought", "vigilant", "standard_qa", "cot_qa", "vigilant_qa"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--self_consistency_runs", type=int, default=None,
                        help="Inference runs per claim for self-consistency")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg["seed"]
    n = args.n if args.n is not None else cfg["evaluation"]["n_examples"]
    poison_rate = args.poison_rate if args.poison_rate is not None else cfg["poisoning"]["poison_rate"]
    strategy = args.strategy if args.strategy is not None else cfg["poisoning"].get("strategy", "opposite_label")
    model = args.model if args.model is not None else cfg["models"]["default"]
    dataset = args.dataset if args.dataset is not None else cfg["dataset"].get("default", "fever")
    sc_runs = (args.self_consistency_runs if args.self_consistency_runs is not None
               else cfg["evaluation"].get("self_consistency_runs", 1))

    emb_cache = os.path.join(cfg["cache"]["dir"], cfg["cache"]["embeddings_subdir"])
    embedder = Embedder(
        model_name=cfg["retrieval"]["embedding_model"],
        cache_dir=emb_cache,
    )
    retriever = Retriever(embedder=embedder, k=cfg["retrieval"]["k"])
    llm = _build_llm(model, cfg)

    with embedder, llm:
        ds = _DATASET_REGISTRY[dataset]
        prompt_type = args.prompt_type if args.prompt_type is not None else ds.default_prompt_fn(cfg)
        metrics = ds.runner(cfg, n, poison_rate, seed, strategy, retriever, llm, prompt_type, sc_runs)

    run_cfg = {
        "dataset": dataset,
        "n": n,
        "poison_rate": poison_rate,
        "strategy": strategy,
        "model": model,
        "prompt_type": prompt_type,
        "seed": seed,
        "self_consistency_runs": sc_runs,
    }
    print(json.dumps({"config": run_cfg, "metrics": metrics}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
