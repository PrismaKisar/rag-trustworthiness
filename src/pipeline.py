"""End-to-end RAG robustness pipeline — CLI entry point."""
import argparse
import json
import os
from pathlib import Path

import yaml

from src.data.fever_loader import load_fever
from src.data.poisoner import poison_dataset
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.llm_client import AnthropicClient, OpenAIClient
from src.evaluation.scorer import run as run_scorer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_llm(model: str, cfg: dict):
    cache_dir = os.path.join(cfg["cache"]["dir"], cfg["cache"]["llm_subdir"])
    temperature = cfg["models"]["temperature"]
    if model.startswith("claude"):
        return AnthropicClient(model=model, temperature=temperature, cache_dir=cache_dir)
    if model.startswith("gpt"):
        return OpenAIClient(model=model, temperature=temperature, cache_dir=cache_dir)
    raise ValueError(f"Unknown model: {model!r}")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="RAG robustness pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--n", type=int, default=None, help="Number of examples (overrides config)")
    parser.add_argument("--poison_rate", type=float, default=None, help="Fraction of evidence replaced")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument(
        "--prompt_type",
        default=None,
        choices=["standard", "chain_of_thought", "vigilant"],
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # CLI overrides take precedence over config values
    seed = args.seed if args.seed is not None else cfg["seed"]
    n = args.n if args.n is not None else cfg["evaluation"]["n_examples"]
    poison_rate = args.poison_rate if args.poison_rate is not None else cfg["poisoning"]["poison_rate"]
    model = args.model if args.model is not None else cfg["models"]["default"]
    prompt_type = args.prompt_type if args.prompt_type is not None else cfg["prompts"]["default"]

    # Load FEVER dev set; wiki-pages dir is optional (falls back to claim-only mode)
    wiki_pages_dir = cfg["dataset"]["fever_pages"]
    if not Path(wiki_pages_dir).exists():
        wiki_pages_dir = None

    examples = load_fever(
        cfg["dataset"]["fever_dev"],
        wiki_pages_dir=wiki_pages_dir,
        max_examples=n,
    )

    if poison_rate > 0.0:
        examples = poison_dataset(examples, poison_rate=poison_rate, seed=seed)

    # Retrieval setup
    emb_cache = os.path.join(cfg["cache"]["dir"], cfg["cache"]["embeddings_subdir"])
    embedder = Embedder(
        model_name=cfg["retrieval"]["embedding_model"],
        cache_dir=emb_cache,
    )
    retriever = Retriever(embedder=embedder, k=cfg["retrieval"]["k"])

    llm = _build_llm(model, cfg)

    metrics = run_scorer(
        examples=examples,
        retriever=retriever,
        llm=llm,
        prompt_type=prompt_type,
        distractor_pool_size=cfg["retrieval"]["distractor_pool_size"],
        seed=seed,
    )

    run_cfg = {
        "n": n,
        "poison_rate": poison_rate,
        "model": model,
        "prompt_type": prompt_type,
        "seed": seed,
    }
    print(json.dumps({"config": run_cfg, "metrics": metrics}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
