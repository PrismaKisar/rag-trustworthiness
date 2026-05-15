"""Shared parallel LLM dispatch for the three-phase scorer pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed


def resolve_raw(cases: list, llm, n_workers: int = 4) -> list[list[str]]:
    """Dispatch parallel LLM calls for all prompts in *cases*.

    Args:
        cases: Objects with ``.prompts`` (list[str]) and ``.max_tokens`` (int).
        llm: LLM client with ``.complete(prompt, max_tokens) -> str``.
        n_workers: Thread-pool size.

    Returns:
        Raw response strings grouped by case index, in prompt order.
        ``result[i][j]`` is the response for ``cases[i].prompts[j]``.
    """
    if not cases:
        return []

    tasks = [
        (case_idx, run_idx, prompt, case.max_tokens)
        for case_idx, case in enumerate(cases)
        for run_idx, prompt in enumerate(case.prompts)
    ]

    raw: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        future_to_key = {
            pool.submit(llm.complete, prompt, max_tokens): (case_idx, run_idx)
            for case_idx, run_idx, prompt, max_tokens in tasks
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            raw[key] = future.result()

    return [
        [raw[(case_idx, run_idx)] for run_idx in range(len(case.prompts))]
        for case_idx, case in enumerate(cases)
    ]
