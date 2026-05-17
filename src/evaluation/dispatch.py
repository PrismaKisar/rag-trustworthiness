"""Shared parallel LLM dispatch for the three-phase scorer pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed


def resolve_raw(cases: list, llm, n_workers: int = 4) -> list[list[str]]:
    """Dispatch parallel LLM calls for all prompts in *cases*.

    Args:
        cases: Objects with ``.prompts`` (list[str]) and ``.prompt_type`` (str).
        llm: LLM client with ``.complete(prompt, prompt_type) -> str``.
        n_workers: Thread-pool size.

    Returns:
        Raw response strings grouped by case index, in prompt order.
        ``result[i][j]`` is the response for ``cases[i].prompts[j]``.
    """
    if not cases:
        return []

    tasks = [
        (case_idx, run_idx, prompt, case.prompt_type)
        for case_idx, case in enumerate(cases)
        for run_idx, prompt in enumerate(case.prompts)
    ]

    raw: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        future_to_key = {
            pool.submit(llm.complete, prompt, prompt_type): (case_idx, run_idx)
            for case_idx, run_idx, prompt, prompt_type in tasks
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            raw[key] = future.result()

    return [
        [raw[(case_idx, run_idx)] for run_idx in range(len(case.prompts))]
        for case_idx, case in enumerate(cases)
    ]
