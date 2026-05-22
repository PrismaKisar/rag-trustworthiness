"""Shared visual constants for all experiment notebooks."""

MODEL_LABELS = {
    "Qwen/Qwen3.5-2B":      "Qwen3.5",
    "google/gemma-4-E2B-it": "Gemma-4",
}

MODEL_COLORS = {
    "Qwen/Qwen3.5-2B":      "#5C6BC0",
    "google/gemma-4-E2B-it": "#26A69A",
}

MODEL_MARKERS = {
    "Qwen/Qwen3.5-2B":      "o",
    "google/gemma-4-E2B-it": "s",
}

MODEL_LS = {
    "Qwen/Qwen3.5-2B":      "-",
    "google/gemma-4-E2B-it": "--",
}

PROMPT_COLORS = {
    "standard":         "#5C6BC0",
    "chain_of_thought": "#26A69A",
    "vigilant":         "#FF7043",
}

PROMPT_COLORS_QA = {
    "standard_qa": "#5C6BC0",
    "cot_qa":      "#26A69A",
    "vigilant_qa": "#FF7043",
}
