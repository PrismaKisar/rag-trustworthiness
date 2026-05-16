"""Shared visual constants for all experiment notebooks."""

MODEL_LABELS = {
    "Qwen/Qwen2.5-1.5B-Instruct":           "Qwen2.5",
    "google/gemma-2-2b-it":                  "Gemma-2",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":  "SmolLM2",
    "microsoft/Phi-3.5-mini-instruct":       "Phi-3.5",
    "meta-llama/Llama-3.2-3B-Instruct":      "Llama-3.2",
}

MODEL_COLORS = {
    "Qwen/Qwen2.5-1.5B-Instruct":           "#5C6BC0",
    "google/gemma-2-2b-it":                  "#26A69A",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":  "#FF7043",
    "microsoft/Phi-3.5-mini-instruct":       "#AB47BC",
    "meta-llama/Llama-3.2-3B-Instruct":      "#FFA726",
}

MODEL_MARKERS = {
    "Qwen/Qwen2.5-1.5B-Instruct":           "o",
    "google/gemma-2-2b-it":                  "s",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":  "^",
    "microsoft/Phi-3.5-mini-instruct":       "D",
    "meta-llama/Llama-3.2-3B-Instruct":      "v",
}

MODEL_LS = {
    "Qwen/Qwen2.5-1.5B-Instruct":           "-",
    "google/gemma-2-2b-it":                  "--",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":  "-.",
    "microsoft/Phi-3.5-mini-instruct":       ":",
    "meta-llama/Llama-3.2-3B-Instruct":      (0, (3, 1, 1, 1)),
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
