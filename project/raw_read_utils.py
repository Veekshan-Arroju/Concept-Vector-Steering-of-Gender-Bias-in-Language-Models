"""
Helpers for raw_read_train.py (pandas only: no torch, testable on CPU).

The read-position ablation: train the SAME concept vector from activations read at the end of the RAW prompt text (the user message
exactly as dataset_builder wrote it: no chat template, no assistant header, no BOS), like the v3 run, instead of at the assistant
header that follows the chat-templated prompt (v5).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

# Anything that would mean a chat template leaked into a "raw" prompt.
CHAT_MARKERS = ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_start|>", "<|im_end|>", "[INST]", "[/INST]")


def raw_prompts(registry: pd.DataFrame, prompt_ids: Sequence[int], bos_text: Optional[str] = None) -> List[str]:
    """
    The user-message text of registry rows `prompt_ids`, in the given order (that order is the stored training order:
    do not re-order, split_train_val relies on it). `bos_text`, if given, is prepended verbatim (e.g. '<|begin_of_text|>').
    """
    ids = [int(i) for i in prompt_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("a training dataset lists the same prompt twice")
    table = registry.set_index("prompt_id")
    missing = [i for i in ids if i not in table.index]
    if missing:
        raise KeyError(f"prompt ids not in the registry: {missing[:5]}")
    texts = table.loc[ids, "user_prompt"].astype(str).tolist()
    for t in texts:
        bad = [m for m in CHAT_MARKERS if m in t]
        if bad:
            raise ValueError(f"a prompt already contains chat-template markers {bad}: {t[:80]!r}")
    return [(bos_text + t) if bos_text else t for t in texts]


def label_summary(labels: Sequence[float]) -> dict:
    """Sanity numbers for a training set: n, number of positives/negatives, and whether rows alternate pos/neg (interleaved)."""
    ys = [float(y) for y in labels]
    if any(y not in (0.0, 1.0) for y in ys):
        raise ValueError("labels must be 0 or 1")
    n1 = int(sum(ys))
    return {"n": len(ys), "n_pos": n1, "n_neg": len(ys) - n1,
            "interleaved": all(ys[i] != ys[i + 1] for i in range(0, len(ys) - 1, 2))}


def dry_run_indices(n: int, limit: int) -> List[int]:
    """`limit` evenly spaced row indices out of n, always an even count starting at an even index so pos/neg pairs stay together."""
    if limit <= 0 or n <= 0:
        raise ValueError("limit and n must be positive")
    limit = min(limit, n)
    pairs = max(1, limit // 2)
    n_pairs = n // 2
    step = max(1, n_pairs // pairs)
    idx = []
    for p in range(0, n_pairs, step):
        idx += [2 * p, 2 * p + 1]
        if len(idx) >= 2 * pairs:
            break
    return idx
