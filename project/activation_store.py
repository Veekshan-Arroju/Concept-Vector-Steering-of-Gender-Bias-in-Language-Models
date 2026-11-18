"""
On-disk store for extracted activations, one entry per REGISTRY PROMPT.

Layout (per model):
    activations/<model_slug>/<experiment>/meta.json        model id, registry hash, layers, dtype, ...
    activations/<model_slug>/<experiment>/layer_neg<k>.pt  Tensor (n_prompts, hidden_dim); row i == prompt_id i
                                                           (layer index -k, e.g. layer_neg1.pt is block -1)

Why this shape
--------------
* Datasets are just lists of prompt_ids, so the 200 neutral negatives shared by all
  subtype datasets are extracted once, not 22 times.
* One file per layer can be memory-mapped, so loading a 400-row dataset does not
  read the whole store.
* meta.json carries a hash of the registry. If prompts.csv changes after extraction
  (new seed, new frames, ...), every loader refuses to run on the stale activations
  instead of silently pairing rows with the wrong prompts.

This module needs torch but NOT transformers, so it can be used on a machine that
only has the saved activations.
"""

import json
from pathlib import Path

import torch

from dataset_io import (
    arm_names,
    get_arm_table,
    get_eval_set,
    get_train_set,
    load_manifest,
)
from experiment import PROJECT_ROOT, exp_dir

ACTIVATIONS_DIR = PROJECT_ROOT / "activations"


def store_dir(model_id: str, root: Path = None) -> Path:
    """activations/<model_slug>/<experiment>"""
    return exp_dir(root or ACTIVATIONS_DIR, model_id)


def layer_path(model_id: str, layer_idx: int, root: Path = None) -> Path:
    if layer_idx >= 0:
        raise ValueError(f"layer indices are negative (-1 = last block); got {layer_idx}")
    return store_dir(model_id, root) / f"layer_neg{-layer_idx}.pt"


# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
def save_prompt_activations(per_layer: dict, model_id: str, registry_sha1: str,
                            extra_meta: dict = None, root: Path = None) -> Path:
    """
    Persist {layer_idx -> Tensor(n_prompts, d)} (row i == prompt_id i).
    Layer files are written first and meta.json last, so a half-written store has no meta
    and is rejected by the loaders.
    """
    out = store_dir(model_id, root)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta_path.unlink()

    n_prompts, hidden_dim, dtype = None, None, None
    for layer_idx, tensor in per_layer.items():
        tensor = tensor.contiguous()
        if n_prompts is None:
            n_prompts, hidden_dim, dtype = tensor.shape[0], tensor.shape[1], str(tensor.dtype)
        assert tensor.shape == (n_prompts, hidden_dim), f"layer {layer_idx}: inconsistent shape {tuple(tensor.shape)}"
        torch.save(tensor, layer_path(model_id, layer_idx, root))

    meta = {
        "model_id": model_id,
        "registry_sha1": registry_sha1,
        "n_prompts": int(n_prompts),
        "hidden_dim": int(hidden_dim),
        "dtype": dtype,
        "layers": sorted(per_layer.keys()),
        **(extra_meta or {}),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[save_prompt_activations] {len(per_layer)} layers x {n_prompts} prompts x {hidden_dim} dims ({dtype}) -> {out}")
    return out


# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------
def load_meta(model_id: str, root: Path = None) -> dict:
    path = store_dir(model_id, root) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No activations for {model_id!r} at {path.parent}. Run activation_extraction.py with MODEL_ID={model_id}."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_fresh(model_id: str, root: Path = None) -> dict:
    meta = load_meta(model_id, root)
    current = load_manifest()["registry_sha1"]
    if meta["registry_sha1"] != current:
        raise RuntimeError(
            f"Stale activations for {model_id!r}: they were extracted from a different prompts.csv "
            f"(stored {meta['registry_sha1'][:10]}..., current {current[:10]}...). Re-run activation_extraction.py."
        )
    return meta


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, mmap=True, weights_only=True)   # torch >= 2.1: no full read
    except (TypeError, RuntimeError):
        return torch.load(path, weights_only=True)


def load_prompt_rows(model_id: str, prompt_ids: list, layers: list = None, root: Path = None) -> dict:
    """
    {layer_idx -> Tensor(len(prompt_ids), d)} for the requested prompts, in the given order.
    Tensors keep the dtype they were saved in (callers cast with .float()).
    """
    meta = check_fresh(model_id, root)
    layers = list(layers) if layers is not None else list(meta["layers"])
    missing = [l for l in layers if l not in meta["layers"]]
    if missing:
        raise KeyError(f"Layers {missing} were not extracted for {model_id!r} (have {meta['layers'][:3]}...)")
    idx = torch.as_tensor(prompt_ids, dtype=torch.long)
    return {l: _load_tensor(layer_path(model_id, l, root))[idx] for l in layers}


def load_train_activations(model_id: str, dataset_name: str, layers: list = None, root: Path = None) -> tuple:
    """(activations, labels) for one training dataset. Same shapes the old load_activations(split='train') returned."""
    ids, labels = get_train_set(dataset_name)
    return load_prompt_rows(model_id, ids, layers, root), labels


def load_eval_activations(model_id: str, set_name: str = "main", concept: str = None,
                          layers: list = None, root: Path = None) -> tuple:
    """
    (activations, labels) for an eval set: neutral=0 vs misogyny=1 pairs.
        set_name "main"    seen-phrase paired test      set_name "heldout"  unseen-phrase test
    """
    ids, labels = get_eval_set(set_name, concept)
    return load_prompt_rows(model_id, ids, layers, root), labels


def load_arm_activations(model_id: str, set_name: str = "main", concept: str = None,
                         layers: list = None, root: Path = None) -> tuple:
    """
    ({arm -> {layer -> Tensor(n_rows, d)}}, arm_table). Row j of every arm belongs to the SAME fresh statement
    (neutral / benign / hateful rows repeat across the held-out phrases of one statement).
    """
    table = get_arm_table(set_name, concept)
    acts = {arm: load_prompt_rows(model_id, table[arm].astype(int).tolist(), layers, root) for arm in arm_names()}
    return acts, table
