"""
Concept-vector training: learn one steering direction v_l per transformer layer.

Primary method  : RFM — kernel ridge regression with iterative AGOP metric updates, then extract the top eigenvector and orient its sign by label
                  correlation. Implements RFMToolkit._compute_directions() from the paper's control_toolkits.py, adapted to use pre-extracted activations
                  from disk (written by activation_extraction.py) instead of re-running the frozen model.

Fallback method : mean_difference — (mean of class-1 activations) minus (mean of class-0 activations), normalized. O(nd), no GPU memory spike,
                  useful for smoke tests or when VRAM is limited. Implements MeanDifferenceToolkit._compute_directions() from the paper.

Both methods apply the same sign-orientation step (pearson_corr) so the output format is identical regardless of method.
"""

import os
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm

from dataset_builder import SEED
from activation_store import load_meta, load_train_activations
from dataset_io import split_train_val, subtype_vector_names   # concept lists come from the manifest
from experiment import PROJECT_ROOT, exp_dir
from gpu_utils import describe_device, warn_if_multi_gpu
from raw_data import (
    AVERAGED_ALL_NAME,
    AVERAGED_NAME,
    POOLED_NAME,
    SINGLE_NAME,
)
import rfm

# Reproducibility 
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Where the learned concept vectors should be written ----
CONCEPT_VECTORS_DIR = PROJECT_ROOT / "concept_vectors"
CONCEPT_VECTORS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Utility functions carried over from direction_utils.py in the paper.
# -----------------------------------------------------------------------
def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation between x and y.

    Directly adapted from direction_utils.pearson_corr() in the paper.
    Used during sign orientation: if a direction anti-correlates with the label, we flip it so positive projections always mean "concept present".

    Args:
        x, y : 1-D or shape-matched tensors

    Returns:
        Scalar Pearson r in [-1.0, +1.0]
    """
    assert x.shape == y.shape, "x and y must have the same shape for Pearson r"

    # Cast to float32 to avoid overflow when inputs come from fp16 activations
    x = x.float()
    y = y.float()

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator   = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return numerator / denominator


def project_onto_direction(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    # device: str = "cuda",
    device: str = None, # None means "use whatever DEVICE is"
) -> torch.Tensor:
    """
    Dot each row of hidden_states with the direction vector.

    Directly adapted from direction_utils.project_onto_direction() in the paper.
    Used to compute per-sample projection scores during sign orientation 
    — the sign of the Pearson r between these scores and the labels tells us whether to flip the direction.

    Args:
        hidden_states : (N, d) tensor — one row per sample
        direction     : (d,)   tensor — the candidate concept vector
        device        : Device for the matmul (default: "cuda")

    Returns:
        (N,) tensor of scalar projection scores
    """
    # Fall back to the globally detected device (CPU or CUDA) if caller didn't specify
    if device is None:
        device = DEVICE
        
    assert len(hidden_states.shape) == 2, "hidden_states must be 2-D: (N, d)"
    assert hidden_states.shape[1] == direction.shape[0], (
        f"Dimension mismatch: hidden_states has d={hidden_states.shape[1]} "
        f"but direction has shape {direction.shape}"
    )

    return (
        hidden_states.to(device=device)
        @ direction.to(device=device, dtype=hidden_states.dtype)
    )



# -----------------------------------------------------------------------
# 2. RFM direction training (primary method)
#    Adapts direction_utils.train_rfm_probe_on_concept() from the paper.
#    Sweeps over bandwidths ∈ {1, 10, 100} and norm ∈ {True, False},
#    keeps the direction that maximizes Pearson |r| on the inner-val split.
# -----------------------------------------------------------------------
def train_rfm_direction(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X:   torch.Tensor,
    val_y:   torch.Tensor,
    bandwidths: list  = [1, 10, 100],
    reg:        float = 1e-3,
    num_iters:  int   = 10,
) -> torch.Tensor:
    """
    Train one concept direction per layer using kernel ridge regression + AGOP updates.

    Adapts direction_utils.train_rfm_probe_on_concept() from the paper.
    For each (bandwidth, norm) combination:
        - Run rfm.rfm() for num_iters AGOP update steps.
        - Record the best direction + its validation Pearson |r|.
    Return the direction that achieved the highest validation correlation across all hyperparameter configurations.

    All data passed in must already be on DEVICE (GPU) and cast to float32.
    The inner-val split is only used to SELECT the best direction — it never influences the AGOP gradient computation (which uses only train_X/train_y).

    Args:
        train_X    : (n_inner_train, d) float32 activations — inner training split
        train_y    : (n_inner_train, 1) float32 labels {0.0, 1.0}
        val_X      : (n_inner_val,   d) float32 activations — inner validation split
        val_y      : (n_inner_val,   1) float32 labels
        bandwidths : Laplacian kernel bandwidth values to sweep (paper default: [1, 10, 100])
        reg        : KRR regularization strength (paper default: 1e-3)
        num_iters  : AGOP update steps per configuration (paper default: 10)

    Returns:
        best_u : (d, 1) float32 tensor on CPU — the raw (unsigned) concept direction.
                 Sign orientation is applied separately in compute_all_layer_signs().
    """
    best_r  = -float("inf")
    best_u  = None
    best_bw   = None
    best_norm = None

    for bw in bandwidths:
        for norm in [True, False]:    # paper sweeps both settings and keeps the winner
            u, val_r = rfm.rfm(
                traindata=(train_X, train_y),
                testdata=(val_X,   val_y),
                L=bw,
                reg=reg,
                num_iters=num_iters,
                norm=norm,
            )

            # val_r is the Pearson |r| on the inner-val split for this config
            if val_r >= best_r:
                best_u    = u       # (d, 1) on DEVICE from lobpcg
                best_r    = val_r
                best_bw   = bw
                best_norm = norm

            torch.cuda.empty_cache()    # free GPU memory between rfm() calls

    print(f"    Best RFM | val-r: {best_r:.4f} | bandwidth: {best_bw} | norm: {best_norm}")

    # If the solver diverged for every config (very rare), best_u stays None
    if best_u is None:
        raise RuntimeError(
            "RFM solver diverged for all (bandwidth, norm) combinations. "
            "Try increasing reg or switching to method='mean_difference'."
        )

    return best_u.cpu()     # move to CPU; caller pushes to GPU when needed



# -----------------------------------------------------------------------
# 3. Mean-difference fallback (fast alternative method)
#    Adapts MeanDifferenceToolkit._compute_directions() from control_toolkits.py in paper.
#    No kernel training — just subtract class centroids and normalize.
# -----------------------------------------------------------------------
def mean_difference_direction(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the concept direction as (class-1 centroid) - (class-0 centroid), normalized.

    Adapts MeanDifferenceToolkit._compute_directions() from the paper.
    Cost is O(n * d) — just two mean computations and a subtraction.
    Good for:
        - Smoke-testing the pipeline quickly
        - Low-VRAM environments where the kernel matrix doesn't fit on GPU
        - A simple baseline to compare RFM performance against

    Args:
        train_X : (n_inner_train, d) float32 activations (inner-training split only)
        train_y : (n_inner_train, 1) float32 labels {0.0, 1.0}

    Returns:
        direction : (1, d) float32 unit-norm vector — the mean-difference concept direction.
                    Stored as (1, d) to match the paper's direction dict convention.
    """
    # Boolean index masks for each class — paper's pos_indices / neg_indices
    pos_mask = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)    # label == 1.0
    neg_mask = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)   # label == 0.0

    pos_mean = train_X[pos_mask].mean(dim=0)    # centroid of class-1 samples
    neg_mean = train_X[neg_mask].mean(dim=0)    # centroid of class-0 samples

    # Direction from class-0 to class-1, then L2-normalize to unit length
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()

    # Return as (1, d) to match the (1, d) convention used throughout directions dict
    return direction.reshape(1, -1)


# -----------------------------------------------------------------------
# 4. Sign orientation across all layers
#    Adapts RFMToolkit._compute_signs() from control_toolkits.py.
#    Ensures v_l is oriented so that positive projections = "concept present".
# -----------------------------------------------------------------------
def compute_all_layer_signs(
    all_train_activations: dict,
    all_train_y:           torch.Tensor,
    directions:            dict,
) -> dict:
    """
    Orient each per-layer direction so class-1 samples project to positive scores.

    Adapts RFMToolkit._compute_signs() from control_toolkits.py in the paper.

    For each layer:
      1. Project ALL training samples onto the (unsigned) candidate direction.
      2. Compute Pearson r between those projections and the training labels.
      3. If r > 0 -> direction already points toward class-1 -> sign = +1.
         If r < 0 -> direction points away from class-1 -> sign = -1 (flip it).

    Information-leakage note:
        Only TRAINING activations are used here — test data is never touched.
        The sign is a post-hoc geometric orientation of a direction that was already determined purely from training data.

    Args:
        all_train_activations : Dict[layer_idx -> Tensor(N_train, d)]   -> The FULL training set (not inner-train only).
        all_train_y           : (N_train, 1) float32 labels for the full training set.
        directions            : Dict[layer_idx -> Tensor(1, d)] — unsigned directions.

    Returns:
        signs : Dict[layer_idx -> Dict[component_idx -> float (+1.0 or -1.0)]]
                Keyed by layer index; component_idx is 0 (one direction per layer).
    """
    signs = {}

    for layer_idx in all_train_activations.keys():
        # Pull the full-training activations for this layer — fp32 for safe arithmetic
        xs = all_train_activations[layer_idx].float()   # (N_train, d)

        # The direction at this layer is stored as (1, d); take row 0 to get (d,)
        direction = directions[layer_idx][0]            # (d,) on CPU

        # Project every training sample onto this candidate direction
        projections = project_onto_direction(xs, direction).to(all_train_y.device)

        # Pearson correlation: positive means direction aligns with label, negative means flip
        corr = pearson_corr(all_train_y.squeeze(1), projections)

        # +1.0 if correlation is positive, -1.0 if negative
        sign_val = (2.0 * (corr > 0).float() - 1.0).item()

        # Store per component index (c_idx=0 since RFM yields one direction per layer)
        signs[layer_idx] = {0: sign_val}

    return signs



# -----------------------------------------------------------------------
# 5. Per-layer concept-vector training loop
#    Adapts RFMToolkit._compute_directions() from control_toolkits.py in paper.
# -----------------------------------------------------------------------
def learn_concept_vectors(
    train_activations: dict,
    train_labels:      list,
    method:            str = "rfm",
) -> tuple:
    """
    Learn one signed concept direction v_l for every transformer layer.

    Adapts RFMToolkit._compute_directions() from control_toolkits.py, using pre-extracted activations instead of re-running the model.

    Per-layer pipeline:
        For each layer l:
            1. Cast fp16 activations to fp32.
            2. Carve inner-train / inner-val from the TRAINING activations.
            3. Run RFM (or mean_difference) to get a raw direction u ∈ R^d.
            4. Store u as (1, d) in directions dict.
        After all layers:
            5. Compute orientation signs using the FULL training set.
            6. Apply signs: directions[l][0] *= sign[l][0].

    Information-leakage controls:
        - inner-train / inner-val split is from TRAINING data only, using the same split_train_val() function as linear_probe.py — consistent splits.
        - Test activations are NEVER passed into this function.
        - Sign computation uses ALL training samples but still only training data.
        - torch.no_grad() is implicit (no model forward passes here at all).

    Args:
        train_activations : Dict[layer_idx -> Tensor(N_train, d)]
                            Loaded from disk by load_train_activations() — training rows only.
        train_labels      : List[int] of length N_train (0 or 1)
        method            : 'rfm'             — full RFM with AGOP metric learning (default)
                            'mean_difference' — fast O(nd) baseline, no kernel training

    Returns:
        directions : Dict[layer_idx -> Tensor(1, d)] — one signed concept vector per layer
        signs      : Dict[layer_idx -> Dict[0 -> float]] — +1/-1 orientation per layer
    """
    # Build (N_train, 1) float32 label tensor — paper's standard probe format
    all_train_y = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)

    # Carve inner-train / inner-val from training data — NO test data here.
    # Reuses split_train_val() from linear_probe.py so both steps use the same split.
    N_train = len(train_labels)
    inner_train_idx, inner_val_idx = split_train_val(N_train)

    # Corresponding label slices for the inner splits
    inner_train_y = all_train_y[list(inner_train_idx)].to(DEVICE)   # (n_inner_train, 1)
    inner_val_y   = all_train_y[list(inner_val_idx)].to(DEVICE)     # (n_inner_val,   1)

    print(
        f"  Inner-train: {len(inner_train_idx)} samples | "
        f"Inner-val: {len(inner_val_idx)} samples | "
        f"Layers: {len(train_activations)}"
    )

    directions = {}
    layers = list(train_activations.keys())

    for layer_idx in tqdm(layers, desc=f"  Learning concept vectors [{method}]"):
        # Cast fp16 -> fp32 before any linear algebra — the paper always does this
        train_acts = train_activations[layer_idx].float()   # (N_train, d)

        # Slice to the inner splits — test activations never touched inside this loop
        inner_train_X = train_acts[list(inner_train_idx)].to(DEVICE)   # (n_inner_train, d)
        inner_val_X   = train_acts[list(inner_val_idx)].to(DEVICE)     # (n_inner_val,   d)

        if method == "rfm":
            # Full RFM: iterative kernel ridge regression + AGOP metric updates.
            # Returns (d, 1) on CPU — we reshape to (1, d) for the paper's convention.
            u = train_rfm_direction(
                train_X=inner_train_X,
                train_y=inner_train_y,
                val_X=inner_val_X,
                val_y=inner_val_y,
            )
            directions[layer_idx] = u.reshape(1, -1)    # (1, d) — paper format

        elif method == "mean_difference":
            # Fast fallback: class centroid difference, normalized.
            # Returns (1, d) on the same device as inner_train_X — move to CPU.
            direction = mean_difference_direction(inner_train_X, inner_train_y)
            directions[layer_idx] = direction.cpu()     # (1, d) on CPU

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'rfm' or 'mean_difference'."
            )

        torch.cuda.empty_cache()    # release GPU memory before the next layer

    # ---- Sign orientation — binary classification only ----
    # The paper's RFMToolkit only orients signs for num_classes == 1 (binary).
    # Our dataset is binary (misogyny vs. neutral), so this always runs.
    # Signs are computed from ALL training data (not just inner-train).
    print("\n  Computing orientation signs (using full training set) ...")
    signs = compute_all_layer_signs(
        all_train_activations=train_activations,
        all_train_y=all_train_y,
        directions=directions,
    )

    # Apply the sign flip where needed: directions[l][0] *= ±1
    # After this, positive projection => "concept present" for every layer.
    for layer_idx in layers:
        directions[layer_idx][0] *= signs[layer_idx][0]

    return directions, signs



# -----------------------------------------------------------------------
# 6. Persistence - one file per named vector
#    concept_vectors/<model_slug>/<experiment>/<vector_name>.pkl
#    vector names: subtype__<slug> (x20), single__misogyny, pooled__misogyny (non-held-out subtypes),
#                  averaged__misogyny (mean of the non-held-out subtype vectors), averaged_all__misogyny (mean of all)
# -----------------------------------------------------------------------
DEFAULT_VECTOR_NAME = os.getenv("VECTOR_NAME", AVERAGED_NAME)


def vector_path(model_id: str, vector_name: str, save_dir: Path = CONCEPT_VECTORS_DIR) -> Path:
    """concept_vectors/<model_slug>/<experiment>/<vector_name>.pkl"""
    return exp_dir(save_dir, model_id) / f"{vector_name}.pkl"


def save_concept_vectors(
    directions:  dict,
    signs:       dict,
    model_id:    str,
    method:      str,
    vector_name: str = DEFAULT_VECTOR_NAME,
    extra:       dict = None,
    save_dir:    Path = CONCEPT_VECTORS_DIR,
) -> Path:
    """
    Save one named set of per-layer concept vectors.

    The payload bundles directions, signs, method, model_id and vector_name so each file is
    self-describing for the probes and the steering scripts.
    """
    out_path = vector_path(model_id, vector_name, save_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "directions":  directions,    # Dict[layer_idx -> Tensor(1, d)]
        "signs":       signs,         # Dict[layer_idx -> Dict[0 -> float]]
        "method":      method,        # which algorithm produced these vectors
        "model_id":    model_id,      # for auditing / documentation
        "vector_name": vector_name,
        **(extra or {}),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[save_concept_vectors] {vector_name} -> {out_path}")
    return out_path


def load_concept_vectors(
    model_id:    str,
    vector_name: str = DEFAULT_VECTOR_NAME,
    save_dir:    Path = CONCEPT_VECTORS_DIR,
) -> tuple:
    """
    Load a named vector set saved by save_concept_vectors().

    Returns:
        (directions, signs, method)
            directions : Dict[layer_idx -> Tensor(1, d)]
            signs      : Dict[layer_idx -> Dict[0 -> float]]
            method     : str - 'rfm', 'mean_difference' or 'average(...)'
    """
    path = vector_path(model_id, vector_name, save_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No vector {vector_name!r} for {model_id!r} at {path}. "
            f"Run concept_vector_training.py (it also builds {AVERAGED_NAME!r})."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)

    print(
        f"[load_concept_vectors] {vector_name}: {len(payload['directions'])} layer vectors "
        f"(method: {payload['method']})"
    )
    return payload["directions"], payload["signs"], payload["method"]


def vector_exists(model_id: str, vector_name: str, save_dir: Path = CONCEPT_VECTORS_DIR) -> bool:
    return vector_path(model_id, vector_name, save_dir).exists()


def vector_covers(model_id: str, vector_name: str, wanted_layers: list, save_dir: Path = CONCEPT_VECTORS_DIR,
                  method: str = None) -> bool:
    """
    True if the saved vector exists, contains every layer in `wanted_layers`, and (if `method` is given) was
    trained with that method. Used by the resume logic so that
      * a quick run on a few layers (--layers ...) is never mistaken for a finished full run, and
      * a quick mean-difference sanity vector is never mistaken for an RFM vector.
    """
    path = vector_path(model_id, vector_name, save_dir)
    if not path.exists():
        return False
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if method is not None and payload.get("method") != method:
        return False
    return set(wanted_layers) <= set(payload["directions"].keys())


# -----------------------------------------------------------------------
# 7. Averaging and similarity
# -----------------------------------------------------------------------
def average_concept_vectors(vector_sets: list) -> tuple:
    """
    Plain mean of several per-layer direction dicts, renormalised to unit length.

    Every input direction is already sign-oriented (positive projection = misogyny present)
    and unit-norm, so the mean is meaningful: if the subtype directions agree, the average
    keeps most of each; if they were unrelated it would be a weak match for all of them
    (check the cosine report). Layers missing from any input are skipped.

    Args:
        vector_sets : list of Dict[layer_idx -> Tensor(1, d)]
    Returns:
        (directions, signs) - same structure as learn_concept_vectors()
    """
    layers = sorted(set.intersection(*[set(v.keys()) for v in vector_sets]))
    directions, signs = {}, {}
    for layer_idx in layers:
        stacked = torch.stack([v[layer_idx][0].float() for v in vector_sets])   # (K, d)
        mean = stacked.mean(dim=0)
        directions[layer_idx] = (mean / mean.norm()).reshape(1, -1)
        signs[layer_idx] = {0: 1.0}
    return directions, signs


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def similarity_report(named: dict, train_names: list, heldout_names: list) -> dict:
    """
    Per layer, cosines that show how coherent "misogyny" is across subtypes and how close the combined vectors
    (S, P, A) are to each other and to the vectors of the held-out phrases.

    Args:
        named         : {vector_name -> Dict[layer -> Tensor(1, d)]}
        train_names   : per-subtype vector names that P and A are built from
        heldout_names : per-subtype vector names of the held-out phrases (never used by P or A)
    Returns:
        {layer -> {subtype_mean_cos, subtype_min_cos, A_member_mean_cos, A_member_min_cos, S_P, S_A, P_A,
                   S_held_mean, P_held_mean, A_held_mean}}
        *_held_mean = mean cosine between that vector and the vectors trained directly on the held-out phrases
        (how close it is to what you would get by training on the unseen wording).
    """
    layers = sorted(set.intersection(*[set(v.keys()) for v in named.values()]))
    report = {}
    for l in layers:
        sub = [named[n][l][0].float() for n in train_names if n in named]
        held = [named[n][l][0].float() for n in heldout_names if n in named]
        pair_cos = [_cos(sub[i], sub[j]) for i in range(len(sub)) for j in range(i + 1, len(sub))]
        nan = float("nan")
        row = {
            "subtype_mean_cos": sum(pair_cos) / len(pair_cos) if pair_cos else nan,
            "subtype_min_cos":  min(pair_cos) if pair_cos else nan,
        }
        # how well the averaged vector matches each of its members (1 = identical direction)
        if AVERAGED_NAME in named and sub:
            to_avg = [_cos(named[AVERAGED_NAME][l][0], v) for v in sub]
            row["A_member_mean_cos"] = sum(to_avg) / len(to_avg)
            row["A_member_min_cos"] = min(to_avg)
        else:
            row["A_member_mean_cos"] = row["A_member_min_cos"] = nan
        for tag, (a, b) in {"S_P": (SINGLE_NAME, POOLED_NAME), "S_A": (SINGLE_NAME, AVERAGED_NAME),
                            "P_A": (POOLED_NAME, AVERAGED_NAME)}.items():
            row[tag] = _cos(named[a][l][0], named[b][l][0]) if a in named and b in named else nan
        for tag, name in {"S_held_mean": SINGLE_NAME, "P_held_mean": POOLED_NAME, "A_held_mean": AVERAGED_NAME}.items():
            row[tag] = (sum(_cos(named[name][l][0], h) for h in held) / len(held)) if (name in named and held) else nan
        report[l] = row
    return report


def print_similarity_report(report: dict) -> None:
    print("\n" + "=" * 100)
    print(f"  {'Layer':>6}  {'subtype pairs':>22}  {'average vs its members':>26}  {'S-P':>6}  {'S-A':>6}  {'P-A':>6}")
    print(f"  {'':>6}  {'mean cos':>10}  {'min cos':>10}  {'mean cos':>12}  {'min cos':>12}")
    print("=" * 100)
    for l in sorted(report):
        r = report[l]
        print(f"  {l:+6d}  {r['subtype_mean_cos']:10.3f}  {r['subtype_min_cos']:10.3f}  "
              f"{r['A_member_mean_cos']:12.3f}  {r['A_member_min_cos']:12.3f}  "
              f"{r['S_P']:6.3f}  {r['S_A']:6.3f}  {r['P_A']:6.3f}")
    print("=" * 100)
    print("  Closeness to the vectors trained directly on the HELD-OUT phrases (mean cosine; higher = closer to what training on the unseen wording gives)")
    print(f"  {'Layer':>6}  {'S':>8}  {'P':>8}  {'A':>8}")
    for l in sorted(report):
        r = report[l]
        print(f"  {l:+6d}  {r['S_held_mean']:8.3f}  {r['P_held_mean']:8.3f}  {r['A_held_mean']:8.3f}")
    print("=" * 100 + "\n")


# -----------------------------------------------------------------------
# 8. Summary display
# -----------------------------------------------------------------------
def print_concept_vector_summary(directions: dict, signs: dict) -> None:
    """
    Print a per-layer summary: direction shape, L2 norm, and orientation sign.

    Useful for a quick sanity check after training - all norms should be close to 1.0 (unit-norm directions), and signs should be +/-1.
    """
    print("\n" + "=" * 62)
    print(f"  {'Layer':>6}    {'Shape':>10}    {'L2 norm':>10}    {'Sign':>6}")
    print("=" * 62)

    for layer_idx in sorted(directions.keys()):
        vec  = directions[layer_idx]          # (1, d)
        norm = vec.norm().item()
        sign = signs[layer_idx][0] if layer_idx in signs else "n/a"
        print(
            f"  {layer_idx:+6d}    {str(tuple(vec.shape)):>10}    "
            f"{norm:10.4f}    {sign:>+6.1f}"
        )

    print("=" * 62 + "\n")


# -----------------------------------------------------------------------
# 9. End-to-end pipeline
# -----------------------------------------------------------------------
def train_one_vector(
    model_id:     str,
    dataset_name: str,
    method:       str,
    layers:       list = None,
    save_dir:     Path = CONCEPT_VECTORS_DIR,
) -> tuple:
    """Train + sign-orient + save the vector for ONE training dataset (name == vector name)."""
    train_activations, train_labels = load_train_activations(model_id, dataset_name, layers=layers)
    print(f"\n[train] {dataset_name}: {len(train_labels)} rows | layers: {len(train_activations)} | method: {method}")
    t0 = time.perf_counter()
    directions, signs = learn_concept_vectors(train_activations, train_labels, method=method)
    elapsed = time.perf_counter() - t0
    per_layer = elapsed / max(1, len(directions))
    print(f"[train] {dataset_name}: {elapsed:.0f} s for {len(directions)} layer(s) = {per_layer:.1f} s per layer "
          f"(the first layer also pays GPU warm-up, so this is a slight over-estimate)")
    all_layers = load_meta(model_id)["layers"]
    if len(directions) < len(all_layers):
        n_vectors = len(subtype_vector_names("all")) + 2   # every subtype + single + pooled
        print(f"[train] Projection for the FULL run ({len(all_layers)} layers, {n_vectors} vectors): "
              f"{per_layer * len(all_layers) / 60:.1f} min per vector, {per_layer * len(all_layers) * n_vectors / 3600:.1f} h in total")
    save_concept_vectors(directions, signs, model_id, method, vector_name=dataset_name,
                         extra={"dataset": dataset_name, "layers": sorted(directions.keys())}, save_dir=save_dir)
    return directions, signs


def build_averaged_vector(model_id: str, method: str, vector_names: list, out_name: str,
                          save_dir: Path = CONCEPT_VECTORS_DIR) -> tuple:
    """
    Average the saved per-subtype vectors in `vector_names` into `out_name` and save it (no training involved).

    The label records the method the COMPONENTS were really trained with (read from the saved files), for example
    "average(16 subtypes, mean_difference)", or "average(16 subtypes, mixed: mean_difference+rfm)" if they differ.
    `method` (the method of the current run) is NOT used for the label: a run that trains only the single-phrase vector
    with RFM must not relabel averaged vectors that are made of mean-difference vectors.
    """
    loaded = [load_concept_vectors(model_id, n, save_dir) for n in vector_names]
    sets = [l[0] for l in loaded]
    used = sorted({l[2] for l in loaded})
    label = used[0] if len(used) == 1 else "mixed: " + "+".join(used)
    directions, signs = average_concept_vectors(sets)
    save_concept_vectors(directions, signs, model_id, f"average({len(sets)} subtypes, {label})",
                         vector_name=out_name, extra={"components": list(vector_names),
                                                      "layers": sorted(directions.keys())}, save_dir=save_dir)
    return directions, signs


def main(
    model_id: str  = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    method:   str  = os.getenv("CONCEPT_VECTOR_METHOD", "rfm"),
    vectors:  list = None,
    layers:   list = None,
    skip_existing: bool = True,
) -> dict:
    """
    Train every concept vector, then build the two averaged vectors and the similarity report.

        1. For each dataset in [single, pooled, every subtype] (or `vectors`):
               load its 400 rows of activations, learn one direction per layer, orient signs, save.
               (pooled was built from the NON-held-out subtypes only; every subtype incl. the held-out ones is trained
               on its own phrase: the held-out vectors are references and are never used by P or A.)
        2. averaged__misogyny     = normalised mean of the NON-held-out subtype vectors   (compare with S and P)
           averaged_all__misogyny = normalised mean of ALL subtype vectors                (for final use)
        3. Print / save the cosine report.

    Args:
        vectors       : subset of dataset names to train (default: single, pooled and every subtype)
        layers        : subset of negative layer indices (default: every extracted layer).
                        RFM cost scales with this: ~6 RFM fits per layer per vector.
        skip_existing : reuse vectors already on disk (lets a long run resume). A vector is only reused if it
                        contains every requested layer AND was trained with the same method, so a timing run on a few
                        layers, or a quick mean-difference check, is redone by a full RFM run.

    Information-leakage controls (unchanged from v3):
        - Only TRAINING rows are loaded here; the test sets are never touched.
        - inner-val (last 20% of rows = last 40 statements of each pool) only selects RFM hyperparameters.
    """
    print(f"\n[main] ---- Concept-Vector Training ----")
    print(f"[main] Model  : {model_id}")
    print(f"[main] Method : {method}")
    warn_if_multi_gpu("RFM training (it uses cuda:0)")
    print(f"[main] Device : {describe_device()}")

    all_sub = subtype_vector_names("all")
    train_sub = subtype_vector_names("train")
    held_sub = subtype_vector_names("heldout")
    names = vectors if vectors is not None else [SINGLE_NAME, POOLED_NAME] + all_sub
    wanted_layers = list(layers) if layers is not None else load_meta(model_id)["layers"]
    trained = []
    for name in names:
        if skip_existing and vector_covers(model_id, name, wanted_layers, method=method):
            print(f"[main] {name}: already on disk with the requested layers and method ({method}), skipping")
            continue
        train_one_vector(model_id, name, method, layers=layers)
        trained.append(name)

    result = {}
    sim_path = exp_dir(CONCEPT_VECTORS_DIR, model_id) / "_similarity.pkl"
    if all(vector_exists(model_id, n) for n in all_sub):
        derived_exist = (vector_exists(model_id, AVERAGED_NAME) and vector_exists(model_id, AVERAGED_ALL_NAME)
                         and sim_path.exists())
        if any(n in all_sub for n in trained) or not derived_exist:
            # the averaged vectors and the similarity report are derived from the subtype vectors: rebuild them only
            # when a subtype vector changed (or they are missing), never as a side effect of training another vector
            build_averaged_vector(model_id, method, train_sub, AVERAGED_NAME)
            build_averaged_vector(model_id, method, all_sub, AVERAGED_ALL_NAME)
            named = {n: load_concept_vectors(model_id, n)[0]
                     for n in [SINGLE_NAME, POOLED_NAME, AVERAGED_NAME, AVERAGED_ALL_NAME] + all_sub if vector_exists(model_id, n)}
            report = similarity_report(named, train_sub, held_sub)
            print_similarity_report(report)
            with open(sim_path, "wb") as f:
                pickle.dump(report, f)
            print(f"[main] similarity report -> {sim_path}")
            result["similarity"] = report
        else:
            print("[main] No subtype vector changed: keeping the existing averaged vectors and similarity report.")
    else:
        print("[main] Not all subtype vectors exist yet - skipping the averaged vectors and the similarity report.")

    print("[main] Done.")
    return result


if __name__ == "__main__":
    import argparse
    import experiment
    parser = argparse.ArgumentParser(description="Train concept vectors for every dataset.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--method", default=os.getenv("CONCEPT_VECTOR_METHOD", "rfm"), choices=["rfm", "mean_difference"])
    parser.add_argument("--vectors", nargs="*", default=None, help="dataset names to train (default: single, pooled and every subtype)")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="negative layer indices, e.g. -8 -12 -16 (default: all)")
    parser.add_argument("--retrain", action="store_true", help="retrain vectors that already exist")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(model_id=args.model_id, method=args.method, vectors=args.vectors, layers=args.layers,
         skip_existing=not args.retrain)
