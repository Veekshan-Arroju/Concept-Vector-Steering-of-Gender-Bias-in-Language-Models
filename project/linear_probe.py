"""
Per-layer linear probe on raw hidden-state activations.

Trains one ridge-regression probe per transformer layer using the full
activation vector a_l ∈ R^k (the raw hidden state) and reports AUROC +
accuracy on the held-out test set for every layer.

This implements the 'linear' baseline described in SI Appendix H of the
paper (Beaglehole et al., arXiv:2502.03708) and the per-block probing
strategy from Appendix L used for monitoring.  Results provide a layer-wise
signal for where concept information lives in the network, and serve as a
comparison point for the RFM-based direction-learning step that follows.

Design notes:
- Activations are loaded from disk (saved by activation_extraction.py).
  The expensive model forward passes are NOT repeated here.
- The inner train/val split for λ-tuning is drawn from TRAINING data only —
  test data is touched exactly once, for the final AUROC and accuracy.
- The regularization sweep and least-squares solver mirror the paper's
  direction_utils.train_linear_probe_on_concept() precisely.
- AUROC is the primary monitoring metric in the paper (Fig. 6 / Table S8);
  accuracy is reported as a secondary metric using the paper's 0.5 threshold.
"""

import os
import pickle
from copy import deepcopy
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset_builder import SEED                              # global seed
from experiment import PROJECT_ROOT, exp_dir                  # project root, experiment-namespaced output dirs
from activation_store import load_eval_activations, load_train_activations   # I/O helpers
from raw_data import POOLED_NAME

# Reproducibility 
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Output directory ----
PROBE_RESULTS_DIR = PROJECT_ROOT / "probe_results"
PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Inner train/val split
#    Adapted from paper's utils.split_indices() (random_split=False default)
# -----------------------------------------------------------------------
# split_train_val is a property of the DATA LAYOUT (rows are strictly interleaved pos/neg, never
# shuffled), so it now lives in dataset_io.py, where it can be tested without torch.
# It is re-exported here so `from linear_probe import split_train_val` keeps working everywhere.
from dataset_io import split_train_val   # noqa: E402,F401



# -----------------------------------------------------------------------
# 2. Low-level helpers
#    Adapted from paper's direction_utils.py
# -----------------------------------------------------------------------
def batch_transpose_multiply(
    A: torch.Tensor,
    B: torch.Tensor,
    mb_size: int = 5000,
) -> torch.Tensor:
    """
    Compute A^T @ B in VRAM-friendly mini-batches.

    Adapted from direction_utils.batch_transpose_multiply() in the paper.
    Used in the primal-form path (n > d) so that XtX = X^T X is built
    chunk-by-chunk, avoiding out-of-memory errors on large datasets.

    Args:
        A, B    : Tensors sharing the same first (sample-count) dimension
        mb_size : Rows per chunk — reduce if you hit an OOM error

    Returns:
        A^T @ B, computed on DEVICE; shape (A.shape[1], B.shape[1])
    """
    n = len(A)
    assert len(A) == len(B), "Row count mismatch between A and B"

    batches = torch.split(torch.arange(n), mb_size)
    result  = 0.0    # float sentinel — first real addition sets the dtype
    for b in batches:
        Ab = A[b].to(DEVICE)
        Bb = B[b].to(DEVICE)
        result = result + Ab.T @ Bb
        del Ab, Bb   # free GPU memory after each chunk

    return result    # shape: (A.shape[1], B.shape[1]) on DEVICE


def accuracy_fn(preds: torch.Tensor, truth: torch.Tensor) -> float:
    """
    Binary classification accuracy (%).

    Directly adapted from direction_utils.accuracy_fn() in the paper.
    Both preds and truth are thresholded at 0.5 — the convention used
    throughout the paper for 0/1-label linear probes.

    Args:
        preds : (N, 1) or (N,) float tensor — raw probe scores
        truth : (N, 1) or (N,) float tensor — ground-truth labels {0.0, 1.0}

    Returns:
        Percentage accuracy as a float in [0.0, 100.0]
    """
    assert len(preds) == len(truth), "Prediction / truth length mismatch"

    preds = preds.reshape(truth.shape)   # align shapes before comparison

    # Binary threshold at 0.5 — mirrors paper's accuracy_fn for shape[1]==1
    preds_bin = torch.where(preds >= 0.5, 1, 0)
    truth_bin = torch.where(truth >= 0.5, 1, 0)

    acc = torch.sum(preds_bin == truth_bin).float() / len(preds) * 100
    return acc.item()



# -----------------------------------------------------------------------
# 3. Linear probe training
#    Adapted from direction_utils.train_linear_probe_on_concept()
#    with use_bias=False (paper's LinearProbeToolkit default)
# -----------------------------------------------------------------------
def train_linear_probe(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X:   torch.Tensor,
    val_y:   torch.Tensor,
) -> tuple:
    """
    Fit a ridge-regression probe via closed-form least-squares with λ sweeping.

    Adapted from direction_utils.train_linear_probe_on_concept() in the paper,
    called with use_bias=False (the LinearProbeToolkit default).

    The solver picks the regularization λ that minimises MSE on the inner-val
    split — the only role val data plays here.  Test data is never seen.

    Two solution paths (paper's dual-form selection):
        n > d  (primal) : β = (X^T X + λI)^{-1}  X^T y
        n ≤ d  (dual)   : α = (X X^T + λI)^{-1}  y,  β = X^T α
    The dual path is more efficient when training samples < hidden dim,
    which is typical for our dataset (n≈128, d=2048 for TinyLlama).

    Regularization grid (identical to paper's sweep):
        [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1, 1, 10, 100, 1000, 10000]

    Args:
        train_X : (n_inner_train, d) float32 activations — inner training set
        train_y : (n_inner_train, 1) float32 labels {0.0, 1.0}
        val_X   : (n_inner_val,   d) float32 activations — inner validation set
        val_y   : (n_inner_val,   1) float32 labels {0.0, 1.0}

    Returns:
        (beta, bias):
            beta — (d, 1) weight vector returned on CPU
            bias — scalar 0 (use_bias=False per paper's default)
    """
    n, d = train_X.shape

    # Same λ grid as paper's direction_utils.train_linear_probe_on_concept
    reg_grid = [0., 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100, 1000, 10000]

    best_val_loss = float("inf")
    best_beta     = None

    for reg in reg_grid:
        if n > d:
            # ---- Primal form ---- efficient when we have more samples than features
            # Builds XtX in memory-safe mini-batches
            XtX  = batch_transpose_multiply(train_X, train_X)      # (d, d)
            XtY  = batch_transpose_multiply(train_X, train_y)      # (d, 1)
            beta = torch.linalg.pinv(
                XtX + reg * torch.eye(d, device=DEVICE)
            ) @ XtY                                                 # (d, 1)

        else:
            # ---- Dual form ---- efficient when n << d (our typical scenario)
            # Works in the smaller (n × n) space instead of (d × d)
            X_d   = train_X.to(DEVICE)         # (n, d)
            y_d   = train_y.to(DEVICE)         # (n, 1)

            XXt   = X_d @ X_d.T               # (n, n)
            alpha = torch.linalg.pinv(
                XXt + reg * torch.eye(n, device=DEVICE)
            ) @ y_d                            # (n, 1)
            beta  = X_d.T @ alpha              # (d, 1)

        # Evaluate on inner-val — this is how we pick the best λ
        # Test data is never touched inside this function
        val_preds = val_X.to(DEVICE) @ beta    # (n_val, 1)
        val_loss  = torch.mean(
            (val_preds - val_y.to(DEVICE)) ** 2
        ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_beta     = deepcopy(beta)     # save a copy of the winning weights

    # Return beta on CPU — callers decide when/whether to push to GPU
    return best_beta.cpu(), 0   # bias = 0 (use_bias=False)




# -----------------------------------------------------------------------
# 4. Per-layer probe evaluation
# -----------------------------------------------------------------------
def probe_all_layers(
    train_activations: dict,
    train_labels:      list,
    test_activations:  dict,
    test_labels:       list,
) -> dict:
    """
    Fit one linear probe per transformer layer and evaluate on the test set.

    Mirrors the per-layer loop inside LinearProbeToolkit._compute_directions()
    (control_toolkits.py in the paper) but uses pre-extracted activations
    instead of re-running the frozen model.

    Information-leakage controls:
    • split_train_val() carves inner-val from TRAINING data only.
    • Regularization λ is chosen on inner-val.
    • Test activations are accessed exactly once — after the probe is locked —
      solely to compute the final AUROC and accuracy.
    • No gradient flows anywhere; everything is closed-form linear algebra.

    Per-layer pipeline:
    For each layer l:
        1. Pull raw hidden-state tensors a_l for train + test.
        2. Split training data into inner-train / inner-val via split_train_val.
        3. Fit probe:  train_linear_probe(inner_train, inner_val).
        4. Score test: raw scores -> AUROC (ranking metric, paper's primary)
                                  + accuracy (threshold at 0.5, secondary).

    Args:
        train_activations : Dict[neg_layer_idx -> Tensor(N_train, d)]
        train_labels      : List[int] of length N_train  (0 or 1)
        test_activations  : Dict[neg_layer_idx -> Tensor(N_test, d)]
        test_labels       : List[int] of length N_test   (0 or 1)

    Returns:
        results : Dict[int -> {'auroc': float, 'accuracy': float}]
                  Keyed by the same negative layer indices as the input dicts.
    """
    # Convert label lists to (N, 1) float32 tensors — paper's standard probe shape
    all_train_y = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)
    all_test_y  = torch.tensor(test_labels,  dtype=torch.float32).reshape(-1, 1)

    # Inner train/val split indices — drawn from TRAINING data exclusively
    N_train = len(train_labels)
    inner_train_idx, inner_val_idx = split_train_val(N_train)

    # Slice out the corresponding label vectors for the inner split
    inner_train_y = all_train_y[list(inner_train_idx)]    # (n_inner_train, 1)
    inner_val_y   = all_train_y[list(inner_val_idx)]      # (n_inner_val,   1)

    print(f"  Inner-train: {len(inner_train_idx)} samples | "
          f"Inner-val: {len(inner_val_idx)} samples | "
          f"Test: {len(test_labels)} samples")

    results = {}
    layers  = list(train_activations.keys())   # e.g. [-1, -2, ..., -21]

    for layer_idx in tqdm(layers, desc="  Probing layer"):

        # ---- Pull activations at this layer ----
        # fp16 -> fp32: the paper always calls .float() before linear algebra
        # to avoid overflow/underflow in matrix multiplications
        train_acts = train_activations[layer_idx].float()   # (N_train, d)
        test_acts  = test_activations[layer_idx].float()    # (N_test,  d)

        # ---- Inner train/val split (test activations NOT touched here) -----
        inner_train_X = train_acts[list(inner_train_idx)]   # (n_inner_train, d)
        inner_val_X   = train_acts[list(inner_val_idx)]     # (n_inner_val,   d)

        # ---- Fit probe — λ tuned on inner-val only ----
        beta, bias = train_linear_probe(
            inner_train_X, inner_train_y,
            inner_val_X,   inner_val_y,
        )
        # beta: (d, 1) on CPU;  bias: scalar 0 (no intercept, use_bias=False)

        # ---- Evaluate on TEST set — one and only access to test data ----
        # Linear probe score: x · β  (a single real-valued score per sample)
        test_preds = (test_acts.to(DEVICE) @ beta.to(DEVICE) + bias).cpu()
        # test_preds: (N_test, 1) raw float scores

        # Accuracy — threshold raw scores at 0.5 (paper's accuracy_fn rule)
        acc = accuracy_fn(test_preds, all_test_y)

        # AUROC — uses continuous (unthresholded) scores; rank-based, no cutoff.
        # Primary monitoring metric in the paper: Fig. 6 and Table S8.
        try:
            auroc = roc_auc_score(
                y_true  = all_test_y.numpy().flatten(),    # (N_test,) ints {0,1}
                y_score = test_preds.numpy().flatten(),    # (N_test,) float scores
            )
        except ValueError:
            # Only happens when the test set contains a single class
            auroc = float("nan")
            print(f"\n  [WARNING] Layer {layer_idx}: AUROC undefined "
                  f"(single class in test labels).")

        results[layer_idx] = {
            "auroc":    round(auroc, 4),
            "accuracy": round(acc,   2),
        }

        print(f"  Layer {layer_idx:+4d} -> "
              f"AUROC: {auroc:.4f}  |  Acc: {acc:.2f}%")

    return results



# -----------------------------------------------------------------------
# 5. Persistence — save and load probe results
# -----------------------------------------------------------------------
def _results_path(model_id: str, tag: str, save_dir: Path) -> Path:
    return exp_dir(save_dir, model_id) / f"probe_results__{tag}.pkl"


def save_probe_results(
    results:  dict,
    model_id: str,
    tag:      str = "default",
    save_dir: Path = PROBE_RESULTS_DIR,
) -> Path:
    """
    Persist per-layer AUROC + accuracy results to a .pkl file.

    File written: probe_results/<model_slug>/<experiment>/probe_results__<tag>.pkl
    `tag` identifies the training dataset, eval set and scope, e.g. "pooled__misogyny__heldout__all".
    """
    out_path = _results_path(model_id, tag, save_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[save_probe_results] Saved -> {out_path}")
    return out_path


def load_probe_results(
    model_id: str,
    tag:      str = "default",
    save_dir: Path = PROBE_RESULTS_DIR,
) -> dict:
    """Load previously saved per-layer probe results -> Dict[int -> {'auroc': float, 'accuracy': float}]."""
    out_path = _results_path(model_id, tag, save_dir)

    with open(out_path, "rb") as f:
        results = pickle.load(f)

    print(f"[load_probe_results] Loaded {len(results)} layers from {out_path}")
    return results




# -----------------------------------------------------------------------
# 6. Summary display
# -----------------------------------------------------------------------
def print_summary(results: dict) -> None:
    """
    Print a per-layer AUROC + accuracy table and highlight the best layer.

    Layers are shown earliest -> latest (most negative index -> -1), matching
    the intuition that concept-relevant information accumulates through depth.
    Also flags the best single layer by AUROC — this is the 'best per-block
    probe' the paper compares against the aggregated probe in Fig. 6 / Table S8.
    """
    print("\n" + "=" * 58)
    print(f"  {'Layer':>6}    {'AUROC':>8}    {'Accuracy':>10}")
    print("=" * 58)

    best_auroc = -1.0
    best_layer = None

    # Sort ascending: most negative (earliest layer) -> -1 (last transformer block)
    for layer_idx in sorted(results.keys()):
        auroc = results[layer_idx]["auroc"]
        acc   = results[layer_idx]["accuracy"]
        flag  = " -> best" if (not np.isnan(auroc) and auroc > best_auroc) else ""

        print(f"  {layer_idx:+6d}    {auroc:8.4f}    {acc:8.2f}%{flag}")

        if not np.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer_idx

    print("=" * 58)
    if best_layer is not None:
        print(f"  Best layer by AUROC: {best_layer:+d}  ->  {best_auroc:.4f}")
    print("=" * 58 + "\n")



# -----------------------------------------------------------------------
# 7. End-to-end pipeline
# -----------------------------------------------------------------------
def main(
    model_id: str = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    dataset_name: str = POOLED_NAME,
    eval_concept: str = None,
    eval_set: str = "main",
) -> dict:
    """
    End-to-end per-layer linear probe pipeline.

        1. Load the 400 training rows of `dataset_name` and an evaluation set (neutral=0 vs misogyny=1 pairs)
        2. For each transformer layer l:
               a. Split training activations into inner-train / inner-val via split_train_val
               b. Sweep lambda, fit ridge-regression probe on inner-train, pick best lambda by inner-val MSE
               c. Score the evaluation set -> AUROC + accuracy
        3. Print per-layer summary table
        4. Save results

    Args:
        dataset_name : training dataset for the probe (default: the pooled 400-row set)
        eval_set     : "main" = fresh statements with SEEN subtype phrases; "heldout" = with the HELD-OUT phrases
        eval_concept : restrict the evaluation set to pairs rendered with this one subtype phrase (default: all)

    Information-leakage controls:
    - Training rows and the evaluation sets are loaded separately; the test statements are not in any training pool
      (checked by dataset_builder.validate), and held-out phrases are not in the pooled dataset.
    - The inner-val fold for lambda selection is drawn from the training rows only.
    - Evaluation activations are accessed once, after the probe is locked.
    """
    print(f"\n[main] ---- Linear Probe (per-layer, raw activations) ----")
    print(f"[main] Model   : {model_id}")
    print(f"[main] Train   : {dataset_name}")
    print(f"[main] Test    : eval set = {eval_set}, scope = {eval_concept or 'all phrases'}")
    print(f"[main] Device  : {DEVICE}")

    train_activations, train_labels = load_train_activations(model_id, dataset_name)
    test_activations, test_labels = load_eval_activations(model_id, eval_set, concept=eval_concept)

    print(
        f"[main] Train: {len(train_labels)} samples | "
        f"Test: {len(test_labels)} samples | "
        f"Layers: {len(train_activations)}"
    )

    layer_results = probe_all_layers(
        train_activations = train_activations,
        train_labels      = train_labels,
        test_activations  = test_activations,
        test_labels       = test_labels,
    )

    print_summary(layer_results)

    save_probe_results(layer_results, model_id=model_id,
                       tag=f"{dataset_name}__{eval_set}__{eval_concept or 'all'}".replace(" ", "_"))

    print("[main] Done.")
    return layer_results


if __name__ == "__main__":
    import argparse
    import experiment
    parser = argparse.ArgumentParser(description="Per-layer ridge linear probe.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--dataset", default=POOLED_NAME, help="training dataset name (default: pooled__misogyny)")
    parser.add_argument("--set", dest="eval_set", default="main", choices=["main", "heldout"],
                        help="main = seen phrases, heldout = unseen phrases")
    parser.add_argument("--eval-concept", default=None, help="restrict the evaluation set to one subtype's phrase")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(model_id=args.model_id, dataset_name=args.dataset, eval_concept=args.eval_concept, eval_set=args.eval_set)
