"""
Per-layer direction projection probe on learned concept vectors.

For each transformer layer l, this step evaluates how well the single scalar quantity  <a_l, v_l>  
— the dot product of the raw hidden state with the trained concept vector — can separate the two classes.

This implements the "direction probe" (also called "projection probe") from Appendix H / Table S8 / Fig. 6 of Beaglehole et al. (arXiv:2502.03708),
adapted to use pre-extracted activations and pre-trained concept vectors instead of re-running the frozen model.

Pipeline per layer l:
    1. Load the signed unit-norm concept direction v_l  (from concept_vector_training.py)
    2. Compute scalar projections:   proj = a_l @ v_l     shape: (N,)
    3. Fit a 1D affine calibration (slope m, intercept b) on inner-train projections so that the output lives in the same scale as the binary labels {0, 1}.
       Adapted from direction_utils.linear_solve(use_bias=True) in the paper, the same step used by LinearProbeToolkit and MeanDifferenceToolkit.
    4. Apply calibration to test projections:  test_preds = proj_test * m + b
    5. AUROC  (primary metric, continuous scores) + Accuracy (secondary metric, threshold at 0.5)

Information leakage controls (matching the paper's design):
    - concept vectors v_l were learned on TRAINING data only
    - The 1D calibration (m, b) is fit on the INNER TRAINING split only (same split as linear_probe.py, carved from training data)
    - Inner-val was used only for concept-direction hyperparameter selection
    - Test activations are accessed exactly once, after calibration is locked
    - No model forward passes happen here — everything works on saved tensors
"""


import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset_builder import SEED
from experiment import PROJECT_ROOT, exp_dir

# These load pre-extracted hidden states from disk (no model re-run)
from activation_store import load_eval_activations, load_train_activations
from dataset_io import concept_for_vector
from raw_data import train_dataset_for_vector

# split_train_val splits training indices into inner-train / inner-val.
# accuracy_fn thresholds scores at 0.5 and computes binary classification accuracy.
# Both are reused here unchanged — same split sizes, same accuracy convention.
from linear_probe import split_train_val, accuracy_fn

# load_concept_vectors loads the trained RFM/mean-difference directions v_l.
# project_onto_direction computes a_l @ v_l for batches of hidden states.
from concept_vector_training import DEFAULT_VECTOR_NAME, load_concept_vectors, project_onto_direction

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Pick GPU when available, fall back to CPU 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# All direction projection probe results land here
DIRECTION_PROBE_RESULTS_DIR = PROJECT_ROOT / "direction_probe_results"
DIRECTION_PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# 1. 1D affine calibration
#    Adapted from direction_utils.linear_solve(use_bias=True) in the paper, the exact function called inside LinearProbeToolkit._compute_directions()
#    and MeanDifferenceToolkit._compute_directions() to calibrate raw scalar projections into label-scale predictions.
# -----------------------------------------------------------------------
def calibrate_1d_projection(
    proj_inner_train: torch.Tensor,
    inner_train_y:    torch.Tensor,
) -> tuple:
    """
    Fit a 1D affine map  (slope m, intercept b)  that converts raw scalar projection scores into label-scale predictions.

    Directly adapted from direction_utils.linear_solve(use_bias=True) in the paper.  This is the calibration step that bridges the gap between
        • the raw dot-product  <a_l, v_l>  (unit-norm direction; values near 0)
        • binary labels {0.0, 1.0} that the classifier must predict

    The solve is standard OLS (no regularization — only 2 unknowns, so over-fitting risk is negligible regardless of dataset size).

    Information leakage control:
        proj_inner_train and inner_train_y MUST come from the inner training split only.  
        Inner-val was already "spent" on direction selection; test data must never enter this function.

    Primal form (n >> 2 always for our dataset sizes):
        Augmented design matrix  X = [proj | 1]  shape (n, 2)
        Normal equations:        beta = pinv(X^T X) @ X^T y   shape (2, 1)
        Returns:  m = beta[0]  (slope, shape (1,1))
                  b = beta[1]  (intercept, scalar float)

    Args:
        proj_inner_train : (n_inner_train, 1) float32 — raw scalar projections for the inner-training split
        inner_train_y    : (n_inner_train, 1) float32 — labels {0.0, 1.0} for the inner-training split

    Returns:
        (m, b):
            m — (1, 1) float32 tensor — slope / regression coefficient
            b — float — intercept / bias term
    """
    # Ensure float32 on CPU for the linear algebra (no GPU needed for 2x2 system)
    proj  = proj_inner_train.float().cpu()
    y_cal = inner_train_y.float().cpu()

    # Make sure both tensors are (n, 1) — not (n,) — so matmul shapes are consistent
    if proj.dim() == 1:
        proj = proj.unsqueeze(1)       # (n,) -> (n, 1)
    if y_cal.dim() == 1:
        y_cal = y_cal.unsqueeze(1)     # (n,) -> (n, 1)

    n = proj.shape[0]

    # Augment projection column with a constant-1 column to absorb the intercept.
    # The system to solve:   [ proj_1,  1 ]          [ y_1 ]
    #                        [ proj_2,  1 ]  [m]   =  [ y_2 ]
    #                        [  ...  ,  . ]  [b]      [ ... ]
    X_aug = torch.cat(
        [proj, torch.ones(n, 1, dtype=torch.float32)],
        dim=1,
    )   # (n, 2)

    # n >> 2 for all realistic dataset sizes, so always use the primal OLS path:
    # beta = (X^T X)^{-1} X^T y
    XtX  = X_aug.T @ X_aug        # (2, 2) Gram matrix — tiny, fast to invert
    XtY  = X_aug.T @ y_cal        # (2, 1) cross-correlation vector
    beta = torch.linalg.pinv(XtX) @ XtY   # (2, 1) — [slope, intercept]

    # Split slope and intercept — mirrors paper's `sol = beta[:-1]; bias = beta[-1].item()`
    m = beta[:-1]                  # (1, 1) slope tensor
    b = beta[-1].item()            # float  intercept scalar

    return m, b



# -----------------------------------------------------------------------
# 2. Per-layer direction projection evaluation
#    Adapted from LinearProbeToolkit._compute_directions() (direction eval path) and MeanDifferenceToolkit._compute_directions() in control_toolkits.py,
#    plus the evaluation loop in NeuralController.evaluate_directions() in neural_controllers.py — all from the research paper.
# -----------------------------------------------------------------------
def evaluate_direction_projection_all_layers(
    train_activations:  dict,
    train_labels:       list,
    test_activations:   dict,
    test_labels:        list,
    concept_directions: dict,
) -> dict:
    """
    Evaluate the scalar projection  <a_l, v_l>  as a binary concept detector
    on every transformer layer and return per-layer AUROC and accuracy.

    Implements the 'direction probe' evaluation described in Appendix H and
    reported in Table S8 / Fig. 6 of the paper.  Unlike the linear probe
    (linear_probe.py) which trains a full d-dimensional weight vector, this
    probe uses only the single scalar  <a_l, v_l>  — hence it tests whether
    the pre-trained concept direction v_l is sufficient for detection, not
    just for steering.

    Per-layer algorithm:
        1. Extract signed concept direction v_l (shape d,) from concept_directions.
           v_l is unit-norm; positive projection = concept present (already oriented
           by compute_all_layer_signs() in concept_vector_training.py).
        2. Carve inner-train / inner-val from TRAINING activations
           (same split_train_val() call and same split sizes as linear_probe.py —
           guarantees apples-to-apples comparisons).
        3. Compute scalar projections for inner-train:
               proj_inner = inner_train_acts @ v_l        shape: (n_inner_train,)
        4. Fit 1D affine calibration on inner-train projections:
               [m, b] = OLS(proj_inner, inner_train_y)    2 free parameters
        5. Compute scalar projections for the test set (only access to test data):
               proj_test = test_acts @ v_l                shape: (N_test,)
        6. Apply calibration -> calibrated test predictions:
               test_preds = proj_test * m + b             shape: (N_test, 1)
        7. AUROC from continuous (uncalibrated ranking) scores.
           A positive monotone affine transform (m > 0, which follows from correct
           sign orientation) preserves the ranking, so AUROC is unchanged vs. raw
           projections.  Using calibrated scores is consistent with the paper.
        8. Accuracy from threshold at 0.5 — the paper's standard binary convention.

    Information leakage controls:
        - concept_directions were trained exclusively on TRAINING data (no test).
        - The calibration (m, b) is fit on the INNER TRAINING split only.
        - Inner-val was consumed only for direction hyperparameter selection.
        - Test activations are touched exactly once: steps 5–8 above.
        - No gradient computation — everything is closed-form linear algebra.

    Args:
        train_activations  : Dict[neg_layer_idx -> Tensor(N_train, d)]
                             from load_train_activations()
        train_labels       : List[int] of length N_train  (0 or 1)
        test_activations   : Dict[neg_layer_idx -> Tensor(N_test, d)]
                             from load_eval_activations()
        test_labels        : List[int] of length N_test   (0 or 1)
        concept_directions : Dict[neg_layer_idx -> Tensor(1, d)]
                             from load_concept_vectors() — already sign-oriented

    Returns:
        results : Dict[int -> {'auroc': float, 'accuracy': float}]
                  Keyed by the same negative layer indices as the input dicts.
    """
    # Convert label lists to float32 (N, 1) tensors — matches the paper's convention
    # for binary probes throughout direction_utils.py and linear_probe.py
    all_train_y = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)
    all_test_y  = torch.tensor(test_labels,  dtype=torch.float32).reshape(-1, 1)

    # Carve inner-train / inner-val from TRAINING data only.
    # Reusing split_train_val() from linear_probe.py ensures that both the linear
    # probe and the direction projection probe see the *identical* inner splits,
    # making their per-layer results directly comparable.
    N_train = len(train_labels)
    inner_train_idx, inner_val_idx = split_train_val(N_train)

    # Label slice for the inner-training portion — used only for calibration
    inner_train_y = all_train_y[list(inner_train_idx)]    # (n_inner_train, 1)

    print(
        f"  Inner-train: {len(inner_train_idx)} samples | "
        f"Inner-val: {len(inner_val_idx)} samples | "
        f"Test: {len(test_labels)} samples"
    )

    # Warn early if any layers are missing concept vectors so users can debug
    layers   = list(train_activations.keys())   # e.g. [-1, -2, ..., -21]
    missing  = [l for l in layers if l not in concept_directions]
    if missing:
        print(
            f"  [WARNING] Concept vectors not found for {len(missing)} "
            f"layer(s): {missing}. They will be skipped."
        )

    results = {}

    for layer_idx in tqdm(layers, desc="  Direction projection layer"):

        # Skip silently if no concept vector was trained for this layer
        if layer_idx not in concept_directions:
            continue

        # ---- Pull raw activations for this layer ----
        # Cast fp16 -> fp32 before any linear algebra to prevent overflow.
        # This mirrors the .float() calls in direction_utils.get_hidden_states() and in every toolkit's _compute_directions() loop.
        train_acts = train_activations[layer_idx].float()   # (N_train, d)
        test_acts  = test_activations[layer_idx].float()    # (N_test,  d)

        # ---- Extract the signed concept direction v_l ----
        # concept_directions[layer_idx] is shape (1, d).
        # Index 0 selects the primary (and only) component.
        # After compute_all_layer_signs() in concept_vector_training.py, the sign is already oriented: positive projection => "concept present."
        v_l = concept_directions[layer_idx][0].float()      # (d,) unit-norm vector on CPU

        # ---- Step 3: Scalar projections for the inner-train split ----
        # project_onto_direction() wraps the matmul so it handles device moves.
        # We slice inner-train activations first — test activations stay untouched.
        inner_train_acts = train_acts[list(inner_train_idx)]    # (n_inner_train, d)
        proj_inner_train = project_onto_direction(
            hidden_states=inner_train_acts,   # (n_inner_train, d)
            direction=v_l,                    # (d,) unit-norm direction
            device=DEVICE,
        ).cpu()                               # (n_inner_train,) float32 on CPU

        # ---- Step 4: 1D affine calibration on inner-train projections ----
        # Maps  proj -> m * proj + b  to put predictions on the {0, 1} label scale.
        # Fit happens here; test data is not yet touched.
        m, b = calibrate_1d_projection(
            proj_inner_train=proj_inner_train.unsqueeze(1),  # (n_inner_train, 1)
            inner_train_y=inner_train_y,                     # (n_inner_train, 1)
        )
        # m: (1, 1) slope tensor   b: float intercept

        # ---- Step 5: Scalar projections for the TEST set ----
        # This is the one and only time test activations are accessed in this layer loop.
        # project_onto_direction moves tensors to DEVICE for the matmul and brings them back to CPU to stay consistent with the rest of the pipeline.
        proj_test = project_onto_direction(
            hidden_states=test_acts,   # (N_test, d)
            direction=v_l,             # (d,) unit-norm direction
            device=DEVICE,
        ).cpu().unsqueeze(1)           # (N_test, 1) float32 on CPU

        # ---- Step 6: Calibrated test predictions ----
        # proj_test @ m + b  maps the raw projection scalar to the label scale.
        # Shape: (N_test, 1) @ (1, 1) + float = (N_test, 1)
        test_preds = proj_test @ m + b    # (N_test, 1)

        # ---- Step 7: Accuracy — threshold at 0.5 ----
        # accuracy_fn is reused from linear_probe.py unchanged.
        # Both preds and truth are thresholded at 0.5 -> binary, then compared.
        acc = accuracy_fn(test_preds, all_test_y)

        # ---- Step 8: AUROC — continuous scores, no thresholding needed ----
        # AUROC only cares about ranking, not absolute values.
        # Calibration with m > 0 (correct sign) is a monotone transform: it preserves the ranking so AUROC from test_preds equals AUROC from proj_test.
        # Using test_preds here is consistent with how the paper evaluates direction probes (e.g. compute_classification_metrics(projected_preds, test_y) in the paper).
        try:
            auroc = roc_auc_score(
                y_true  = all_test_y.numpy().flatten(),    # ground-truth labels
                y_score = test_preds.numpy().flatten(),    # continuous prediction scores
            )
        except ValueError:
            # Can happen when the test set accidentally contains only one class
            auroc = float("nan")
            print(
                f"\n  [WARNING] Layer {layer_idx}: AUROC is undefined "
                f"(test set contains only one class)."
            )

        results[layer_idx] = {
            "auroc":    round(auroc, 4),
            "accuracy": round(acc,   2),
        }

        print(
            f"  Layer {layer_idx:+4d} -> "
            f"AUROC: {auroc:.4f}  |  Acc: {acc:.2f}%"
        )

    return results


# -----------------------------------------------------------------------
# 3. Persistence — save and load direction projection probe results
# -----------------------------------------------------------------------

def _results_path(model_id: str, tag: str, save_dir: Path) -> Path:
    return exp_dir(save_dir, model_id) / f"direction_probe_results__{tag}.pkl"


def save_direction_probe_results(
    results:  dict,
    model_id: str,
    tag:      str = "default",
    save_dir: Path = DIRECTION_PROBE_RESULTS_DIR,
) -> Path:
    """
    Persist per-layer AUROC + accuracy results for the direction projection probe.
    File written: direction_probe_results/<model_slug>/<experiment>/direction_probe_results__<tag>.pkl
    `tag` is "<vector_name>__<eval_set>__<scope>", e.g. "averaged__misogyny__heldout__all".
    """
    out_path = _results_path(model_id, tag, save_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[save_direction_probe_results] Saved -> {out_path}")
    return out_path


def load_direction_probe_results(
    model_id: str,
    tag:      str = "default",
    save_dir: Path = DIRECTION_PROBE_RESULTS_DIR,
) -> dict:
    """Load direction projection probe results -> Dict[int -> {'auroc': float, 'accuracy': float}]."""
    out_path = _results_path(model_id, tag, save_dir)
    with open(out_path, "rb") as f:
        results = pickle.load(f)
    print(
        f"[load_direction_probe_results] Loaded {len(results)} layers "
        f"from {out_path}"
    )
    return results


def print_direction_probe_summary(results: dict) -> None:
    """
    Print a per-layer AUROC + accuracy table for the direction projection probe.

    Format is identical to print_summary() in linear_probe.py so the two tables can be visually compared side by side.

    Layers are shown earliest -> latest (most negative index -> -1), following the intuition that deeper layers accumulate more concept-relevant geometry.
    The best layer by AUROC is flagged — this is the 'best per-block direction probe' that the paper compares against the aggregated probe (Fig. 6 / Table S8).
    """
    print("\n" + "=" * 58)
    print("  Direction Projection Probe — Per-Layer Results")
    print("=" * 58)
    print(f"  {'Layer':>6}    {'AUROC':>8}    {'Accuracy':>10}")
    print("=" * 58)

    best_auroc = -1.0
    best_layer = None

    # Sort from most-negative (earliest transformer block) to -1 (last block)
    for layer_idx in sorted(results.keys()):
        auroc = results[layer_idx]["auroc"]
        acc   = results[layer_idx]["accuracy"]

        # Mark the running best so users can spot it at a glance
        is_best = (not np.isnan(auroc)) and (auroc > best_auroc)
        flag    = " -> best" if is_best else ""

        print(f"  {layer_idx:+6d}    {auroc:8.4f}    {acc:8.2f}%{flag}")

        if is_best:
            best_auroc = auroc
            best_layer = layer_idx

    print("=" * 58)
    if best_layer is not None:
        print(f"  Best layer by AUROC: {best_layer:+d}  ->  {best_auroc:.4f}")
    print("=" * 58 + "\n")


# -----------------------------------------------------------------------
# 5. End-to-end pipeline
# -----------------------------------------------------------------------
def main(
    model_id: str = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    vector_name: str = DEFAULT_VECTOR_NAME,
    eval_scope: str = "all",
    eval_set: str = "main",
) -> dict:
    """
    End-to-end direction projection probe pipeline for ONE named vector.

        1. Load the vector (concept_vector_training.py) and the training rows of its dataset
           (the averaged vectors borrow the pooled dataset for calibration).
        2. For each layer: calibrate <a_l, v_l> on inner-train, score the evaluation set.
        3. Print per-layer AUROC + accuracy; save.

    Args:
        vector_name : e.g. "averaged__misogyny", "pooled__misogyny", "single__misogyny", "subtype__hostile_sexism"
        eval_set    : "main" = fresh statements with SEEN phrases; "heldout" = with the HELD-OUT phrases
        eval_scope  : "all" = every pair of the eval set;
                      "own" = only the pairs rendered with this subtype's phrase (subtype__* vectors only; use
                      eval_set "main" for a training subtype and "heldout" for a held-out one)
    """
    print(f"\n[main] ---- Direction Projection Probe ----")
    print(f"[main] Model  : {model_id}")
    print(f"[main] Vector : {vector_name}   (eval set: {eval_set}, scope: {eval_scope})")
    print(f"[main] Device : {DEVICE}")

    directions, signs, method = load_concept_vectors(model_id=model_id, vector_name=vector_name)
    layers = sorted(directions.keys())
    print(f"[main] Concept vector method  : {method}")
    print(f"[main] Layers with directions : {len(layers)}")

    eval_concept = None
    if eval_scope == "own":
        eval_concept = concept_for_vector(vector_name)
        if eval_concept is None:
            raise ValueError("eval_scope='own' needs a subtype__* vector")

    train_dataset = train_dataset_for_vector(vector_name)
    train_activations, train_labels = load_train_activations(model_id, train_dataset, layers=layers)
    test_activations, test_labels = load_eval_activations(model_id, eval_set, concept=eval_concept, layers=layers)
    print(
        f"[main] Calibration data: {train_dataset} ({len(train_labels)} rows) | "
        f"Test: {len(test_labels)} rows | Layers: {len(train_activations)}"
    )

    layer_results = evaluate_direction_projection_all_layers(
        train_activations  = train_activations,
        train_labels       = train_labels,
        test_activations   = test_activations,
        test_labels        = test_labels,
        concept_directions = directions,    # already sign-oriented by concept_vector_training.py
    )

    print_direction_probe_summary(layer_results)
    save_direction_probe_results(layer_results, model_id=model_id, tag=f"{vector_name}__{eval_set}__{eval_scope}")

    print("[main] Done.")
    return layer_results


if __name__ == "__main__":
    import argparse
    import experiment
    parser = argparse.ArgumentParser(description="Per-layer direction projection probe for one named vector.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--set", dest="eval_set", default="main", choices=["main", "heldout"], help="main = seen phrases, heldout = unseen phrases")
    parser.add_argument("--scope", default="all", choices=["all", "own"])
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(model_id=args.model_id, vector_name=args.vector, eval_scope=args.scope, eval_set=args.eval_set)
