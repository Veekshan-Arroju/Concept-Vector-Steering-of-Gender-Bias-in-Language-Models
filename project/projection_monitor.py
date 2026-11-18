"""
Projection-based monitoring: aggregate projection monitor.

Builds on the per-layer direction projection probe (direction_projection_probe.py) by aggregating the concept-direction projection scores from 
ALL transformer layers into a single feature vector per sample, then training a logistic monitor on those stacked features.

Corresponds to the 'linear_agg' / 'agg' path inside:
    NeuralController.evaluate_directions()  ->  direction_utils.aggregate_layers()

and is the aggregate monitoring approach reported in Figure 6 and Table S8 from paper.

Algorithm (Appendix L of the paper):
    For each transformer layer l:
        proj_l = a_l @ v_l                             shape: (N,)
    Stack across all L layers:
        X = [proj_l1 | proj_l2 | ... | proj_lL]        shape: (N, L)
    Build feature matrices separately for inner-val and test:
        X_val  : (n_inner_val, L)   used to fit the logistic monitor
        X_test : (N_test,   L)      used for final evaluation only
    Fit logistic monitor:
        Tune C in {1000, 100, 10, 1, 0.1, 0.01} on inner-val AUROC (Appendix L)
        monitor_coef, monitor_bias = logistic_solve(X_val, inner_val_y)
    Evaluate:
        agg_preds = X_test @ monitor_coef + monitor_bias  ->  AUROC + accuracy

Information leakage controls:
    - Concept vectors v_l were learned on inner-train (concept_vector_training.py).
    - 1D per-layer calibrations in direction_projection_probe.py also used inner-train.
    - The logistic monitor is fit on the INNER-VAL split ONLY (no overlap with inner-train).
    - Test activations are accessed exactly once — during feature construction + final scoring.
    - No model forward passes anywhere; all computation runs on pre-extracted tensors.
"""


import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset_builder import SEED
from experiment import PROJECT_ROOT, exp_dir

# These bring pre-extracted hidden states from disk (no model re-run needed)
from activation_store import load_eval_activations, load_train_activations
from dataset_io import concept_for_vector
from raw_data import train_dataset_for_vector

# split_train_val carves inner-train / inner-val from training indices.
# accuracy_fn thresholds continuous scores at 0.5 and returns binary accuracy.
# Both are reused unchanged — same semantics as in all upstream steps.
from linear_probe import split_train_val, accuracy_fn

# load_concept_vectors retrieves the trained concept directions v_l from disk.
# project_onto_direction computes a_l @ v_l for a batch of hidden states.
from concept_vector_training import DEFAULT_VECTOR_NAME, load_concept_vectors, project_onto_direction

# Reproducibility 
torch.manual_seed(SEED)
np.random.seed(SEED)

# Prefer GPU for matmuls; fall back to CPU 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All aggregate monitor results (metrics + fitted coefficients) 
MONITOR_RESULTS_DIR = PROJECT_ROOT / "monitor_results"
MONITOR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Build the stacked projection feature matrix
#    Mirrors the per-layer projection collection inside:
#        LinearProbeToolkit._compute_directions()  ->  direction_outputs['val'/'test']
#        MeanDifferenceToolkit._compute_directions()  ->  direction_outputs['val'/'test']
#    and then the concatenation step at the top of direction_utils.aggregate_layers():
#        val_X = torch.concat(layer_outputs['val'], dim=1)
# -----------------------------------------------------------------------
def build_layer_projection_features(
    activations: dict,
    concept_directions: dict,
) -> torch.Tensor:
    """
    Compute per-layer scalar projections and stack them into a single feature matrix.

    For each layer l:
        proj_l = a_l @ v_l     element-wise dot product     shape: (N,)

    Then concatenate across all L available layers:
        X = [proj_l1 | proj_l2 | ... | proj_lL]             shape: (N, L)

    This matches how the paper's aggregate_layers() receives its input:
        val_X = torch.concat(direction_outputs['val'], dim=1)   shape: (n, L*n_components)
    With n_components=1 (single concept direction per layer), the shapes are identical.

    Layers are iterated in a consistent sorted order (most-negative to -1) so that column positions in X are the same whether we're building the val or test matrix.

    Args:
        activations        : Dict[neg_layer_idx -> Tensor(N, d)] from load_train_activations() / load_eval_activations()
        concept_directions : Dict[neg_layer_idx -> Tensor(1, d)] from load_concept_vectors()

    Returns:
        X : (N, L) float32 tensor on CPU — one column per layer
    """
    layer_projections = []

    # Sort to ensure a deterministic and consistent column order across calls
    for layer_idx in sorted(activations.keys()):

        if layer_idx not in concept_directions:
            # Warn and skip layers without a trained direction (rare in practice)
            print(f"  [WARNING] No concept direction for layer {layer_idx}, skipping.")
            continue

        # fp16 -> fp32 before matmul to prevent overflow
        # This mirrors the .float() calls throughout LinearProbeToolkit and direction_utils
        acts = activations[layer_idx].float()   # (N, d)

        # The concept direction is stored as (1, d); index [0] picks the single direction
        # After compute_all_layer_signs() in concept_vector_training.py the sign is already
        # oriented: positive dot product => "concept present" for this layer
        v_l = concept_directions[layer_idx][0].float()   # (d,)

        # project_onto_direction handles the device movement and computes acts @ v_l
        proj_l = project_onto_direction(
            hidden_states=acts,
            direction=v_l,
            device=DEVICE,
        ).cpu()   # (N,) float32 on CPU

        # Keep as (N, 1) so torch.cat stacks cleanly along dim=1
        layer_projections.append(proj_l.unsqueeze(1))   # (N, 1)

    if not layer_projections:
        raise ValueError(
            "No layers had matching concept directions. "
            "Verify that concept_vector_training.py used the same model_id "
            "as the activations passed here."
        )

    # Horizontal stack -> (N, L), mirroring torch.concat(direction_outputs['val'], dim=1)
    X = torch.cat(layer_projections, dim=1)   # (N, L)
    return X




# -----------------------------------------------------------------------
# 2. Logistic aggregate monitor training
#    Adapted from direction_utils.logistic_solve() in the paper (the aggregation path inside aggregate_layers() when use_logistic=True), 
#    extended with the C-grid hyperparameter search described in Appendix L of the paper.
#
#    Appendix L from paper: "For logistic regression, we tuned the regularization parameter C = 1/λ among {1000, 100, 10, 1, 0.1, 0.01}."
#    Model selection criterion: AUROC on the inner-val set.
# -----------------------------------------------------------------------
def train_logistic_monitor(
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    C_grid: list = None,
) -> tuple:
    """
    Fit a logistic regression monitor over the stacked layer projection features.

    Directly adapts direction_utils.logistic_solve() from the paper and extends it with the C-grid search described in Appendix L of the paper.

    The selection criterion is AUROC on the inner-val set (not accuracy), which matches the paper's monitoring evaluation metric throughout Fig. 6 / Table S8.

    Information leakage control:
        val_X and val_y MUST come from the inner-val split only.
        Concept directions were fit on inner-train. Test data must never enter here.

    Args:
        val_X  : (n_inner_val, L) float32 — stacked projection features for inner-val
        val_y  : (n_inner_val, 1) float32 — binary labels {0.0, 1.0} for inner-val
        C_grid : list of C values to sweep; defaults to Appendix L grid

    Returns:
        (monitor_coef, monitor_bias):
            monitor_coef : (L, 1) float32 tensor — weight vector over the L layers
            monitor_bias : (1,)  float32 tensor  — intercept
    """
    # Default C grid from Appendix L: "C = 1/λ among {1000, 100, 10, 1, 0.1, 0.01}"
    if C_grid is None:
        C_grid = [1000, 100, 10, 1, 0.1, 0.01]

    # Convert to numpy for sklearn — logistic regression has no native GPU impl
    X_np   = val_X.cpu().numpy()
    y_flat = val_y.cpu().numpy().flatten()   # logistic expects 1-D labels

    best_auroc = -1.0
    best_clf   = None

    for C in C_grid:
        # sklearn LogisticRegression with fit_intercept=True mirrors the paper's logistic_solve() which calls LogisticRegression(fit_intercept=True, max_iter=1000)
        clf = LogisticRegression(
            C=C,
            fit_intercept=True,
            max_iter=1000,
            solver="lbfgs",   # stable for moderately-sized feature vectors
        )
        clf.fit(X_np, y_flat)

        # Score on inner-val with AUROC (paper's primary monitoring metric)
        # We use the raw decision function (scores) rather than predict_proba so the output type is consistent with how we later apply the monitor to test data
        val_scores = clf.decision_function(X_np)   # (n_inner_val,) continuous scores
        try:
            val_auroc = roc_auc_score(y_true=y_flat, y_score=val_scores)
        except ValueError:
            val_auroc = 0.0   # happens only if val set is single-class (edge case)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_clf   = clf

        print(f"    C={C:<8.3g}  val AUROC: {val_auroc:.4f}")

    print(f"  Best C: {best_clf.C}  (val AUROC: {best_auroc:.4f})")

    # Mirror paper's logistic_solve() return format:
    # coef_ shape: (1, L) for binary -> .T gives (L, 1)
    # intercept_ shape: (1,)
    monitor_coef = torch.from_numpy(best_clf.coef_).float().T     # (L, 1)
    monitor_bias = torch.from_numpy(best_clf.intercept_).float()  # (1,)

    return monitor_coef, monitor_bias



# -----------------------------------------------------------------------
# 3. Linear aggregate monitor (alternative to logistic — both in the paper)
#    Adapted from direction_utils.linear_solve(use_bias=True) called inside direction_utils.aggregate_layers() when use_logistic=False.
#    With λ-grid tuning as described in Appendix L for linear regression: "For linear regression, we tuned regularization among λ in {1e-4, 1e-3, 1e-2, 0.1, 1, 10}."
# -----------------------------------------------------------------------
def train_linear_aggregate_monitor(
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    lambda_grid: list = None,
) -> tuple:
    """
    Fit a ridge OLS monitor over the stacked layer projection features.

    Adapted from direction_utils.linear_solve(use_bias=True) in the paper.
    Extended with the λ sweep from Appendix L: "For linear regression, we tuned regularization among λ in {1e-4, 1e-3, 1e-2, 0.1, 1, 10}."

    Provided for completeness and direct comparison with the logistic monitor.
    The logistic monitor is the primary approach described in the task.

    Args:
        val_X       : (n_inner_val, L) — stacked projection features for inner-val
        val_y       : (n_inner_val, 1) — binary labels {0.0, 1.0} for inner-val
        lambda_grid : regularization values to sweep; defaults to Appendix L grid

    Returns:
        (monitor_coef, monitor_bias) as float32 tensors on CPU
    """
    if lambda_grid is None:
        # Appendix L: "λ in {10^-4, 10^-3, 10^-2, 0.1, 1, 10}"
        lambda_grid = [1e-4, 1e-3, 1e-2, 0.1, 1, 10]

    X   = val_X.float().cpu()      # (n_inner_val, L) keep on CPU — the system is tiny
    y   = val_y.float().cpu()      # (n_inner_val, 1)
    n, L = X.shape

    # Augment with a bias column to absorb the intercept — mirrors direction_utils.append_one()
    X_aug = torch.cat([X, torch.ones(n, 1, dtype=torch.float32)], dim=1)   # (n, L+1)

    best_auroc   = -1.0
    best_monitor_coef = None
    best_monitor_bias = None

    for reg in lambda_grid:
        # OLS with Tikhonov regularization: beta = pinv(X^T X + reg*I) @ X^T y
        # Paper's linear_solve uses pinv without reg in the aggregate path, but Appendix L explicitly adds regularization for the monitoring experiments.
        XtX  = X_aug.T @ X_aug                                       # (L+1, L+1)
        XtX  = XtX + reg * torch.eye(L + 1, dtype=torch.float32)    # ridge term
        XtY  = X_aug.T @ y                                            # (L+1, 1)
        beta = torch.linalg.pinv(XtX) @ XtY                          # (L+1, 1)

        # Split slope from bias (paper's convention: sol = beta[:-1], bias = beta[-1])
        coef = beta[:-1]    # (L, 1)
        bias = beta[-1]     # (1,) scalar tensor

        # Evaluate on inner-val using AUROC (same metric as logistic tuning)
        val_scores = (X @ coef + bias).numpy().flatten()
        try:
            val_auroc = roc_auc_score(y_true=y.numpy().flatten(), y_score=val_scores)
        except ValueError:
            val_auroc = 0.0

        if val_auroc > best_auroc:
            best_auroc        = val_auroc
            best_monitor_coef = coef
            best_monitor_bias = bias

    print(f"  Best linear monitor  ->  val AUROC: {best_auroc:.4f}")

    return best_monitor_coef, best_monitor_bias


# -----------------------------------------------------------------------
# 4. Full aggregate projection monitor evaluation
#    Mirrors the full flow of direction_utils.aggregate_layers() (called from NeuralController.evaluate_directions()) but uses pre-extracted activations
#    and pre-trained concept directions instead of running the live model.
# -----------------------------------------------------------------------
def run_aggregate_projection_monitor(
    train_activations:  dict,
    train_labels:       list,
    test_activations:   dict,
    test_labels:        list,
    concept_directions: dict,
    use_logistic:       bool = True,
) -> dict:
    """
    Build the aggregate projection monitor and evaluate it on the test set.

    Implements the 'aggregate projection monitor' from Fig. 6 / Table S8 and Appendix L of the paper. Concept-direction projection scores from every
    transformer layer are concatenated into one feature vector per sample, and a logistic (or linear) monitor is trained on those features.

    Full data flow:
        1. Convert labels to float32; carve inner-train / inner-val from TRAINING data using the same split_train_val() as all upstream steps.
        2. Build inner-val projection feature matrix (used to fit the monitor):
               X_val  = build_layer_projection_features(inner_val_acts, v)
               shape: (n_inner_val, L)
        3. Build test projection feature matrix (used only for final evaluation):
               X_test = build_layer_projection_features(test_acts, v)
               shape: (N_test, L)
        4. Fit logistic monitor on (X_val, inner_val_y) with C-grid sweep.
        5. Apply monitor to test:
               agg_scores = X_test @ monitor_coef + monitor_bias    shape: (N_test, 1)
        6. AUROC (continuous scores) + accuracy (threshold at 0.5).

    Information leakage controls:
        - Concept directions v_l: learned on inner-train (not touched here).
        - The logistic monitor is fit on INNER-VAL only — disjoint from inner-train.
        - Test activations are accessed exactly once: steps 3 + 6 above.
        - No model forward passes here; all computation is on pre-extracted tensors.

    Args:
        train_activations  : Dict[neg_layer_idx -> Tensor(N_train, d)] from load_train_activations()
        train_labels       : List[int] of length N_train  (0 or 1)
        test_activations   : Dict[neg_layer_idx -> Tensor(N_test, d)] from load_eval_activations()
        test_labels        : List[int] of length N_test   (0 or 1)
        concept_directions : Dict[neg_layer_idx -> Tensor(1, d)] from load_concept_vectors() — already sign-oriented
        use_logistic       : True  -> logistic monitor (paper default for monitoring, described in task spec and Appendix L)
                             False -> linear ridge monitor (alternative, also in paper)

    Returns:
        results : dict with keys
            'auroc'        : float  — AUROC on the test set
            'accuracy'     : float  — binary accuracy on test set (threshold at 0.5)
            'monitor_coef' : (L, 1) float32 tensor — fitted aggregator weights
            'monitor_bias' : (1,) float32 tensor   — fitted intercept
            'n_layers'     : int  — number of layers in the feature vector
            'method'       : str  — 'logistic' or 'linear'
    """
    # Float32 tensors in (N, 1) shape — standard convention throughout the pipeline
    all_train_y = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)
    all_test_y  = torch.tensor(test_labels,  dtype=torch.float32).reshape(-1, 1)

    # Carve inner-train / inner-val from TRAINING data only.
    # Reusing split_train_val() from linear_probe.py ensures that the inner splits used here are bit-for-bit identical to those used for concept direction
    # training (concept_vector_training.py) and per-layer calibration (direction_projection_probe.py). 
    # This is the guarantee of apples-to-apples comparison across all probing variants.
    N_train = len(train_labels)
    inner_train_idx, inner_val_idx = split_train_val(N_train)

    # Labels for the inner-val portion — the only label set the monitor sees
    inner_val_y = all_train_y[list(inner_val_idx)]   # (n_inner_val, 1)

    print(
        f"  Inner-train: {len(inner_train_idx)} | "
        f"Inner-val: {len(inner_val_idx)} | "
        f"Test: {len(test_labels)}"
    )

    # Slice inner-val activations from the full training tensors.
    # We do this per-layer to avoid duplicating the entire training activation dict.
    inner_val_activations = {
        layer_idx: acts[list(inner_val_idx)]
        for layer_idx, acts in train_activations.items()
    }

    # Build feature matrices — these are the only places where test activations are used
    print("\n  Building inner-val projection features (all layers) ...")
    X_val = build_layer_projection_features(
        activations=inner_val_activations,
        concept_directions=concept_directions,
    )   # (n_inner_val, L)

    print("  Building test projection features (all layers) ...")
    X_test = build_layer_projection_features(
        activations=test_activations,
        concept_directions=concept_directions,
    )   # (N_test, L)

    L = X_val.shape[1]   # number of layers in the feature vector
    print(f"  Feature dimensionality: {L} layers")

    # Fit the aggregate monitor on inner-val projections.
    # The monitor is trained on the inner-val split because:
    #   a) concept directions were learned on inner-train (concept_vector_training.py)
    #   b) per-layer 1D calibrations were fit on inner-train (direction_projection_probe.py)
    # Inner-val is the remaining clean split that can be used for the aggregator.
    # This matches the paper's evaluate_directions() -> aggregate_layers() flow.
    method_name = "logistic" if use_logistic else "linear"
    print(f"\n  Fitting {method_name} aggregate monitor on inner-val ...")

    if use_logistic:
        # Primary monitor: logistic regression with C-grid sweep (Appendix L)
        monitor_coef, monitor_bias = train_logistic_monitor(X_val, inner_val_y)
    else:
        # Alternative: linear ridge regression with λ-grid sweep (Appendix L)
        monitor_coef, monitor_bias = train_linear_aggregate_monitor(X_val, inner_val_y)

    # Apply the fitted monitor to the test set.
    # Raw linear score: X_test @ monitor_coef + monitor_bias  shape: (N_test, 1)
    # This mirrors: agg_preds = test_X @ agg_beta + agg_bias in aggregate_layers().
    agg_scores = X_test.float() @ monitor_coef.float() + monitor_bias   # (N_test, 1)

    # AUROC — the paper's primary monitoring metric (Fig. 6, Table S8, Appendix L).
    # AUROC is rank-based: the monotone linear transform (score -> same score) preserves rankings, so using raw decision-function scores gives the same AUROC as
    # using sigmoid probabilities. This is consistent with how roc_auc_score is applied throughout the paper (Appendix L: "we used sklearn.metrics.roc_auc_score").
    try:
        auroc = roc_auc_score(
            y_true  = all_test_y.numpy().flatten(),        # ground-truth labels
            y_score = agg_scores.detach().numpy().flatten(),  # continuous monitor scores
        )
    except ValueError:
        auroc = float("nan")
        print("  [WARNING] AUROC undefined — test set may contain only one class.")

    # Accuracy — secondary metric using 0.5 threshold (paper's standard convention in accuracy_fn from direction_utils.py / linear_probe.py).
    # This is the same threshold used in compute_classification_metrics(): preds = torch.where(preds >= 0.5, 1, 0)
    acc = accuracy_fn(agg_scores, all_test_y)

    print(f"\n  Aggregate monitor -> AUROC: {auroc:.4f}  |  Acc: {acc:.2f}%")

    return {
        "auroc":        round(float(auroc), 4),
        "accuracy":     round(float(acc),   2),
        "monitor_coef": monitor_coef,    # (L, 1) fitted aggregator weight vector
        "monitor_bias": monitor_bias,    # (1,)  fitted intercept
        "n_layers":     L,
        "method":       method_name,
    }



# -----------------------------------------------------------------------
# 5. Persistence — save and load monitor results
#    Naming convention mirrors save_direction_probe_results() and save_probe_results() from the upstream steps, but uses a dedicated file prefix and directory.
# -----------------------------------------------------------------------
def _results_path(model_id: str, tag: str, save_dir: Path) -> Path:
    return exp_dir(save_dir, model_id) / f"aggregate_monitor__{tag}.pkl"


def save_monitor_results(
    results:  dict,
    model_id: str,
    tag:      str = "default",
    save_dir: Path = MONITOR_RESULTS_DIR,
) -> Path:
    """
    Persist aggregate monitor results - both the evaluation metrics and the fitted logistic/linear coefficients - to a .pkl file.

    File: monitor_results/<model_slug>/<experiment>/aggregate_monitor__<tag>.pkl
    `tag` is "<vector_name>__<eval_set>__<scope>". The prefix is distinct from probe_results and direction_probe_results
    so outputs from each probing variant are never confused on disk.
    """
    out_path = _results_path(model_id, tag, save_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[save_monitor_results] Saved -> {out_path}")
    return out_path


def load_monitor_results(
    model_id: str,
    tag:      str = "default",
    save_dir: Path = MONITOR_RESULTS_DIR,
) -> dict:
    """
    Load aggregate monitor results saved by save_monitor_results().

    Returns:
        Dict with keys: 'auroc', 'accuracy', 'monitor_coef', 'monitor_bias',
                        'n_layers', 'method'
    """
    out_path = _results_path(model_id, tag, save_dir)

    with open(out_path, "rb") as f:
        results = pickle.load(f)

    print(
        f"[load_monitor_results] Loaded from {out_path} "
        f"(method: {results.get('method', '?')}, "
        f"n_layers: {results.get('n_layers', '?')})"
    )
    return results



# -----------------------------------------------------------------------
# 6. Summary display
# -----------------------------------------------------------------------
def print_monitor_summary(results: dict) -> None:
    """
    Print a human-readable summary for the aggregate projection monitor.

    Unlike the per-layer tables in linear_probe.py and direction_projection_probe.py, the aggregate monitor produces a single prediction, so we show one metric row.

    We also print the top-5 most influential layer positions by absolute monitor weight. 
    This tells us whether early, middle, or late transformer layers drive the detection signal — the paper's aggregate monitor captures all layers, 
    but in practice a few layers dominate.
    """
    auroc    = results.get("auroc",    float("nan"))
    acc      = results.get("accuracy", float("nan"))
    method   = results.get("method",   "unknown")
    n_layers = results.get("n_layers", "?")
    coef     = results.get("monitor_coef")   # (L, 1) or None

    print("\n" + "=" * 66)
    print("  Aggregate Projection Monitor — Summary")
    print("=" * 66)
    print(f"  Method     : {method} regression over {n_layers} layer projections")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  Accuracy   : {acc:.2f}%")
    print("=" * 66)

    # Show which layer positions contribute most — useful for interpretability
    if coef is not None:
        weights = coef.detach().cpu().numpy().flatten()   # (L,)
        top_k   = min(5, len(weights))
        # Sort by absolute value (largest influence first)
        top_idx = np.argsort(np.abs(weights))[::-1][:top_k]

        print(f"\n  Top-{top_k} most influential layer positions (by |weight|):")
        print(f"  {'Rank':>4}  {'Layer pos':>10}  {'Weight':>10}")
        print("  " + "-" * 30)
        for rank, idx in enumerate(top_idx, start=1):
            print(f"  {rank:>4}  {idx:>10}  {weights[idx]:>+10.4f}")

    print("=" * 66 + "\n")


# -----------------------------------------------------------------------
# 7. End-to-end pipeline
# -----------------------------------------------------------------------
def main(
    model_id:     str  = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    vector_name:  str  = DEFAULT_VECTOR_NAME,
    eval_scope:   str  = "all",
    use_logistic: bool = True,
    eval_set:     str  = "main",
) -> dict:
    """
    End-to-end aggregate projection monitor pipeline for ONE named vector.

    Steps:
        1. Load the vector (concept_vector_training.py) and the training rows of its dataset
           (the averaged vector borrows the pooled dataset).
        2. Build inner-val and paired-test projection feature matrices.
        3. Tune the logistic monitor on inner-val using the C-grid (Appendix L).
        4. Evaluate on the paired test -> AUROC + accuracy. Print and save.

    Information leakage controls (unchanged):
        - Training rows and the paired test are separate objects.
        - The logistic monitor is fit on the INNER-VAL split only.
        - Test activations are touched exactly once.
        - No model forward passes occur - all work is on pre-extracted tensors.

    Args:
        vector_name  : e.g. "averaged__misogyny", "pooled__misogyny", "subtype__hostile_sexism"
        eval_set     : "main" = fresh statements with SEEN phrases; "heldout" = with the HELD-OUT phrases
        eval_scope   : "all" or "own" (own = only pairs rendered with this subtype's phrase; subtype__* vectors only)
        use_logistic : True  -> logistic monitor (paper default for monitoring)
                       False -> linear ridge monitor (paper alternative)
    """
    print(f"\n[main] ---- Aggregate Projection Monitor ----")
    print(f"[main] Model      : {model_id}")
    print(f"[main] Vector     : {vector_name}   (eval set: {eval_set}, scope: {eval_scope})")
    print(f"[main] Device     : {DEVICE}")
    print(f"[main] Method     : {'logistic' if use_logistic else 'linear'} regression")

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
        f"[main] Train rows ({train_dataset}): {len(train_labels)} | "
        f"Test: {len(test_labels)} | Layers: {len(train_activations)}"
    )

    print("\n[main] ---- Building & fitting aggregate projection monitor ----")
    monitor_results = run_aggregate_projection_monitor(
        train_activations  = train_activations,
        train_labels       = train_labels,
        test_activations   = test_activations,
        test_labels        = test_labels,
        concept_directions = directions,   # already sign-oriented
        use_logistic       = use_logistic,
    )

    print_monitor_summary(monitor_results)
    save_monitor_results(monitor_results, model_id=model_id, tag=f"{vector_name}__{eval_set}__{eval_scope}")

    print("[main] Done.")
    return monitor_results


if __name__ == "__main__":
    import argparse
    import experiment
    parser = argparse.ArgumentParser(description="Aggregate projection monitor for one named vector.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--vector", default=DEFAULT_VECTOR_NAME)
    parser.add_argument("--set", dest="eval_set", default="main", choices=["main", "heldout"], help="main = seen phrases, heldout = unseen phrases")
    parser.add_argument("--scope", default="all", choices=["all", "own"])
    parser.add_argument("--linear", action="store_true", help="use the ridge-linear monitor instead of logistic")
    experiment.add_experiment_arg(parser)
    args = parser.parse_args()
    experiment.apply_experiment_arg(args)
    main(model_id=args.model_id, vector_name=args.vector, eval_scope=args.scope, use_logistic=not args.linear, eval_set=args.eval_set)
