"""
Recursive Feature Machine (RFM) for concept vector direction learning.

Implements kernel ridge regression with a Laplacian kernel, iterative AGOP (Average Gradient Outer Product) metric updates, 
and top eigenvector extraction via LOBPCG. This produces the raw concept direction per layer before sign orientation is applied.
"""

import torch
from torch.linalg import solve
import time

# RFM math is always done in float32 — this sets the global default for new tensors.
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from tqdm import tqdm

# torcheval is only used in get_err() for diagnostic R² scores.
# The main rfm() training loop doesn't call get_err(), so this import is optional.

try:
    from torcheval.metrics.functional import r2_score as _torcheval_r2
    _HAS_TORCHEVAL = True
except ImportError:
    _HAS_TORCHEVAL = False

    def _torcheval_r2(preds, y):
        # Manual R² fallback so get_err() still works without torcheval installed.
        ss_res = torch.sum((preds - y) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        return (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    



# -----------------------------------------------------------------------
# 1. Distance and kernel helpers
# -----------------------------------------------------------------------
def euclidean_distances_M_2(samples, samples_M_applied, centers, centers_M_applied, threshold=1e-3):
    """
    Compute pairwise Mahalanobis-weighted L2 distances between samples and centers.

    Uses the identity  ||a - b||_M^2 = a^T M a - 2 a^T M b + b^T M b
    where M is the current learned metric matrix.
    M is already factored into samples_M_applied = samples @ M so we don't recompute it inside the loop.

    The diagonal is zeroed for the self-distance case (samples is centers), and distances below the threshold 
    are clamped to zero to avoid sqrt(neg) from floating-point errors.
    """
    # a^T M a  term — (n,) of per-sample squared norms under metric M
    samples_norm = torch.sum(samples_M_applied * samples, dim=1)

    if samples is centers:
        # Self-distance case: reuse the same norms for both sides
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers_M_applied * centers, dim=1)

    # Reshape norms for broadcasting: (n, 1) and (1, m)
    samples_norm = samples_norm.reshape(-1, 1)
    centers_norm = centers_norm.reshape(1, -1)

    # Cross term  -2 a^T M b  via matrix multiply
    distances = samples_M_applied @ centers.T
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)

    if samples is centers:
        # Self-distances are exactly 0 by definition — zero the diagonal in-place
        distances.fill_diagonal_(0)
    else:
        # Tiny negative values from floating-point rounding: clamp to zero
        distances = torch.where(abs(distances) < threshold, 0, distances)

    distances.clamp_(min=0)
    distances.sqrt_()

    return distances


def laplacian_M_2(bandwidth, distances):
    """
    Laplacian kernel K(x, x') = exp(-||x - x'||_M / L).

    Takes pre-computed distances so the kernel matrix can be reused without recomputing distances. 
    bandwidth (L) controls how quickly similarity falls off with distance.
    """
    assert bandwidth > 0, "Bandwidth must be positive"
    kernel_mat = distances.clone()
    kernel_mat.clamp_(min=0)
    gamma = 1.0 / bandwidth    # -1/L exponent factor
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def laplacian_M_3(X, X_M_applied, x, x_M_applied, L):
    """
    Derivative kernel: K(x_i, x_j) / ||x_i - x_j||_M.

    Used inside get_grads_2() to compute the AGOP gradient matrix.
    Points where the distance is zero (self-pairs) produce inf → zeroed out.
    """
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)
    K = laplacian_M_2(L, dist.clone())
    K = K / dist                       # element-wise division
    K[K == float("Inf")] = 0.0         # zero out self-pairs (0/0 = inf -> 0)
    return K


def laplacian_M(X, X_M_applied, x, x_M_applied, L):
    """
    Standard Laplacian kernel (no derivative division).

    Used by solve_kr() to build the kernel matrix for the regression solve.
    """
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)
    K = laplacian_M_2(L, dist.clone())
    K[K == float("Inf")] = 0.0
    return K



# -----------------------------------------------------------------------
# 2. AGOP gradient computation
# -----------------------------------------------------------------------
def get_grads_2(X, x, sol, L, P):
    """
    Compute the AGOP (Average Gradient Outer Product) matrix M = G^T G.

    This is the heart of the iterative metric update.  M captures how the kernel solution changes when inputs are perturbed 
    — its top eigenvector is the direction that best explains the label variation.

    G_i = grad_{x_i} f(x_i)  where f is the kernel predictor.
    M = (G - mean(G))^T (G - mean(G)) / (m * L^2)

    Args:
        X   : (n, d) training inputs (already mean-centered by rfm())
        x   : (m, d) query points (same as X in our loop — self-gradient)
        sol : (n,)   kernel ridge regression weights alpha
        L   : Laplacian bandwidth
        P   : (d, d) current metric matrix M  (confusingly named P here to
              match the paper's variable names)

    Returns:
        M : (d, d) updated metric matrix (not yet normalized — caller does M /= M.max())
    """
    # Apply current metric P to get M-weighted versions of the inputs
    X_M_applied = X @ P
    x_M_applied = X_M_applied if x is X else x @ P

    # Build derivative kernel K_ij = K(x_i, x_j) / dist(x_i, x_j)
    K = laplacian_M_3(X, X_M_applied, x, x_M_applied, L)

    n, d = X.shape
    m, d = x.shape

    # Step 1: weighted feature contribution from training points
    # sol_X1[i] = alpha[i] * (M x_i)
    X1 = X_M_applied.reshape(n, d)
    sol_X1 = sol.reshape(n, 1).expand(n, d) * X1    # (n, d)
    step_1 = K.T @ sol_X1                           # (m, d) — kernel-weighted average

    # Step 2: subtract the query point's own contribution
    # (sol @ K)[j] = sum_i alpha[i] K(x_j, x_i) = f(x_j) in primal form
    sol_K = sol @ K                                  # (m,)
    del K

    sol_K = sol_K.reshape(m, 1).expand(m, d)
    x1 = x_M_applied.reshape(m, d)
    step_2 = sol_K * x1                             # (m, d)

    # G: the per-sample gradient difference, mean-centered for stability
    G = step_1 - step_2
    G = G - torch.mean(G, axis=0, keepdims=True)

    # AGOP matrix: outer product of the gradient matrix with itself
    M = torch.mm(G.T, G)
    M = M / (m * L * L)    # normalize by dataset size and bandwidth squared

    return M



# -----------------------------------------------------------------------
# 3. Kernel ridge regression solver
# -----------------------------------------------------------------------
def solve_kr(X, X_M_applied, y, L, reg):
    """
    Solve kernel ridge regression: (K + reg * I) alpha = y.

    Builds the full (n x n) Laplacian kernel matrix and solves the system via torch.linalg.solve (more stable than explicit inversion).

    Returns None if the matrix is ill-conditioned (LinAlgError), so the rfm() loop can break early rather than crashing.
    """
    K = laplacian_M(X, X_M_applied, X, X_M_applied, L)

    try:
        # Solve the regularized system — returns alpha^T as (n,) or (1, n)
        sol = solve(K + reg * torch.eye(len(K), device=X.device), y).T
    except torch._C._LinAlgError:
        # Matrix is too ill-conditioned at this bandwidth/metric combo
        return None

    return sol

# -----------------------------------------------------------------------
# 4. Validation error (diagnostic only — not called inside the main rfm() loop)
# -----------------------------------------------------------------------
def get_err(sol, X, x, X_M_applied, x_M_applied, y, L):
    """
    Evaluate kernel ridge regression predictions on held-out data.

    Returns MSE and R² score. This is a diagnostic utility called outside the main rfm() training loop 
    — useful for debugging bandwidth selection.
    """
    K_test = laplacian_M(X, X_M_applied, x, x_M_applied, L)
    preds = (sol @ K_test).T
    return torch.mean(torch.square(preds - y)), _torcheval_r2(preds, y)


# -----------------------------------------------------------------------
# 5. Top eigenvector extraction
# -----------------------------------------------------------------------
def get_top_dir_err(X, y, M):
    """
    Extract the top eigenvector of M via LOBPCG and measure its alignment with y.

    The AGOP matrix M accumulates feature-gradient information.
    Its top eigenvector is the direction that most efficiently explains label variation under the current learned metric 
    — this becomes the candidate concept vector v_l.

    Validation correlation (Pearson r between X @ u and y) tells rfm() how good this candidate is, 
    so it can track the best across all iterations.

    Remark: the research paper's version had 'concept_features' instead of 'M' in the retry block 
    — that was an undefined-variable bug. Fixed here.

    Args:
        X : (m, d) validation activations (mean-centered by the caller)
        y : (m, 1) validation labels
        M : (d, d) current AGOP metric matrix

    Returns:
        (val_r, u):
            val_r — absolute Pearson correlation of projections with y
            u     — (d, 1) top eigenvector of M
    """
    # Matches xRFM's get_top_eigenvector() (xrfm/rfm_src/utils.py) — the
    # official library the paper's own code now delegates this exact step to.
    #
    # M is torch.eye(d) on the very first AGOP iteration of every rfm() call
    # (before any update), so its spectrum is exactly degenerate (every
    # vector is a top eigenvector with eigenvalue 1). torch.lobpcg's internal
    # Rayleigh-Ritz subspace becomes rank-deficient on this and errors out —
    # and since M stays extremely close to the identity even with light
    # regularization, torch.linalg.eigh hits the *same* cuSOLVER clustered-
    # eigenvalue failure on GPU (pytorch/pytorch#147544, #159741), so it's
    # not a safe fallback either.
    #
    # torch.linalg.svd doesn't share that failure mode (different cuSOLVER
    # routine), so it's used as the fallback here instead — no CPU round
    # trip needed. One light regularization attempt with LOBPCG, then
    # straight to the robust SVD fallback on any failure.
    epsilon = 1e-6
    try:
        M_reg = M + epsilon * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        s, u = torch.lobpcg(M_reg, k=1)
    except Exception:
        print("Warning: LOBPCG did not converge — falling back to torch.linalg.svd")
        _, _, Vt = torch.linalg.svd(M, full_matrices=False)
        u = Vt[0].reshape(-1, 1)

    # Project validation samples onto this candidate direction
    preds = X @ u      # (m, 1)

    # Pearson |r| between projections and labels — the selection criterion for rfm()
    val_r = torch.abs(
        torch.corrcoef(torch.cat((preds, y), dim=-1).T)
    )[0, 1].item()

    return val_r, u


# -----------------------------------------------------------------------
# 6. Main RFM training loop
# -----------------------------------------------------------------------
def rfm(traindata, testdata, L=10, reg=1e-3, num_iters=10, norm=False):
    """
    Learn the concept direction via iterative RFM: KRR solve -> AGOP update -> repeat.

    Each iteration:
      1. Build Laplacian kernel K under the current metric M.
      2. Solve (K + reg*I) alpha = y  to get kernel-ridge weights.
      3. Compute AGOP matrix  M = G^T G  from the kernel-weighted gradients.
      4. Normalize: M /= M.max()
      5. Extract the top eigenvector u of M via LOBPCG.
      6. Evaluate |Pearson r| of (X_test @ u) vs y_test — keep the best u seen.

    The "testdata" here is the inner-val split (not the held-out test set).
    concept_vector_training.py is responsible for ensuring that only training activations are ever passed in.

    Args:
        traindata  : (X_train, y_train) both float32 on the same device
        testdata   : (X_val, y_val)   inner-val split for tracking best direction
        L          : Laplacian kernel bandwidth
        reg        : Kernel ridge regularization strength
        num_iters  : Number of AGOP update steps
        norm       : If True, row-normalize inputs before training (helps for some layers)

    Returns:
        (best_u, best_r):
            best_u — (d, 1) concept direction that maximized val correlation
            best_r — the peak Pearson |r| achieved on the val set
    """
    X_train, y_train = traindata
    X_test, y_test   = testdata

    # Mean-center inputs — removes the DC offset so metric updates focus on directions of variance rather than the global mean
    mean = torch.mean(X_train, dim=0, keepdims=True)
    X_train = X_train - mean
    X_test  = X_test  - mean

    # Optional row-wise L2 normalization — can help when activation magnitudes vary a lot across samples
    if norm:
        X_train = X_train / torch.norm(X_train, dim=-1).reshape(-1, 1)
        X_test  = X_test  / torch.norm(X_test,  dim=-1).reshape(-1, 1)

    n, d = X_train.shape

    # Initialize M = I (identity metric = plain Euclidean distance)
    M = torch.eye(d, device=X_train.device)

    best_r = -float("inf")
    best_u = None

    for i in range(num_iters):
        # Apply current metric to training inputs (cached so we don't recompute M @ x_i inside the kernel function on every call)
        X_train_M_applied = X_train @ M

        # Solve kernel ridge regression to get alpha weights
        sol = solve_kr(X_train, X_train_M_applied, y_train, L, reg)
        if sol is None:
            # Solver diverged — stop iterating for this (bw, norm) config
            break

        # Extract top eigenvector of current M and measure val correlation
        val_r, u = get_top_dir_err(X_test, y_test, M)

        # Keep the direction that gave the best validation alignment so far
        if val_r > best_r:
            best_r = val_r
            best_u = u.clone()

        # AGOP metric update: M ← G^T G  (feature-gradient outer product)
        M = get_grads_2(X_train, X_train, sol, L, M)
        M /= M.max()    # scale so the largest entry is 1 (numerical stability)

    return best_u, best_r



# -----------------------------------------------------------------------
# 7. Standalone smoke test (mirrors the paper's __main__ block)
# -----------------------------------------------------------------------
def main():
    """Quick sanity check: RFM should recover the direction x_0 + x_1 from random data."""
    n, d = 1000, 100
    torch.manual_seed(0)
    # Detect the best available device so the smoke test runs on CPU-only machines too
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RFM smoke test on: {device}") 

    # X_train = torch.randn(n, d).cuda()
    # X_test  = torch.randn(n, d).cuda()
    X_train = torch.randn(n, d).to(device)
    X_test  = torch.randn(n, d).to(device)

    # Label is a linear function of the first two features — RFM should find this direction
    y_train = (X_train[:, 0] + X_train[:, 1]).reshape(-1, 1)
    y_test  = (X_test[:, 0]  + X_test[:, 1]).reshape(-1, 1)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    start = time.time()
    best_u, best_r = rfm(
        (X_train, y_train),
        (X_test,  y_test),
        reg=1e-3,
        L=10,
        num_iters=10,
    )
    end = time.time()

    print(f"Top direction (first 3 dims): {best_u[:3].squeeze()}")
    print(f"Best val Pearson |r|: {best_r:.4f}")
    print(f"Training time: {end - start:.2f}s")


if __name__ == "__main__":
    main()