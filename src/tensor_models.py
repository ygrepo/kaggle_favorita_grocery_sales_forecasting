import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, tucker
import torch  # Import torch to check for CUDA
from typing import Tuple

# --- MODIFICATION: Set backend and device ---
# 1. Set backend to PyTorch
tl.set_backend("pytorch")

# 2. Check for CUDA and set the global device
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- End MODIFICATION ---


from src.utils import get_logger, build_multifeature_X_matrix

logger = get_logger(__name__)
logger.info(f"Tensorly backend set to 'pytorch'. Using device: {device}")


def fit_and_decompose(
    method: str,
    df: pd.DataFrame,
    features: str | list[str],
    ranks: tuple[int, int, int] | int | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
):
    logger.info("Multifeature mode: reshaping data to (I, J, D)")

    # Parse features (no change needed)
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
        logger.info(f"Parsed features: {features}")

    X, M, row_names, col_names = build_multifeature_X_matrix(df, features)
    logger.info(f"X shape:{X.shape}")

    # Move data to the selected device (GPU or CPU)
    # Assuming X_mat is float and M is boolean
    X_mat = tl.tensor(X, device=device, dtype=tl.float32)
    M = tl.tensor(M, device=device, dtype=torch.bool)

    # X is already a tensor, center_scale_signed now accepts tensors
    X, mus, sds = center_scale_signed(X_mat, M)

    I, J, D = X_mat.shape

    # Use provided ranks or compute defaults (no change needed)
    if ranks is None:
        rank_tuple = (max(2, I // 4), max(2, J // 4), max(2, D // 4))
        logger.info(f"No ranks provided, using defaults: {rank_tuple}")
    else:
        rank_tuple = ranks

    if method == "tucker":
        logger.info(f"Performing Tucker decomposition with rank={rank_tuple}")
        # Pass the device tensors to the function
        weights, factors = tucker_decomposition(X_mat, rank_tuple, n_iter, tol)
    elif method == "ntf":
        logger.info(f"Performing NTF decomposition with rank={rank_tuple}")
        # Pass the device tensors to the function
        weights, factors = nonneg_parafac(X_mat, rank_tuple)
    else:
        raise ValueError(f"Invalid method: {method}")

    pve_percent, rmse = errors(X_mat, weights, factors)
    return pve_percent, rmse


def center_scale_signed(
    # --- MODIFICATION: Accept tensors ---
    X: tl.tensor,
    M: tl.tensor,
    eps: float = 1e-8,
) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    # --- End MODIFICATION ---
    """
    Z-score each feature using only observed entries; then fill missing with 0,
    which is the feature mean after centering.
    """
    I, J, D = X.shape
    # --- MODIFICATION: Log device ---
    logger.info(f"Centering and scaling {I*J*D} values on device={X.device}")

    # Create new tensors on the same device and with the same dtype as X
    mus = tl.zeros(D, device=X.device, dtype=X.dtype)
    sds = tl.ones(D, device=X.device, dtype=X.dtype)
    # --- End MODIFICATION ---

    for d in range(D):
        # --- MODIFICATION: Use tensorly (PyTorch) functions ---
        vals = X[..., d][M]  # Boolean mask works on tensors

        # Handle case where a feature has no observed values
        if vals.shape[0] == 0:
            mu = 0.0
            sd = 1.0
        else:
            mu = tl.nanmean(vals)
            sd = tl.nanstd(vals, ddof=1)  # ddof=1 for sample std dev

        sd = sd if sd > eps else 1.0
        X[..., d] = (X[..., d] - mu) / sd
        mus[d] = mu
        sds[d] = sd
        # --- End MODIFICATION ---

    # Neutral imputation (this indexing works on tensors)
    X[~M, :] = 0.0
    return X, mus, sds


def errors(
    X: tl.tensor, weights: tl.tensor, factors: list[tl.tensor]
) -> Tuple[float, float]:
    """Calculates PVE using tensorly (backend-agnostic) functions."""
    X_hat = tl.cp_to_tensor((weights, factors))

    # 5. Calculate SSE and RMSE (on device)
    finite_mask = tl.isfinite(X)
    X_finite_orig = X[finite_mask]
    if not tl.any(finite_mask):
        return np.nan, np.nan

    X_finite_hat = X_hat[finite_mask]
    sse_vec = X_finite_orig - X_finite_hat
    sse = tl.vdot(sse_vec, sse_vec)  # vdot uses the backend
    num_finite = X_finite_orig.shape[0]
    rmse = tl.sqrt(sse / num_finite)  # tl.sqrt uses the backend

    # Calculate PVE (on device)
    mu = tl.mean(X_finite_orig)  # tl.mean uses the backend
    tss_vec = X_finite_orig - mu
    tss = tl.vdot(tss_vec, tss_vec)

    if tss == 0:
        pve_percent = 100.0 if sse == 0 else np.nan
    else:
        pve = 1.0 - (sse / tss)
        pve_percent = pve.item() * 100.0

    # Return scalar floats
    return pve_percent, rmse.item()


def tucker_decomposition(
    X: tl.tensor,
    ranks: tuple[int, int, int] | int | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[tl.tensor, list[tl.tensor]]:

    logger.info(f"Performing Tucker decomposition with rank={ranks}")

    # `tucker` will run on the device of X (GPU or CPU)
    core, factors = tucker(
        X,  # No longer need tl.tensor(X)
        rank=ranks,
        init="svd",
        tol=tol,
        n_iter_max=n_iter,
        verbose=True,
    )

    # Return device tensors
    return core, factors


def nonneg_parafac(
    X: tl.tensor, rank: int = 10
) -> Tuple[tl.tensor, list[tl.tensor]]:
    """
    ... (docstring) ...
    Args:
        X_tensor_np (np.ndarray): Input data tensor (3D). May contain NaNs.
    ...
    """

    # 2. Prepare data for NTF (on device)
    X_imputed = tl.nan_to_num(X, nan=0.0)
    X_imputed[X_imputed < 0] = 0.0  # In-place op on device tensor

    # 3. Fit the Non-negative PARAFAC (CP) model (on device)
    try:
        weights, factors = non_negative_parafac(
            X_imputed,  # This is a device tensor
            rank=rank,
            init="random",
            random_state=42,
            n_iter_max=500,
            tol=1e-6,
        )
    except ValueError as e:
        logger.error(f"Error during NTF fit: {e}")
        return None, None
    return weights, factors
