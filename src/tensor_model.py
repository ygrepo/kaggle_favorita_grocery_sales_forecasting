import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, tucker
from typing import Tuple

# Set NumPy as the backend for consistency
tl.set_backend("numpy")

from src.utils import get_logger, build_multifeature_X_matrix

logger = get_logger(__name__)


def fit_and_decompose(
    df: pd.DataFrame,
    features: str | list[str],
    ranks: int | list[int] | range | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
):
    logger.info("Multifeature mode: reshaping data to (I, J, D)")

    # Parse features if it's a comma-separated string
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
        logger.info(f"Parsed features: {features}")

    # Convert range to list for tensorly compatibility
    if isinstance(ranks, range):
        ranks = list(ranks)
        logger.info(f"Converted ranks range to list: {ranks}")

    X_mat, M, row_names, col_names = build_multifeature_X_matrix(df, features)
    logger.info(f"X_mat shape:{X_mat.shape}")
    tucker_decomposition(X_mat, M, ranks, n_iter, tol)


def center_scale_signed(
    X: np.ndarray, M: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score each feature using only observed entries; then fill missing with 0,
    which is the feature mean after centering.
    """
    I, J, D = X.shape
    logger.info(f"Centering and scaling {I*J*D} values")
    mus = np.zeros(D)
    sds = np.ones(D)
    for d in range(D):
        vals = X[..., d][M]
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=1)
        sd = sd if sd > eps else 1.0
        X[..., d] = (X[..., d] - mu) / sd
        mus[d] = mu
        sds[d] = sd
    # Neutral imputation after centering
    X[~M, :] = 0.0
    return X, mus, sds


def pve(X: np.ndarray, Xhat: np.ndarray):
    num = np.linalg.norm(X - Xhat) ** 2
    den = np.linalg.norm(X) ** 2 + 1e-12
    return 1.0 - num / den


def tucker_decomposition(
    X: np.ndarray,
    M: np.ndarray,
    ranks: int | list[int] | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, mus, sds = center_scale_signed(X, M)
    core, factors = tucker(
        tl.tensor(X), rank=ranks, init="svd", tol=tol, n_iter_max=n_iter
    )
    Xhat = tl.tucker_to_tensor((core, factors))
    logger.info(f"PVE:{pve(X, Xhat):.2f}")
    return core, factors, mus, sds


def fit_ntf_and_get_factors(
    X_3d: np.ndarray, rank: int = 10, n_iter: int = 500, tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Non-Negative CP Decomposition (PARAFAC) on a 3D tensor.

    Args:
        X_3d (I, J, D): Input data tensor (Stores x Items x Features).
        rank: The factorization rank (R), which defines the number of clusters.
        n_iter: Maximum number of iterations.
        tol: Tolerance for convergence.

    Returns:
        (weights, U, V, D) factors, where U and V are the row/column factors.
        U is the Store Membership Factor (I x Rank)
        V is the Item Membership Factor (J x Rank)
    """

    # 1. Prepare Data
    # TensorLy expects the tensor input
    X_tensor = tl.tensor(X_3d)

    # 2. Fit Non-Negative CP Decomposition (PARAFAC)
    # The CP decomposition factors the tensor into a sum of rank-one components.
    # The factors U, V, and D represent the continuous, non-negative cluster memberships
    # for rows (stores), columns (items), and the features dimension.
    weights, factors = non_negative_parafac(
        tensor=X_tensor,
        rank=rank,
        init="random",
        n_iter_max=n_iter,
        tol=tol,
        verbose=True,
        random_state=42,
    )

    # Factors are returned as a list: [Factor_U, Factor_V, Factor_D]
    U_factor = factors[0]  # Store factors (I x Rank)
    V_factor = factors[1]  # Item factors (J x Rank)
    D_factor = factors[2]  # Feature factors (D x Rank)

    return weights, U_factor, V_factor, D_factor
