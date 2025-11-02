import os
import itertools
from typing import List, Union
import pandas as pd
import numpy as np
import tensorly as tl
from pathlib import Path
from tensorly.decomposition import (
    non_negative_parafac,
    tucker,
    parafac,
)
import torch  # Import torch to check for CUDA
from typing import Tuple


tl.set_backend("pytorch")

device = "cuda" if torch.cuda.is_available() else "cpu"


from src.utils import (
    get_logger,
    build_multifeature_X_matrix,
    save_csv_or_parquet,
)

logger = get_logger(__name__)
logger.info(f"Tensorly backend set to 'pytorch'. Using device: {device}")


def tune_ranks(
    method: str,
    df: pd.DataFrame,
    features: Union[str, List[str]],
    output_path: Path,
    rank_list: List[int] = None,
    store_ranks: List[int] = None,
    item_ranks: List[int] = None,
    feature_ranks: List[int] = None,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Calls fit_and_decompose for every combination of ranks and saves
    PVE/RMSE results to a file.

    Args:
        method: 'tucker', 'parafac', or 'ntf'
        df: Input DataFrame
        features: List of feature names to use
        output_path: Path to save the resulting file
        rank_list: (For 'parafac'/'ntf') A list of ranks to test (e.g., [5, 10, 15])
        store_ranks: (For 'tucker') List of ranks for mode 0
        item_ranks: (For 'tucker') List of ranks for mode 1
        feature_ranks: (For 'tucker') List of ranks for mode 2
        n_iter: Max iterations
        tol: Tolerance
    """
    results = []

    if method == "tucker":
        # Check for correct inputs
        if not (store_ranks and item_ranks and feature_ranks):
            logger.error(
                "For 'tucker' method, you must provide store_ranks, item_ranks, and feature_ranks."
            )
            return pd.DataFrame()

        # Generate all combinations
        rank_combinations = list(
            itertools.product(store_ranks, item_ranks, feature_ranks)
        )
        total_runs = len(rank_combinations)
        logger.info(
            f"--- Starting Tucker rank tuning. Testing {total_runs} combinations. ---"
        )

        for i, rank_tuple in enumerate(rank_combinations):
            logger.info(
                f"*** Testing Tucker combo {i+1}/{total_runs}: {rank_tuple} ***"
            )
            try:
                pve, rmse = fit_and_decompose(
                    method=method,
                    df=df,
                    features=features,
                    ranks=rank_tuple,
                    n_iter=n_iter,
                    tol=tol,
                )
                results.append(
                    {
                        "rank_store": rank_tuple[0],
                        "rank_item": rank_tuple[1],
                        "rank_feature": rank_tuple[2],
                        "pve": pve,
                        "rmse": rmse,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed on rank {rank_tuple}: {e}", exc_info=True
                )
                results.append(
                    {
                        "rank_store": rank_tuple[0],
                        "rank_item": rank_tuple[1],
                        "rank_feature": rank_tuple[2],
                        "pve": np.nan,
                        "rmse": np.nan,
                    }
                )

    elif method in ["parafac", "ntf"]:
        # Check for correct inputs
        if not rank_list:
            logger.error(f"For '{method}' method, you must provide rank_list.")
            return pd.DataFrame()

        total_runs = len(rank_list)
        logger.info(
            f"--- Starting {method} rank tuning. Testing {total_runs} ranks. ---"
        )

        for i, rank in enumerate(rank_list):
            logger.info(
                f"*** Testing {method} rank {i+1}/{total_runs}: {rank} ***"
            )
            try:
                pve, rmse = fit_and_decompose(
                    method=method,
                    df=df,
                    features=features,
                    ranks=rank,
                    n_iter=n_iter,
                    tol=tol,
                )
                results.append({"rank": rank, "pve": pve, "rmse": rmse})
            except Exception as e:
                logger.error(f"Failed on rank {rank}: {e}", exc_info=True)
                results.append({"rank": rank, "pve": np.nan, "rmse": np.nan})

    else:
        logger.error(
            f"Invalid method provided: {method}. Must be 'tucker', 'parafac', or 'ntf'."
        )
        return pd.DataFrame()

    # --- Save results ---
    if not results:
        logger.warning("No results were generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Check if all results are NaN (all failed)
    if results_df[["pve", "rmse"]].isna().all().all():
        logger.error("All rank tuning runs failed. Returning empty DataFrame.")
        return pd.DataFrame()

    # Ensure output directory exists
    output_dir = output_path.parent
    if output_dir:  # Check if it's not an empty string
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Tuning complete. Saving results to {output_path}")
    save_csv_or_parquet(results_df, output_path)

    # Log the best results
    best_pve = results_df.sort_values(by="pve", ascending=False)
    logger.info(f"Best results by PVE:\n{best_pve.head()}")

    best_rmse = results_df.sort_values(by="rmse", ascending=True)
    logger.info(f"Best results by RMSE:\n{best_rmse.head()}")

    return results_df


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
    elif method == "parafac":
        logger.info(f"Performing PARAFAC decomposition with rank={rank_tuple}")
        # Pass the device tensors to the function
        weights, factors = parafac_decomposition(X_mat, rank_tuple)
    else:
        raise ValueError(f"Invalid method: {method}")

    pve_percent, rmse = errors(X_mat, weights, factors)
    return pve_percent, rmse


def _nanstd(tensor, dim=None, keepdim=False, ddof=1):
    """
    Calculates nan-safe standard deviation, implementing ddof.
    torch.nanstd wasn't available in older torch versions.
    """
    # Calculate the mean, keeping the dimensions to enable broadcasting
    tensor_mean = torch.nanmean(tensor, dim=dim, keepdim=True)

    # Calculate the squared differences from the mean
    squared_diffs = torch.pow(tensor - tensor_mean, 2)

    # Sum the squared differences
    sum_sq_diff = torch.nansum(squared_diffs, dim=dim, keepdim=keepdim)

    # Count non-NaN elements
    count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=keepdim)

    # Apply ddof (delta degrees of freedom)
    n = count - ddof
    n = torch.clamp(n, min=0)  # n cannot be negative

    # Calculate variance: sum( (x-mu)^2 ) / (N - ddof)
    nan_variance = sum_sq_diff / n

    # Handle division by zero if n=0 (e.g., all NaNs or N <= ddof)
    nan_variance = torch.where(n > 0, nan_variance, float("nan"))

    # Take the square root to get the standard deviation
    output = torch.sqrt(nan_variance)

    return output


def center_scale_signed(
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
    logger.info(f"Centering and scaling {I*J*D} values on device={X.device}")

    # Create new tensors on the same device and with the same dtype as X
    mus = tl.zeros(D, device=X.device, dtype=X.dtype)
    sds = tl.ones(D, device=X.device, dtype=X.dtype)

    for d in range(D):
        vals = X[..., d][M]  # Boolean mask works on tensors

        # Handle case where a feature has no observed values
        if vals.shape[0] == 0:
            mu = 0.0
            sd = 1.0
        else:
            mu = torch.nanmean(vals)
            sd = _nanstd(
                vals, dim=0, keepdim=False, ddof=1
            )  # ddof=1 for sample std dev

        sd = sd if sd > eps else 1.0
        X[..., d] = (X[..., d] - mu) / sd
        mus[d] = mu
        sds[d] = sd

    # Neutral imputation (this indexing works on tensors)
    X[~M, :] = 0.0
    return X, mus, sds


def errors(
    X: tl.tensor, weights: tl.tensor, factors: list[tl.tensor]
) -> Tuple[float, float]:
    """Calculates PVE using tensorly (backend-agnostic) functions."""
    X_hat = tl.cp_to_tensor((weights, factors))

    # 5. Calculate SSE and RMSE (on device)
    finite_mask = torch.isfinite(X)
    X_finite_orig = X[finite_mask]
    if not tl.any(finite_mask):
        return np.nan, np.nan

    X_finite_hat = X_hat[finite_mask]
    sse_vec = X_finite_orig - X_finite_hat
    sse = tl.dot(sse_vec, sse_vec)  # vdot uses the backend
    num_finite = X_finite_orig.shape[0]
    rmse = tl.sqrt(sse / num_finite)  # tl.sqrt uses the backend

    # Calculate PVE (on device)
    mu = tl.mean(X_finite_orig)  # tl.mean uses the backend
    tss_vec = X_finite_orig - mu
    tss = tl.dot(tss_vec, tss_vec)

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

    # Prepare data for NTF (on device)
    X_imputed = torch.nan_to_num(X, nan=0.0)
    X_imputed[X_imputed < 0] = 0.0  # In-place op on device tensor

    # Fit the Non-negative PARAFAC (CP) model (on device)
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


def parafac_decomposition(  # Renamed function for clarity
    X: tl.tensor, rank: int = 10
) -> Tuple[tl.tensor, list[tl.tensor]]:
    """
    Performs PARAFAC (CP) decomposition on a 3D tensor, suitable for signed data.

    Args:
        X (tl.tensor): Input data tensor (3D). May contain NaNs.
        rank: The factorization rank (R).

    Returns:
        (weights, factors)
    """

    # Prepare data (Impute NaNs, but DO NOT clip negative values)
    X_imputed = torch.nan_to_num(X, nan=0.0)
    # Fit the standard PARAFAC (CP) model (on device)
    try:
        # USE 'parafac' instead of 'non_negative_parafac'
        weights, factors = parafac(
            X_imputed,  # This is a device tensor
            rank=rank,
            init="random",  # 'svd' is also a good init for 'parafac'
            random_state=42,
            n_iter_max=500,
            tol=1e-6,
        )
    except ValueError as e:
        logger.error(f"Error during PARAFAC fit: {e}")
        return None, None

    return weights, factors
