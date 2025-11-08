import os
import itertools
from typing import List, Union, Any, Dict, Literal, Optional
import pandas as pd
import numpy as np
import torch
import tensorly as tl
from pathlib import Path
from tensorly.decomposition import (
    non_negative_parafac,
    tucker,
    parafac,
)
from datetime import datetime
from typing import Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

tl.set_backend("pytorch")

device = "cuda" if torch.cuda.is_available() else "cpu"


from src.utils import (
    get_logger,
    build_multifeature_X_matrix,
    save_csv_or_parquet,
)

logger = get_logger(__name__)
logger.info(f"Tensorly backend set to 'pytorch'. Using device: {device}")


def compute_factor_stats(
    factors: list[tl.tensor], factor_names: list[str] = None
) -> dict:
    """
    Computes utilization statistics for each factor matrix and returns
    them as a flat dictionary.

    Args:
        factors: List of factor matrices [factor_0, factor_1, ...]
        factor_names: List of names for each factor (e.g., "Store", "SKU")

    Returns:
        A flat dictionary with prefixed keys (e.g., "Store_empty_components").
    """
    stats_dict = {}
    if not factors:
        return stats_dict

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    for i, F in enumerate(factors):
        name = factor_names[i]

        # Define keys for this factor
        key_total = f"{name}_total_components"
        key_empty = f"{name}_empty_components"
        key_mean = f"{name}_mean_strength"
        key_q25 = f"{name}_q25_strength"
        key_q75 = f"{name}_q75_strength"

        if F is None:
            stats_dict[key_total] = 0
            stats_dict[key_empty] = 0
            stats_dict[key_mean] = np.nan
            stats_dict[key_q25] = np.nan
            stats_dict[key_q75] = np.nan
            continue

        total_components = F.shape[1]
        stats_dict[key_total] = total_components

        if total_components == 0:
            stats_dict[key_empty] = 0
            stats_dict[key_mean] = np.nan
            stats_dict[key_q25] = np.nan
            stats_dict[key_q75] = np.nan
            continue

        try:
            # Calculate the L2 norm (strength) of each component vector (column)
            component_strengths = torch.norm(F, p=2, dim=0)

            stats_dict[key_empty] = torch.sum(
                component_strengths < 1e-8
            ).item()
            stats_dict[key_mean] = torch.mean(component_strengths).item()
            stats_dict[key_q25] = torch.quantile(
                component_strengths, 0.25
            ).item()
            stats_dict[key_q75] = torch.quantile(
                component_strengths, 0.75
            ).item()

        except Exception as e:
            logger.error(
                f"Error computing stats for factor '{name}': {e}",
                exc_info=True,
            )
            stats_dict[key_empty] = np.nan
            stats_dict[key_mean] = np.nan
            stats_dict[key_q25] = np.nan
            stats_dict[key_q75] = np.nan

    return stats_dict


def compute_tucker_core_stats(core: tl.tensor) -> dict:
    """
    Computes magnitude statistics on the core tensor of a Tucker decomposition.

    Args:
        core: The core tensor.

    Returns:
        A flat dictionary of statistics.
    """
    stats_dict = {}
    if core is None:
        return stats_dict

    try:
        # Get the magnitude of all interaction values in the core
        core_magnitudes = torch.abs(core.flatten())

        stats_dict["core_mean_strength"] = torch.mean(core_magnitudes).item()
        stats_dict["core_q25_strength"] = torch.quantile(
            core_magnitudes, 0.25
        ).item()
        stats_dict["core_q75_strength"] = torch.quantile(
            core_magnitudes, 0.75
        ).item()
        stats_dict["core_max_strength"] = torch.max(core_magnitudes).item()

    except Exception as e:
        logger.error(f"Error computing core stats: {e}", exc_info=True)
        stats_dict["core_mean_strength"] = np.nan
        stats_dict["core_q25_strength"] = np.nan
        stats_dict["core_q75_strength"] = np.nan
        stats_dict["core_max_strength"] = np.nan

    return stats_dict


def compute_item_membership_stats(
    factors: list[tl.tensor], factor_names: list[str] = None
) -> dict:
    """
    Computes summary statistics for the per-item membership strengths.

    For each factor matrix (e.g., Stores), it first calculates the
    mean absolute loading for EACH item (e.g., each store).
    It then returns the mean, Q1, and Q3 of those per-item values.
    This measures the "softness" or "sparsity" of item memberships.

    Args:
        factors: List of factor matrices [factor_0, factor_1, ...]
        factor_names: List of names for each factor (e.g., "Store", "SKU")

    Returns:
        A flat dictionary with prefixed keys (e.g., "Store_item_mean_loading_avg").
    """
    stats_dict = {}
    if not factors:
        return stats_dict

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    for i, F in enumerate(factors):
        name = factor_names[i]

        # Define keys for this factor's summary
        key_avg = (
            f"{name}_item_mean_loading_avg"  # The mean of the per-item means
        )
        key_q25 = (
            f"{name}_item_mean_loading_q25"  # The Q1 of the per-item means
        )
        key_q75 = (
            f"{name}_item_mean_loading_q75"  # The Q3 of the per-item means
        )

        if F is None or F.shape[0] == 0:
            stats_dict[key_avg] = np.nan
            stats_dict[key_q25] = np.nan
            stats_dict[key_q75] = np.nan
            continue

        try:
            # 1. Get absolute loadings
            abs_loadings = torch.abs(F)

            # 2. Calculate the mean loading for EACH item (across its components)
            # F shape is [n_items, n_components]. We average across dim=1.
            per_item_mean_loading = torch.mean(
                abs_loadings, dim=1
            )  # Shape [n_items]

            # 3. Compute the summary stats of that per-item vector
            stats_dict[key_avg] = torch.mean(per_item_mean_loading).item()
            stats_dict[key_q25] = torch.quantile(
                per_item_mean_loading, 0.25
            ).item()
            stats_dict[key_q75] = torch.quantile(
                per_item_mean_loading, 0.75
            ).item()

        except Exception as e:
            logger.error(
                f"Error computing item membership stats for '{name}': {e}",
                exc_info=True,
            )
            stats_dict[key_avg] = np.nan
            stats_dict[key_q25] = np.nan
            stats_dict[key_q75] = np.nan

    return stats_dict


def log_tucker_core_stats(stats_dict: dict):
    """
    Logs the statistics computed from the core tensor.
    """
    logger.info("--- Tucker Core Utilization Check ---")

    mean_s = stats_dict.get("core_mean_strength", np.nan)
    q25_s = stats_dict.get("core_q25_strength", np.nan)
    q75_s = stats_dict.get("core_q75_strength", np.nan)
    max_s = stats_dict.get("core_max_strength", np.nan)

    logger.info(
        f"Core Strengths (Mean={mean_s:.4f}, Q1(25%)={q25_s:.4f}, "
        f"Q3(75%)={q75_s:.4f}, Max={max_s:.4f})"
    )
    logger.info("-------------------------------------")


def log_factor_utilization(
    factors: list[tl.tensor],
    stats_dict: dict,
    factor_names: list[str] = None,
):
    """
    Computes and logs the utilization and statistics of factors/clusters.
    This is a logging wrapper around compute_factor_stats.
    """
    if not factors:
        logger.warning("No factors provided to log_factor_utilization.")
        return

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    for name in factor_names:
        # Use .get() for safety, defaulting to np.nan
        total = stats_dict.get(f"{name}_total_components", 0)
        empty = stats_dict.get(f"{name}_empty_components", np.nan)
        mean_s = stats_dict.get(f"{name}_mean_strength", np.nan)
        q25_s = stats_dict.get(f"{name}_q25_strength", np.nan)
        q75_s = stats_dict.get(f"{name}_q75_strength", np.nan)

        logger.info(
            f"Factor '{name}': {empty}/{total} empty components. "
            f"Strength (Mean={mean_s:.4f}, Q1(25%)={q25_s:.4f}, Q3(75%)={q75_s:.4f})"
        )

    logger.info("--------------------------------------")


def log_item_membership_stats(stats_dict: dict, factor_names: list[str]):
    """
    Logs the summary statistics for per-item mean loadings.
    """
    logger.info("--- Item Membership 'Softness' Check ---")

    for name in factor_names:
        avg = stats_dict.get(f"{name}_item_mean_loading_avg", np.nan)
        q25 = stats_dict.get(f"{name}_item_mean_loading_q25", np.nan)
        q75 = stats_dict.get(f"{name}_item_mean_loading_q75", np.nan)

        logger.info(
            f"Factor '{name}' Item Mean Loading: "
            f"Avg={avg:.4f}, Q1(25%)={q25:.4f}, Q3(75%)={q75:.4f}"
        )
    logger.info("----------------------------------------")


def compute_cluster_stats(df: pd.DataFrame) -> pd.Series:
    """
    Computes statistics on the number of clusters assigned to each item.
    """
    # First, count how many clusters each item belongs to
    cluster_counts = (
        df.groupby(["factor_name", "item_name"])
        .size()
        .reset_index(name="num_clusters")
    )

    # Then calculate statistics by factor_name
    stats = (
        cluster_counts.groupby("factor_name")["num_clusters"]
        .agg(
            [
                "min",
                "mean",
                "max",
                lambda x: np.percentile(x, 75) - np.percentile(x, 25),  # IQR
            ]
        )
        .round(2)
    )

    # Rename the lambda column to IQR
    stats = stats.rename(columns={"<lambda_0>": "IQR"})
    return stats


def tune_ranks(
    method: str,
    df: pd.DataFrame,
    features: Union[str, List[str]],
    output_path: Path,
    rank_list: List[int] = None,
    store_ranks: List[int] = None,
    sku_ranks: List[int] = None,
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
        sku_ranks: (For 'tucker') List of ranks for mode 1
        feature_ranks: (For 'tucker') List of ranks for mode 2
        n_iter: Max iterations
        tol: Tolerance
    """
    results = []

    if method == "tucker":
        # Check for correct inputs
        if not (store_ranks and sku_ranks and feature_ranks):
            logger.error(
                "For 'tucker' method, you must provide store_ranks, sku_ranks, and feature_ranks."
            )
            return pd.DataFrame()

        # Generate all combinations
        rank_combinations = list(
            itertools.product(store_ranks, sku_ranks, feature_ranks)
        )
        logger.info(f"Rank combinations: {rank_combinations}")
        total_runs = len(rank_combinations)
        logger.info(
            f"--- Starting Tucker rank tuning. Testing {total_runs} combinations. ---"
        )

        for i, rank_tuple in enumerate(rank_combinations):
            logger.info(
                f"*** Testing Tucker combo {i+1}/{total_runs}: {rank_tuple} ***"
            )
            try:
                pve, rmse, stats_dict = fit_and_decompose(
                    method=method,
                    df=df,
                    features=features,
                    ranks=rank_tuple,
                    n_iter=n_iter,
                    tol=tol,
                )
                logger.info(f"PVE: {pve:.2f}%, RMSE: {rmse:.3f}")
                results.append(
                    {
                        "store_rank": rank_tuple[0],
                        "sku_rank": rank_tuple[1],
                        "feature_rank": rank_tuple[2],
                        "pve": pve,
                        "rmse": rmse,
                        **stats_dict,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Failed on rank {rank_tuple}: {e}", exc_info=True
                )
                results.append(
                    {
                        "store_rank": rank_tuple[0],
                        "sku_rank": rank_tuple[1],
                        "feature_rank": rank_tuple[2],
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
                pve, rmse, stats_dict = fit_and_decompose(
                    method=method,
                    df=df,
                    features=features,
                    ranks=rank,
                    n_iter=n_iter,
                    tol=tol,
                )
                results.append(
                    {
                        "rank": rank,
                        "pve": pve,
                        "rmse": rmse,
                        **stats_dict,
                    }
                )
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
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Log the best results
    best_pve = results_df.sort_values(by="pve", ascending=False)
    logger.info(f"Best results by PVE:\n{best_pve.head()}")

    best_rmse = results_df.sort_values(by="rmse", ascending=True)
    logger.info(f"Best results by RMSE:\n{best_rmse.head()}")
    date = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(date) / output_path
    save_csv_or_parquet(best_rmse, output_path)

    return results_df


def fit_and_decompose(
    method: str,
    df: pd.DataFrame,
    features: str | list[str],
    ranks: tuple[int, int, int] | int | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[float, float, dict]:
    """
    Builds, pre-processes, and decomposes the tensor, returning
    PVE, RMSE, and a dictionary of factor/core statistics.
    """
    # Parse features
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
        logger.info(f"Parsed features: {features}")

    # Build the RAW tensor (contains NaNs)
    X_raw, M_raw, _, _ = build_multifeature_X_matrix(df, features)
    logger.info(f"X shape:{X_raw.shape}")

    # Create tensors FIRST
    X_mat = tl.tensor(X_raw, device=device, dtype=tl.float32)
    M_tensor = tl.tensor(M_raw, device=device, dtype=torch.bool)

    # Conditionally pre-process the data
    if method in ["tucker", "parafac"]:
        # These models support signed data, so we z-score
        logger.info(
            f"Applying z-score (center_scale_signed) for '{method}' model."
        )
        # X is now the processed (z-scored, imputed) tensor
        X, mus, sds = center_scale_signed(X_mat, M_tensor)
    elif method == "ntf":
        # NTF requires non-negative data. We do NOT z-score.
        logger.info(f"Skipping z-score for non-negative model '{method}'.")
        # X is the raw tensor (with NaNs)
        X = X_mat
    else:
        raise ValueError(f"Invalid method: {method}")

    I, J, D = X_mat.shape
    logger.info(f"Multifeature mode: reshaping data: ({I}, {J}, {D})")

    # Use provided ranks or compute defaults
    if ranks is None:
        rank_tuple = (max(2, I // 4), max(2, J // 4), max(2, D // 4))
        logger.info(f"No ranks provided, using defaults: {rank_tuple}")
    else:
        rank_tuple = ranks

    # 4. Decompose using the correct tensor 'X'
    if method == "tucker":
        logger.info(f"Performing Tucker decomposition with rank={rank_tuple}")
        weights, factors = tucker_decomposition(X, rank_tuple, n_iter, tol)
    elif method == "ntf":
        logger.info(f"Performing NTF decomposition with rank={rank_tuple}")
        # 'nonneg_parafac' will receive the RAW X and impute NaNs
        weights, factors = nonneg_parafac(X, rank_tuple)
    elif method == "parafac":
        logger.info(f"Performing PARAFAC decomposition with rank={rank_tuple}")
        # 'parafac_decomposition' will receive the Z-SCORED X
        weights, factors = parafac_decomposition(X, rank_tuple)
    else:
        raise ValueError(f"Invalid method: {method}")

    # 5. Log factor/core statistics
    stats_dict = {}  # Initialize an empty dict

    if factors:
        mode_names = ["Store", "SKU", "Feature"]

        if method == "tucker":
            # 1. Compute stats from the CORE (which is 'weights')
            core_stats = compute_tucker_core_stats(weights)
            log_tucker_core_stats(core_stats)
            stats_dict.update(core_stats)  # Add core stats to dict
        else:
            # For 'parafac' and 'ntf', compute component strength
            comp_stats = compute_factor_stats(factors, factor_names=mode_names)
            log_factor_utilization(
                factors, comp_stats, factor_names=mode_names
            )
            stats_dict.update(comp_stats)  # Add component stats to dict

        # 2. Compute ITEM MEMBERSHIP stats (for ALL methods)
        item_stats = compute_item_membership_stats(
            factors, factor_names=mode_names
        )
        log_item_membership_stats(item_stats, factor_names=mode_names)
        stats_dict.update(item_stats)

    else:
        logger.error(
            f"Decomposition failed for method {method}, skipping errors."
        )
        return np.nan, np.nan, stats_dict

    # 6. Calculate PVE/RMSE using the correct tensor 'X'
    pve_percent, rmse = errors(X, weights, factors, method=method)
    logger.info(f"PVE: {pve_percent:.2f}%, RMSE: {rmse:.3f}")

    # 7. Return all three items
    return pve_percent, rmse, stats_dict


def fit(
    method: str,
    df: pd.DataFrame,
    features: str | list[str],
    ranks: tuple[int, int, int] | int | None = None,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[
    tl.tensor,
    list[tl.tensor],
    list,
    list,
    list,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Fits a single, final model and returns the model components
    (weights, factors) and metadata (names).
    """
    # Parse features
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]
        logger.info(f"Parsed features: {features}")

    # Build the RAW tensor (contains NaNs)
    X_raw, M_raw, row_names, col_names = build_multifeature_X_matrix(
        df, features
    )
    logger.info(f"X shape:{X_raw.shape}")
    logger.info(
        f"Got {len(row_names)} row names (Stores) and {len(col_names)} col names (SKUs)"
    )

    # Create tensors FIRST
    X_mat = tl.tensor(X_raw, device=device, dtype=tl.float32)
    M_tensor = tl.tensor(M_raw, device=device, dtype=torch.bool)

    # Conditionally pre-process the data
    I, J, D = X_mat.shape
    mus = tl.zeros(D, device=X_mat.device, dtype=X_mat.dtype)
    sds = tl.ones(D, device=X_mat.device, dtype=X_mat.dtype)
    if method in ["tucker", "parafac"]:
        logger.info(
            f"Applying z-score (center_scale_signed) for '{method}' model."
        )
        X, mus, sds = center_scale_signed(X_mat, M_tensor)
    elif method == "ntf":
        logger.info(f"Skipping z-score for non-negative model '{method}'.")
        X = X_mat
    else:
        raise ValueError(f"Invalid method: {method}")

    logger.info(f"Multifeature mode: reshaping data: ({I}, {J}, {D})")

    if ranks is None:
        rank_tuple = (max(2, I // 4), max(2, J // 4), max(2, D // 4))
        logger.info(f"No ranks provided, using defaults: {rank_tuple}")
    else:
        rank_tuple = ranks

    # Decompose using the correct tensor 'X'
    if method == "tucker":
        logger.info(f"Performing Tucker decomposition with rank={rank_tuple}")
        weights, factors = tucker_decomposition(X, rank_tuple, n_iter, tol)
    elif method == "ntf":
        logger.info(f"Performing NTF decomposition with rank={rank_tuple}")
        weights, factors = nonneg_parafac(X, rank_tuple)
    elif method == "parafac":
        logger.info(f"Performing PARAFAC decomposition with rank={rank_tuple}")
        weights, factors = parafac_decomposition(X, rank_tuple)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Log stats (but don't return them)
    if factors:
        mode_names = ["Store", "SKU", "Feature"]
        if method == "tucker":
            core_stats = compute_tucker_core_stats(weights)
            log_tucker_core_stats(core_stats)
        else:
            comp_stats = compute_factor_stats(factors, factor_names=mode_names)
            log_factor_utilization(
                factors, comp_stats, factor_names=mode_names
            )

        item_stats = compute_item_membership_stats(
            factors, factor_names=mode_names
        )
        log_item_membership_stats(item_stats, factor_names=mode_names)
    else:
        logger.error(
            f"Decomposition failed for method {method}, skipping errors."
        )
        return None, None, None, None, None, None, None, None, None

    # Calculate and log PVE/RMSE
    pve_percent, rmse = errors(X, weights, factors, method=method)
    logger.info(f"FINAL MODEL PVE: {pve_percent:.2f}%, RMSE: {rmse:.3f}")

    # Return the model components and names
    return (
        weights,
        factors,
        row_names,
        col_names,
        features,
        X_raw,
        M_raw,
        mus,
        sds,
    )


def compute_reconstruction_metrics(
    X: np.ndarray,
    weights: tl.tensor,
    factors: List[tl.tensor],
    method: str,
    mus: tl.tensor,
    sds: tl.tensor,
    *,
    M: np.ndarray = None,
    max_scatter_points: int = 200_000,
) -> Dict[str, Any]:
    """
    Computes reconstruction metrics after scaling back to the original data space.

    Args:
        X: The *original* data tensor (numpy, float, with NaNs) (I,J,D)
        weights: The core tensor (Tucker) or weights vector (CP/NTF)
        factors: List of factor matrices
        method: 'tucker', 'parafac', or 'ntf'
        mus: The 1D tensor of means (length D) from scaling
        sds: The 1D tensor of std devs (length D) from scaling
        M: (I,J) boolean mask of observed entries
        max_scatter_points: Max points for scatter plot data

    Returns:
        A dictionary containing all computed metrics for plotting.
    """
    tl.set_backend("pytorch")
    logger.info(f"Computing reconstruction metrics for method '{method}'...")
    X = np.asarray(X)
    assert X.ndim == 3, f"Expected (I,J,D), got {X.shape}"
    I, J, D = X.shape

    # Reconstruct the Z-Scored Tensor ---
    logger.info("Reconstructing z-scored tensor...")
    try:
        if method == "tucker":
            X_rec_zscored_torch = tl.tucker_to_tensor((weights, factors))
        elif method in ["parafac", "ntf"]:
            X_rec_zscored_torch = tl.cp_to_tensor((weights, factors))
        else:
            raise ValueError(f"Unknown method for reconstruction: {method}")
        X_rec_zscored = X_rec_zscored_torch.cpu().numpy()
    except Exception as e:
        logger.error(f"!!! Reconstruction failed: {e}")
        X_rec_zscored = np.full(X.shape, np.nan)

    if X_rec_zscored.shape != X.shape:
        raise ValueError(
            f"X_rec_zscored shape {X_rec_zscored.shape} != X shape {X.shape}"
        )

    # Scale Back to Original Data Space
    logger.info("Scaling reconstruction back to original data space...")
    mus_np = mus.cpu().numpy()  # (D,)
    sds_np = sds.cpu().numpy()  # (D,)
    # Use broadcasting: (I, J, D) = (I, J, D) * (D,) + (D,)
    X_rec_unscaled = (X_rec_zscored * sds_np) + mus_np

    # Build Masks
    if M is not None:
        assert M.shape == (I, J), f"M must be (I,J), got {M.shape}"
        Md = np.repeat(M[:, :, None], D, axis=2)  # (I,J,D)
    else:
        Md = ~np.isnan(X)

    # Get Data for Scatter Plot
    x_flat = X[Md]
    # IMPORTANT: Use the unscaled reconstruction here
    xr_flat = X_rec_unscaled[Md]

    n = x_flat.size
    if n == 0:
        logger.warning("Warning: No observed data points found in mask M.")
        x_plot, xr_plot = np.array([]), np.array([])
    elif n > max_scatter_points:
        idx = np.random.RandomState(0).choice(
            n, size=max_scatter_points, replace=False
        )
        x_plot, xr_plot = x_flat[idx], xr_flat[idx]
    else:
        x_plot, xr_plot = x_flat, xr_flat

    # Per-feature Metrics (PVE, RMSE)
    n_obs_per = np.zeros(D, dtype=int)
    rss_per = np.full(D, np.nan)
    rmse_per = np.full(D, np.nan)
    pve_per = np.full(D, np.nan)

    for d in range(D):
        md = Md[:, :, d]
        n_obs = int(md.sum())
        n_obs_per[d] = n_obs
        if n_obs == 0:
            logger.warning(f"Warning: No observed data for feature {d}")
            continue

        xd = X[:, :, d][md]
        # IMPORTANT: Use the unscaled reconstruction here
        xhd = X_rec_unscaled[:, :, d][md]

        resid = xd - xhd
        rss = float(np.sum(resid * resid))
        rss_per[d] = rss
        rmse_per[d] = float(np.sqrt(rss / n_obs))
        mu = float(xd.mean())
        tss = float(np.sum((xd - mu) ** 2))
        pve_per[d] = (
            (1.0 - rss / max(tss, 1e-12)) * 100.0 if tss > 0 else np.nan
        )

    # Get Rank String
    try:
        if method == "tucker":
            rank_str = f"({factors[0].shape[1]}, {factors[1].shape[1]}, {factors[2].shape[1]})"
        else:
            rank_str = f"({factors[0].shape[1]})"
    except Exception:
        rank_str = "(unknown rank)"

    # Package results
    metrics = {
        "x_plot": x_plot,
        "xr_plot": xr_plot,
        "n_obs_per": n_obs_per,
        "rss_per": rss_per,
        "rmse_per": rmse_per,
        "pve_per": pve_per,
        "D": D,
        "rank_str": rank_str,
    }
    return metrics


def get_top_k_assignments(
    factors: list[tl.tensor], factor_names: list[str] = None, k: int = 3
) -> dict:
    """
    Implements the "Fixed-K" approach.
    Converts soft factor loadings into "Top-K" cluster assignments
    based on the magnitude of the loadings.
    """
    assignments = {}
    if not factors:
        return assignments

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    logger.info(f"--- Generating Top-K (k={k}) Assignments ---")

    for i, F in enumerate(factors):
        name = factor_names[i]
        factor_assignments = []

        # Check if k is larger than the number of components
        num_components = F.shape[1]
        if k > num_components:
            k_actual = num_components
            logger.warning(
                f"k={k} is larger than num_components={num_components} for '{name}'. Using k={k_actual}."
            )
        else:
            k_actual = k

        # Iterate over each item (row) in the factor matrix
        for item_loadings in F:
            abs_loadings = torch.abs(item_loadings)
            top_k = torch.topk(abs_loadings, k=k_actual)
            top_k_indices = top_k.indices.cpu().numpy().tolist()
            factor_assignments.append(top_k_indices)

        assignments[name] = factor_assignments

    logger.info("------------------------------------------")
    return assignments


def get_threshold_k_assignments(
    factors: list[tl.tensor],
    factor_names: list[str] = None,
    threshold: float = 0.9,
) -> dict:
    """
    Implements the "Data-Driven" approach.
    Assigns a variable number of clusters to each item based on a
    cumulative strength threshold (e.g., 0.9 = 90% of identity).
    """
    assignments = {}
    if not factors:
        return assignments

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    logger.info(
        f"--- Generating Threshold-K (threshold={threshold*100}%) Assignments ---"
    )

    for i, F in enumerate(factors):
        name = factor_names[i]
        factor_assignments = []

        # Iterate over each item (row)
        for item_loadings in F:
            abs_loadings = torch.abs(item_loadings)
            total_identity = torch.sum(abs_loadings)

            # Handle case where all loadings are zero
            if total_identity <= 1e-8:
                factor_assignments.append([])
                continue

            # Sort loadings and indices to find the strongest ones
            sorted_loadings, sorted_indices = torch.sort(
                abs_loadings, descending=True
            )

            # Get cumulative sum
            cum_sum = torch.cumsum(sorted_loadings, dim=0)

            # Find the point where cum_sum exceeds the threshold
            threshold_val = total_identity * threshold

            # Find all indices that meet or exceed the threshold
            indices_meeting_threshold = torch.where(cum_sum >= threshold_val)[
                0
            ]

            if len(indices_meeting_threshold) == 0:
                # Safeguard: if no single loading meets it (e.g. threshold > 1.0)
                # or if all loadings are zero (already handled), just take the top-1.
                cutoff_index = 0
            else:
                # Get the *first* index that crosses the bar
                cutoff_index = indices_meeting_threshold[0].item()

            # Get the cluster indices (from the original list) up to the cutoff
            final_indices = (
                sorted_indices[: cutoff_index + 1].cpu().numpy().tolist()
            )
            factor_assignments.append(final_indices)

        assignments[name] = factor_assignments

    logger.info("--------------------------------------------------")
    return assignments


def get_hard_assignments(
    factors: list[tl.tensor],
    factor_names: list[str] = None,
) -> dict:
    """
    Implements a "hard" or "argmax" assignment.
    Assigns each item to the single factor it has the highest loading for.
    """
    assignments = {}
    if not factors:
        return assignments

    if factor_names is None:
        factor_names = [f"Mode {i}" for i in range(len(factors))]

    logger.info("--- Generating Hard (Argmax) Assignments ---")

    for i, F in enumerate(factors):
        name = factor_names[i]
        factor_assignments = []

        # Iterate over each item (row)
        for item_loadings in F:

            # --- START MODIFICATION ---

            # Find the index of the single factor with the max absolute loading
            if (
                len(item_loadings) == 0
                or torch.sum(torch.abs(item_loadings)) <= 1e-8
            ):
                final_indices = []  # Or assign to a default, e.g., [0]
            else:
                top_index = torch.argmax(torch.abs(item_loadings)).item()
                final_indices = [top_index]  # Assign to *only* that one factor

            # --- END MODIFICATION ---

            factor_assignments.append(final_indices)

        assignments[name] = factor_assignments

    logger.info("--------------------------------------------------")
    return assignments


def create_assignments(
    assignments: dict,
    name_map: dict,  # Map factor names to their corresponding item name lists.
) -> pd.DataFrame:
    """
    Saves the cluster assignments as a flat DataFrame (wide-format)
    to a CSV or Parquet file.

    Each row represents a single item-to-cluster assignment.
    """
    all_rows = []  # This will hold the flat data

    for _, data in assignments.items():
        # factor_name is e.g., "Store"
        for factor_name, cluster_lists in data.items():
            item_names = name_map.get(factor_name, [])

            # Zip item names (e.g., "Store_1") with their assignments (e.g., [1, 5, 30])
            for i, cluster_list in enumerate(cluster_lists):
                item_name = (
                    item_names[i] if i < len(item_names) else f"index_{i}"
                )

                # If an item has no clusters (e.g., from thresholding)
                if not cluster_list:
                    all_rows.append(
                        {
                            "factor_name": factor_name,
                            "item_name": str(item_name),
                            "cluster_id": np.nan,  # Use NaN for "no cluster"
                        }
                    )
                else:
                    # "Explode" the list: [1, 5, 30] becomes 3 rows
                    for cluster_id in cluster_list:
                        all_rows.append(
                            {
                                "factor_name": factor_name,
                                "item_name": str(item_name),
                                "cluster_id": cluster_id,
                            }
                        )

    # Convert the list of flat dicts into a DataFrame
    df = pd.DataFrame(all_rows)
    return df


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

        # Check for NaN (from _nanstd) or a value too small
        if torch.isnan(sd) or sd <= eps or sd == 0:
            sd = 1.0
        X[..., d] = (X[..., d] - mu) / sd
        mus[d] = mu
        sds[d] = sd

    # Replace ALL NaNs (from original data or from scaling) with 0.0
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, mus, sds


def errors(
    X: tl.tensor, weights: tl.tensor, factors: list[tl.tensor], method: str
) -> Tuple[float, float]:
    """Calculates PVE using tensorly (backend-agnostic) functions."""
    if method == "tucker":
        # 'weights' is actually the core tensor in this case
        X_hat = tl.tucker_to_tensor((weights, factors))
    else:
        # 'parafac' and 'ntf' use this
        X_hat = tl.cp_to_tensor((weights, factors))

    # 5. Calculate SSE and RMSE (on device)
    finite_mask = torch.isfinite(X)
    X_finite_orig = X[finite_mask]
    if not tl.any(finite_mask):
        return np.nan, np.nan

    X_finite_hat = X_hat[finite_mask]
    sse_vec = X_finite_orig - X_finite_hat
    sse = tl.dot(sse_vec, sse_vec)  # dot uses the backend
    num_finite = X_finite_orig.shape[0]
    rmse = tl.sqrt(sse / num_finite)  # tl.sqrt uses the backend

    # Calculate PVE
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
        X,
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


def calculate_cluster_profiles(
    model_dict: Dict[str, Any],
    mask: str,
    max_clusters: int = 5,  # Reduced default to 5
    selection_method: Literal[
        "size", "variance", "random", "hierarchical"
    ] = "size",
) -> Optional[pd.DataFrame]:
    """
    Extract and calculate mean feature profiles for clusters.
    """
    try:
        X_raw = np.copy(model_dict["X_raw"])
        M_raw = model_dict["M_raw"]
        feature_names = model_dict["feature_names"]
        assignments_df = model_dict["assignments"]

        # Replace masked values with NaN
        X_raw[M_raw == 0] = np.nan

        # Average features based on the mask
        if mask == "Store":
            avg_features = np.nanmean(X_raw, axis=1)
            item_names = model_dict["row_names"]
        elif mask == "SKU":
            avg_features = np.nanmean(X_raw, axis=0)
            item_names = model_dict["col_names"]
        else:
            logger.error(f"Unknown mask '{mask}'. Use 'Store' or 'SKU'.")
            return None

        # Create feature DataFrame
        feature_df = pd.DataFrame(avg_features, columns=feature_names)
        feature_df["item_name"] = item_names

        # Filter and merge assignments
        assignments_filtered = assignments_df.query("factor_name == @mask")
        merged_df = pd.merge(feature_df, assignments_filtered, on="item_name")

        if merged_df.empty:
            logger.error("Merge failed. No matching 'item_name' found.")
            return None

        # Calculate cluster means and sizes
        cluster_means = merged_df.groupby("cluster_id")[feature_names].mean()
        cluster_sizes = merged_df.groupby("cluster_id").size()
        cluster_means["cluster_size"] = cluster_sizes

        logger.info(f"Total clusters found: {len(cluster_means)}")

        # FORCE limit clusters if we have too many
        if len(cluster_means) > max_clusters:
            if selection_method == "size":
                # Select clusters with the most members
                top_clusters = cluster_sizes.nlargest(max_clusters).index
            elif selection_method == "variance":
                # Select clusters with highest variance
                cluster_variance = cluster_means[feature_names].var(axis=1)
                top_clusters = cluster_variance.nlargest(max_clusters).index
            elif selection_method == "hierarchical":
                # Group similar clusters using hierarchical clustering
                # Standardize the cluster profiles
                scaler = StandardScaler()
                profiles_scaled = scaler.fit_transform(
                    cluster_means[feature_names]
                )

                # Apply hierarchical clustering to group similar clusters
                hierarchical = AgglomerativeClustering(n_clusters=max_clusters)
                meta_clusters = hierarchical.fit_predict(profiles_scaled)

                # Select one representative from each meta-cluster (largest cluster)
                top_clusters = []
                for meta_id in range(max_clusters):
                    clusters_in_meta = cluster_means.index[
                        meta_clusters == meta_id
                    ]
                    # Pick the largest cluster from this meta-cluster
                    largest_in_meta = cluster_sizes.loc[
                        clusters_in_meta
                    ].idxmax()
                    top_clusters.append(largest_in_meta)

                top_clusters = pd.Index(top_clusters)
            else:  # random
                top_clusters = cluster_means.sample(n=max_clusters).index

            cluster_means = cluster_means.loc[top_clusters]
            logger.info(
                f"Reduced to {len(cluster_means)} clusters using {selection_method} method"
            )

        return cluster_means

    except Exception as e:
        logger.error(f"Error calculating cluster profiles: {e}")
        return None
