import numpy as np
from typing import Dict, Any, Iterable, Optional, Callable, Tuple
import pandas as pd
from sklearn.cluster import KMeans
from src.BinaryTriFactorizationEstimator import (
    BinaryTriFactorizationEstimator,
    model_loss,
    gaussian_loss,
    poisson_nll,
)
from src.data_utils import normalize_data
from src.utils import save_csv_or_parquet
from dataclasses import dataclass
from src.utils import get_logger
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

logger = get_logger(__name__)


def get_normalized_assignments(
    block_id_mat: np.ndarray,
    norm_data: pd.DataFrame,
    *,
    row_col: str = "store",
    col_col: str = "item",
    value_col: str = "growth_rate_1",
    drop_unassigned: bool = True,  # set True to remove block_id == -1
) -> pd.DataFrame:

    I, J = norm_data.shape
    assert block_id_mat.shape == (
        I,
        J,
    ), "block_id matrix must match norm_data shape"

    # Build a DF aligned to norm_data’s index/columns
    blk_df = pd.DataFrame(
        block_id_mat, index=norm_data.index, columns=norm_data.columns
    )

    # Long (row,item,block_id)
    blk_long = (
        blk_df.stack()
        .rename("block_id")
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # Long values from the SAME matrix used for clustering
    val_long = (
        norm_data.stack()
        .rename(value_col)
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    df = val_long.merge(blk_long, on=[row_col, col_col], how="left")

    # Optional cleanup
    if drop_unassigned:
        df = df[df["block_id"] >= 0]

    df["block_id"] = df["block_id"].astype("Int64")  # pandas nullable int

    return df


def merge_assignments(
    df: pd.DataFrame,
    assign: Dict[str, Any],
    norm_data: pd.DataFrame,
    *,
    row_col: str = "store",
    col_col: str = "item",
    value_col: str = "growth_rate_1",
) -> pd.DataFrame:

    block_id_mat = assign["block_id"]

    # long (store,item,block_id)
    blk_long = (
        pd.DataFrame(
            block_id_mat, index=norm_data.index, columns=norm_data.columns
        )
        .stack()
        .rename("block_id")
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # long values from the SAME matrix used for clustering
    val_long = (
        norm_data.stack()
        .rename(value_col)
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # plotting table: store, item, value, block_id (no date column)
    val_long = val_long.merge(blk_long, on=[row_col, col_col], how="left")

    # attach per-(store,item) block_id
    val_long = val_long[["store", "item", "block_id"]].drop_duplicates()

    df = df.merge(val_long, on=["store", "item"], how="left", validate="m:1")
    return df


def summarize_blocks_for_merge(
    est: BinaryTriFactorizationEstimator,
    X: np.ndarray,
    assign: Dict[str, Any],
) -> pd.DataFrame:
    """Per-block stats from the current assignment."""
    R, C = est.B_.shape
    bid = assign["block_id"]
    used = np.unique(bid)
    used = used[used >= 0]
    rows = []
    for b in used:
        mask = bid == b
        vals = X[mask]
        r, c = int(b // C), int(b % C)
        rows.append(
            {
                "block_id": int(b),
                "r": r,
                "c": c,
                "B_rc": float(est.B_[r, c]),
                "n_cells": int(mask.sum()),
                "coverage_%": 100.0 * mask.mean(),
                "mean": float(vals.mean()) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)


def explain_coverage(
    est: BinaryTriFactorizationEstimator,
    assign: Dict[str, Any],
) -> pd.DataFrame:

    # per-block coverage as a matrix
    cov = pd.DataFrame(
        [
            [
                (assign["block_id"] == (r * est.B_.shape[1] + c)).sum()
                for c in range(est.B_.shape[1])
            ]
            for r in range(est.B_.shape[0])
        ],
        index=[f"r{r}" for r in range(est.B_.shape[0])],
        columns=[f"c{c}" for c in range(est.B_.shape[1])],
    )
    cov = cov.style.background_gradient(cmap="Blues", axis=None)
    return cov


def merge_blocks_by_stat(
    assign: Dict[str, Any],
    df: pd.DataFrame,
    norm_data: pd.DataFrame,
    est: BinaryTriFactorizationEstimator,
    *,
    stat: str = "mean",
    scheme: str = "sign",
    k: int = 2,
) -> pd.DataFrame:
    """
    Merge active blocks using a single statistic of their cells.

    stat:   "mean" or "median"   (drives grouping)
    scheme: "sign" | "kmeans"
      - "sign": positives vs negatives (by chosen stat)
      - "kmeans": k-means in 1D on the chosen stat (k groups)
    """
    # Handle both DataFrame and NumPy array cases
    if hasattr(norm_data, "to_numpy"):
        X_array = norm_data.to_numpy()
    else:
        X_array = norm_data
    summ = summarize_blocks_for_merge(est, X_array, assign)
    stat_col = "mean" if stat == "mean" else "median"
    s = summ[stat_col].to_numpy()

    if scheme == "sign":
        # 0 = nonnegative, 1 = negative → two groups
        merged = (s < 0).astype(int)
        # order: group 0 (>=0) first, group 1 (<0) next
        if stat == "mean":
            order = np.argsort(
                [
                    s[merged == g].mean() if (merged == g).any() else np.inf
                    for g in [0, 1]
                ]
            )
        else:
            medians = [
                np.median(s[merged == g]) if (merged == g).any() else np.inf
                for g in [0, 1]
            ]
            order = np.argsort(np.array(medians))
        remap = {int(g_old): int(g_new) for g_new, g_old in enumerate(order)}
        summ["merged_id"] = [remap[int(g)] for g in merged]

    elif scheme == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labs = km.fit_predict(s.reshape(-1, 1))
        # reorder clusters by mean stat so ids are consistent
        if stat == "mean":
            stat_df = pd.Series(s).groupby(labs).mean().sort_values()
        else:
            stat_df = pd.Series(s).groupby(labs).median().sort_values()
        remap = {int(old): int(new) for new, old in enumerate(stat_df.index)}
        summ["merged_id"] = [remap[int(g)] for g in labs]

    else:
        raise ValueError("scheme must be 'sign' or 'kmeans'")

    # map original block_id -> merged_id
    mapping = dict(zip(summ["block_id"].tolist(), summ["merged_id"].tolist()))

    # build merged block-id matrix
    bid = assign["block_id"].copy()
    bid_merged = np.full_like(bid, fill_value=-1, dtype=int)
    # vectorized map
    flat = bid.ravel()
    out = np.array([mapping.get(int(b), -1) for b in flat], dtype=int)
    bid_merged = out.reshape(bid.shape)

    # Use bid_merged
    df_assign = (
        pd.DataFrame(
            bid_merged, index=norm_data.index, columns=norm_data.columns
        )
        .stack()
        .rename("merged_block_id")
        .reset_index()
        .rename(columns={"level_0": "store", "level_1": "item"})
    )

    # attach to your training df
    df = df.drop(columns=["block_id"], errors="ignore").merge(
        df_assign[["store", "item", "merged_block_id"]],
        on=["store", "item"],
        how="left",
        validate="m:1",
    )
    df.rename(columns={"merged_block_id": "block_id"}, inplace=True)
    return df


def _mode_min(s: pd.Series) -> int:
    m = s.mode()
    return int(m.min()) if not m.empty else -1


def get_sorted_row_col(
    df: pd.DataFrame,
    *,
    row_col: str = "store",
    col_col: str = "item",
    block_col: str = "block_id",
) -> tuple[list, list]:
    """
    Return row_order, col_order for the heatmap.

    Strategy:
      1) Primary: order rows/cols by representative block (mode_min).
      2) If no variation (all same mode), fallback to lexicographic sort
         of the entire block_id pattern (row-wise / column-wise).
      3) Within each step, use deterministic tie-breaks (min, mean, then full pattern).
    """
    # pivot matrix of block ids
    blk = df.pivot(index=row_col, columns=col_col, values=block_col)

    # Fill missing with a sentinel that sorts first
    blk = blk.fillna(-1).astype(int)

    # representative per row/col
    row_rep = blk.apply(_mode_min, axis=1)
    col_rep = blk.apply(_mode_min, axis=0)

    # quick tie-break stats
    row_min = blk.min(axis=1)
    row_mean = blk.mean(axis=1)
    col_min = blk.min(axis=0)
    col_mean = blk.mean(axis=0)

    # Decide if mode has enough variation
    row_rep_varies = row_rep.nunique() > 1
    col_rep_varies = col_rep.nunique() > 1

    # ----- Row order -----
    if row_rep_varies:
        # sort by representative, then min, then mean, then lexicographic row pattern
        row_order = (
            blk.assign(__rep=row_rep, __min=row_min, __mean=row_mean)
            .sort_values(
                by=["__rep", "__min", "__mean", *list(blk.columns)],
                kind="mergesort",  # stable
            )
            .index.tolist()
        )
    else:
        # fallback: pure lexicographic by full row pattern
        row_order = blk.sort_values(
            by=list(blk.columns), kind="mergesort"
        ).index.tolist()

    # ----- Column order -----
    if col_rep_varies:
        col_df = blk.T.assign(__rep=col_rep, __min=col_min, __mean=col_mean)
        col_order = col_df.sort_values(
            by=["__rep", "__min", "__mean", *list(col_df.columns)],
            kind="mergesort",
        ).index.tolist()
    else:
        # fallback: pure lexicographic by full column pattern
        col_order = blk.T.sort_values(
            by=list(blk.index), kind="mergesort"
        ).index.tolist()

    return row_order, col_order


# ----------------------------
# PVE (percent variance explained)
# ----------------------------
def compute_pve(
    X: np.ndarray, Xhat: np.ndarray, loss_name="gaussian", mask=None, eps=1e-12
) -> float:
    """
    PVE = 1 - Loss(X, Xhat) / Loss(X, baseline)
    baseline:
      - gaussian: constant global mean of X over mask
      - poisson : constant mean-rate μ = mean(X over mask)
    """
    if mask is None:
        obs = X
    else:
        obs = X[mask]

    if loss_name == "gaussian":
        baseline_val = float(np.mean(obs))
        baseline = np.full_like(X, baseline_val, dtype=float)
    else:  # poisson
        mu = float(np.mean(obs))
        baseline = np.full_like(X, mu, dtype=float)

    L_model = model_loss(X, Xhat, loss_name, mask)
    L_base = model_loss(X, baseline, loss_name, mask)
    denom = max(L_base, eps)
    return 1.0 - (L_model / denom)


# ----------------------------
# Fast per-block ablation ΔLoss
# ----------------------------


def ablate_block_delta_loss(
    est: BinaryTriFactorizationEstimator,
    X: np.ndarray,
    r: int,
    c: int,
    B: np.ndarray,
    mask: np.ndarray = None,
):
    """
    ΔLoss_rc = Loss(X, Xhat_without_rc) - Loss(X, Xhat_full)
    where removing block (r,c) is a rank-1 update:
       Xhat_without = Xhat - B_rc * (U[:,r] ⊗ V[:,c])
    Positive Δ means the block is helpful.
    """
    U, _, V = est.factors()
    Xhat = est.reconstruct()
    # U, V, B, Xhat = est.U_, est.V_, Bview, est.Xhat_

    b_rc = float(B[r, c])
    if b_rc == 0.0:
        return 0.0
    loss_name = est.loss
    L_full = model_loss(X, Xhat, loss_name, mask)
    # rank-1 contribution with current U/V
    outer = np.outer(U[:, r].astype(float), V[:, c].astype(float))
    Xhat_wo = Xhat - b_rc * outer
    L_wo = model_loss(X, Xhat_wo, loss_name, mask)
    return L_wo - L_full


# ----------------------------
# WCV / BCV / silhouette-like
# ----------------------------
def compute_block_wcv_bcv_silhouette(
    X: np.ndarray,
    assign_dict: Dict[str, Any],
    R: int,
    C: int,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Using per-cell assignments from assign_unique_blocks(...):
      - assign_dict["r_star"], assign_dict["c_star"] with -1 for unassigned.

    Returns per-block rows with columns:
      ["r", "c", "n", "mean", "WCV", "BCV_nearest", "silhouette_like"].

    Notes:
      - WCV: mean squared deviation within block (r,c).
      - BCV_nearest: min squared distance between this block mean and any other defined block's mean.
      - silhouette_like in [-1, 1]: (BCV - WCV) / max(BCV, WCV).
    """
    r_star = np.asarray(assign_dict["r_star"])
    c_star = np.asarray(assign_dict["c_star"])
    I, J = r_star.shape
    assert X.shape == (I, J), "X shape must match r_star/c_star shapes"

    # observed & assigned mask
    if mask is None:
        obs_mask = np.ones((I, J), dtype=bool)
    else:
        obs_mask = np.asarray(mask, dtype=bool)
        assert obs_mask.shape == (I, J), "mask must match X shape"

    assigned = (r_star >= 0) & (c_star >= 0)
    obs_mask = obs_mask & assigned

    block_means = np.full((R, C), np.nan, dtype=float)
    block_ns = np.zeros((R, C), dtype=int)

    # ---- pass 1: means & counts
    for r in range(R):
        for c in range(C):
            sel = (r_star == r) & (c_star == c) & obs_mask
            n = int(sel.sum())
            block_ns[r, c] = n
            if n > 0:
                block_means[r, c] = float(X[sel].mean())

    # ---- pass 2: WCV, BCV_nearest, silhouette-like
    rows = []
    # Precompute list of defined block means for BCV lookup
    defined_idx = np.argwhere(~np.isnan(block_means))
    defined_vals = block_means[~np.isnan(block_means)]

    for r in range(R):
        for c in range(C):
            n = block_ns[r, c]
            if n == 0:
                rows.append((r, c, 0, np.nan, np.nan, np.nan, np.nan))
                continue

            sel = (r_star == r) & (c_star == c) & obs_mask
            vals = X[sel].astype(float)
            mu = block_means[r, c]

            # WCV: mean squared deviation within block
            wcv = float(np.mean((vals - mu) ** 2))

            # BCV_nearest: min squared distance to any other defined block mean
            if defined_vals.size <= 1:
                bcv = np.nan
            else:
                # exclude this block's own mean by masking by coordinates
                # (safer than comparing floats)
                mask_self = ~(
                    (defined_idx[:, 0] == r) & (defined_idx[:, 1] == c)
                )
                others = defined_vals[mask_self]
                if others.size == 0:
                    bcv = np.nan
                else:
                    diffs2 = (others - mu) ** 2
                    bcv = float(diffs2.min())

            # silhouette-like
            if np.isnan(bcv):
                sil = np.nan
            else:
                denom = max(bcv, wcv)
                sil = (bcv - wcv) / denom if denom > 0 else 0.0

            rows.append((r, c, n, mu, wcv, bcv, sil))

    return pd.DataFrame(
        rows,
        columns=[
            "r",
            "c",
            "n",
            "mean",
            "WCV",
            "BCV_nearest",
            "silhouette_like",
        ],
    )


# ----------------------------
# Sparsity & coverage helpers
# ----------------------------
def sparsity_of_B(est: BinaryTriFactorizationEstimator, tol=1e-6):
    _, _, B = est.factors()
    return float(np.mean(np.abs(B) < tol))  # fraction near-zero


def coverage_from_assign(assign_dict: Dict[str, Any]):
    r_star = assign_dict["r_star"]
    return float(np.mean(r_star >= 0))  # fraction of cells assigned


def _baseline_array(X: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    Constant-mean baseline:
      - Gaussian: baseline = mean(X[mask])
    """
    obs = X if mask is None else X[mask]

    # Handle empty observations
    if obs.size == 0:
        logger.warning(
            "Empty observations for baseline calculation, using 0.0"
        )
        mu = 0.0
    else:
        mu = float(np.mean(obs))
        # Handle NaN/inf results
        if not np.isfinite(mu):
            logger.warning(f"Non-finite baseline mean: {mu}, using 0.0")
            mu = 0.0

    return np.full_like(X, mu, dtype=float)


# ---------- AIC/BIC helpers ----------
def _obs_count(mask: np.ndarray | None, X: np.ndarray) -> int:
    """Number of observed cells; if no mask, count all cells in X."""
    if mask is None:
        return int(X.size)
    mask = mask.astype(bool)
    if mask.shape != X.shape:
        raise ValueError(
            f"mask shape {mask.shape} must match X shape {X.shape}"
        )
    return int(np.sum(mask))


def neg_loglik_from_loss(
    loss_name: str, X: np.ndarray, Xhat: np.ndarray, mask=None
) -> float:
    """
    Return a quantity proportional to negative log-likelihood.
    (Exact constants cancel in AIC/BIC comparisons across same data.)
    """
    if loss_name == "gaussian":
        return gaussian_loss(
            X, Xhat, mask=mask
        )  # SSE ~ 2*negLL up to sigma^2 scale
    elif loss_name == "poisson":
        return poisson_nll(X, Xhat, mask=mask)  # exact NLL (up to const)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def aic_bic_scores(
    loss_name: str,
    X: np.ndarray,
    Xhat: np.ndarray,
    R: int,
    C: int,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    # Effective number of free parameters in the model
    # Here, we only count entries in B (R×C matrix).
    # U and V are binary membership matrices, not treated as free parameters here.
    k = int(R * C)

    # Number of observed data points (cells in X).
    # If a mask is provided, only those entries count as "observed".
    N = _obs_count(mask, X)

    # Compute negative log-likelihood (NLL) of the fitted model
    # - Gaussian loss → SSE-based surrogate
    # - Poisson loss  → exact Poisson NLL
    nll = neg_loglik_from_loss(loss_name, X, Xhat, mask=mask)

    # Put both Gaussian and Poisson families on the same "2*negLL" scale
    # (AIC/BIC are usually defined in terms of -2 log likelihood)
    two_nll = 2.0 * nll

    # Akaike Information Criterion (AIC):
    # AIC = 2k - 2logL ≈ 2k + 2*NLL (constants dropped)
    # Penalizes model complexity (k parameters) but not as strongly as BIC.
    AIC = 2.0 * k + two_nll

    # Bayesian Information Criterion (BIC):
    # BIC = log(N)*k - 2logL ≈ log(N)*k + 2*NLL
    # Penalizes model complexity more strongly for large N.
    BIC = np.log(max(N, 1)) * k + two_nll

    # Return both scores as floats
    return float(AIC), float(BIC)


# ----------------------------
# Gini concentration
# ----------------------------
def gini_concentration(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gini in [0,1]: 0 = perfectly even, 1 = all mass on one entry.
    Works on any nonnegative array. Flattens and ignores NaNs.
    0 means all entries contribute equally.
    1 means one entry dominates all others.
    """
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    v = v[v >= 0]
    if v.size == 0:
        return np.nan
    s = v.sum()
    if s <= eps:
        return 0.0
    v = np.sort(v)
    n = v.size
    # Gini = 1 + 1/n - 2 * sum_i ((n+1-i)/n * v_i / sum(v))
    coef = np.arange(1, n + 1)
    return 1.0 + 1.0 / n - 2.0 * np.sum((n + 1 - coef) * v) / (n * s)


# ----------------------------
# A thin wrapper to fit with restarts
# ----------------------------


@dataclass
class FitResult:
    est: BinaryTriFactorizationEstimator
    loss: float
    percent_loss: float
    rmse: float
    percent_rmse: float
    seed: int


def _fit_with_restarts(
    est_maker: Callable[..., BinaryTriFactorizationEstimator],
    X: np.ndarray,
    n_row: int,
    n_col: int,
    *,
    restarts: int = 3,
    seeds: Optional[Iterable[int]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> FitResult:
    fit_kwargs = fit_kwargs or {}
    if seeds is None:
        # deterministic but varied
        seeds = range(10_000, 10_000 + restarts)

    best = None
    for s in list(seeds)[:restarts]:
        est = est_maker(
            n_row_clusters=n_row, n_col_clusters=n_col, random_state=s
        )
        est.fit(X)
        # training loss (use the same metric family as model_loss)
        Xhat = est.reconstruct()
        loss = model_loss(X, Xhat, getattr(est, "loss", "gaussian"))
        if (best is None) or (loss < best.loss):
            percent_loss = 100.0 * loss / np.sum(X**2)
            rmse = np.sqrt(np.mean((X - Xhat) ** 2))
            percent_rmse = 100.0 * rmse / np.sqrt(np.mean(X**2))

            best = FitResult(
                est=est,
                loss=loss,
                percent_loss=percent_loss,
                rmse=rmse,
                percent_rmse=percent_rmse,
                seed=s,
            )
    return best


def make_gap_cellmask(min_keep: int = 6):
    """
    Returns a callable that, given a *fitted* estimator est, builds a cell-level (I, J) mask
    by (a) selecting strong blocks with the gap heuristic, then (b) projecting to (I, J).
    """

    def _fn(est: BinaryTriFactorizationEstimator) -> np.ndarray:
        allowed = est.allowed_mask_from_gap(min_keep=min_keep)  # (R, C)
        # Sanity checks
        U, B, V = est.factors()
        assert (
            U is not None and V is not None and B is not None
        ), "Estimator must be fitted"
        assert (
            allowed.shape == B.shape
        ), f"allowed {allowed.shape} != B {B.shape}"
        cell_mask = est.blockmask_to_cellmask(allowed)  # (I, J)
        return cell_mask

    return _fn


# ----------------------------
# Sweep R×C and compute metrics
# ----------------------------
def sweep_btf_grid(
    est_maker: Callable[..., BinaryTriFactorizationEstimator],
    X: np.ndarray,
    R_list: Iterable[int],
    C_list: Iterable[int],
    *,
    restarts: int = 3,
    seeds: Optional[Iterable[int]] = None,
    min_keep: int = 6,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,  # NEW: Number of parallel processes
    batch_size: int = 4,  # NEW: Number of (R,C) pairs per batch
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (n_row, n_col) containing:
      - loss, val-like PVE (on X vs Xhat), sil_mean,
        ΔLoss_total, ΔLoss_gini, frac_weak (20th pct), B_sparsity, coverage,
        plus shapes and seed of the chosen restart.

    Parameters
    ----------
    R_list : Iterable[int] or range
        Row cluster counts to try. Can be a list [2, 3, 5] or range(2, 6)
    C_list : Iterable[int] or range
        Column cluster counts to try. Can be a list [2, 3, 5] or range(2, 6)
    n_jobs : int, default=1
        Number of parallel processes. Use -1 for all CPU cores, 1 for single-threaded
    batch_size : int, default=4
        Number of (R,C) pairs per batch for multiprocessing
    """

    # Convert ranges to lists if needed
    if isinstance(R_list, range):
        R_list = list(R_list)
    if isinstance(C_list, range):
        C_list = list(C_list)

    # Create all (R,C) combinations
    rc_pairs = [(R, C) for R in R_list for C in C_list]
    total_combinations = len(rc_pairs)

    logger.info(f"Total (R,C) combinations to process: {total_combinations}")

    # Determine number of processes
    if n_jobs == -1:
        import multiprocessing

        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    logger.info(f"Using {n_jobs} processes for BTF grid search")

    # Use multiprocessing if n_jobs > 1
    if n_jobs == 1:
        # Single-threaded processing
        return _sweep_btf_grid_sequential(
            est_maker,
            X,
            rc_pairs,
            restarts,
            seeds,
            min_keep,
            fit_kwargs,
        )
    else:
        # Multi-threaded processing
        return _sweep_btf_grid_parallel(
            est_maker,
            X,
            rc_pairs,
            restarts,
            seeds,
            min_keep,
            fit_kwargs,
            n_jobs,
            batch_size,
        )


def suggest_min_keep_elbow(est: BinaryTriFactorizationEstimator) -> int:
    """
    Compute block energies = squared block strength × cluster sizes.
    Sort them descending.
    Compute cumulative explained energy curve.
    Compare it to the diagonal baseline (uniform case).
    Pick the k with the maximum deviation — the elbow.
    Return k (≥1).
    """
    if est.U_ is None or est.V_ is None or est.B_ is None:
        raise ValueError("Fit the model first.")
    U, V, B = est.U_, est.V_, est.B_
    nr = U.sum(axis=0).astype(float)  # (R,)
    nc = V.sum(axis=0).astype(float)  # (C,)
    R, C = B.shape

    # block "energy"
    E = (B**2) * (nr[:, None] * nc[None, :])  # (R,C)
    e = np.sort(E.ravel())[::-1]  # descending
    m = e.size
    if m == 0 or np.all(e == 0):
        return 1

    frac = np.cumsum(e) / (e.sum() + 1e-12)  # cumulative explained share
    x = np.arange(1, m + 1) / m  # normalized k
    # Knee as max vertical distance above the diagonal
    k_star = int(np.argmax(frac - x)) + 1  # convert to 1-based count
    return max(1, k_star)


def _process_single_rc_pair(
    R: int,
    C: int,
    est_maker,
    X,
    restarts,
    seeds,
    min_keep,
    fit_kwargs,
):
    """Process a single (R,C) pair and return the metrics row."""

    # Suppress numpy warnings for the entire computation to avoid noise
    # These warnings are expected when dealing with sparse data and large (R,C) values
    with (
        np.errstate(invalid="ignore", divide="ignore"),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*Mean of empty slice.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*invalid value encountered in divide.*",
        )

        logger.info(f"Fitting BTF with R={R}, C={C}")
        fitres = _fit_with_restarts(
            est_maker,
            X,
            n_row=R,
            n_col=C,
            restarts=restarts,
            seeds=seeds,
            fit_kwargs=fit_kwargs,
        )
        est = fitres.est
        loss_name = getattr(est, "loss", "gaussian")

        # --- PVE ---
        Xhat = est.reconstruct()

        # Defensive check for min_keep
        if min_keep is None:
            logger.warning(f"min_keep is None for R={R}, C={C}, using 0")
            min_keep = 0

        if min_keep > 0:
            logger.info(f"Computing cell mask for R={R}, C={C}")
            mask = make_gap_cellmask(min_keep=min_keep)(est)
        else:
            mask = None
        pve = compute_pve(X, Xhat, loss_name=loss_name, mask=mask)

        N_all = X.size
        N_obs = _obs_count(mask, X)

        # Defensive checks for N_obs
        if N_obs is None:
            logger.warning(f"N_obs is None for R={R}, C={C}, using 0")
            N_obs = 0

        mask_coverage = N_obs / N_all if N_all > 0 else np.nan

        # --- Assignments (for WCV/BCV and coverage) ---
        assign = est.assign_unique_blocks(
            X=X,
            method=(
                "gaussian_delta" if est.loss == "gaussian" else "poisson_delta"
            ),
            allowed_mask=(
                np.abs(est.B_) >= np.percentile(np.abs(est.B_), 20)
            ),  # drop weakest 20% blocks
        )

        # --- Silhouette-like (WCV/BCV) ---
        wcvdf = compute_block_wcv_bcv_silhouette(
            X, assign, *est.B_.shape, mask=mask
        )
        col = wcvdf["silhouette_like"]
        if not wcvdf.empty and col.notna().any():
            sil_mean = float(np.nanmean(col))
        else:
            sil_mean = np.nan

        # --- Ablations ---
        # Compute ΔLoss for every block (r,c) in the factorization
        # ΔLoss_rc = Loss_without_block - Loss_full
        # Positive values mean the block is "helpful"
        ablation_mat = est.compute_all_block_delta_losses(X, mask=mask)

        # Flatten to a 1-D array and keep only finite entries (ignore NaN/inf)
        ablation_flat = ablation_mat[np.isfinite(ablation_mat)]
        abl_pos = np.clip(ablation_flat, 0.0, None)

        # Total contribution of all blocks (sum of ΔLoss values)
        # Measures how much the model benefits overall from all blocks
        total_block_contribution = float(np.sum(abl_pos))

        # 20th percentile of block ΔLoss values (a "weak block" threshold)
        # If many blocks have ΔLoss below this, they are considered "weak"
        thresh = np.percentile(abl_pos, 20) if abl_pos.size else np.nan

        # Fraction of blocks whose ΔLoss is below the 20th percentile
        # i.e., what proportion of blocks are "weak" compared to the rest
        # frac_weak: proportion of blocks considered weak.
        frac_weak = (
            float(np.mean(abl_pos < thresh)) if abl_pos.size else np.nan
        )

        gini = (
            gini_concentration(np.maximum(ablation_flat, 0.0))
            if ablation_flat.size
            else np.nan
        )

        # N = _obs_count(mask, X)
        # delta_per_cell = (total_block_contribution / N_obs) if N_obs > 0 else np.nan
        per_cell_block_contribution = total_block_contribution / max(N_obs, 1)

        base = _baseline_array(X, mask)
        baseline_loss = neg_loglik_from_loss(loss_name, X, base, mask=mask)

        # Ensure baseline_loss is a valid number for comparison
        if baseline_loss is None or not np.isfinite(baseline_loss):
            baseline_loss = np.nan

        rel_block_contribution = (
            (total_block_contribution / baseline_loss)
            if (
                baseline_loss is not None
                and np.isfinite(baseline_loss)
                and baseline_loss > 0
            )
            else np.nan
        )

        # --- Extras ---
        b_sparsity = sparsity_of_B(est, tol=1e-4)
        coverage = coverage_from_assign(assign)

        # --- AIC/BIC-like penalized scores ---
        Xhat = est.reconstruct()
        AIC, BIC = aic_bic_scores(loss_name, X, Xhat, R, C, mask=mask)

        return {
            "n_row": R,
            "n_col": C,
            "Mask_Nobs": N_obs,
            "Mask_Coverage": mask_coverage,
            "seed": fitres.seed,
            "Loss": fitres.loss,
            "Percent_Loss": fitres.percent_loss,
            "RMSE": fitres.rmse,
            "Percent_RMSE": fitres.percent_rmse,
            "PVE": pve,
            "Mean Silhouette": sil_mean,
            "BlockContribution_Total": total_block_contribution,
            # "DeltaLoss_PerCell": delta_per_cell,
            "BlockContribution_PerCell": per_cell_block_contribution,
            "BlockContribution_RelBaseline": rel_block_contribution,
            "BlockContribution_FracWeak20": frac_weak,
            "BlockContribution_Gini": gini,
            "B_Sparsity": b_sparsity,
            "Coverage": coverage,
            "AIC": AIC,
            "BIC": BIC,
        }


def _sweep_btf_grid_sequential(
    est_maker,
    X,
    rc_pairs,
    restarts,
    seeds,
    min_keep,
    fit_kwargs,
):
    """Sequential processing of (R,C) pairs."""
    rows = []
    for R, C in rc_pairs:
        row = _process_single_rc_pair(
            R,
            C,
            est_maker,
            X,
            restarts,
            seeds,
            min_keep,
            fit_kwargs,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _process_rc_batch(args):
    """Process a batch of (R,C) pairs in parallel."""
    (
        rc_batch,
        est_maker,
        X,
        restarts,
        seeds,
        min_keep,
        fit_kwargs,
    ) = args

    results = []
    for R, C in rc_batch:
        try:
            row = _process_single_rc_pair(
                R,
                C,
                est_maker,
                X,
                restarts,
                seeds,
                min_keep,
                fit_kwargs,
            )
            results.append(row)
        except Exception as e:
            logger.error(f"Error processing R={R}, C={C}: {e}")
            # Add detailed traceback for debugging
            import traceback

            logger.error(f"Full traceback for R={R}, C={C}:")
            logger.error(traceback.format_exc())
            # Add a row with NaN values to maintain structure
            results.append(
                {
                    "n_row": R,
                    "n_col": C,
                    "Mask_Nobs": np.nan,
                    "Mask_Coverage": np.nan,
                    "seed": np.nan,
                    "Loss": np.nan,
                    "Percent_Loss": np.nan,
                    "RMSE": np.nan,
                    "Percent_RMSE": np.nan,
                    "PVE": np.nan,
                    "Mean Silhouette": np.nan,
                    "BlockContribution_Total": np.nan,
                    "BlockContribution_PerCell": np.nan,
                    "BlockContribution_RelBaseline": np.nan,
                    "BlockContribution_FracWeak20": np.nan,
                    "BlockContribution_Gini": np.nan,
                    "B_Sparsity": np.nan,
                    "Coverage": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                }
            )

    return results


def _validate_est_maker_for_multiprocessing(est_maker):
    """Validate that est_maker is picklable for multiprocessing."""
    import pickle

    try:
        # Test if est_maker can be pickled
        pickle.dumps(est_maker)
    except Exception as e:
        raise ValueError(
            f"est_maker is not picklable for multiprocessing: {e}\n"
            f"Hint: Use BinaryTriFactorizationEstimator.factory(**kwargs) instead of lambda functions.\n"
            f"Example:\n"
            f"  # Instead of: lambda **kw: BinaryTriFactorizationEstimator(loss='gaussian', **kw)\n"
            f"  # Use: BinaryTriFactorizationEstimator.factory(loss='gaussian')"
        ) from e


def _sweep_btf_grid_parallel(
    est_maker,
    X,
    rc_pairs,
    restarts,
    seeds,
    min_keep,
    fit_kwargs,
    n_jobs,
    batch_size,
):
    """Parallel processing of (R,C) pairs using multiprocessing."""

    # Validate that est_maker is picklable
    _validate_est_maker_for_multiprocessing(est_maker)

    logger.info(f"Starting parallel BTF grid search with {n_jobs} processes")

    # Create batches of (R,C) pairs
    batches = [
        rc_pairs[i : i + batch_size]
        for i in range(0, len(rc_pairs), batch_size)
    ]

    logger.info(
        f"Created {len(batches)} batches of ~{batch_size} (R,C) pairs each"
    )

    # Prepare arguments for each batch
    batch_args = [
        (
            batch,
            est_maker,
            X,
            restarts,
            seeds,
            min_keep,
            fit_kwargs,
        )
        for batch in batches
    ]

    # Process batches in parallel
    all_results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Use tqdm for progress tracking if available
        try:
            from tqdm import tqdm

            batch_results = list(
                tqdm(
                    executor.map(_process_rc_batch, batch_args),
                    total=len(batches),
                    desc="Processing BTF batches",
                )
            )
        except ImportError:
            batch_results = list(executor.map(_process_rc_batch, batch_args))

    # Flatten results from all batches
    for batch_result in batch_results:
        all_results.extend(batch_result)

    logger.info(
        f"Parallel BTF processing completed. Processed {len(rc_pairs)} (R,C) combinations"
    )

    return pd.DataFrame(all_results)


# ----------------------------
# Rank settings — flexible order
# ----------------------------
def pick_best_btf_setting(
    df: pd.DataFrame,
    max_pve_drop: float = 0.01,
    min_sil: float = -0.05,
) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()

    # --- Filter by silhouette (with single back-off) ---
    if "Mean Silhouette" in d.columns:
        d1 = d[d["Mean Silhouette"] >= min_sil]
        if not d1.empty:
            d = d1  # keep filtered; else keep original d (back off)

    # --- Filter by PVE window around best ---
    if "PVE" in d.columns:
        pve_series = pd.to_numeric(d["PVE"], errors="coerce")
        if pve_series.notna().any():
            pve_star = float(pve_series.max())
            d = d[pve_series >= pve_star - float(max_pve_drop)]
        # if all NaN, skip PVE filtering

    if d.empty:
        raise ValueError(
            "No candidate settings remain after filters (PVE/Silhouette)."
        )

    # --- Sorting: prefer high PVE/structure, then smaller models, then deterministic tie ---
    sort_cols = [
        "PVE",  # maximize
        "Mean Silhouette",  # maximize
        "BlockContribution_RelBaseline",  # maximize
        "BlockContribution_Gini",  # maximize
        "BlockContribution_FracWeak20",  # minimize
        "n_row",  # minimize
        "n_col",  # minimize
    ]
    ascending = [False, False, False, False, True, True, True]

    # Only use columns that exist; preserve paired asc flags
    cols_use, asc_use = (
        zip(*[(c, a) for c, a in zip(sort_cols, ascending) if c in d.columns])
        if any(c in d.columns for c in sort_cols)
        else ([], [])
    )

    if cols_use:
        # Convert numeric-like cols safely to numeric for consistent ordering
        for c in cols_use:
            if c in (
                "n_row",
                "n_col",
                "BlockContribution_FracWeak20",
                "PVE",
                "Mean Silhouette",
                "BlockContribution_RelBaseline",
                "BlockContribution_Gini",
            ):
                d[c] = pd.to_numeric(d[c], errors="coerce")
        # Deterministic tie-breaker even if all sort cols tie
        d = d.sort_values(
            list(cols_use) + ["n_row", "n_col"],
            ascending=list(asc_use) + [True, True],
            kind="mergesort",
            na_position="last",
            ignore_index=True,
        )

    # --- Ensure n_row/n_col present and pick best unique pair ---
    if not {"n_row", "n_col"}.issubset(d.columns):
        raise ValueError("Missing required columns: n_row and/or n_col.")

    settings_sorted = d.drop_duplicates(
        subset=["n_row", "n_col"], keep="first"
    ).reset_index(drop=True)
    if settings_sorted.empty:
        raise ValueError(
            "After de-duplicating (n_row, n_col), no settings remain."
        )

    best = settings_sorted.iloc[0].copy()
    return settings_sorted, best


def normalize_data_and_fit_estimator(
    df: pd.DataFrame,
    est_maker: Callable[..., BinaryTriFactorizationEstimator],
    n_row: int,
    n_col: int,
    keep_strategy: str = "delta_then_size",
    random_state: int = 0,
    min_keep: int = 6,
) -> tuple[BinaryTriFactorizationEstimator, dict]:
    # Build normalized + raw matrices
    norm_data = normalize_data(
        df,
        column_name="growth_rate_1",
        log_transform=False,
        median_transform=True,
        mean_transform=False,
        zscore_rows=False,
        zscore_cols=True,
    ).fillna(0)
    est = est_maker(n_row, n_col, random_state=random_state)

    # Handle both DataFrame and NumPy array cases
    if hasattr(norm_data, "to_numpy"):
        X_array = norm_data.to_numpy()
    else:
        X_array = norm_data

    est.fit(X_array)
    assign = est.filter_blocks(
        X=X_array,
        min_keep=min_keep,
        keep_strategy=keep_strategy,
        return_frame=False,
    )
    return est, assign


def _prep_matrix_for_btnmf(
    df: pd.DataFrame,
    id_cols: list[str] | None,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns: X_mat (I×J nonnegative), row_names (I,), feat_cols (len=J)
    - If normalize=True: per-column min-max to [0,1] after shifting to nonnegative.
    - If normalize=False: assume already nonnegative & comparably scaled (e.g., M_btnmf).
    """
    X = df.copy()

    # infer id cols
    if id_cols is None:
        if {"store", "item"}.issubset(X.columns):
            id_cols = ["store", "item"]
        elif "store_item" in X.columns:
            id_cols = ["store_item"]
        else:
            id_cols = []

    # numeric feature columns
    feat_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in feat_cols if c not in id_cols]
    if not feat_cols:
        raise ValueError("No numeric feature columns found to cluster on.")

    # row names
    if id_cols == ["store", "item"]:
        row_names = (
            X["store"].astype(str) + "▮" + X["item"].astype(str)
        ).to_numpy()
    elif id_cols == ["store_item"]:
        row_names = X["store_item"].astype(str).to_numpy()
    else:
        row_names = X.index.astype(str).to_numpy()

    # matrix
    M = X[feat_cols].to_numpy(dtype=np.float32)
    M = np.where(np.isfinite(M), M, 0.0)

    # ensure nonnegative
    min_per_col = np.nanmin(M, axis=0)
    neg_cols = min_per_col < 0
    if np.any(neg_cols):
        M[:, neg_cols] = M[:, neg_cols] - min_per_col[neg_cols][None, :]

    if normalize:
        # per-column min-max to [0,1]
        col_min = np.nanmin(M, axis=0)
        M0 = M - col_min
        col_max = np.nanmax(M0, axis=0)
        col_max[col_max == 0] = 1.0
        M = M0 / col_max

    # clip any tiny numeric junk
    M = np.clip(M, 0.0, None)

    return M, row_names, feat_cols


def _build_assign_df(
    row_names: np.ndarray,
    block_ids: np.ndarray,
    df: pd.DataFrame,
    id_cols: tuple = ("store_item",),
) -> pd.DataFrame:
    """
    Create tidy (key, block_id) frame aligned to BTNMF row order,
    robust to 1D/2D block_ids and tuple row_names.
    """
    # --- collapse block_ids ---
    bid = np.asarray(block_ids)
    if bid.ndim == 1:
        bid = bid.astype(int)
    elif bid.ndim == 2:
        I, J = bid.shape
        out = np.full(I, -1, dtype=int)
        for i in range(I):
            vals = bid[i][bid[i] >= 0]
            if vals.size:
                u, c = np.unique(vals, return_counts=True)
                out[i] = int(u[np.argmax(c)])
        bid = out
    else:
        raise ValueError("block_ids must be 1-D or 2-D")

    rn = np.asarray(row_names, dtype=object).ravel()

    if id_cols == ("store_item",):
        assign_df = pd.DataFrame(
            {"store_item": rn.astype(str), "block_id": bid}
        )
        if "store_item" not in df.columns and {"store", "item"}.issubset(
            df.columns
        ):
            df = df.assign(
                store_item=df["store"].astype(str)
                + "_"
                + df["item"].astype(str)
            )
        df_out = df.copy()
        df_out["store_item"] = df_out["store_item"].astype(str)
        return df_out.merge(assign_df, on="store_item", how="left")

    else:
        raise ValueError(f"Unsupported id_cols={id_cols}")


def cluster_data_and_explain_blocks(
    df: pd.DataFrame,
    row_range: range,
    col_range: range,
    *,
    normalize: bool = True,
    alpha: float = 1e-2,
    beta: float = 0.6,
    block_l1: float = 0.0,
    b_inner: int = 15,
    max_iter: int = 50,
    k_row: int = 1,
    k_col: int = 1,
    keep_strategy: str = "delta_then_size",
    tol: float = 1e-5,
    max_pve_drop: float = 0.01,
    min_sil: float = -0.05,
    min_keep: int = 6,
    top_k: Optional[int] = None,
    top_rank_fn: Optional[Path] = None,
    summary_fn: Optional[Path] = None,
    block_id_fn: Optional[Path] = None,
    output_fn: Optional[Path] = None,
    n_jobs: int = 1,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Works with BOTH:
      • raw feature tables  -> set normalize=True (min-max per column to [0,1])
      • prebuilt M_btnmf    -> set normalize=False (already nonnegative [0,1])
    df must contain ids: ["store","item"] or ["store_item"] plus numeric feature columns.
    """
    # overlap budgets: <=0 means data-driven stopping
    if k_row <= 0 or k_col <= 0:
        k_row = None
        k_col = None

    # Build factory (same as before)
    make_btf = BinaryTriFactorizationEstimator.factory(
        k_row=k_row,
        k_col=k_col,
        loss="gaussian",
        alpha=alpha,
        beta=beta,
        block_l1=block_l1,
        b_inner=b_inner,
        max_iter=max_iter,
        tol=tol,
    )

    # ----- INPUT PREP (new) -----
    # auto-detect id cols; min-max if normalize=True

    id_cols = ["store_item"]

    X_mat, row_names, feat_cols = _prep_matrix_for_btnmf(
        df, id_cols=id_cols, normalize=normalize
    )

    # small stats
    num_total = X_mat.size
    num_nans = int(np.isnan(X_mat).sum())
    num_finite = int(np.isfinite(X_mat).sum())
    logger.info(
        "Finite: %d, NaNs: %d (%.1f%%)",
        num_finite,
        num_nans,
        100.0 * num_nans / max(1, num_total),
    )

    # ----- grid sweep over (R,C) -----
    R_list = list(row_range)
    C_list = list(col_range)
    grid_df = sweep_btf_grid(
        make_btf,
        X_mat,
        R_list,
        C_list,
        restarts=3,
        seeds=range(123, 999),
        min_keep=None,
        fit_kwargs={"max_iter": max_iter, "tol": tol},
        n_jobs=n_jobs,
        batch_size=batch_size,
    )
    logger.info(f"Grid search completed. Found {len(grid_df)} combinations")

    ranked_df, best = pick_best_btf_setting(
        grid_df, max_pve_drop=max_pve_drop, min_sil=min_sil
    )
    if top_rank_fn is not None and top_k is not None:
        ranked_df.iloc[:top_k].to_csv(top_rank_fn, index=False)

    n_row = int(best["n_row"])
    n_col = int(best["n_col"])
    est = make_btf(n_row, n_col, random_state=42)

    # ----- fit final model -----
    est.fit(X_mat)
    suggested_min_keep_elbow = suggest_min_keep_elbow(est)
    logger.info(
        f"Current min_keep: {min_keep}-Suggested min_keep: {suggested_min_keep_elbow}"
    )
    logger.info(f"keep_strategy: {keep_strategy}")
    if keep_strategy == "TopK":
        min_keep = min(min_keep, suggested_min_keep_elbow)
    else:
        min_keep = suggested_min_keep_elbow

    assign = est.filter_blocks(
        X=X_mat,
        min_keep=min_keep,
        keep_strategy=keep_strategy,
        return_frame=False,
    )

    # optional summary
    if summary_fn is not None:
        summary = est.explain_blocks(
            X=X_mat,
            assign=assign,
            row_names=row_names,
            col_names=np.array(feat_cols),
            top_k=5,
        )
        summary.to_csv(summary_fn, index=False)

    # diagnostics
    U, _, V = est.factors()

    block_ids = assign["block_id"]
    out = _build_assign_df(row_names, block_ids, df, id_cols=("store_item",))

    if block_id_fn is not None:
        np.save(block_id_fn, np.asarray(out["block_id"]))

    unique_assignments = np.unique(out["block_id"])
    logger.info(
        f"unique block assignments (first 20): {unique_assignments[:20]}  "
        f"count: {unique_assignments.size}"
    )
    logger.info(f"row-cluster counts: {U.sum(axis=0).astype(int)}")
    logger.info(f"col-cluster counts: {V.sum(axis=0).astype(int)}")

    if output_fn is not None:
        save_csv_or_parquet(out, output_fn)

    return out
