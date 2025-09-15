import numpy as np
from typing import Dict, Any, Iterable, Optional, Callable
import pandas as pd
from sklearn.cluster import KMeans
from src.BinaryTriFactorizationEstimator import BinaryTriFactorizationEstimator
from src.data_utils import normalize_data
from src.utils import save_csv_or_parquet
from dataclasses import dataclass
from src.plot_util import plot_block_annot_heatmap
from src.utils import get_logger
from pathlib import Path

logger = get_logger(__name__)


def get_normalized_assignments(
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
        pd.DataFrame(block_id_mat, index=norm_data.index, columns=norm_data.columns)
        .stack()
        .rename("block_id")
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # long values from the SAME matrix used for clustering
    df = (
        norm_data.stack()
        .rename(value_col)
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # store, item, value, block_id (no date column)
    df = df.merge(blk_long, on=[row_col, col_col], how="left")

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
        pd.DataFrame(block_id_mat, index=norm_data.index, columns=norm_data.columns)
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
    summ = summarize_blocks_for_merge(est, norm_data.to_numpy(), assign)
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
        pd.DataFrame(bid_merged, index=norm_data.index, columns=norm_data.columns)
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
            by=["__rep", "__min", "__mean", *list(col_df.columns)], kind="mergesort"
        ).index.tolist()
    else:
        # fallback: pure lexicographic by full column pattern
        col_order = blk.T.sort_values(
            by=list(blk.index), kind="mergesort"
        ).index.tolist()

    return row_order, col_order


# ----------------------------
# Losses
# ----------------------------
def gaussian_loss(X: np.ndarray, Xhat: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is None:
        diff = X - Xhat
    else:
        diff = (X - Xhat)[mask]
    return float(np.sum(diff * diff))


def poisson_nll(
    X: np.ndarray, Xhat: np.ndarray, mask: np.ndarray = None, eps=1e-9
) -> float:
    # NLL(μ; x) = μ - x*log μ (+ const)
    if mask is None:
        x = X
        mu = np.maximum(Xhat, eps)
    else:
        x = X[mask]
        mu = np.maximum(Xhat[mask], eps)
    return float(np.sum(mu - x * np.log(mu)))


def model_loss(
    X: np.ndarray, Xhat: np.ndarray, loss_name, mask: np.ndarray = None
) -> float:
    if loss_name == "gaussian":
        return gaussian_loss(X, Xhat, mask)
    elif loss_name == "poisson":
        return poisson_nll(X, Xhat, mask)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


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


def compute_all_block_delta_losses(
    est: BinaryTriFactorizationEstimator, X: np.ndarray, mask: np.ndarray = None
):

    # If B is larger, trim a view
    U, B, V = est.factors()
    # Use aligned dims from U and V
    R = U.shape[1]
    C = V.shape[1]
    # B = est.B_
    if B.shape != (R, C):
        B = B[:R, :C]

    dmat = np.zeros((R, C), dtype=float)
    for r in range(R):
        for c in range(C):
            dmat[r, c] = ablate_block_delta_loss(est, X, r, c, B=B, mask=mask)
    return dmat


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

    Returns a DataFrame with per-block rows (r,c) and columns:
      ["r", "c", "n", "mean", "WCV", "BCV_nearest", "silhouette_like"].

    Notes:
      - WCV is mean squared deviation within the (r,c) block.
      - BCV_nearest is the minimum squared separation between this block mean
        and any other block's mean.
      - silhouette_like is in [-1, 1]; higher is better separation vs compactness.
    """
    # Per-cell block assignments (shape must match X)
    r_star = assign_dict["r_star"]  # shape (I,J), values in {0..R-1} or -1
    c_star = assign_dict["c_star"]  # shape (I,J), values in {0..C-1} or -1
    I, J = r_star.shape

    # Observed mask: include only cells that are both observed and assigned to some block
    if mask is None:
        obs_mask = np.ones_like(r_star, dtype=bool)
    else:
        obs_mask = mask.astype(bool)

    # Prepare outputs: per-block mean and count (n)
    stats = []  # list of rows for the final DataFrame
    block_means = np.full((R, C), np.nan)  # block centroids (means)
    block_ns = np.zeros((R, C), dtype=int)  # cell counts per block

    # First pass: compute per-block means and counts
    for r in range(R):
        for c in range(C):
            # cells that belong to block (r,c), are observed, and not unassigned
            sel = (r_star == r) & (c_star == c) & obs_mask
            n = int(sel.sum())
            block_ns[r, c] = n
            if n > 0:
                vals = X[sel].astype(float)
                block_means[r, c] = float(vals.mean())

    # Second pass: compute WCV, nearest-BCV, and silhouette-like per block
    for r in range(R):
        for c in range(C):
            n = block_ns[r, c]

            # If this block has no cells, emit a row with NaNs for stats
            if n == 0:
                # BUGFIX: include the 'mean' slot; previously one field short
                stats.append((r, c, 0, np.nan, np.nan, np.nan, np.nan))
                continue

            # Gather this block's values and mean
            sel = (r_star == r) & (c_star == c) & obs_mask
            vals = X[sel].astype(float)
            mu = block_means[r, c]

            # WCV = mean squared deviation from the block mean
            wcv = float(np.mean((vals - mu) ** 2))

            # BCV_nearest = min squared distance to any other defined block mean
            diffs = []
            for p in range(R):
                for q in range(C):
                    if p == r and q == c:
                        continue
                    other_mu = block_means[p, q]
                    if not np.isnan(other_mu):
                        diffs.append((mu - other_mu) ** 2)
            bcv = float(np.min(diffs)) if diffs else np.nan

            # Silhouette-like: (bcv - wcv) / max(bcv, wcv), in [-1, 1]
            if np.isnan(bcv):
                sil = np.nan
            else:
                denom = max(bcv, wcv)
                sil = (bcv - wcv) / denom if denom > 0 else 0.0

            stats.append((r, c, n, mu, wcv, bcv, sil))

    # Assemble tidy per-block DataFrame
    df = pd.DataFrame(
        stats,
        columns=["r", "c", "n", "mean", "WCV", "BCV_nearest", "silhouette_like"],
    )
    return df


# ----------------------------
# Sparsity & coverage helpers
# ----------------------------
def sparsity_of_B(est: BinaryTriFactorizationEstimator, tol=1e-6):
    _, _, B = est.factors()
    return float(np.mean(np.abs(B) < tol))  # fraction near-zero


def coverage_from_assign(assign_dict: Dict[str, Any]):
    r_star = assign_dict["r_star"]
    return float(np.mean(r_star >= 0))  # fraction of cells assigned


def _baseline_array(
    X: np.ndarray, loss_name: str, mask: np.ndarray | None
) -> np.ndarray:
    """
    Constant-mean baseline:
      - Gaussian: baseline = mean(X[mask])
      - Poisson : baseline = mean-rate μ = mean(X[mask])
    """
    obs = X if mask is None else X[mask]
    mu = float(np.mean(obs))
    return np.full_like(X, mu, dtype=float)


# ---------- AIC/BIC helpers ----------
def _obs_count(mask: np.ndarray | None, X: np.ndarray) -> int:
    """Number of observed cells; if no mask, count all cells in X."""
    if mask is None:
        return int(X.size)
    mask = mask.astype(bool)
    if mask.shape != X.shape:
        raise ValueError(f"mask shape {mask.shape} must match X shape {X.shape}")
    return int(np.sum(mask))


def neg_loglik_from_loss(
    loss_name: str, X: np.ndarray, Xhat: np.ndarray, mask=None
) -> float:
    """
    Return a quantity proportional to negative log-likelihood.
    (Exact constants cancel in AIC/BIC comparisons across same data.)
    """
    if loss_name == "gaussian":
        return gaussian_loss(X, Xhat, mask=mask)  # SSE ~ 2*negLL up to sigma^2 scale
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
        est = est_maker(n_row, n_col, random_state=s)
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
        assert allowed.shape == B.shape, f"allowed {allowed.shape} != B {B.shape}"
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
    """

    # Convert ranges to lists if needed
    if isinstance(R_list, range):
        R_list = list(R_list)
    if isinstance(C_list, range):
        C_list = list(C_list)

    rows = []
    for R in R_list:
        for C in C_list:
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
            if min_keep > 0:
                logger.info(f"Computing cell mask for R={R}, C={C}")
                mask = make_gap_cellmask(min_keep=min_keep)(est)
            else:
                mask = None
            pve = compute_pve(X, Xhat, loss_name=loss_name, mask=mask)

            N_all = X.size
            N_obs = _obs_count(mask, X)
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
            sil_mean = (
                float(np.nanmean(wcvdf["silhouette_like"]))
                if not wcvdf.empty
                else np.nan
            )

            # --- Ablations ---
            # Compute ΔLoss for every block (r,c) in the factorization
            # ΔLoss_rc = Loss_without_block - Loss_full
            # Positive values mean the block is "helpful"
            ablation_mat = compute_all_block_delta_losses(est, X, mask=mask)

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
            frac_weak = float(np.mean(abl_pos < thresh)) if abl_pos.size else np.nan

            gini = (
                gini_concentration(np.maximum(ablation_flat, 0.0))
                if ablation_flat.size
                else np.nan
            )

            # N = _obs_count(mask, X)
            # delta_per_cell = (total_block_contribution / N_obs) if N_obs > 0 else np.nan
            per_cell_block_contribution = total_block_contribution / max(N_obs, 1)

            base = _baseline_array(X, loss_name, mask)
            baseline_loss = neg_loglik_from_loss(loss_name, X, base, mask=mask)
            rel_block_contribution = (
                (total_block_contribution / baseline_loss)
                if baseline_loss > 0
                else np.nan
            )

            # --- Extras ---
            b_sparsity = sparsity_of_B(est, tol=1e-4)
            coverage = coverage_from_assign(assign)

            # --- AIC/BIC-like penalized scores ---
            Xhat = est.reconstruct()
            AIC, BIC = aic_bic_scores(loss_name, X, Xhat, R, C, mask=mask)

            rows.append(
                {
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
            )

    return pd.DataFrame(rows)


# ----------------------------
# Rank settings — flexible order
# ----------------------------


def pick_best_btf_setting(
    df: pd.DataFrame, max_pve_drop: float = 0.01, min_sil: float = -0.05
) -> tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    # Per-obs scores help compare across N
    if {"AIC", "BIC", "Coverage"}.issubset(d.columns):
        N = d["Coverage"] * 0 + 1  # placeholder if you don’t have N per row
    # Filter by silhouette
    if "Mean Silhouette" in d.columns:
        d = d[d["Mean Silhouette"] >= min_sil]
        if d.empty:  # if too strict, back off once
            d = df.copy()
    # PVE within epsilon of best
    if "PVE" in d.columns:
        pve_star = d["PVE"].max()
        d = d[d["PVE"] >= pve_star - max_pve_drop]

    # Now rank: favor simplicity and structure after the filter
    sort_cols = [
        "PVE",  # maximize: explained variance
        "Mean Silhouette",  # maximize: cluster separation
        "BlockContribution_RelBaseline",  # maximize: relative gain
        "BlockContribution_Gini",  # maximize: concentrated signal
        "BlockContribution_FracWeak20",  # minimize: few weak blocks
        "n_row",
        "n_col",  # minimize model size
    ]
    ascending = [
        False,  # PVE
        False,  # Silhouette
        False,  # BlockContribution_RelBaseline
        False,  # BlockContribution_Gini
        True,  # BlockContribution_FracWeak20
        True,
        True,  # size
    ]

    used = [(c, a) for c, a in zip(sort_cols, ascending) if c in d.columns]
    if used:
        by, asc = zip(*used)
        d = d.sort_values(
            list(by),
            ascending=list(asc),
            kind="mergesort",
            na_position="last",
            ignore_index=True,
        )

    # best unique (n_row,n_col)
    if not {"n_row", "n_col"}.issubset(d.columns):
        raise ValueError("Missing n_row/n_col")
    settings_sorted = d.drop_duplicates(subset=["n_row", "n_col"], keep="first")
    best = settings_sorted.iloc[0].copy()
    return settings_sorted, best


def normalize_data_and_fit_estimator(
    df: pd.DataFrame,
    est_maker: Callable[..., BinaryTriFactorizationEstimator],
    n_row: int,
    n_col: int,
    random_state: int = 0,
    min_keep: int = 6,
) -> tuple[BinaryTriFactorizationEstimator, dict]:
    # Build normalized + raw matrices
    norm_data = normalize_data(
        df,
        column_name="growth_rate_1",
        log_transform=False,
        median_transform=False,
        mean_transform=True,
        zscore_rows=False,
        zscore_cols=True,
    ).fillna(0)
    est = est_maker(n_row, n_col, random_state=random_state)
    est.fit(norm_data.to_numpy())
    assign = est.filter_blocks(
        X=norm_data.to_numpy(), min_keep=min_keep, return_frame=False
    )
    return est, assign


def cluster_data_and_explain_blocks(
    df: pd.DataFrame,
    row_range: range,
    col_range: range,
    alpha: float = 1e-2,
    beta: float = 0.6,
    block_l1: float = 0.0,
    b_inner: int = 15,
    max_iter: int = 50,
    tol: float = 1e-5,
    max_pve_drop: float = 0.01,
    min_sil: float = -0.05,
    min_keep: int = 6,
    top_k: Optional[int] = None,
    top_rank_fn: Optional[Path] = None,
    summary_fn: Optional[Path] = None,
    output_fn: Optional[Path] = None,
    figure_fn: Optional[Path] = None,
) -> pd.DataFrame:
    make_btf = BinaryTriFactorizationEstimator.factory(
        k_row=None,
        k_col=None,
        loss="gaussian",
        alpha=alpha,
        beta=beta,
        block_l1=block_l1,  # 0 = off; >0 = L1 on B (0.01 = good start)
        b_inner=b_inner,  # inner prox steps for B when block_l1>0
        max_iter=max_iter,
        tol=tol,
    )

    # Define your grid
    R_list = row_range
    C_list = col_range
    norm_data = normalize_data(
        df,
        column_name="growth_rate_1",
        log_transform=False,
        median_transform=False,
        mean_transform=True,
        zscore_rows=False,
        zscore_cols=True,
    ).fillna(0)

    # Run the sweep
    grid_df = sweep_btf_grid(
        make_btf,
        norm_data.to_numpy(dtype=np.float32),
        R_list,
        C_list,
        restarts=3,
        seeds=range(123, 999),  # optional
        min_keep=min_keep,
        fit_kwargs={"max_iter": max_iter, "tol": tol},
    )

    # Rank and pick the best
    ranked_df, best = pick_best_btf_setting(
        grid_df, max_pve_drop=max_pve_drop, min_sil=min_sil
    )

    if top_rank_fn is not None:
        logger.info(f"Saving top {top_k} ranked_df to {top_rank_fn}")
        top_k_df = ranked_df.iloc[:top_k]
        top_k_df.to_csv(top_rank_fn, index=False)

    est_maker = BinaryTriFactorizationEstimator.factory(
        n_row_clusters=best["n_row"],
        n_col_clusters=best["n_col"],
        k_row=None,
        k_col=None,
        loss="gaussian",
        alpha=alpha,
        beta=beta,
        block_l1=block_l1,
        b_inner=b_inner,
        max_iter=max_iter,
        tol=tol,
    )
    n_row = best["n_row"]
    n_col = best["n_col"]
    est, assign = normalize_data_and_fit_estimator(
        df, est_maker, n_row, n_col, random_state=0, min_keep=min_keep
    )
    if summary_fn is not None:
        summary = est.explain_blocks(
            X=norm_data.to_numpy(),
            assign=assign,
            row_names=norm_data.index.to_numpy(),
            col_names=norm_data.columns.to_numpy(),
            top_k=5,
        )
        logger.info(f"Saving summary to {summary_fn}")
        summary.to_csv(summary_fn, index=False)
    norm_data = normalize_data(
        df,
        column_name="growth_rate_1",
        log_transform=False,
        median_transform=False,
        mean_transform=True,
        zscore_rows=False,
        zscore_cols=False,
    ).fillna(0)
    df2 = get_normalized_assignments(
        assign, norm_data
    )  # contains unique per-cell block_id

    # plot (no dates in this table)
    row_order, col_order = get_sorted_row_col(df2)
    plot_block_annot_heatmap(
        df2,
        ttl="Store-SKU Clusters",
        value_col="growth_rate_1",
        block_col="block_id",
        row_col="store",
        col_col="item",
        date_col=None,
        row_order=row_order,
        col_order=col_order,
        fmt="{:.0f}",
        cell_h=0.6,
        cell_w=0.75,
        font_size=11,
        # figsize=(6, 4),
        xlabel_size=14,
        ylabel_size=14,
        label_weight="bold",
        fn=figure_fn,
        xtick_rotation=45,
        show_plot=False,
    )
    df = df.merge(
        df2.drop(columns="growth_rate_1", axis=1), on=["store", "item"], how="left"
    )

    if output_fn is not None:
        logger.info(f"Saving output to {output_fn}")
        save_csv_or_parquet(df, output_fn)

    return df
