import numpy as np
from typing import Dict, Any, Iterable, Tuple, Optional
import pandas as pd
from sklearn.cluster import KMeans
from src.BinaryTriFactorizationEstimator import BinaryTriFactorizationEstimator
from dataclasses import dataclass

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
def ablate_block_delta_loss(
    est: BinaryTriFactorizationEstimator,
    X: np.ndarray,
    r: int,
    c: int,
    mask: np.ndarray = None,
) -> float:
    """
    ΔLoss_rc = Loss(X, Xhat_without_rc) - Loss(X, Xhat_full)
    where removing block (r,c) is a rank-1 update:
       Xhat_without = Xhat - B_rc * (U[:,r] ⊗ V[:,c])
    Positive Δ means the block is helpful.
    """
    U, V, B = est.factors()
    Xhat = est.reconstruct()
    loss_name = est.loss

    b_rc = float(B[r, c])
    if b_rc == 0.0:
        return 0.0

    # rank-1 contribution of block (r,c)
    outer = np.outer(U[:, r].astype(float), V[:, c].astype(float))
    Xhat_wo = Xhat - b_rc * outer

    L_full = model_loss(X, Xhat, loss_name, mask)
    L_wo = model_loss(X, Xhat_wo, loss_name, mask)
    return L_wo - L_full


def compute_all_block_delta_losses(
    est: BinaryTriFactorizationEstimator, X: np.ndarray, mask: np.ndarray = None
):
    _, _, B = est.factors()
    R, C = B.shape
    dmat = np.zeros((R, C), dtype=float)
    for r in range(R):
        for c in range(C):
            dmat[r, c] = ablate_block_delta_loss(est, X, r, c, mask=mask)
    return dmat  # ΔLoss per block


# ----------------------------
# WCV / BCV / silhouette-like
# ----------------------------
def compute_block_wcv_bcv_silhouette(
    X: np.ndarray, assign_dict: Dict[str, Any], R: int, C: int, mask: np.ndarray = None
):
    """
    Using per-cell assignments from assign_unique_blocks(...):
      - assign_dict["r_star"], ["c_star"] with -1 for unassigned.
    Returns:
      DataFrame with per-block: n, mean, WCV, BCV_nearest, silhouette_like
    """
    r_star = assign_dict["r_star"]
    c_star = assign_dict["c_star"]
    I, J = r_star.shape

    # observed mask
    if mask is None:
        obs_mask = np.ones_like(r_star, dtype=bool)
    else:
        obs_mask = mask.astype(bool)

    # collect per-block cells
    stats = []
    block_means = np.full((R, C), np.nan, dtype=float)
    block_ns = np.zeros((R, C), dtype=int)

    for r in range(R):
        for c in range(C):
            sel = (r_star == r) & (c_star == c) & obs_mask
            n = int(np.sum(sel))
            block_ns[r, c] = n
            if n > 0:
                vals = X[sel].astype(float)
                mu = float(np.mean(vals))
                block_means[r, c] = mu

    # WCV per block
    for r in range(R):
        for c in range(C):
            n = block_ns[r, c]
            if n == 0:
                stats.append((r, c, 0, np.nan, np.nan, np.nan))
                continue
            sel = (r_star == r) & (c_star == c) & obs_mask
            vals = X[sel].astype(float)
            mu = block_means[r, c]
            wcv = float(np.mean((vals - mu) ** 2))

            # BCV as nearest-centroid separation in mean-space
            diffs = []
            for p in range(R):
                for q in range(C):
                    if (p == r and q == c) or np.isnan(block_means[p, q]):
                        continue
                    diffs.append((mu - block_means[p, q]) ** 2)
            bcv = float(np.min(diffs)) if diffs else np.nan

            # silhouette-like score in [−1,1]
            if np.isnan(bcv):
                sil = np.nan
            else:
                denom = max(bcv, wcv)
                sil = (bcv - wcv) / denom if denom > 0 else 0.0

            stats.append((r, c, n, mu, wcv, bcv, sil))

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


# ----------------------------
# Gini concentration
# ----------------------------
def gini_concentration(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gini in [0,1]: 0 = perfectly even, 1 = all mass on one entry.
    Works on any nonnegative array. Flattens and ignores NaNs.
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
    est: Any
    train_loss: float
    seed: int


def _fit_with_restarts(
    estimator: BinaryTriFactorizationEstimator,
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
        est = estimator.copy()
        est.n_row_clusters = n_row
        est.n_col_clusters = n_col
        # If your estimator accepts random_state, pass it via fit_kwargs
        if hasattr(est, "random_state"):
            est.random_state = s
        est.fit(X)
        # training loss (use the same metric family as model_loss)
        tr_loss = model_loss(X, est.Xhat_, getattr(est, "loss", "gaussian"))
        if (best is None) or (tr_loss < best.train_loss):
            best = FitResult(est=est, train_loss=tr_loss, seed=s)
    return best


# ----------------------------
# Sweep R×C and compute metrics
# ----------------------------
def sweep_btf_grid(
    estimator: BinaryTriFactorizationEstimator,
    X: np.ndarray,
    R_list: Iterable[int],
    C_list: Iterable[int],
    *,
    restarts: int = 3,
    seeds: Optional[Iterable[int]] = None,
    mask: Optional[np.ndarray] = None,
    assign_method_if_gauss: str = "gaussian_delta",
    assign_method_if_pois: str = "poisson_delta",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (n_row, n_col) containing:
      - train_loss, val-like PVE (on X vs Xhat), sil_mean,
        ΔLoss_total, ΔLoss_gini, frac_weak (20th pct), B_sparsity, coverage,
        plus shapes and seed of the chosen restart.
    """
    rows = []
    for R in R_list:
        for C in C_list:
            fitres = _fit_with_restarts(
                estimator,
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
            pve = compute_pve(X, est.Xhat_, loss_name=loss_name, mask=mask)

            # --- Assignments (for WCV/BCV and coverage) ---
            assign = est.assign_unique_blocks(
                X=X,
                method=(
                    assign_method_if_gauss
                    if loss_name == "gaussian"
                    else assign_method_if_pois
                ),
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
            delta_mat = compute_all_block_delta_losses(est, X, mask=mask)
            delta_flat = delta_mat[np.isfinite(delta_mat)]
            total_delta = float(np.sum(delta_flat))
            thresh = np.percentile(delta_flat, 20) if delta_flat.size else np.nan
            frac_weak = (
                float(np.mean(delta_flat < thresh)) if delta_flat.size else np.nan
            )
            gini = (
                gini_concentration(np.maximum(delta_flat, 0.0))
                if delta_flat.size
                else np.nan
            )

            # --- Extras ---
            b_sparsity = sparsity_of_B(est, tol=1e-4)
            coverage = coverage_from_assign(assign)

            rows.append(
                {
                    "n_row": R,
                    "n_col": C,
                    "seed": fitres.seed,
                    "TrainLoss": fitres.train_loss,
                    "PVE": pve,
                    "Mean Silhouette": sil_mean,
                    "DeltaLoss_Total": total_delta,
                    "DeltaLoss_Gini": gini,
                    "DeltaLoss_FracWeak20": frac_weak,
                    "B_Sparsity": b_sparsity,
                    "Coverage": coverage,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# Rank settings — flexible order
# ----------------------------
def pick_best_btf_setting(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rank by a sensible default:
      Maximize:  PVE, Mean Silhouette, DeltaLoss_Gini, Coverage
      Minimize:  TrainLoss, DeltaLoss_FracWeak20, B_Sparsity, n_row, n_col
    Feel free to edit the order to your preference.
    """
    if df is None or df.empty:
        raise ValueError("No settings to rank (empty DF).")

    d = df.copy()
    # Coerce numerics & clean
    for c in [
        "PVE",
        "Mean Silhouette",
        "DeltaLoss_Gini",
        "Coverage",
        "TrainLoss",
        "DeltaLoss_FracWeak20",
        "B_Sparsity",
        "n_row",
        "n_col",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)

    # Desired order:
    sort_cols = [
        "PVE",
        "Mean Silhouette",
        "DeltaLoss_Gini",
        "Coverage",
        "TrainLoss",
        "DeltaLoss_FracWeak20",
        "B_Sparsity",
        "n_row",
        "n_col",
    ]
    # False = descending (maximize), True = ascending (minimize)
    ascending = [False, False, False, False, True, True, True, True, True]

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

    # Best representative per (n_row,n_col)
    if not {"n_row", "n_col"}.issubset(d.columns):
        raise ValueError("Missing n_row/n_col; cannot rank settings.")
    settings_sorted = d.drop_duplicates(subset=["n_row", "n_col"], keep="first")
    if settings_sorted.empty:
        raise ValueError("No unique (n_row, n_col) settings to rank.")
    best_row = settings_sorted.iloc[0].copy()
    return settings_sorted, best_row
