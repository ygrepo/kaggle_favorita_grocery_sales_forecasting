import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.cluster import (
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
)
from pathlib import Path
from src.data_utils import normalize_data, mav_by_cluster
from src.gdkm import GeneralizedDoubleKMeans
from src.solver_gdkm import TiedGDKM
from typing import Optional, Dict, Any, Tuple


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_ClusterClass(model_class_name: str) -> object:
    """
    Get a clustering model class from a string name.

    Parameters
    ----------
    model_class_name : str
        Name of the clustering model class

    Returns
    -------
    model_class : class
        The clustering model class
    """
    MODEL_CLASSES = {
        "SpectralBiclustering": SpectralBiclustering,
        "SpectralCoclustering": SpectralCoclustering,
        "SpectralClustering": SpectralClustering,
        "GeneralizedDoubleKMeans": GeneralizedDoubleKMeans,
        "TiedGDKM": TiedGDKM,
    }

    if model_class_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model class: {model_class_name}. Available: {list(MODEL_CLASSES.keys())}"
        )

    return MODEL_CLASSES[model_class_name]


def get_ClusterModel(
    model_class: str, n_row: int, n_col: int, **model_kwargs
) -> tuple[object, int]:
    """
    Create a clustering model instance from a string name.

    Parameters
    ----------
    model_class : str
        Name of the clustering model class
    n_row : int
        Number of row clusters
    n_col : int
        Number of column clusters
    **model_kwargs
        Additional keyword arguments for the model constructor

    Returns
    -------
    model : object
        Instantiated clustering model
    use_n_col : int
        Effective number of column clusters used
    """
    # Map string names to actual class objects
    model_cls = get_ClusterClass(model_class)
    logger.info(f"Building model {model_class} with n_row={n_row}, n_col={n_col}")
    use_n_col = n_col
    try:
        # --- Build the model with the right n_clusters ---
        if model_class == "SpectralBiclustering":
            model = model_cls(n_clusters=(n_row, n_col), **model_kwargs)
            use_n_col = n_col
        elif model_class == "SpectralCoclustering":
            model = model_cls(n_clusters=n_row, **model_kwargs)
            # n_col not defined for this estimator
            use_n_col = n_row
        elif model_class == "GeneralizedDoubleKMeans":
            # GDKM uses n_row_clusters and n_col_clusters
            model = model_cls(
                n_row_clusters=n_row,
                n_col_clusters=n_col,  # Global V (tied columns)
                tie_columns=True,  # Use tied columns mode
                **model_kwargs,
            )
            use_n_col = n_col
        elif model_class == "TiedGDKM":
            model = model_cls(
                n_row_clusters=n_row,
                n_col_clusters=n_col,
                **model_kwargs,
            )
            use_n_col = n_col
        else:
            # Default case for other clustering models (SpectralClustering, HDBSCAN, etc.)
            model = model_cls(n_clusters=n_row, **model_kwargs)
            use_n_col = n_row
    except Exception as e:
        logger.error(f"Error building model {model_class}: {e}")
        raise
    return model, use_n_col


# ----------------------------- helpers -----------------------------


def _valid_silhouette_input(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    min_cluster_size: int = 2,
    axis_name: str = "row",
    max_report: int = 20,
) -> bool:
    """Validate inputs for silhouette computation."""
    if labels is None:
        logger.warning(f"[silhouette:{axis_name}] labels is None")
        return False

    labs, counts = np.unique(labels, return_counts=True)
    if labs.size < 2:
        logger.warning(f"[silhouette:{axis_name}] only {labs.size} cluster(s): {labs}")
        return False

    too_small = labs[counts < min_cluster_size]
    if too_small.size > 0:
        logger.warning(
            f"[silhouette:{axis_name}] clusters with <{min_cluster_size} members: {too_small}"
        )
        return False

    if not np.isfinite(X).all():
        nan_mask = ~np.isfinite(X)
        nbad = int(nan_mask.sum())
        logger.warning(
            f"[silhouette:{axis_name}] Found {nbad} invalid entries (NaN/inf)."
        )
        if nbad > 0:
            bad_rows, bad_cols = np.where(nan_mask)
            for r, c in zip(bad_rows[:max_report], bad_cols[:max_report]):
                logger.warning(f"  - Row {r}, Col {c}, Value={X[r, c]}")
        return False

    return True


def _huber_distance(u: np.ndarray, v: np.ndarray, *, delta: float = 1.0) -> float:
    """Huber dissimilarity between two vectors."""
    d = np.abs(u - v)
    mask = d <= delta
    # 0.5 * d^2  (quadratic region),  delta * (d - 0.5*delta) (linear region)
    return float(0.5 * np.sum(d[mask] ** 2) + delta * np.sum(d[~mask] - 0.5 * delta))


def _mav_ratio_distance(u: np.ndarray, v: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Symmetric MAV-normalized L1:
        mae = mean(|u - v|)
        denom = 0.5 * (mean(|u|) + mean(|v|))
        d = mae / max(denom, eps)
    """
    mae = float(np.mean(np.abs(u - v)))
    denom = 0.5 * (float(np.mean(np.abs(u))) + float(np.mean(np.abs(v))))
    return mae / max(denom, eps)


def _metric_from_norm(norm: str | None, *, huber_delta: float = 1.0):
    """
    Map training norm → silhouette distance metric (string or callable).
    """
    if norm is None or norm == "l2":
        return "euclidean"
    if norm == "l1":
        return "manhattan"
    if norm == "huber":
        # Capture delta in a closure so sklearn can call metric(u, v)
        def huber_metric(u, v, delta=huber_delta):
            return _huber_distance(u, v, delta=delta)

        return huber_metric
    if norm == "mav_ratio":

        def mav_metric(u, v, eps=1e-12):
            return _mav_ratio_distance(u, v, eps=eps)

        return mav_metric
    # Fallback
    logger.warning(f"[silhouette] Unknown norm '{norm}', falling back to euclidean.")
    return "euclidean"


def safe_silhouette(
    X: np.ndarray,
    row_labels: np.ndarray | None = None,
    col_labels: np.ndarray | None = None,
    *,
    model=None,
    norm: str | None = None,
    huber_delta: float = 1.0,
    prefer: str = "row",  # 'row' or 'col'
    min_cluster_size: int = 2,
    log_level: str = "INFO",
) -> float:
    """
    Compute silhouette using a distance metric matched to the model's norm.
    Prefer row silhouette; fall back to columns; else NaN.

    If `model` has attribute `norm`, it overrides the `norm` arg.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Resolve norm from model if available
    if hasattr(model, "norm"):
        norm = getattr(model, "norm")
    metric = _metric_from_norm(norm, huber_delta=huber_delta)

    # Try preferred axis first
    if prefer == "row":
        order = (("row", X, row_labels), ("col", X.T, col_labels))
    else:
        order = (("col", X.T, col_labels), ("row", X, row_labels))

    for axis_name, Xax, labs in order:
        if labs is None:
            continue
        if not _valid_silhouette_input(
            Xax, labs, min_cluster_size=min_cluster_size, axis_name=axis_name
        ):
            continue
        try:
            return float(silhouette_score(Xax, labs, metric=metric))
        except Exception as e:
            logger.warning(f"[silhouette:{axis_name}] failed with metric={metric}: {e}")

    logger.warning("[silhouette] not computable (degenerate labels or non-finite X)")
    return np.nan


# --- Check cluster sizes (rows & columns) ---
def _small_clusters(labels, k, min_size, *, log_level="INFO"):
    """Return indices of clusters with fewer than `min_size` members."""
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Effective k (handle None/NaN): use observed label range if k not usable
    if k is None or (isinstance(k, float) and np.isnan(k)):
        k_eff = int(np.max(labels)) + 1
    else:
        k_eff = int(k)

    counts = (
        pd.Series(labels).value_counts().reindex(range(k_eff), fill_value=0).to_numpy()
    )
    return np.where(counts < min_size)[0]


def log_cluster_sizes(labels, kind="row", k_expected=None, log_level="INFO"):
    """
    Log cluster membership counts. If k_expected is given, also show empty/missing clusters.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    labs, counts = np.unique(labels, return_counts=True)
    present = dict(zip(labs.tolist(), counts.tolist()))

    if k_expected is None:
        logger.info(f"[{kind} clusters] present={len(labs)}")
        for lab, cnt in present.items():
            logger.info(f"  - Cluster {lab}: {cnt} members")
        return

    # With k_expected: include empty clusters 0..k_expected-1
    logger.info(f"[{kind} clusters] expected={int(k_expected)}; present={len(labs)}")
    for lab in range(int(k_expected)):
        cnt = present.get(lab, 0)
        lvl = logger.warning if cnt == 0 else logger.info
        lvl(f"  - Cluster {lab}: {cnt} members")


def _canon_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Map arbitrary integer labels to 0..K-1 (stable order).
    Returns (mapped_labels, K).
    This function canonicalizes any integer labels
    to that range and returns both the mapped labels and K
    (the number of unique labels)
    """
    labs = np.asarray(labels, dtype=int)
    uniq = np.unique(labs)
    remap = {v: i for i, v in enumerate(uniq)}
    mapped = np.vectorize(remap.get)(labs)
    return mapped, uniq.size


def block_reconstruct_from_labels(
    X: np.ndarray,
    row_labels: np.ndarray,
    col_labels: Optional[np.ndarray] = None,
    *,
    stat: str = "mean",  # 'mean' (L2-consistent) or 'median' (L1-consistent)
) -> Dict[str, Any]:
    """
    Build one-hot U,V from labels, compute block means C, reconstruction Xhat, and true PVE.

    Shapes:
      - X: (I, J)
      - U: (I, P), V: (J, Q)
      - If col_labels is None (or identity), C: (P, J) and Xhat = U @ C
      - Else C: (P, Q) and Xhat = U @ C @ V.T

    Returns dict with:
      - 'U', 'V' (or None if identity), 'C', 'Xhat', 'pve',
        'row_labels', 'col_labels' (canonicalized 0..K-1), 'P', 'Q'
    """
    X = np.asarray(X)
    I, J = X.shape

    # Canonicalize labels
    row_labels, P = _canon_labels(np.asarray(row_labels))
    if col_labels is not None:
        col_labels, Q = _canon_labels(np.asarray(col_labels))
    else:
        Q = None

    # One-hot U
    U = np.eye(P, dtype=int)[row_labels]  # (I,P)

    # Detect “no real” column clustering (identity)
    no_col_clustering = (col_labels is None) or (
        Q == J and np.array_equal(col_labels, np.arange(J))
    )

    if no_col_clustering:
        logger.info("No column clustering detected.")
        # Row-only clustering: C is per-row-cluster centroid across columns
        if stat == "mean":
            counts = U.sum(axis=0, keepdims=True).T.astype(float)  # (P,1)
            C = (U.T @ X) / np.maximum(counts, 1e-12)  # (P,J)
        elif stat == "median":
            C = np.zeros((P, J), dtype=float)
            for p in range(P):
                rmask = row_labels == p
                if rmask.any():
                    C[p] = np.median(X[rmask], axis=0)
        else:
            raise ValueError("stat must be 'mean' or 'median'")

        Xhat = U @ C  # (I,J)
        V = None

    else:
        # True biclustering: build V and block means C[p,q]
        logger.info("Biclustering with column detected.")
        V = np.eye(Q, dtype=int)[col_labels]  # (J,Q)

        if stat == "mean":
            row_counts = U.sum(axis=0).astype(float)  # (P,)
            col_counts = V.sum(axis=0).astype(float)  # (Q,)
            counts = np.outer(row_counts, col_counts)  # (P,Q)
            sums = U.T @ X @ V  # (P,Q)
            C = sums / np.maximum(counts, 1e-12)
            C[counts == 0] = 0.0
        elif stat == "median":
            C = np.zeros((P, Q), dtype=float)
            for p in range(P):
                rmask = row_labels == p
                if not rmask.any():  # empty row cluster
                    continue
                Xp = X[rmask]
                for q in range(Q):
                    cmask = col_labels == q
                    if not cmask.any():  # empty col cluster
                        continue
                    blk = Xp[:, cmask]
                    if blk.size > 0:
                        C[p, q] = np.median(blk)
        else:
            raise ValueError("stat must be 'mean' or 'median'")

        Xhat = U @ C @ V.T  # (I,J)

    # True block-model PVE
    Xc = X - X.mean(axis=0, keepdims=True)
    tss = float(np.sum(Xc * Xc))
    rss = float(np.sum((X - Xhat) ** 2))
    # 1) Verify the pieces
    logger.info(f"TSS:{tss},RSS:{rss}, PVE%:{100 * (1 - rss / max(tss, 1e-12))}")

    # 2) Sanity checks
    assert tss >= 0 and rss >= 0
    # If PVE < 0 → RSS > TSS: model worse than per-column mean baseline.

    # 3) Alignment sanity (common pitfall)
    # row_labels should have len == X.shape[0]
    # col_labels should have len == X.shape[1] (if used)

    pve = 100.0 * (1.0 - rss / max(tss, 1e-12)) if tss > 0 else np.nan

    return {
        "U": U,
        "V": V,
        "C": C,
        "Xhat": Xhat,
        "pve": pve,
        "row_labels": row_labels,
        "col_labels": (None if no_col_clustering else col_labels),
        "P": P,
        "Q": (None if no_col_clustering else Q),
    }


def true_block_pve(
    X: np.ndarray,
    row_labels: np.ndarray,
    col_labels: Optional[np.ndarray] = None,
    *,
    stat: str = "mean",
) -> float:
    """Convenience wrapper: return only the true block-model PVE (%)."""
    return float(
        block_reconstruct_from_labels(X, row_labels, col_labels, stat=stat)["pve"]
    )


def compute_biclustering_scores(
    data,
    *,
    model_name: str = "SpectralBiclustering",
    model_kwargs=None,
    row_range=range(2, 6),
    col_range=range(2, 6),
    col_mav_name: str = "store_item_mav",
    col_cluster_mav_name: str = "store_cluster_item_cluster_mav",
    true_row_labels=None,  # kept for future ARI if you want it
    min_cluster_size=2,
    skip_invalid=True,
    log_level="INFO",
):
    """
    Returns a tidy DataFrame with one row per (store, item) including:
      - model / n_row / n_col
      - Explained Variance (%), Mean Silhouette
      - Within_Cluster_Var, Between_Cluster_Var, Ratio_Between_Within
      - {col_mav_name} (per (store,item))
      - {col_cluster_mav_name}_mean (mean MAV of the item's (store_cluster, item_cluster))
    Notes:
      - Assumes `mav_by_cluster` returns:
          per_store_item: columns ['store','item','store_cluster','item_cluster', col_mav_name]
          per_cluster:     columns ['store_cluster','item_cluster','n_obs',
                                    f'{col_cluster_mav_name}_mean',
                                    f'{col_cluster_mav_name}_within_var']
    """
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_state", 42)

    X = np.asarray(data)
    n_rows, n_cols = X.shape
    results = []  # collect per (store,item) rows across grid search
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for n_row in row_range:
        if n_row > n_rows:
            continue
        for n_col in col_range:
            if n_col is not None and n_col > n_cols:
                continue

            use_n_col = n_col  # reset per iteration
            try:
                model, use_n_col = get_ClusterModel(
                    model_name, n_row, n_col, **model_kwargs
                )
                # fit
                model.fit(X)

                # row labels
                if hasattr(model, "row_labels_"):
                    row_labels = model.row_labels_
                elif hasattr(model, "labels_"):
                    row_labels = model.labels_
                else:
                    # e.g., models exposing binary row-indicator matrix
                    row_labels = np.argmax(model.rows_, axis=0)

                # column labels
                if hasattr(model, "column_labels_"):
                    col_labels = model.column_labels_
                elif hasattr(model, "columns_"):
                    col_labels = np.argmax(model.columns_, axis=0)
                else:
                    # identity (no column clustering)
                    col_labels = np.arange(n_cols)

                bad_row = _small_clusters(
                    row_labels, n_row, min_cluster_size, log_level=log_level
                )
                bad_col = _small_clusters(
                    col_labels, n_col, min_cluster_size, log_level=log_level
                )
                if (bad_row.size > 0) or (bad_col.size > 0):
                    log_cluster_sizes(
                        row_labels, kind="row", k_expected=n_row, log_level=log_level
                    )
                    log_cluster_sizes(
                        col_labels, kind="col", k_expected=n_col, log_level=log_level
                    )
                    msg = (
                        f"[skip] n_row={n_row}, n_col={n_col}: "
                        f"rows<{min_cluster_size}={bad_row.tolist()} "
                        f"cols<{min_cluster_size}={bad_col.tolist()}"
                    )
                    logger.warning(msg)
                    if skip_invalid:
                        continue
                    else:
                        raise RuntimeError(msg)

                # --- Build cluster assignment tables (unchanged) ---
                store_clusters = pd.DataFrame(
                    {"store": data.index, "store_cluster": row_labels.astype(int)}
                )
                item_clusters = pd.DataFrame(
                    {"item": data.columns, "item_cluster": col_labels.astype(int)}
                )
                df_assignments = pd.DataFrame(
                    {
                        "store": np.repeat(data.index.values, len(data.columns)),
                        "item": np.tile(data.columns.values, len(data.index)),
                    }
                )

                df_assignments = df_assignments.merge(
                    store_clusters, on="store", how="left"
                ).merge(item_clusters, on="item", how="left")

                # --- MAV computation (per (store,item) and per (store_cluster,item_cluster)) ---
                per_store_item, per_cluster = mav_by_cluster(
                    df_assignments,
                    data,
                    col_mav_name=col_mav_name,
                    col_cluster_mav_name=col_cluster_mav_name,
                )

                # --- Global variance metrics across all (store,item) points ---
                global_mean = per_store_item[col_mav_name].mean()

                # Between-cluster variance (weighted by cluster size)
                if len(per_cluster) > 1:
                    between_num = (
                        (per_cluster[f"{col_cluster_mav_name}_mean"] - global_mean) ** 2
                        * per_cluster["n_obs"]
                    ).sum()
                    between_var = between_num / max(per_store_item.shape[0] - 1, 1)
                else:
                    between_var = np.nan

                # Within-cluster variance (pooled)
                if len(per_cluster) > 0:
                    within_num = (
                        per_cluster[f"{col_cluster_mav_name}_within_var"]
                        * (per_cluster["n_obs"] - 1)
                    ).sum()
                    denom = per_store_item.shape[0] - len(per_cluster)
                    within_var = within_num / denom if denom > 0 else np.nan
                else:
                    within_var = np.nan
                assert np.isfinite(within_var)
                assert np.isfinite(between_var)
                assert within_var >= 0.0
                assert between_var >= 0.0
                assert within_var <= between_var

                ratio_between_within = (
                    between_var / within_var
                    if (np.isfinite(within_var) and within_var > 0)
                    else np.nan
                )

                # ================================
                # True block-model PVE (no proxy)
                # ================================
                # Build one-hot U, V from labels
                # If `data` is a DataFrame from your pipeline:
                X = data.values

                # From any fitted model:
                row_labels = model.row_labels_  # or np.argmax(model.rows_, axis=0)
                col_labels = getattr(model, "column_labels_", None)
                pve_row = true_block_pve(X, row_labels, None, stat="mean")  # row-only
                pve_block = true_block_pve(
                    X, row_labels, col_labels, stat="mean"
                )  # tied block
                logger.info(
                    f"PVE row-only: {pve_row:.2f}%  |  PVE block: {pve_block:.2f}%"
                )

                try:
                    # --- Silhouette (use model's own labels, fallback to columns) ---
                    sil = safe_silhouette(
                        X,
                        row_labels=row_labels,
                        col_labels=col_labels,
                        model=model,  # <- pass the fitted model
                        norm=getattr(
                            model, "norm", None
                        ),  # <- explicit norm if no attribute
                        huber_delta=getattr(model, "huber_delta", 1.0),
                        prefer="row",
                        min_cluster_size=min_cluster_size,
                        log_level=log_level,
                    )

                except Exception:
                    sil = np.nan

                # --- Join cluster mean onto each (store,item) ---
                per_store_item = per_store_item.merge(
                    per_cluster[
                        [
                            "store_cluster",
                            "item_cluster",
                            f"{col_cluster_mav_name}_mean",
                        ]
                    ],
                    on=["store_cluster", "item_cluster"],
                    how="left",
                    validate="many_to_one",
                )

                # --- Attach metadata columns (same values repeated for all rows of this setting) ---
                logger.info(f"Model: {model}")
                per_store_item = per_store_item.assign(
                    n_row=int(n_row),
                    n_col=use_n_col,
                    Model=model,
                    **{
                        "Explained Variance PVE Row (%)": pve_row,
                        "Explained Variance PVE Block (%)": pve_block,
                        "Mean Silhouette": sil,
                        "Within_Cluster_Var": within_var,
                        "Between_Cluster_Var": between_var,
                        "Ratio_Between_Within": ratio_between_within,
                    },
                )

            except Exception as e:
                logger.error(f"[FAIL] n_row={n_row}, n_col={n_col} → {e}")
                # Build an empty-but-typed frame to keep schema consistent
                per_store_item = pd.DataFrame(
                    columns=[
                        "store",
                        "item",
                        "store_cluster",
                        "item_cluster",
                        col_mav_name,
                        f"{col_cluster_mav_name}_mean",
                        "n_row",
                        "n_col",
                        "Model",
                        "Explained Variance PVE Row (%)",
                        "Explained Variance PVE Block (%)",
                        "Mean Silhouette",
                        "Within_Cluster_Var",
                        "Between_Cluster_Var",
                        "Ratio_Between_Within",
                    ]
                )

            results.append(per_store_item)

    # Concatenate all settings
    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    expected_cols = [
        "Model",
        "n_row",
        "n_col",
        "Explained Variance PVE Row (%)",
        "Explained Variance PVE Block (%)",
        "Mean Silhouette",
        "store",
        "item",
        "store_cluster",
        "item_cluster",
        col_mav_name,
        f"{col_cluster_mav_name}_mean",
        "Within_Cluster_Var",
        "Between_Cluster_Var",
        "Ratio_Between_Within",
    ]
    if out.empty:
        logger.error(
            "No valid biclustering results (all skipped by min_cluster_size?)."
        )
        # Return an empty frame with expected columns so downstream code doesn’t KeyError
        logger.info(f"Returning empty frame with expected columns: {expected_cols}")
        return pd.DataFrame(columns=[c for c in expected_cols])

    cols = [c for c in expected_cols if c in out.columns] + [
        c for c in out.columns if c not in expected_cols
    ]
    return out[cols]


def pick_best_biclustering_setting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Guard: empty or None
    if df is None or df.empty:
        raise ValueError("mav_df is empty after filtering; no settings to rank.")

    df = df.copy()

    # Create ratio if missing
    if "Ratio_Between_Within" not in df.columns:
        if {"Within_Cluster_Var", "Between_Cluster_Var"}.issubset(df.columns):
            w = pd.to_numeric(df["Within_Cluster_Var"], errors="coerce")
            b = pd.to_numeric(df["Between_Cluster_Var"], errors="coerce")
            df["Ratio_Between_Within"] = np.where(
                (w > 0) & np.isfinite(w), b / w, np.nan
            )
        else:
            df["Ratio_Between_Within"] = np.nan

    # Coerce numerics & clean
    for c in [
        "Ratio_Between_Within",
        "Explained Variance PVE Row (%)",
        "Explained Variance PVE Block (%)",
        "Mean Silhouette",
        "Within_Cluster_Var",
        "Between_Cluster_Var",
        "n_row",
        "n_col",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Keep rows that have a usable ratio if any
    has_ratio = (
        df["Ratio_Between_Within"].notna()
        if "Ratio_Between_Within" in df
        else pd.Series(False, index=df.index)
    )
    d = df[has_ratio].copy() if has_ratio.any() else df.copy()

    if d.empty:
        raise ValueError("No valid rows to rank (all metrics are NaN).")

    # Desired order:
    # - Maximize (desc): Ratio, PVE Row, PVE Block, Silhouette
    # - Minimize (asc):  Within Var, Between Var, n_row, n_col
    sort_cols = [
        "Ratio_Between_Within",
        "Explained Variance PVE Row (%)",
        "Explained Variance PVE Block (%)",
        "Mean Silhouette",
        "Within_Cluster_Var",
        "Between_Cluster_Var",
        "n_row",
        "n_col",
    ]
    asc = [False, False, False, False, False, True, True, True]  # 1:1 with sort_cols
    order = dict(zip(sort_cols, asc))

    used = [(c, order[c]) for c in sort_cols if c in d.columns]
    if used:
        by, ascending = zip(*used)
        d = d.sort_values(
            list(by),
            ascending=list(ascending),
            kind="mergesort",
            na_position="last",
            ignore_index=True,
        )
    else:
        d = d.copy()

    # Best representative per (n_row, n_col)
    if not {"n_row", "n_col"}.issubset(d.columns):
        raise ValueError("Missing n_row/n_col; cannot rank settings.")

    settings_sorted = d.drop_duplicates(subset=["n_row", "n_col"], keep="first")
    if settings_sorted.empty:
        raise ValueError("No unique (n_row, n_col) settings to rank.")

    best_row = settings_sorted.iloc[0].copy()
    return settings_sorted, best_row


def cluster_data(
    df: pd.DataFrame,
    *,
    store_item_matrix_fn: Optional[Path] = None,
    mav_df_fn: Optional[Path] = None,
    only_best_model: bool = True,
    only_best_model_path: Optional[Path] = None,
    only_top_n_clusters: int = 2,
    only_top_n_clusters_path: Optional[Path] = None,
    store_fn: Optional[Path] = None,
    item_fn: Optional[Path] = None,
    output_fn: Optional[Path] = None,
    model_name="SpectralBiclustering",
    row_range: range = range(2, 5),
    col_range: range = range(2, 5),
    min_cluster_size: int = 2,
    skip_invalid: bool = True,
    model_kwargs=None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if model_kwargs is None:
        model_kwargs = {}

    # Drop old cluster columns if present
    df = df.drop(
        columns=[c for c in ["store_cluster", "item_cluster"] if c in df.columns]
    )
    original_df = df.copy()

    # Build normalized + raw matrices
    norm_data = normalize_data(
        df, log_transform=False, median_transform=False, mean_transform=True
    ).fillna(0)

    logger.info(f"Number of items: {df['item'].nunique()}")
    logger.info(f"Number of stores: {df['store'].nunique()}")

    if store_item_matrix_fn is not None:
        path = Path(store_item_matrix_fn)
        if path.is_dir() or str(path) == ".":
            path = path / "store_item_matrix.csv"
        logger.info(f"Saving store_item_matrix to {path}")
        norm_data.to_csv(path, index=False)

    # Run biclustering grid search
    mav_df = compute_biclustering_scores(
        data=norm_data,
        model_name=model_name,
        row_range=row_range,
        col_range=col_range,
        true_row_labels=None,
        min_cluster_size=min_cluster_size,
        skip_invalid=skip_invalid,
        model_kwargs=model_kwargs,
    )

    # Select best result
    try:
        _, best_row = pick_best_biclustering_setting(mav_df)
    except ValueError as e:
        logger.error(f"No valid clustering settings: {e}")
        # Optional: keep schema but no assignments
        df = original_df.copy()
        df["store_cluster"] = np.nan
        df["item_cluster"] = np.nan
        df["cluster"] = np.nan
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)
        return df

    #    _, best_row = pick_best_biclustering_setting(mav_df)

    save_mav(
        mav_df,
        mav_df_fn=mav_df_fn,
        only_best_model=only_best_model,
        only_best_model_path=only_best_model_path,
        only_top_n_clusters=only_top_n_clusters,
        only_top_n_clusters_path=only_top_n_clusters_path,
        best_row=best_row,
    )

    logger.info(f"Best clustering result: {best_row}")
    best_model = best_row["Model"]

    # Optimal row/col numbers
    n_row = int(best_row["n_row"])
    n_col = int(best_row["n_col"]) if pd.notna(best_row["n_col"]) else n_row
    logger.info(f"Optimal n_row: {n_row}, n_col: {n_col}")
    logger.info(f"Optimal n_row: {n_row}, n_col: {n_col}")
    mav_df = mav_df.query("n_row == @n_row and n_col == @n_col")

    # Extract row/column labels from model
    store_labels = best_model.row_labels_
    item_labels = best_model.column_labels_

    store_ids = norm_data.index.tolist()
    item_ids = norm_data.columns.tolist()

    # Build cluster assignment DataFrames
    store_clusters = pd.DataFrame(
        {"store": store_ids, "store_cluster": [str(lbl) for lbl in store_labels]}
    )
    item_clusters = pd.DataFrame(
        {"item": item_ids, "item_cluster": [str(lbl) for lbl in item_labels]}
    )

    # Merge cluster assignments into original df
    df = original_df.merge(store_clusters, on="store", how="left")
    df = df.merge(item_clusters, on="item", how="left")

    # Final adjustments
    df["cluster"] = df["store_cluster"] + "_" + df["item_cluster"]
    df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].astype(bool)

    if store_fn is not None:
        path = Path(store_fn)
        if path.is_dir() or str(path) == ".":
            path = path / "store_clusters.csv"
        logger.info(f"Saving store_fn to {path}")
        df[["date", "store", "store_cluster"]].drop_duplicates().to_csv(
            path, index=False
        )

    if item_fn is not None:
        path = Path(item_fn)
        if path.is_dir() or str(path) == ".":
            path = path / "item_clusters.csv"
        logger.info(f"Saving item_fn to {path}")
        df[["date", "item", "item_cluster"]].drop_duplicates().to_csv(path, index=False)

    if output_fn is not None:
        path = Path(output_fn)
        if path.is_dir() or str(path) == ".":
            path = path / "clustered_data.parquet"
        logger.info(f"Saving df to {path}")
        df.to_parquet(path)

    return df


def save_mav(
    mav_df: pd.DataFrame,
    *,
    mav_df_fn: Optional[Path] = None,
    only_best_model: bool = True,
    only_best_model_path: Optional[Path] = None,
    only_top_n_clusters: int = 2,
    only_top_n_clusters_path: Optional[Path] = None,
    best_row,
):

    if only_best_model and best_row is not None and only_best_model_path is not None:
        logger.info("Saving only best model's MAV scores")
        # Get n_row and n_col from best_row and filter mav_df
        best_n_row = best_row["n_row"]
        best_n_col = best_row["n_col"]
        df = mav_df[(mav_df["n_row"] == best_n_row) & (mav_df["n_col"] == best_n_col)]
        logger.info(f"Saving best model mav_df to {only_best_model_path}")
        df.to_csv(only_best_model_path, index=False)

    if only_top_n_clusters > 0 and only_top_n_clusters_path is not None:
        # Get the best Ratio_Between_Within for each (n_row, n_col) combination
        best_per_cluster = (
            mav_df.groupby(["n_row", "n_col"])["Ratio_Between_Within"]
            .max()
            .reset_index()
            .sort_values("Ratio_Between_Within", ascending=False)
            .head(only_top_n_clusters)[["n_row", "n_col"]]
        )
        mav_df = mav_df.merge(best_per_cluster, on=["n_row", "n_col"], how="inner")
        logger.info(
            f"Saving top {only_top_n_clusters} mav_df to {only_top_n_clusters_path}"
        )
        mav_df.to_csv(only_top_n_clusters_path, index=False)

    logger.info(f"Saving mav_df to {mav_df_fn}")
    mav_df.to_csv(mav_df_fn, index=False)
