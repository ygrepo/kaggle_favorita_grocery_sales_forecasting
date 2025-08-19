import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import (
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
    HDBSCAN,
)
from pathlib import Path
from src.data_utils import normalize_data, mav_by_cluster, median_mean_transform
from typing import Optional

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_spectral_clustering_cv_scores(
    data,
    *,
    model_class=SpectralBiclustering,
    n_clusters_row_range=range(2, 6),
    cv_folds=3,
    true_row_labels=None,
    model_kwargs=None,
):
    """Cross‑validate Spectral[Bic|Co]clustering and SpectralClustering."""

    def _safe_mean(arr):
        return np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_state", 42)

    X = np.asarray(data)
    n_rows, n_cols = X.shape
    results = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ───────────────────────────────────────────────────────────────────
    # 0.  *NEW* — find the *smallest* training‑fold size
    # ───────────────────────────────────────────────────────────────────
    min_train_samples = int(np.floor(n_rows * (cv_folds - 1) / cv_folds))
    max_row_clusters = min(n_rows, min_train_samples)
    n_clusters_row_range = [k for k in n_clusters_row_range if k <= max_row_clusters]

    # Decide how many loop dimensions we actually have
    if model_class is SpectralClustering:
        col_range = [None]
        loop_over_col = False
    elif model_class is SpectralCoclustering:
        col_range = [None]
        loop_over_col = False
    else:  # SpectralBiclustering
        col_range = [None]
        #        col_range = [c for c in n_clusters_col_range if c <= n_cols]
        loop_over_col = False

    # ───────────────────────────────────────────────────────────────────
    # 1.  Grid search
    # ───────────────────────────────────────────────────────────────────
    for n_row in n_clusters_row_range:
        logger.info(f"Evaluating n_row={n_row}")
        for n_col in col_range:
            msg = f"Evaluating n_row={n_row}"
            if loop_over_col:
                msg += f", n_col={n_col}"
            logger.info(msg)

            pve_list, sil_list, ari_list = [], [], []

            for train_idx, test_idx in kf.split(X):

                logger.info(f"\tFold {kf.get_n_splits()}")
                # *NEW* — skip this fold if it has too few samples
                if n_row > len(train_idx):
                    pve_list.append(np.nan)
                    sil_list.append(np.nan)
                    ari_list.append(np.nan)
                    continue

                X_train, X_test = X[train_idx], X[test_idx]

                try:
                    # Instantiate model
                    if model_class is SpectralClustering:
                        model = model_class(n_clusters=n_row, **model_kwargs)
                    elif model_class is SpectralCoclustering:
                        model = model_class(n_clusters=n_row, **model_kwargs)
                    else:  # SpectralBiclustering
                        n_col_eff = n_row if n_col is None else n_col
                        model = model_class(
                            n_clusters=(n_row, n_col_eff), **model_kwargs
                        )

                    model.fit(X_train)

                    # Row labels
                    if hasattr(model, "row_labels_"):
                        row_labels = model.row_labels_
                    elif hasattr(model, "labels_"):
                        row_labels = model.labels_
                    else:  # SpectralCoclustering
                        row_labels = np.argmax(model.rows_, axis=0)

                    # % Variance explained
                    global_mean = X_train.mean(axis=0)
                    total_ss = np.sum((X_test - global_mean) ** 2)

                    recon_error = 0.0
                    for xi in X_test:
                        best_err = np.inf
                        for cid in range(n_row):
                            mask = row_labels == cid
                            if np.any(mask):
                                centroid = X_train[mask].mean(axis=0)
                                best_err = min(
                                    best_err,
                                    np.linalg.norm(xi - centroid) ** 2,
                                )
                        recon_error += best_err
                    pve_list.append(100 * (1 - recon_error / total_ss))

                    # Predicted labels for test rows
                    test_labels = np.array(
                        [
                            np.argmin(
                                [
                                    (
                                        np.linalg.norm(
                                            xi - X_train[row_labels == cid].mean(axis=0)
                                        )
                                        if np.any(row_labels == cid)
                                        else np.inf
                                    )
                                    for cid in range(n_row)
                                ]
                            )
                            for xi in X_test
                        ]
                    )

                    # ARI
                    if true_row_labels is not None:
                        ari_list.append(
                            adjusted_rand_score(true_row_labels[test_idx], test_labels)
                        )

                    # Silhouette
                    try:
                        sil_list.append(silhouette_score(X_test, test_labels))
                    except ValueError:
                        sil_list.append(np.nan)

                except Exception as e:
                    fail_msg = f"[FAIL] n_row={n_row}"
                    if loop_over_col:
                        fail_msg += f", n_col={n_col}"
                    logger.error(f"{fail_msg} → {e}")
                    pve_list.append(np.nan)
                    sil_list.append(np.nan)
                    ari_list.append(np.nan)

            results.append(
                {
                    "n_row": n_row,
                    "n_col": n_col if loop_over_col else np.nan,
                    "Explained Variance (%)": _safe_mean(pve_list),
                    "Mean Silhouette": _safe_mean(sil_list),
                    "Mean ARI": _safe_mean(ari_list),
                }
            )

    return pd.DataFrame(results)


def _valid_silhouette_input(X: np.ndarray, labels: np.ndarray) -> bool:
    if labels is None:
        return False
    labs, counts = np.unique(labels, return_counts=True)
    if len(labs) < 2:
        return False
    if (counts < 2).any():
        return False
    if not np.isfinite(X).all():
        return False
    return True


def safe_silhouette(
    X: np.ndarray,
    row_labels: np.ndarray | None = None,
    col_labels: np.ndarray | None = None,
    log_level="INFO",
) -> float:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    """Prefer row silhouette; fall back to columns; else NaN."""
    # rows first
    if row_labels is not None and _valid_silhouette_input(X, row_labels):
        try:
            return float(silhouette_score(X, row_labels))
        except Exception as e:
            logger.warning(f"[silhouette] row failed: {e}")
    # columns fallback
    if col_labels is not None and _valid_silhouette_input(X.T, col_labels):
        try:
            return float(silhouette_score(X.T, col_labels))
        except Exception as e:
            logger.warning(f"[silhouette] col failed: {e}")
    logger.warning("[silhouette] not computable (degenerate labels or non-finite X)")
    return np.nan


def compute_biclustering_scores(
    data,
    raw_data,
    *,
    model_class,
    row_range=range(2, 6),
    col_range=range(2, 6),
    col_mav_name: str = "store_item_mav",
    col_cluster_mav_name: str = "store_cluster_item_cluster_mav",
    true_row_labels=None,  # kept for future ARI if you want it
    model_kwargs=None,
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
            logger.info(f"Evaluating n_row={n_row}, n_col={n_col}")
            try:
                # --- Build the model with the right n_clusters ---
                if model_class.__name__ == "SpectralBiclustering":
                    model = model_class(n_clusters=(n_row, n_col), **model_kwargs)
                elif model_class.__name__ == "SpectralCoclustering":
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    # n_col not defined for this estimator
                    use_n_col = np.nan
                else:
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    use_n_col = np.nan

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

                # if not set above, keep explicit n_col value
                if "use_n_col" not in locals():
                    use_n_col = n_col

                # --- Build cluster assignment tables ---
                store_clusters = pd.DataFrame(
                    {"store": raw_data.index, "store_cluster": row_labels.astype(int)}
                )
                item_clusters = pd.DataFrame(
                    {"item": raw_data.columns, "item_cluster": col_labels.astype(int)}
                )

                df_assignments = pd.DataFrame(
                    {
                        "store": np.repeat(
                            raw_data.index.values, len(raw_data.columns)
                        ),
                        "item": np.tile(raw_data.columns.values, len(raw_data.index)),
                    }
                )
                df_assignments = df_assignments.merge(
                    store_clusters, on="store", how="left"
                ).merge(item_clusters, on="item", how="left")

                # --- MAV computation (per (store,item) and per (store_cluster,item_cluster)) ---
                per_store_item, per_cluster = mav_by_cluster(
                    df_assignments,
                    raw_data,
                    col_mav_name=col_mav_name,
                    col_cluster_mav_name=col_cluster_mav_name,
                )
                # Expect per_store_item to include: store, item, store_cluster, item_cluster, col_mav_name
                # Expect per_cluster to include: store_cluster, item_cluster, n_obs,
                #                                f'{col_cluster_mav_name}_mean',
                #                                f'{col_cluster_mav_name}_within_var'

                # --- Global variance metrics across all (store,item) points ---
                global_mean = per_store_item[col_mav_name].mean()

                # Between-cluster variance (weighted by cluster size)
                if len(per_cluster) > 1:
                    between_num = (
                        (per_cluster[f"{col_cluster_mav_name}_mean"] - global_mean) ** 2
                        * per_cluster["n_obs"]
                    ).sum()
                    # df = N - 1
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

                if np.isnan(within_var) or within_var <= 0:
                    ratio_between_within = np.nan
                else:
                    ratio_between_within = between_var / within_var

                # --- PVE (by nearest-row-centroid reconstruction) & Silhouette on rows ---
                global_mean_vec = X.mean(axis=0)
                total_ss = float(np.sum((X - global_mean_vec) ** 2))
                # row centroids
                row_centroids = []
                for cid in range(n_row):
                    mask = row_labels == cid
                    if np.any(mask):
                        row_centroids.append(X[mask].mean(axis=0))
                    else:
                        row_centroids.append(None)

                # reconstruction error: nearest centroid
                recon_error = 0.0
                for xi in X:
                    dists = [
                        np.linalg.norm(xi - c) if c is not None else np.inf
                        for c in row_centroids
                    ]
                    recon_error += float(np.min(dists) ** 2)
                pve = (
                    100.0 * (1.0 - (recon_error / total_ss)) if total_ss > 0 else np.nan
                )

                # predicted labels by nearest centroid (for silhouette)
                predicted_labels = np.array(
                    [
                        np.argmin(
                            np.array(
                                np.linalg.norm(xi - c) if c is not None else np.inf
                                for c in row_centroids
                            )
                        )
                        for xi in X
                    ],
                    dtype=int,
                )

                try:
                    # --- Silhouette (use model's own labels, fallback to columns) ---
                    sil = safe_silhouette(
                        X,
                        row_labels=row_labels,
                        col_labels=col_labels,
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
                per_store_item = per_store_item.assign(
                    n_row=int(n_row),
                    n_col=use_n_col,
                    Model=model,
                    **{
                        "Explained Variance (%)": pve,
                        "Mean Silhouette": sil,
                        "Within_Cluster_Var": within_var,
                        "Between_Cluster_Var": between_var,
                        "Ratio_Between_Within": ratio_between_within,
                    },
                )

            except Exception as e:
                _log(f"[FAIL] n_row={n_row}, n_col={n_col} → {e}")
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
                        "Explained Variance (%)",
                        "Mean Silhouette",
                        "Within_Cluster_Var",
                        "Between_Cluster_Var",
                        "Ratio_Between_Within",
                    ]
                )

            results.append(per_store_item)

    # Concatenate all settings
    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    # Order columns nicely if present
    pref_order = [
        "Model",
        "n_row",
        "n_col",
        "Explained Variance (%)",
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
    cols = [c for c in pref_order if c in out.columns] + [
        c for c in out.columns if c not in pref_order
    ]
    return out[cols]


def pick_best_biclustering_setting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pick best biclustering setting by lexicographic priority:
      1) Ratio_Between_Within (max)
      2) Explained Variance (%) (max)
      3) Mean Silhouette (max)
      4) Within_Cluster_Var (min)
      5) Between_Cluster_Var (max)
      6) n_row (max)
      7) n_col (max)
    Does not sort/group by Model, but keeps it.
    """
    # Ensure numeric types
    num_cols = [
        "Ratio_Between_Within",
        "Explained Variance (%)",
        "Mean Silhouette",
        "Within_Cluster_Var",
        "Between_Cluster_Var",
        "n_row",
        "n_col",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ratio_Between_Within"])

    # Sort whole df by priority
    d = df.sort_values(
        [
            "Ratio_Between_Within",
            "Explained Variance (%)",
            "Mean Silhouette",
            "Within_Cluster_Var",
            "Between_Cluster_Var",
            "n_row",
            "n_col",
        ],
        ascending=[False, False, False, True, False, False, False],
        kind="mergesort",
    )

    # Representative row per (n_row, n_col): the best one by this sort
    per_setting = d.drop_duplicates(subset=["n_row", "n_col"], keep="first")

    # Now rank the settings themselves again by the same priority
    settings_sorted = per_setting.sort_values(
        [
            "Ratio_Between_Within",
            "Explained Variance (%)",
            "Mean Silhouette",
            "Within_Cluster_Var",
            "Between_Cluster_Var",
            "n_row",
            "n_col",
        ],
        ascending=[False, False, False, True, False, False, False],
        kind="mergesort",
    )

    best_row = settings_sorted.iloc[0].copy()
    return settings_sorted, best_row


def cluster_data(
    df: pd.DataFrame,
    *,
    freq: str = "W",
    store_item_matrix_fn: Optional[Path] = None,
    mav_df_fn: Optional[Path] = None,
    store_fn: Optional[Path] = None,
    item_fn: Optional[Path] = None,
    output_fn: Optional[Path] = None,
    model_class=SpectralBiclustering,
    row_range: range = range(2, 5),
    col_range: range = range(2, 5),
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
        df, freq=freq, log_transform=True, median_transform=False, mean_transform=True
    ).fillna(0)
    raw_data = median_mean_transform(
        df,
        freq=freq,
        median_transform=False,
        mean_transform=True,
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
        raw_data=raw_data,
        model_class=model_class,
        row_range=row_range,
        col_range=col_range,
        true_row_labels=None,
        model_kwargs=model_kwargs,
    )

    if mav_df_fn is not None:
        path = Path(mav_df_fn)
        if path.is_dir() or str(path) == ".":
            path = path / "mav_df.csv"
        logger.info(f"Saving mav_df to {path}")
        mav_df.to_csv(path, index=False)

    # Select best result
    _, best_row = pick_best_biclustering_setting(mav_df)

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
