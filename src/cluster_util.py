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


def compute_biclustering_scores(
    data,
    raw_data,
    *,
    model_class,
    row_range=range(2, 6),
    col_range=range(2, 6),
    col_mav_name: str = "store_item_mav",
    col_cluster_mav_name: str = "store_cluster_item_cluster_mav",
    true_row_labels=None,
    model_kwargs=None,
):
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_state", 42)

    X = np.asarray(data)
    n_rows, n_cols = X.shape
    all_mav = pd.DataFrame()

    for n_row in row_range:
        if n_row > n_rows:
            continue
        for n_col in col_range:
            if n_col is not None and n_col > n_cols:
                continue

            logger.info(f"Evaluating n_row={n_row}, n_col={n_col}")
            try:
                # --- Fit model
                if model_class.__name__ == "SpectralBiclustering":
                    model = model_class(n_clusters=(n_row, n_col), **model_kwargs)
                elif model_class.__name__ == "SpectralCoclustering":
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    n_col = np.nan
                else:
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    n_col = np.nan

                model.fit(X)

                # --- Get row/col labels
                row_labels = (
                    model.row_labels_
                    if hasattr(model, "row_labels_")
                    else (
                        model.labels_
                        if hasattr(model, "labels_")
                        else np.argmax(model.rows_, axis=0)
                    )
                )
                col_labels = (
                    model.column_labels_
                    if hasattr(model, "column_labels_")
                    else (
                        np.argmax(model.columns_, axis=0)
                        if hasattr(model, "columns_")
                        else np.arange(n_cols)
                    )
                )

                # --- Cluster assignments
                store_clusters = pd.DataFrame(
                    {"store": raw_data.index, "store_cluster": row_labels.astype(str)}
                )
                item_clusters = pd.DataFrame(
                    {"item": raw_data.columns, "item_cluster": col_labels.astype(str)}
                )

                df_assignments = pd.DataFrame(
                    {
                        "store": raw_data.index.repeat(len(raw_data.columns)),
                        "item": np.tile(raw_data.columns, len(raw_data.index)),
                    }
                )
                logger.info("Merging cluster assignments")
                df_assignments = df_assignments.merge(
                    store_clusters, on="store", how="left"
                )
                df_assignments = df_assignments.merge(
                    item_clusters, on="item", how="left"
                )

                # --- MAV computation (per store-item + per cluster)
                per_store_item, per_cluster = mav_by_cluster(
                    df_assignments,
                    raw_data,
                    col_mav_name=col_mav_name,
                    col_cluster_mav_name=col_cluster_mav_name,
                )

                # --- Global variance metrics
                global_mean = per_store_item[col_mav_name].mean()

                between_num = (
                    (per_cluster[f"{col_cluster_mav_name}_mean"] - global_mean) ** 2
                    * per_cluster["n_obs"]
                ).sum()
                between_var = between_num / (per_store_item.shape[0] - 1)

                within_num = (
                    per_cluster[f"{col_cluster_mav_name}_within_var"]
                    * (per_cluster["n_obs"] - 1)
                ).sum()

                denom = per_store_item.shape[0] - len(per_cluster)
                within_var = within_num / denom if denom > 0 else np.nan

                ratio_between_within = (
                    between_var / within_var
                    if within_var and within_var > 0
                    else np.nan
                )

                # --- PVE, ARI, silhouette
                global_mean_vec = X.mean(axis=0)
                total_ss = np.sum((X - global_mean_vec) ** 2)

                recon_error = sum(
                    min(
                        np.linalg.norm(xi - X[row_labels == cid].mean(axis=0)) ** 2
                        for cid in range(n_row)
                        if np.any(row_labels == cid)
                    )
                    for xi in X
                )
                pve = 100 * (1 - recon_error / total_ss)

                predicted_labels = np.array(
                    [
                        np.argmin(
                            np.array(
                                (
                                    np.linalg.norm(
                                        xi - X[row_labels == cid].mean(axis=0)
                                    )
                                    if np.any(row_labels == cid)
                                    else np.inf
                                )
                                for cid in range(n_row)
                            )
                        )
                        for xi in X
                    ]
                )

                # ari = (
                #     adjusted_rand_score(true_row_labels, predicted_labels)
                #     if true_row_labels is not None
                #     else np.nan
                # )
                try:
                    sil = silhouette_score(X, predicted_labels)
                except ValueError:
                    sil = np.nan

            except Exception as e:
                logger.error(f"[FAIL] n_row={n_row}, n_col={n_col} → {e}")
                pve, sil, row_labels, col_labels, model = (
                    np.nan,
                    np.nan,
                    None,
                    None,
                    None,
                )
                within_var, between_var, ratio_between_within = np.nan, np.nan, np.nan
                per_store_item = pd.DataFrame()
                per_cluster = pd.DataFrame()

            # --- Attach metadata
            per_store_item["n_row"] = n_row
            per_store_item["n_col"] = n_col
            per_store_item["Model"] = model
            per_store_item["Explained Variance (%)"] = pve
            per_store_item["Mean Silhouette"] = sil
            per_store_item["Within_Cluster_Var"] = within_var
            per_store_item["Between_Cluster_Var"] = between_var
            per_store_item["Ratio_Between_Within"] = ratio_between_within

            all_mav = pd.concat([all_mav, per_store_item], ignore_index=True)
            MAV_COLUMNS = [
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
                "Within_Cluster_Var",
                "Between_Cluster_Var",
                "Ratio_Between_Within",
            ]
            all_mav = all_mav[MAV_COLUMNS]

    return pd.DataFrame(all_mav)


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
    if mav_df["Explained Variance (%)"].notna().any():
        best_idx = mav_df["Explained Variance (%)"].idxmax()
        best_row = mav_df.loc[best_idx]
    else:
        logger.error(
            "No valid clustering results: all entries have NaN Explained Variance"
        )
        return pd.DataFrame()

    logger.info(f"Best clustering result: {best_row}")
    best_model = best_row["Model"]

    # Optimal row/col numbers
    n_row = int(best_row["n_row"])
    n_col = int(best_row["n_col"]) if best_row["n_col"] is not None else n_row
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
