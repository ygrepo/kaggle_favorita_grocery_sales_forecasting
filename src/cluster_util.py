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
from src.utils import normalize_store_item_matrix

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
    *,
    model_class,
    row_range=range(2, 6),
    col_range=range(2, 6),
    true_row_labels=None,
    model_kwargs=None,
    return_models=False,
):
    """
    Evaluate clustering performance using both row and column cluster settings.
    Optionally returns model and cluster assignments for plotting.
    """

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_state", 42)

    X = np.asarray(data)
    n_rows, n_cols = X.shape
    results = []

    for n_row in row_range:
        if n_row > n_rows:
            continue
        for n_col in col_range:
            if n_col is not None and n_col > n_cols:
                continue

            logger.info(f"Evaluating n_row={n_row}, n_col={n_col}")

            try:
                # Instantiate model
                if model_class.__name__ == "SpectralBiclustering":
                    model = model_class(n_clusters=(n_row, n_col), **model_kwargs)
                elif model_class.__name__ == "SpectralCoclustering":
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    n_col = np.nan
                else:
                    model = model_class(n_clusters=n_row, **model_kwargs)
                    n_col = np.nan

                model.fit(X)

                # Get row/col labels
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

                # Compute % variance explained
                global_mean = X.mean(axis=0)
                total_ss = np.sum((X - global_mean) ** 2)

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
                            [
                                (
                                    np.linalg.norm(
                                        xi - X[row_labels == cid].mean(axis=0)
                                    )
                                    if np.any(row_labels == cid)
                                    else np.inf
                                )
                                for cid in range(n_row)
                            ]
                        )
                        for xi in X
                    ]
                )

                ari = (
                    adjusted_rand_score(true_row_labels, predicted_labels)
                    if true_row_labels is not None
                    else np.nan
                )
                try:
                    sil = silhouette_score(X, predicted_labels)
                except ValueError:
                    sil = np.nan

            except Exception as e:
                logger.error(f"[FAIL] n_row={n_row}, n_col={n_col} → {e}")
                pve, sil, ari, row_labels, col_labels, model = (
                    np.nan,
                    np.nan,
                    np.nan,
                    None,
                    None,
                    None,
                )

            result = {
                "n_row": n_row,
                "n_col": n_col,
                "Explained Variance (%)": pve,
                "Mean Silhouette": sil,
                "Mean ARI": ari,
                "row_labels": row_labels,
                "col_labels": col_labels,
            }
            if return_models:
                result["model"] = model

            results.append(result)

    return pd.DataFrame(results)


def cluster_data(
    df: pd.DataFrame,
    *,
    freq: str = "W",
    store_item_matrix_fn: Path = None,
    cluster_output_fn: Path = None,
    output_fn: Path = None,
    model_class=SpectralBiclustering,
    row_range: range = range(2, 5),
    col_range: range = range(2, 5),
    model_kwargs=None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if model_kwargs is None:
        model_kwargs = {}
    norm_data = normalize_store_item_matrix(df, freq=freq)
    norm_data = norm_data.fillna(0)
    logger.info(f"Number of items: {df['item'].nunique()}")
    logger.info(f"Number of stores: {df['store'].nunique()}")
    if store_item_matrix_fn:
        logger.info(f"Saving store_item_matrix to {store_item_matrix_fn}")
        norm_data.to_csv(store_item_matrix_fn, index=False)
    cluster_df = compute_biclustering_scores(
        data=norm_data.values,
        model_class=model_class,
        row_range=row_range,
        col_range=col_range,
        true_row_labels=None,
        model_kwargs=model_kwargs,
        return_models=True,
    )
    if cluster_output_fn:
        logger.info(f"Saving cluster_df to {cluster_output_fn}")
        cluster_df.to_csv(cluster_output_fn, index=False)
    # Select the best clustering result
    if cluster_df["Explained Variance (%)"].notna().any():
        best_idx = cluster_df["Explained Variance (%)"].idxmax()
        best_row = cluster_df.loc[best_idx]
    else:
        logger.error(
            "No valid clustering results: all entries have NaN Explained Variance"
        )
        return
    logger.info(f"Best clustering result: {best_row}")
    best_model = best_row["model"]

    store_labels = best_model.row_labels_
    item_labels = best_model.column_labels_

    # Map cluster labels to actual store and item IDs
    store_ids = norm_data.index.tolist()
    item_ids = norm_data.columns.tolist()

    store_cluster_map = dict(zip(store_ids, store_labels))
    item_cluster_map = dict(zip(item_ids, item_labels))

    # Apply to original DataFrame
    df["store_cluster"] = df["store"].map(store_cluster_map)
    df["item_cluster"] = df["item"].map(item_cluster_map)
    df["cluster"] = (
        df["store_cluster"].astype(int).astype(str)
        + "_"
        + df["item_cluster"].astype(int).astype(str)
    )
    if output_fn:
        logger.info(f"Saving df to {output_fn}")
        df.to_parquet(output_fn, index=False)
    return df
