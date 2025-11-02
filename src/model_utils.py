import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.strategies import DeepSpeedStrategy
from sklearn.preprocessing import RobustScaler
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from typing import (
    List,
    Tuple,
    Dict,
    Protocol,
    runtime_checkable,
    Optional,
    Union,
)
import pickle
from pathlib import Path
from collections import defaultdict
import re
from tqdm.auto import tqdm
import time
from contextlib import contextmanager
from xgboost.callback import TrainingCallback


from lightning.pytorch.loggers import CSVLogger
from src.model import LightningWrapper, ModelType
from src.model import (
    model_factory,
    init_weights,
    model_factory_from_str,
)
from src.data_utils import (
    sort_df,
    build_feature_and_label_cols,
    get_X_feature_idx,
    get_y_idx,
    WEIGHT_COLUMN,
    META_FEATURES,
    X_CYCLICAL_FEATURES,
    X_FEATURES,
)

# Set up logger
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    return accelerator


def select_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref.startswith("cuda"):
        return (
            torch.device(pref)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    if pref == "mps":
        return (
            torch.device("mps")
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def is_gpu_available():
    return torch.cuda.is_available()


class StoreItemDataset(Dataset):
    def __init__(
        self, df, store_item_id, feature_cols, target_col, weight_col
    ):
        self.store_df = df[df["store_item"] == store_item_id].reset_index(
            drop=True
        )
        self.X = torch.tensor(
            self.store_df[feature_cols].values, dtype=torch.float32
        )
        self.y = torch.tensor(
            self.store_df[target_col].values, dtype=torch.float32
        ).unsqueeze(1)
        self.w = torch.tensor(
            self.store_df[weight_col].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.store_df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


@runtime_checkable
class InverseTransformer(Protocol):
    def inverse_transform(self, X: ArrayLike, /) -> np.ndarray: ...


def inverse_transform(
    arr: ArrayLike, scaler: InverseTransformer
) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr).ravel()


def fit_rf_with_tqdm(
    model,
    X,
    y,
    *,
    step=50,
    desc="RandomForest",
    patience=2,
    min_delta=1e-4,
    time_budget_s=None,
    leave=False,
    sample_weight=None,
):
    """
    Incrementally grows a RandomForest with a tqdm bar and OOB early stopping.

    - Requires bootstrap=True and oob_score=True.
    - Only increases n_estimators; uses warm_start to add trees in chunks.
    - Early-stops when OOB R^2 stops improving by > min_delta for 'patience' chunks.
    """
    # ensure bootstrap/OOB/warm_start
    if not getattr(model, "bootstrap", True):
        model.set_params(bootstrap=True)
    if not getattr(model, "oob_score", False):
        model.set_params(oob_score=True)
    model.set_params(warm_start=True)

    # target cap = current n_estimators (or default), then start from 0
    target_n = int(getattr(model, "n_estimators", 0)) or 600
    model.set_params(n_estimators=0)
    n_now = 0

    pbar = tqdm(total=target_n, desc=desc, leave=leave)
    start = time.time()
    best_oob, best_n, bad = -np.inf, 0, 0

    while n_now < target_n:
        n_next = min(n_now + step, target_n)
        model.set_params(n_estimators=n_next)

        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)

        curr_oob = getattr(model, "oob_score_", None)
        pbar.update(n_next - n_now)
        if curr_oob is not None and np.isfinite(curr_oob):
            pbar.set_postfix(
                oob=f"{curr_oob:.4f}", best=f"{best_oob:.4f}", trees=n_next
            )

        n_now = n_next

        if curr_oob is not None and np.isfinite(curr_oob):
            if curr_oob > best_oob + min_delta:
                best_oob, best_n, bad = curr_oob, n_now, 0
            else:
                bad += 1

        if (
            time_budget_s is not None
            and (time.time() - start) >= time_budget_s
        ):
            break
        if bad >= patience:
            break

    pbar.close()

    # trim back to best_n trees (optional)
    if hasattr(model, "estimators_") and best_n and best_n < n_now:
        model.estimators_ = model.estimators_[:best_n]
        model.n_estimators = best_n

    model.set_params(warm_start=False)
    return model


def fit_gbr_with_tqdm(model, X, y, step=25, desc="GBM"):
    # GradientBoostingRegressor supports warm_start too
    assert hasattr(model, "n_estimators")
    n_final = model.n_estimators
    model.set_params(warm_start=True)

    pbar = tqdm(total=n_final, desc=desc)
    n_now = 0
    while n_now < n_final:
        n_next = min(n_now + step, n_final)
        model.set_params(n_estimators=n_next)
        model.fit(X, y)
        pbar.update(n_next - n_now)
        n_now = n_next
    pbar.close()
    model.set_params(warm_start=False)
    return model


def fit_hgb_with_tqdm(model, X, y, step=20, desc="HistGBM"):
    # HistGradientBoostingRegressor grows by max_iter
    assert hasattr(model, "max_iter")
    iters = model.max_iter
    model.set_params(warm_start=True)

    pbar = tqdm(total=iters, desc=desc)
    n_now = 0
    while n_now < iters:
        n_next = min(n_now + step, iters)
        model.set_params(max_iter=n_next)
        model.fit(X, y)
        pbar.update(n_next - n_now)
        n_now = n_next
    pbar.close()
    model.set_params(warm_start=False)
    return model


@contextmanager
def spinner(desc="fitting"):
    p = tqdm(total=1, desc=desc)
    try:
        yield
    finally:
        p.update(1)
        p.close()


class XGBoostTQDMCallback(TrainingCallback):
    def __init__(
        self, total: int, desc: str = "XGB train", leave: bool = False
    ):
        self.total = total
        self.desc = desc
        self.leave = leave
        self.pbar = None

    def before_training(self, model):
        self.pbar = tqdm(total=self.total, desc=self.desc, leave=self.leave)
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        # Optionally show last RMSE on val if present
        try:
            rmse = evals_log["val"]["rmse"][-1]
            self.pbar.set_postfix(val_rmse=f"{rmse:.4f}", it=epoch + 1)
        except Exception:
            pass
        self.pbar.update(1)
        return False  # don't stop training

    def after_training(self, model):
        if self.pbar is not None:
            self.pbar.close()
        return model


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────
def create_X_y_dataset(
    df: pd.DataFrame,
    *,
    val_horizon: int = 30,
    test_horizon: int = 30,
    y_col: str = "y",
    weight_col: str = "weight",
    x_cols: list[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    RobustScaler,
    tuple[float, float],
]:
    """
    Returns:
      X_train, X_val, X_test,
      y_train, y_val, y_test,
      w_train, w_val, w_test,
      x_scaler, y_scaler, (y_clip_lower, y_clip_upper)
    """

    assert "date" in df.columns, "`df` must have a 'date' column (datetime64)"
    if x_cols is None:
        raise ValueError("Provide x_cols (list of feature column names).")

    # ---- Cutoffs (inclusive test span, inclusive val span) ----
    cutoff_test = df["date"].max() - pd.Timedelta(days=test_horizon - 1)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_horizon)

    train_mask = df["date"] < cutoff_val
    val_mask = (df["date"] >= cutoff_val) & (df["date"] < cutoff_test)
    test_mask = df["date"] >= cutoff_test
    valid = df[y_col].notna()
    train_mask &= valid
    val_mask &= valid
    test_mask &= valid

    # ---- Work in pandas until after splitting (keeps indices aligned) ----
    X_df = df.loc[:, x_cols]
    y_s = df.loc[:, y_col].astype(float)  # Series
    w_s = (
        df.loc[:, weight_col].fillna(1.0)
        if weight_col in df
        else pd.Series(1.0, index=df.index)
    )

    # ---- Compute clip bounds on *training only* (avoid leakage) ----
    q_low, q_high = y_s[train_mask].quantile([0.01, 0.99])
    y_clip_lower, y_clip_upper = float(np.floor(q_low)), float(np.ceil(q_high))

    y_clipped = y_s.clip(lower=y_clip_lower, upper=y_clip_upper)

    # ---- Split BEFORE scaling; then fit scalers on train only ----
    X_train_df, X_val_df, X_test_df = (
        X_df[train_mask],
        X_df[val_mask],
        X_df[test_mask],
    )

    med = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(med)
    X_val_df = X_val_df.fillna(med)
    X_test_df = X_test_df.fillna(med)

    y_train_s, y_val_s, y_test_s = (
        y_clipped[train_mask],
        y_clipped[val_mask],
        y_clipped[test_mask],
    )
    w_train_s, w_val_s, w_test_s = (
        w_s[train_mask],
        w_s[val_mask],
        w_s[test_mask],
    )

    # ---- Feature scaler ----
    x_scaler = MinMaxScaler().fit(X_train_df.values)
    X_train = x_scaler.transform(X_train_df.values).astype(np.float32)
    X_val = x_scaler.transform(X_val_df.values).astype(np.float32)
    X_test = x_scaler.transform(X_test_df.values).astype(np.float32)

    # ---- Target scaler (Robust) — fit on train only; keep 2D shape ----
    y_scaler = RobustScaler().fit(y_train_s.values.reshape(-1, 1))
    y_train = (
        y_scaler.transform(y_train_s.values.reshape(-1, 1))
        .astype(np.float32)
        .ravel()
    )
    y_val = (
        y_scaler.transform(y_val_s.values.reshape(-1, 1))
        .astype(np.float32)
        .ravel()
    )
    y_test = (
        y_scaler.transform(y_test_s.values.reshape(-1, 1))
        .astype(np.float32)
        .ravel()
    )

    # ---- Weights ----
    w_train = w_train_s.values.astype(np.float32)
    w_val = w_val_s.values.astype(np.float32)
    w_test = w_test_s.values.astype(np.float32)

    assert (
        np.isfinite(y_train).all()
        and np.isfinite(y_val).all()
        and np.isfinite(y_test).all()
    )
    assert X_train.shape[0] == y_train.shape[0] == w_train.shape[0]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        w_train,
        w_val,
        w_test,
        x_scaler,
        y_scaler,
        (y_clip_lower, y_clip_upper),
    )


def generate_loaders(
    df: pd.DataFrame,
    window_size: int,
    dataloader_dir: Path,
    scalers_dir: Path,
    *,
    val_horizon: int = 30,
    batch_size: int = 32,
    num_workers: int = 15,
    log_level: str = "INFO",
):
    """
    Generate DataLoaders row-aligned with original df.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    dataloader_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare DataFrame ---
    df = sort_df(df, window_size=window_size)

    df["store_cluster"] = pd.to_numeric(df["store_cluster"], errors="coerce")
    df["item_cluster"] = pd.to_numeric(df["item_cluster"], errors="coerce")
    store_cluster = df["store_cluster"].unique()
    item_cluster = df["item_cluster"].unique()
    assert len(store_cluster) == 1
    assert len(item_cluster) == 1
    store_cluster, item_cluster = store_cluster[0], item_cluster[0]
    cluster_key = f"{store_cluster}_{item_cluster}"

    logger.info(f"Unique store items: {df['store_item'].nunique()}")
    logger.info(
        f"Preparing loaders for cluster {cluster_key} with {len(df)} rows"
    )

    if WEIGHT_COLUMN not in df.columns:
        df[WEIGHT_COLUMN] = 1.0

    # --- Compute global cutoff date ---
    cutoff_date = df["start_date"].max() - pd.Timedelta(days=val_horizon - 1)

    # --- Split masks ---
    train_mask = df["start_date"] < cutoff_date
    val_mask = df["start_date"] >= cutoff_date

    logger.info(f"Global validation cutoff date: {cutoff_date.date()}")
    logger.info(f"Train rows: {train_mask.sum()}, Val rows: {val_mask.sum()}")

    features = build_feature_and_label_cols(window_size=window_size)

    # --- Early exit if no valid samples ---
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        logger.warning("No valid samples for train/val split")
        empty = torch.empty(
            (0, len(features[X_FEATURES])), dtype=torch.float32
        )
        empty_y = torch.empty((0, len(features[LABELS])), dtype=torch.float32)
        empty_w = torch.empty((0, 1), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(empty, empty_y, empty_w),
            batch_size=batch_size,
            num_workers=0,
        )
        meta_df = pd.DataFrame(columns=features[META_FEATURES])
        for split in ["train", "val"]:
            torch.save(
                loader, dataloader_dir / f"{cluster_key}_{split}_loader.pt"
            )
            meta_df.to_parquet(
                dataloader_dir / f"{cluster_key}_{split}_meta.parquet"
            )
        return loader, loader

    # --- Features & labels ---
    X = df[features[X_FEATURES]].fillna(0).values.astype(np.float32)
    Y = df[features[LABELS]].fillna(0).values.astype(np.float32)
    W = df[[WEIGHT_COLUMN]].values.astype(np.float32)

    X_train, X_val = X[train_mask], X[val_mask]
    Y_train, Y_val = Y[train_mask], Y[val_mask]
    W_train, W_val = W[train_mask], W[val_mask]

    meta_train_df = df.loc[train_mask, features[META_FEATURES]].reset_index(
        drop=True
    )
    meta_val_df = df.loc[val_mask, features[META_FEATURES]].reset_index(
        drop=True
    )

    # --- Transform & Scale ---
    idx_features = get_X_feature_idx(window_size)
    idy_features = get_y_idx(window_size)

    def _transform_xy(X_data, Y_data):
        x_log1p = np.log1p(
            np.clip(X_data[:, idx_features[X_TO_LOG_FEATURES]], 0, None)
        )
        x_log_raw = X_data[:, idx_features[X_LOG_FEATURES]]
        x_cyc = X_data[:, idx_features[X_CYCLICAL_FEATURES]]
        y_transformed = np.hstack(
            [
                np.log1p(
                    np.clip(
                        Y_data[:, idy_features[Y_TO_LOG_FEATURES]], 0, None
                    )
                ),
                Y_data[:, idy_features[Y_LOG_FEATURES]],
            ]
        )
        return x_log1p, x_log_raw, x_cyc, y_transformed

    x_log1p_train, x_log_raw_train, x_cyc_train, Y_train = _transform_xy(
        X_train, Y_train
    )
    x_log1p_val, x_log_raw_val, x_cyc_val, Y_val = _transform_xy(X_val, Y_val)

    # Fit scalers on train
    scaler_log1p = MinMaxScaler().fit(x_log1p_train)
    scaler_log_raw = MinMaxScaler().fit(x_log_raw_train)
    scaler_cyc = MinMaxScaler().fit(x_cyc_train)

    # Apply scalers, keeping original feature order
    def scale_x(x_log1p, x_log_raw, x_cyc):
        X_scaled = np.zeros(
            (x_log1p.shape[0], len(idx_features[X_FEATURES])), dtype=np.float32
        )
        X_scaled[:, idx_features[X_TO_LOG_FEATURES]] = scaler_log1p.transform(
            x_log1p
        )
        X_scaled[:, idx_features[X_LOG_FEATURES]] = scaler_log_raw.transform(
            x_log_raw
        )
        X_scaled[:, idx_features[X_CYCLICAL_FEATURES]] = scaler_cyc.transform(
            x_cyc
        )
        return X_scaled

    X_train_scaled = scale_x(x_log1p_train, x_log_raw_train, x_cyc_train)
    X_val_scaled = scale_x(x_log1p_val, x_log_raw_val, x_cyc_val)

    # --- Sanity logger ---
    for group_name, idxs in [
        ("X_TO_LOG_FEATURES", idx_features[X_TO_LOG_FEATURES]),
        ("X_LOG_FEATURES", idx_features[X_LOG_FEATURES]),
        ("X_CYCLICAL_FEATURES", idx_features[X_CYCLICAL_FEATURES]),
    ]:
        min_val = X_train_scaled[:, idxs].min()
        max_val = X_train_scaled[:, idxs].max()
        logger.info(
            f"[Sanity] {group_name} scaled range: min={min_val:.4f}, max={max_val:.4f}"
        )

    # --- Create DataLoaders ---
    persistent = num_workers > 0
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled),
            torch.tensor(Y_train),
            torch.tensor(W_train),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_scaled),
            torch.tensor(Y_val),
            torch.tensor(W_val),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
    )

    # --- Save loaders and meta ---
    for split, loader, meta_df in [
        ("train", train_loader, meta_train_df),
        ("val", val_loader, meta_val_df),
    ]:
        torch.save(loader, dataloader_dir / f"{cluster_key}_{split}_loader.pt")
        meta_df.to_parquet(
            dataloader_dir / f"{cluster_key}_{split}_meta.parquet"
        )

    # --- Save scalers ---
    for scaler, name in zip(
        [scaler_log1p, scaler_log_raw, scaler_cyc],
        ["x_log1p", "x_log_raw", "x_cyc"],
    ):
        fn = scalers_dir / f"{cluster_key}_{name}_scaler.pkl"
        with open(fn, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler {name} for {cluster_key} to {fn}")

    logger.info(
        f"Saved loaders and scalers for store_cluster={store_cluster}, item_cluster={item_cluster}"
    )
    return train_loader, val_loader


def dataloader_to_dataframe(
    loader_path: Path,
    meta_path: Path,
    scaler_dir: Path,
    window_size: int,
    *,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # --- Load DataLoader ---
    data = torch.load(loader_path)
    loader = DataLoader(data.dataset, batch_size=32)

    all_x, all_y, all_w = [], [], []
    with torch.no_grad():
        for xb, yb, wb in loader:
            all_x.append(xb.cpu())
            all_y.append(yb.cpu())
            all_w.append(wb.cpu())

    X = torch.cat(all_x, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()
    W = torch.cat(all_w, dim=0).numpy()

    # --- Load scalers ---
    stem = loader_path.stem
    cluster_key = "_".join(stem.split("_")[:2])
    x_log1p_scaler = pickle.load(
        (scaler_dir / f"{cluster_key}_x_log1p_scaler.pkl").open("rb")
    )
    x_log_raw_scaler = pickle.load(
        (scaler_dir / f"{cluster_key}_x_log_raw_scaler.pkl").open("rb")
    )
    x_cyc_scaler = pickle.load(
        (scaler_dir / f"{cluster_key}_x_cyc_scaler.pkl").open("rb")
    )

    idx_features = get_X_feature_idx(window_size)

    # --- Inverse transform ---
    x_log1p_scaled = X[:, idx_features[X_TO_LOG_FEATURES]]
    x_log_scaled = X[:, idx_features[X_LOG_FEATURES]]
    x_cyc_scaled = X[:, idx_features[X_CYCLICAL_FEATURES]]

    x_tolog_orig = np.expm1(x_log1p_scaler.inverse_transform(x_log1p_scaled))
    x_log_orig = x_log_raw_scaler.inverse_transform(x_log_scaled)
    x_cyc_orig = x_cyc_scaler.inverse_transform(x_cyc_scaled)

    X_orig = np.zeros(
        (X.shape[0], len(idx_features[X_FEATURES])), dtype=np.float32
    )
    for i, idx in enumerate(idx_features[X_TO_LOG_FEATURES]):
        X_orig[:, idx] = x_tolog_orig[:, i]
    for i, idx in enumerate(idx_features[X_LOG_FEATURES]):
        X_orig[:, idx] = x_log_orig[:, i]
    for i, idx in enumerate(idx_features[X_CYCLICAL_FEATURES]):
        X_orig[:, idx] = x_cyc_orig[:, i]

    idy_features = get_y_idx(window_size)
    y_tolog_orig = np.expm1(
        np.clip(Y[:, : len(idy_features[Y_TO_LOG_FEATURES])], 0, None)
    )
    y_log_orig = Y[:, len(idy_features[Y_TO_LOG_FEATURES]) :]
    Y_orig = np.zeros(
        (Y.shape[0], len(idy_features[LABELS])), dtype=np.float32
    )
    for i, idx in enumerate(idy_features[Y_TO_LOG_FEATURES]):
        Y_orig[:, idx] = y_tolog_orig[:, i]
    for i, idx in enumerate(idy_features[Y_LOG_FEATURES]):
        Y_orig[:, idx] = y_log_orig[:, i]

    features = build_feature_and_label_cols(window_size)

    # --- Create DataFrame ---
    df_x = pd.DataFrame(X_orig, columns=features["X_FEATURES"])
    df_y = pd.DataFrame(Y_orig, columns=features["LABELS"])
    df_w = pd.DataFrame(W, columns=[WEIGHT_COLUMN])
    df = pd.concat([df_x, df_y, df_w], axis=1)

    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        assert len(meta_df) == len(
            df
        ), f"Meta rows: {len(meta_df)}, Feature rows: {len(df)}"
        df = pd.concat(
            [meta_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1
        )
    else:
        logger.warning(f"Meta file not found at: {meta_path}")

    # --- Preserve original feature order for output ---
    return sort_df(df, window_size=window_size)


def combine_loaders_to_dataframe(
    dataloader_dir: Path,
    x_feature_cols: List[str],
    label_cols: List[str],
    *,
    store_cluster: Optional[int] = None,
    item_cluster: Optional[int] = None,
    loader_type: str = "val",  # or "train"
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Combines multiple loaders and meta files into a single DataFrame for a given store or item cluster.

    Parameters
    ----------
    dataloader_dir : Path
        Directory containing the *_loader.pt and *_meta.parquet files.
    x_feature_cols : List[str]
        Column names for input features.
    label_cols : List[str]
        Column names for label features.
    store_cluster : Optional[int]
        If set, only include files with this store cluster.
    item_cluster : Optional[int]
        If set, only include files with this item cluster.
    loader_type : str
        'train' or 'val' to indicate which type of loader to load.
    log_level : str
        Logging level for the logger.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all matching loaders and meta files.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    assert (store_cluster is None) != (
        item_cluster is None
    ), "Provide only one of store_cluster or item_cluster"
    pattern = re.compile(
        r"(\d+)_(\d+)_" + re.escape(loader_type) + r"_loader\.pt"
    )
    logger.info(f"Loading {dataloader_dir}")
    logger.info(f"Loader type: {loader_type}")
    logger.info(f"Store cluster: {store_cluster}")
    logger.info(f"Item cluster: {item_cluster}")
    dfs = []
    for file in dataloader_dir.glob(f"*_{loader_type}_loader.pt"):
        match = pattern.match(file.name)
        if not match:
            logger.warning(f"Skipping {file.name}")
            continue

        sc, ic = int(match.group(1)), int(match.group(2))
        logger.info(
            f"Loading {file.name} (store_cluster={sc}, item_cluster={ic})"
        )
        if store_cluster is not None and sc != store_cluster:
            logger.warning(f"Skipping {file.name} (store_cluster={sc})")
            continue
        if item_cluster is not None and ic != item_cluster:
            logger.warning(f"Skipping {file.name} (item_cluster={ic})")
            continue

        logger.info(f"Loading {file.name}")
        df = dataloader_to_dataframe(
            file, x_feature_cols, label_cols, "weight"
        )
        meta_path = dataloader_dir / f"{sc}_{ic}_{loader_type}_meta.parquet"
        logger.info(f"Loading {meta_path}")
        if meta_path.exists():
            meta_df = pd.read_parquet(meta_path)
            df = pd.concat(
                [meta_df.reset_index(drop=True), df.reset_index(drop=True)],
                axis=1,
            )
        else:
            logger.warning(f"Meta file not found for {file.name}")
        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()


def train_all_models_for_cluster_pair(
    model_types: List[ModelType],
    model_dir: Path,
    dataloader_dir: Path,
    scaler_dir: Path,
    model_logger_dir: Path,
    store_cluster: int,
    item_cluster: int,
    *,
    window_size: int = 1,
    lr: float = 3e-4,
    epochs: int = 5,
    hidden_dim: int = 128,
    h1: int = 64,
    h2: int = 32,
    depth: int = 3,
    dropout: float = 0.0,
    seed: int = 2025,
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Train multiple model types for a single (store_cluster, item_cluster) pair,
    appending results to the same history CSV.
    """

    for model_type in model_types:
        try:
            train(
                model_dir=model_dir,
                model_type=model_type,
                dataloader_dir=dataloader_dir,
                scaler_dir=scaler_dir,
                model_logger_dir=model_logger_dir,
                window_size=window_size,
                store_cluster=store_cluster,
                item_cluster=item_cluster,
                lr=lr,
                epochs=epochs,
                hidden_dim=hidden_dim,
                h1=h1,
                h2=h2,
                depth=depth,
                dropout=dropout,
                seed=seed,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                enable_progress_bar=enable_progress_bar,
                log_level=log_level,
            )
        except Exception as e:
            logger.exception(
                f"Failed to train {model_type.value} for ({store_cluster}, {item_cluster}): {e}"
            )


def train(
    model_dir: Path,
    dataloader_dir: Path,
    model_logger_dir: Path,
    scaler_dir: Path,
    model_type: ModelType,  # Your enum for feedforward models
    store_cluster: int,
    item_cluster: int,
    *,
    window_size: int = 1,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    h1: int = 64,
    h2: int = 32,
    depth: int = 3,
    dropout: float = 0.0,
    epochs: int = 30,
    seed: int = 2025,
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Unified training for feedforward per-cluster models and sequence models.
    """

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    pl.seed_everything(seed)

    checkpoints_dir = model_dir / "checkpoints"
    for d in [checkpoints_dir, model_logger_dir]:
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    model_name = f"{store_cluster}_{item_cluster}_{model_type.value}"
    # Load pre-saved metadata and loaders
    train_meta_fn = (
        dataloader_dir / f"{store_cluster}_{item_cluster}_train_meta.parquet"
    )
    val_meta_fn = (
        dataloader_dir / f"{store_cluster}_{item_cluster}_val_meta.parquet"
    )
    meta_df = pd.read_parquet(train_meta_fn)
    val_meta_df = pd.read_parquet(val_meta_fn)

    if meta_df.empty or val_meta_df.empty:
        logger.warning(
            f"Skipping pair ({store_cluster}, {item_cluster}) due to insufficient data."
        )
        return

    train_loader = torch.load(
        dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt",
        weights_only=False,
    )
    logger.info(
        f"Loaded train loader from {dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt"}"
    )
    val_loader = torch.load(
        dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt",
        weights_only=False,
    )
    logger.info(
        f"Loaded val loader from {dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt"}"
    )

    inferred_batch_size = train_loader.batch_size or 32
    logger.info(f"Inferred batch size: {inferred_batch_size}")
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    # Get number of training samples
    logger.info(f"Number of training samples: {len(train_loader.dataset)}")

    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")

    # Get input dimension
    input_dim = train_loader.dataset.tensors[0].shape[1]

    features = build_feature_and_label_cols(window_size)

    output_dim = len(features[LABELS])
    logger.info(
        f"Input dimension: {input_dim}, Output dimension: {output_dim}"
    )

    # Load scalers
    stem = train_meta_fn.stem
    cluster_key = "_".join(stem.split("_")[:2])
    x_log1p_scaler = pickle.load(
        (scaler_dir / f"{cluster_key}_x_log1p_scaler.pkl").open("rb")
    )
    logger.info(f"Loaded x_log1p_scaler for {cluster_key}")

    # Build model and wrapper
    base_model = model_factory(
        ModelType(model_type.value),
        input_dim,
        hidden_dim,
        h1,
        h2,
        depth,
        output_dim,
        dropout,
    )
    base_model.apply(init_weights)
    lightning_model = LightningWrapper(
        base_model,
        model_name=model_name,
        store=store_cluster,
        item=item_cluster,
        window_size=window_size,
        lr=lr,
        log_level=log_level,
        inverse_scaler=x_log1p_scaler,
    )

    checkpoint_dir = checkpoints_dir / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=checkpoint_dir,
        filename=model_name,
    )

    early_stop = EarlyStopping(
        monitor="val_mae", patience=5, mode="min", min_delta=1e-4
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize CSV logger
    csv_logger_name = f"{model_name}_{store_cluster}_{item_cluster}"
    csv_logger = CSVLogger(name=csv_logger_name, save_dir=model_logger_dir)
    strategy = DeepSpeedStrategy(stage=2)  # ZeRO-2, no external config
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=2,
        strategy=strategy,
        accumulate_grad_batches=2,
        precision="bf16-mixed",
        deterministic=True,
        max_epochs=epochs,
        logger=csv_logger,
        enable_progress_bar=enable_progress_bar,
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
    )

    logger.info(f"Training feedforward model: {model_name}")
    trainer.fit(lightning_model, train_loader, val_loader)


def load_latest_model(
    checkpoints_dir: Path,
    input_dim: int,
    output_dim: int,
    model_type: ModelType,
    *,
    log_level: str = "INFO",
) -> dict[tuple[int, int], LightningWrapper]:
    """
    Load the latest checkpoint per (store_cluster, item_cluster).
    Prioritizes -v3 > -v2 > -v1 > base (version=0).
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    candidates = defaultdict(list)
    pattern = re.compile(r"model_(\d+)_(\d+)_([A-Za-z0-9]+)(?:-v(\d+))?")
    version, best_ckpt, model_name = max(versioned_ckpts, key=lambda x: x[0])

    for ckpt_path in checkpoints_dir.rglob("*.ckpt"):
        match = pattern.match(ckpt_path.parent.name)  # match folder name
        if not match:
            continue
        sc, ic = int(match[1]), int(match[2])
        model_name = match[3]
        version = int(match[4]) if match[4] is not None else 0
        candidates[(sc, ic)].append((version, ckpt_path, model_name))

    model_dict = {}
    for (sc, ic), versioned_ckpts in candidates.items():
        try:
            model = model_factory_from_str(
                model_name,
                input_dim,
                hidden_dim=128,  # default
                h1=64,  # default
                h2=32,  # default
                depth=3,  # default
                output_dim=output_dim,
                dropout=0.0,  # default
            )
            wrapper = LightningWrapper.load_from_checkpoint(
                best_ckpt, model=model, strict=False
            )
            model_dict[(sc, ic)] = wrapper
        except Exception as e:
            logger.warning(f"Skipping {best_ckpt.name}: {e}")

    return model_dict


def load_scalers(
    scalers_dir: Path, *, log_level: str = "INFO"
) -> Dict[Tuple[int, int], Dict[str, MinMaxScaler]]:
    """
    Load x_cyc and x_sales scalers per (store_cluster, item_cluster)
    from a flat directory.

    Returns
    -------
    Dict[(int, int), {"x_cyc": MinMaxScaler, "x_sales": MinMaxScaler}]
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    scaler_dict: Dict[Tuple[int, int], Dict[str, MinMaxScaler]] = {}

    for filename in os.listdir(scalers_dir):
        if not filename.endswith("_scaler.pkl"):
            continue

        parts = filename.replace(".pkl", "").split("_")
        if len(parts) != 4:
            logger.warning(f"Unexpected filename format: {filename}")
            continue

        try:
            sc, ic = int(parts[0]), int(parts[1])
            scaler_type = f"{parts[2]}_{parts[3]}"  # "x_cyc" or "x_sales"
            key = (sc, ic)

            with open(scalers_dir / filename, "rb") as f:
                scaler = pickle.load(f)

            if key not in scaler_dict:
                scaler_dict[key] = {}

            scaler_dict[key][scaler_type] = scaler

        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")

    return scaler_dict


def load_lightning_wrapper(ckpt_path: Path) -> LightningWrapper:
    """
    Load a LightningWrapper from checkpoint with automatic model creation.

    Parameters
    ----------
    ckpt_path : Path
        Path to the .ckpt file.

    Returns
    -------
    LightningWrapper
        Loaded LightningWrapper instance with restored weights.
    """
    # Load just the hyperparameters first
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    model_name = extract_model_name(hparams["model_name"])
    model = model_factory_from_str(
        model_name,
        hparams["input_dim"],
        hidden_dim=128,  # default
        h1=64,  # default
        h2=32,  # default
        depth=3,  # default
        output_dim=hparams["output_dim"],
        dropout=0.0,  # default
    )
    # Load the wrapper with model injected
    wrapper = LightningWrapper.load_from_checkpoint(ckpt_path, model=model)
    return wrapper


def load_latest_models_from_checkpoints(
    checkpoints_dir: Path,
    input_dim: int,
    output_dim: int,
    *,
    log_level: str = "INFO",
) -> dict[tuple[int, int], LightningWrapper]:
    """
    Load the latest checkpoint per (store_cluster, item_cluster).
    Prioritizes -v3 > -v2 > -v1 > base (version=0).
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    candidates = defaultdict(list)
    pattern = re.compile(r"model_(\d+)_(\d+)_([A-Za-z0-9]+)(?:-v(\d+))?")

    for ckpt_path in checkpoints_dir.rglob("*.ckpt"):
        match = pattern.match(ckpt_path.parent.name)  # match folder name
        if not match:
            continue
        sc, ic = int(match[1]), int(match[2])
        model_name = match[3]
        version = int(match[4]) if match[4] is not None else 0
        candidates[(sc, ic)].append((version, ckpt_path, model_name))

    model_dict = {}
    for (sc, ic), versioned_ckpts in candidates.items():
        version, best_ckpt, model_name = max(
            versioned_ckpts, key=lambda x: x[0]
        )
        try:
            model = model_factory_from_str(
                model_name,
                input_dim,
                hidden_dim=128,  # default
                h1=64,  # default
                h2=32,  # default
                depth=3,  # default
                output_dim=output_dim,
                dropout=0.0,  # default
            )
            wrapper = LightningWrapper.load_from_checkpoint(
                best_ckpt, model=model, strict=False
            )
            model_dict[(sc, ic)] = wrapper
        except Exception as e:
            logger.warning(f"Skipping {best_ckpt.name}: {e}")

    return model_dict


def load_models_from_dir(model_dir="../output/models/", date_str: str = ""):
    """
    Loads models from the specified directory, optionally filtering by date_str.
    Returns a dictionary mapping store_item identifiers to their model and feature columns.

    Args:
        model_dir (str): Directory containing saved model files.
        date_str (str): Date string to filter models (e.g., '2025-05-29'). If empty, loads all.

    Returns:
        dict: Mapping store_item ID -> (model, feature_cols)
    """
    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".pth"):
            if date_str and not filename.endswith(f"_{date_str}.pth"):
                continue  # Skip non-matching date files
            model_path = os.path.join(model_dir, filename)
            sid, model, feature_cols = load_model(model_path)
            models[sid] = (model, feature_cols)
    return models


def batch_predict_all_store_items(
    meta_df: pd.DataFrame,
    input_df: pd.DataFrame,
    model_dict: dict[tuple[int, int], LightningWrapper],
    input_feature_cols: list[str],
    store_col: str = "store",
    item_col: str = "item",
    store_cluster_col: str = "store_cluster",
    item_cluster_col: str = "item_cluster",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    predictions = []

    for (store, item), row in meta_df.groupby([store_col, item_col]):
        try:
            store_cluster = row[store_cluster_col].iloc[0]
            item_cluster = row[item_cluster_col].iloc[0]
            model_key = (store_cluster, item_cluster)

            if model_key not in model_dict:
                continue

            model = model_dict[model_key].model
            model.eval()
            model.to(device)

            # Filter rows for this store-item pair
            rows = input_df.query(
                f"{store_col} == @store and {item_col} == @item"
            )
            if rows.empty:
                continue

            X = torch.tensor(
                rows[input_feature_cols].values, dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                preds_log = model(X)
                preds_log = torch.clamp(preds_log, min=1e-6)
                preds_pct = torch.expm1(preds_log).cpu().numpy()

            preds_df = pd.DataFrame(
                {
                    "store": store,
                    "item": item,
                    "log_pct_change": preds_log.cpu().numpy().squeeze(),
                    "pct_change": preds_pct.squeeze(),
                    "index": rows.index,
                }
            )

            predictions.append(preds_df)

        except Exception as e:
            print(f"Error predicting for store {store}, item {item}: {e}")

    return pd.concat(predictions, ignore_index=True)


def predict_next_days_for_sid(
    sid: str,
    last_date_df: pd.DataFrame,
    models: dict,
    y_scalers: dict,
    days_to_predict: int = 16,
) -> pd.DataFrame:
    model, feature_cols = models[sid]
    model.eval()

    input_data = last_date_df.query(f"store_item == '{sid}'")
    x = input_data[feature_cols].values.astype("float32")
    x_tensor = torch.tensor(x)

    with torch.no_grad():
        y_pred = model(x_tensor)

    y_pred_scaled = pd.DataFrame(y_pred.numpy(), columns=feature_cols)
    y_pred_df = pd.DataFrame(
        y_scalers[sid].inverse_transform(y_pred_scaled), columns=feature_cols
    )

    sales_day_cols = [
        col for col in y_pred_df.columns if col.startswith("sales_day_")
    ]
    sales_pred_df = y_pred_df[sales_day_cols]
    sales_pred_df = np.expm1(sales_pred_df)

    meta = input_data.iloc[0][["store_item", "store", "item"]].to_dict()
    start_date = pd.to_datetime(
        input_data.iloc[0]["start_date"]
    ) + pd.Timedelta(days=15)

    rows = []
    for i, col in enumerate(sales_day_cols[:days_to_predict]):
        row = {
            "date": start_date + pd.Timedelta(days=i + 1),
            **meta,
            "unit_sales": sales_pred_df.at[0, col],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def predict_next_days_for_sids(
    last_date_df: pd.DataFrame,
    models: dict,
    y_scalers: dict,
    days_to_predict: int = 16,
) -> pd.DataFrame:
    """
    Predicts the next `days_to_predict` days for all store_items present in the models dict.

    Args:
        last_date_df (pd.DataFrame): Latest input row per store_item.
        models (dict): Dict of store_item -> (model, feature_cols).
        y_scalers (dict): Dict of store_item -> MinMaxScaler to inverse-transform predictions.
        days_to_predict (int): Number of days to forecast per store_item.

    Returns:
        pd.DataFrame: Combined predictions for all store_items.
    """
    all_preds = []
    for sid in models.keys():
        try:
            pred_df = predict_next_days_for_sid(
                sid, last_date_df, models, y_scalers, days_to_predict
            )
            all_preds.append(pred_df)
        except Exception as e:
            logger.info(f"Skipping {sid} due to error: {e}")

    return pd.concat(all_preds, ignore_index=True)
