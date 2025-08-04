import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import lightning.pytorch as pl
import random
import logging
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import pickle
from pathlib import Path
from enum import Enum
from tqdm import tqdm
from collections import defaultdict
import re
from typing import Optional
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from src.model import LightningWrapper, ModelType
from src.model import compute_mav, model_factory, init_weights, model_factory_from_str
from torch.utils.data import TensorDataset, Dataset, DataLoader

logger = logging.getLogger(__name__)

# Set up logger
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    return accelerator


def safe_append_to_history(history_fn: Path, new_history: pd.DataFrame) -> pd.DataFrame:
    """
    Safely appends new training history to an existing CSV file if it exists and is valid.

    Parameters
    ----------
    history_fn : Path
        Path to the CSV history file.
    new_history : pd.DataFrame
        New training history to append.

    Returns
    -------
    pd.DataFrame
        The combined history DataFrame.
    """
    if (
        history_fn is not None
        and history_fn.exists()
        and os.path.getsize(history_fn) > 0
    ):
        try:
            previous_history = pd.read_csv(history_fn)

            # Check for valid structure
            if not previous_history.empty and set(new_history.columns).issubset(
                previous_history.columns
            ):
                combined_history = pd.concat(
                    [previous_history, new_history], ignore_index=True
                )
            else:
                combined_history = new_history

        except Exception as e:
            print(
                f"Warning: Failed to read or validate {history_fn}. Using new history only. Reason: {e}"
            )
            combined_history = new_history
    else:
        combined_history = new_history

    # Save combined history
    if history_fn is not None:
        logger.info(f"Saving history to {history_fn}")
        combined_history.to_csv(history_fn, index=False)

    return combined_history


class StoreItemDataset(Dataset):
    def __init__(self, df, store_item_id, feature_cols, target_col, weight_col):
        self.store_df = df[df["store_item"] == store_item_id].reset_index(drop=True)
        self.X = torch.tensor(self.store_df[feature_cols].values, dtype=torch.float32)
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


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────


def generate_loaders(
    df: pd.DataFrame,
    all_features: List[str],
    meta_cols: List[str],
    x_feature_cols: List[str],
    x_to_log_features: List[str],
    x_log_features: List[str],
    x_cyclical_features: List[str],
    label_cols: List[str],
    y_log_features: List[str],
    y_to_log_features: List[str],
    dataloader_dir: Path,
    scalers_dir: Path,
    *,
    weight_col: str = "weight",
    window_val: int = 30,
    val_horizon: int = 30,
    batch_size: int = 32,
    num_workers: int = 15,
    log_level: str = "INFO",
):
    """
    Generate DataLoaders using sliding windows with a hold-out validation block
    aligned with sequence model loaders.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    dataloader_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare DataFrame ---
    df = (
        df[all_features]
        .sort_values(["start_date", "store_item"])
        .reset_index(drop=True)
    )
    df["store_cluster"] = pd.to_numeric(df["store_cluster"], errors="coerce")
    df["item_cluster"] = pd.to_numeric(df["item_cluster"], errors="coerce")
    store_cluster = df["store_cluster"].unique()
    item_cluster = df["item_cluster"].unique()
    assert len(store_cluster) == 1
    assert len(item_cluster) == 1
    store_cluster, item_cluster = store_cluster[0], item_cluster[0]
    cluster_key = f"{store_cluster}_{item_cluster}"

    logger.info(f"Preparing loaders for cluster {cluster_key} with {len(df)} rows")

    if weight_col not in df.columns:
        df[weight_col] = 1.0

    num_samples = len(df)
    num_windows = num_samples - window_val
    validation_cutoff = (
        num_samples - val_horizon
    )  # 🔹 windows ending after this go to validation

    if num_windows <= 0:
        logger.warning("No valid windows")
        empty = torch.empty((0, len(x_feature_cols)), dtype=torch.float32)
        empty_y = torch.empty((0, len(label_cols)), dtype=torch.float32)
        empty_w = torch.empty((0, 1), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(empty, empty_y, empty_w), batch_size=batch_size, num_workers=0
        )
        meta_df = pd.DataFrame(columns=meta_cols)
        for split in ["train", "val"]:
            torch.save(loader, dataloader_dir / f"{cluster_key}_{split}_loader.pt")
            meta_df.to_parquet(dataloader_dir / f"{cluster_key}_{split}_meta.parquet")
        return loader, loader

    # --- Storage for windows ---
    X_train_raw, y_train_raw, W_train_raw, meta_train_raw = [], [], [], []
    X_val_raw, y_val_raw, W_val_raw, meta_val_raw = [], [], [], []

    for i in tqdm(range(num_windows), desc="Generating windows"):
        df_train = df.iloc[i : i + window_val].fillna(0)

        X_win = df_train[x_feature_cols].values
        y_win = df_train[label_cols].values
        W_win = df_train[[weight_col]].values
        meta_win = df_train[meta_cols]

        if i + window_val <= validation_cutoff:
            # Training window
            X_train_raw.append(X_win)
            y_train_raw.append(y_win)
            W_train_raw.append(W_win)
            meta_train_raw.append(meta_win)
        else:
            # Validation window
            X_val_raw.append(X_win)
            y_val_raw.append(y_win)
            W_val_raw.append(W_win)
            meta_val_raw.append(meta_win)

    # --- Stack arrays ---
    X_train = np.vstack(X_train_raw).astype(np.float32)
    y_train = np.vstack(y_train_raw).astype(np.float32)
    W_train = np.vstack(W_train_raw).astype(np.float32)
    X_val = np.vstack(X_val_raw).astype(np.float32)
    y_val = np.vstack(y_val_raw).astype(np.float32)
    W_val = np.vstack(W_val_raw).astype(np.float32)

    meta_train_df = pd.concat(meta_train_raw, ignore_index=True)
    meta_val_df = pd.concat(meta_val_raw, ignore_index=True)

    # --- Feature indices for scaling ---
    col_x_index_map = {col: idx for idx, col in enumerate(x_feature_cols)}
    x_to_log_idx = [col_x_index_map[c] for c in x_to_log_features]
    x_log_idx = [col_x_index_map[c] for c in x_log_features]
    x_cyc_idx = [col_x_index_map[c] for c in x_cyclical_features]

    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    y_to_log_idx = [col_y_index_map[c] for c in y_to_log_features]
    y_log_idx = [col_y_index_map[c] for c in y_log_features]

    # --- Transform & Scale ---
    x_log1p_train = np.log1p(np.clip(X_train[:, x_to_log_idx], 0, None))
    x_log_raw_train = X_train[:, x_log_idx]
    x_cyc_train = X_train[:, x_cyc_idx]
    y_train = np.hstack(
        [np.log1p(np.clip(y_train[:, y_to_log_idx], 0, None)), y_train[:, y_log_idx]]
    )

    x_log1p_val = np.log1p(np.clip(X_val[:, x_to_log_idx], 0, None))
    x_log_raw_val = X_val[:, x_log_idx]
    x_cyc_val = X_val[:, x_cyc_idx]
    y_val = np.hstack(
        [np.log1p(np.clip(y_val[:, y_to_log_idx], 0, None)), y_val[:, y_log_idx]]
    )

    scaler_log1p = MinMaxScaler().fit(x_log1p_train)
    scaler_log_raw = MinMaxScaler().fit(x_log_raw_train)
    scaler_cyc = MinMaxScaler().fit(x_cyc_train)

    X_train_scaled = np.hstack(
        [
            scaler_log1p.transform(x_log1p_train),
            scaler_log_raw.transform(x_log_raw_train),
            scaler_cyc.transform(x_cyc_train),
        ]
    )
    X_val_scaled = np.hstack(
        [
            scaler_log1p.transform(x_log1p_val),
            scaler_log_raw.transform(x_log_raw_val),
            scaler_cyc.transform(x_cyc_val),
        ]
    )

    # --- Create DataLoaders ---
    persistent = num_workers > 0
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled), torch.tensor(y_train), torch.tensor(W_train)
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_scaled), torch.tensor(y_val), torch.tensor(W_val)
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
        meta_df.to_parquet(dataloader_dir / f"{cluster_key}_{split}_meta.parquet")

    # --- Save scalers ---
    for scaler, name in zip(
        [scaler_log1p, scaler_log_raw, scaler_cyc], ["x_log1p", "x_log_raw", "x_cyc"]
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
    x_feature_cols: List[str],
    label_cols: List[str],
    x_to_log_features: List[str],
    x_log_features: List[str],
    x_cyclical_features: List[str],
    y_to_log_features: List[str],
    y_log_features: List[str],
    meta_cols: List[str],
    *,
    weight_col: str = "weight",
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

    # --- Feature indices ---
    col_x_index_map = {col: idx for idx, col in enumerate(x_feature_cols)}
    x_to_log_idx = [col_x_index_map[c] for c in x_to_log_features]
    x_log_idx = [col_x_index_map[c] for c in x_log_features]
    x_cyc_idx = [col_x_index_map[c] for c in x_cyclical_features]

    # --- Inverse transform ---
    n_tolog = len(x_to_log_idx)
    n_log = len(x_log_idx)

    x_log1p_scaled = X[:, :n_tolog]
    x_log_scaled = X[:, n_tolog : n_tolog + n_log]
    x_cyc_scaled = X[:, n_tolog + n_log :]

    x_tolog_orig = np.expm1(x_log1p_scaler.inverse_transform(x_log1p_scaled))
    x_log_orig = x_log_raw_scaler.inverse_transform(x_log_scaled)
    x_cyc_orig = x_cyc_scaler.inverse_transform(x_cyc_scaled)

    X_orig = np.zeros((X.shape[0], len(x_feature_cols)), dtype=np.float32)
    for i, idx in enumerate(x_to_log_idx):
        X_orig[:, idx] = x_tolog_orig[:, i]
    for i, idx in enumerate(x_log_idx):
        X_orig[:, idx] = x_log_orig[:, i]
    for i, idx in enumerate(x_cyc_idx):
        X_orig[:, idx] = x_cyc_orig[:, i]

    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    y_to_log_idx = [col_y_index_map[c] for c in y_to_log_features]
    y_log_idx = [col_y_index_map[c] for c in y_log_features]

    y_tolog_orig = np.expm1(np.clip(Y[:, : len(y_to_log_idx)], 0, None))
    y_log_orig = Y[:, len(y_to_log_idx) :]
    Y_orig = np.zeros((Y.shape[0], len(label_cols)), dtype=np.float32)
    for i, idx in enumerate(y_to_log_idx):
        Y_orig[:, idx] = y_tolog_orig[:, i]
    for i, idx in enumerate(y_log_idx):
        Y_orig[:, idx] = y_log_orig[:, i]

    df_x = pd.DataFrame(X_orig, columns=x_feature_cols)
    df_y = pd.DataFrame(Y_orig, columns=label_cols)
    df_w = pd.DataFrame(W, columns=[weight_col])
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

    all_cols = meta_cols + x_feature_cols + label_cols + [weight_col]
    df = df[all_cols]
    df.sort_values(["start_date", "store_item"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


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
    pattern = re.compile(r"(\d+)_(\d+)_" + re.escape(loader_type) + r"_loader\.pt")
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
        logger.info(f"Loading {file.name} (store_cluster={sc}, item_cluster={ic})")
        if store_cluster is not None and sc != store_cluster:
            logger.warning(f"Skipping {file.name} (store_cluster={sc})")
            continue
        if item_cluster is not None and ic != item_cluster:
            logger.warning(f"Skipping {file.name} (item_cluster={ic})")
            continue

        logger.info(f"Loading {file.name}")
        df = dataloader_to_dataframe(file, x_feature_cols, label_cols, "weight")
        meta_path = dataloader_dir / f"{sc}_{ic}_{loader_type}_meta.parquet"
        logger.info(f"Loading {meta_path}")
        if meta_path.exists():
            meta_df = pd.read_parquet(meta_path)
            df = pd.concat(
                [meta_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1
            )
        else:
            logger.warning(f"Meta file not found for {file.name}")
        dfs.append(df)

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()


def train_all_models_for_cluster_pair(
    model_types: List[ModelType],
    model_dir: Path,
    dataloader_dir: Path,
    label_cols: list[str],
    y_log_features: list[str],
    store_cluster: int,
    item_cluster: int,
    *,
    history_dir: Optional[Path] = None,
    lr: float = 3e-4,
    epochs: int = 5,
    seed: int = 2025,
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    train_logger: bool = False,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Train multiple model types for a single (store_cluster, item_cluster) pair,
    appending results to the same history CSV.
    """
    all_histories = []

    for model_type in model_types:
        try:
            history = train_model_unified(
                model_dir=model_dir,
                model_type=model_type,
                dataloader_dir=dataloader_dir,
                model_family="feedforward",
                label_cols=label_cols,
                y_log_features=y_log_features,
                store_cluster=store_cluster,
                item_cluster=item_cluster,
                history_dir=history_dir,
                lr=lr,
                epochs=epochs,
                seed=seed,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                enable_progress_bar=enable_progress_bar,
                train_logger=train_logger,
                log_level=log_level,
            )
            break
            # history = train_per_cluster_pair(
            #     model_dir=model_dir,
            #     model_type=model_type,
            #     dataloader_dir=dataloader_dir,
            #     label_cols=label_cols,
            #     y_log_features=y_log_features,
            #     store_cluster=store_cluster,
            #     item_cluster=item_cluster,
            #     history_dir=history_dir,
            #     lr=lr,
            #     epochs=epochs,
            #     seed=seed,
            #     num_workers=num_workers,
            #     persistent_workers=persistent_workers,
            #     enable_progress_bar=enable_progress_bar,
            #     train_logger=train_logger,
            #     log_level=log_level,
            # )
            all_histories.append(history)
        except Exception as e:
            logger.exception(
                f"Failed to train {model_type.value} for ({store_cluster}, {item_cluster}): {e}"
            )

    if all_histories:
        return pd.concat(all_histories, ignore_index=True)
    else:
        return pd.DataFrame()


def train_per_cluster_pair(
    model_dir: Path,
    model_type: ModelType,
    dataloader_dir: Path,
    label_cols: list[str],
    y_log_features: list[str],
    store_cluster: int,
    item_cluster: int,
    *,
    history_dir: Optional[Path] = None,
    lr: float = 3e-4,
    epochs: int = 5,
    seed: int = 2025,
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    train_logger: bool = False,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    pl.seed_everything(seed)
    logger.info(
        f"Training model: {model_type.value} for cluster pair: ({store_cluster}, {item_cluster})"
    )

    # Setup paths
    checkpoints_dir = model_dir / "checkpoints"
    for d in [checkpoints_dir, history_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load pre-saved metadata
    train_meta_fn = (
        dataloader_dir / f"{store_cluster}_{item_cluster}_train_meta.parquet"
    )
    val_meta_fn = dataloader_dir / f"{store_cluster}_{item_cluster}_val_meta.parquet"
    meta_df = pd.read_parquet(train_meta_fn)
    val_meta_df = pd.read_parquet(val_meta_fn)

    if meta_df.empty or val_meta_df.empty:
        logger.warning(
            f"Skipping pair ({store_cluster}, {item_cluster}) due to insufficient data."
        )
        return pd.DataFrame()

    # Load pre-filtered dataloaders
    train_loader = torch.load(
        dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt",
        weights_only=False,
    )
    val_loader = torch.load(
        dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt",
        weights_only=False,
    )

    inferred_batch_size = train_loader.batch_size or 32
    logger.info(f"Inferred batch size: {inferred_batch_size}")

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )

    input_dim = train_dataset.tensors[0].shape[1]
    output_dim = len(label_cols)

    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    logger.debug(f"Checking label order consistency: {col_y_index_map}")

    y_log_idx = [col_y_index_map[c] for c in y_log_features]
    train_mav = compute_mav(train_loader, y_log_idx)
    val_mav = compute_mav(val_loader, y_log_idx)

    base_model = model_factory(model_type, input_dim, output_dim)
    base_model.apply(init_weights)
    model_name = f"{store_cluster}_{item_cluster}_{model_type.value}"

    lightning_model = LightningWrapper(
        base_model,
        model_name=model_name,
        store=store_cluster,
        item=item_cluster,
        sales_idx=y_log_idx,
        train_mav=train_mav,
        val_mav=val_mav,
        lr=lr,
        log_level=log_level,
    )
    checkpoint_dir = checkpoints_dir / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="best_train_avg_mae",
        mode="min",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename=model_name,
    )

    logger.info("Training model...")
    trainer = pl.Trainer(
        accelerator=get_device(),
        deterministic=True,
        max_epochs=epochs,
        logger=train_logger,
        enable_progress_bar=enable_progress_bar,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    # Collect history
    history = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "store_cluster": store_cluster,
                "item_cluster": item_cluster,
                "train_mav": train_mav,
                "val_mav": val_mav,
                "best_train_avg_mae": lightning_model.best_train_avg_mae,
                "best_val_avg_mae": lightning_model.best_val_avg_mae,
                "best_train_avg_rmse": lightning_model.best_train_avg_rmse,
                "best_val_avg_rmse": lightning_model.best_val_avg_rmse,
                "best_train_avg_mae_percent_mav": lightning_model.best_train_avg_mae_percent_mav,
                "best_val_avg_mae_percent_mav": lightning_model.best_val_avg_mae_percent_mav,
            }
        ]
    )

    # Save history
    if history_dir:
        history_fn = (
            history_dir / f"{store_cluster}_{item_cluster}_{model_type.value}.csv"
        )
        history = safe_append_to_history(history_fn, history)
    return history


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
            model = model_factory_from_str(model_name, input_dim, output_dim)
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
        model_name, hparams["input_dim"], hparams["output_dim"]
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
        version, best_ckpt, model_name = max(versioned_ckpts, key=lambda x: x[0])
        try:
            model = model_factory_from_str(model_name, input_dim, output_dim)
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
            rows = input_df.query(f"{store_col} == @store and {item_col} == @item")
            if rows.empty:
                continue

            X = torch.tensor(rows[input_feature_cols].values, dtype=torch.float32).to(
                device
            )
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

    sales_day_cols = [col for col in y_pred_df.columns if col.startswith("sales_day_")]
    sales_pred_df = y_pred_df[sales_day_cols]
    sales_pred_df = np.expm1(sales_pred_df)

    meta = input_data.iloc[0][["store_item", "store", "item"]].to_dict()
    start_date = pd.to_datetime(input_data.iloc[0]["start_date"]) + pd.Timedelta(
        days=15
    )

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
    last_date_df: pd.DataFrame, models: dict, y_scalers: dict, days_to_predict: int = 16
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


# ------------------------
# Sequence to Sequence
# ------------------------


def generate_sequence_model_loaders(
    df: pd.DataFrame,
    meta_cols: List[str],
    all_features: List[str],
    x_feature_cols: List[str],  # multi-input x
    label_cols: List[str],  # multi-target y
    dataloader_dir: Path,
    *,
    weight_col: str = "weight",
    max_encoder_length: int = 30,  # historical window size, e.g., 30 days
    max_prediction_length: int = 1,  # usually 1 for next-day forecasting
    val_horizon: int = 30,  # Last N days for validation
    batch_size: int = 64,
    num_workers: int = 8,
    log_level: str = "INFO",
):
    """
    Generate PyTorch Forecasting DataLoaders (train & val) for TFT.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with meta, features, and labels for 2 years.
    meta_cols : list of str
        Columns like ['start_date','store_item','store_cluster','item_cluster'].
    x_sales_cols : list of str
        Time-varying unknown features (sales, lags, medians, etc.).
    x_cyclical_cols : list of str
        Time-varying known future features (calendar/cyclical).
    label_cols : list of str
        Multi-target y columns (e.g., sales + logpct changes).
    dataloader_dir : Path
        Directory to save loaders and meta.
    weight_col : str
        Optional weight column for loss.
    max_encoder_length : int
        Days of history for the encoder (like your old window_size).
    max_prediction_length : int
        Days to predict (usually 1 for next-day forecasting).
    val_horizon : int
        How many last days to keep for validation.
    batch_size : int
        Loader batch size.
    num_workers : int
        DataLoader workers.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    dataloader_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare and sort DF ---
    df = (
        df[all_features]
        .sort_values(["start_date", "store_item"])
        .reset_index(drop=True)
    )
    df["store_cluster"] = df["store_cluster"].astype(str)
    df["item_cluster"] = df["item_cluster"].astype(str)

    store_cluster = df["store_cluster"].unique()
    item_cluster = df["item_cluster"].unique()
    assert len(store_cluster) == 1
    assert len(item_cluster) == 1
    store_cluster, item_cluster = store_cluster[0], item_cluster[0]
    cluster_key = f"{store_cluster}_{item_cluster}"
    logger.info(
        f"Preparing Sequence loaders for cluster {cluster_key} with {len(df)} rows"
    )

    # --- Create time index per series ---
    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("store_item").cumcount()

    if weight_col not in df.columns:
        df[weight_col] = 1.0

    # --- Train/validation split ---
    validation_cutoff = df["time_idx"].max() - val_horizon
    train_df = df[df.time_idx <= validation_cutoff]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        group_ids=["store_item"],
        weight=weight_col,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=x_feature_cols,
        time_varying_unknown_reals=label_cols,
        target=label_cols,
        static_categoricals=["store_cluster", "item_cluster"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # --- Validation using same scaling & encoding ---
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    # --- Convert to DataLoaders ---
    train_loader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # --- Save loaders and meta ---
    torch.save(train_loader, dataloader_dir / f"{cluster_key}_seq_train_loader.pt")
    torch.save(val_loader, dataloader_dir / f"{cluster_key}_seq_val_loader.pt")

    df[df.time_idx <= validation_cutoff][meta_cols].to_parquet(
        dataloader_dir / f"{cluster_key}_seq_train_meta.parquet"
    )
    df[df.time_idx > validation_cutoff][meta_cols].to_parquet(
        dataloader_dir / f"{cluster_key}_seq_val_meta.parquet"
    )

    logger.info(f"Saved Sequence Model loaders and meta for {cluster_key}")
    return train_loader, val_loader


def train_sequence_model(
    df: pd.DataFrame,
    meta_cols: list[str],
    x_historical_cols: list[str],
    x_cyclical_cols: list[str],
    label_cols: list[str],
    dataloader_dir: Path,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train a sequence-to-sequence model using PyTorch Lightning and TorchForecasting.
    This function prepares the data, defines the model, and trains it.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing time series data.
    meta_cols : list[str]
        List of metadata columns.
    x_historical_cols : list[str]
        List of historical feature columns.
    x_cyclical_cols : list[str]
        List of cyclical feature columns.
    label_cols : list[str]
        List of label columns to predict.
    dataloader_dir : Path
        Directory to save the DataLoaders.

    Returns
    ----------
    Tuple[DataLoader, DataLoader]
    """

    # --- Get the underlying TimeSeriesDataSet from train loader ---
    training_dataset = train_loader.dataset.dataset  # DataLoader -> TimeSeriesDataSet
    # Get number of training samples - handle different dataset types
    try:
        num_samples = len(training_dataset)
    except (TypeError, AttributeError):
        # For datasets that don't implement __len__ properly
        num_samples = getattr(training_dataset, "length", "unknown")
    print("Training samples:", num_samples)

    # --- Define TemporalFusionTransformer model ---
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=1e-3,
        hidden_size=32,  # number of LSTM units
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,  # size for continuous variables
        output_size=len(label_cols),  # multi-target
        loss=RMSE(),  # or QuantileLoss([0.1,0.5,0.9]) for probabilistic
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    print(f"Number of parameters in model: {tft.size()/1e3:.1f}k")

    # --- Define PyTorch Lightning Trainer ---
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        enable_progress_bar=True,
        deterministic=True,
    )

    # --- Train the model ---
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def train_model_unified(
    model_dir: Path,
    dataloader_dir: Path,
    model_type: ModelType,  # Your enum for feedforward models
    model_family: str,  # "feedforward" or "sequence"
    label_cols: list[str],
    y_log_features: list[str],
    store_cluster: int,
    item_cluster: int,
    *,
    history_dir: Optional[Path] = None,
    lr: float = 3e-4,
    epochs: int = 30,
    seed: int = 2025,
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    train_logger: bool = False,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Unified training for feedforward per-cluster models and sequence models.
    """

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    pl.seed_everything(seed)
    checkpoints_dir = model_dir / "checkpoints"
    for d in [checkpoints_dir, history_dir]:
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    model_name = f"{store_cluster}_{item_cluster}_{model_type.value if model_family=='feedforward' else model_family}"

    # --------------------------------------------------------
    # 1) FEEDFORWARD MODEL BRANCH
    # --------------------------------------------------------
    if model_family == "feedforward":
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
            return pd.DataFrame()

        train_loader = torch.load(
            dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt",
            weights_only=False,
        )
        val_loader = torch.load(
            dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt",
            weights_only=False,
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
        # Get number of training samples - handle different dataset types
        try:
            num_samples = len(train_loader.dataset)
        except (TypeError, AttributeError):
            # For datasets that don't implement __len__ properly
            num_samples = getattr(train_loader.dataset, "length", "unknown")
        logger.info(f"Number of training samples: {num_samples}")
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=inferred_batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )
        try:
            num_samples = len(val_loader.dataset)
        except (TypeError, AttributeError):
            # For datasets that don't implement __len__ properly
            num_samples = getattr(val_loader.dataset, "length", "unknown")
        logger.info(f"Number of training samples: {num_samples}")

        # Get input dimension from the first batch
        # For TensorDataset, we can access tensors directly
        if hasattr(train_loader.dataset, "tensors"):
            input_dim = train_loader.dataset.tensors[0].shape[1]
        else:
            # Fallback: get from first batch
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].shape[1]
        output_dim = len(label_cols)

        # Compute MAV for normalization
        col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
        y_log_idx = [col_y_index_map[c] for c in y_log_features]
        logger.info(f"y_log_idx: {y_log_idx}")
        train_mav = compute_mav(train_loader, y_log_idx)
        val_mav = compute_mav(val_loader, y_log_idx)

        # Build model and wrapper
        base_model = model_factory(model_type, input_dim, output_dim)
        base_model.apply(init_weights)
        lightning_model = LightningWrapper(
            base_model,
            model_name=model_name,
            store=store_cluster,
            item=item_cluster,
            sales_idx=y_log_idx,
            train_mav=train_mav,
            val_mav=val_mav,
            lr=lr,
            log_level=log_level,
        )

        checkpoint_dir = checkpoints_dir / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="best_train_avg_mae",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_dir,
            filename=model_name,
        )

        trainer = pl.Trainer(
            accelerator=get_device(),
            deterministic=True,
            max_epochs=epochs,
            logger=True,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback],
        )

        logger.info(f"Training feedforward model: {model_name}")
        trainer.fit(lightning_model, train_loader, val_loader)

        # Collect training history
        history = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "store_cluster": store_cluster,
                    "item_cluster": item_cluster,
                    "train_mav": train_mav,
                    "val_mav": val_mav,
                    "best_train_avg_mae": lightning_model.best_train_avg_mae,
                    "best_val_avg_mae": lightning_model.best_val_avg_mae,
                    "best_train_avg_rmse": lightning_model.best_train_avg_rmse,
                    "best_val_avg_rmse": lightning_model.best_val_avg_rmse,
                    "best_train_avg_mae_percent_mav": lightning_model.best_train_avg_mae_percent_mav,
                    "best_val_avg_mae_percent_mav": lightning_model.best_val_avg_mae_percent_mav,
                }
            ]
        )

    # --------------------------------------------------------
    # 2) SEQUENCE MODEL BRANCH (TFT or other)
    # --------------------------------------------------------
    elif model_family == "sequence":

        # Load dataloaders from disk (assume generated by generate_sequence_model_loaders)
        train_loader = torch.load(
            dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt",
            weights_only=False,
        )
        val_loader = torch.load(
            dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt",
            weights_only=False,
        )

        training_dataset = train_loader.dataset.dataset  # underlying TimeSeriesDataSet
        # Get number of training samples - handle different dataset types
        try:
            num_samples = len(training_dataset)
        except (TypeError, AttributeError):
            # For datasets that don't implement __len__ properly
            num_samples = getattr(training_dataset, "length", "unknown")
        logger.info(f"Training samples: {num_samples}")

        # Build TFT model
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=lr,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=len(label_cols),
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )
        logger.info(f"TFT model params: {tft.size()/1e3:.1f}k")

        # Callbacks and Trainer
        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=get_device(),
            deterministic=True,
            enable_progress_bar=enable_progress_bar,
            callbacks=[early_stop],
        )

        logger.info(f"Training sequence model: {model_name}")
        trainer.fit(tft, train_loader, val_loader)

        # Collect history
        history = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "store_cluster": store_cluster,
                    "item_cluster": item_cluster,
                    "best_val_loss": trainer.callback_metrics.get(
                        "val_loss", float("nan")
                    ).item(),
                }
            ]
        )

    else:
        raise ValueError("`model_family` must be either 'feedforward' or 'sequence'")

    # Save history
    if history_dir:
        history_fn = history_dir / f"{model_name}.csv"
        history = safe_append_to_history(history_fn, history)

    return history
