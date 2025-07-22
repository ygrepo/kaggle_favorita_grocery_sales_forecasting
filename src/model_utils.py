import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import lightning.pytorch as pl
import random
import logging
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import LightningModule
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import pickle
from pathlib import Path
from enum import Enum
from tqdm import tqdm

import re
from collections import defaultdict
from typing import Optional


# Set up logger
logger = logging.getLogger(__name__)


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


class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.0):
        """
        Shallow 1-layer neural network for log-space regression targets.
        Outputs are unbounded real numbers.
        """
        super().__init__()

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(hidden_dim, output_dim),
            nn.Identity(),  # no activation at output
        )

    def forward(self, x):
        return self.net(x)


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim=3, h1=128, h2=64, dropout=0.2):
        """
        Two-layer feedforward NN for log-transformed targets.
        All outputs are unbounded real values, no final activation.
        """
        super().__init__()

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(h2, output_dim),
            nn.Identity(),  # no activation
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden=128, depth=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, input_dim),  # residual matches input_dim
                )
            )
        self.out_proj = nn.Linear(input_dim, output_dim)  # final projection

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(out)  # residual connection
        return self.out_proj(out)  # raw outputs, unbounded


# Enum for model types
class ModelType(Enum):
    SHALLOW_NN = "ShallowNN"
    TWO_LAYER_NN = "TwoLayerNN"
    RESIDUAL_MLP = "ResidualMLP"


MODEL_TYPES = list(ModelType)


# Model Factory Function
def model_factory(model_type: ModelType, input_dim: int, output_dim: int) -> nn.Module:
    """Factory function to return the correct model based on the model_type."""
    if model_type == ModelType.SHALLOW_NN:
        return ShallowNN(input_dim, output_dim)
    elif model_type == ModelType.TWO_LAYER_NN:
        return TwoLayerNN(input_dim, output_dim)
    elif model_type == ModelType.RESIDUAL_MLP:
        return ResidualMLP(input_dim, output_dim)


# ----- initialise every nn.Linear in the network ----------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.01)  # He initialisation for LeakyReLU
        nn.init.zeros_(m.bias)


class NWRMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        logger.setLevel(logging.INFO)

    def forward(self, y_pred, y_true, w):
        # add 1 to avoid log(0), clamp to eps to keep preds positive
        eps = 1e-6
        y_pred = torch.clamp(y_pred, min=eps)
        log_diff = torch.log(y_pred + 1.0) - torch.log(y_true + 1.0)
        num = torch.sum(w * log_diff**2)
        den = torch.sum(w)
        return torch.sqrt(num / den)


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true, w):
        eps = 1e-6
        y_pred = torch.clamp(y_pred, min=eps)
        log_diff = torch.log(y_pred + 1.0) - torch.log(y_true + 1.0)
        nwrmsle = torch.sqrt(torch.sum(w * log_diff**2) / torch.sum(w))
        mae = torch.sum(w * torch.abs(y_pred - y_true)) / torch.sum(w)
        return self.alpha * nwrmsle + (1 - self.alpha) * mae


def set_seed(seed: int = 42):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For full reproducibility in future versions
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────
def generate_loaders(
    df: pd.DataFrame,
    meta_cols: List[str],
    x_feature_cols: List[str],
    x_to_log_features: List[str],
    x_log_features: List[str],
    x_cyclical_features: List[str],
    label_cols: List[str],
    y_log_features: List[str],
    y_to_log_features: List[str],
    scalers_dir: Path,
    dataloader_dir: Path,
    store_cluster: int,
    item_cluster: int,
    *,
    weight_col: str = "weight",
    window_size: int = 16,
    batch_size: int = 32,
    num_workers: int = 15,
    log_level: str = "INFO",
):
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    scalers_dir.mkdir(parents=True, exist_ok=True)
    dataloader_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
    num_samples = len(df)
    logger.info(
        f"Preparing loaders from {num_samples} samples: {store_cluster}, {item_cluster}"
    )

    all_cols = meta_cols + x_feature_cols
    X_train_raw, y_train_raw, W_train_raw = [], [], []
    X_val_raw, y_val_raw, W_val_raw = [], [], []
    meta_train_raw, meta_val_raw = [], []

    num_windows = num_samples - window_size
    if num_windows <= 0:
        raise ValueError("Not enough samples to generate windows.")

    logger.info(f"Processing {num_windows} windows")

    if weight_col not in df.columns:
        df[weight_col] = 1.0

    for i in tqdm(range(num_windows), desc="Processing windows", unit="window"):
        train_start = i
        train_end = i + window_size
        val_idx = train_end

        df_train = df.iloc[train_start:train_end].fillna(0)
        df_val = df.iloc[[val_idx]].fillna(0)

        all_train = df_train[all_cols].values
        y_train = df_train[label_cols].values
        w_train = df_train[[weight_col]].values

        all_val = df_val[all_cols].values
        y_val = df_val[label_cols].values
        w_val = df_val[[weight_col]].values

        X_train_raw.append(all_train)
        y_train_raw.append(y_train)
        W_train_raw.append(w_train)
        X_val_raw.append(all_val)
        y_val_raw.append(y_val)
        W_val_raw.append(w_val)

        meta_train_raw.append(df_train[meta_cols].reset_index(drop=True))
        meta_val_raw.append(df_val[meta_cols].reset_index(drop=True))

    X_train = np.vstack(X_train_raw)
    del X_train_raw
    y_train = np.vstack(y_train_raw)
    del y_train_raw
    W_train = np.vstack(W_train_raw)
    del W_train_raw
    X_val = np.vstack(X_val_raw)
    del X_val_raw
    y_val = np.vstack(y_val_raw)
    del y_val_raw
    W_val = np.vstack(W_val_raw)
    del W_val_raw

    meta_train_df = pd.concat(meta_train_raw, ignore_index=True)
    del meta_train_raw
    meta_val_df = pd.concat(meta_val_raw, ignore_index=True)
    del meta_val_raw

    combined_cols = meta_cols + x_feature_cols
    col_x_index_map = {col: idx for idx, col in enumerate(combined_cols)}
    x_to_log_idx = [col_x_index_map[c] for c in x_to_log_features]
    x_log_idx = [col_x_index_map[c] for c in x_log_features]
    x_cyc_idx = [col_x_index_map[c] for c in x_cyclical_features]
    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    y_log_idx = [col_y_index_map[c] for c in y_log_features]
    y_to_log_idx = [col_y_index_map[c] for c in y_to_log_features]

    x_tolog_train = np.log1p(
        np.clip(X_train[:, x_to_log_idx].astype(np.float32), 0, None)
    )
    x_log_train = np.clip(X_train[:, x_log_idx].astype(np.float32), 0, None)
    x_sales_train = np.hstack(
        [x_tolog_train.astype(np.float32), x_log_train.astype(np.float32)]
    )
    x_cyc_train = X_train[:, x_cyc_idx].astype(np.float32)
    y_to_log_train = np.log1p(
        np.clip(y_train[:, y_to_log_idx].astype(np.float32), 0, None)
    )
    y_log_train = np.log1p(np.clip(y_train[:, y_log_idx].astype(np.float32), 0, None))
    y_train = np.hstack(
        [y_log_train.astype(np.float32), y_to_log_train.astype(np.float32)]
    )
    x_tolog_val = np.log1p(np.clip(X_val[:, x_to_log_idx].astype(np.float32), 0, None))
    x_log_val = np.clip(X_val[:, x_log_idx].astype(np.float32), 0, None)

    x_sales_val = np.hstack(
        [x_tolog_val.astype(np.float32), x_log_val.astype(np.float32)]
    )
    x_cyc_val = X_val[:, x_cyc_idx].astype(np.float32)
    y_to_log_val = np.log1p(np.clip(y_val[:, y_to_log_idx].astype(np.float32), 0, None))
    y_log_val = np.log1p(np.clip(y_val[:, y_log_idx].astype(np.float32), 0, None))
    y_val = np.hstack([y_log_val.astype(np.float32), y_to_log_val.astype(np.float32)])

    x_sales_scaler = MinMaxScaler().fit(x_sales_train)
    x_cyc_scaler = MinMaxScaler().fit(x_cyc_train)

    X_train_scaled = np.hstack(
        [
            x_sales_scaler.transform(x_sales_train).astype(np.float32),
            x_cyc_scaler.transform(x_cyc_train).astype(np.float32),
        ]
    )
    X_val_scaled = np.hstack(
        [
            x_sales_scaler.transform(x_sales_val).astype(np.float32),
            x_cyc_scaler.transform(x_cyc_val).astype(np.float32),
        ]
    )
    persistent = num_workers > 0
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_scaled),
            torch.tensor(y_train),
            torch.tensor(W_train.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_scaled),
            torch.tensor(y_val),
            torch.tensor(W_val.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent,
    )

    # Debug: Check shapes of inputs/outputs
    x_shape = X_train_scaled.shape[1]
    y_shape = y_train.shape[1]
    w_shape = W_train.shape[1]

    logger.info(f"Input features shape: {x_shape}")
    logger.info(f"Target (y) shape: {y_shape}")
    logger.info(f"Weight shape: {w_shape}")

    # Save loaders and scalers
    fn = dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt"
    torch.save(train_loader, fn)
    logger.info(f"Saved train_loader: {fn}")
    fn = dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt"
    torch.save(val_loader, fn)
    logger.info(f"Saved val_loader: {fn}")
    fn = scalers_dir / f"{store_cluster}_{item_cluster}_x_sales_scaler.pkl"
    pickle.dump(x_sales_scaler, open(fn, "wb"))
    logger.info(f"Saved x_sales_scaler: {fn}")
    fn = scalers_dir / f"{store_cluster}_{item_cluster}_x_cyc_scaler.pkl"
    pickle.dump(x_cyc_scaler, open(fn, "wb"))
    logger.info(f"Saved x_cyc_scaler: {fn}")
    fn = dataloader_dir / f"{store_cluster}_{item_cluster}_train_meta.parquet"
    meta_train_df.to_parquet(fn)
    logger.info(f"Saved meta_train_df: {fn}")
    fn = dataloader_dir / f"{store_cluster}_{item_cluster}_val_meta.parquet"
    meta_val_df.to_parquet(fn)
    logger.info(f"Saved meta_val_df: {fn}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────
# LightningWrapper
# ─────────────────────────────────────────────────────────────────────
class LightningWrapper(pl.LightningModule):
    """Minimal Lightning module wrapping the forecasting model."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        store: int,
        item: int,
        sales_idx: list[int],
        train_mav: float,
        val_mav: float,
        lr: float = 3e-4,
        log_level: str = "INFO",
    ):
        super().__init__()
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.sales_idx = sales_idx
        self.train_mav = train_mav
        self.val_mav = val_mav
        self.loss_fn = NWRMSLELoss()

        # Initialize metrics accumulators for each epoch
        self.train_error_history = []
        self.val_error_history = []
        self.train_mae_history = []
        self.val_mae_history = []
        self.train_rmse_history = []
        self.val_rmse_history = []
        self.best_train_avg_mae = float("inf")
        self.best_val_avg_mae = float("inf")
        self.best_train_avg_rmse = float("inf")
        self.best_val_avg_rmse = float("inf")
        self.best_train_avg_percent_mav = float("inf")
        self.best_val_avg_percent_mav = float("inf")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xb, yb, wb = batch
        if torch.any(torch.isnan(xb)):
            logger.warning(f"NaN in input (xb) at batch {batch_idx}")
        if torch.any(torch.isinf(xb)):
            logger.warning(f"Inf in input (xb) at batch {batch_idx}")
        preds = self.model(xb)
        logger.debug(f"Training Batch {batch_idx} preds:\n", preds)
        if torch.any(torch.isinf(preds)):
            logger.warning(f"Inf in raw preds at batch {batch_idx}")
        preds = torch.clamp(preds, min=1e-6)

        loss = self.loss_fn(preds, yb, wb)

        # Compute training MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_mae = compute_mae(xb, yb, self.model, device, self.sales_idx)
        train_rmse = compute_rmse(preds, yb, device)

        self.train_error_history.append(loss)
        self.train_mae_history.append(train_mae)
        self.train_rmse_history.append(train_rmse)

        # Log the batch loss, MAE, and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", train_mae, prog_bar=True)
        self.log("train_rmse", train_rmse, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb, wb = batch
        if (
            torch.any(torch.isnan(xb))
            or torch.any(torch.isnan(yb))
            or torch.any(torch.isnan(wb))
        ):
            logger.info(f"NaN detected in data at batch {batch_idx}")
        preds = self.model(xb)
        logger.debug(f"Validation Batch {batch_idx} preds:\n", preds)
        if torch.any(torch.isnan(preds)):
            logger.warning(f"NaN detected in predictions at batch {batch_idx}")
        preds = torch.clamp(preds, min=1e-6)
        if torch.all(torch.eq(yb, 0)):
            logger.warning(f"Zero detected in targets at batch {batch_idx}")
        yb = torch.clamp(yb, min=1e-6)
        loss = self.loss_fn(preds, yb, wb)

        # Compute validation MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_mae = compute_mae(xb, yb, self.model, device, self.sales_idx)
        val_rmse = compute_rmse(preds, yb, device)

        self.val_error_history.append(loss)
        self.val_mae_history.append(val_mae)
        self.val_rmse_history.append(val_rmse)

        # Log the batch loss, MAE, and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_rmse", val_rmse, prog_bar=True)

        return loss

    def on_epoch_start(self) -> None:
        logger.info(f"\nModel: {self.model_name}-Epoch {self.current_epoch} started!")

    def on_train_epoch_end(self) -> None:
        logger.info(f"\nModel: {self.model_name}-Epoch {self.current_epoch} ended!")
        # Compute average metrics for the epoch
        avg_train_mae = np.mean(self.train_mae_history)
        avg_train_rmse = np.mean(self.train_rmse_history)
        avg_train_percent_mav = avg_train_mae / self.train_mav * 100
        avg_val_mae = np.mean(self.val_mae_history)
        avg_val_percent_mav = avg_val_mae / self.val_mav * 100
        avg_val_rmse = np.mean(self.val_rmse_history)

        # Log epoch-level metrics
        self.log("avg_train_mav", self.train_mav, prog_bar=False)
        self.log("avg_train_mae", avg_train_mae, prog_bar=False)
        self.log("avg_train_percent_mav", avg_train_percent_mav, prog_bar=False)
        self.log("avg_train_rmse", avg_train_rmse, prog_bar=False)
        self.log("avg_val_mav", self.val_mav, prog_bar=False)
        self.log("avg_val_mae", avg_val_mae, prog_bar=False)
        self.log("avg_val_percent_mav", avg_val_percent_mav, prog_bar=False)
        self.log("avg_val_rmse", avg_val_rmse, prog_bar=False)

        # Reset accumulators for the next epoch
        self.train_mae_history = []
        self.val_mae_history = []
        self.train_rmse_history = []
        self.val_rmse_history = []

        if avg_train_mae < self.best_train_avg_mae:
            self.best_train_avg_mae = avg_train_mae
            self.best_train_avg_mae_percent_mav = avg_train_percent_mav
        if avg_train_rmse < self.best_train_avg_rmse:
            self.best_train_avg_rmse = avg_train_rmse
        if avg_val_mae < self.best_val_avg_mae:
            self.best_val_avg_mae = avg_val_mae
            self.best_val_avg_mae_percent_mav = avg_val_percent_mav
        if avg_val_rmse < self.best_val_avg_rmse:
            self.best_val_avg_rmse = avg_val_rmse

        self.log("best_train_avg_mae", self.best_train_avg_mae, prog_bar=False)
        self.log(
            "best_train_avg_mae_percent_mav",
            self.best_train_avg_mae_percent_mav,
            prog_bar=False,
        )
        self.log("best_train_avg_rmse", self.best_train_avg_rmse, prog_bar=False)
        self.log("best_val_avg_mae", self.best_val_avg_mae, prog_bar=False)
        self.log(
            "best_val_avg_mae_percent_mav",
            self.best_val_avg_mae_percent_mav,
            prog_bar=False,
        )
        self.log("best_val_avg_rmse", self.best_val_avg_rmse, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ─────────────────────────────────────────────────────────────────────
# Helper metrics
# ─────────────────────────────────────────────────────────────────────


def compute_mav(
    loader: DataLoader,
    sales_idx: list[int],  # positions of the sales targets in y
) -> float:
    """
    Mean absolute value of the sales targets in original units.
    Assumes log1p scaling. Cyclical sin/cos columns are ignored.
    """
    abs_sum, count = 0.0, 0
    with torch.no_grad():
        for _, yb, _ in loader:
            y_np = yb.cpu().numpy()

            sales = y_np[:, sales_idx]  # directly select the relevant y columns

            # Undo log1p
            sales = np.expm1(sales)

            abs_sum += np.abs(sales).sum()
            count += sales.size

    return abs_sum / count


def compute_rmse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute Root Mean Squared Error (RMSE)."""
    # Ensure both tensors are on the correct device
    preds = preds.to(device)
    targets = targets.to(device)

    # Detach the tensors from the computation graph to avoid gradient tracking issues
    preds = preds.detach()
    targets = targets.detach()

    # Calculate squared differences
    squared_diff = torch.square(preds - targets)

    # Compute the mean squared error (MSE)
    mse = torch.mean(squared_diff)

    # Return the square root of MSE (RMSE)
    rmse = torch.sqrt(mse)
    return rmse


def compute_mae(
    xb: torch.Tensor,  # Input batch of features
    yb: torch.Tensor,  # Input batch of targets
    model: torch.nn.Module,  # The model used for predictions
    device: torch.device,  # The device (cpu or cuda)
    sales_idx: list[int],  # Indices of the sales targets in the output
) -> float:
    """
    Mean Absolute Error on SALES ONLY, expressed in original units.
    Cyclical targets are excluded from the calculation.
    """

    with torch.no_grad():
        # ----- predictions -----
        preds_np = model(xb.to(device)).cpu().numpy()  # (batch, n_targets)
        p_sales_units = preds_np[:, sales_idx]  # Extract the sales block
        p_sales_units = np.expm1(p_sales_units)  # Convert back to unit sales

        # ----- ground truth ----
        yb_np = yb.cpu().numpy()
        y_sales_units = yb_np[:, sales_idx]
        y_sales_units = np.expm1(y_sales_units)

        # ----- calculate batch-wise MAE -----
        batch_mae = (
            np.abs(p_sales_units - y_sales_units).sum() / y_sales_units.size
        )  # MAE for this batch

    return batch_mae


# def train(
#     today_str: str,
#     model_dir: Path,
#     model_type: ModelType,
#     data_dir: Path,
#     label_cols: list[str],
#     y_log_features: list[str],
#     *,
#     lr: float = 3e-4,
#     epochs: int = 5,
#     seed: int = 2025,
#     enable_progress_bar: bool = True,
#     train_logger: bool = False,
#     log_level: str = "INFO",
# ) -> pd.DataFrame:
#     # Set seed
#     logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
#     pl.seed_everything(seed)

#     # Setup paths
#     dataloader_dir = data_dir / "dataloader"
#     checkpoints_dir = model_dir / "checkpoints"
#     history_dir = model_dir / "history"
#     for d in [checkpoints_dir, history_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # Load loaders and scalers
#     train_loader = torch.load(dataloader_dir / f"{today_str}_train_loader.pt")
#     val_loader = torch.load(dataloader_dir / f"{today_str}_val_loader.pt")

#     # Infer input/output dimensions
#     input_dim = next(iter(train_loader))[0].shape[1]

#     # Compute scaling stats
#     col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
#     y_log_idx = [col_y_index_map[c] for c in y_log_features]

#     train_mav = compute_mav(train_loader, y_log_idx)
#     val_mav = compute_mav(val_loader, y_log_idx)

#     # Build model
#     model_name = f"{today_str}_model_global_{model_type.value}"
#     output_dim = len(label_cols)
#     base_model = model_factory(model_type, input_dim, output_dim)
#     base_model.apply(init_weights)
#     logger.info(f"Built model: {base_model}, epochs={epochs}, lr={lr}")

#     lightning_model = LightningWrapper(
#         base_model,
#         model_name=model_name,
#         lr=lr,
#         sales_idx=y_log_idx,
#         train_mav=train_mav,
#         val_mav=val_mav,
#     )

#     # Callbacks
#     checkpoint_callback = ModelCheckpoint(
#         monitor="best_train_avg_mae",
#         mode="min",
#         save_top_k=1,
#         dirpath=checkpoints_dir,
#         filename=model_name,
#     )

#     # Trainer
#     trainer = pl.Trainer(
#         deterministic=True,
#         max_epochs=epochs,
#         logger=train_logger,
#         enable_progress_bar=enable_progress_bar,
#         callbacks=[checkpoint_callback],
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     )

#     # Train
#     trainer.fit(lightning_model, train_loader, val_loader)

#     # Save training history
#     history = pd.DataFrame(
#         [
#             {
#                 "model_name": model_name,
#                 "train_mav": train_mav,
#                 "val_mav": val_mav,
#                 "best_train_avg_mae": lightning_model.best_train_avg_mae,
#                 "best_val_avg_mae": lightning_model.best_val_avg_mae,
#                 "best_train_avg_rmse": lightning_model.best_train_avg_rmse,
#                 "best_val_avg_rmse": lightning_model.best_val_avg_rmse,
#                 "best_train_avg_mae_percent_mav": lightning_model.best_train_avg_mae_percent_mav,
#                 "best_val_avg_mae_percent_mav": lightning_model.best_val_avg_mae_percent_mav,
#             }
#         ]
#     )
#     history.to_excel(history_dir / f"{today_str}_history.xlsx", index=False)
#     return history


def train_all_models_for_cluster_pair(
    model_types: List[ModelType],
    model_dir: Path,
    dataloader_dir: Path,
    label_cols: list[str],
    y_log_features: list[str],
    store_cluster: int,
    item_cluster: int,
    *,
    history_fn: Optional[Path] = None,
    lr: float = 3e-4,
    epochs: int = 5,
    seed: int = 2025,
    num_workers: int = 15,
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
            history = train_per_cluster_pair(
                model_dir=model_dir,
                model_type=model_type,
                dataloader_dir=dataloader_dir,
                label_cols=label_cols,
                y_log_features=y_log_features,
                store_cluster=store_cluster,
                item_cluster=item_cluster,
                history_fn=history_fn,
                lr=lr,
                epochs=epochs,
                seed=seed,
                num_workers=num_workers,
                enable_progress_bar=enable_progress_bar,
                train_logger=train_logger,
                log_level=log_level,
            )
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
    history_fn: Optional[Path] = None,
    lr: float = 3e-4,
    epochs: int = 5,
    seed: int = 2025,
    num_workers: int = 15,
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
    history_dir = model_dir / "history"
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
        dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt"
    )
    val_loader = torch.load(
        dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt"
    )

    inferred_batch_size = train_loader.batch_size or 32
    logger.info(f"Inferred batch size: {inferred_batch_size}")

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=inferred_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    input_dim = train_dataset.tensors[0].shape[1]
    output_dim = len(label_cols)

    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    y_log_idx = [col_y_index_map[c] for c in y_log_features]
    train_mav = compute_mav(train_loader, y_log_idx)
    val_mav = compute_mav(val_loader, y_log_idx)

    model_name = f"model_{store_cluster}_{item_cluster}_{model_type.value}"
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
        deterministic=True,
        max_epochs=epochs,
        logger=train_logger,
        enable_progress_bar=enable_progress_bar,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
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

    # Load existing history CSV if provided and exists
    if history_fn is not None and history_fn.exists():
        previous_history = pd.read_csv(history_fn)
        history = pd.concat([previous_history, history], ignore_index=True)

    # Save updated history
    if history_fn is not None:
        history.to_csv(history_fn, index=False)

    return history


# def train_per_cluster_pair(
#     model_dir: Path,
#     model_type: ModelType,
#     dataloader_dir: Path,
#     label_cols: list[str],
#     y_log_features: list[str],
#     store_cluster: int,
#     item_cluster: int,
#     *,
#     lr: float = 3e-4,
#     epochs: int = 5,
#     seed: int = 2025,
#     num_workers: int = 15,
#     enable_progress_bar: bool = True,
#     train_logger: bool = False,
#     log_level: str = "INFO",
# ) -> pd.DataFrame:
#     logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
#     pl.seed_everything(seed)
#     logger.info(f"Training model for cluster pair: ({store_cluster}, {item_cluster})")

#     # Setup paths
#     # dataloader_dir = data_dir / "dataloader"
#     checkpoints_dir = model_dir / "checkpoints"
#     history_dir = model_dir / "history"
#     for d in [checkpoints_dir, history_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # Load pre-saved metadata (optional, for logging or inspection)
#     train_meta_fn = (
#         dataloader_dir / f"{store_cluster}_{item_cluster}_train_meta.parquet"
#     )
#     val_meta_fn = dataloader_dir / f"{store_cluster}_{item_cluster}_val_meta.parquet"

#     meta_df = pd.read_parquet(train_meta_fn)
#     val_meta_df = pd.read_parquet(val_meta_fn)

#     # Skip if metadata is empty
#     if meta_df.empty or val_meta_df.empty:
#         logger.warning(
#             f"Skipping pair ({store_cluster}, {item_cluster}) due to insufficient data."
#         )
#         return pd.DataFrame()  # or `continue` if inside a loop

#     # Load pre-filtered dataloaders
#     train_loader_path = (
#         dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt"
#     )
#     val_loader_path = dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt"

#     train_loader = torch.load(train_loader_path)
#     val_loader = torch.load(val_loader_path)

#     inferred_batch_size = train_loader.batch_size or 32
#     logger.info(f"Inferred batch size: {inferred_batch_size}")

#     # Re-wrap into new loaders if necessary (e.g., to control num_workers)
#     train_dataset = train_loader.dataset
#     val_dataset = val_loader.dataset

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=inferred_batch_size,
#         num_workers=num_workers,
#         persistent_workers=True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=inferred_batch_size,
#         num_workers=num_workers,
#         persistent_workers=True,
#     )

#     # Infer dimensions
#     input_dim = train_dataset.tensors[0].shape[1]
#     output_dim = len(label_cols)

#     # Compute MAVs
#     col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
#     y_log_idx = [col_y_index_map[c] for c in y_log_features]
#     train_mav = compute_mav(train_loader, y_log_idx)
#     logger.info(f"{store_cluster}_{item_cluster} Train MAV: {train_mav}")
#     val_mav = compute_mav(val_loader, y_log_idx)
#     logger.info(f"{store_cluster}_{item_cluster} Val MAV: {val_mav}")

#     model_name = f"model_{store_cluster}_{item_cluster}_{model_type.value}"
#     base_model = model_factory(model_type, input_dim, output_dim)
#     base_model.apply(init_weights)

#     lightning_model = LightningWrapper(
#         base_model,
#         model_name=model_name,
#         lr=lr,
#         sales_idx=y_log_idx,
#         train_mav=train_mav,
#         val_mav=val_mav,
#     )

#     checkpoint_callback = ModelCheckpoint(
#         monitor="best_train_avg_mae",
#         mode="min",
#         save_top_k=1,
#         dirpath=checkpoints_dir,
#         filename=model_name,
#     )

#     trainer = pl.Trainer(
#         deterministic=True,
#         max_epochs=epochs,
#         logger=train_logger,
#         enable_progress_bar=enable_progress_bar,
#         callbacks=[checkpoint_callback],
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     )

#     trainer.fit(lightning_model, train_loader, val_loader)

#     # Log history for this model
#     history = {
#         "model_name": model_name,
#         "store_cluster": store_cluster,
#         "item_cluster": item_cluster,
#         "train_mav": train_mav,
#         "val_mav": val_mav,
#         "best_train_avg_mae": lightning_model.best_train_avg_mae,
#         "best_val_avg_mae": lightning_model.best_val_avg_mae,
#         "best_train_avg_rmse": lightning_model.best_train_avg_rmse,
#         "best_val_avg_rmse": lightning_model.best_val_avg_rmse,
#         "best_train_avg_mae_percent_mav": lightning_model.best_train_avg_mae_percent_mav,
#         "best_val_avg_mae_percent_mav": lightning_model.best_val_avg_mae_percent_mav,
#     }
#     all_histories.append(history)

#     # Save full history
#     history_df = pd.DataFrame(all_histories)
#     fn = history_dir / f"{store_cluster}_{item_cluster}_xls"
#     history_df.to_excel(fn, index=False)
#     return history_df


def load_model(model_path: str) -> Tuple[int, nn.Module, List[str]]:
    """Loads a saved model from the given path and returns it along with the
    associated store_item identifier and feature columns.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        tuple: A tuple containing the store_item identifier, the model, and its
            feature columns.
    """
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    store_item_id = checkpoint["sid"]
    model_state_dict = checkpoint["model_state_dict"]
    feature_columns = checkpoint["feature_cols"]

    model = ShallowNN(input_dim=len(feature_columns))
    model.load_state_dict(model_state_dict)
    model.eval()

    return store_item_id, model, feature_columns


def load_scalers(
    model_dir: str = "../output/data/", date_str: str = ""
) -> Tuple[Dict[str, MinMaxScaler], Dict[str, MinMaxScaler]]:
    x_scalers = {}
    y_scalers = {}

    for filename in os.listdir(model_dir):
        if date_str and not filename.startswith(f"{date_str}"):
            continue

        full_path = os.path.join(model_dir, filename)
        parts = filename.replace(".pkl", "").split("_")

        if filename.startswith(f"{date_str}_x_scaler_"):
            # Join last two parts to form store_item ID
            store_item = f"{parts[-2]}_{parts[-1]}"
            with open(full_path, "rb") as f:
                x_scalers[store_item] = pickle.load(f)

        elif filename.startswith(f"{date_str}_y_scaler_"):
            store_item = f"{parts[-2]}_{parts[-1]}"
            with open(full_path, "rb") as f:
                y_scalers[store_item] = pickle.load(f)

    return x_scalers, y_scalers


def load_latest_models_from_checkpoints(
    date: str,
    checkpoints_dir: Path,
    model_type: type[LightningModule],
    input_dim: int,
    output_dim: int,
) -> dict[tuple[int, int], LightningWrapper]:
    """
    Load the latest checkpoint per (store_cluster, item_cluster).
    Prioritizes -v3 > -v2 > -v1 > base.
    """
    # Group all matching checkpoints by (sc, ic)
    candidates = defaultdict(list)
    pattern = re.compile(rf"{date}_model_sc(\d+)_ic(\d+)_.*?(?:-v(\d+))?\.ckpt")

    for ckpt_path in checkpoints_dir.glob(f"{date}_model_sc*_ic*_*.ckpt"):
        match = pattern.match(ckpt_path.name)
        if not match:
            continue
        sc, ic, version = int(match[1]), int(match[2]), match[3]
        version = int(version) if version is not None else 0
        candidates[(sc, ic)].append((version, ckpt_path))

    model_dict = {}
    for (sc, ic), versioned_paths in candidates.items():
        # Sort versions descending, take the highest
        best_ckpt = max(versioned_paths, key=lambda x: x[0])[1]
        try:
            model = model_factory(model_type, input_dim, output_dim)
            wrapper = LightningWrapper.load_from_checkpoint(
                best_ckpt, model=model, strict=False
            )
            model_dict[(sc, ic)] = wrapper
        except Exception as e:
            print(f"Skipping {best_ckpt.name}: {e}")

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
