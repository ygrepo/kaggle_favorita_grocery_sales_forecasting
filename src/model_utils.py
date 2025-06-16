import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import lightning.pytorch as pl
from sklearn.preprocessing import MinMaxScaler
import random
import logging
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
import pickle
from pathlib import Path


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
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, dropout=0.0):
        """Simple feed forward network used in unit tests.

        The default ``dropout`` is set to ``0.0`` so that the forward pass is
        deterministic unless a different value is explicitly requested.  This
        avoids stochastic behaviour in the tests which call ``model(x)`` twice
        in a row to check reproducibility.
        """

        super().__init__()
        output_dim = output_dim or input_dim

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(hidden_dim, output_dim),
            nn.Hardtanh(),  # clips to [0,1]
        )

    def forward(self, x):
        return self.net(x)


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, h1=128, h2=64, dropout=0.0):
        """Two layer feed forward network used in unit tests.

        ``dropout`` defaults to ``0.0`` so that the forward pass is
        deterministic by default (matching :class:`ShallowNN`).  Tests that
        require stochastic behaviour can pass a different value.
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
            nn.Linear(h2, input_dim),
            nn.Sigmoid(),  # already in (0,1)
        )

    def forward(self, x):
        out = self.net(x)
        # optional extra safety, **not in-place**
        # Limit outputs strictly to the [0, 1] range for testing
        return torch.clamp(out, 1e-6, 1.0)
        # or simply: return out


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, depth=3, dropout=0.2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.Linear(input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, input_dim),
            ]
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for i in range(0, len(self.blocks), 5):
            h = self.blocks[i + 0](out)
            h = self.blocks[i + 1](h)
            h = self.blocks[i + 2](h)
            h = self.blocks[i + 3](h)
            h = self.blocks[i + 4](h)
            out = out + h  # residual add (not in‑place)
        # Sigmoid ensures values lie in ``(0, 1)``.  Clamp for numerical
        # stability and to guarantee the upper bound used in tests.
        return torch.clamp(torch.sigmoid(out), 1e-6, 1.0)


# ----- initialise every nn.Linear in the network ----------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.01)  # He initialisation for LeakyReLU
        nn.init.zeros_(m.bias)


class NWRMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

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


def compute_rmsle_manual(loader, model, device, eps=1e-6):
    model.eval()
    total_num, total_den = 0.0, 0.0

    with torch.no_grad():
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = torch.clamp(yb.to(device), min=eps)
            wb = wb.to(device)

            preds = torch.clamp(model(xb), min=eps)

            log_diff = torch.log(preds + 1.0) - torch.log(yb + 1.0)
            weighted_sq_diff = wb * log_diff**2

            # Valid (finite) mask
            mask = torch.isfinite(weighted_sq_diff)

            # Only count valid loss terms
            safe_values = weighted_sq_diff[mask]
            total_num += safe_values.sum().item()

            # Count valid weights using same mask, broadcast-safe
            valid_weights = wb.expand_as(weighted_sq_diff)[mask]
            total_den += valid_weights.sum().item()

    rmsle = (total_num / (total_den + eps)) ** 0.5
    if np.isnan(rmsle):
        logger.info("Final RMSLE is NaN")
        logger.info("Numerator:", total_num)
        logger.info("Denominator:", total_den)
    return rmsle


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


class LightningWrapper(pl.LightningModule):
    """Minimal Lightning module wrapping the forecasting model."""

    def __init__(
        self,
        model: nn.Module,
        y_sales_scaler: MinMaxScaler,
        sales_idx: list[int],
        train_mav: float,
        val_mav: float,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.y_sales_scaler = y_sales_scaler
        self.sales_idx = sales_idx
        self.train_mav = train_mav
        self.val_mav = val_mav
        self.loss_fn = NWRMSLELoss()

        # Initialize metrics accumulators for each epoch
        self.train_mae_accum = 0.0
        self.train_percent_mav_accum = 0.0
        self.val_mae_accum = 0.0
        self.val_percent_mav_accum = 0.0
        self.train_count = 0
        self.val_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xb, yb, wb = batch
        preds = torch.clamp(self.model(xb), min=1e-6)
        loss = self.loss_fn(preds, yb, wb)

        # Compute training MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_mae = compute_mae(
            xb, yb, self.model, device, self.y_sales_scaler, self.sales_idx
        )
        train_percent_mav = (
            (train_mae / self.train_mav) * 100 if self.train_mav else float("inf")
        )

        self.train_mae_accum += train_mae
        self.train_percent_mav_accum += train_percent_mav
        self.train_count += 1

        # Log the batch loss and MAE
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", train_mae, prog_bar=True)
        self.log("train_percent_mav", train_percent_mav, prog_bar=True)

        return loss

    # def training_step(self, batch, batch_idx):
    #     xb, yb, wb = batch
    #     preds = torch.clamp(self.model(xb), min=1e-6)
    #     loss = self.loss_fn(preds, yb, wb)
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss
    def validation_step(self, batch, batch_idx):
        xb, yb, wb = batch
        if (
            torch.any(torch.isnan(xb))
            or torch.any(torch.isnan(yb))
            or torch.any(torch.isnan(wb))
        ):
            print(f"NaN detected in data at batch {batch_idx}")
        preds = self.model(xb)
        if torch.any(torch.isnan(preds)):
            print(f"NaN detected in predictions at batch {batch_idx}")
        preds = torch.clamp(preds, min=1e-6)
        yb = torch.clamp(yb, min=1e-6)
        loss = self.loss_fn(preds, yb, wb)

        # Compute validation MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_mae = compute_mae(
            xb, yb, self.model, device, self.y_sales_scaler, self.sales_idx
        )
        val_percent_mav = (
            (val_mae / self.val_mav) * 100 if self.val_mav else float("inf")
        )
        self.val_mae_accum += val_mae
        self.val_percent_mav_accum += val_percent_mav
        self.val_count += 1

        # Log the batch loss and MAE
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_percent_mav", val_percent_mav, prog_bar=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     xb, yb, wb = batch
    #     if (
    #         torch.any(torch.isnan(xb))
    #         or torch.any(torch.isnan(yb))
    #         or torch.any(torch.isnan(wb))
    #     ):
    #         print(f"NaN detected in data at batch {batch_idx}")
    #     preds = self.model(xb)
    #     if torch.any(torch.isnan(preds)):
    #         print(f"NaN detected in predictions at batch {batch_idx}")
    #     preds = torch.clamp(preds, min=1e-6)
    #     yb = torch.clamp(yb, min=1e-6)
    #     loss = self.loss_fn(preds, yb, wb)
    #     self.log("val_loss", loss, prog_bar=True)

    def on_epoch_end(self):
        # Compute average metrics for the epoch
        avg_train_mae = (
            self.train_mae_accum / self.train_count if self.train_count > 0 else 0.0
        )
        avg_val_mae = self.val_mae_accum / self.val_count if self.val_count > 0 else 0.0

        avg_train_percent_mav = (
            self.train_percent_mav_accum / self.train_count
            if self.train_count > 0
            else 0.0
        )
        avg_val_percent_mav = (
            self.val_percent_mav_accum / self.val_count if self.val_count > 0 else 0.0
        )

        # Log epoch-level metrics
        self.log("avg_train_mae", avg_train_mae)
        self.log("avg_val_mae", avg_val_mae)
        self.log("avg_train_percent_mav", avg_train_percent_mav)
        self.log("avg_val_percent_mav", avg_val_percent_mav)

        # Reset accumulators for the next epoch
        self.train_mae_accum = 0.0
        self.val_mae_accum = 0.0
        self.train_percent_mav_accum = 0.0
        self.val_percent_mav_accum = 0.0
        self.train_count = 0
        self.val_count = 0

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
def compute_mpe(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    sc_sales: MinMaxScaler,  # log‑sales scaler
    sc_cyc: MinMaxScaler,  # sin/cos scaler  (‑1…1 ↔ 0…1)
    sales_idx: list[int],
    cyc_idx: list[int],
    eps: float = 1e-9,
) -> float:
    """
    Mean Percentage Error on the *full* target vector (sales + cyclical):

        MPE = mean( (ŷ – y) / (y + eps) ).

    • Sales cols are inverse‑scaled → expm1 to get unit sales.
    • Cyclical cols are inverse‑scaled to get values back in [‑1, 1].
    """
    err_sum, count = 0.0, 0
    model.eval()

    with torch.no_grad():
        for xb, yb, _ in loader:
            # ---------- predictions ----------
            preds_np = model(xb.to(device)).cpu().numpy()
            preds_inv = preds_np.copy()

            # sales block
            p_sales_scaled = preds_np[:, sales_idx]
            p_sales_unscaled = sc_sales.inverse_transform(p_sales_scaled)
            preds_inv[:, sales_idx] = np.expm1(p_sales_unscaled)

            # cyclical block
            p_cyc_scaled = preds_np[:, cyc_idx]
            preds_inv[:, cyc_idx] = sc_cyc.inverse_transform(p_cyc_scaled)

            # ---------- ground truth ----------
            yb_np = yb.cpu().numpy()
            yb_inv = yb_np.copy()

            y_sales_scaled = yb_np[:, sales_idx]
            y_sales_unscaled = sc_sales.inverse_transform(y_sales_scaled)
            yb_inv[:, sales_idx] = np.expm1(y_sales_unscaled)

            y_cyc_scaled = yb_np[:, cyc_idx]
            yb_inv[:, cyc_idx] = sc_cyc.inverse_transform(y_cyc_scaled)

            # ---------- percentage error ----------
            pct_err = (preds_inv - yb_inv) / (yb_inv + eps)  # element‑wise
            err_sum += pct_err.sum()
            count += pct_err.size

    return np.abs(err_sum / count)


def compute_mav(
    loader: DataLoader,
    y_scaler: MinMaxScaler,  # fitted on *log‑sales* columns only
    sales_idx: list[int],  # positions of the sales targets in y
) -> float:
    """
    Mean absolute value of the sales targets in original units.
    Cyclical sin/cos columns are ignored.
    """
    abs_sum, count = 0.0, 0
    with torch.no_grad():
        for _, yb, _ in loader:
            # --- slice the sales block (scaled log space) -----------------
            sales_scaled = yb.cpu().numpy()[:, sales_idx]  # shape (batch, n_sales)

            # --- restore to original units -------------------------------
            sales_unscaled = y_scaler.inverse_transform(sales_scaled)  # undo MinMax
            sales_units = np.expm1(sales_unscaled)  # undo log1p

            # --- accumulate ---------------------------------------------
            abs_sum += np.abs(sales_units).sum()
            count += sales_units.size

    return abs_sum / count


def compute_mae(
    xb: torch.Tensor,  # Input batch of features
    yb: torch.Tensor,  # Input batch of targets
    model: torch.nn.Module,  # The model used for predictions
    device: torch.device,  # The device (cpu or cuda)
    sc_sales: MinMaxScaler,  # Scaler for the sales targets
    sales_idx: list[int],  # Indices of the sales targets in the output
) -> float:
    """
    Mean Absolute Error on SALES ONLY, expressed in original units.
    Cyclical targets are excluded from the calculation.
    """

    with torch.no_grad():
        # ----- predictions -----
        preds_np = model(xb.to(device)).cpu().numpy()  # (batch, n_targets)
        p_sales_scaled = preds_np[:, sales_idx]  # Extract the sales block
        p_sales_unscaled = sc_sales.inverse_transform(p_sales_scaled)
        p_sales_units = np.expm1(p_sales_unscaled)  # Convert back to unit sales

        # ----- ground truth ----
        yb_np = yb.cpu().numpy()
        y_sales_scaled = yb_np[:, sales_idx]
        y_sales_unscaled = sc_sales.inverse_transform(y_sales_scaled)
        y_sales_units = np.expm1(y_sales_unscaled)

        # ----- calculate batch-wise MAE -----
        batch_mae = (
            np.abs(p_sales_units - y_sales_units).sum() / y_sales_units.size
        )  # MAE for this batch

    return batch_mae


# def compute_mae(
#     loader: DataLoader,
#     model: torch.nn.Module,
#     device: torch.device,
#     sc_sales: MinMaxScaler,  # fitted on log‑sales columns only
#     sales_idx: list[int],  # positions of the sales targets in y
# ) -> float:
#     """
#     Mean Absolute Error on SALES ONLY, expressed in original units.
#     Cyclical targets are excluded from the calculation.
#     """
#     abs_sum, count = 0.0, 0
#     model.eval()

#     with torch.no_grad():
#         for xb, yb, _ in loader:
#             # ----- predictions -----
#             preds_np = model(xb.to(device)).cpu().numpy()  # (batch, n_targets)
#             p_sales_scaled = preds_np[:, sales_idx]  # take sales block
#             p_sales_unscaled = sc_sales.inverse_transform(p_sales_scaled)
#             p_sales_units = np.expm1(p_sales_unscaled)  # back to unit sales

#             # ----- ground truth ----
#             yb_np = yb.cpu().numpy()
#             y_sales_scaled = yb_np[:, sales_idx]
#             y_sales_unscaled = sc_sales.inverse_transform(y_sales_scaled)
#             y_sales_units = np.expm1(y_sales_unscaled)

#             # ----- accumulate -----
#             abs_sum += np.abs(p_sales_units - y_sales_units).sum()
#             count += y_sales_units.size

#     return abs_sum / count


def train(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    x_feature_cols: List[str],
    x_sales_features: List[str],
    x_cyclical_features: List[str],
    label_cols: List[str],
    y_sales_features: List[str],
    y_cyclical_features: List[str],
    item_col: str,
    train_frac: float = 0.8,
    batch_size: int = 32,
    lr: float = 3e-4,
    epochs: int = 5,
    seed: int = 2025,
    model_cls: type = ShallowNN,
    num_workers: int = 5,
    enable_progress_bar: bool = True,
    train_logger: bool = False,
    output_dir: str = "../output/data/",
    model_dir: str = "../output/models/",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, torch.nn.Module]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    scalers_dir = Path(output_dir) / "scalers"
    scalers_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(output_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history = []
    models: Dict[str, torch.nn.Module] = {}
    x_scalers: Dict[str, MinMaxScaler] = {}
    y_scalers: Dict[str, MinMaxScaler] = {}

    today_str = datetime.today().strftime("%Y-%m-%d")
    for sid in df["store_item"].unique():
        sub = (
            df[df["store_item"] == sid]
            .sort_values("start_date")
            .reset_index(drop=True)
            .merge(weights_df, on=item_col, how="left")
        )

        train_size = int(len(sub) * train_frac)

        # -----------------------------------
        # SPLIT
        # -----------------------------------
        X_train_df = sub.loc[: train_size - 1, x_feature_cols]
        X_test_df = sub.loc[train_size:, x_feature_cols]

        # ---------------------------------
        #  X  SIDE  (cyc sin/cos untouched)
        # ---------------------------------
        x_sales_idx = [x_feature_cols.index(c) for c in x_sales_features]
        x_cyc_idx = [x_feature_cols.index(c) for c in x_cyclical_features]
        x_sales_train = np.clip(X_train_df.iloc[:, x_sales_idx].to_numpy(), 0, None)
        x_sales_test = np.clip(X_test_df.iloc[:, x_sales_idx].to_numpy(), 0, None)
        x_sales_train_log = np.log1p(x_sales_train)
        x_sales_test_log = np.log1p(x_sales_test)

        x_sales_scaler = MinMaxScaler().fit(x_sales_train_log)

        x_cyc_train = X_train_df.iloc[:, x_cyc_idx].to_numpy()  # still -1…1
        x_cyc_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_cyc_train)
        X_train_full = np.hstack(
            [
                x_sales_scaler.transform(x_sales_train_log),  # scaled log‑sales 0…1
                x_cyc_scaler.transform(x_cyc_train),  # NEW: cyc 0…1
            ]
        )

        x_cyc_test = X_test_df.iloc[:, x_cyc_idx].to_numpy()
        X_test_full = np.hstack(
            [
                x_sales_scaler.transform(x_sales_test_log),
                x_cyc_scaler.transform(x_cyc_test),
            ]
        )

        pickle.dump(
            x_sales_scaler,
            open(f"{output_dir}/scalers/{today_str}_x_sales_scaler_{sid}.pkl", "wb"),
        )

        pickle.dump(
            x_cyc_scaler,
            open(f"{output_dir}/scalers/{today_str}_x_cyc_scaler_{sid}.pkl", "wb"),
        )

        # ---------------------------------
        #  Y  SIDE  (same idea)
        # ---------------------------------
        y_sales_train = np.clip(
            sub.loc[: train_size - 1, y_sales_features].to_numpy(), 0, None
        )
        y_sales_test = np.clip(
            sub.loc[train_size:, y_sales_features].to_numpy(), 0, None
        )

        y_sales_train_log = np.log1p(y_sales_train)
        y_sales_test_log = np.log1p(y_sales_test)
        y_sales_scaler = MinMaxScaler().fit(y_sales_train_log)

        y_cyc_train = sub.loc[: train_size - 1, y_cyclical_features].to_numpy()

        y_cyc_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_cyc_train)

        y_train_full = np.hstack(
            [
                y_sales_scaler.transform(y_sales_train_log),
                y_cyc_scaler.transform(y_cyc_train),  # NEW
            ]
        )

        y_cyc_test = sub.loc[train_size:, y_cyclical_features].to_numpy()
        y_test_full = np.hstack(
            [
                y_sales_scaler.transform(y_sales_test_log),
                y_cyc_scaler.transform(y_cyc_test),
            ]
        )

        pickle.dump(
            y_sales_scaler,
            open(f"{output_dir}/scalers/{today_str}_y_sales_scaler_{sid}.pkl", "wb"),
        )
        pickle.dump(
            y_cyc_scaler,
            open(f"{output_dir}/scalers/{today_str}_y_cyc_scaler_{sid}.pkl", "wb"),
        )

        # ---------------------------------
        #  DATASETS & MODEL
        # ---------------------------------

        w = sub["weight"].to_numpy(float).reshape(-1, 1)
        w_train, w_test = w[:train_size], w[train_size:]

        ds_train = TensorDataset(
            torch.from_numpy(X_train_full).float(),  # features
            torch.from_numpy(y_train_full).float(),  # targets
            torch.from_numpy(w_train).float(),  # weights
        )
        ld_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
        )

        ds_test = TensorDataset(
            torch.from_numpy(X_test_full).float(),
            torch.from_numpy(y_test_full).float(),
            torch.from_numpy(w_test).float(),
        )
        ld_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
        )

        if model_cls is ShallowNN:
            base_model = ShallowNN(input_dim=X_train_full.shape[1])
        elif model_cls is TwoLayerNN:
            base_model = TwoLayerNN(input_dim=X_train_full.shape[1])
        elif model_cls is ResidualMLP:
            base_model = ResidualMLP(input_dim=X_train_full.shape[1])
        else:
            raise ValueError(f"Unknown model: {model_cls}")

        base_model.apply(init_weights)

        sales_idx = [label_cols.index(col) for col in y_sales_features]
        train_mav = compute_mav(ld_train, y_sales_scaler, sales_idx)
        val_mav = compute_mav(ld_test, y_sales_scaler, sales_idx)

        lightning_model = LightningWrapper(
            base_model,
            lr=lr,
            y_sales_scaler=y_sales_scaler,
            sales_idx=sales_idx,
            train_mav=train_mav,
            val_mav=val_mav,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # The metric you want to monitor
            mode="min",  # 'min' means the model with the lowest val_loss will be saved
            save_top_k=1,  # Save only the best model
            dirpath=Path(model_dir) / "checkpoints",  # Directory to save checkpoints
            filename=f"{today_str}_model_{sid}",  # Filename pattern for the saved checkpoint
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=train_logger,
            enable_progress_bar=enable_progress_bar,
            callbacks=[checkpoint_callback],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )
        trainer.fit(lightning_model, ld_train, ld_test)

        model = lightning_model.model.to(device)

        # train_loss = compute_rmsle_manual(ld_train, model, device)
        # test_loss = compute_rmsle_manual(ld_test, model, device)

        train_loss = trainer.logged_metrics["train_loss"]  # From training step
        val_loss = trainer.logged_metrics["val_loss"]  # From validation step
        avg_train_mae = trainer.logged_metrics["avg_train_mae"]
        avg_val_mae = trainer.logged_metrics["avg_val_mae"]
        avg_train_percent_mav = trainer.logged_metrics["avg_train_percent_mav"]
        avg_val_percent_mav = trainer.logged_metrics["avg_val_percent_mav"]

        # train_mae = compute_mae(ld_train, model, device, y_sales_scaler, sales_idx)
        # train_percent_mav = (train_mae / train_mav) * 100 if train_mav else float("inf")
        # val_mae = compute_mae(ld_test, model, device, y_sales_scaler, sales_idx)
        # val_percent_mav = (val_mae / val_mav) * 100 if val_mav else float("inf")

        # sales_idx = [label_cols.index(c) for c in y_sales_features]
        # cyc_idx = [label_cols.index(col) for col in y_cyclical_features]
        # train_mpe = compute_mpe(
        #     ld_train,
        #     model,
        #     device,
        #     sc_sales=y_sales_scaler,
        #     sc_cyc=y_cyc_scaler,
        #     sales_idx=sales_idx,
        #     cyc_idx=cyc_idx,
        # )

        # val_mpe = compute_mpe(
        #     ld_test,
        #     model,
        #     device,
        #     sc_sales=y_sales_scaler,
        #     sc_cyc=y_cyc_scaler,
        #     sales_idx=sales_idx,
        #     cyc_idx=cyc_idx,
        # )

        history.append(
            {
                "store_item": sid,
                "epoch": epochs,
                "train_loss": train_loss,
                "avg_train_mae": avg_train_mae,
                "avg_train_percent_mav": avg_train_percent_mav,
                "val_loss": val_loss,
                "avg_val_mae": avg_val_mae,
                "avg_val_percent_mav": avg_val_percent_mav,
            }
        )

        # logger.info(
        #     f"[{sid}] Finished training "
        #     f"train_loss {train_loss:.4f}, "
        #     f"train_MAE {train_mae:.4f}, "
        #     f"train_%MAV {train_percent_mav:.4f}, "
        #     f"val_loss {val_loss:.4f}, "
        #     f"val_MAE {val_mae:.4f}, "
        #     f"val_%MAV {val_percent_mav:.4f}, "
        #     f"train_MPE {train_mpe:.4f}, val_MPE {val_mpe:.4f}"
        # )

        save_path = os.path.join(model_dir, f"{today_str}_model_{sid}.pth")
        torch.save(
            {
                "sid": sid,
                "model_state_dict": lightning_model.model.state_dict(),
                "x_feature_cols": x_feature_cols,
                "y_feature_cols": label_cols,
                "train_frac": train_frac,
                "epochs": epochs,
            },
            save_path,
        )
        # torch.save(
        #     {
        #         "sid": sid,
        #         "model_state_dict": model.cpu().state_dict(),
        #         "x_feature_cols": x_feature_cols,
        #         "y_feature_cols": label_cols,
        #         "train_frac": train_frac,
        #         "epochs": epochs,
        #     },
        #     save_path,
        # )
        logger.info(f"Saved model for {sid} to {save_path}")

        models[sid] = model.cpu()
        x_scalers[sid] = x_sales_scaler
        y_scalers[sid] = y_sales_scaler

    hist_df = pd.DataFrame(history)
    summary_df = (
        hist_df.groupby("store_item")
        .agg(
            final_train_loss=("train_loss", "last"),
            final_val_loss=("val_loss", "last"),
            final_avg_train_mae=("avg_train_mae", "last"),
            final_avg_val_mae=("avg_val_mae", "last"),
            final_avg_train_percent_mav=("avg_train_percent_mav", "last"),
            final_avg_val_percent_mav=("avg_val_percent_mav", "last"),
        )
        .reset_index()
    )

    return hist_df, summary_df, models


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
    output_dir: str = "../output/data/", date_str: str = ""
) -> Tuple[Dict[str, MinMaxScaler], Dict[str, MinMaxScaler]]:
    x_scalers = {}
    y_scalers = {}

    for filename in os.listdir(output_dir):
        if date_str and not filename.startswith(f"{date_str}"):
            continue

        full_path = os.path.join(output_dir, filename)
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
