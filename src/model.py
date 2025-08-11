import torch
from torch import nn
import lightning.pytorch as pl
import logging
import numpy as np
from pathlib import Path
from enum import Enum
from torch.nn import functional as F
from torchmetrics import Metric
from typing import List
import pandas as pd

from src.utils import get_logger


# Set up logger
logger = logging.getLogger(__name__)


def extract_model_name(model_name_chkp: Path) -> str:
    """
    Extract the model name without store/item cluster prefix.
    E.g. from "7_7_ShallowNN.ckpt" → "ShallowNN"
    """
    parts = model_name_chkp.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected checkpoint name format: {model_name_chkp}")

    return "_".join(parts[2:])


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
    def __init__(self, input_dim, output_dim=3, h1=64, h2=32, dropout=0.4):
        # def __init__(self, input_dim, output_dim=3, h1=128, h2=64, dropout=0.2):
        """
        Two-layer feedforward NN for log-transformed targets.
        All outputs are unbounded real values, no final activation.
        """
        super().__init__()

        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.LeakyReLU(),
            dropout_layer,
            nn.Linear(h2, output_dim),
            nn.Softplus(beta=1.0),  # guarantees strictly positive outputs
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(
        self, input_dim, output_dim=3, hidden_dim: int = 128, depth=3, dropout=0.2
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # changed from BatchNorm1d
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),  # residual connection
                )
            )
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(out)
        return self.out_proj(out)


# Enum for model types
class ModelType(Enum):
    SHALLOW_NN = "ShallowNN"
    TWO_LAYER_NN = "TwoLayerNN"
    RESIDUAL_MLP = "ResidualMLP"


MODEL_TYPES = list(ModelType)


# Model Factory Function
def model_factory(
    model_type: ModelType,
    input_dim: int,
    hidden_dim: int,
    h1: int,
    h2: int,
    depth: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    """Factory function to return the correct model based on the model_type."""
    if model_type == ModelType.SHALLOW_NN:
        return ShallowNN(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == ModelType.TWO_LAYER_NN:
        return TwoLayerNN(input_dim, output_dim, h1, h2, dropout)
    elif model_type == ModelType.RESIDUAL_MLP:
        return ResidualMLP(input_dim, output_dim, hidden_dim, depth, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_factory_from_str(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    h1: int,
    h2: int,
    depth: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    """Factory function to return the correct model based on the model_type."""
    if model_type == "ShallowNN":
        return ShallowNN(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == "TwoLayerNN":
        return TwoLayerNN(input_dim, output_dim, h1, h2, dropout)
    elif model_type == "ResidualMLP":
        return ResidualMLP(input_dim, output_dim, hidden_dim, depth, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        log_diff = y_pred - y_true  # Already log-scaled
        num = torch.sum(w * log_diff**2)
        den = torch.sum(w)
        return torch.sqrt(num / den)


class PercentMAVLossLog1p(torch.nn.Module):
    """
    Batch-wise normalized MAE in original units for log1p-scaled preds/targets.
    Equals (%MAV / 100) computed on the batch, using the SAME mask: target > 0.
    Supports optional per-row weights `w` (shape [B]).
    """

    def __init__(self, sales_idx: list[int], eps: float = 1e-12):
        super().__init__()
        self.sales_idx = sales_idx
        self.eps = eps

    def forward(
        self,
        y_pred_log: torch.Tensor,
        y_true_log: torch.Tensor,
        w: torch.Tensor | None = None,
    ):
        # Select only the sales targets and go back to original units
        y_pred = torch.expm1(y_pred_log[:, self.sales_idx])
        y_true = torch.expm1(y_true_log[:, self.sales_idx])

        # Same mask as your metric: finite & positive targets only
        valid = torch.isfinite(y_true) & torch.isfinite(y_pred)
        mask = valid & (y_true > 0)

        # Sum absolute error and target volume per row (across target columns)
        per_row_err = torch.where(mask, (y_pred - y_true).abs(), 0.0).sum(dim=1)
        per_row_vol = torch.where(mask, y_true.abs(), 0.0).sum(dim=1)

        if w is not None:
            w = w.reshape(-1)
            num = (w * per_row_err).sum()
            den = (w * per_row_vol).sum()
        else:
            num = per_row_err.sum()
            den = per_row_vol.sum()

        return num / (den + self.eps)  # this is %MAV / 100


class MSELossOriginalFromLog1p(torch.nn.Module):
    def __init__(self, sales_idx: list[int], mask_positive: bool = True):
        super().__init__()
        self.sales_idx = sales_idx
        self.mask_positive = mask_positive

    def forward(
        self,
        y_pred_log: torch.Tensor,
        y_true_log: torch.Tensor,
        w: torch.Tensor | None = None,
    ):
        y_pred = torch.expm1(y_pred_log[:, self.sales_idx])
        y_true = torch.expm1(y_true_log[:, self.sales_idx])

        valid = torch.isfinite(y_pred) & torch.isfinite(y_true)
        mask = valid & (
            (y_true > 0)
            if self.mask_positive
            else torch.ones_like(y_true, dtype=torch.bool)
        )
        if not mask.any():
            return torch.zeros((), device=y_pred.device)

        se = (y_pred - y_true).pow(2)

        # mean over masked elements (unweighted)
        if w is None:
            return se[mask].mean()

        # weighted mean: weight per row, then average
        per_row_se = torch.where(mask, se, 0.0).sum(dim=1)  # sum over targets
        per_row_cnt = mask.sum(dim=1).clamp_min(1)  # how many targets used
        per_row_mse = per_row_se / per_row_cnt

        w = w.reshape(-1).to(per_row_mse.device).float()
        return (w * per_row_mse).sum() / w.sum().clamp_min(1e-12)


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


# ─────────────────────────────────────────────────────────────────────
# LightningWrapper
# ─────────────────────────────────────────────────────────────────────


class LightningWrapper(pl.LightningModule):
    """Lightning module wrapping the forecasting model with full metric logging."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        store: int,
        item: int,
        sales_idx: List[int],
        # train_mav: float,
        # val_mav: float,
        *,
        lr: float = 3e-4,
        log_level: str = "INFO",
    ):
        super().__init__()
        self.logger_ = get_logger(f"{__name__}.{model_name}", log_level)

        # Ensure at least one handler exists
        if not self.logger_.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger_.addHandler(handler)

        self.model = model
        self.model_name = model_name
        self.store = store
        self.item = item
        self.lr = lr
        self.sales_idx = sales_idx
        # self.train_mav = train_mav
        # self.val_mav = val_mav

        # Loss and metrics
        # self.loss_fn = NWRMSLELoss()
        # self.loss_fn = PercentMAVLossLog1p(sales_idx)
        # self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.loss_fn = MSELossOriginalFromLog1p(sales_idx)
        self.train_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
        self.val_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
        self.train_mav_metric = MeanAbsTargetLog1p(sales_idx)
        self.val_mav_metric = MeanAbsTargetLog1p(sales_idx)
        self.train_rmse_metric = RootMeanSquaredErrorLog1p(sales_idx, log_level)
        self.val_rmse_metric = RootMeanSquaredErrorLog1p(sales_idx, log_level)

        # Save hparams (ignoring model object)
        self.save_hyperparameters(ignore=["model"])

    # ---------------------------
    # Core Methods
    # ---------------------------

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xb, yb, wb = batch
        preds = self.model(xb)

        # Compute loss
        # loss = self.loss_fn(preds, yb)
        loss = self.loss_fn(preds, yb, wb)

        # Log training loss

        # keep per-batch logging **if you want it**
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

        # add epoch-level logging for the LR scheduler
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=xb.size(0),
        )

        # Update & log MAE
        self.train_mae_metric.update(preds, yb)
        self.log(
            "train_mae",
            self.train_mae_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.train_mav_metric.update(yb)

        # Update & log RMSE
        self.train_rmse_metric.update(preds, yb)
        self.log(
            "train_rmse",
            self.train_rmse_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb, wb = batch
        preds = self.model(xb)

        # Compute loss
        loss = self.loss_fn(preds[:, self.sales_idx], yb[:, self.sales_idx])
        #        loss = self.loss_fn(preds, yb, wb)
        # keep per-batch logging **if you want it**
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)

        # add epoch-level logging for the LR scheduler
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=xb.size(0),
        )

        # Update & log MAE
        self.val_mae_metric.update(preds, yb)
        self.log(
            "val_mae",
            self.val_mae_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.val_mav_metric.update(yb)

        # Update & log RMSE
        self.val_rmse_metric.update(preds, yb)
        self.log(
            "val_rmse",
            self.val_rmse_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    # ---------------------------
    # Metric Reset Hooks
    # ---------------------------

    def on_train_epoch_start(self):
        self.train_mae_metric.reset()
        self.train_mav_metric.reset()
        self.train_rmse_metric.reset()

    def on_validation_epoch_start(self):
        self.val_mae_metric.reset()
        self.val_mav_metric.reset()
        self.val_rmse_metric.reset()

    # ---------------------------
    # Epoch-End Hooks for %MAV
    # ---------------------------

    def on_train_epoch_end(self):
        avg_train_mae = float(self.train_mae_metric.compute().item())
        avg_train_mav = float(self.train_mav_metric.compute().item())
        self.log(
            "train_percent_mav",
            100.0 * avg_train_mae / max(avg_train_mav, 1e-12),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # avg_train_mae = float(self.train_mae_metric.compute().item())
        # # avg_train_percent_mav = (
        # #     math.nan if self.train_mav == 0 else avg_train_mae / self.train_mav * 100
        # # )
        # avg_train_percent_mav = 100.0 * avg_train_mae / max(self.train_mav, 1e-12)

        # self.log(
        #     "train_percent_mav",
        #     avg_train_percent_mav,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

    def on_validation_epoch_end(self):
        avg_val_mae = float(self.val_mae_metric.compute().item())
        avg_val_mav = float(self.val_mav_metric.compute().item())
        self.logger_.info(
            f"[VAL] MAE={avg_val_mae:.6f}  MAV={avg_val_mav:.6f}  %MAV={100*avg_val_mae/max(avg_val_mav,1e-12):.2f}"
        )

        self.log(
            "val_percent_mav",
            100.0 * avg_val_mae / max(avg_val_mav, 1e-12),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    # ---------------------------
    # Optimizer
    # ---------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=1e-5, lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-5, threshold=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }


# ─────────────────────────────────────────────────────────────────────
# Helper metrics
# ─────────────────────────────────────────────────────────────────────
class MeanAbsoluteErrorLog1p(Metric):
    """
    MAE in original units for log1p-scaled predictions & targets.
    Computes over POSITIVE targets only (aligns with your %MAV).
    """

    full_state_update = False

    def __init__(self, sales_idx: list[int], log_level: str = "INFO"):
        super().__init__()
        self.logger_ = get_logger(f"{__name__}.MeanAbsoluteErrorLog1p", log_level)
        self.sales_idx = sales_idx

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Select relevant columns
        preds = preds[:, self.sales_idx]
        target = target[:, self.sales_idx]

        # Revert log1p
        preds = torch.expm1(preds)
        target = torch.expm1(target)

        # Valid & positive-target mask
        valid = torch.isfinite(preds) & torch.isfinite(target)
        pos = target > 0
        mask = valid & pos
        n = torch.count_nonzero(mask)
        if n == 0:
            return

        abs_error = torch.abs(preds[mask] - target[mask]).sum()
        self.sum_abs_error += abs_error
        self.count += n

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_abs_error.device)
        return self.sum_abs_error / self.count


class MeanAbsTargetLog1p(Metric):
    """
    Mean(|target|) in original units for log1p-scaled targets.
    Computes over POSITIVE targets only (aligns with MeanAbsoluteErrorLog1p).
    """

    full_state_update = False

    def __init__(self, sales_idx: list[int]):
        super().__init__()
        self.sales_idx = sales_idx
        self.add_state(
            "sum_abs_target", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, target: torch.Tensor):
        target = target[:, self.sales_idx]
        target = torch.expm1(target)  # back to original units
        mask = torch.isfinite(target) & (target > 0)
        n = torch.count_nonzero(mask)
        if n == 0:
            return
        self.sum_abs_target += torch.abs(target[mask]).sum()
        self.count += n

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_abs_target.device)
        return self.sum_abs_target / self.count


class RootMeanSquaredErrorLog1p(Metric):
    """
    RMSE in original units for log1p-scaled predictions & targets.
    Aggregates across steps automatically.
    """

    full_state_update = False  # avoids expensive DDP communication

    def __init__(self, sales_idx: list[int], log_level: str = "INFO"):
        super().__init__()
        self.logger_ = get_logger(f"{__name__}.RootMeanSquaredErrorLog1p", log_level)
        self.sales_idx = sales_idx

        # Buffers for sum of squared error and count
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Select relevant columns
        preds = preds[:, self.sales_idx]
        target = target[:, self.sales_idx]

        # Revert log1p
        preds = torch.expm1(preds)
        target = torch.expm1(target)

        # Mask zero targets
        mask = target > 0
        if mask.sum() == 0:
            return

        squared_error = torch.square(preds[mask] - target[mask]).sum()
        self.sum_squared_error += squared_error
        self.count += mask.sum()  # ✅ increment count before compute()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_squared_error.device)
        return torch.sqrt(self.sum_squared_error / self.count)


def compute_rmse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    logger: logging.Logger,
    *,
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
    logger.debug(f"RMSE: {rmse.item():.6f}")
    return rmse


def naive_percent_mav(
    df,
    y_col="y_sales_day_1",  # target in original units
    lag1_col="sales_day_1",  # yesterday's sales in original units
):
    y_true = df[y_col].to_numpy()
    y_pred = df[lag1_col].to_numpy()

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0)
    if not mask.any():
        return float("nan")

    mae = np.abs(y_pred[mask] - y_true[mask]).mean()
    mav = np.abs(y_true[mask]).mean()  # same as mean(y_true[mask]) since >0
    return 100.0 * mae / max(mav, 1e-12)
