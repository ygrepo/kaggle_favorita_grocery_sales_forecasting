import torch
from torch import nn
import lightning.pytorch as pl
import logging
import numpy as np
from pathlib import Path
from enum import Enum
from torch.utils.data import DataLoader
from torchmetrics import Metric
import math
from typing import List

from src.utils import get_logger


logger = logging.getLogger(__name__)

# Set up logger
logger = logging.getLogger(__name__)


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)


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
    def __init__(self, input_dim, output_dim=3, h1=128, h2=64, dropout=0.2):
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
                    nn.LayerNorm(hidden),  # changed from BatchNorm1d
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, input_dim),  # residual connection
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
def model_factory(model_type: ModelType, input_dim: int, output_dim: int) -> nn.Module:
    """Factory function to return the correct model based on the model_type."""
    if model_type == ModelType.SHALLOW_NN:
        return ShallowNN(input_dim, output_dim)
    elif model_type == ModelType.TWO_LAYER_NN:
        return TwoLayerNN(input_dim, output_dim)
    elif model_type == ModelType.RESIDUAL_MLP:
        return ResidualMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_factory_from_str(
    model_type: str, input_dim: int, output_dim: int
) -> nn.Module:
    """Factory function to return the correct model based on the model_type."""
    if model_type == "ShallowNN":
        return ShallowNN(input_dim, output_dim)
    elif model_type == "TwoLayerNN":
        return TwoLayerNN(input_dim, output_dim)
    elif model_type == "ResidualMLP":
        return ResidualMLP(input_dim, output_dim)
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
        eps = 1e-6
        y_pred = torch.clamp(y_pred, min=eps)
        log_diff = y_pred - y_true  # Already log-scaled
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
        train_mav: float,
        val_mav: float,
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
        self.train_mav = train_mav
        self.val_mav = val_mav

        # Loss and metrics
        self.loss_fn = NWRMSLELoss()
        self.train_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
        self.val_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
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
        preds = torch.clamp(preds, min=1e-6)

        # Compute loss
        loss = self.loss_fn(preds, yb, wb)

        # Log training loss
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
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
        preds = torch.clamp(preds, min=1e-6)
        yb = torch.clamp(yb, min=1e-6)

        # Compute loss
        loss = self.loss_fn(preds, yb, wb)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
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

    def on_epoch_start(self):
        self.logger_.info(
            f"\nModel: {self.model_name}-Epoch {self.current_epoch} started!"
        )
        self.train_mae_metric.reset()
        self.train_rmse_metric.reset()

    def on_validation_epoch_start(self):
        self.val_mae_metric.reset()
        self.val_rmse_metric.reset()

    # ---------------------------
    # Epoch-End Hooks for %MAV
    # ---------------------------

    def on_train_epoch_end(self):
        avg_train_mae = self.train_mae_metric.compute().item()
        avg_train_percent_mav = (
            math.nan if self.train_mav == 0 else avg_train_mae / self.train_mav * 100
        )

        self.log(
            "train_percent_mav",
            avg_train_percent_mav,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        avg_val_mae = self.val_mae_metric.compute().item()
        avg_val_percent_mav = (
            math.nan if self.val_mav == 0 else avg_val_mae / self.val_mav * 100
        )

        self.log(
            "val_percent_mav",
            avg_val_percent_mav,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    # ---------------------------
    # Optimizer
    # ---------------------------

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


class MeanAbsoluteErrorLog1p(Metric):
    """
    MAE in original units for log1p-scaled predictions & targets.
    Aggregates across steps automatically.
    """

    full_state_update = False  # avoids expensive DDP communication

    def __init__(self, sales_idx: list[int], log_level: str = "INFO"):
        super().__init__()
        self.logger_ = get_logger(f"{__name__}.MeanAbsoluteErrorLog1p", log_level)
        self.sales_idx = sales_idx

        # Buffers for sum of abs error and count
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
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

        abs_error = torch.abs(preds[mask] - target[mask]).sum()
        self.sum_abs_error += abs_error
        self.count += mask.sum()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum_abs_error.device)
        return self.sum_abs_error / self.count


def compute_mae(
    preds: torch.Tensor,
    yb: torch.Tensor,
    sales_idx: list[int],
    logger: logging.Logger,
) -> float:
    """
    Compute Mean Absolute Error (MAE) in original units from predictions and targets.
    Both preds and yb are assumed to be log1p-scaled.
    """
    preds_np = preds[:, sales_idx].detach().cpu().numpy()
    preds_np = np.expm1(preds_np)  # revert log1p

    yb_np = yb[:, sales_idx].detach().cpu().numpy()
    yb_np = np.expm1(yb_np)  # revert log1p

    mask = yb_np > 0
    if mask.sum() == 0:
        logger.warning("All targets are zero in this batch. Skipping MAE.")
        return 0.0

    batch_mae = np.abs(preds_np[mask] - yb_np[mask]).mean()
    return float(batch_mae)


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


def compute_mav(
    loader: DataLoader,
    sales_idx: list[int],
    logger: logging.Logger,
) -> float:
    """
    Mean absolute value of the sales targets in original units.
    Assumes log1p scaling. Cyclical sin/cos columns are ignored.
    Returns 0.0 if all sales are zero or count is zero.
    """
    logger.setLevel(getattr(logging, "DEBUG", logging.INFO))

    abs_sum, count = 0.0, 0
    with torch.no_grad():
        for _, yb, _ in loader:
            y_np = yb.cpu().numpy()
            sales = np.expm1(y_np[:, sales_idx])  # Undo log1p
            abs_sum += np.abs(sales).sum()
            count += sales.size

    if count == 0:
        logger.warning(
            "MAV computation: No sales entries found (count=0). Returning 0.0."
        )
        return 0.0

    mav = abs_sum / count

    if mav == 0.0:
        logger.warning("MAV computation: All sales are zero. Returning 0.0.")

    return mav
