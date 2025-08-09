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
from typing import List, Optional, Dict, Any, Union

from src.utils import get_logger

# PyTorch Forecasting imports
try:
    from pytorch_forecasting import TemporalFusionTransformer, NBeats, DeepAR
    from pytorch_forecasting.models import LSTM as PytorchForecastingLSTM
    from pytorch_forecasting.metrics import RMSE, MAE, MAPE

    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False
    print("Warning: pytorch_forecasting not available. Sequence models will not work.")


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


# Unified model types enum
class MODEL_TYPE(str, Enum):
    # Feedforward models
    SHALLOW_NN = "ShallowNN"
    TWO_LAYER_NN = "TwoLayerNN"
    RESIDUAL_MLP = "ResidualMLP"

    # Sequence models
    TFT = "TFT"
    NBEATS = "NBEATS"
    DEEPAR = "DEEPAR"
    LSTM = "LSTM"


# Model type lists for convenience
FF_MODEL_TYPES = [
    MODEL_TYPE.SHALLOW_NN,
    MODEL_TYPE.TWO_LAYER_NN,
    MODEL_TYPE.RESIDUAL_MLP,
]
SEQ_MODEL_TYPES = [
    MODEL_TYPE.TFT,
    MODEL_TYPE.NBEATS,
    MODEL_TYPE.DEEPAR,
    MODEL_TYPE.LSTM,
]


# Unified Model Factory Function
def model_factory(
    model_type: MODEL_TYPE,
    input_dim: int = None,
    hidden_dim: int = 128,
    h1: int = 64,
    h2: int = 32,
    depth: int = 3,
    output_dim: int = 1,
    dropout: float = 0.0,
    # Sequence model specific parameters
    training_dataset=None,
    learning_rate: float = 1e-3,
    attention_head_size: int = 4,
    hidden_continuous_size: int = 16,
    **kwargs,
) -> Union[nn.Module, pl.LightningModule]:
    """Unified factory function for both feedforward and sequence models."""

    # Feedforward models
    if model_type == MODEL_TYPE.SHALLOW_NN:
        return ShallowNN(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == MODEL_TYPE.TWO_LAYER_NN:
        return TwoLayerNN(input_dim, output_dim, h1, h2, dropout)
    elif model_type == MODEL_TYPE.RESIDUAL_MLP:
        return ResidualMLP(input_dim, output_dim, hidden_dim, depth, dropout)

    # Sequence models
    elif model_type == MODEL_TYPE.TFT:
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("pytorch_forecasting is required for sequence models")
        return TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_dim,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=3,
            **kwargs,
        )
    elif model_type == MODEL_TYPE.NBEATS:
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("pytorch_forecasting is required for sequence models")
        return NBeats.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_dim,
            loss=RMSE(),
            log_interval=10,
            **kwargs,
        )
    elif model_type == MODEL_TYPE.DEEPAR:
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("pytorch_forecasting is required for sequence models")
        return DeepAR.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_dim,
            dropout=dropout,
            loss=RMSE(),
            log_interval=10,
            **kwargs,
        )
    elif model_type == MODEL_TYPE.LSTM:
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("pytorch_forecasting is required for sequence models")
        return PytorchForecastingLSTM.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_dim,
            dropout=dropout,
            loss=RMSE(),
            log_interval=10,
            **kwargs,
        )
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
    """Lightning module wrapping both feedforward and sequence models with full metric logging."""

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
        is_sequence_model: bool = False,
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
        self.is_sequence_model = is_sequence_model

        # Loss and metrics (only for feedforward models)
        if not is_sequence_model:
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
        if self.is_sequence_model:
            # Let the sequence model handle its native training and logging
            result = self.model.training_step(batch, batch_idx)

            # Extract predictions and targets for our custom metrics
            try:
                x, y = batch
                with torch.no_grad():
                    # Get predictions from the model
                    output = self.model(x)
                    if isinstance(output, dict):
                        preds = output.get("prediction", output.get("output", None))
                    else:
                        preds = output

                    # Extract targets - sequence models use different format
                    if isinstance(y, dict):
                        yb = y.get("target", y.get("decoder_target", None))
                    else:
                        yb = y

                    # Compute our custom metrics if we have valid predictions and targets
                    if preds is not None and yb is not None:
                        # Ensure shapes are compatible
                        if preds.dim() > 2:
                            preds = preds.view(-1, preds.size(-1))
                        if yb.dim() > 2:
                            yb = yb.view(-1, yb.size(-1))

                        # Update our custom metrics
                        self.train_mae_metric.update(preds, yb)
                        self.train_rmse_metric.update(preds, yb)
            except Exception as e:
                # If metric extraction fails, just continue with native metrics
                self.logger_.debug(f"Could not extract metrics for sequence model: {e}")

            return result
        else:
            # For feedforward models, use the original logic
            xb, yb, wb = batch
            preds = self.model(xb)
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

        # Update MAE and RMSE (only for feedforward models - sequence models handle this in their step methods)
        if not self.is_sequence_model:
            self.train_mae_metric.update(preds, yb)
            self.train_rmse_metric.update(preds, yb)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_sequence_model:
            # Let the sequence model handle its native validation and logging
            result = self.model.validation_step(batch, batch_idx)

            # Extract predictions and targets for our custom metrics
            try:
                x, y = batch
                with torch.no_grad():
                    # Get predictions from the model
                    output = self.model(x)
                    if isinstance(output, dict):
                        preds = output.get("prediction", output.get("output", None))
                    else:
                        preds = output

                    # Extract targets - sequence models use different format
                    if isinstance(y, dict):
                        yb = y.get("target", y.get("decoder_target", None))
                    else:
                        yb = y

                    # Compute our custom metrics if we have valid predictions and targets
                    if preds is not None and yb is not None:
                        # Ensure shapes are compatible
                        if preds.dim() > 2:
                            preds = preds.view(-1, preds.size(-1))
                        if yb.dim() > 2:
                            yb = yb.view(-1, yb.size(-1))

                        # Update our custom metrics
                        self.val_mae_metric.update(preds, yb)
                        self.val_rmse_metric.update(preds, yb)
            except Exception as e:
                # If metric extraction fails, just continue with native metrics
                self.logger_.debug(f"Could not extract metrics for sequence model: {e}")

            return result
        else:
            # For feedforward models, use the original logic
            xb, yb, wb = batch
            preds = self.model(xb)
            loss = self.loss_fn(preds, yb, wb)

            # keep per-batch logging **if you want it**
            self.log(
                "val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False
            )

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

        # Update MAE and RMSE (only for feedforward models - sequence models handle this in their step methods)
        if not self.is_sequence_model:
            self.val_mae_metric.update(preds, yb)
            self.val_rmse_metric.update(preds, yb)

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
        # Compute custom metrics for both feedforward and sequence models
        try:
            avg_train_mae = self.train_mae_metric.compute().item()
            avg_train_percent_mav = (
                math.nan
                if self.train_mav == 0
                else avg_train_mae / self.train_mav * 100
            )

            self.log(
                "train_percent_mav",
                avg_train_percent_mav,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            # Also log the custom MAE and RMSE for consistency
            self.log(
                "train_mae",
                self.train_mae_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                "train_rmse",
                self.train_rmse_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as e:
            # If metric computation fails (e.g., no data), skip
            self.logger_.debug(f"Could not compute train metrics: {e}")

    def on_validation_epoch_end(self):
        # Compute custom metrics for both feedforward and sequence models
        try:
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

            # Also log the custom MAE and RMSE for consistency
            self.log(
                "val_mae",
                self.val_mae_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                "val_rmse",
                self.val_rmse_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as e:
            # If metric computation fails (e.g., no data), skip
            self.logger_.debug(f"Could not compute val metrics: {e}")

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


# Custom callback for sequence models to compute feedforward-style metrics
class SequenceModelMetricsCallback(pl.Callback):
    """Callback to compute custom MAE, RMSE, and %MAV metrics for sequence models."""

    def __init__(
        self,
        model_name: str,
        store: int,
        item: int,
        sales_idx: List[int],
        train_mav: float = 1.0,
        val_mav: float = 1.0,
        log_level: str = "INFO",
    ):
        super().__init__()
        self.model_name = model_name
        self.store = store
        self.item = item
        self.sales_idx = sales_idx
        self.train_mav = train_mav
        self.val_mav = val_mav
        self.logger_ = get_logger(f"{__name__}.{model_name}", log_level)

        # Initialize metrics
        self.train_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
        self.val_mae_metric = MeanAbsoluteErrorLog1p(sales_idx, log_level)
        self.train_rmse_metric = RootMeanSquaredErrorLog1p(sales_idx, log_level)
        self.val_rmse_metric = RootMeanSquaredErrorLog1p(sales_idx, log_level)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Extract predictions and targets from sequence model and update metrics."""
        try:
            x, y = batch
            with torch.no_grad():
                # Get predictions from the model
                output = pl_module(x)
                if isinstance(output, dict):
                    preds = output.get("prediction", output.get("output", None))
                else:
                    preds = output

                # Extract targets
                if isinstance(y, dict):
                    yb = y.get("target", y.get("decoder_target", None))
                else:
                    yb = y

                # Update metrics if we have valid data
                if preds is not None and yb is not None:
                    # Ensure shapes are compatible
                    if preds.dim() > 2:
                        preds = preds.view(-1, preds.size(-1))
                    if yb.dim() > 2:
                        yb = yb.view(-1, yb.size(-1))

                    self.train_mae_metric.update(preds, yb)
                    self.train_rmse_metric.update(preds, yb)
        except Exception as e:
            self.logger_.debug(f"Could not extract train metrics: {e}")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Extract predictions and targets from sequence model and update metrics."""
        try:
            x, y = batch
            with torch.no_grad():
                # Get predictions from the model
                output = pl_module(x)
                if isinstance(output, dict):
                    preds = output.get("prediction", output.get("output", None))
                else:
                    preds = output

                # Extract targets
                if isinstance(y, dict):
                    yb = y.get("target", y.get("decoder_target", None))
                else:
                    yb = y

                # Update metrics if we have valid data
                if preds is not None and yb is not None:
                    # Ensure shapes are compatible
                    if preds.dim() > 2:
                        preds = preds.view(-1, preds.size(-1))
                    if yb.dim() > 2:
                        yb = yb.view(-1, yb.size(-1))

                    self.val_mae_metric.update(preds, yb)
                    self.val_rmse_metric.update(preds, yb)
        except Exception as e:
            self.logger_.debug(f"Could not extract val metrics: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Log custom training metrics."""
        try:
            avg_train_mae = self.train_mae_metric.compute().item()
            avg_train_percent_mav = (
                math.nan
                if self.train_mav == 0
                else avg_train_mae / self.train_mav * 100
            )

            # Log custom metrics
            pl_module.log(
                "train_percent_mav",
                avg_train_percent_mav,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            pl_module.log(
                "train_mae_custom",
                self.train_mae_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            pl_module.log(
                "train_rmse_custom",
                self.train_rmse_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as e:
            self.logger_.debug(f"Could not compute train metrics: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log custom validation metrics."""
        try:
            avg_val_mae = self.val_mae_metric.compute().item()
            avg_val_percent_mav = (
                math.nan if self.val_mav == 0 else avg_val_mae / self.val_mav * 100
            )

            # Log custom metrics
            pl_module.log(
                "val_percent_mav",
                avg_val_percent_mav,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            pl_module.log(
                "val_mae_custom",
                self.val_mae_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            pl_module.log(
                "val_rmse_custom",
                self.val_rmse_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        except Exception as e:
            self.logger_.debug(f"Could not compute val metrics: {e}")

    def on_epoch_start(self, trainer, pl_module):
        """Reset metrics at the start of each epoch."""
        self.train_mae_metric.reset()
        self.val_mae_metric.reset()
        self.train_rmse_metric.reset()
        self.val_rmse_metric.reset()
