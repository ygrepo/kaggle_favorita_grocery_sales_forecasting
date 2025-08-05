import torch
from torch import nn
import lightning.pytorch as pl
import logging
import numpy as np
from pathlib import Path
from enum import Enum
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import TensorDataset, Dataset, DataLoader

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


# class NWRMSLELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         logger.setLevel(logging.INFO)

#     def forward(self, y_pred, y_true, w):
#         # add 1 to avoid log(0), clamp to eps to keep preds positive
#         eps = 1e-6
#         y_pred = torch.clamp(y_pred, min=eps)
#         log_diff = torch.log(y_pred + 1.0) - torch.log(y_true + 1.0)
#         num = torch.sum(w * log_diff**2)
#         den = torch.sum(w)
#         return torch.sqrt(num / den)


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
        *,
        lr: float = 3e-4,
        log_level: str = "INFO",
    ):
        super().__init__()
        self.logger_ = logging.getLogger(f"{__name__}.{model_name}")
        self.logger_.setLevel(getattr(logging, log_level.upper(), logging.INFO))

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
        self.loss_fn = NWRMSLELoss()

        # Save all except model
        self.save_hyperparameters(ignore=["model"])

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

        # --- Check inputs ---
        if torch.any(torch.isnan(xb)):
            self.logger_.error(f"NaN detected in input features at batch {batch_idx}")
            raise ValueError("NaN in input features detected!")
        if torch.any(torch.isinf(xb)):
            self.logger_.error(f"Inf detected in input features at batch {batch_idx}")
            raise ValueError("Inf in input features detected!")

        # --- Forward pass ---
        preds = self.model(xb)

        # --- Detect NaNs in predictions ---
        if torch.any(torch.isnan(preds)):
            self.logger_.error(f"NaN detected in predictions at batch {batch_idx}")
            self.logger_.debug(
                "Sample predictions:\n%s", preds[:5].detach().cpu().numpy()
            )
            self.logger_.debug("Input sample:\n%s", xb[:5].detach().cpu().numpy())
            raise ValueError("NaN in predictions detected!")

        preds = torch.clamp(preds, min=1e-6)

        # --- Compute loss ---
        loss = self.loss_fn(preds, yb, wb)
        # self.logger_.debug("Training Batch %d loss: %.6f", batch_idx, loss.item())

        # --- Detect NaNs in loss ---
        if torch.isnan(loss):
            self.logger_.error(f"NaN detected in loss at batch {batch_idx}")
            self.logger_.debug("Preds sample:\n%s", preds[:5].detach().cpu().numpy())
            self.logger_.debug("Targets sample:\n%s", yb[:5].detach().cpu().numpy())
            self.logger_.debug("Weights sample:\n%s", wb[:5].detach().cpu().numpy())
            raise ValueError("NaN loss detected!")

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        # Compute training MAE for this batch
        train_mae = compute_mae_from_preds(preds, yb, self.sales_idx)
        train_rmse = compute_rmse(preds, yb, preds.device)
        # self.logger_.info("Training Batch %d RMSE: %.6f", batch_idx, train_rmse.item())
        self.train_error_history.append(loss)
        self.train_mae_history.append(train_mae)
        self.train_rmse_history.append(train_rmse)
        self.log("train_mae", train_mae, prog_bar=True, sync_dist=True)
        self.log("train_rmse", train_rmse, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb, wb = batch
        if (
            torch.any(torch.isnan(xb))
            or torch.any(torch.isnan(yb))
            or torch.any(torch.isnan(wb))
        ):
            self.logger_.info(f"NaN detected in data at batch {batch_idx}")
            raise ValueError("NaN detected in validation data!")
        if (
            torch.any(torch.isinf(xb))
            or torch.any(torch.isinf(yb))
            or torch.any(torch.isinf(wb))
        ):
            self.logger_.info(f"Inf detected in data at batch {batch_idx}")
            raise ValueError("Inf detected in validation data!")

        preds = self.model(xb)
        if torch.any(torch.isnan(preds)):
            self.logger_.warning(f"NaN detected in predictions at batch {batch_idx}")
        preds = torch.clamp(preds, min=1e-6)
        if torch.all(torch.eq(yb, 0)):
            self.logger_.warning(f"Zero detected in targets at batch {batch_idx}")
        yb = torch.clamp(yb, min=1e-6)
        loss = self.loss_fn(preds, yb, wb)
        # self.logger_.debug("Validation Batch %d loss: %.6f", batch_idx, loss.item())

        # Compute validation MAE for this batch
        val_mae = compute_mae_from_preds(xb, yb, self.sales_idx)
        val_rmse = compute_rmse(preds, yb, preds.device)

        self.val_error_history.append(loss)
        self.val_mae_history.append(val_mae)
        self.val_rmse_history.append(val_rmse)

        # Log the batch loss, MAE, and accuracy
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", val_mae, prog_bar=True, sync_dist=True)
        self.log("val_rmse", val_rmse, prog_bar=True, sync_dist=True)

        return loss

    def on_epoch_start(self) -> None:
        self.logger_.info(
            f"\nModel: {self.model_name}-Epoch {self.current_epoch} started!"
        )

    def on_train_epoch_end(self) -> None:
        self.logger_.info(
            f"\nModel: {self.model_name}-Epoch {self.current_epoch} ended!"
        )
        # Compute average metrics for the epoch
        avg_train_mae = np.mean([_to_float(t) for t in self.train_mae_history])
        avg_train_rmse = np.mean([_to_float(t) for t in self.train_rmse_history])
        if self.train_mav == 0:
            self.logger_.warning(
                f"Train mav is 0 for store {self.store} and item {self.item}"
            )
            avg_train_percent_mav = float("inf")
        else:
            avg_train_percent_mav = avg_train_mae / self.train_mav * 100

        avg_val_mae = np.mean([_to_float(t) for t in self.val_mae_history])
        avg_val_rmse = np.mean([_to_float(t) for t in self.val_rmse_history])
        if self.val_mav == 0:
            self.logger_.warning(
                f"Train mav is 0 for store {self.store} and item {self.item}"
            )
            avg_val_percent_mav = float("inf")
        else:
            avg_val_percent_mav = avg_val_mae / self.val_mav * 100

        # Log epoch-level metrics
        self.log("avg_train_mav", self.train_mav, prog_bar=False, sync_dist=True)
        self.log(
            "avg_train_mae",
            _to_float(avg_train_mae),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "avg_train_percent_mav",
            _to_float(avg_train_percent_mav),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "avg_train_rmse",
            _to_float(avg_train_rmse),
            prog_bar=False,
            sync_dist=True,
        )
        self.log("avg_val_mav", self.val_mav, prog_bar=False, sync_dist=True)
        self.log("avg_val_mae", _to_float(avg_val_mae), prog_bar=False, sync_dist=True)
        self.log(
            "avg_val_percent_mav",
            _to_float(avg_val_percent_mav),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "avg_val_rmse",
            _to_float(avg_val_rmse),
            prog_bar=False,
            sync_dist=True,
        )

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

        self.log(
            "best_train_avg_mae",
            _to_float(self.best_train_avg_mae),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "best_train_avg_mae_percent_mav",
            _to_float(self.best_train_avg_mae_percent_mav),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "best_train_avg_rmse",
            _to_float(self.best_train_avg_rmse),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "best_val_avg_mae",
            _to_float(self.best_val_avg_mae),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "best_val_avg_mae_percent_mav",
            _to_float(self.best_val_avg_mae_percent_mav),
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "best_val_avg_rmse",
            _to_float(self.best_val_avg_rmse),
            prog_bar=False,
            sync_dist=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
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


def compute_mae(
    xb: torch.Tensor,
    yb: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    sales_idx: list[int],
) -> float:
    """
    Mean Absolute Error on SALES ONLY, expressed in original units.
    Cyclical targets are excluded from the calculation.
    Returns 0.0 if batch is empty or only zeros.
    """
    if xb.numel() == 0 or yb.numel() == 0:
        logger.warning("Empty batch encountered in compute_mae. Returning 0.0.")
        return 0.0

    with torch.no_grad():
        # ----- predictions -----
        preds_np = model(xb.to(device)).cpu().numpy()
        p_sales_units = preds_np[:, sales_idx]
        p_sales_units = np.expm1(p_sales_units)

        # ----- ground truth ----
        yb_np = yb.cpu().numpy()
        y_sales_units = yb_np[:, sales_idx]
        y_sales_units = np.expm1(y_sales_units)

        # ----- calculate batch-wise MAE -----
        if y_sales_units.size == 0:
            logger.warning("No sales values found in compute_mae. Returning 0.0.")
            return 0.0
        mask = y_sales_units > 0
        logger.debug(f"Non-zero targets in batch: {mask.sum()} / {y_sales_units.size}")

        if mask.sum() == 0:
            logger.warning("All targets are zero in this batch. Skipping MAE.")
            return 0.0

    batch_mae = np.abs(p_sales_units[mask] - y_sales_units[mask]).mean()

    return batch_mae


def compute_mae_from_preds(
    preds: torch.Tensor, yb: torch.Tensor, sales_idx: list[int]
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
        return 0.0

    batch_mae = np.abs(preds_np[mask] - yb_np[mask]).mean()
    return float(batch_mae)


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


def compute_mav(
    loader: DataLoader,
    sales_idx: list[int],
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

            # nonzero_count = np.count_nonzero(sales)
            # logger.debug(
            #     f"Sales: shape={sales.shape}, non-zero={nonzero_count}, "
            #     f"min={np.min(sales):.2f}, max={np.max(sales):.2f}, mean={np.mean(sales):.2f}"
            # )

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
