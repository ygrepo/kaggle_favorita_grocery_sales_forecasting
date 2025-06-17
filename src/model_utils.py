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
from typing import List, Tuple, Dict, override
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
        model_name: str,
        y_sales_scaler: MinMaxScaler,
        sales_idx: list[int],
        train_mav: float,
        val_mav: float,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.y_sales_scaler = y_sales_scaler
        self.sales_idx = sales_idx
        self.train_mav = train_mav
        self.val_mav = val_mav
        self.loss_fn = NWRMSLELoss()

        # Initialize metrics accumulators for each epoch
        self.train_error_history = []
        self.val_error_history = []
        self.train_mae_history = []
        self.val_mae_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.best_train_avg_mae = float("inf")
        self.best_val_avg_mae = float("inf")
        self.best_train_avg_accuracy = -1
        self.best_val_avg_accuracy = -1
        self.best_train_avg_percent_mav = -1
        self.best_val_avg_percent_mav = -1

    def forward(self, x):
        return self.model(x)

    def calculate_accuracy(self, preds, targets, threshold=0.1):
        """Calculate accuracy as the percentage of exact matches (with a tolerance)."""
        correct = (torch.abs(preds - targets) < threshold).float()
        accuracy = correct.sum() / correct.size(0)
        return accuracy

    def training_step(self, batch, batch_idx):
        xb, yb, wb = batch
        preds = torch.clamp(self.model(xb), min=1e-6)
        loss = self.loss_fn(preds, yb, wb)

        # Compute training MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_mae = compute_mae(
            xb, yb, self.model, device, self.y_sales_scaler, self.sales_idx
        )
        train_accuracy = self.calculate_accuracy(preds, yb)  # Calculate accuracy

        self.train_error_history.append(loss)
        self.train_mae_history.append(train_mae)
        self.train_accuracy_history.append(train_accuracy)

        # Log the batch loss, MAE, and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", train_mae, prog_bar=True)
        self.log("train_accuracy", train_accuracy, prog_bar=True)

        return loss

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
        if torch.all(torch.eq(yb, 0)):
            print(f"Zero detected in targets at batch {batch_idx}")
        yb = torch.clamp(yb, min=1e-6)
        loss = self.loss_fn(preds, yb, wb)

        # Compute validation MAE for this batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_mae = compute_mae(
            xb, yb, self.model, device, self.y_sales_scaler, self.sales_idx
        )
        val_accuracy = self.calculate_accuracy(preds, yb)  # Calculate accuracy

        self.val_error_history.append(loss)
        self.val_mae_history.append(val_mae)
        self.val_accuracy_history.append(val_accuracy)

        # Log the batch loss, MAE, and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)

        return loss

    def on_epoch_start(self) -> None:
        print(f"\nModel: {self.model_name}-Epoch {self.current_epoch} started!")

    def on_train_epoch_end(self) -> None:
        print(f"\nModel: {self.model_name}-Epoch {self.current_epoch} ended!")
        # Compute average metrics for the epoch
        avg_train_mae = np.mean(self.train_mae_history)
        avg_train_accuracy = np.mean(self.train_accuracy_history)  # Average accuracy
        avg_train_percent_mav = avg_train_mae / self.train_mav * 100
        avg_val_mae = np.mean(self.val_mae_history)
        avg_val_percent_mav = avg_val_mae / self.val_mav * 100
        avg_val_accuracy = np.mean(
            self.val_accuracy_history
        )  # Average validation accuracy

        # Log epoch-level metrics
        self.log("avg_train_mae", avg_train_mae, prog_bar=False)
        self.log("avg_train_percent_mav", avg_train_percent_mav, prog_bar=False)
        self.log(
            "avg_train_accuracy", avg_train_accuracy, prog_bar=False
        )  # Log accuracy
        self.log("avg_val_mae", avg_val_mae, prog_bar=False)
        self.log("avg_val_percent_mav", avg_val_percent_mav, prog_bar=False)
        self.log("avg_val_accuracy", avg_val_accuracy, prog_bar=False)  # Log accuracy

        # Reset accumulators for the next epoch
        self.train_mae_history = []
        self.val_mae_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []

        if avg_train_mae < self.best_train_avg_mae:
            self.best_train_avg_mae = avg_train_mae
            self.best_train_avg_mae_percent_mav = avg_train_percent_mav
        if avg_train_accuracy > self.best_train_avg_accuracy:
            self.best_train_avg_accuracy = avg_train_accuracy
        if avg_val_mae < self.best_val_avg_mae:
            self.best_val_avg_mae = avg_val_mae
            self.best_val_avg_mae_percent_mav = avg_val_percent_mav
        if avg_val_accuracy > self.best_val_avg_accuracy:
            self.best_val_avg_accuracy = avg_val_accuracy
        if avg_val_accuracy > self.best_val_avg_accuracy:
            self.best_val_avg_accuracy = avg_val_accuracy

        self.log("best_train_avg_mae", self.best_train_avg_mae, prog_bar=False)
        self.log(
            "best_train_avg_mae_percent_mav",
            self.best_train_avg_mae_percent_mav,
            prog_bar=False,
        )
        self.log(
            "best_train_avg_accuracy",
            self.best_train_avg_accuracy,
            prog_bar=False,
        )  # Log best accuracy
        self.log("best_val_avg_mae", self.best_val_avg_mae, prog_bar=False)
        self.log(
            "best_val_avg_mae_percent_mav",
            self.best_val_avg_mae_percent_mav,
            prog_bar=False,
        )
        self.log(
            "best_val_avg_accuracy", self.best_val_avg_accuracy, prog_bar=False
        )  # Log best accuracy

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
    model_dir: str = "../output/models/",
) -> pd.DataFrame:

    pl.seed_everything(seed)

    # Ensure directories exist
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    scalers_dir = model_dir_path / "scalers"
    scalers_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = model_dir_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_dir = model_dir_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    history = []

    today_str = datetime.today().strftime("%Y-%m-%d")
    for sid in df["store_item"].unique():
        sub = (
            df[df["store_item"] == sid].sort_values("start_date").reset_index(drop=True)
        )
        sub = sub.merge(weights_df, on=item_col, how="left")

        train_size = int(len(sub) * train_frac)

        # SPLIT
        X_train_df = sub.loc[: train_size - 1, x_feature_cols]
        X_test_df = sub.loc[train_size:, x_feature_cols]

        # Prepare input features (cyclical, sales)
        x_sales_idx = [x_feature_cols.index(c) for c in x_sales_features]
        x_cyc_idx = [x_feature_cols.index(c) for c in x_cyclical_features]

        # Prepare sales data
        x_sales_train = np.clip(X_train_df.iloc[:, x_sales_idx].to_numpy(), 0, None)
        x_sales_test = np.clip(X_test_df.iloc[:, x_sales_idx].to_numpy(), 0, None)
        x_sales_train_log = np.log1p(x_sales_train)
        x_sales_test_log = np.log1p(x_sales_test)

        x_sales_scaler = MinMaxScaler().fit(x_sales_train_log)
        x_cyc_train = X_train_df.iloc[:, x_cyc_idx].to_numpy()  # cyclical features
        x_cyc_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_cyc_train)

        # Stack sales and cyclical features for training
        X_train_full = np.hstack(
            [
                x_sales_scaler.transform(x_sales_train_log),
                x_cyc_scaler.transform(x_cyc_train),
            ]
        )
        X_test_full = np.hstack(
            [
                x_sales_scaler.transform(x_sales_test_log),
                x_cyc_scaler.transform(X_test_df.iloc[:, x_cyc_idx].to_numpy()),
            ]
        )

        # Save scalers
        pickle.dump(
            x_sales_scaler,
            open(scalers_dir / f"{today_str}_x_sales_scaler_{sid}.pkl", "wb"),
        )
        pickle.dump(
            x_cyc_scaler,
            open(scalers_dir / f"{today_str}_x_cyc_scaler_{sid}.pkl", "wb"),
        )

        # Prepare output features
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

        # Stack output features for training
        y_train_full = np.hstack(
            [
                y_sales_scaler.transform(y_sales_train_log),
                y_cyc_scaler.transform(y_cyc_train),
            ]
        )
        y_test_full = np.hstack(
            [
                y_sales_scaler.transform(y_sales_test_log),
                y_cyc_scaler.transform(
                    sub.loc[train_size:, y_cyclical_features].to_numpy()
                ),
            ]
        )

        # Save scalers for output
        pickle.dump(
            y_sales_scaler,
            open(scalers_dir / f"{today_str}_y_sales_scaler_{sid}.pkl", "wb"),
        )
        pickle.dump(
            y_cyc_scaler,
            open(scalers_dir / f"{today_str}_y_cyc_scaler_{sid}.pkl", "wb"),
        )

        # Prepare dataset and dataloaders
        w = sub["weight"].to_numpy(float).reshape(-1, 1)
        w_train, w_test = w[:train_size], w[train_size:]

        ds_train = TensorDataset(
            torch.from_numpy(X_train_full).float(),
            torch.from_numpy(y_train_full).float(),
            torch.from_numpy(w_train).float(),
        )
        ld_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
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

        # Initialize model
        base_model = model_cls(input_dim=X_train_full.shape[1])
        base_model.apply(init_weights)

        # Compute MAV (Mean Absolute Value)
        sales_idx = [label_cols.index(c) for c in y_sales_features]
        train_mav = compute_mav(ld_train, y_sales_scaler, sales_idx)
        val_mav = compute_mav(ld_test, y_sales_scaler, sales_idx)

        # Initialize Lightning model
        model_name = f"{today_str}_model_{sid}"
        lightning_model = LightningWrapper(
            base_model,
            model_name=model_name,
            lr=lr,
            y_sales_scaler=y_sales_scaler,
            sales_idx=sales_idx,
            train_mav=train_mav,
            val_mav=val_mav,
        )

        # ModelCheckpoint to monitor best_train_avg_mae
        checkpoint_callback = ModelCheckpoint(
            monitor="best_train_avg_mae",  # Monitor the best training MAE
            mode="min",  # Save the model with the lowest train MAE
            save_top_k=1,
            dirpath=checkpoints_dir,
            filename=f"{today_str}_model_{model_name}",  # Model file naming pattern
        )

        trainer = pl.Trainer(
            deterministic=True,
            max_epochs=epochs,
            logger=train_logger,
            enable_progress_bar=enable_progress_bar,
            callbacks=[checkpoint_callback],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        # Train the model
        trainer.fit(lightning_model, ld_train, ld_test)

        # Save training history for the model
        history.append(
            {
                "model_name": model_name,
                "best_train_avg_mae": lightning_model.best_train_avg_mae,
                "best_val_avg_mae": lightning_model.best_val_avg_mae,
                "best_train_avg_accuracy": lightning_model.best_train_avg_accuracy,
                "best_val_avg_accuracy": lightning_model.best_val_avg_accuracy,
                "best_train_avg_mae_percent_mav": lightning_model.best_train_avg_mae_percent_mav,
                "best_val_avg_mae_percent_mav": lightning_model.best_val_avg_mae_percent_mav,
            }
        )

    # Save history to file
    history_save_path = history_dir / f"{today_str}_history.xlsx"
    history = pd.DataFrame(history)
    history.to_excel(history_save_path, index=False)
    return history


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
