import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit, KFold
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime


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
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Hardtanh(min_val=0.0, max_val=1.0),  # clips to [0,1]
            nn.Sigmoid(),  # outputs in (0,1)
        )

    def forward(self, x):
        return self.net(x)


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


def train(
    df: pd.DataFrame,
    y_scaler: object,
    weights_df: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str],
    y_sales_features: list[str],
    item_col: str,
    train_frac: float = 0.8,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 5,
    seed: int = 2025,
    model_dir: str = "../output/models/",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, torch.nn.Module]]:
    """
    Trains a separate ShallowNN on each store_item's time series (no CV),
    does an 80/20 split by time, saves each model, and returns:
      - hist_df: per-epoch train/test losses
      - summary_df: final losses by store_item
      - models: dict mapping store_item -> trained model (on CPU)
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)

    history = []
    models: Dict[str, torch.nn.Module] = {}

    for sid in df["store_item"].unique():
        # --- Prepare sid-specific data
        sub = (
            df[df["store_item"] == sid]
            .sort_values("start_date")
            .reset_index(drop=True)
            .merge(weights_df, on=item_col, how="left")
        )

        X = sub[feature_cols].to_numpy(float)
        y = sub[label_cols].to_numpy(float)
        w = sub["weight"].to_numpy(float).reshape(-1, 1)

        # train/test 80/20 by time
        train_size = int(len(sub) * train_frac)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        w_train, w_test = w[:train_size], w[train_size:]

        # DataLoaders
        ds_train = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
            torch.from_numpy(w_train).float(),
        )
        ds_test = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).float(),
            torch.from_numpy(w_test).float(),
        )
        ld_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        ld_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        # init model, loss, optimizer
        model = ShallowNN(input_dim=len(feature_cols)).to(device)
        loss_fn = NWRMSLELoss()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # train epochs
        for epoch in range(1, epochs + 1):
            # train
            model.train()
            tr_loss_acc = 0.0
            for xb, yb, wb in ld_train:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb, wb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                tr_loss_acc += loss.item() * xb.size(0)
            tr_loss = tr_loss_acc / len(ds_train)

            # test
            model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for xb, yb, wb in ld_test:
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                    if torch.any(torch.isnan(yb)) or torch.any(yb < 0):
                        print(
                            f"[{sid}] Warning: yb has NaN or negative values in test set at epoch {epoch}"
                        )
                        print(yb)
                    p = model(xb).clamp(min=1e-6)
                    if torch.any(torch.isnan(p)) or torch.any(p < 0):
                        print(
                            f"[{sid}] Warning: p has NaN or negative values at epoch {epoch}"
                        )
                        print(p)
                    ld = torch.log(p + 1) - torch.log(yb + 1)
                    num += (wb * ld**2).sum().item()
                    den += wb.sum().item()
            te_loss = (num / den) ** 0.5

            # — TRUE TRAIN MAE —
            model.eval()
            abs_tr_sum = 0.0
            count_tr = 0
            with torch.no_grad():
                for xb, yb, _ in ld_train:
                    xb, yb = xb.to(device), yb.to(device)
                    p = model(xb)

                    # Inverse MinMax scaling
                    p_inv = y_scaler.inverse_transform(p.cpu().numpy())
                    yb_inv = y_scaler.inverse_transform(yb.cpu().numpy())

                    # Apply expm1 ONLY to the sales columns
                    sales_idx = [label_cols.index(col) for col in y_sales_features]

                    p_actual = p_inv.copy()
                    yb_actual = yb_inv.copy()
                    p_actual[:, sales_idx] = np.expm1(p_actual[:, sales_idx])
                    yb_actual[:, sales_idx] = np.expm1(yb_actual[:, sales_idx])

                    # MAE in original scale
                    abs_tr_sum += np.sum(np.abs(p_actual - yb_actual))
                    count_tr += yb_actual.size

            true_train_mae = abs_tr_sum / count_tr

            # TRUE TEST MAE
            abs_te_sum = 0.0
            count_te = 0
            with torch.no_grad():
                for xb, yb, _ in ld_test:
                    xb, yb = xb.to(device), yb.to(device)
                    p = model(xb)

                    # Inverse MinMax scaling
                    p_inv = y_scaler.inverse_transform(p.cpu().numpy())
                    yb_inv = y_scaler.inverse_transform(yb.cpu().numpy())

                    # Apply expm1 ONLY to the sales columns
                    p_actual = p_inv.copy()
                    yb_actual = yb_inv.copy()
                    p_actual[:, sales_idx] = np.expm1(p_actual[:, sales_idx])
                    yb_actual[:, sales_idx] = np.expm1(yb_actual[:, sales_idx])

                    # MAE in original scale
                    abs_te_sum += np.sum(np.abs(p_actual - yb_actual))
                    count_te += yb_actual.size

            true_test_mae = abs_te_sum / count_te

            # record everything
            history.append(
                {
                    "store_item": sid,
                    "epoch": epoch,
                    "train_loss": tr_loss,  # your original
                    "train_mae": true_train_mae,  # NEW MAE on train
                    "test_loss": te_loss,  # RMSLE on test
                    "test_mae": true_test_mae,  # NEW MAE on test
                }
            )

            print(
                f"[{sid}] Epoch {epoch}/{epochs} "
                f"train_RMSLE {tr_loss:.4f}, train_MAE {true_train_mae:.4f}, "
                f"test_RMSLE {te_loss:.4f}, test_MAE {true_test_mae:.4f}"
            )

        # save to disk
        today_str = datetime.today().strftime("%Y-%m-%d")
        save_path = os.path.join(model_dir, f"model_{sid}_{today_str}.pth")
        torch.save(
            {
                "sid": sid,
                "model_state_dict": model.cpu().state_dict(),
                "feature_cols": feature_cols,
                "label_cols": label_cols,
                "train_frac": train_frac,
                "epochs": epochs,
            },
            save_path,
        )
        print(f"Saved model for {sid} to {save_path}")

        # keep in memory (on CPU) as well
        models[sid] = model.cpu()

    # build DataFrames
    hist_df = pd.DataFrame(history)
    summary_df = (
        hist_df.groupby("store_item")
        .agg(
            final_train_loss=("train_loss", "last"),
            final_test_loss=("test_loss", "last"),
        )
        .reset_index()
    )

    return hist_df, summary_df, models


def train_one_model_per_sid_timeseries(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    item_col: str,
    k: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 5,
    seed: int = 2025,
    model_dir: str = "output/models/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = []
    avg_history = []

    for sid in df["store_item"].unique():
        sub = df[df["store_item"] == sid].sort_values("date").reset_index(drop=True)
        sub = sub.merge(weights_df, on=item_col, how="left")

        X = sub[feature_cols].to_numpy(float)
        y = sub[label_cols].to_numpy(float)
        w = sub["weight"].to_numpy(float).reshape(-1, 1)

        full_ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(w).float(),
        )

        tscv = TimeSeriesSplit(n_splits=k)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = ShallowNN(input_dim=len(feature_cols)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = NWRMSLELoss()

            for _ in range(epochs):
                model.train()
                for xb, yb, wb in train_ld:
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                    preds = model(xb)
                    loss = loss_fn(preds, yb, wb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for xb, yb, wb in val_ld:
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                    p = model(xb).clamp(min=1e-6)
                    ld = torch.log(p + 1) - torch.log(yb + 1)
                    num += torch.sum(wb * ld**2).item()
                    den += torch.sum(wb).item()
            val_loss = (num / den) ** 0.5
            fold_metrics.append((val_loss, fold_idx))

        best_fold = min(fold_metrics)[1]
        train_idx, _ = list(tscv.split(X))[best_fold - 1]
        train_ds = Subset(full_ds, train_idx)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        model = ShallowNN(input_dim=len(feature_cols)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = NWRMSLELoss()

        epoch_train_losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for xb, yb, wb in train_ld:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb, wb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            epoch_loss = total_loss / len(train_ds)
            epoch_train_losses.append(epoch_loss)
            history.append(
                {
                    "store_item": sid,
                    "fold": best_fold,
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": None,
                }
            )
            print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}")

        for epoch, avg_loss in enumerate(epoch_train_losses, 1):
            avg_history.append(
                {"store_item": sid, "epoch": epoch, "avg_train_loss": avg_loss}
            )

        print(f"Training for store_item {sid} completed.")
        os.makedirs(model_dir, exist_ok=True)
        model = model.cpu()
        model_path = os.path.join(model_dir, f"model_{sid}.pth")
        torch.save(
            {
                "sid": sid,
                "model_state_dict": model.state_dict(),
                "fold": best_fold,
                "epochs": epochs,
                "feature_cols": feature_cols,
                "label_cols": label_cols,
            },
            model_path,
        )
        print(f"Model for store_item {sid} saved at {model_path}")

    history_df = pd.DataFrame(history)
    avg_history_df = pd.DataFrame(avg_history)
    return history_df, avg_history_df


def train_one_model_per_sid_kfold(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    item_col: str,
    k: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 5,
    shuffle: bool = True,
    seed: int = 2025,
    model_dir: str = "output/models/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = []
    fold_metrics = {}
    # all_results = {}

    for sid in df["store_item"].unique():
        print(f"Training for store_item {sid}...")
        sub = df[df["store_item"] == sid].sort_values("date").reset_index(drop=True)
        sub = sub.merge(weights_df, on=item_col, how="left")

        X = sub[feature_cols].to_numpy(float)
        y = sub[label_cols].to_numpy(float)
        w = sub["weight"].to_numpy(float).reshape(-1, 1)

        full_ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(w).float(),
        )

        kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
            val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = ShallowNN(input_dim=len(feature_cols)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = NWRMSLELoss()

            for _ in range(epochs):
                model.train()
                for xb, yb, wb in train_ld:
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                    preds = model(xb)
                    loss = loss_fn(preds, yb, wb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            num, den = 0.0, 0.0
            with torch.no_grad():
                for xb, yb, wb in val_ld:
                    xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                    p = model(xb).clamp(min=1e-6)
                    ld = torch.log(p + 1) - torch.log(yb + 1)
                    num += torch.sum(wb * ld**2).item()
                    den += torch.sum(wb).item()
            val_loss = (num / den) ** 0.5
            fold_metrics.append((val_loss, fold_idx))

        best_fold = min(fold_metrics)[1]
        train_idx, _ = list(kf.split(X))[best_fold - 1]
        train_ds = Subset(full_ds, train_idx)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

        model = ShallowNN(input_dim=len(feature_cols)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = NWRMSLELoss()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for xb, yb, wb in train_ld:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb, wb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            history.append(
                {
                    "store_item": sid,
                    "fold": best_fold,
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_ds),
                    "val_loss": None,
                }
            )
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {total_loss / len(train_ds):.4f}"
            )
        print(f"Training for store_item {sid} completed.")
        # Save the model
        # After training loop
        model = model.cpu()  # Move to CPU for portability
        model_path = os.path.join(model_dir, f"model_{sid}.pth")
        torch.save(
            {
                "sid": sid,
                "model_state_dict": model.state_dict(),
                "fold": best_fold,
                "epochs": epochs,
                "feature_cols": feature_cols,
                "label_cols": label_cols,
            },
            model_path,
        )

        print(f"Model for store_item {sid} saved at {model_path}")

        # all_results[sid] = {'model': model.cpu(), 'best_fold': best_fold}

    history_df = pd.DataFrame(history)
    return history_df


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
    y_scaler,
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
        y_scaler.inverse_transform(y_pred_scaled), columns=feature_cols
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
    last_date_df: pd.DataFrame, models: dict, y_scaler, days_to_predict: int = 16
) -> pd.DataFrame:
    """
    Predicts the next `days_to_predict` days for all store_items present in the models dict.

    Args:
        last_date_df (pd.DataFrame): Latest input row per store_item.
        models (dict): Dict of store_item -> (model, feature_cols).
        y_scaler (MinMaxScaler): Fitted scaler to inverse-transform predictions.
        days_to_predict (int): Number of days to forecast per store_item.

    Returns:
        pd.DataFrame: Combined predictions for all store_items.
    """
    all_preds = []
    for sid in models.keys():
        try:
            pred_df = predict_next_days_for_sid(
                sid, last_date_df, models, y_scaler, days_to_predict
            )
            all_preds.append(pred_df)
        except Exception as e:
            print(f"Skipping {sid} due to error: {e}")

    return pd.concat(all_preds, ignore_index=True)
