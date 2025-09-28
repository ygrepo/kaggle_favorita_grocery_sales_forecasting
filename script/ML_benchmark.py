# scripts/ML_benchmark.py
import sys
from pathlib import Path
import argparse
import os
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.isotonic import IsotonicRegression

import pickle

from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import setup_logging, get_logger, get_n_jobs
from src.data_utils import (
    load_raw_data,
    sort_df,
    build_feature_and_label_cols,
    X_FEATURES,
)
from src.model_utils import (
    create_X_y_dataset,
    InverseTransformer,
    inverse_transform,
    fit_rf_with_tqdm,
    fit_gbr_with_tqdm,
    fit_hgb_with_tqdm,
    spinner,
    XGBoostTQDMCallback,
)

logger = get_logger(__name__)

SEED = 42


def smape(y_true: np.ndarray, y_pred: np.ndarray, denom_floor=1e-8) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (array-like): ground truth values
        y_pred (array-like): predicted values
        denom_floor (float): to cap the denominator

    Returns:
        float: SMAPE value in percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = np.maximum(
        (np.abs(y_true) + np.abs(y_pred)) / 2.0, denom_floor
    )
    return float(np.mean(numerator / denominator) * 100)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate regression metrics including RMSE, MAE, MSE, R2, Pearson correlation,
    Median Absolute Error, and Explained Variance.
    """
    # Ensure both arrays have the same shape (flatten to 1D)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate Pearson correlation coefficient (returns coefficient and p-value)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # Calculate Median Absolute Error
    median_ae = median_absolute_error(y_true, y_pred)

    # Calculate Explained Variance Score
    explained_variance = explained_variance_score(y_true, y_pred)

    smape_val = smape(y_true, y_pred)
    # Convert all values to Python float type
    return (
        float(np.asarray(rmse).item()),
        float(np.asarray(mse).item()),
        float(np.asarray(r2).item()),
        float(np.asarray(pearson_corr).item()),
        float(np.asarray(mae).item()),
        float(np.asarray(median_ae).item()),
        smape_val,
        float(np.asarray(explained_variance).item()),
    )


# ---- Helper to append one set of metrics ----
def _append_rows(
    tag: str,
    model_name: str,
    units: str,
    rows: list[dict],
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    ptr: np.ndarray,
    pva: np.ndarray,
    pte: np.ndarray,
):
    for dataset, yt, yp in (
        ("Training", ytr, ptr),
        ("Validation", yva, pva),
        ("Test", yte, pte),
    ):
        (
            rmse,
            mse,
            r2,
            pearson_corr,
            mae,
            median_ae,
            smape_val,
            explained_variance,
        ) = calculate_metrics(yt, yp)
        rows.append(
            {
                "Model": model_name,
                "Dataset": dataset,
                "Units": units,  # "scaled" or "original"
                "Calibrated": tag,  # "raw" or "calibrated"
                "RMSE": rmse,
                "MSE": mse,
                "R2": r2,
                "Pearson": pearson_corr,
                "MAE": mae,
                "Median_AE": median_ae,
                "SMAPE": smape_val,
                "Explained_Variance": explained_variance,
            }
        )


# Apply mapping to val/test preds
def _apply(m, x: np.ndarray):
    x = np.asarray(x).ravel()
    return (
        m.transform(x)
        if hasattr(m, "transform")
        else m.predict(x.reshape(-1, 1))
    )


def evaluate_model(
    metrics_df: pd.DataFrame,
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_scaler: Optional["InverseTransformer"] = None,
    units: str = "scaled",  # "scaled" or "original"
    calibrate: Optional[str] = None,  # None | "isotonic" | "linear"
) -> pd.DataFrame:
    """
    Evaluate a trained model and append rows to metrics_df.
    - If units == "original" and y_scaler is provided, predictions & targets are inverse-transformed.
    - If calibrate is set, a post-hoc mapping is fit on (val_pred, y_val) and applied to val/test preds.
      Calibration is reported in separate rows with Calibrated=True.
    """

    # ---- Raw predictions (model is already fitted outside) ----
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # ---- Work in chosen units ----
    if y_scaler is not None and units == "original":
        y_train_eval = inverse_transform(y_train, y_scaler)
        y_val_eval = inverse_transform(y_val, y_scaler)
        y_test_eval = inverse_transform(y_test, y_scaler)
        train_pred = inverse_transform(train_pred, y_scaler)
        val_pred = inverse_transform(val_pred, y_scaler)
        test_pred = inverse_transform(test_pred, y_scaler)
    else:
        y_train_eval, y_val_eval, y_test_eval = y_train, y_val, y_test

    rows = []

    # ---- Raw metrics ----
    _append_rows(
        "raw",
        model_name,
        units,
        rows,
        y_train_eval,
        y_val_eval,
        y_test_eval,
        train_pred,
        val_pred,
        test_pred,
    )

    # ---- Optional calibration on validation ----
    if calibrate is not None:
        cal = None
        val_x = np.asarray(val_pred).ravel()
        val_y = np.asarray(y_val_eval).ravel()

        # Need at least 2 distinct prediction values to calibrate meaningfully
        can_calibrate = (
            np.isfinite(val_x).all()
            and np.isfinite(val_y).all()
            and (np.unique(val_x).size > 1)
        )

        if can_calibrate:
            if calibrate.lower() == "isotonic":

                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(val_x, val_y)
            elif calibrate.lower() == "linear":

                cal = LinearRegression()
                cal.fit(val_x.reshape(-1, 1), val_y)
            else:
                raise ValueError(
                    "calibrate must be None, 'isotonic', or 'linear'."
                )

            train_pred_cal = _apply(cal, train_pred)
            val_pred_cal = _apply(cal, val_pred)
            test_pred_cal = _apply(cal, test_pred)

            _append_rows(
                "calibrated",
                model_name,
                units,
                rows,
                y_train_eval,
                y_val_eval,
                y_test_eval,
                train_pred_cal,
                val_pred_cal,
                test_pred_cal,
            )
        else:
            # Append an empty row
            rows.append(
                {
                    "Model": model_name,
                    "Dataset": "Validation",
                    "Units": units,
                    "Calibrated": "calibration_skipped",
                    "RMSE": np.nan,
                    "MSE": np.nan,
                    "R2": np.nan,
                    "Pearson": np.nan,
                    "MAE": np.nan,
                    "Median_AE": np.nan,
                    "SMAPE": np.nan,
                    "Explained_Variance": np.nan,
                }
            )

    # ---- Concat into metrics_df ----
    return pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)


def save_model(model, model_name, model_filename: Path):
    # Save the model to disk
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"{model_name} model saved to {model_filename}")


def parse_args():
    p = argparse.ArgumentParser(description="Create and load PLM model")
    p.add_argument("--log_dir", type=str, default="")
    p.add_argument("--log_fn", type=str, default="")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--data_fn", type=str, default="")
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--n_jobs", type=int, default=1)
    return p.parse_args()


def main():

    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"N_jobs: {args.n_jobs}")
        n_jobs = get_n_jobs(args.n_jobs)
        logger.info(f"Effective n_jobs: {n_jobs}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Log level: {args.log_level}")
        logger.info(f"Log dir: {args.log_dir}")
        logger.info(f"Data fn: {args.data_fn}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"data_fn: {args.data_fn}")
        data_fn = Path(args.data_fn).resolve()
        log_dir = Path(args.log_dir).resolve()

        df = load_raw_data(data_fn)
        df = sort_df(df)

        features = build_feature_and_label_cols()

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _,
            _,
            _,
            _,
            y_scaler,
            _,
        ) = create_X_y_dataset(
            df,
            val_horizon=7,
            test_horizon=7,
            y_col="y",
            weight_col="weight",
            x_cols=features["X_FEATURES"],
        )

        logger.info("Running models...")

        model_name = "Random Forest"
        logger.info(f"Model:{model_name}")
        y_train_raveled = y_train.ravel()

        model = RandomForestRegressor(
            n_estimators=600,  # cap
            max_depth=12,  # shallower
            min_samples_leaf=50,  # bigger leaves
            min_samples_split=100,
            max_features=0.6,  # feature subsampling
            bootstrap=True,  # required for OOB
            oob_score=True,  # monitored by early-stop
            n_jobs=n_jobs,
            random_state=SEED,
        )

        model = fit_rf_with_tqdm(
            model,
            X_train,
            y_train_raveled,
            step=100,
            desc="RF fit (OOB EST)",
            patience=3,
            min_delta=1e-4,
            time_budget_s=None,
        )

        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Dataset",
                "Units",
                "Calibrated",
                "RMSE",
                "MSE",
                "R2",
                "Pearson",
                "MAE",
                "Median_AE",
                "SMAPE",
                "Explained_Variance",
            ]
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )
        model_dir = Path(args.model_dir)
        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        # model_name = "SVR"
        # logger.info(f"Model:{model_name}")
        # model = SVR(kernel="rbf")
        # with spinner("SVR fit"):
        #     model.fit(X_train, y_train_raveled)
        # metrics_df = evaluate_model(
        #     metrics_df,
        #     model_name,
        #     model,
        #     X_train,
        #     y_train,
        #     X_val,
        #     y_val,
        #     X_test,
        #     y_test,
        #     y_scaler,
        #     "scaled",
        #     calibrate=None,
        # )
        # metrics_df = evaluate_model(
        #     metrics_df,
        #     model_name,
        #     model,
        #     X_train,
        #     y_train,
        #     X_val,
        #     y_val,
        #     X_test,
        #     y_test,
        #     y_scaler,
        #     "original",
        #     calibrate="isotonic",
        # )
        # model_filename = (
        #     model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        # )
        # save_model(model, model_name, model_filename)

        model_name = "HistGBM"
        logger.info(f"Model:{model_name}")

        model = HistGradientBoostingRegressor(
            loss="squared_error",  # or "absolute_error" for L1-like robustness
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,  # tree size control (analogous to depth)
            min_samples_leaf=50,  # combats overfit; tune
            l2_regularization=1.0,
            # max_bins=255,  # more precise splits
            early_stopping=False,
            # validation_fraction=0.1,
            # n_iter_no_change=30,
            random_state=SEED,
        )

        model = fit_hgb_with_tqdm(
            model, X_train, y_train_raveled, step=25, desc="HistGBM fit"
        )

        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )

        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        model_name = "HuberLossGBM"
        logger.info(f"Model:{model_name}")
        model = GradientBoostingRegressor(
            loss="huber",  # robust to outliers
            alpha=0.9,  # Huber quantile
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            random_state=SEED,
        )
        model = fit_gbr_with_tqdm(
            model,
            X_train,
            y_train_raveled,
            step=25,
            desc="HuberLossGBM fit",
        )

        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )
        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        model_name = "Linear Regression"
        logger.info(f"Model:{model_name}")
        model = LinearRegression()
        with spinner("LR fit"):
            model.fit(X_train, y_train_raveled)
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )
        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        model_name = "MLP"
        logger.info(f"Model:{model_name}")
        # MLP
        model = MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            random_state=SEED,
        )
        with spinner("MLP fit"):
            model.fit(X_train, y_train_raveled)
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )
        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        model_name = "XGBoost"
        logger.info(f"Model:{model_name}")
        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,  # shallower trees
            min_child_weight=10,  # stronger leaf mins
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,  # L2
            reg_alpha=0.0,  # L1 if you want it
            objective="reg:squarederror",
            random_state=SEED,
        )
        nrounds = model.get_params().get("n_estimators", 0)
        model.fit(
            X_train,
            y_train_raveled,
            eval_set=[(X_val, y_val.ravel())],
            callbacks=[XGBoostTQDMCallback(total=nrounds, desc="XGBoost fit")],
            verbose=False,
            early_stopping_rounds=100,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "scaled",
            calibrate=None,
        )
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_scaler,
            "original",
            calibrate="isotonic",
        )
        model_filename = (
            model_dir / f"{model_name.replace(' ', '_')}_model_regression.pkl"
        )
        save_model(model, model_name, model_filename)

        logger.info("Done!")

        datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_csv = log_dir / f"{datestamp}_ML_metrics.csv"

        logger.info(f"Saving metrics to {result_csv}")
        # Save metrics_df to CSV
        metrics_df.to_csv(result_csv, index=False)
        logger.info("Metrics saved!")

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
