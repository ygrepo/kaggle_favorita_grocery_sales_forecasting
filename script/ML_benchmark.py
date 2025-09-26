# scripts/ML_benchmark.py
import sys
from pathlib import Path
import argparse
import os

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
import pickle

from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import setup_logging, get_logger
from src.data_utils import load_raw_data
from src.model_utils import create_X_y_dataset

logger = get_logger(__name__)

SEED = 42


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculate regression metrics including RMSE, MAE, MSE, R2, Pearson correlation,
    Median Absolute Error, and Explained Variance.
    """
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

    return rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance


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
) -> pd.DataFrame:
    """
    Train and evaluate a model, and save metrics to a global DataFrame.

    Parameters:
      - model_name: Name of the model.
      - model: The regression model object with a .predict() method.
      - X_train, y_train: Training features and labels.
      - X_val, y_val: Validation features and labels.
      - X_test, y_test: Test features and labels.
      - data_dir: Directory to save the model file.
    """

    # Get predictions and calculate metrics for each split.
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred)

    val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_pred)

    test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_pred)

    # Prepare a list of datasets and corresponding metrics.
    datasets = ["Training", "Validation", "Test"]
    metrics = [train_metrics, val_metrics, test_metrics]
    rows = []
    for dataset, metric in zip(datasets, metrics):
        rmse, mae, mse, r2, pearson_corr, median_ae, explained_variance = metric
        rows.append(
            {
                "Model": model_name,
                "Dataset": dataset,
                "RMSE": rmse,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
                "Pearson": pearson_corr,
                "Median_AE": median_ae,
                "Explained_Variance": explained_variance,
            }
        )

    # Update the global metrics DataFrame
    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    return metrics_df


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
    return p.parse_args()


def main():

    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Log level: {args.log_level}")
        logger.info(f"Log dir: {args.log_dir}")
        logger.info(f"Data fn: {args.data_fn}")
        logger.info(f"Model dir: {args.model_dir}")
        logger.info(f"data_fn: {args.data_fn}")

        df = load_raw_data(args.data_fn)
        X_train, X_val, X_test, y_train, y_val, y_test = create_X_y_dataset(df)

        logger.info("Running models...")

        logger.info("Random Forest")
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf_model.fit(X_train, y_train)
        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Dataset",
                "RMSE",
                "MAE",
                "MSE",
                "R2",
                "Pearson",
                "Median_AE",
                "Explained_Variance",
            ]
        )
        model_name = "Random Forest"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            rf_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_dir = Path(args.model_dir)
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(rf_model, model_name, model_filename)

        logger.info("SVR")
        # SVR
        svr_model = SVR(kernel="rbf")
        svr_model.fit(X_train, y_train)
        model_name = "SVR"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            svr_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(svr_model, model_name, model_filename)

        logger.info("GBM")
        # GBM
        gbm_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=SEED
        )
        gbm_model.fit(X_train, y_train)
        model_name = "GBM"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            gbm_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(gbm_model, model_name, model_filename)

        logger.info("Linear Regression")
        # Linear Regression
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)
        model_name = "Linear Regression"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            lin_reg_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(lin_reg_model, model_name, model_filename)

        logger.info("MLP")
        # MLP
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            random_state=SEED,
        )
        mlp_model.fit(X_train, y_train)
        model_name = "MLP"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            mlp_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(mlp_model, model_name, model_filename)

        logger.info("XGBoost")
        # XGBoost
        xgb_model = XGBRegressor(random_state=SEED, eval_metric="rmse")
        xgb_model.fit(X_train, y_train)
        model_name = "XGBoost"
        metrics_df = evaluate_model(
            metrics_df,
            model_name,
            xgb_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        model_filename = (
            model_dir
            / f"{model_name.replace(' ', '_')}_{args.embedding}_{args.dataset}_{args.splitmode}_model_regression.pkl"
        )
        save_model(xgb_model, model_name, model_filename)

        logger.info("Done!")

        output_dir = Path(args.output_dir)
        datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_csv = (
            output_dir
            / f"{datestamp}_ML_metrics_{args.embedding}_{args.dataset}_{args.splitmode}.csv"
        )

        logger.info(f"Saving metrics to {result_csv}")
        # Save metrics_df to CSV
        metrics_df.to_csv(result_csv, index=False)
        logger.info("Metrics saved!")

    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
