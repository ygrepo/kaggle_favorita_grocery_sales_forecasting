from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def compare_metrics_by_model(
    paths: Dict[str, Path],
    metrics: Optional[List[str]] = None,
    model_filters: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[Path] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Compare metrics by Model across multiple experiment settings, including
    both mean and standard deviation for each metric.

    Args:
        paths: dict mapping setting_name -> Path to CSV file.
        metrics: list of metric columns to aggregate.
                 Default = ["RMSSE", "SMAPE", "MARRE"].
        model_filters: optional dict mapping setting_name -> list of models
                       to keep for that file.
        output_path: optional path to save output CSV.
        agg: aggregation for central tendency ("mean" or "median").

    Returns:
        DataFrame with columns:
            Setting, Model,
            <metric>_mean, <metric>_std
    """

    if metrics is None:
        metrics = ["RMSSE", "SMAPE", "MARRE"]

    # Load all CSVs
    dfs = {}
    for key, path in paths.items():
        df = pd.read_csv(path)

        # Apply model filtering if provided
        if model_filters and key in model_filters:
            df = df[df["Model"].isin(model_filters[key])]

        dfs[key] = df

    summary_rows = []
    for setting, df in dfs.items():

        if df.empty:
            continue

        grouped = df.groupby("Model")[metrics]

        # Compute mean/median
        if agg == "mean":
            center_df = grouped.mean().add_suffix("_mean")
        elif agg == "median":
            center_df = grouped.median().add_suffix("_median")
        else:
            raise ValueError(f"Unsupported agg function: {agg}")

        # Compute std always
        std_df = grouped.std().add_suffix("_std")

        # Merge mean/median + std
        merged = pd.concat([center_df, std_df], axis=1)

        # reorder so each metric's mean/std are adjacent
        ordered_cols = []
        for m in metrics:
            ordered_cols.extend([f"{m}_mean", f"{m}_std"])

        merged = merged[ordered_cols]

        merged = merged.reset_index()
        merged.insert(0, "Setting", setting)

        summary_rows.append(merged)

    # Combine all settings
    comparison_df = pd.concat(summary_rows, ignore_index=True)

    comparison_df.sort_values(by="RMSSE_mean", inplace=True)
    print(comparison_df)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"Comparison saved to {output_path}")

    return comparison_df


def main():

    paths = {
        # "ml_no_covs": Path(
        #     "output/metrics/20251130_2013_2014_store_2000_item_cyc_features_ml_no_covs_metrics.csv"
        # ),
        # "ml_past_covs": Path(
        #     "output/metrics/20251205_2013_2014_store_2000_item_cyc_features_all_ml_past_covs_metrics.csv"
        # ),
        # "ml_future_covs": Path(
        #     "output/metrics/20251201_2013_2014_store_2000_item_cyc_features_ml_future_covs_metrics.csv"
        # ),
        "Global_ML_past_future_covs": Path(
            # "output/metrics/20251206_2013_2014_store_2000_item_cyc_features_ml_past_future_covs_metrics.csv"
            "output/metrics/20251207_2013_2014_store_2000_item_cyc_features_ML_GLOBAL_past_future_covs_metrics.csv"
        ),
        # "dl_no_covs": Path(
        #     "output/metrics/20251130_2013_2014_store_2000_item_cyc_features_dl_no_covs_metrics.csv"
        # ),
        # "dl_past_covs": Path(
        #     "output/metrics/20251204_2013_2014_store_2000_item_cyc_features_all_dl_past_covs_metrics.csv"
        # ),
        # "tcn_future_covs": Path(
        #     "output/metrics/20251205_2013_2014_store_2000_item_cyc_features_tcn_all_dl_future_covs_metrics.csv"
        # ),
        # "tcn_past_future_covs_1": Path(
        #     "output/metrics/20251205_2013_2014_store_2000_item_cyc_features_tcn_all_dl_past_future_covs_metrics_1.csv"
        # ),
        # "tcn_past_future_covs": Path(
        #     "output/metrics/20251205_2013_2014_store_2000_item_cyc_features_tcn_all_dl_past_future_covs_metrics.csv"
        # ),
        "Global_DL_past_future_covs": Path(
            # "output/metrics/20251206_2013_2014_store_2000_item_cyc_features_DL_past_future_covs_metrics.csv"
            "output/metrics/20251207_2013_2014_store_2000_item_cyc_features_DL_GLOBAL_past_future_covs_metrics.csv"
        ),
    }

    model_filters = {
        # "dl_no_covs": ["TFT"],
        # "dl_past_future": ["BLOCK_RNN"],
        # "dl_future": ["TSMixer"],
        # "dl_past": ["BLOCK_RNN"],
    }
    output_path = Path(
        "output/metrics/20251207_compare_time_series_models.csv"
    )

    compare_metrics_by_model(
        paths, model_filters=model_filters, output_path=output_path
    )


if __name__ == "__main__":
    main()
