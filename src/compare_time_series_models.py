import pandas as pd
from pathlib import Path

paths = {
    "ml": Path(
        "output/metrics/20251124_2013_2014_store_2000_item_cyc_features_ml_metrics.csv"
    ),
    "dl_no_covs": Path(
        "output/metrics/20251125_2013_2014_store_2000_item_cyc_features_dl_no_covs_metrics.csv"
    ),
    "dl_past": Path(
        "output/metrics/20251124_2013_2014_store_2000_item_cyc_features_dl_past_covs_metrics.csv"
    ),
    "dl_future": Path(
        "output/metrics/20251124_2013_2014_store_2000_item_cyc_features_dl_future_covs_metrics.csv"
    ),
    "dl_past_future": Path(
        "output/metrics/20251124_2013_2014_store_2000_item_cyc_features_dl_past_future_covs_metrics.csv"
    ),
}

dfs = {k: pd.read_csv(v) for k, v in paths.items()}

# Compute mean metrics per setting
metrics = ["RMSSE", "MASE", "SMAPE", "RMSE", "MAE"]
summary = {k: df[metrics].mean() for k, df in dfs.items()}

# Create comparison DataFrame
comparison_df = pd.DataFrame(summary).T
comparison_df.index.name = "Setting"
comparison_df = comparison_df.reset_index()

# Save to CSV
output_path = Path("../output/metrics/20251125_compare_time_series_models.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(output_path, index=False)

print(f"Comparison saved to {output_path}")
print("\nComparison Summary:")
print(comparison_df)
