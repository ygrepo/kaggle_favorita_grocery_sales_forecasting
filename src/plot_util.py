import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import sem, t


def plot_loss_with_comparison(
    hist_df_current,
    hist_df_previous=None,
    overall_title=None,
    fn=None,
    xvline=None,
    show_ci=False,
):
    def compute_summary_stats(df, loss_col):
        grouped = df.groupby("epoch")[loss_col]
        median = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        if show_ci:
            mean = grouped.mean()
            stderr = grouped.apply(sem)
            ci_range = t.ppf(0.975, df.groupby("epoch").count()[loss_col] - 1) * stderr
            ci_low = mean - ci_range
            ci_high = mean + ci_range
        else:
            ci_low = ci_high = None
        return median, q1, q3, ci_low, ci_high

    epochs = sorted(hist_df_current["epoch"].unique())

    med_tr, q1_tr, q3_tr, ci_tr_low, ci_tr_high = compute_summary_stats(
        hist_df_current, "train_loss"
    )
    med_te, q1_te, q3_te, ci_te_low, ci_te_high = compute_summary_stats(
        hist_df_current, "test_loss"
    )

    fig, (ax_tr, ax_te) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- Train Loss Plot ----
    ax_tr.plot(epochs, med_tr, color="blue", label="Median Train (Current)")
    ax_tr.fill_between(
        epochs, q1_tr, q3_tr, color="blue", alpha=0.3, label="IQR (Current)"
    )
    if show_ci and ci_tr_low is not None:
        ax_tr.fill_between(
            epochs, ci_tr_low, ci_tr_high, color="blue", alpha=0.15, label="95% CI"
        )

    if hist_df_previous is not None:
        prev_median_tr = hist_df_previous.groupby("epoch")["train_loss"].median()
        ax_tr.plot(
            epochs,
            prev_median_tr,
            color="gray",
            linestyle="--",
            label="Median Train (Previous)",
        )

    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)
    ax_tr.axvline(x=xvline, color="green", linestyle="--", label=f"Epoch {xvline}")

    # ---- Validation Loss Plot ----
    ax_te.plot(epochs, med_te, color="orange", label="Median Val (Current)")
    ax_te.fill_between(
        epochs, q1_te, q3_te, color="orange", alpha=0.3, label="IQR (Current)"
    )
    if show_ci and ci_te_low is not None:
        ax_te.fill_between(
            epochs, ci_te_low, ci_te_high, color="orange", alpha=0.15, label="95% CI"
        )

    if hist_df_previous is not None:
        prev_median_te = hist_df_previous.groupby("epoch")["test_loss"].median()
        ax_te.plot(
            epochs,
            prev_median_te,
            color="gray",
            linestyle="--",
            label="Median Val (Previous)",
        )

    ax_te.set_title("Validation Loss by Epoch", fontsize=16, fontweight="bold")
    ax_te.set_xlabel("Epoch", fontsize=14)
    ax_te.set_ylabel("Validation Loss", fontsize=14)
    ax_te.axvline(x=50, color="green", linestyle="--", label="Epoch 50")

    # Shared formatting
    for ax in (ax_tr, ax_te):
        ax.legend(fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Overall title
    if overall_title:
        fig.suptitle(overall_title, fontsize=20, fontweight="bold")
        plt.subplots_adjust(top=0.9)

    plt.tight_layout(pad=3)
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_all_sids_losses(hist_df, overall_title=None, fn=None):
    sids = hist_df["store_item"].unique()
    epochs = sorted(hist_df["epoch"].unique())

    fig, (ax_tr, ax_te) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # — Train Loss —
    for sid in sids:
        df_sid = hist_df[hist_df["store_item"] == sid]
        ax_tr.plot(
            df_sid["epoch"], df_sid["train_loss"], marker="o", linewidth=1, label=sid
        )
    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)

    # — Validation Loss —
    for sid in sids:
        df_sid = hist_df[hist_df["store_item"] == sid]
        ax_te.plot(
            df_sid["epoch"], df_sid["test_loss"], marker="o", linewidth=1, label=sid
        )
    ax_te.set_title("Validation Loss by Epoch", fontsize=16, fontweight="bold")
    ax_te.set_xlabel("Epoch", fontsize=14)
    ax_te.set_ylabel("Validation Loss", fontsize=14)

    # --- Dynamic x-axis tick setup ---
    x_min, x_max = min(epochs), max(epochs)
    x_range = x_max - x_min
    if x_range > 50:
        x_major, x_minor = 10, 2
    elif x_range > 20:
        x_major, x_minor = 5, 1
    elif x_range > 10:
        x_major, x_minor = 2, 1
    else:
        x_major, x_minor = 1, 0.5

    for ax in (ax_tr, ax_te):
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))

        # --- Dynamic y-axis tick setup ---
        y_data = hist_df["train_loss"] if ax == ax_tr else hist_df["test_loss"]
        y_min, y_max = y_data.min(), y_data.max()
        y_range = y_max - y_min
        if y_range < 1e-3:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        else:
            margin = y_range * 0.1
            y_min -= margin
            y_max += margin
        ax.set_ylim(y_min, y_max)

        tick_range = y_max - y_min
        if tick_range > 300:
            y_major, y_minor = 50, 10
        elif tick_range > 100:
            y_major, y_minor = 20, 5
        elif tick_range > 50:
            y_major, y_minor = 10, 2
        elif tick_range > 10:
            y_major, y_minor = 2, 0.5
        elif tick_range > 1:
            y_major, y_minor = 0.5, 0.1
        else:
            y_major, y_minor = 0.2, 0.05

        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

        ax.grid(True, which="major", linestyle="--", alpha=0.5)

        # Format ticks
        ax.tick_params(axis="x", which="major", length=6)
        ax.tick_params(axis="x", which="minor", length=4)
        ax.tick_params(axis="y", which="major", length=6)
        ax.tick_params(axis="y", which="minor", length=4)

        # Tick label formatting
        for lbl in ax.get_xticklabels(which="major"):
            lbl.set_fontsize(10)
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels(which="major"):
            lbl.set_fontsize(10)
            lbl.set_fontweight("bold")

    # — Overall title —
    if overall_title:
        fig.suptitle(overall_title, fontsize=20, fontweight="bold")
        # Adjust top spacing dynamically based on title length
        title_len = len(overall_title)
        top_margin = 0.99 - min(0.02, 0.001 * title_len)  # Clamp adjustment
        plt.subplots_adjust(top=top_margin)

    plt.tight_layout(pad=3)
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_median_iqr_loss(hist_df, overall_title=None, fn=None, xvline=None):
    epochs = sorted(hist_df["epoch"].unique())

    # Group data by epoch
    grouped = hist_df.groupby("epoch")

    # Compute median and IQR for each epoch
    median_train = grouped["train_loss"].median()
    q1_train = grouped["train_loss"].quantile(0.25)
    q3_train = grouped["train_loss"].quantile(0.75)

    median_val = grouped["test_loss"].median()
    q1_val = grouped["test_loss"].quantile(0.25)
    q3_val = grouped["test_loss"].quantile(0.75)

    fig, (ax_tr, ax_te) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Plot Train Loss ---

    ax_tr.plot(
        epochs,
        median_train,
        color="blue",
        label="Training Loss per Epoch (All Store–Item Pairs)",
    )
    ax_tr.fill_between(epochs, q1_train, q3_train, color="blue", alpha=0.6, label="IQR")
    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)

    # --- Plot Validation Loss ---
    ax_te.plot(
        epochs,
        median_val,
        color="orange",
        label="Validation Loss per Epoch (All Store–Item Pairs)",
    )
    ax_te.fill_between(epochs, q1_val, q3_val, color="orange", alpha=0.6, label="IQR")
    ax_te.set_title("Validation Loss by Epoch", fontsize=16, fontweight="bold")
    ax_te.set_xlabel("Epoch", fontsize=14)
    ax_te.set_ylabel("Validation Loss", fontsize=14)

    for ax in (ax_tr, ax_te):
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", which="major", length=6, labelsize=12)
        ax.tick_params(axis="y", which="major", length=6, labelsize=12)
        ax.legend(fontsize=12)
        ax.axvline(x=xvline, linestyle="--", color="green", label=f"Epoch {xvline}")

    if overall_title:
        fig.suptitle(overall_title, fontsize=20, fontweight="bold")
        top_margin = 0.99 - min(0.02, 0.001 * len(overall_title))
        plt.subplots_adjust(top=top_margin)

    plt.tight_layout(pad=3)
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_loss_per_sid(
    hist_df,
    title,
    sid,
    fn,
    y_label="Loss",
    train_col="train_loss",
    test_col="test_loss",
):
    """
    Plot train and test loss vs. epoch for a given store_item (sid),
    save to file `fn`.
    """
    df = hist_df[hist_df["store_item"] == sid].sort_values("epoch")
    if df.empty:
        raise ValueError(f"No data found for store_item = {sid!r}")

    fig, ax = plt.subplots(figsize=(12, 5))

    # plot
    ax.plot(df["epoch"], df[train_col], marker="o", label="Train Loss")
    ax.plot(df["epoch"], df[test_col], marker="o", label="Validation Loss")

    # titles & labels
    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=16, fontweight="bold")

    # --- Dynamic x-axis ticks ---
    x_min, x_max = df["epoch"].min(), df["epoch"].max()
    x_range = x_max - x_min
    if x_range > 50:
        x_major = 10
        x_minor = 2
    elif x_range > 20:
        x_major = 5
        x_minor = 1
    elif x_range > 10:
        x_major = 2
        x_minor = 1
    else:
        x_major = 1
        x_minor = 0.5
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))

    # --- Dynamic y-axis ticks ---
    y_values = pd.concat([df[train_col], df[test_col]])
    y_min, y_max = y_values.min(), y_values.max()
    y_range = y_max - y_min

    if y_range < 1e-3:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    else:
        margin = y_range * 0.1
        y_min -= margin
        y_max += margin
    ax.set_ylim(y_min, y_max)
    if y_range > 300:
        y_major = 50
        y_minor = 10
    elif y_range > 100:
        y_major = 20
        y_minor = 5
    elif y_range > 50:
        y_major = 10
        y_minor = 2
    elif y_range > 10:
        y_major = 2
        y_minor = 0.5
    elif y_range > 1:
        y_major = 0.5
        y_minor = 0.1
    else:
        y_major = 0.2
        y_minor = 0.05

    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

    # grid, legend
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.legend()

    # tick styling
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=4)
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=4)

    plt.tight_layout(pad=2)
    plt.savefig(fn, dpi=300)
    plt.show()
    plt.close(fig)


def plot_sales_histogram(
    df: pd.DataFrame, sid: str, bins: int = 50, log_scale: bool = False, fn: str = None
):
    """
    Plot histogram of unit_sales for a specific store_item (sid).

    Parameters:
        df: pd.DataFrame – main dataset containing 'unit_sales' and 'store_item'
        sid: str – store_item ID to filter the data
        bins: int – number of histogram bins
        log_scale: bool – if True, applies log10 to x-axis
    """
    sub_df = df[df["store_item"] == sid]

    if sub_df.empty:
        raise ValueError(f"No data found for store_item = {sid!r}")

    sales = sub_df["unit_sales"].astype(float)
    if log_scale:
        sales = sales[sales > 0]  # avoid log(0)
        sales = np.log10(sales)

    plt.figure(figsize=(10, 6))
    sns.histplot(sales, bins=bins, kde=True, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Unit Sales for {sid}", fontsize=24, fontweight="bold")
    plt.xlabel(
        "log10(Unit Sales)" if log_scale else "Unit Sales",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Frequency", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def plot_final_percent_mae_per_sid(
    summary_df: pd.DataFrame,
    title: str | None = None,
    fn: str | None = None,
    y_lim: tuple[float, float] | None = None,
):
    """
    Plot final train/test percent MAE for each store_item (sid).

    Parameters
    ----------
    summary_df : DataFrame
        Must contain columns ``store_item``, ``final_train_percent_mae`` and
        ``final_test_percent_mae``.
    title : str, optional
        Overall title for the plot.
    fn : str, optional
        If provided, save the figure to this filename (dpi=300).
    y_lim : (float, float), optional
        y‑axis limits as (ymin, ymax).  Pass None to let Matplotlib auto‑scale.
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty")

    df = summary_df.sort_values("store_item").reset_index(drop=True)
    x = np.arange(len(df))

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["final_train_percent_mae"], marker="o", label="Train %MAV")
    ax.plot(x, df["final_test_percent_mae"], marker="o", label="Validation %MAV")

    ax.set_xlabel("store_item", fontsize=16, fontweight="bold")
    ax.set_ylabel("%MAV", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["store_item"], rotation=90, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title:
        ax.set_title(title, fontsize=24, fontweight="bold")

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def visualize_clustered_matrix(X, U, V_list, title="Clustered Matrix", fn: str = None):
    row_order = np.argsort(np.argmax(U, axis=1))
    col_cluster_ids = np.zeros(X.shape[1], dtype=int)
    for p, Vp in enumerate(V_list):
        cluster_ids = np.argmax(Vp, axis=1)
        mask = np.any(Vp, axis=1)
        col_cluster_ids[mask] = cluster_ids[mask] + sum(
            [v.shape[1] for v in V_list[:p]]
        )
    col_order = np.argsort(col_cluster_ids)
    clustered_X = X[row_order, :][:, col_order]
    plt.figure(figsize=(10, 6))
    sns.heatmap(clustered_X, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Variables (Clustered)")
    plt.ylabel("Objects (Clustered)")
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def visualize_spectral_biclustering(
    data, row_labels, col_labels, title="Spectral Biclustering", fn: str = None
):
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)
    reordered_data = data[np.ix_(row_order, col_order)]

    plt.figure(figsize=(6, 6))
    sns.heatmap(reordered_data, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def visualize_gdkm_cv_scores(results_df):
    pivot_sil = results_df.pivot(index="P", columns="Q", values="Mean Silhouette")
    pivot_loss = results_df.pivot(index="P", columns="Q", values="Mean Loss")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(pivot_sil, annot=True, fmt=".2f", ax=axes[0], cmap="viridis")
    axes[0].set_title("Mean Silhouette Score")
    axes[0].set_xlabel("Q (Column Clusters)")
    axes[0].set_ylabel("P (Row Clusters)")

    sns.heatmap(pivot_loss, annot=True, fmt=".2f", ax=axes[1], cmap="viridis_r")
    axes[1].set_title("Mean Loss (SSR)")
    axes[1].set_xlabel("Q (Column Clusters)")
    axes[1].set_ylabel("P (Row Clusters)")

    plt.tight_layout()
    plt.show()


def plot_gdkm_elbow_curve(results_df):
    """
    Plots an elbow curve of GDKM loss vs. number of row clusters (P) for each Q.
    This helps visualize the tradeoff in reconstruction loss as P increases.

    Parameters:
    - results_df: DataFrame returned by compute_gdkm_cv_scores
    """
    plt.figure(figsize=(8, 6))

    for q in sorted(results_df["Q"].unique()):
        subset = results_df[results_df["Q"] == q]
        plt.plot(subset["P"], subset["Mean Loss"], marker="o", label=f"Q = {q}")

    plt.title("GDKM Elbow Curve (Loss vs P for each Q)")
    plt.xlabel("Number of Row Clusters (P)")
    plt.ylabel("Mean Reconstruction Loss (SSR)")
    plt.legend(title="Column Clusters (Q)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_biclustering_elbow(
    df: pd.DataFrame,
    *,
    title: str = "Biclustering Elbow",
    title_fontsize: int = 24,
    x_col: str = "n_clusters",
    metric: str = "Explained Variance (%)",  # or "Mean Loss", "Mean Silhouette"
    tick_step: int = 2,
    vline_x: int | None = None,
    vline_kwargs: dict | None = None,
    figsize: tuple[int, int] = (9, 5),
    fn: str | None = None,  # optional png filename
):
    """
    Plot an elbow curve for Spectral Biclustering results.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'n_clusters', and the chosen `metric`.
    x_col : str
        Column name in `df` to plot on the x‑axis.
    metric : str
        Column name in `df` to plot on the y‑axis.
    fn : str or None
        If provided, the plot is saved to this filename (dpi=300).
    """
    # --------------------  Plot  --------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        df[x_col],
        df[metric],
        marker="o",
        label=metric,
    )

    ax.set_xlabel("Number of Store_Item Clusters", fontsize=16, fontweight="bold")
    ax.set_ylabel(metric, fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Force integer x‑ticks
    # keep only every `tick_step`‑th integer
    ticks = df[x_col]
    sparse_ticks = ticks[::tick_step]
    ax.set_xticks(sparse_ticks)
    ax.set_xticklabels(sparse_ticks, rotation=0)

    # -------- optional vertical line --------
    if vline_x is not None:
        default_line_style = dict(color="red", linestyle="--", linewidth=1.5)
        if vline_kwargs:  # allow custom overrides
            default_line_style.update(vline_kwargs)
        ax.axvline(vline_x, **default_line_style)
        # Optional text label
        ax.text(
            vline_x,
            ax.get_ylim()[0],
            f"  k={vline_x}",
            color=default_line_style["color"],
            va="bottom",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_spectral_biclustering_heatmap(results_df, fn: str = None):
    pivot = results_df.pivot(
        index="n_row", columns="n_col", values="Explained Variance (%)"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(
        "Spectral Biclustering – % Variance Explained",
        fontsize=24,
        fontweight="bold",
    )
    plt.xlabel("n_col (column clusters)", fontsize=16, fontweight="bold")
    plt.ylabel("n_row (row clusters)", fontsize=16, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def plot_spectral_clustering_elbows(
    result_dfs: list[pd.DataFrame],
    *,
    metric: str = "Explained Variance (%)",
    titles: list[str] | None = None,
    figsize: tuple[int, int] = (8, 12),
    fn: str | None = None,
):
    """Plot elbow curves from multiple spectral clustering results.

    Parameters
    ----------
    result_dfs : list of DataFrame
        Each DataFrame is returned by :func:`compute_spectral_clustering_cv_scores`.
    metric : str, optional
        Metric column to plot on the y-axis.
    titles : list of str, optional
        Optional subplot titles. Must match the length of ``result_dfs`` if
        provided.
    figsize : tuple of int, optional
        Overall figure size.
    fn : str, optional
        If given, save the figure to this path.
    """

    if titles and len(titles) != len(result_dfs):
        raise ValueError("Length of titles must match result_dfs")

    fig, axes = plt.subplots(len(result_dfs), 1, figsize=figsize, sharex=True)
    if len(result_dfs) == 1:
        axes = [axes]

    for idx, (df, ax) in enumerate(zip(result_dfs, axes)):
        col_values = (
            sorted(df["n_col"].dropna().unique())
            if "n_col" in df.columns and df["n_col"].notna().any()
            else [None]
        )

        for n_col in col_values:
            if n_col is None:
                subset = df.copy()
                label = None
            else:
                subset = df[df["n_col"] == n_col]
                label = f"n_col={n_col}"

            ax.plot(subset["n_row"], subset[metric], marker="o", label=label)

        ax.set_ylabel(metric, fontsize=12, fontweight="bold")
        if titles:
            ax.set_title(titles[idx], fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if len(col_values) > 1 or col_values[0] is not None:
            ax.legend(title="n_col")

    axes[-1].set_xlabel("n_row", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
