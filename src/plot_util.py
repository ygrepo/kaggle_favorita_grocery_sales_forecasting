from src.utils import get_logger

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import sem

from matplotlib.ticker import MaxNLocator
from typing import (
    Sequence,
    Union,
    Mapping,
    Callable,
    Optional,
    Tuple,
    List,
    Any,
    Literal,
    Dict,
)
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA


Number = Union[int, float]

from src.utils import get_logger
from src.tensor_models import calculate_cluster_profiles

logger = get_logger(__name__)


# Colors for the two metrics
METRIC_COLORS = {
    "RMSSE": "#B82020",  # dark red
    "MASE": "#395B9B",  # navy blue
}


def plot_store_sku_heatmap(
    df: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = (8, 6),
    tick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 16,
    fn: Optional[Path] = None,
):
    # Create the heatmap
    plt.figure(figsize=figsize)

    # Create heatmap with custom styling
    sns.heatmap(
        df.values,
        annot=True,  # Show values in cells
        fmt=".3f",  # Format numbers to 4 decimal places
        cmap="RdBu_r",  # Color scheme (red-blue reversed)
        center=0,  # Center colormap at 0
        cbar_kws={"label": "Growth Rate"},
        linewidths=0.5,  # Add grid lines
        square=False,
    )  # Don't force square cells

    # Customize the plot
    plt.title(
        "Store-SKU Heatmap", fontsize=title_fontsize, fontweight="bold", pad=20
    )
    plt.xlabel("SKU", fontsize=label_fontsize, fontweight="bold")
    plt.ylabel("Store", fontsize=label_fontsize, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)

    plt.tight_layout()
    if fn:
        logger.info(f"Saving plot to {fn}")
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def multi_membership_order(
    M, method="average", metric="cosine", optimal_ordering=True
) -> np.ndarray:
    jitter = 1e-9 * np.random.RandomState(0).randn(*M.shape)
    Z = linkage(
        M + jitter,
        method=method,
        metric=metric,
        optimal_ordering=optimal_ordering,
    )
    return leaves_list(Z)


def plot_graded_overlapping_patches_df(
    df: pd.DataFrame,  # pandas DataFrame: rows = stores, cols = items
    U: np.ndarray,  # (n_rows, R)
    V: np.ndarray,  # (n_cols, C)
    *,
    center: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (8, 6),
    tick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 16,
    # ---- NEW annotation controls ----
    annotate: str = "pct",  # 'none' | 'abs' | 'pct' | 'zscore'
    median_mode: str = "global",  # 'global' | 'row' | 'col'
    min_abs: float = 0.005,  # only used when annotate='abs'
    min_pct: float = 1.0,  # % threshold when annotate='pct'
    min_z: float = 0.5,  # |z| threshold when annotate='zscore'
    fn: Optional[Path] = None,
):
    """
    Graded-overlap heatmap (no GridSpec) with optional per-cell annotation:
      abs:  value - median
      pct: (value - median) / median
      zscore: (value - median) / MAD*1.4826 (robust)
    """
    X = df.values
    row_names = df.index.to_list()
    col_names = df.columns.to_list()

    n_rows, n_cols = X.shape
    assert U.shape[0] == n_rows, "U rows must match X_df.index"
    assert V.shape[0] == n_cols, "V cols must match X_df.columns"

    # --- membership-similarity order ---
    row_order = multi_membership_order(U)
    col_order = multi_membership_order(V)
    logger.info(f"row_order: {row_order}")
    logger.info(f"col_order: {col_order}")

    # reorder
    Xr = X[np.ix_(row_order, col_order)]

    # sensible center for colormap
    if center is None:
        center = np.nanmedian(X) if np.nanmin(X) >= 0 else 0.0

    # ---------- build annotation matrix (strings) ----------
    labels = None
    if annotate.lower() != "none":
        # medians aligned with the reordered matrix
        if median_mode == "global":
            med = np.nanmedian(X)
            diff = Xr - med
            med_den = med
        elif median_mode == "row":
            med_row = np.nanmedian(X, axis=1, keepdims=True)
            med = med_row[row_order, :]
            diff = Xr - med
            med_den = med
        elif median_mode == "col":
            med_col = np.nanmedian(X, axis=0, keepdims=True)
            med = med_col[:, col_order]
            diff = Xr - med
            med_den = med
        else:
            raise ValueError("median_mode must be 'global', 'row', or 'col'")

        # format strings with thresholds to avoid clutter
        labels = np.empty_like(Xr, dtype=object)
        labels[:] = ""

        if annotate == "abs":
            mask = np.abs(diff) >= float(min_abs)
            labels[mask] = np.where(
                diff[mask] >= 0,
                np.char.add("+", np.char.mod("%.3f", diff[mask])),
                np.char.mod("%.3f", diff[mask]),
            )
        elif annotate == "pct":
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = 100.0 * diff / med_den
            mask = np.isfinite(pct) & (np.abs(pct) >= float(min_pct))
            labels[mask] = np.where(
                pct[mask] >= 0,
                np.char.add("+", np.char.mod("%.1f%%", pct[mask])),
                np.char.mod("%.1f%%", pct[mask]),
            )
        elif annotate == "zscore":
            # robust z using MAD per chosen median_mode
            if median_mode == "global":
                mad = np.nanmedian(np.abs(X - med))
                scale = (mad * 1.4826) or np.nan
                z = (Xr - med) / scale
            elif median_mode == "row":
                mad_row = np.nanmedian(
                    np.abs(X - med_row), axis=1, keepdims=True
                )
                scale = mad_row[row_order, :] * 1.4826
                z = (Xr - med) / np.where(scale == 0, np.nan, scale)
            else:  # 'col'
                mad_col = np.nanmedian(
                    np.abs(X - med_col), axis=0, keepdims=True
                )
                scale = mad_col[:, col_order] * 1.4826
                z = (Xr - med) / np.where(scale == 0, np.nan, scale)

            mask = np.isfinite(z) & (np.abs(z) >= float(min_z))
            labels[mask] = np.where(
                z[mask] >= 0,
                np.char.add("+", np.char.mod("%.2fσ", z[mask])),
                np.char.mod("%.2fσ", z[mask]),
            )
        else:
            raise ValueError(
                "annotate must be 'none', 'abs', 'pct', or 'zscore'"
            )

    # ---- draw ----
    fig, ax = plt.subplots(figsize=figsize)
    row_labels = [str(row_names[i]) for i in row_order]
    col_labels = [str(col_names[j]) for j in col_order]

    sns.heatmap(
        Xr,
        cmap=cmap,
        center=center,
        cbar=True,
        ax=ax,
        yticklabels=row_labels,
        xticklabels=col_labels,
        annot=labels,
        fmt="",
        # annot_kws={"fontsize": max(6, fontsize - 1)},
    )
    ax.tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
    ax.tick_params(axis="y", rotation=0, labelsize=tick_fontsize)
    ax.set_ylabel("Store", fontsize=label_fontsize, fontweight="bold")
    ax.set_xlabel("SKU", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(
        "Combined Clustering",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    if fn:
        logger.info(f"Saving plot to {fn}")
        plt.savefig(fn, dpi=300)
    plt.show()

    plt.close(fig)


def _order_by_memberships(
    M: np.ndarray, method="average", metric="cosine"
) -> np.ndarray:
    """Return a dendrogram leaf order based only on memberships M."""
    rng = np.random.RandomState(0)
    Z = linkage(
        M + 1e-9 * rng.randn(*M.shape),
        method=method,
        metric=metric,
        optimal_ordering=True,
    )
    return leaves_list(Z)


def _format_annot_matrix(
    Xr: np.ndarray, value_fmt: str | Callable[[float], str]
) -> np.ndarray:
    """Build string matrix for seaborn annot from Xr."""
    labels = np.empty_like(Xr, dtype=object)
    if callable(value_fmt):
        it = np.nditer(Xr, flags=["multi_index"])
        for x in it:
            i, j = it.multi_index
            v = x.item()
            labels[i, j] = (
                ""
                if (v is None or (isinstance(v, float) and np.isnan(v)))
                else value_fmt(v)
            )
    else:
        # string format (e.g., '.3f', '.2%')
        mask = ~np.isnan(Xr)
        labels[:] = ""
        labels[mask] = np.char.mod(f"%{value_fmt}", Xr[mask])
    return labels


def plot_by_store_memberships(
    df: pd.DataFrame,  # pandas DataFrame: rows=stores, cols=items
    U: np.ndarray,  # (n_rows, R)
    *,
    center: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (8, 6),
    tick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 16,
    sort_cols_by_mean_after: bool = False,
    show_values: bool = True,
    value_fmt: str | Callable[[float], str] = ".3f",
    x_rotation: int = 45,
    y_rotation: int = 45,
    fn: Optional[Path] = None,
):
    """Reorder rows using ONLY store memberships U; columns unchanged (or mean-sorted)."""
    X = df.values
    row_names = df.index.to_list()
    col_names = df.columns.to_list()
    n_rows, n_cols = X.shape
    assert U.shape[0] == n_rows, "U rows must match X_df.index"

    row_order = _order_by_memberships(U)
    Xr = X[row_order, :]

    if sort_cols_by_mean_after:
        col_order = np.argsort(-np.nanmean(Xr, axis=0))  # descending
    else:
        col_order = np.arange(n_cols)

    Xr = Xr[:, col_order]
    if center is None:
        center = np.nanmedian(X) if np.nanmin(X) >= 0 else 0.0

    fig, ax = plt.subplots(figsize=figsize)

    annot = None
    fmt = ""
    if show_values:
        annot = _format_annot_matrix(Xr, value_fmt)
        fmt = ""  # strings already formatted

    sns.heatmap(
        Xr,
        cmap=cmap,
        center=center,
        cbar=True,
        ax=ax,
        yticklabels=[str(row_names[i]) for i in row_order],
        xticklabels=[str(col_names[j]) for j in col_order],
        annot=annot,
        fmt=fmt,
    )

    # --- ticks: do it once, after heatmap ---
    ax.tick_params(axis="x", labelsize=tick_fontsize, labelrotation=x_rotation)
    ax.tick_params(axis="y", labelsize=tick_fontsize, labelrotation=y_rotation)

    # Optional: align tick labels nicely
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment("right")  # helps when y_rotation != 0

    ax.set_ylabel("store", fontsize=label_fontsize, fontweight="bold")
    ax.set_xlabel("SKU", fontsize=label_fontsize, fontweight="bold")

    ax.set_title(
        "Heatmap ordered by STORE memberships (rows only)",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    if fn:
        logger.info(f"Saving plot to {fn}")
        plt.savefig(fn, dpi=300, bbox_inches="tight", format="png")
    plt.show()
    plt.close(fig)


def plot_by_item_memberships(
    df: pd.DataFrame,  # pandas DataFrame: rows=stores, cols=items
    V: np.ndarray,  # (n_cols, C)
    *,
    center: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (8, 6),
    tick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 16,
    sort_rows_by_mean_after: bool = False,
    show_values: bool = True,
    value_fmt: str | Callable[[float], str] = ".3f",
    x_rotation: int = 45,
    y_rotation: int = 45,
    fn: Optional[Path] = None,
):
    """Reorder columns using ONLY item memberships V; rows unchanged (or mean-sorted)."""
    X = df.values
    row_names = df.index.to_list()
    col_names = df.columns.to_list()
    n_rows, n_cols = X.shape
    assert V.shape[0] == n_cols, "V rows must match X_df.columns"

    col_order = _order_by_memberships(V)
    Xr = X[:, col_order]

    if sort_rows_by_mean_after:
        row_order = np.argsort(-np.nanmean(Xr, axis=1))  # descending
    else:
        row_order = np.arange(n_rows)

    Xr = Xr[row_order, :]
    if center is None:
        center = np.nanmedian(X) if np.nanmin(X) >= 0 else 0.0

    fig, ax = plt.subplots(figsize=figsize)

    annot = None
    fmt = ""
    if show_values:
        annot = _format_annot_matrix(Xr, value_fmt)
        fmt = ""  # we already formatted strings

    sns.heatmap(
        Xr,
        cmap=cmap,
        center=center,
        cbar=True,
        ax=ax,
        yticklabels=[str(row_names[i]) for i in row_order],
        xticklabels=[str(col_names[j]) for j in col_order],
        annot=annot,
        fmt=fmt,
    )
    ax.tick_params(axis="x", labelsize=tick_fontsize, labelrotation=x_rotation)
    ax.tick_params(axis="y", labelsize=tick_fontsize, labelrotation=y_rotation)
    ax.set_ylabel("store", fontsize=label_fontsize, fontweight="bold")
    ax.set_xlabel("SKU", fontsize=label_fontsize, fontweight="bold")
    # Optional: align tick labels nicely
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment("right")  # helps when y_rotation != 0

    ax.set_title(
        "Heatmap ordered by SKU memberships (columns only)",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight", format="png")
    # plt.savefig(fn, dpi=300, bbox_inches="tight", format="png")
    plt.show()
    plt.close()


def plot_U_V(
    U: np.ndarray,  # (n_rows, R)
    V: np.ndarray,  # (n_cols, C)
    row_names: Sequence[str],  # len = n_rows
    col_names: Sequence[str],  # len = n_cols
    *,
    figsize: Tuple[int, int] = (12, 5),
    label_fontsize: int = 12,
    title_fontsize: int = 16,
    fn: Optional[Path] = None,
):
    row_names = np.asarray(row_names)
    col_names = np.asarray(col_names)

    # If the first column label is a header like "store", drop it to match item rows
    if (len(col_names) == V.shape[0] + 1) and (
        str(col_names[0]).lower() == "store"
    ):
        item_labels = col_names[1:]
    else:
        item_labels = col_names[: V.shape[0]]

    # Labels
    u_cluster_labels = [f"C{j+1}" for j in range(U.shape[1])]
    v_cluster_labels = [f"C{k+1}" for k in range(V.shape[1])]
    store_labels = [str(s) for s in row_names]
    item_labels = [str(x) for x in item_labels]

    # Black/white colormap; force 0 -> white, 1 -> black
    bw = ListedColormap(["#FFFFFF", "#000000"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- U (stores) ---
    sns.heatmap(
        U,
        annot=True,
        fmt=".0f",  # robust for int/float 0/1
        cmap=bw,
        vmin=0,
        vmax=1,
        cbar=False,  # remove colorbar
        xticklabels=u_cluster_labels,
        yticklabels=store_labels,
        ax=ax1,
        square=True,
        linewidths=0.2,
        linecolor="#CCCCCC",
    )
    ax1.set_title(
        "Store Cluster Memberships (U)",
        fontsize=title_fontsize,
        fontweight="bold",
    )
    ax1.set_xlabel(
        "Row (store) clusters", fontsize=label_fontsize, fontweight="bold"
    )
    ax1.set_ylabel("Store", fontsize=label_fontsize, fontweight="bold")

    # --- V (items/SKUs) ---
    sns.heatmap(
        V,
        annot=True,
        fmt=".0f",
        cmap=bw,
        vmin=0,
        vmax=1,
        cbar=False,  # remove colorbar
        xticklabels=v_cluster_labels,
        yticklabels=item_labels,
        ax=ax2,
        square=True,
        linewidths=0.2,
        linecolor="#CCCCCC",
    )
    ax2.set_title(
        "SKU Cluster Memberships (V)",
        fontsize=title_fontsize,
        fontweight="bold",
    )
    ax2.set_xlabel(
        "Column (SKU) clusters", fontsize=label_fontsize, fontweight="bold"
    )
    ax2.set_ylabel("SKU", fontsize=label_fontsize, fontweight="bold")

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


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
            ci_range = (
                t.ppf(0.975, df.groupby("epoch").count()[loss_col] - 1)
                * stderr
            )
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
            epochs,
            ci_tr_low,
            ci_tr_high,
            color="blue",
            alpha=0.15,
            label="95% CI",
        )

    if hist_df_previous is not None:
        prev_median_tr = hist_df_previous.groupby("epoch")[
            "train_loss"
        ].median()
        ax_tr.plot(
            epochs,
            prev_median_tr,
            color="gray",
            linestyle="--",
            label="Median Train (Previous)",
        )

    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)
    ax_tr.axvline(
        x=xvline, color="green", linestyle="--", label=f"Epoch {xvline}"
    )

    # ---- Validation Loss Plot ----
    ax_te.plot(epochs, med_te, color="orange", label="Median Val (Current)")
    ax_te.fill_between(
        epochs, q1_te, q3_te, color="orange", alpha=0.3, label="IQR (Current)"
    )
    if show_ci and ci_te_low is not None:
        ax_te.fill_between(
            epochs,
            ci_te_low,
            ci_te_high,
            color="orange",
            alpha=0.15,
            label="95% CI",
        )

    if hist_df_previous is not None:
        prev_median_te = hist_df_previous.groupby("epoch")[
            "test_loss"
        ].median()
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
            df_sid["epoch"],
            df_sid["train_loss"],
            marker="o",
            linewidth=1,
            label=sid,
        )
    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)

    # — Validation Loss —
    for sid in sids:
        df_sid = hist_df[hist_df["store_item"] == sid]
        ax_te.plot(
            df_sid["epoch"],
            df_sid["test_loss"],
            marker="o",
            linewidth=1,
            label=sid,
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
    ax_tr.fill_between(
        epochs, q1_train, q3_train, color="blue", alpha=0.6, label="IQR"
    )
    ax_tr.set_title("Train Loss by Epoch", fontsize=16, fontweight="bold")
    ax_tr.set_ylabel("Train Loss", fontsize=14)

    # --- Plot Validation Loss ---
    ax_te.plot(
        epochs,
        median_val,
        color="orange",
        label="Validation Loss per Epoch (All Store–Item Pairs)",
    )
    ax_te.fill_between(
        epochs, q1_val, q3_val, color="orange", alpha=0.6, label="IQR"
    )
    ax_te.set_title("Validation Loss by Epoch", fontsize=16, fontweight="bold")
    ax_te.set_xlabel("Epoch", fontsize=14)
    ax_te.set_ylabel("Validation Loss", fontsize=14)

    for ax in (ax_tr, ax_te):
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", which="major", length=6, labelsize=12)
        ax.tick_params(axis="y", which="major", length=6, labelsize=12)
        ax.legend(fontsize=12)
        ax.axvline(
            x=xvline, linestyle="--", color="green", label=f"Epoch {xvline}"
        )

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
    df: pd.DataFrame,
    sid: str,
    bins: int = 50,
    log_scale: bool = False,
    fn: str = None,
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
    sns.histplot(
        sales, bins=bins, kde=True, color="skyblue", edgecolor="black"
    )
    plt.title(
        f"Distribution of Unit Sales for {sid}", fontsize=24, fontweight="bold"
    )
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


def plot_final_percent_mav_per_sid(
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
    ax.plot(x, df["final_train_percent_mav"], marker="o", label="Train %MAV")
    ax.plot(
        x, df["final_test_percent_mav"], marker="o", label="Validation %MAV"
    )

    ax.set_xlabel("store_item", fontsize=16, fontweight="bold")
    ax.set_ylabel(
        "Mean Absolute Error as %MAV", fontsize=16, fontweight="bold"
    )
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


def visualize_clustered_matrix(
    X, U, V_list, title="Clustered Matrix", fn: str = None
):
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
    pivot_sil = results_df.pivot(
        index="P", columns="Q", values="Mean Silhouette"
    )
    pivot_loss = results_df.pivot(index="P", columns="Q", values="Mean Loss")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(pivot_sil, annot=True, fmt=".2f", ax=axes[0], cmap="viridis")
    axes[0].set_title("Mean Silhouette Score")
    axes[0].set_xlabel("Q (Column Clusters)")
    axes[0].set_ylabel("P (Row Clusters)")

    sns.heatmap(
        pivot_loss, annot=True, fmt=".2f", ax=axes[1], cmap="viridis_r"
    )
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
        plt.plot(
            subset["P"], subset["Mean Loss"], marker="o", label=f"Q = {q}"
        )

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

    ax.set_xlabel(
        "Number of Store_Item Clusters", fontsize=16, fontweight="bold"
    )
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
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.close(fig)


def plot_spectral_clustering_elbows(
    result_dfs: Sequence[pd.DataFrame],
    *,
    metric: str = "Explained Variance (%)",
    titles: Sequence[str] | None = None,
    figsize: tuple[int, int] = (10, 4),
    fn: str | None = None,
    vline_x: Number | None = None,
    hline_y: Number | None = None,
    annotate_intersection: bool = True,
    line_palette: Sequence[str] | None = None,
    vline_kwargs: Mapping | None = None,
    hline_kwargs: Mapping | None = None,
) -> None:
    """
    Plot elbow curves (variance‑explained vs. number of clusters) for one or
    several spectral‑clustering runs.

    Parameters
    ----------
    result_dfs :
        Iterable of DataFrames, each returned by
        ``compute_spectral_clustering_cv_scores``. Must contain 'n_row'
        (number of clusters) and the chosen *metric* column.  If the run
        explored the column dimension the frame should also contain 'n_col'.
    metric :
        Column to plot on the y‑axis.
    titles :
        Sub‑plot titles; length must match *result_dfs* if given.
    figsize :
        Matplotlib figure size in inches.
    fn :
        Optional path to save the figure (300 dpi PNG/TIFF etc.).
    vline_x, hline_y :
        X‑ and Y‑coordinates for decision lines (e.g. *k* = 5, 80 %).
    annotate_intersection :
        If *True*, annotate the point (vline_x, hline_y) when both lines are
        drawn and that point exists in the data.
    line_palette :
        Sequence of colours for different ``n_col`` curves.  Defaults to a
        colour‑blind‑safe palette of up to six hues.
    vline_kwargs, hline_kwargs :
        Extra keyword arguments forwarded to *ax.axvline* / *ax.axhline*.
    """
    # ───────────────────────────────────────── theme ───────────────────────
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
        },
    )

    if titles and len(titles) != len(result_dfs):
        raise ValueError("Length of *titles* must match *result_dfs*")

    if line_palette is None:
        line_palette = sns.color_palette("colorblind", 6)

    fig, axes = plt.subplots(
        1,
        len(result_dfs),
        figsize=figsize,
        sharey=True,
        squeeze=False,  # always returns 2‑D array
    )
    axes = axes.flatten()  # easier iteration

    for idx, (df, ax) in enumerate(zip(result_dfs, axes)):
        # ───── draw curves ────────────────────────────────────────────────
        col_values = (
            sorted(df["n_col"].dropna().unique())
            if "n_col" in df.columns and df["n_col"].notna().any()
            else [None]
        )
        colour_cycle = iter(line_palette)

        for n_col in col_values:
            if n_col is None:  # 1‑D spectral clustering run
                subset = df
                lbl, colour = None, next(colour_cycle)
            else:  # bi/co‑clustering: slice by n_col
                subset = df[df["n_col"] == n_col]
                lbl, colour = f"n_col = {n_col}", next(colour_cycle)

            ax.plot(
                subset["n_row"],
                subset[metric],
                marker="o",
                linewidth=2,
                color=colour,
                label=lbl,
            )

        # ───── decision lines ─────────────────────────────────────────────
        if vline_x is not None:
            ax.axvline(
                vline_x,
                **{
                    "color": "red",
                    "linestyle": "--",
                    "linewidth": 1.4,
                    **(vline_kwargs or {}),
                },
            )
        if hline_y is not None:
            ax.axhline(
                hline_y,
                **{
                    "color": "grey",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    **(hline_kwargs or {}),
                },
            )
        # annotate the intersection if desired and sensible
        if (
            annotate_intersection
            and (vline_x is not None)
            and (hline_y is not None)
            and df["n_row"].isin([vline_x]).any()
        ):
            ax.annotate(
                f"{hline_y:.0f} %",
                xy=(vline_x, hline_y),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

        # ───── axes cosmetics ─────────────────────────────────────────────
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Number of clusters (k)", labelpad=6, fontsize=13)
        if idx == 0:
            ax.set_ylabel(metric, labelpad=6, fontsize=13)
        else:
            ax.set_ylabel(None)

        if len(col_values) > 1 or col_values[0] is not None:
            ax.legend(frameon=False, fontsize=10, loc="best")

        if titles:
            ax.set_title(titles[idx], fontsize=14, fontweight="bold", pad=8)

    sns.despine(fig=fig)
    fig.tight_layout()

    if fn:
        fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_heatmap_with_cluster_boundaries(
    title: str,
    matrix_ordered,
    row_labels,
    col_labels,
    label_in_cell=False,
    fn: str | None = None,
):

    fig, ax = plt.subplots(figsize=(6, 6))

    row_labels = np.asarray(row_labels)
    col_labels = np.asarray(col_labels)

    # Construct (row_id, col_id) cluster identifier matrix
    cluster_ids = np.array(
        [
            [(row_labels[i], col_labels[j]) for j in range(len(col_labels))]
            for i in range(len(row_labels))
        ]
    )

    if label_in_cell:
        annot = np.array(
            [[f"R{r}-C{c}" for r, c in row] for row in cluster_ids]
        )
        fmt = ""
    else:
        annot = matrix_ordered
        fmt = ".1f"

    sns.heatmap(
        matrix_ordered,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "unit sales"},
        annot=annot,
        fmt=fmt,
        annot_kws={"fontsize": 7, "color": "white", "fontweight": "bold"},
        ax=ax,
    )

    # Find boundaries between bicluster blocks
    for i in range(1, matrix_ordered.shape[0]):
        if not np.all(cluster_ids[i, :] == cluster_ids[i - 1, :]):
            ax.axhline(i, color="red", linewidth=1.5)

    for j in range(1, matrix_ordered.shape[1]):
        if not np.all(cluster_ids[:, j] == cluster_ids[:, j - 1]):
            ax.axvline(j, color="red", linewidth=1.5)

    ax.set_xlabel("Store", fontsize=13, fontweight="bold")
    ax.set_ylabel("Item", fontsize=13, fontweight="bold")
    plt.title(title)
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_with_cluster_boundaries_from_model(
    data,
    model,
    title="Reordered Bicluster Matrix",
    max_ticks=20,
    fn: str = None,
):
    """
    Plot a reordered data matrix with red lines indicating bicluster boundaries.
    Shows original row and column indices on the reordered matrix.

    Parameters:
    - data: original 2D numpy array
    - model: must have .row_labels_ and .column_labels_ attributes
    - title: plot title
    - max_ticks: maximum number of ticks to show on each axis
    """
    if hasattr(model, "row_labels_") and hasattr(model, "column_labels_"):
        row_labels = model.row_labels_
        col_labels = model.column_labels_
    else:
        raise ValueError(
            "Model must have row_labels_ and column_labels_ attributes."
        )

    # Reorder data
    reordered_data, row_order, col_order = reorder_data(
        data, row_labels, col_labels
    )

    # Boundary cuts
    row_cuts = np.where(np.diff(row_labels[row_order]) != 0)[0] + 1
    col_cuts = np.where(np.diff(col_labels[col_order]) != 0)[0] + 1

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(reordered_data, aspect="auto", cmap=plt.cm.Blues)
    ax.set_title(title)

    for r in row_cuts:
        ax.axhline(r - 0.5, color="red", linewidth=1)
    for c in col_cuts:
        ax.axvline(c - 0.5, color="red", linewidth=1)

    # Annotate ticks with original indices
    row_ticks = np.linspace(
        0, len(row_order) - 1, min(len(row_order), max_ticks), dtype=int
    )
    col_ticks = np.linspace(
        0, len(col_order) - 1, min(len(col_order), max_ticks), dtype=int
    )

    ax.set_yticks(row_ticks)
    ax.set_yticklabels(row_order[row_ticks], fontsize=8)

    ax.set_xticks(col_ticks)
    ax.set_xticklabels(col_order[col_ticks], fontsize=8, rotation=90)

    ax.set_xlabel("Original Column Index")
    ax.set_ylabel("Original Row Index")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close(fig)


def plot_bicluster_grid(
    matrix,
    row_masks,
    col_masks,
    figsize=(20, 15),
    show_labels=False,
    single_cbar=True,
    cmap="viridis",
    fn: str = None,
):
    """
    Display biclusters from (row_masks, col_masks) as subplots.

    Parameters
    ----------
    matrix : pd.DataFrame
        Original data matrix (do not reorder globally).
    row_masks : np.ndarray
        Boolean array (n_biclusters, n_rows).
    col_masks : np.ndarray
        Boolean array (n_biclusters, n_cols).
    figsize : tuple
        Size of the full figure.
    show_labels : bool
        Whether to display row/column tick labels.
    single_cbar : bool
        Whether to show a single shared colorbar.
    cmap : str
        Colormap for the heatmaps.
    """

    n_biclusters = len(row_masks)
    n_cols_grid = int(np.ceil(np.sqrt(n_biclusters)))
    n_rows_grid = int(np.ceil(n_biclusters / n_cols_grid))

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=figsize)
    axes = np.array(axes).flatten()

    # Track heatmaps for shared colorbar
    vmin = np.min(matrix.values)
    vmax = np.max(matrix.values)
    heatmaps = []

    for k, (row_mask, col_mask) in enumerate(zip(row_masks, col_masks)):
        ax = axes[k]
        submatrix = matrix.loc[row_mask, col_mask]

        hm = sns.heatmap(
            submatrix,
            ax=ax,
            cmap=cmap,
            cbar=False if single_cbar else True,
            xticklabels=show_labels,
            yticklabels=show_labels,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Bicluster {k}", fontsize=10)
        heatmaps.append(hm)

    # Hide unused axes
    for ax in axes[n_biclusters:]:
        ax.axis("off")

    # Add single shared colorbar
    if single_cbar:
        cbar_ax = fig.add_axes(
            [0.92, 0.3, 0.015, 0.4]
        )  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([vmin, vmax])
        fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_bicluster_heatmaps_grid(
    data,
    results_df,
    *,
    heatmaps_per_row: int = 3,
    figsize_per_heatmap: tuple = (5, 4),
    cmap: str = "viridis",
    show_tick_labels: bool = True,
    fn: str | None = None,
):
    """
    Plots bicluster heatmaps in a grid layout using subplots.
    Shows x-axis as stores and y-axis as items, with cluster boundaries.
    """
    X = np.asarray(data)
    store_ids = np.array(data.index)
    item_ids = np.array(data.columns)

    valid_results = [
        row
        for _, row in results_df.iterrows()
        if row["row_labels"] is not None and row["col_labels"] is not None
    ]

    n_heatmaps = len(valid_results)
    if n_heatmaps == 0:
        print("No valid bicluster results to plot.")
        return

    n_rows = math.ceil(n_heatmaps / heatmaps_per_row)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=heatmaps_per_row,
        figsize=(
            figsize_per_heatmap[0] * heatmaps_per_row,
            figsize_per_heatmap[1] * n_rows,
        ),
        squeeze=False,
    )

    for idx, row in enumerate(valid_results):
        i, j = divmod(idx, heatmaps_per_row)
        ax = axes[i][j]

        row_labels = np.asarray(row["row_labels"])
        col_labels = np.asarray(row["col_labels"])
        n_row = row["n_row"]
        n_col = row["n_col"]

        row_order = np.argsort(row_labels)
        col_order = (
            np.argsort(col_labels)
            if np.ndim(col_labels) == 1
            else np.arange(X.shape[1])
        )

        reordered = X[row_order, :][:, col_order]

        sns.heatmap(reordered, cmap=cmap, cbar=False, ax=ax)

        # Draw horizontal boundaries between row clusters
        row_boundaries = np.flatnonzero(np.diff(row_labels[row_order])) + 1
        for y in row_boundaries:
            ax.axhline(y, color="white", linewidth=1.5)

        # Draw vertical boundaries between column clusters
        if np.ndim(col_labels) == 1:
            col_boundaries = np.flatnonzero(np.diff(col_labels[col_order])) + 1
            for x in col_boundaries:
                ax.axvline(x, color="white", linewidth=1.5)

        # Tick labels
        reordered_store_ids = store_ids[row_order]
        reordered_item_ids = item_ids[col_order]

        # Label cluster squares with (row_label, col_label)
        unique_row_labels = np.unique(row_labels)
        unique_col_labels = np.unique(col_labels)

        for rl in unique_row_labels:
            row_mask = row_labels[row_order] == rl
            row_start = np.where(row_mask)[0][0]
            row_end = np.where(row_mask)[0][-1]

            for cl in unique_col_labels:
                col_mask = col_labels[col_order] == cl
                if not np.any(col_mask):
                    continue  # skip if this column cluster is not present
                col_start = np.where(col_mask)[0][0]
                col_end = np.where(col_mask)[0][-1]

                # Coordinates for center of the block
                y_center = (row_start + row_end) / 2
                x_center = (col_start + col_end) / 2

                ax.text(
                    x_center,
                    y_center,
                    f"({rl},{cl})",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"
                    ),
                )

        if show_tick_labels:
            xticks = np.linspace(
                0,
                len(reordered_item_ids) - 1,
                min(10, len(reordered_item_ids)),
                dtype=int,
            )
            yticks = np.linspace(
                0,
                len(reordered_store_ids) - 1,
                min(10, len(reordered_store_ids)),
                dtype=int,
            )
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(reordered_item_ids[xticks], rotation=90)
            ax.set_yticklabels(reordered_store_ids[yticks])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(f"n_row={n_row}, n_col={n_col}", fontsize=10)
        ax.set_xlabel("Items")
        ax.set_ylabel("Stores")

    # Remove empty axes
    for k in range(n_heatmaps, n_rows * heatmaps_per_row):
        i, j = divmod(k, heatmaps_per_row)
        fig.delaxes(axes[i][j])

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_biclustering_elbows(
    df: pd.DataFrame,
    *,
    title: str = "Biclustering Elbow Plot",
    title_fontsize: int = 20,
    metric: str = "Explained Variance (%)",  # or "Mean Loss", "Mean Silhouette"
    tick_step: int = 1,
    vline_index: int | None = None,
    vline_text: str | None = None,
    vline_kwargs: dict | None = None,
    figsize: tuple[int, int] = (10, 5),
    fn: str | None = None,  # optional PNG filename
):
    """
    Plot an elbow curve for biclustering results based on a composite n_row x n_col identifier.

    Parameters
    ----------
    df : DataFrame
        Output from compute_biclustering_scores with 'n_row', 'n_col', and metric columns.
    metric : str
        Column name in `df` to plot on the y-axis.
    vline_index : int or None
        Optional index in the sorted DataFrame where to draw a vertical line.
    fn : str or None
        If provided, the plot is saved to this filename (dpi=300).
    """
    # ───────────────────── Sorting and X-axis Labels ─────────────────────
    df_sorted = (
        df.copy().sort_values(by=metric, ascending=True).reset_index(drop=True)
    )
    df_sorted["label"] = df_sorted.apply(
        lambda row: (
            f"{row['n_row']}x{int(row['n_col'])}"
            if not pd.isna(row["n_col"])
            else f"{row['n_row']}"
        ),
        axis=1,
    )

    # ───────────────────── Plot ─────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        df_sorted["label"],
        df_sorted[metric],
        marker="o",
        label=metric,
    )

    ax.set_xlabel("n_row x n_col", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Auto tick step
    n_points = len(df_sorted)
    tick_step = max(1, n_points // 20)

    ax.set_xticks(df_sorted.index[::tick_step])
    ax.set_xticklabels(
        df_sorted["label"][::tick_step], rotation=60, ha="right"
    )

    fig.subplots_adjust(bottom=0.25)

    # ───────────────────── Optional vertical line ─────────────────────
    if vline_index is not None and 0 <= vline_index < len(df_sorted):
        default_line_style = dict(color="red", linestyle="--", linewidth=1.5)
        if vline_kwargs:
            default_line_style.update(vline_kwargs)

        ax.axvline(vline_index, **default_line_style)
        ax.text(
            vline_index,
            ax.get_ylim()[0],
            vline_text,
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

    return df_sorted


def plot_block_annot_heatmap(
    df: pd.DataFrame,
    *,
    ttl: str | None = None,
    value_col: str = "growth_rate_1",
    block_col: str = "block_id",
    row_col: str = "store",
    col_col: str = "item",
    date_col: str = "start_date",
    date: str | None = None,
    x_label: str = "SKU",
    y_label: str = "Store",
    fmt: str = "{:.2f}",
    cell_h: float = 0.9,
    cell_w: float = 0.6,
    font_size: int = 11,
    row_order: list | None = None,
    col_order: list | None = None,
    on_missing_date: str = "latest",
    fn: Path | None = None,
    # NEW args
    figsize: tuple[float, float] | None = None,
    xlabel_size: int | None = None,
    ylabel_size: int | None = None,
    label_weight: str = "bold",
    xtick_size: int = 9,
    ytick_size: int = 9,
    xtick_rotation: float = 90,
    ytick_rotation: float = 0,
    show_plot: bool = True,
):
    if xlabel_size is None:
        xlabel_size = font_size
    if ylabel_size is None:
        ylabel_size = font_size

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        if date is None:
            date = df[date_col].max()
        else:
            date = pd.to_datetime(date)
        dfx = df[df[date_col] == date]
        if dfx.empty:
            if on_missing_date == "latest":
                date = df[date_col].max()
                dfx = df[df[date_col] == date]
                print(
                    f"[info] requested date not found; using latest available: {date.date()}"
                )
            else:
                raise ValueError(f"No rows for date {date.date()}")
        df = dfx

    key = [row_col, col_col]
    dup_mask = df.duplicated(subset=key, keep=False)
    if dup_mask.any():
        g1 = df.groupby(key)[value_col].nunique()
        g2 = df.groupby(key)[block_col].nunique()
        bad = g1[(g1 > 1) | (g2 > 1)]
        if len(bad) > 0:
            raise ValueError(
                "Duplicates with different values/blocks; refuse to aggregate."
            )
        df = df.drop_duplicates(subset=key)

    val = df.pivot(index=row_col, columns=col_col, values=value_col)
    blk = df.pivot(index=row_col, columns=col_col, values=block_col)

    if row_order is not None:
        row_order = [r for r in row_order if r in blk.index]
        blk = blk.reindex(index=row_order)
        val = val.reindex(index=row_order)
    else:
        blk = blk.sort_index(axis=0)
        val = val.reindex(index=blk.index)

    if col_order is not None:
        col_order = [c for c in col_order if c in blk.columns]
        blk = blk.reindex(columns=col_order)
        val = val.reindex(columns=col_order)
    else:
        blk = blk.sort_index(axis=1)
        val = val.reindex(columns=blk.columns)

    uniq = pd.Series(blk.to_numpy().ravel()).dropna()
    uniq = np.sort(uniq.astype(int).unique())
    n = len(uniq)

    if n == 0:
        logger.warning(
            "[warn] no block ids present on this slice; showing a blank grid."
        )
        H = max(4, cell_h * blk.shape[0])
        W = max(6, cell_w * blk.shape[1])
        if figsize is not None:
            W, H = figsize
        _, ax = plt.subplots(figsize=(W, H))
        ax.set_axis_off()
        return

    id2idx = {bid: i for i, bid in enumerate(uniq)}
    mapped = (
        pd.Series(blk.to_numpy().ravel())
        .map(lambda x: id2idx.get(int(x)) if pd.notna(x) else np.nan)
        .to_numpy()
        .reshape(blk.shape)
    )
    blk_idx = pd.DataFrame(mapped, index=blk.index, columns=blk.columns)
    mask = blk_idx.isna().to_numpy()

    colors = (
        sns.color_palette("tab20", n)
        if n <= 20
        else sns.color_palette("husl", n)
    )
    cmap = ListedColormap(colors)
    cmap.set_bad("#f0f0f0")
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), n)

    A = val.to_numpy(dtype=float)

    # Sanitize data to prevent TIFF encoding issues
    # Replace inf/-inf with NaN, and clip extreme values
    A = np.where(np.isinf(A), np.nan, A)
    finite_mask = np.isfinite(A)
    if finite_mask.any():
        # Clip extreme values to prevent encoding issues
        finite_values = A[finite_mask]
        p1, p99 = np.percentile(finite_values, [1, 99])
        # Allow reasonable range but prevent extreme outliers
        max_safe_value = max(abs(p1), abs(p99)) * 10
        A = np.clip(A, -max_safe_value, max_safe_value)

    annot = np.empty_like(A, dtype=object)
    annot[:] = ""
    m = ~np.isnan(A)
    if m.any():
        try:
            annot[m] = np.vectorize(lambda x: fmt.format(x))(A[m])
        except (ValueError, OverflowError):
            # Fallback for extreme values that can't be formatted
            annot[m] = np.vectorize(
                lambda x: f"{x:.2e}" if abs(x) > 1e6 else f"{x:.2f}"
            )(A[m])

    # Figure size: from args if given, else from cell_h/w and grid size
    H = max(4, cell_h * blk.shape[0])
    W = max(6, cell_w * blk.shape[1])
    if figsize is not None:
        W, H = figsize

    _, ax = plt.subplots(figsize=(W, H))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)

    sns.heatmap(
        blk_idx,
        cmap=cmap,
        norm=norm,
        mask=mask,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"shrink": 1.0, "aspect": 25},
        square=True,
        linewidths=1.0,
        linecolor="white",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": font_size, "fontweight": "bold"},
        ax=ax,
    )

    # Axis labels: bold and sized
    ax.set_xlabel(x_label, fontsize=xlabel_size, fontweight=label_weight)
    ax.set_ylabel(y_label, fontsize=ylabel_size, fontweight=label_weight)

    ttl = ttl or f"Blocks colored by {block_col}"
    if date_col and date is not None:
        ttl += f" — {pd.to_datetime(date).date()}"
    ax.set_title(ttl)

    # Tick labels (row/col names) with controllable rotation + size
    ax.set_xticklabels(
        blk_idx.columns.astype(str),
        rotation=xtick_rotation,
        ha="center",
        fontsize=xtick_size,
        fontweight=label_weight,
    )
    ax.set_yticklabels(
        blk_idx.index.astype(str),
        rotation=ytick_rotation,
        va="center",
        fontsize=ytick_size,
        fontweight=label_weight,
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(n))
    cbar.set_ticklabels(uniq)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Cluster ID", fontsize=9)

    plt.tight_layout()
    if fn:
        logger.info(f"Saving plot to {fn}")
        # Convert TIFF to PNG if needed to avoid encoding issues
        fn_str = str(fn)
        if fn_str.lower().endswith(".tiff") or fn_str.lower().endswith(".tif"):
            fn_str = fn_str.rsplit(".", 1)[0] + ".png"
            logger.info(f"Converting TIFF to PNG format: {fn_str}")

        try:
            plt.savefig(fn_str, dpi=300, bbox_inches="tight", format="png")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            # Try with lower DPI as fallback
            try:
                plt.savefig(fn_str, dpi=150, bbox_inches="tight", format="png")
                logger.info(
                    f"Saved with reduced DPI (150) due to encoding issues"
                )
            except Exception as e2:
                logger.error(f"Failed to save even with reduced DPI: {e2}")
                raise
    if show_plot:
        plt.show()


def plot_pca_explained_variance(
    X_3d: np.ndarray,
    n_components: int = None,
    label_fontsize: int = 14,
    title_fontsize: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    fn: Optional[Path] = None,
):
    """
    Performs PCA on the mean-aggregated 2D data and plots the
    cumulative explained variance ratio.

    Args:
        X_3d (I, J, D): Input data tensor (Stores x Items x Features).
        n_components: Number of components for PCA. If None, uses min(I, J).
    """

    # 1. Prepare 2D Data (Average across features)
    if X_3d.ndim == 3:
        X_2d = np.mean(X_3d, axis=2)  # Shape (I, J) - Stores x Items
    else:
        # Assuming input might already be 2D
        X_2d = X_3d

    I, J = X_2d.shape

    # Ensure no NaNs (PCA requires finite values)
    # Use mean imputation as a simple strategy
    if np.isnan(X_2d).any():
        col_means = np.nanmean(X_2d, axis=0)
        inds = np.where(np.isnan(X_2d))
        X_2d[inds] = np.take(col_means, inds[1])
        # Handle cases where a whole column might be NaN
        if np.isnan(X_2d).any():
            X_2d = np.nan_to_num(
                X_2d
            )  # Fallback: replace remaining NaNs with 0

    # 2. Perform PCA
    # Decide on the number of components
    if n_components is None:
        # Default to the smaller dimension
        n_components = min(I, J)

    # Apply PCA - typically center data first (implicitly done by sklearn's PCA)
    # We are analyzing variance across items (columns), so PCA fits on rows (stores)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_2d)  # Fit PCA on Stores x Items matrix

    # 3. Calculate Cumulative Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # 4. Plotting
    plt.figure(figsize=figsize)
    plt.plot(
        range(1, len(cumulative_explained_variance) + 1),
        cumulative_explained_variance,
        marker="o",
        linestyle="--",
    )

    # Find points for 90%, 95% variance
    try:
        n_90 = np.argmax(cumulative_explained_variance >= 0.90) + 1
        plt.axhline(
            0.90,
            color="r",
            linestyle=":",
            label=f"90% Variance ({n_90} components)",
        )
        plt.axvline(n_90, color="r", linestyle=":")
    except ValueError:
        pass  # Handle case where 90% is never reached

    try:
        n_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
        plt.axhline(
            0.95,
            color="g",
            linestyle=":",
            label=f"95% Variance ({n_95} components)",
        )
        plt.axvline(n_95, color="g", linestyle=":")
    except ValueError:
        pass  # Handle case where 95% is never reached

    plt.title(
        "Cumulative Explained Variance by PCA Components",
        fontsize=title_fontsize,
        fontweight="bold",
    )
    plt.xlabel(
        "Number of Principal Components",
        fontsize=label_fontsize,
        fontweight="bold",
    )
    plt.ylabel(
        "Cumulative Explained Variance Ratio",
        fontsize=label_fontsize,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1.05)
    plt.legend(loc="best")
    plt.tight_layout()
    # Save the plot
    if fn:
        logger.info(f"Saving plot to {fn}")
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close()


def plot_reconstruction_metrics(
    metrics: dict[str, Any],
    method: str,
    *,
    features: List[str] = None,
    show_pve_line: bool = True,
    use_rmse: bool = True,
    fn: Path | None = None,
    figsize=(12, 6),
):
    """
    Plots pre-computed reconstruction quality metrics.
    Includes 1x2 grid: Scatter, Per-Feature RMSE/PVE.
    """
    logger.info(f"Plotting reconstruction metrics for method '{method}'...")

    # Unpack Metrics ---
    x_plot = metrics["x_plot"]
    xr_plot = metrics["xr_plot"]
    n_obs_per = metrics["n_obs_per"]
    rss_per = metrics["rss_per"]
    rmse_per = metrics["rmse_per"]
    pve_per = metrics["pve_per"]
    D = metrics["D"]
    rank_str = metrics["rank_str"]

    names = (
        list(features)
        if features is not None
        else [f"feat_{d}" for d in range(D)]
    )

    # Figure layout
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Reconstruction Quality: {method.upper()} @ rank={rank_str}",
        fontsize=16,
    )

    # Plot Scatter
    ax = axes[0]
    if x_plot.size > 0:
        ax.scatter(x_plot, xr_plot, alpha=0.35, s=2)
        lo, hi = float(np.nanmin(x_plot)), float(np.nanmax(x_plot))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Original values", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reconstructed values", fontsize=14, fontweight="bold")
    ax.set_title(
        "Reconstruction",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2)

    # Plot Per-feature bars
    ax = axes[1]
    xloc = np.arange(D)
    vals = rmse_per if use_rmse else rss_per
    label = "RMSE" if use_rmse else "RSS"
    heights = np.array(
        [v if np.isfinite(v) else 0.0 for v in vals], dtype=float
    )
    bars = ax.bar(xloc, heights)
    y_max_val = np.nanmax(heights)
    y_max = y_max_val if (np.isfinite(y_max_val) and y_max_val > 0) else 1.0

    for d, b in enumerate(bars):
        if not np.isfinite(vals[d]) or n_obs_per[d] == 0:
            b.set_height(0.02 * y_max)
            b.set_color("lightgray")
            b.set_hatch("//")

    ax.set_xticks(xloc)
    ax.set_xticklabels(
        names, rotation=45, ha="right", fontsize=12, fontweight="heavy"
    )
    ax.set_ylabel(label, fontsize=14, fontweight="bold")
    ax.set_title(f"Per-feature {label}", fontsize=14, fontweight="bold")

    if show_pve_line:
        ax2 = ax.twinx()
        for d in range(D):
            # This print statement was in your original code
            print(f"pve_per[{d}]={pve_per[d]}")
            if np.isfinite(pve_per[d]):
                ax2.plot(
                    xloc[d],
                    pve_per[d],
                    marker="o",
                    markersize=12,
                    color="orange",
                )
            else:
                ax2.plot(xloc[d], 5, marker="o", markersize=6, color="red")

        ax2.set_ylabel("PVE (%)")
        ax2.set_ylim(0, 100)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=6,
                label="Valid PVE",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=6,
                label="Invalid / No PVE",
            ),
        ]
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Finalize Plot
    plt.tight_layout()
    if fn:
        print(f"Saving plot to {fn}")
        plt.savefig(fn, dpi=300)
    plt.show()
    plt.close(fig)


def plot_factor_matrix(
    factor_matrix,
    title="Factor Matrix Loadings",
    x_labels=None,
    y_labels=None,
    max_factors=10,  # New parameter to limit factors
    top_factors_only=True,  # Show only most important factors
):
    """
    Visualizes a factor matrix using a heatmap with improved readability.

    Args:
        factor_matrix (np.ndarray or torch.Tensor): The 2D factor matrix (e.g., F_features).
        title (str): The title for the plot.
        x_labels (list[str], optional): Labels for the columns.
        y_labels (list[str], optional): Labels for the rows.
        max_factors (int): Maximum number of factors to display.
        top_factors_only (bool): If True, select factors with highest variance.
    """
    # Convert to numpy if it's a tensor
    if hasattr(factor_matrix, "detach"):  # PyTorch tensor
        factor_matrix = factor_matrix.detach().cpu().numpy()
    elif hasattr(factor_matrix, "numpy"):  # Other tensor types
        factor_matrix = factor_matrix.numpy()

    if factor_matrix.ndim != 2:
        raise ValueError(
            f"Factor matrix must be 2D, but got {factor_matrix.ndim} dimensions."
        )

    # Select subset of factors if there are too many
    if factor_matrix.shape[1] > max_factors:
        if top_factors_only:
            # Select factors with highest variance (most informative)
            factor_variances = np.var(factor_matrix, axis=0)
            top_indices = np.argsort(factor_variances)[-max_factors:]
            factor_matrix = factor_matrix[:, top_indices]
            print(
                f"Showing top {max_factors} factors (by variance) out of {len(top_indices)} selected"
            )
        else:
            # Just take first max_factors
            original_shape = factor_matrix.shape[1]
            factor_matrix = factor_matrix[:, :max_factors]
            top_indices = range(max_factors)
            print(
                f"Showing first {max_factors} factors out of {original_shape} total"
            )
    else:
        top_indices = range(factor_matrix.shape[1])

    # Create labels
    if x_labels is None:
        x_labels = [f"Factor {i+1}" for i in top_indices]
    else:
        x_labels = [
            x_labels[i] if i < len(x_labels) else f"Factor {i+1}"
            for i in top_indices
        ]

    if y_labels is None:
        y_labels = [f"Row {i+1}" for i in range(factor_matrix.shape[0])]

    # Dynamic figure sizing with better proportions
    n_factors = factor_matrix.shape[1]
    n_features = factor_matrix.shape[0]

    fig_width = min(max(6, n_factors * 1.2), 20)  # Cap at 20 inches
    fig_height = min(max(4, n_features * 0.6), 12)  # Cap at 12 inches

    plt.figure(figsize=(fig_width, fig_height))

    # Adjust annotation based on matrix size
    show_annot = (
        n_factors <= 15 and n_features <= 20
    )  # Only annotate if not too crowded
    font_size = max(
        8, min(12, 120 // (n_factors + n_features))
    )  # Dynamic font size

    ax = sns.heatmap(
        factor_matrix,
        annot=show_annot,
        fmt=".2f" if show_annot else None,
        annot_kws=(
            {"fontsize": font_size, "fontweight": "bold"}
            if show_annot
            else None
        ),
        cmap="RdBu_r",  # Better color map for readability
        xticklabels=x_labels,
        yticklabels=y_labels,
        center=0,
        cbar_kws={"label": "Loadings", "shrink": 0.8},
        linewidths=(
            0.5 if n_factors <= 20 else 0
        ),  # Add gridlines for smaller matrices
    )

    # Improved formatting
    ax.set_title(title, fontsize=16, pad=20, fontweight="bold")
    plt.xlabel("Factor Components", fontsize=12, fontweight="bold")
    plt.ylabel("Features", fontsize=12, fontweight="bold")

    # Rotate x-labels if there are many factors
    rotation = 45 if n_factors > 8 else 0
    ax.set_xticklabels(
        ax.get_xticklabels(), fontweight="bold", rotation=rotation
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", rotation=0)

    # Add summary statistics as text if annotations are hidden
    if not show_annot:
        plt.figtext(
            0.02,
            0.02,
            f"Matrix: {factor_matrix.shape[0]}×{factor_matrix.shape[1]} | "
            f"Range: [{factor_matrix.min():.2f}, {factor_matrix.max():.2f}]",
            fontsize=10,
            style="italic",
        )


def plot_feature_factor_interpretation(
    model_dict, fn=None, max_factors=10, top_factors_only=True
):
    """
    Visualizes feature factor matrix with improved readability.

    Args:
        model_dict (dict): The loaded model dictionary.
        fn (str, optional): Filename to save the plot.
        max_factors (int): Maximum number of factors to display.
        top_factors_only (bool): Whether to select most important factors.
    """
    try:
        f_features = model_dict["factors"][2]
        feature_names = model_dict["feature_names"]

        # Handle tensor conversion early
        if hasattr(f_features, "detach"):  # PyTorch tensor
            total_factors = f_features.shape[1]
        else:  # NumPy array
            total_factors = f_features.shape[1]

    except KeyError:
        print("ERROR: 'factors' or 'feature_names' not found in model_dict.")
        return
    except IndexError:
        print("ERROR: Feature factor matrix not found at index 2.")
        return

    print(f"Original matrix shape: {f_features.shape}")
    print(
        f"Plotting Feature Factor Loadings ({f_features.shape[0]} features × {total_factors} factors)..."
    )

    plot_factor_matrix(
        f_features,
        title=f"Feature Factor Loadings (Top {min(max_factors, total_factors)} factors)",
        y_labels=feature_names,
        max_factors=max_factors,
        top_factors_only=top_factors_only,
    )

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# Create separate plots for factor subsets
def plot_feature_factors_multiplot(
    model_dict, fn_prefix=None, factors_per_plot=8
):
    """
    Create multiple smaller plots when you have many factors.
    """
    try:
        f_features = model_dict["factors"][2]
        feature_names = model_dict["feature_names"]
        total_factors = f_features.shape[1]
    except (KeyError, IndexError):
        print("ERROR: Could not extract factor matrix.")
        return

    n_plots = (total_factors + factors_per_plot - 1) // factors_per_plot

    for i in range(n_plots):
        start_idx = i * factors_per_plot
        end_idx = min((i + 1) * factors_per_plot, total_factors)

        factor_subset = f_features[:, start_idx:end_idx]
        factor_labels = [f"Factor {j+1}" for j in range(start_idx, end_idx)]

        plot_factor_matrix(
            factor_subset,
            title=f"Feature Factors {start_idx+1}-{end_idx}",
            x_labels=factor_labels,
            y_labels=feature_names,
            max_factors=factors_per_plot + 1,  # No additional limiting
        )

        plt.tight_layout()
        if fn_prefix:
            plt.savefig(
                f"{fn_prefix}_part_{i+1}.png", dpi=300, bbox_inches="tight"
            )
        plt.show()
        plt.close()


def plot_cluster_combined_profiles(
    model_dict: Dict[str, Any],
    mask: str,
    plot_types: List[Literal["heatmap", "parallel", "bar", "radar"]] = [
        "heatmap",
        "parallel",
    ],
    max_clusters: int = 5,
    selection_method: Literal["size", "variance", "hierarchical"] = "size",
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
):
    """
    Plot cluster profiles using two visualization methods side by side horizontally.

    Args:
        model_dict: The loaded model dictionary
        mask: 'Store' or 'SKU'
        plot_types: List of exactly 2 plot types to show side by side
        max_clusters: Maximum number of clusters to show
        selection_method: How to select top clusters
        figsize: Figure size for the combined plot
        save_path: Path to save the plot
    """
    # Validate plot_types
    if len(plot_types) != 2:
        logger.error("plot_types must contain exactly 2 plot types")
        return

    valid_types = ["heatmap", "parallel", "bar", "radar"]
    if not all(pt in valid_types for pt in plot_types):
        logger.error(f"Invalid plot types. Must be from {valid_types}")
        return

    # Get cluster data
    cluster_means = calculate_cluster_profiles(
        model_dict, mask, max_clusters, selection_method
    )
    if cluster_means is None:
        return

    feature_names = model_dict["feature_names"]
    feature_data = cluster_means[feature_names]

    # Normalize for better comparison
    feature_data_norm = (
        feature_data - feature_data.mean()
    ) / feature_data.std()
    value_label = "Normalized Values"

    # Get total clusters for title
    total_clusters = len(
        model_dict["assignments"]
        .query("factor_name == @mask")["cluster_id"]
        .unique()
    )
    title_suffix = f" (Top {len(feature_data_norm)}/{total_clusters} clusters by {selection_method})"

    # Handle radar plot special case (needs polar projection)
    if "radar" in plot_types:
        fig = plt.figure(figsize=figsize)

        # Create subplots with appropriate projections
        if plot_types[0] == "radar":
            ax1 = fig.add_subplot(1, 2, 1, projection="polar")
            ax2 = fig.add_subplot(1, 2, 2)
        else:  # plot_types[1] == "radar"
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection="polar")

        axes = [ax1, ax2]
    else:
        # Standard subplots for non-radar plots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Set overall title
    fig.suptitle(
        f"{mask} Factor Profiles{title_suffix}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create each plot using the helper functions
    for i, plot_type in enumerate(plot_types):
        if plot_type == "heatmap":
            _plot_heatmap(
                feature_data_norm,
                mask,
                14,
                figsize,
                value_label,
                "",
                cluster_means,
                axes[i],
            )
            axes[i].set_title("Heatmap", fontsize=14, fontweight="bold")

        elif plot_type == "parallel":
            _plot_parallel_coordinates(
                feature_data_norm,
                mask,
                14,
                figsize,
                value_label,
                "",
                cluster_means,
                axes[i],
            )
            axes[i].set_title(
                "Parallel Coordinates", fontsize=14, fontweight="bold"
            )

        elif plot_type == "bar":
            _plot_grouped_bars(
                feature_data_norm,
                mask,
                14,
                figsize,
                value_label,
                "",
                cluster_means,
                axes[i],
            )
            axes[i].set_title("Grouped Bars", fontsize=14, fontweight="bold")

        elif plot_type == "radar":
            _plot_radar(
                feature_data_norm,
                mask,
                14,
                figsize,
                "",
                cluster_means,
                axes[i],
            )
            axes[i].set_title(
                "Radar Chart", fontsize=14, fontweight="bold", y=1.08
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cluster_profiles(
    model_dict: Dict[str, Any],
    mask: str,
    plot_type: Literal["heatmap", "parallel", "bar", "radar"] = "heatmap",
    title_fontsize: int = 16,
    figsize: tuple = (12, 8),
    normalize: bool = True,
    save_path: Optional[str] = None,
    max_clusters: int = 10,
    selection_method: Literal["size", "variance", "hierarchical"] = "size",
):
    """
    Plot cluster profiles using various visualization methods.

    Args:
        model_dict: The loaded model dictionary
        mask: 'Store' or 'SKU'
        plot_type: Type of visualization
        title_fontsize: Font size for the plot title
        figsize: Figure size
        normalize: Whether to normalize features for better comparison
        save_path: Path to save the plot
        max_clusters: Maximum number of clusters to show (default: 10)
        selection_method: How to select top clusters ('size', 'variance', 'hierarchical')
    """

    cluster_means = calculate_cluster_profiles(
        model_dict, mask, max_clusters, selection_method
    )
    if cluster_means is None:
        return

    feature_names = model_dict["feature_names"]
    feature_data = cluster_means[feature_names]

    # Normalize features if requested (z-score normalization)
    if normalize:
        feature_data = (
            feature_data - feature_data.mean()
        ) / feature_data.std()
        value_label = "Normalized Values"
    else:
        value_label = "Feature Values"

    # Add cluster size info to the title
    total_clusters = len(
        model_dict["assignments"]
        .query("factor_name == @mask")["cluster_id"]
        .unique()
    )
    title_suffix = f" (Top {len(feature_data)}/{total_clusters} clusters by {selection_method})"

    if plot_type == "heatmap":
        _plot_heatmap(
            feature_data,
            mask,
            title_fontsize,
            figsize,
            value_label,
            title_suffix,
            cluster_means,
        )
    elif plot_type == "parallel":
        _plot_parallel_coordinates(
            feature_data,
            mask,
            title_fontsize,
            figsize,
            value_label,
            title_suffix,
            cluster_means,
        )
    elif plot_type == "bar":
        _plot_grouped_bars(
            feature_data,
            mask,
            title_fontsize,
            figsize,
            value_label,
            title_suffix,
            cluster_means,
        )
    elif plot_type == "radar":
        _plot_radar(
            feature_data,
            mask,
            title_fontsize,
            figsize,
            title_suffix,
            cluster_means,
        )
    else:
        logger.error(f"Unknown plot_type: {plot_type}")
        return

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def _plot_heatmap(
    feature_data: pd.DataFrame,
    mask: str,
    title_fontsize: int,
    figsize: tuple,
    value_label: str,
    title_suffix: str,
    cluster_means: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
):
    """Heatmap visualization with cluster size annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    # Create heatmap with better styling
    sns.heatmap(
        feature_data,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": value_label},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        f"{mask} Factor - Heatmap{title_suffix}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cluster ID (size)", fontsize=12, fontweight="bold")

    # Update y-axis labels to include cluster sizes
    y_labels = [
        f"Cluster {idx} (n={cluster_means.loc[idx, 'cluster_size']})"
        for idx in feature_data.index
    ]
    ax.set_yticklabels(y_labels, rotation=0, fontweight="bold")

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold"
    )

    if standalone:
        plt.tight_layout()


def _plot_parallel_coordinates(
    feature_data: pd.DataFrame,
    mask: str,
    title_fontsize: int,
    figsize: tuple,
    value_label: str,
    title_suffix: str,
    cluster_means: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
):
    """Create parallel coordinates plot with cluster size info."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    # Create color palette with better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_data)))

    # Plot each cluster
    for idx, (cluster_id, row) in enumerate(feature_data.iterrows()):
        cluster_size = cluster_means.loc[cluster_id, "cluster_size"]
        ax.plot(
            range(len(feature_data.columns)),
            row.values,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=f"Cluster {cluster_id} (n={cluster_size})",
            color=colors[idx],
            alpha=0.8,
        )

    ax.set_xticks(range(len(feature_data.columns)))
    ax.set_xticklabels(
        feature_data.columns, rotation=45, ha="right", fontweight="bold"
    )
    ax.set_ylabel(value_label, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{mask} Factor - Parallel Coordinates{title_suffix}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if standalone:
        plt.tight_layout()


def _plot_grouped_bars(
    feature_data: pd.DataFrame,
    mask: str,
    title_fontsize: int,
    figsize: tuple,
    value_label: str,
    title_suffix: str,
    cluster_means: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
):
    """Create grouped bar chart with cluster size info."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False

    # Transpose for better bar grouping
    feature_data_t = feature_data.T

    # Create grouped bar plot with better colors
    feature_data_t.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")

    ax.set_title(
        f"{mask} Factor - Grouped Bars{title_suffix}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel(value_label, fontsize=12, fontweight="bold")

    # Update legend with cluster sizes
    legend_labels = [
        f"Cluster {col} (n={cluster_means.loc[col, 'cluster_size']})"
        for col in feature_data.index
    ]
    ax.legend(
        legend_labels,
        title="Cluster ID",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="y")

    if standalone:
        plt.tight_layout()


def _plot_radar(
    feature_data: pd.DataFrame,
    mask: str,
    title_fontsize: int,
    figsize: tuple,
    title_suffix: str,
    cluster_means: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
):
    """Create radar plot with cluster size info."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        standalone = True
    else:
        standalone = False

    num_vars = len(feature_data.columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Color palette with better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_data)))

    # Plot each cluster
    for idx, (cluster_id, row) in enumerate(feature_data.iterrows()):
        values = row.tolist()
        values += values[:1]  # Complete the circle
        cluster_size = cluster_means.loc[cluster_id, "cluster_size"]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2.5,
            markersize=6,
            label=f"Cluster {cluster_id} (n={cluster_size})",
            color=colors[idx],
            alpha=0.8,
        )
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_data.columns, fontsize=10, fontweight="bold")
    ax.set_title(
        f"{mask} Factor - Radar Chart{title_suffix}",
        fontsize=title_fontsize,
        fontweight="bold",
        y=1.08,
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    if standalone:
        plt.tight_layout()


def plot_grouped_bar(metrics_stats):
    models = metrics_stats["Model"].values
    n_models = len(models)
    n_metrics = len(metrics)

    # Spacing between model groups
    x = np.arange(n_models) * 1.3

    # Width of each bar
    width = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))

    eps = 1e-6

    for i, metric in enumerate(metrics):
        means = np.clip(metrics_stats[(metric, "mean")].values, eps, None)
        stds = np.clip(metrics_stats[(metric, "std")].values, eps, None)

        ax.bar(
            x + (i - (n_metrics - 1) / 2) * width,
            means,
            yerr=stds,
            capsize=4,
            width=width,
            edgecolor="black",
            linewidth=2.2,
            color=colors[metric],
            label=metric,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)

    ax.set_ylabel("Metric Value", fontsize=13)
    ax.set_yscale("linear")  # <-- force linear scale

    # Aesthetic style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(width=1.5, labelsize=11)

    ax.legend(title="Metric", fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_grouped_bar_iqr(
    metric_names: list[str],
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    y_tick_step: float = 0.1,
    y_min: float | None = None,
    y_max: float | None = None,
    fn: Path | None = None,
):
    models = df["Model"].unique()
    n_models = len(models)
    n_metrics = len(metric_names)

    # ---- compute median + IQR ----
    stats = {}
    for metric in metric_names:
        grouped = df.groupby("Model")[metric]
        med = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)

        stats[metric] = {
            "median": med.reindex(models).to_numpy(),
            "lower": (med - q1).reindex(models).to_numpy(),
            "upper": (q3 - med).reindex(models).to_numpy(),
        }

    # ---- spacing ----
    width = 0.32  # bar width
    gap = 0.08  # horizontal gap between bars within a group

    # step between group centers (controls space between model groups)
    group_step = 1.2
    x = np.arange(n_models) * group_step

    fig, ax = plt.subplots(figsize=figsize)

    # ---- draw bars ----
    for j, metric in enumerate(metric_names):
        med = stats[metric]["median"]
        lower = stats[metric]["lower"]
        upper = stats[metric]["upper"]
        yerr = np.vstack([lower, upper])

        # for 2 bars: j=0,1 → offsets -0.5, +0.5
        offset_factor = j - (n_metrics - 1) / 2  # -0.5, +0.5 for 2 metrics
        offset = offset_factor * (width + gap)  # distance from group center

        ax.bar(
            x + offset,
            med,
            yerr=yerr,
            capsize=4,
            width=width,
            edgecolor="black",
            linewidth=2.2,
            color=METRIC_COLORS[metric],
            label=metric,
        )
    # axis style…
    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        fontsize=10,
        fontweight="bold",
        rotation=0,
        ha="center",
        rotation_mode="anchor",
    )
    if y_min is not None and y_max is not None:
        # Use provided y_min and y_max
        y_ticks = np.arange(y_min, y_max + y_tick_step / 2, y_tick_step)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min, y_max)
    else:
        # Fall back to automatic limits if not provided
        current_y_min, current_y_max = ax.get_ylim()
        y_min_auto = np.floor(current_y_min / y_tick_step) * y_tick_step
        y_max_auto = np.ceil(current_y_max / y_tick_step) * y_tick_step
        y_ticks = np.arange(
            y_min_auto, y_max_auto + y_tick_step / 2, y_tick_step
        )
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_min_auto, y_max_auto)

    ax.set_ylabel("Metric Value", fontsize=15, fontweight="bold")
    # y_min, y_max = ax.get_ylim()

    # # Round to nearest tick step and add padding
    # y_min_rounded = np.floor(y_min / y_tick_step) * y_tick_step
    # y_max_rounded = np.ceil(y_max / y_tick_step) * y_tick_step

    # # Create y-ticks at specified intervals
    # y_ticks = np.arange(
    #     y_min_rounded, y_max_rounded + y_tick_step / 2, y_tick_step
    # )
    # ax.set_yticks(y_ticks)
    # ax.set_ylim(y_min_rounded, y_max_rounded)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", pad=2)
    # ax.tick_params(width=1.6, labelsize=12)
    ax.legend(title="Metric", fontsize=12)
    plt.subplots_adjust(bottom=0.1)  # bring entire figure closer

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_grouped_bar_iqr_comparison(
    metric_names: list[str],
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_title: str = "Left Plot",
    right_title: str = "Right Plot",
    figsize: tuple = (16, 6),
    fn: Path | None = None,
):
    # Get models from both dataframes
    models_left = df_left["Model"].unique()
    models_right = df_right["Model"].unique()
    n_models_left = len(models_left)
    n_models_right = len(models_right)
    n_metrics = len(metric_names)

    # ---- compute median + IQR for both dataframes ----
    def compute_stats(df, models):
        stats = {}
        for metric in metric_names:
            grouped = df.groupby("Model")[metric]
            med = grouped.median()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)

            stats[metric] = {
                "median": med.reindex(models).to_numpy(),
                "lower": (med - q1).reindex(models).to_numpy(),
                "upper": (q3 - med).reindex(models).to_numpy(),
            }
        return stats

    stats_left = compute_stats(df_left, models_left)
    stats_right = compute_stats(df_right, models_right)

    # ---- determine global y-axis range for consistent scaling ----
    all_values = []
    for df in [df_left, df_right]:
        for metric in metric_names:
            # Filter out infinite and NaN values
            values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                all_values.extend(values.values)

    if not all_values:
        raise ValueError("No valid values found in the data")

    y_min = min(all_values)
    y_max = max(all_values)

    print(f"Debug: y_min = {y_min}, y_max = {y_max}")

    # Check for reasonable range
    y_range = y_max - y_min
    if y_range > 100:  # If range is too large, use automatic ticks
        print(f"Warning: Large y-range ({y_range:.2f}), using automatic ticks")
        use_custom_ticks = False
        y_ticks = None
        y_min_rounded = y_min
        y_max_rounded = y_max
    else:
        # Round to nearest 0.1 and add some padding
        y_min_rounded = np.floor(y_min * 10) / 10 - 0.1
        y_max_rounded = np.ceil(y_max * 10) / 10 + 0.1

        # Create y-ticks every 0.1, but limit the number of ticks
        tick_range = y_max_rounded - y_min_rounded
        if tick_range <= 10:  # Only use 0.1 intervals if range is reasonable
            y_ticks = np.arange(y_min_rounded, y_max_rounded + 0.05, 0.1)
            use_custom_ticks = True
        else:
            # Use larger intervals for larger ranges
            tick_step = 0.5 if tick_range <= 50 else 1.0
            y_ticks = np.arange(
                y_min_rounded, y_max_rounded + tick_step / 2, tick_step
            )
            use_custom_ticks = True

    # ---- spacing ----
    width = 0.32  # bar width
    gap = 0.08  # horizontal gap between bars within a group
    group_step = 1.2

    # Create subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    def plot_bars(ax, stats, models, title):
        n_models = len(models)
        x = np.arange(n_models) * group_step

        # ---- draw bars ----
        for j, metric in enumerate(metric_names):
            med = stats[metric]["median"]
            lower = stats[metric]["lower"]
            upper = stats[metric]["upper"]

            # Filter out invalid values
            valid_mask = ~(
                np.isnan(med)
                | np.isnan(lower)
                | np.isnan(upper)
                | np.isinf(med)
                | np.isinf(lower)
                | np.isinf(upper)
            )

            if not np.any(valid_mask):
                print(f"Warning: No valid data for metric {metric}")
                continue

            yerr = np.vstack([lower, upper])

            # for 2 bars: j=0,1 → offsets -0.5, +0.5
            offset_factor = j - (n_metrics - 1) / 2  # -0.5, +0.5 for 2 metrics
            offset = offset_factor * (
                width + gap
            )  # distance from group center

            ax.bar(
                x + offset,
                med,
                yerr=yerr,
                capsize=4,
                width=width,
                edgecolor="black",
                linewidth=2.2,
                color=METRIC_COLORS[metric],
                label=metric,
            )

        # ---- styling ----
        ax.set_xticks(x)
        ax.set_xticklabels(
            models,
            fontsize=10,
            fontweight="bold",
            rotation=0,
            ha="center",
            rotation_mode="anchor",
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        # Set consistent y-axis
        if use_custom_ticks and y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_ylim(y_min_rounded, y_max_rounded)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["bottom"].set_linewidth(1.8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=2)

    # Plot left subplot
    plot_bars(ax_left, stats_left, models_left, left_title)
    ax_left.set_ylabel("Metric Value", fontsize=15, fontweight="bold")

    # Plot right subplot
    plot_bars(ax_right, stats_right, models_right, right_title)

    # Add legend to the right subplot only (to avoid duplication)
    ax_right.legend(title="Metric", fontsize=12, loc="upper right")

    plt.tight_layout()
    if fn:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
