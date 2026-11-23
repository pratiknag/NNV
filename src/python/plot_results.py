import argparse
from pathlib import Path
import torch
import math
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import seaborn as sns
import numpy as np
import pandas as pd
from sparse_functions import *

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

plt.rcParams.update({
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2.5,
    "lines.markersize": 7,
})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-locs",
        type=int,
        nargs="+",
        default=None,
        help="N_locs values to plot (default: all)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Folder for the figures (default: figures/).",
    )
    return parser.parse_args()


COLORS = {
    "HV": "red",
    "LR": "blue",
    "sparse": "darkgreen",
    "sparse_stage1": "lightgreen",
    "sparse_stage2": "lightblue",
}

STYLES = {
    "HV": "-",
    "LR": ":",
    "sparse": "-",
    "sparse_stage1": "--",
    "sparse_stage2": "-.",
}

LABELS = {
    "HV": "HV",
    "LR": "low-rank",
    "sparse": "NNV",
    "sparse_stage1": "sparse stage 1",
    "sparse_stage2": "NNV + IC0",
}

# Colors for the sparse-only plot
SPARSE_COLORS = {
    "sparse": "#0072B2",
    "sparse_stage1": "#D55E00",
    "sparse_stage2": "#009E73",
}


def main():
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.read_csv("complete_results.csv")
    if args.n_locs is not None:
        missing = set(args.n_locs) - set(results_df["N_locs"].unique())
        if missing:
            raise ValueError(
                f"N_locs values not in results: {sorted(missing)}. "
                f"Available: {sorted(results_df['N_locs'].unique())}"
            )
        results_df = results_df[results_df["N_locs"].isin(args.n_locs)]
    # Older result files contain only the Gaussian model
    if "model" not in results_df:
        results_df["model"] = "gaussian"
    models = [
        m for m in DATA_MODELS
        if m in results_df["model"].unique()
    ]

    # RRMSPE values cannot be less than 1
    rrmspe_cols = [
        c for c in results_df.columns
        if c.startswith("rrmspe_")
    ]
    results_df[rrmspe_cols] = results_df[rrmspe_cols].clip(lower=1.0)
    # Replace negative dLS values with 0.00003
    dls_cols = [
        c for c in results_df.columns
        if c.startswith("dLS_")
    ]
    # results_df[dls_cols] = results_df[dls_cols].clip(lower=0.00003)

    # RRMSPE plot
    plot_grid(
        results_df,
        models,
        prefix="rrmspe_",
        methods=["HV", "LR", "sparse", "sparse_stage2"],
        ylabel="RRMSPE",
        filename=args.figure_dir / "spatial-RRMSPE.pdf",
        log_y=True,
    )

    # RRMSPE plot: sparse methods only
    plot_grid(
        results_df,
        models,
        prefix="rrmspe_",
        methods=["sparse", "sparse_stage1", "sparse_stage2"],
        ylabel="RRMSPE",
        filename=args.figure_dir / "spatial-RRMSPE-3models.pdf",
        log_y=True,
        colors=SPARSE_COLORS,
        max_n=25,
    )

    # dLS plot
    plot_grid(
        results_df,
        models,
        prefix="dLS_",
        methods=["HV", "LR", "sparse", "sparse_stage2"],
        ylabel="dLS",
        filename=args.figure_dir / "spatial-dLS.pdf",
        log_y=False,
    )


def set_log_ticks(ax):
    low, high = ax.get_ylim()
    ticks = [t for t in (1, 1.2, 1.5, 2, 3, 5, 10) if low <= t <= high]
    if len(ticks) < 3:
        ticks = [t for t in (1, 1.1, 1.2, 1.4) if low <= t <= high]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def n_locs_title(n_locs):
    return rf"$N_{{\mathrm{{locs}}}} = {int(n_locs)}$"


def add_legend(fig, ax, title):
    # Single legend in one row, with the top N_locs heading above it
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        title=title,
        title_fontsize=plt.rcParams["axes.titlesize"],
        frameon=False,
    )


def plot_grid(
    results_df,
    models,
    prefix,
    methods,
    ylabel,
    filename,
    log_y,
    colors=COLORS,
    max_n=None,
    x_column="N",
    xlabel="conditioning-set size $m$",
    y_min=None,
):
    # One column per model, one row per N_locs value
    n_locs_values = sorted(results_df["N_locs"].unique())
    n_rows, n_cols = len(n_locs_values), len(models)

    # Extra height in the top row for the legend
    row_heights = [3.2 + 1.0] + [3.2] * (n_rows - 1)
    fig = plt.figure(
        figsize=(3.5 * n_cols, sum(row_heights)),
        layout="constrained",
    )
    # One sub-figure per row so each row gets its own N_locs heading
    subfigs = np.atleast_1d(
        fig.subfigures(n_rows, 1, height_ratios=row_heights)
    )
    axes = np.array([
        subfig.subplots(1, n_cols, squeeze=False)[0]
        for subfig in subfigs
    ])

    for row, n_locs in enumerate(n_locs_values):
        # N_locs heading above each row (the top one is the legend title)
        if row > 0:
            subfigs[row].suptitle(
                n_locs_title(n_locs),
                fontsize=plt.rcParams["axes.titlesize"],
            )
        for col, model in enumerate(models):
            ax = axes[row, col]
            scores = results_df[
                (results_df["model"] == model)
                & (results_df["N_locs"] == n_locs)
            ]
            if max_n is not None:
                scores = scores[scores[x_column] <= max_n]
            scores = scores.sort_values(x_column)
            neighbors = scores[x_column].to_numpy()

            for method in methods:
                ax.plot(
                    neighbors,
                    scores[prefix + method].to_numpy(),
                    color=colors[method],
                    linestyle=STYLES[method],
                    marker="o",
                    label=LABELS[method],
                )

            if log_y:
                # Reference line RRMSPE = 1
                ax.axhline(
                    y=1.0,
                    color="black",
                    linestyle="--",
                )
                ax.set_yscale("log")
                # A few plain tick labels (no 1.2x10^0 style, no crowded
                # minor labels)
                set_log_ticks(ax)
            if max_n is not None:
                ax.set_xlim(left=0, right=max_n)
            if y_min is not None:
                ax.set_ylim(bottom=y_min)

            # Model heading on the top row only
            if row == 0:
                ax.set_title(model.capitalize())

            ax.grid(
                True,
                which="both",
                alpha=0.3,
            )

    # Single axis labels for the whole figure
    fig.supxlabel(
        xlabel,
        fontsize=plt.rcParams["axes.labelsize"],
    )
    fig.supylabel(
        ylabel,
        fontsize=plt.rcParams["axes.labelsize"],
    )

    add_legend(subfigs[0], axes[0, 0], n_locs_title(n_locs_values[0]))

    fig.savefig(
        filename,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
