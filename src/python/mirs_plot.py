"""
Section 4 figures: TPW data and filtering means (as Figure 5 of Jurek and
Katzfuss, 2022), widths of the pixelwise 95% prediction intervals, and
held-out scores over time (as their Figure 6).
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd
from plot_results import COLORS, FIGURE_DIR, LABELS, STYLES

METHODS = ["sparse", "HV", "LR"]

# Colormap of the uncertainty maps:
# colorRampPalette(c("brown", "blue", "green")) in R
INTERVAL_CMAP = LinearSegmentedColormap.from_list(
    "brown_blue_green", ["#A52A2A", "#0000FF", "#00FF00"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("application_mirs"))
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--times", default="1,6,11")
    return parser.parse_args()


def map_grid(fields, grid, side, row_labels, times, filename, cmap, vmin, vmax,
             colorbar_label):
    lon = grid["lon"].to_numpy().reshape(side, side)
    lat = grid["lat"].to_numpy().reshape(side, side)
    fig, axes = plt.subplots(len(fields), len(times),
                             figsize=(3.4 * len(times), 3.0 * len(fields)),
                             sharex=True, sharey=True, squeeze=False)
    for r, (label, field) in enumerate(zip(row_labels, fields)):
        for c, t in enumerate(times):
            ax = axes[r, c]
            ax.set_facecolor("0.6")
            image = ax.pcolormesh(lon, lat, field[t - 1].reshape(side, side),
                                  cmap=cmap, vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
            if r == 0:
                ax.set_title(f"t = {t}", fontsize=13)
            if c == 0:
                ax.set_ylabel(f"{label}\nlatitude", fontsize=12)
            if r == len(fields) - 1:
                ax.set_xlabel("longitude", fontsize=12)
            ax.tick_params(labelsize=10)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.895, 0.2, 0.02, 0.6))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(colorbar_label, fontsize=16, labelpad=10)
    colorbar.ax.tick_params(labelsize=13)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interval_grid(widths, grid, side, row_labels, times, filename):
    """Interval-width maps on a common color scale."""
    lon = grid["lon"].to_numpy().reshape(side, side)
    lat = grid["lat"].to_numpy().reshape(side, side)
    fig, axes = plt.subplots(len(widths), len(times),
                             figsize=(3.6 * len(times), 3.0 * len(widths)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    shown = np.stack([w[[t - 1 for t in times]] for w in widths])
    vmin, vmax = float(np.nanmin(shown)), float(np.nanmax(shown))
    for r, (label, width) in enumerate(zip(row_labels, widths)):
        for c, t in enumerate(times):
            ax = axes[r, c]
            image = ax.pcolormesh(lon, lat, width[t - 1].reshape(side, side),
                                  cmap=INTERVAL_CMAP, vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
            if r == 0:
                ax.set_title(f"t = {t}", fontsize=13)
            if c == 0:
                ax.set_ylabel(f"{label}\nlatitude", fontsize=12)
            if r == len(widths) - 1:
                ax.set_xlabel("longitude", fontsize=12)
            ax.tick_params(labelsize=10)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.6)
    colorbar.set_label("95% interval width (mm)", fontsize=16, labelpad=10)
    colorbar.ax.tick_params(labelsize=13)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    times = [int(t) for t in args.times.split(",")]
    data = np.load(args.data_dir / "data.npz")
    side = int(data["side"])
    grid = pd.read_csv(args.data_dir / "grid.csv")
    filters = {m: np.load(args.data_dir / f"filter_{m}.npz") for m in METHODS}
    labels = [LABELS[m] for m in METHODS]

    map_grid([data["Y"]] + [filters[m]["mean"] for m in METHODS], grid, side,
             ["data"] + labels, times,
             args.figure_dir / "application-maps.pdf", "Spectral_r", 0, 50,
             "TPW (mm)")
    # Width 2 * 1.96 * sd of the pixelwise 95% prediction intervals for
    # y_{t,i}, with predictive variance = filtering variance + tau^2
    with open(args.data_dir / "params.json") as f:
        noise = json.load(f)["noise_variance"]
    widths = [2 * 1.96 * np.sqrt(filters[m]["sd"].astype(float) ** 2 + noise)
              for m in METHODS]
    interval_grid(widths, grid, side, labels, times,
                  args.figure_dir / "application-sd.pdf")

    scores = pd.read_csv(args.data_dir / "filter_scores.csv")
    # Doubling ticks, so that equal ratios are equally spaced on the log axis
    panels = [("rmspe", "RMSPE", [0.25, 0.5, 1, 2, 4, 8]),
              ("crps", "CRPS", [0.125, 0.25, 0.5, 1, 2, 4]),
              ("logscore", "log score", None)]
    # One row; log score on a linear scale, as it can be negative
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.8))
    for ax, (column, ylabel, ticks) in zip(axes, panels):
        for method in METHODS:
            s = scores[scores["method"] == method]
            ax.plot(s["t"], s[column], STYLES[method], color=COLORS[method],
                    marker="o", label=LABELS[method])
        ax.set_xlabel("time t", fontsize=21)
        ax.set_ylabel(ylabel, fontsize=21)
        ax.set_xticks(scores["t"].unique())
        if ticks is not None:
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(ticks[0] * 0.9, ticks[-1] * 1.1)
        ax.grid(True, which="major", axis="y", alpha=0.4)
        ax.tick_params(labelsize=17)
    axes[0].legend(fontsize=17)
    fig.tight_layout()
    fig.savefig(args.figure_dir / "application-scores.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
