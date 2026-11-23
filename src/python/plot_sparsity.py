"""
Figures for Sections 3.2-3.3 with all methods matched on the average number
of nonzeros per row of the posterior covariance factor (sim_sparsity.py):
  * sparsity-spatial.pdf (main text): dLS against nonzeros per row, one
    column per data model; sparsity-spatial-rmspe.pdf (Supplementary
    Material): the same for RRMSPE;
  * sparsity-filter.pdf: the same for filtering, averaged over time points;
  * sparsity-filter-time.pdf (main text): dLS over time at one sparsity
    level; sparsity-filter-time-rmspe.pdf (Supplementary Material): RRMSPE.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from plot_results import FIGURE_DIR

# dLS is shown on a log scale; values below DLS_FLOOR (Monte Carlo error,
# including slightly negative values) are shown at DLS_FLOOR
DLS_FLOOR = 0.01

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["gaussian", "logistic", "poisson", "gamma"]
STYLE = {  # label, color, linestyle, marker
    "NNV": ("NNV", "darkgreen", "-", "o"),
    "NNV-exact-S3": ("NNV + IC0", "#009E73", "--", "x"),
    "NNV-stage1": ("Stage 1 only", "#56B4E9", ":", "+"),
    "HV": ("HV", "red", "-", "s"),
    "LR": ("low-rank", "blue", ":", "^"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=ROOT / "results_sparsity")
    p.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    p.add_argument("--level", type=int, default=4,
                   help="Sparsity level for the plot over time (4 = about 80).")
    return p.parse_args()


def rrmspe_axis(axes_row):
    """Log scale with a few plain-number ticks, from 1 up to the largest value
    in the row (the axes of a row share their y-axis)."""
    axes_row = np.atleast_1d(axes_row)
    top = max(np.max(np.asarray(line.get_ydata())) for ax in axes_row
              for line in ax.get_lines() if len(line.get_ydata()) > 1)
    candidates = [1, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3] if top < 1.06 \
        else [1, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 3]
    ticks = [t for t in candidates if t <= top * 1.02]
    for ax in axes_row:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())


def legend(fig, axes):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels),
               frameon=False)


def _format_rows(fig, axes, rows, xlabel):
    """Axis scales, labels and legend for the rows ("dLS" and/or "RRMSPE")."""
    for r, y in enumerate(rows):
        for ax in axes[r]:
            ax.grid(True, alpha=0.3)
            if y == "dLS":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            else:
                ax.axhline(1, color="black", linestyle="--", linewidth=1)
        if y == "RRMSPE":
            rrmspe_axis(axes[r])
        axes[r, 0].set_ylabel(y)
    fig.supxlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
    legend(fig, axes)


def by_sparsity(df, methods, xlabel, filename, rows=("dLS", "RRMSPE")):
    s = df.groupby(["model", "method", "level"])[
        ["nnz_per_row", "dLS", "RRMSPE"]].mean().reset_index()
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.7 * len(rows)),
                             layout="constrained", sharex=True, sharey="row",
                             squeeze=False)
    for c, model in enumerate(MODELS):
        for method in methods:
            d = s[(s.model == model) & (s.method == method)].sort_values("level")
            if d.empty:
                continue
            label, color, ls, mk = STYLE[method]
            for r, y in enumerate(rows):
                v = d[y].clip(lower=DLS_FLOOR) if y == "dLS" else d[y]
                axes[r, c].plot(d.nnz_per_row, v, color=color, linestyle=ls,
                                marker=mk, label=label)
        axes[0, c].set_title(model.capitalize())
    _format_rows(fig, axes, rows, xlabel)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


def over_time(df, level, filename, rows=("dLS", "RRMSPE")):
    d = df[df.level == level]
    s = d.groupby(["model", "method", "t"])[["dLS", "RRMSPE"]].mean().reset_index()
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.7 * len(rows)),
                             layout="constrained", sharex=True, sharey="row",
                             squeeze=False)
    for c, model in enumerate(MODELS):
        for method in ["LR", "HV", "NNV"]:
            e = s[(s.model == model) & (s.method == method)]
            label, color, ls, mk = STYLE[method]
            for r, y in enumerate(rows):
                v = e[y].clip(lower=DLS_FLOOR) if y == "dLS" else e[y]
                axes[r, c].plot(e.t, v, color=color, linestyle=ls, marker=mk,
                                markersize=4, label=label)
        axes[0, c].set_title(model.capitalize())
    _format_rows(fig, axes, rows, "time $t$")
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


# Sparsity levels shown (level 1, about 20 nonzeros per row, is left out)
LEVELS_SHOWN = [2, 3, 4, 5, 6]


def read(directory, study):
    """Results of all runs of one study (main run and added levels)."""
    files = sorted(directory.glob(f"{study}*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["level"].isin(LEVELS_SHOWN)]


def main():
    args = parse_args()
    xlabel = (r"average number of nonzeros per row of $\widehat{\mathbf{L}}$, "
              r"$\sum_i|\mathcal{J}_i|/n$")
    df = read(args.dir, "spatial")
    if df is not None:
        spatial_methods = ["LR", "HV", "NNV-stage1", "NNV-exact-S3", "NNV"]
        # main text: dLS only; Supplementary Material: RRMSPE only
        by_sparsity(df[df.method != "Exact"], spatial_methods, xlabel,
                    args.figure_dir / "sparsity-spatial.pdf", rows=("dLS",))
        by_sparsity(df[df.method != "Exact"], spatial_methods, xlabel,
                    args.figure_dir / "sparsity-spatial-rmspe.pdf", rows=("RRMSPE",))
    df = read(args.dir, "filter")
    if df is not None:
        by_sparsity(df, ["LR", "HV", "NNV"], xlabel,
                    args.figure_dir / "sparsity-filter.pdf")
        over_time(df, args.level, args.figure_dir / "sparsity-filter-time.pdf",
                  rows=("dLS",))
        over_time(df, args.level, args.figure_dir / "sparsity-filter-time-rmspe.pdf",
                  rows=("RRMSPE",))


if __name__ == "__main__":
    main()
