"""
Section 4: comparison of NNV, HV and LR with the exact Kalman filter
(mirs_exact_full.py), random hold-outs. Reads application_mirs/results/ and writes
  figures/application-exact.pdf        Figure 7: held-out RRMSPE and dLS relative
                                       to the exact filter, time per step
  figures/application-exact-rmsd.pdf   Figure S8: RMS differences of the filtering
                                       means and sds from the exact filter
  application_mirs/results/table_exact.tex   Table 1
  application_mirs/results/{exact_rmsd.csv, scores_random_exact_kf.csv}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np
import pandas as pd
from plot_results import FIGURE_DIR
from plot_sparsity import STYLE
from mirs_estimate7 import load

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "application_mirs" / "results"
METHODS = ["LR", "HV", "NNV"]


def rmsd_table():
    _, _, an, _ = load(ROOT / "application_mirs", scheme="random")
    obs = ~np.isnan(an)
    ex = np.load(DIR / "fields_random_exact.npz")
    files = {m: DIR / f"fields_random_{m}.npz" for m in METHODS}
    rows = []
    for m, f in files.items():
        fl = np.load(f)
        for t in range(an.shape[0]):
            for cells, mask in [("with data", obs[t]), ("without data", ~obs[t])]:
                dm = fl["mean"][t, mask].astype(float) - ex["mean"][t, mask]
                ds = fl["sd"][t, mask].astype(float) - ex["sd"][t, mask]
                rows.append({"method": m, "t": t + 1, "cells": cells,
                             "n_cells": int(mask.sum()),
                             "rmsd_mean": np.sqrt(np.mean(dm ** 2)),
                             "rmsd_sd": np.sqrt(np.mean(ds ** 2))})
    d = pd.DataFrame(rows)
    d.to_csv(DIR / "exact_rmsd.csv", index=False)
    return d


def figure_rmsd(rmsd):
    """Supplement figure (application-exact-rmsd.pdf): root-mean-square
    differences of the filtering means (left) and sds (right) of NNV, HV and
    LR from those of the exact filter over time, at cells with training data
    at time t (dashed, open markers) and at all other cells (solid)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained", sharex=True)
    for method in ["NNV", "HV", "LR"]:
        label, color, _, mk = STYLE[method]
        for cells, ls, fill in [("without data", "-", color), ("with data", "--", "white")]:
            e = rmsd[(rmsd.method == method) & (rmsd.cells == cells)].sort_values("t")
            for ax, col in zip(axes, ["rmsd_mean", "rmsd_sd"]):
                ax.plot(e.t, e[col], color=color, linestyle=ls, marker=mk, markersize=4,
                        markerfacecolor=fill, label=f"{label}, {cells}")
    for ax, title in zip(axes, ["filtering means", "filtering standard deviations"]):
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 12, 2))
    axes[0].set_ylabel("RMS difference")
    fig.supxlabel("time $t$", fontsize=plt.rcParams["axes.labelsize"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(FIGURE_DIR / "application-exact-rmsd.pdf", bbox_inches="tight")
    plt.close(fig)


DLS_FLOOR = 1e-3


def figure(scores):
    """Three panels over time: held-out RRMSPE and dLS relative to the exact
    filter (dLS below DLS_FLOOR, incl. slightly negative values, shown at the
    floor), and time per filtering step (incl. the exact filter)."""
    ex = scores[scores.method == "exact"].set_index("t")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), layout="constrained", sharex=True)
    styles = dict(STYLE)
    styles["exact"] = ("exact", "black", "-.", "D")
    for method in ["exact", "LR", "HV", "NNV"]:
        label, color, ls, mk = styles[method]
        e = scores[scores.method == method].set_index("t").loc[ex.index]
        kw = dict(color=color, linestyle=ls, marker=mk, markersize=4, label=label)
        if method != "exact":
            axes[0].plot(ex.index, e.rmspe / ex.rmspe, **kw)
            axes[1].plot(ex.index, np.maximum(e.logscore - ex.logscore, DLS_FLOOR), **kw)
        axes[2].plot(ex.index, e.time_total, **kw)
    titles = ["RRMSPE", "dLS", "time per step (s)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[0].set_yticks([1, 1.5, 2, 3])
    axes[1].set_yticks([0.001, 0.01, 0.1, 1])
    # log minor ticks and grid lines, so that the log scale is visible
    from matplotlib.ticker import NullFormatter
    for ax in axes[:2]:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", axis="y", alpha=0.15)
    axes[2].set_yticks([10, 30, 100, 300])
    axes[0].set_xticks(range(1, 12, 2))
    fig.supxlabel("time $t$", fontsize=plt.rcParams["axes.labelsize"])
    h, l = axes[2].get_legend_handles_labels()
    order = [l.index(styles[m][0]) for m in ["NNV", "HV", "LR", "exact"]]
    fig.legend([h[k] for k in order], [l[k] for k in order],
               loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(FIGURE_DIR / "application-exact.pdf", bbox_inches="tight")
    plt.close(fig)


def exact_scores():
    """Held-out scores of the exact filter: y ~ N(mean + trend, sd^2 + tau^2)."""
    import json
    from mirs_filter import gaussian_scores
    data = np.load(ROOT / "application_mirs" / "data.npz")
    test = data["split"] == 2
    noise = json.loads((DIR / "params_used.json").read_text())["noise"]
    ex = np.load(DIR / "fields_random_exact.npz")
    times = pd.read_csv(DIR / "exact_random.csv")
    recs = []
    for t in range(test.shape[0]):
        k = test[t]
        recs.append({"t": t + 1, "method": "exact",
                     "time_total": times.time_total[t],
                     **gaussian_scores(data["Y"][t, k], ex["mean"][t, k].astype(float),
                                       np.sqrt(ex["sd"][t, k].astype(float) ** 2 + noise))})
    d = pd.DataFrame(recs)
    d.to_csv(DIR / "scores_random_exact_kf.csv", index=False)
    return d


def table(scores):
    """Table 1: averages over t of the held-out scores; dLS = LS - LS(exact),
    RRMSPE = RMSPE / RMSPE(exact) per time point, averaged over t."""
    ex = scores[scores.method == "exact"].set_index("t")
    rows = []
    for m in ["exact", "NNV", "HV", "LR"]:
        e = scores[scores.method == m].set_index("t").loc[ex.index]
        rows.append({"method": m, "rmspe": e.rmspe.mean(),
                     "rrmspe": (e.rmspe / ex.rmspe).mean(),
                     "crps": e.crps.mean(), "logscore": e.logscore.mean(),
                     "dls": (e.logscore - ex.logscore).mean(),
                     "picp95": e.picp95.mean(), "mpiw95": e.mpiw95.mean(),
                     "time_total": e.time_total.sum()})
    tab = pd.DataFrame(rows)
    names = {"exact": "exact", "NNV": r"\NNV{}", "HV": "HV", "LR": "LR"}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"method & RRMSPE & dLS & CRPS & PICP & MPIW & total time (s) \\",
             r"\midrule"]
    for _, v in tab.iterrows():
        lines.append(f"{names[v.method]} & {v.rrmspe:.3f} & {v.dls:.3f} & "
                     f"{v.crps:.3f} & "
                     f"{v.picp95:.3f} & {v.mpiw95:.2f} & {v.time_total:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DIR / "table_exact.tex").write_text("\n".join(lines) + "\n")
    print(tab.round(4).to_string())


def main():
    rmsd = rmsd_table()
    scores = pd.concat([exact_scores()] + [pd.read_csv(DIR / f"scores_random_{m}.csv")
                                           for m in METHODS])
    figure(scores)
    figure_rmsd(rmsd[rmsd.method.isin(METHODS)])
    table(scores)
    g = rmsd.groupby(["method", "cells"])[["rmsd_mean", "rmsd_sd"]]
    print(g.mean().round(4).to_string())
    print(g.max().round(4).to_string())


if __name__ == "__main__":
    main()
