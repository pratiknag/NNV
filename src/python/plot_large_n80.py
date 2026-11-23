"""
Section 3.5: large-n comparison with one setting per method, matched at about
80 nonzeros per row of the posterior covariance factor (run_large_n80.sh).

  * large_n/table_spatial80.tex: spatial-only time per data set, log score
    relative to NNV and RMSPE for n = 10,000, 40,000 and 90,000,
    averaged over data sets (and the GPvecchia times of HV and LR, if run);
  * figures/large-n-filter80.pdf: filtering on the 300 x 300 grid over
    time: log score relative to NNV, RMSPE and time per step.

The log score is -log q(x_true | y); relative to NNV, it equals
the dLS up to a constant common to all methods (Section 3.1).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from plot_results import FIGURE_DIR
from plot_sparsity import STYLE, legend, rrmspe_axis

ROOT = Path(__file__).resolve().parents[2] / "large_n"
MODELS = ["gaussian", "logistic", "poisson", "gamma"]
METHODS = ["NNV", "HV", "LR"]
# Filtering: logistic data left out for now (Stage-3 error accumulates at this
# sparsity for n = 90,000; see the notes in the project memory)
FILTER_MODELS = ["gaussian", "poisson", "gamma"]
FLOOR = 0.1  # relative log scores below this are shown at this value
FLOOR_PCT = 0.01  # RMSPE excess (%) below this is shown at this value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=ROOT)
    p.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return p.parse_args()


def relative(df, keys):
    """Log score minus that of NNV, and RMSPE relative to that of NNV (in
    percent above NNV), for the same data set (and time point)."""
    ref = df[df.method == "NNV"].set_index(keys)[["ls", "rmspe"]].rename(
        columns={"ls": "ls_nnv", "rmspe": "rmspe_nnv"})
    df = df.join(ref, on=keys)
    df["dls"] = df.ls - df.ls_nnv
    df["rmspe_excess"] = 100 * (df.rmspe / df.rmspe_nnv - 1)
    return df


def spatial_table(args):
    frames = [pd.read_csv(f) for f in sorted(args.dir.glob("spatial80_side*.csv"))]
    if not frames:
        return
    d = relative(pd.concat(frames), ["side", "model", "dataset"])
    s = d.groupby(["side", "model", "method"])[
        ["nnz_per_row", "time_total", "dls", "rmspe_excess"]].mean()
    g = None
    gfiles = sorted(args.dir.glob("gpvecchia80_side*.csv"))
    if gfiles:
        g = pd.concat([pd.read_csv(f) for f in gfiles]).groupby(
            ["side", "model", "method"]).time_total.mean()
    print(s.round(3).to_string())
    names = {"NNV": r"\NNV", "HV": "HV", "LR": "LR"}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"$n$ & & \multicolumn{4}{c}{time per data set (s)} \\",
             r"\cmidrule(lr){3-6}",
             r" & & Gaussian & Logistic & Poisson & Gamma \\", r"\midrule"]
    # time block, then log-score and RMSPE blocks
    blocks = [("time_total", "time per data set (s)", "%.2f"),
              ("dls", "log score relative to NNV", "%.0f"),
              ("rmspe_excess", "RMSPE above \\NNV{} (\\%)", "%.1f")]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"$n$ & method & Gaussian & Logistic & Poisson & Gamma \\"]
    for col, title, fmt in blocks:
        lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\"]
        for side in sorted(d.side.unique()):
            for k, method in enumerate(METHODS):
                vals = [fmt % s.loc[(side, m, method), col] for m in MODELS]
                lead = f"${side ** 2:,}$".replace(",", "{,}") if k == 0 else ""
                lines.append(f"{lead} & {names[method]} & " + " & ".join(vals) + r" \\")
                if col == "time_total" and g is not None and method != "NNV":
                    gv = [fmt % g.loc[(side, m, method)] if (side, m, method) in g.index
                          else "--" for m in MODELS]
                    lines.append(f" & {names[method]} (GPvecchia) & " + " & ".join(gv) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = args.dir / "table_spatial80.tex"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


YLABEL = {"dls": "log score\nminus NNV", "rmspe_excess": "RMSPE above\nNNV (%)",
          "time_total": "time per\ndata set (s)"}


def spatial_figure(args, rows=("dls", "time_total"), name="large-n-spatial80.pdf"):
    """Rows (default: log score relative to NNV and time per data set, for
    the main text; ("rmspe_excess",) for the Supplementary Material);
    columns: data models; x-axis: number of locations n."""
    frames = [pd.read_csv(f) for f in sorted(args.dir.glob("spatial80_side*.csv"))]
    if not frames:
        return
    d = relative(pd.concat(frames), ["side", "model", "dataset"])
    s = d.groupby(["model", "method", "n"])[["dls", "rmspe_excess", "time_total"]].mean().reset_index()
    ns = sorted(s.n.unique())
    gfiles = sorted(args.dir.glob("gpvecchia80_side*.csv"))
    g = (pd.concat([pd.read_csv(f) for f in gfiles]).groupby(["model", "method", "n"])
         .time_total.mean().reset_index()) if gfiles else None
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.5 * len(rows)), layout="constrained",
                             sharex=True, sharey="row", squeeze=False)
    for c, model in enumerate(MODELS):
        for method in ["LR", "HV", "NNV"]:
            e = s[(s.model == model) & (s.method == method)].sort_values("n")
            label, color, ls, mk = STYLE[method]
            for r, y in enumerate(rows):
                if method == "NNV" and y != "time_total":
                    continue  # the reference: zero by definition
                axes[r, c].plot(e.n, e[y], color=color, linestyle=ls, marker=mk, label=label)
        if g is not None and "time_total" in rows:  # GPvecchia (no Stage 3)
            for method in ["LR", "HV"]:
                e = g[(g.model == model) & (g.method == method)].sort_values("n")
                label, color, _, mk = STYLE[method]
                axes[rows.index("time_total"), c].plot(
                    e.n, e.time_total, color=color, linestyle="--", marker=mk,
                                markerfacecolor="none", label=f"{label} (GPvecchia)")
        axes[0, c].set_title(model.capitalize())
        for r in range(len(rows)):
            axes[r, c].set_xscale("log")
            axes[r, c].set_yscale("log")
            axes[r, c].set_xticks(ns, [f"{n // 1000}" for n in ns])
            axes[r, c].xaxis.set_minor_formatter(NullFormatter())
            axes[r, c].yaxis.set_major_formatter(FormatStrFormatter("%g"))
            axes[r, c].yaxis.set_minor_formatter(NullFormatter())
            axes[r, c].grid(True, alpha=0.3)
        for r in range(len(rows)):
            axes[r, c].yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    for r, y in enumerate(rows):
        axes[r, 0].set_ylabel(YLABEL[y])
    fig.supxlabel("number of locations $n$ (thousands)", fontsize=plt.rcParams["axes.labelsize"])
    legend(fig, axes)
    out = args.figure_dir / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def filter_figure(args):
    f = args.dir / "filter80_side300.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = relative(d[d.model.isin(FILTER_MODELS)], ["model", "dataset", "t"])
    nnz = d.groupby("method").nnz_per_row.mean()
    s = d.groupby(["model", "method", "t"])[["dls", "rmspe_excess", "time_total"]].mean().reset_index()
    fig, axes = plt.subplots(3, len(FILTER_MODELS), figsize=(13, 10.5),
                             layout="constrained", sharex=True, sharey="row")
    for c, model in enumerate(FILTER_MODELS):
        for method in ["LR", "HV", "NNV"]:
            e = s[(s.model == model) & (s.method == method)]
            label, color, ls, mk = STYLE[method]
            for r, y in enumerate(["dls", "rmspe_excess", "time_total"]):
                if method == "NNV" and r < 2:
                    continue  # the reference: zero by definition
                axes[r, c].plot(e.t, e[y], color=color, linestyle=ls, marker=mk,
                                markersize=4, label=label)
        axes[0, c].set_title(model.capitalize())
        axes[0, c].set_yscale("log")
        axes[0, c].yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[1, c].set_yscale("log")
        axes[1, c].yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[2, c].set_yscale("log")
        axes[2, c].yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[2, c].yaxis.set_minor_formatter(NullFormatter())
        for r in range(3):
            axes[r, c].yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            axes[r, c].yaxis.set_minor_formatter(NullFormatter())
            axes[r, c].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("log score\nminus NNV")
    axes[1, 0].set_ylabel("RMSPE above\nNNV (%)")
    axes[2, 0].set_ylabel("time per\nstep (s)")
    axes[2, 0].set_xticks([2, 4, 6, 8, 10])
    fig.supxlabel("time $t$", fontsize=plt.rcParams["axes.labelsize"])
    legend(fig, axes)
    out = args.figure_dir / "large-n-filter80.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print(d.groupby(["model", "method"])[["nnz_per_row", "time_total", "dls", "rmspe_excess",
                                          "newton_iters"]].mean().round(3).to_string())


def main():
    args = parse_args()
    spatial_table(args)
    spatial_figure(args)                       # main text
    spatial_figure(args, rows=("rmspe_excess",),  # Supplementary Material
                   name="large-n-spatial80-rmspe.pdf")
    filter_figure(args)


if __name__ == "__main__":
    main()
