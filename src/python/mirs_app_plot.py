"""
Section 4 figures and table for the outputs of mirs_app.py
(application_mirs/nnz100): data and filtering means, widths of the pixelwise
95% prediction intervals (random hold-out scheme), held-out scores over time
for both hold-out schemes, and the table of average scores and times per step.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import (FormatStrFormatter, LogLocator, MaxNLocator,
                               NullFormatter, NullLocator)
import numpy as np
import pandas as pd
from plot_results import FIGURE_DIR
from plot_sparsity import STYLE, legend
from mirs_plot import map_grid, interval_grid

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "application_mirs"
import os
OUT = Path(os.environ.get("APP_OUT", DATA / "nnz150"))
METHODS = ["NNV", "HV", "LR"]
SCHEMES = {"random": "random hold-out", "gaps": "swath-gap hold-out"}
TIMES = [1, 6, 11]
# Linear range of the symmetric log scale (log score, RMSPE in %)
LINTHRESH = {"dls": 0.01, "rmspe_excess": 1.0}


def _map_axes(ax):
    """Few longitude/latitude ticks so that labels of the shared font size fit."""
    ax.set_xticks([-120, -105, -90])
    ax.set_yticks([25, 35, 45])


def maps_by_time(fields, grid, side, labels, filename, vmin=0, vmax=50):
    """Rows: time points TIMES; columns: data and the filtering means of the
    methods (gray: unobserved cells), on a common color scale; font sizes as
    in the other figures (plot_results rcParams)."""
    lon = grid["lon"].to_numpy().reshape(side, side)
    lat = grid["lat"].to_numpy().reshape(side, side)
    fig, axes = plt.subplots(len(TIMES), len(fields),
                             figsize=(4.0 * len(fields), 3.7 * len(TIMES)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for r, t in enumerate(TIMES):
        for c, (label, field) in enumerate(zip(labels, fields)):
            ax = axes[r, c]
            ax.set_facecolor("0.6")
            image = ax.pcolormesh(lon, lat, field[t - 1].reshape(side, side),
                                  cmap="Spectral_r", vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
            _map_axes(ax)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"$t={t}$\nlatitude")
            if r == len(TIMES) - 1:
                ax.set_xlabel("longitude")
    cb = fig.colorbar(image, ax=axes, shrink=0.6)
    cb.set_label("TPW (mm)")
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def widths_by_method(widths, grid, side, labels, times, filename):
    """Figure 10: rows are methods, columns time points; widths of the 95%
    prediction intervals on a common color scale, shared font sizes."""
    from mirs_plot import INTERVAL_CMAP
    lon = grid["lon"].to_numpy().reshape(side, side)
    lat = grid["lat"].to_numpy().reshape(side, side)
    fig, axes = plt.subplots(len(widths), len(times),
                             figsize=(4.3 * len(times), 3.7 * len(widths)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    shown = np.stack([w[[t - 1 for t in times]] for w in widths])
    vmin, vmax = float(np.nanmin(shown)), float(np.nanmax(shown))
    for r, (label, width) in enumerate(zip(labels, widths)):
        for c, t in enumerate(times):
            ax = axes[r, c]
            image = ax.pcolormesh(lon, lat, width[t - 1].reshape(side, side),
                                  cmap=INTERVAL_CMAP, vmin=vmin, vmax=vmax,
                                  shading="nearest", rasterized=True)
            _map_axes(ax)
            if r == 0:
                ax.set_title(f"$t={t}$")
            if c == 0:
                ax.set_ylabel(f"{label}\nlatitude")
            if r == len(widths) - 1:
                ax.set_xlabel("longitude")
    cb = fig.colorbar(image, ax=axes, shrink=0.6)
    cb.set_label("95% interval width (mm)")
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _hv_partition_lines(ax, lon, lat, side):
    """Coarse partition lines of HV (GPvecchia MRA, J = 2: splits at 1/2,
    then 1/4 and 3/4 of the domain in each coordinate)."""
    idx = np.arange(side)
    for f in (0.25, 0.5, 0.75):
        c = f * (side - 1)
        ax.axvline(np.interp(c, idx, lon[0]), color="black", linestyle="--",
                   linewidth=1.2)
        ax.axhline(np.interp(c, idx, lat[:, 0]), color="black", linestyle="--",
                   linewidth=1.2)


def _zoom_inset(ax, lon, lat, field, side, cmap, norm, r0, r1, c0, c1, corner,
                lim=None):
    """Magnified inset of rows r0:r1, columns c0:c1 of a width map, with the HV
    partition lines (dashed) and the zoomed region marked on the panel.
    corner: inset position (x0, y0) in axes fraction."""
    ins = ax.inset_axes([corner[0], corner[1], 0.56, 0.56])
    # own (stretched, continuous) color scale lim, shared by the insets of
    # the same time point, so that small jumps are visible
    kw = dict(norm=norm) if lim is None else dict(vmin=lim[0], vmax=lim[1])
    ins.pcolormesh(lon[r0:r1, c0:c1], lat[r0:r1, c0:c1], field[r0:r1, c0:c1],
                   cmap=cmap, shading="nearest", rasterized=True, **kw)
    # partition lines only as short ticks at the inset edges, so that the
    # jump itself stays visible
    idx = np.arange(side)
    tick = dict(color="white", linewidth=2.0, solid_capstyle="butt")
    for f in (0.25, 0.5, 0.75):
        c = f * (side - 1)
        if c0 <= c <= c1:
            x = (c - c0 + 0.5) / (c1 - c0)
            ins.plot([x, x], [0.0, 0.12], transform=ins.transAxes, **tick)
            ins.plot([x, x], [0.88, 1.0], transform=ins.transAxes, **tick)
        if r0 <= c <= r1:
            y = (c - r0 + 0.5) / (r1 - r0)
            ins.plot([0.0, 0.12], [y, y], transform=ins.transAxes, **tick)
            ins.plot([0.88, 1.0], [y, y], transform=ins.transAxes, **tick)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp_ in ins.spines.values():
        sp_.set_edgecolor("white")
        sp_.set_linewidth(1.5)
    ax.indicate_inset_zoom(ins, edgecolor="white", alpha=1, linewidth=1.2)


def means_and_widths(data, means, widths, grid, side, labels, filename,
                     hv_lines=(), vmin=0, vmax=50, panel=(2.55, 2.5),
                     width_step=1.0, zooms=None, zoom_methods=()):
    """Rows: time points TIMES; columns: data, the filtering means of the
    methods (common TPW scale), then the widths of their 95% prediction
    intervals (common width scale, discrete bands of width_step mm). hv_lines:
    labels of methods whose mean maps get the HV partition lines."""
    lon = grid["lon"].to_numpy().reshape(side, side)
    lat = grid["lat"].to_numpy().reshape(side, side)
    k = len(labels)
    ncol = 1 + 2 * k
    fig, axes = plt.subplots(len(TIMES), ncol, figsize=(panel[0] * ncol, panel[1] * len(TIMES)),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    shown = np.stack([w[[t - 1 for t in TIMES]] for w in widths])
    wmin, wmax = float(np.nanmin(shown)), float(np.nanmax(shown))
    width_cmap = plt.get_cmap("turbo")
    width_norm = BoundaryNorm(np.arange(0, wmax + width_step, width_step), width_cmap.N)
    for r, t in enumerate(TIMES):
        panels = ([("data", data[t - 1], "mean")]
                  + [(f"{l} mean", mu[t - 1], "mean") for l, mu in zip(labels, means)]
                  + [(f"{l} width", w[t - 1], "width") for l, w in zip(labels, widths)])
        for c, (title, field, kind) in enumerate(panels):
            ax = axes[r, c]
            if kind == "mean":
                ax.set_facecolor("0.6")
                im_mean = ax.pcolormesh(lon, lat, field.reshape(side, side),
                                        cmap="Spectral_r", vmin=vmin, vmax=vmax,
                                        shading="nearest", rasterized=True)
            else:
                # discrete 2-mm color bands (turbo), so that the small jumps
                # of the HV widths at its partition boundaries show up as
                # breaks in the bands
                im_width = ax.pcolormesh(lon, lat, field.reshape(side, side),
                                         cmap=width_cmap, norm=width_norm,
                                         shading="nearest", rasterized=True)
            if (kind == "width" and zooms and t in zooms
                    and any(title.startswith(h + " ") for h in zoom_methods)):
                r0, r1, c0, c1, _ = zooms[t]
                win = np.concatenate([w[t - 1].reshape(side, side)[r0:r1, c0:c1].ravel()
                                      for l, w in zip(labels, widths)
                                      if l in zoom_methods])
                _zoom_inset(ax, lon, lat, field.reshape(side, side), side,
                            width_cmap, width_norm, *zooms[t],
                            lim=np.quantile(win, [0.01, 0.99]))
            if kind == "mean" and any(title.startswith(h + " ") for h in hv_lines):
                _hv_partition_lines(ax, lon, lat, side)
            _map_axes(ax)
            ax.set_xticks([-115, -100])     # narrow panels: two ticks
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"$t={t}$\nlatitude")
            if r == len(TIMES) - 1:
                ax.set_xlabel("longitude")
    cb = fig.colorbar(im_mean, ax=axes[:, :1 + k], location="bottom", shrink=0.5,
                      aspect=40)
    cb.set_label("TPW (mm)")
    cb = fig.colorbar(im_width, ax=axes[:, 1 + k:], location="bottom", shrink=0.8,
                      aspect=30)
    tick_step = 10 * np.ceil(wmax / 30)     # three ticks, e.g. 0, 20, 40
    cb.set_ticks(np.arange(0, wmax, tick_step))
    cb.minorticks_off()
    cb.set_label("95% interval width (mm)")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def maps(scheme="random"):
    data = np.load(DATA / "data.npz")
    side = int(data["side"])
    grid = pd.read_csv(DATA / "grid.csv")
    fields = {m: np.load(OUT / f"fields_{scheme}_{m}.npz") for m in METHODS}
    labels = [STYLE[m][0] for m in METHODS]
    maps_by_time([data["Y"]] + [fields[m]["mean"] for m in METHODS], grid, side,
                 ["data"] + labels, FIGURE_DIR / "application-maps.pdf")
    noise = json.loads((OUT / f"params_{scheme}.json").read_text())["noise_variance"]
    widths = [2 * 1.96 * np.sqrt(fields[m]["sd"].astype(float) ** 2 + noise)
              for m in METHODS]
    widths_by_method(widths, grid, side, labels, TIMES, FIGURE_DIR / "application-sd.pdf")


def scores_figure(scores):
    """Rows: hold-out schemes; columns: held-out log score, RMSPE (log scale)
    and time per filtering step (log scale) of all three methods over time."""
    d = scores
    cols = [("logscore", "log score", True), ("rmspe", "RMSPE", True),
            ("time_total", "time per step (s)", False)]
    fig, axes = plt.subplots(len(SCHEMES), 3, figsize=(14, 3.6 * len(SCHEMES) + 0.8),
                             layout="constrained", sharex=True, squeeze=False)
    for r, scheme in enumerate(SCHEMES):
        for method in ["LR", "HV", "NNV"]:
            e = d[(d.scheme == scheme) & (d.method == method)].sort_values("t")
            label, color, ls, mk = STYLE[method]
            for c, (y, _, _) in enumerate(cols):
                axes[r, c].plot(e.t, e[y], color=color, linestyle=ls, marker=mk,
                                markersize=4, label=label)
        axes[r, 0].set_ylabel(SCHEMES[scheme])
    for c, (y, title, log) in enumerate(cols):
        axes[0, c].set_title(title)
        for ax in axes[:, c]:
            if log:
                ax.set_yscale("log")
                # Few, plain ticks: log score spans less than a factor of 3
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4) if y == "logscore"
                                           else LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xticks(range(1, int(d.t.max()) + 1, 2 if d.t.max() > 6 else 1))
    fig.supxlabel("time $t$", fontsize=plt.rcParams["axes.labelsize"])
    legend(fig, axes)
    fig.savefig(FIGURE_DIR / "application-scores.pdf", bbox_inches="tight")
    plt.close(fig)


def table(scores):
    names = {"NNV": r"\NNV{}", "HV": "HV", "LR": "LR"}
    s = scores.groupby(["scheme", "method"])[
        ["rmspe", "crps", "logscore", "picp95", "mpiw95"]].mean()
    # total computation time over all time points
    s["time_total"] = scores.groupby(["scheme", "method"])["time_total"].sum()
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"hold-out & method & RMSPE & CRPS & LS & PICP & MPIW & total time (s) \\",
             r"\midrule"]
    for k, scheme in enumerate(SCHEMES):
        if k:
            lines.append(r"\midrule")
        for j, m in enumerate(METHODS):
            v = s.loc[(scheme, m)]
            lead = scheme if j == 0 else ""
            lines.append(f"{lead} & {names[m]} & {v.rmspe:.3f} & {v.crps:.3f} & "
                         f"{v.logscore:.3f} & {v.picp95:.3f} & {v.mpiw95:.2f} & "
                         f"{v.time_total:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table.tex").write_text("\n".join(lines) + "\n")
    print(s.round(3).to_string())


def main():
    # Only the hold-out schemes that were run
    for s in [s for s in SCHEMES if not (OUT / f"scores_{s}.csv").exists()]:
        del SCHEMES[s]
    scores = pd.concat([pd.read_csv(OUT / f"scores_{s}.csv") for s in SCHEMES])
    table(scores)
    scores_figure(scores)
    # The maps (Figures 9-10) are only redrawn on request (APP_MAPS=1)
    if os.environ.get("APP_MAPS") == "1":
        maps()


if __name__ == "__main__":
    main()
