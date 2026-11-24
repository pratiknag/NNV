"""
Section 3.2 maps: exact (Laplace) posterior mean and posterior means of HV and
NN-triple for one simulated spatial data set (N = 1600), to show the
block-boundary artifacts of HV. Second row: the differences of the two
approximations from the exact posterior mean. Exact and HV means come from
src/R/export_spatial_fields.R.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from veccs.orderings import maxmin_cpp_ancestor
from plot_results import FIGURE_DIR
from sparse_functions import (DATA_MODELS, calibrate_rho_values,
                              exponential_covariance, get_likelihood, ichol0,
                              make_sparse_patterns, newton_laplace,
                              sparse_inverse_cholesky, spai_from_V)

TITLES = {"gaussian": "Gaussian", "logistic": "logistic",
          "poisson": "Poisson", "gamma": "gamma"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DATA_MODELS))
    parser.add_argument("--n", type=int, default=1600)
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--conditioning-size", type=int, default=10)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def nn_triple_mean(data, model, rho, range_param=0.15, variance=1.0,
                   noise_variance=0.2, gamma_shape=2.0):
    """Posterior mean of NN-triple, as in run_experiment."""
    dtype = torch.float64
    X = torch.as_tensor(data[["x1", "x2"]].to_numpy(), dtype=dtype)
    Z = torch.as_tensor(data["z"].to_numpy(), dtype=dtype)
    n = X.shape[0]
    ret = maxmin_cpp_ancestor(X.numpy(), np.empty((0, 2)), rho)
    S_rho, S_A = make_sparse_patterns(ret.sparsity, ret.ancestor_set_reduced,
                                      n, dtype=dtype)
    order = torch.as_tensor(ret.maximin_order, dtype=torch.long)
    X, Z = X[order], Z[order]
    U = sparse_inverse_cholesky(
        exponential_covariance(X, variance=variance, range_param=range_param),
        S_rho)
    UUT = (U @ U.t()).coalesce()
    score, hess = get_likelihood(model, noise_variance, gamma_shape)

    def solve(h, rhs):
        D = torch.sparse_coo_tensor(torch.arange(n).repeat(2, 1), h, (n, n))
        W = (UUT + D).coalesce()
        V = ichol0(W, upper=True)
        _, Winv = spai_from_V(V, S_A)
        return torch.sparse.mm(Winv, rhs.unsqueeze(1)).squeeze(1), W, V

    pred, _, _ = newton_laplace(solve, score, hess, Z,
                                linear=model == "gaussian")
    mean = np.empty(n)
    mean[order.numpy()] = pred["mean"].numpy()
    return mean


def plot_model(fields, side, size, hv_m, nn_k, filename, model):
    """
    Row 1: exact posterior mean, HV and NN-triple posterior means. Row 2:
    observations (bottom left) and differences of HV and NN-triple from the exact posterior mean, below the
    corresponding maps. Tick labels only on the outer edges of the panels.
    """
    def img(v):
        return np.asarray(v).reshape(side, side)  # x1 varies fastest: rows are x2

    extent = (0, 1, 0, 1)
    hv, nn, exact = fields[f"hv_{size}"], fields["nn"], fields["exact"]
    top = [("exact posterior mean", exact),
           (f"HV (m = {hv_m})", hv),
           (f"NNV (m = {nn_k})", nn)]
    bottom = [("HV $-$ exact", hv - exact), ("NNV $-$ exact", nn - exact)]
    lim = np.abs(np.r_[exact, hv, nn]).max()
    dlim = max(np.abs(d).max() for _, d in bottom)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), sharex=True,
                             sharey=True, layout="constrained")
    for ax, (title, v) in zip(axes[0], top):
        im_top = ax.imshow(img(v), origin="lower", extent=extent,
                           cmap="Spectral_r", vmin=-lim, vmax=lim,
                           interpolation="nearest")
        ax.set_title(title, fontsize=17)
    for ax, (title, v) in zip(axes[1, 1:], bottom):
        im_diff = ax.imshow(img(v), origin="lower", extent=extent,
                            cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                            interpolation="nearest")
        ax.set_title(title, fontsize=17)
    # Bottom left: the observations z (binary or counts)
    z = np.asarray(fields["z"])
    ax = axes[1, 0]
    if model == "logistic":
        cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#252525"])
        im_z = ax.imshow(img(z), origin="lower", extent=extent, cmap=cmap,
                         vmin=-0.5, vmax=1.5, interpolation="nearest")
        ticks = [0, 1]
    else:
        zmax = int(z.max())
        cmap = plt.get_cmap("viridis", zmax + 1)
        im_z = ax.imshow(img(z), origin="lower", extent=extent, cmap=cmap,
                         vmin=-0.5, vmax=zmax + 0.5, interpolation="nearest")
        ticks = np.arange(0, zmax + 1, 5 if zmax > 10 else 1)
    ax.set_title("data", fontsize=17)
    # colorbar in an inset to the left of the panel (outside its y label),
    # so that the panels of the two rows stay aligned
    cax = ax.inset_axes([-0.42, 0.075, 0.06, 0.85])
    cb = fig.colorbar(im_z, cax=cax, ticks=ticks)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cb.set_label("observation", fontsize=17)
    cb.ax.tick_params(labelsize=14)
    # Coarse partition lines of HV (J = 2: first splits at 1/2, then 1/4, 3/4)
    for ax in (axes[0, 1], axes[1, 1]):
        for v in (0.25, 0.5, 0.75):
            ax.axvline(v, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
            ax.axhline(v, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
    for ax in axes.flat:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=14, labelleft=False, labelbottom=False)
    # Outer edges: left of each row, bottom of each column
    for ax in (axes[0, 0], axes[1, 0]):
        ax.tick_params(labelleft=True)
        ax.set_ylabel("$s_2$", fontsize=20)
    for ax in axes[1]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("$s_1$", fontsize=20)
    cb = fig.colorbar(im_top, ax=axes[0], shrink=0.85)
    cb.set_label("posterior mean", fontsize=17)
    cb.ax.tick_params(labelsize=14)
    cb = fig.colorbar(im_diff, ax=axes[1], shrink=0.85)
    cb.set_label("difference", fontsize=17)
    cb.ax.tick_params(labelsize=14)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    side = int(round(np.sqrt(args.n)))
    size = args.conditioning_size
    for model in args.models.split(","):
        data_dir = Path(f"simulation_{model}")
        fields = pd.read_csv(data_dir / f"fields_seed{args.dataset}_n{args.n}.csv")
        if f"hv_{size}" not in fields:
            raise SystemExit(f"run export_spatial_fields.R with "
                             f"--conditioning_sizes={size} for {model}")
        # achieved HV conditioning-set size, written by export_spatial_fields.R
        hv_m = int(pd.read_csv(data_dir / f"fields_seed{args.dataset}_n{args.n}_hv_m.csv")
                   .query("requested == @size")["hv_m"].iloc[0])
        (rho, nn_k), = calibrate_rho_values(fields[["x1", "x2"]].to_numpy(),
                                            [size])
        # label NNV by its median conditioning-set size m = median_i |K_i| - 1
        # (s_i not counted), as for HV (Section 3.1)
        sp = np.unique(maxmin_cpp_ancestor(fields[["x1", "x2"]].to_numpy(),
                                           np.empty((0, 2)), rho).sparsity,
                       axis=1)
        counts = np.bincount(sp[1, sp[0] != sp[1]], minlength=len(fields))
        nn_avg, nn_max = counts.mean(), int(counts.max())
        nn_k = int(np.median(counts))
        fields["nn"] = nn_triple_mean(fields, model, rho)
        rmse = {k: np.sqrt(np.mean((fields[c] - fields["exact"]) ** 2))
                for k, c in [("HV", f"hv_{size}"), ("NN-triple", "nn")]}
        print(f"{model}: HV m = {hv_m}, NN-triple rho = {rho:.3f} (median m = {nn_k}, mean {nn_avg:.1f}, max {nn_max}); "
              f"RMS difference from exact: HV {rmse['HV']:.4f}, "
              f"NN-triple {rmse['NN-triple']:.4f}")
        out = args.figure_dir / f"spatial-maps-{model}.pdf"
        plot_model(fields, side, size, hv_m, nn_k, out, model)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
