"""
Section 3.6: implied prior basis functions of NN-triple and of hierarchical
Vecchia (HV) for a Matern covariance with smoothness 1.5 on a regular 1D grid.

NN-triple: columns of L_prior, computed by Stage 3 from the Stage-1 factor U
on the maximin ordering. Thin dashed lines: the columns of the exact
Cholesky factor of the prior covariance in the same ordering. HV: columns of L_HV = U_HV^{-T} (Stages 2 and 3 are
exact for HV), with the hierarchical partitioning of Jurek and Katzfuss (2022,
Sec. 3.1): J = 2 regions per level, the knots of a region being its first r
points in the maximin ordering that are not already knots of an ancestor
region, and all remaining points at the finest level M. The number of knots
r_m per region at level m is chosen so that HV has as many basis functions at
each plotted level as NN-triple (r_m = 2 beyond the plotted levels, as in Jurek and Katzfuss).
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from veccs.orderings import maxmin_cpp_ancestor
from basis_functions import (function_colors, nearest_predecessor_distances,
                             select_resolution_columns)
from sparse_functions import make_sparse_patterns, sparse_inverse_cholesky, spai_from_V
import scipy.sparse as sp
from mirs_sparse import (chunk_bounds, ichol0_upper, matern_matrix, sets_to_csr,
                         spai)

plt.rcParams.update({
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1025)
    parser.add_argument("--range-param", type=float, default=0.2)
    parser.add_argument("--smoothness", type=float, default=1.5)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[2] / "figures" / "basis-functions.pdf",
    )
    return parser.parse_args()


def kl_exact_to_factor(Sigma, L):
    """KL(N(0, Sigma) || N(0, L L^T)) for lower-triangular L."""
    n = Sigma.shape[0]
    A = np.linalg.solve(L, np.linalg.cholesky(Sigma))
    return 0.5 * (np.sum(A ** 2) - n + 2 * np.log(np.abs(np.diag(L))).sum()
                  - 2 * np.log(np.diag(np.linalg.cholesky(Sigma))).sum())


def nn_triple_factor(grid, rho, Sigma_fn):
    ret = maxmin_cpp_ancestor(grid[:, None], np.empty((0, 1)), rho)
    order = np.asarray(ret.maximin_order)
    ancestors = np.asarray(ret.ancestor_set_reduced)
    J = sets_to_csr(ancestors[1], ancestors[0], grid.size)
    S_rho, S_A = make_sparse_patterns(ret.sparsity, ret.ancestor_set_reduced,
                                      grid.size)
    Sigma = torch.as_tensor(Sigma_fn(grid[order]))
    U = sparse_inverse_cholesky(Sigma, S_rho)
    L, _ = spai_from_V(U, S_A)
    sizes = np.bincount(S_rho.indices()[1].numpy(), minlength=grid.size) - 1
    return order, L.to_dense().numpy(), sizes, U.to_dense().numpy(), J


def hv_factor(grid, knots_per_level, Sigma_fn):
    """
    Ordering, levels, regions and exact factor U^{-T} of HV with J = 2, with
    knots_per_level[m] knots per region at level m (the last entry for finer levels).
    """
    n = grid.size
    maximin = np.asarray(
        maxmin_cpp_ancestor(grid[:, None], np.empty((0, 1)), 1.0).maximin_order)
    rank = np.empty(n, dtype=int)
    rank[maximin] = np.arange(n)
    chosen = np.zeros(n, dtype=bool)
    order, level_of, region_of = [], [], []
    m = 0
    while not chosen.all():
        r = knots_per_level[min(m, len(knots_per_level) - 1)]
        edges = np.linspace(0, 1, 2 ** m + 1)
        region = np.minimum(np.searchsorted(edges, grid, side="right") - 1,
                            2 ** m - 1)
        remaining = np.flatnonzero(~chosen)
        # Finest level: all remaining points once no region has more than r
        last = np.bincount(region[remaining], minlength=2 ** m).max() <= r
        for j in range(2 ** m):
            candidates = remaining[region[remaining] == j]
            candidates = candidates[np.argsort(rank[candidates])]
            knots = candidates if last else candidates[:r]
            chosen[knots] = True
            order.extend(knots)
            level_of.extend([m] * knots.size)
            region_of.extend([j] * knots.size)
        m += 1
    order = np.array(order)
    level_of, region_of = np.array(level_of), np.array(region_of)
    # C_i: knots of ancestor regions, plus previous knots of the same region
    n_levels = level_of.max() + 1
    rows, cols = [], []
    for i in range(n):
        li, ji = level_of[i], region_of[i]
        ancestor = (level_of < li) & (region_of == ji >> (li - level_of))
        same = (level_of == li) & (region_of == ji) & (np.arange(n) <= i)
        for k in np.flatnonzero(ancestor | same):
            rows.append(k)
            cols.append(i)
    S = torch.sparse_coo_tensor(torch.tensor([rows, cols]),
                                torch.ones(len(rows), dtype=torch.float64),
                                (n, n)).coalesce()
    Sigma = torch.as_tensor(Sigma_fn(grid[order]))
    U = sparse_inverse_cholesky(Sigma, S).to_dense().numpy()
    L = np.linalg.inv(U).T
    sizes = np.bincount(np.array(cols), minlength=n) - 1
    return order, level_of, region_of, L, sizes, n_levels


def posterior_kl(grid, order, U, tau2, Sigma_fn, J=None):
    """
    KL from the exact posterior of x given y = x + noise (variance tau2 at
    every site) to the approximation of the filter update: Stage 1 prior
    factor U, then IC0 and SPAI on the pattern J (or exact Stages 2 and 3 if
    J is None, as for HV).
    """
    n = grid.size
    Sigma = Sigma_fn(grid[order])
    exact = Sigma - Sigma @ np.linalg.solve(Sigma + tau2 * np.eye(n), Sigma)
    U = sp.csc_matrix(U)
    W = (U @ U.T + sp.eye(n) / tau2).tocsr()
    if J is None:
        V = np.linalg.cholesky(W.toarray()[::-1, ::-1])[::-1, ::-1]
        L = np.linalg.inv(V).T
    else:
        V, _ = ichol0_upper(W)
        L = spai(V, *J, chunk_bounds(n)).toarray()
    return kl_exact_to_factor((exact + exact.T) / 2, L)


def overlap(columns, threshold=0.05):
    """Mean number of columns exceeding threshold * their peak, per location."""
    above = np.abs(columns) > threshold * np.abs(columns).max(axis=0)
    covered = above.sum(axis=1)
    return covered[covered > 0].mean()


def main():
    args = parse_args()
    grid = np.linspace(0.0, 1.0, args.n)

    def Sigma_fn(x):
        return matern_matrix(np.column_stack([x, np.zeros_like(x)]), 1.0,
                             args.range_param, args.smoothness)

    exact = Sigma_fn(grid)
    # NN-triple resolution levels depend only on the maximin ordering, not on
    # rho; HV gets the same number of basis functions per level
    maximin = np.asarray(maxmin_cpp_ancestor(
        grid[:, None], np.empty((0, 1)), 1.0).maximin_order)
    nn_counts = [len(c) for c in select_resolution_columns(
        nearest_predecessor_distances(grid[maximin]), grid[maximin],
        args.max_level, 10 ** 6).values()]
    knots = [-(-count // 2 ** m) for m, count in enumerate(nn_counts)] + [2]
    hv_order, hv_level, hv_region, L_hv, hv_sizes, hv_levels = hv_factor(
        grid, knots, Sigma_fn)
    N = int(hv_sizes.max())
    # NN-triple with the same maximum conditioning-set size
    rho = 1.0
    while True:
        nn_order, L_nn, nn_sizes, U_nn, J_nn = nn_triple_factor(
            grid, rho, Sigma_fn)
        if nn_sizes.max() >= N:
            break
        rho += 0.05

    def kl(order, L):
        return kl_exact_to_factor(exact[np.ix_(order, order)], L)

    print(f"HV: knots per region r_m = {knots[:-1]}, then 2, {hv_levels} levels, N = {N}, "
          f"mean |C_i| = {hv_sizes.mean():.2f}, KL = {kl(hv_order, L_hv):.4e}")
    print(f"NN-triple: rho = {rho:.2f}, N = {nn_sizes.max()}, "
          f"mean |C_i| = {nn_sizes.mean():.2f}, KL = {kl(nn_order, L_nn):.4e}, "
          f"KL of Stage 1 alone = {kl(nn_order, np.linalg.inv(U_nn).T):.4e}")
    # Posterior given y = x + noise, as in the update step of the filter
    U_hv = np.linalg.inv(L_hv).T
    for tau2 in (1e-4, 1e-3, 1e-2, 1e-1):
        print(f"posterior KL, tau^2 = {tau2:g}: NN-triple "
              f"{posterior_kl(grid, nn_order, U_nn, tau2, Sigma_fn, J_nn):.3e}, "
              f"HV {posterior_kl(grid, hv_order, U_hv, tau2, Sigma_fn):.3e}")

    # Columns by resolution level
    distances = nearest_predecessor_distances(grid[nn_order])
    nn_columns = select_resolution_columns(distances, grid[nn_order],
                                           args.max_level, 10 ** 6)
    hv_columns = {m: np.flatnonzero(hv_level == m).tolist()
                  for m in range(args.max_level + 1)}

    rows = args.max_level + 1
    fig, axes = plt.subplots(rows, 2, figsize=(20, 1.9 * rows + 0.6),
                             sharex=True, squeeze=False)
    for c, (name, order, L, columns) in enumerate([
            ("NNV", nn_order, L_nn, nn_columns),
            ("HV", hv_order, L_hv, hv_columns)]):
        hv = name == "HV"
        # Exact Cholesky factor of the prior covariance in the same ordering
        L_exact = np.linalg.cholesky(exact[np.ix_(order, order)])
        for m in range(rows):
            ax = axes[m, c]
            basis = np.zeros((grid.size, len(columns[m])))
            # One muted color per basis function, by the location of its point
            by_location = sorted(columns[m], key=lambda col: grid[order[col]])
            palette = function_colors(len(by_location))
            color_of = dict(zip(by_location, palette))
            for k, column in enumerate(columns[m]):
                basis[order, k] = L[:, column]
                # Exact basis function in the background: thin, dashed
                exact_basis = np.zeros(grid.size)
                exact_basis[order] = L_exact[:, column]
                ax.plot(grid, exact_basis, color=color_of[column],
                        linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
                # An NN-triple basis function is exactly zero outside the
                # rows of its column in the ancestor pattern
                support = np.flatnonzero(basis[:, k] != 0)
                shown = ((grid >= grid[support.min()])
                         & (grid <= grid[support.max()]))
                if hv:
                    # An HV basis function is zero outside its region
                    j = hv_region[column]
                    shown = (grid >= j / 2 ** m) & (grid <= (j + 1) / 2 ** m)
                ax.plot(grid[shown], basis[shown, k],
                        color=color_of[column],
                        linewidth=3.0, alpha=0.85, zorder=2)
            print(f"{name}, level {m}: {len(columns[m])} functions, "
                  f"mean overlap {overlap(basis):.2f}")
            if hv:
                # Boundaries of the HV regions at this level (k / 2^m)
                for k in range(1, 2 ** m):
                    ax.axvline(k / 2 ** m, color="0.6", linewidth=0.8, zorder=0)
            if c == 0:
                ax.set_ylabel(f"level = {m}", rotation=0, labelpad=55, va="center")
            if m == 0:
                ax.set_title(name)
            ax.set_xlim(0, 1)
            ax.margins(y=0.08)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[-1, c].set_xlabel("Location")
        axes[-1, c].set_xticks(np.linspace(0, 1, 6))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.12,
                        hspace=0.3, wspace=0.12)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
