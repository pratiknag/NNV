import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from veccs.orderings import maxmin_cpp_ancestor

from sparse_functions import (
    exponential_covariance,
    make_sparse_patterns,
    sparse_inverse_cholesky,
    spai_from_V,
)

# Same font sizes as the other figures (plot_results.py)
plt.rcParams.update({
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


# Paul Tol's "muted" qualitative palette, ordered so that neighbouring
# (overlapping) basis functions get contrasting hues: indigo, rose, teal,
# sand, wine, cyan, green, purple, olive
FUNCTION_PALETTE = ["#332288", "#CC6677", "#44AA99", "#DDCC77", "#882255",
                    "#88CCEE", "#117733", "#AA4499", "#999933"]


def function_colors(k):
    """k colors along FUNCTION_PALETTE, cycling if k exceeds its length."""
    return [FUNCTION_PALETTE[j % len(FUNCTION_PALETTE)] for j in range(k)]


def nearest_predecessor_distances(ordered_grid):
    distances = np.full(ordered_grid.size, np.inf, dtype=float)
    for index in range(1, ordered_grid.size):
        distances[index] = np.min(
            np.abs(ordered_grid[index] - ordered_grid[:index])
        )
    return distances


def select_resolution_columns(
    distances,
    ordered_grid,
    max_level,
    columns_per_level,
):
    reference_distance = distances[1]
    levels = np.zeros(distances.size, dtype=int)
    finite = np.isfinite(distances)
    levels[finite] = np.maximum(
        0,
        np.rint(np.log2(reference_distance / distances[finite])).astype(int),
    )

    selected = {}
    for level in range(max_level + 1):
        columns = np.flatnonzero(levels == level)
        columns = columns[np.argsort(ordered_grid[columns])]
        if columns.size > columns_per_level:
            positions = np.linspace(
                0,
                columns.size - 1,
                columns_per_level,
            ).round().astype(int)
            columns = columns[positions]
        selected[level] = columns.tolist()
    return selected


def sparse_counts(matrix):
    counts = torch.bincount(matrix.indices()[0], minlength=matrix.shape[0])
    column_counts = torch.bincount(matrix.indices()[1], minlength=matrix.shape[1])
    return {
        "nnz": matrix._nnz(),
        "row_mean": counts.double().mean().item(),
        "row_max": int(counts.max().item()),
        "column_mean": column_counts.double().mean().item(),
        "column_max": int(column_counts.max().item()),
    }


def gaussian_kl_from_factor(covariance, factor):
    factor_dense = factor.to_dense() if factor.is_sparse else factor
    covariance_factor = factor_dense @ factor_dense.mT
    factor_cholesky = torch.linalg.cholesky(covariance_factor)
    solved = torch.cholesky_solve(covariance, factor_cholesky)
    exact_cholesky = torch.linalg.cholesky(covariance)
    logdet_approx = 2.0 * torch.log(torch.diagonal(factor_cholesky)).sum()
    logdet_exact = 2.0 * torch.log(torch.diagonal(exact_cholesky)).sum()
    return (
        0.5
        * (
            torch.trace(solved)
            - covariance.shape[0]
            + logdet_approx
            - logdet_exact
        )
    ).item()


def make_grid(n, irregular=False, seed=12345, irregularity=2.0):
    grid = np.linspace(0.0, 1.0, n)
    if irregular:
        rng = np.random.default_rng(seed)
        random_locations = rng.uniform(0.0, 1.0, size=n - 2)
        grid[1:-1] = np.sort(random_locations**irregularity)
    return grid


def plot_basis_functions(
    grid,
    permutation,
    selected_columns,
    factor,
    output_path,
):
    factor_dense = factor.to_dense() if factor.is_sparse else factor
    nrows = len(selected_columns)
    fig, axes = plt.subplots(
        nrows + 1,
        1,
        figsize=(14.0, 1.7 * nrows + 0.7),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * nrows + [0.34]},
    )

    for level, columns in selected_columns.items():
        ax = axes[level, 0]
        if not columns:
            ax.set_ylabel(f"m={level}", rotation=0, labelpad=45, va="center")
            continue

        palette = function_colors(len(columns))
        for column, color in zip(columns, palette):
            basis = np.zeros(grid.size)
            basis[permutation] = factor_dense[:, column].cpu().numpy()
            ax.plot(
                grid,
                basis,
                color=color,
                linewidth=2.0,
                alpha=0.8,
                zorder=2,
            )

        ax.set_ylabel(f"m={level}", rotation=0, labelpad=45, va="center")
        ax.set_xlim(grid[0], grid[-1])
        ax.margins(y=0.08)
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    locations_ax = axes[-1, 0]
    locations_ax.scatter(
        grid,
        np.zeros(grid.size),
        marker="|",
        s=60,
        linewidths=1.0,
        color="0.15",
        alpha=0.8,
    )
    locations_ax.set_ylabel("sites", rotation=0, labelpad=45, va="center")
    locations_ax.set_yticks([])
    locations_ax.set_ylim(-0.8, 0.8)
    locations_ax.spines["left"].set_visible(False)
    locations_ax.spines["right"].set_visible(False)
    locations_ax.spines["top"].set_visible(False)
    locations_ax.set_xlabel("Location")
    locations_ax.set_xticks(np.linspace(grid[0], grid[-1], 6))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.12, hspace=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    n=1025,
    rho=4.0,
    range_param=0.2,
    variance=1.0,
    max_level=3,
    columns_per_level=64,
    output=Path(__file__).resolve().parents[2] / "figures" / "basis-functions.pdf",
    skip_exact=False,
    irregular_grid=False,
    grid_seed=12345,
    irregularity=2.0,
):
    if n < 3:
        raise ValueError("n must be at least 3 to define resolution levels.")
    if rho <= 0 or range_param <= 0 or variance <= 0:
        raise ValueError("rho, range_param, and variance must be positive.")
    if max_level < 0 or columns_per_level < 1:
        raise ValueError("max_level must be nonnegative and columns_per_level positive.")
    if irregularity <= 0:
        raise ValueError("irregularity must be positive.")

    grid = make_grid(
        n,
        irregular=irregular_grid,
        seed=grid_seed,
        irregularity=irregularity,
    )
    locations = grid[:, None]
    ordering = maxmin_cpp_ancestor(
        locations,
        np.empty((0, 1)),
        rho,
    )
    permutation = np.asarray(ordering.maximin_order, dtype=np.int64)
    ordered_grid = grid[permutation]
    distances = nearest_predecessor_distances(ordered_grid)
    selected_columns = select_resolution_columns(
        distances,
        ordered_grid,
        max_level,
        columns_per_level,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ordered_locations = torch.as_tensor(
        locations[permutation], dtype=dtype, device=device
    )
    covariance = exponential_covariance(
        ordered_locations,
        variance=variance,
        range_param=range_param,
    )
    sparsity, spai_pattern = make_sparse_patterns(
        ordering.sparsity,
        ordering.ancestor_set_reduced,
        n,
        dtype=dtype,
        device=device,
    )

    prior_precision_factor = sparse_inverse_cholesky(covariance, sparsity)
    prior_factor, _ = spai_from_V(prior_precision_factor, spai_pattern)

    exact_factor = None
    relative_factor_error = None
    kl_divergence = None
    if not skip_exact:
        exact_factor = torch.linalg.cholesky(covariance)
        prior_dense = prior_factor.to_dense()
        relative_factor_error = (
            torch.linalg.norm(prior_dense - exact_factor)
            / torch.linalg.norm(exact_factor)
        ).item()
        kl_divergence = gaussian_kl_from_factor(covariance, prior_factor)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_basis_functions(
        grid,
        permutation,
        selected_columns,
        prior_factor,
        output,
    )

    counts = sparse_counts(prior_factor)
    grid_type = "irregular" if irregular_grid else "regular"
    print(
        f"n={n}, grid={grid_type}, rho={rho:g}, "
        f"range={range_param:g}, device={device}"
    )
    print(f"saved figure: {output}")
    print(
        "L sparsity: "
        f"nnz={counts['nnz']}, "
        f"row nnz mean/max={counts['row_mean']:.2f}/{counts['row_max']}, "
        f"column nnz mean/max={counts['column_mean']:.2f}/{counts['column_max']}"
    )
    if relative_factor_error is not None:
        print(f"relative Frobenius factor error: {relative_factor_error:.6e}")
        print(f"KL(N(0, Sigma) || N(0, L L^T)): {kl_divergence:.6e}")
    for level, columns in selected_columns.items():
        print(f"resolution {level}: selected columns {columns}")
    return prior_factor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot prior covariance-factor basis functions (implied basis functions figure)."
    )
    parser.add_argument("--n", type=int, default=1025)
    parser.add_argument("--rho", type=float, default=4.0)
    parser.add_argument("--range-param", type=float, default=0.2)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--columns-per-level", type=int, default=64)
    parser.add_argument(
        "--irregular-grid",
        action="store_true",
        help="Use reproducible clustered locations instead of an equally spaced grid.",
    )
    parser.add_argument("--grid-seed", type=int, default=12345)
    parser.add_argument(
        "--irregularity",
        type=float,
        default=2.0,
        help="Power-law clustering strength for irregular grids; larger values cluster near 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "figures" / "basis-functions.pdf",
    )
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="Skip dense Cholesky comparison and Gaussian KL computation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        n=args.n,
        rho=args.rho,
        range_param=args.range_param,
        variance=args.variance,
        max_level=args.max_level,
        columns_per_level=args.columns_per_level,
        output=args.output,
        skip_exact=args.skip_exact,
        irregular_grid=args.irregular_grid,
        grid_seed=args.grid_seed,
        irregularity=args.irregularity,
    )