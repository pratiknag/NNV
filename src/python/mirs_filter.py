"""
Section 4: NN-triple, HV and LR filters for the MIRS total precipitable water
data, with the advection-diffusion evolution of Section 3.3 and the parameters
of mirs_estimate.py. All three filters use the sparse CPU implementation of
mirs_sparse.py and differ only in their ordering and sparsity patterns. The
filtering distributions are scored at the held-out observations.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from veccs.orderings import maxmin_cpp_ancestor
from mirs_sparse import SparseFilter, advection_diffusion_matrix, sets_to_csr
from sparse_functions import calibrate_rho_values

METHODS = ["sparse", "HV", "LR"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("application_mirs"))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--conditioning-size",
        type=int,
        default=None,
        help="Target mean conditioning size of NN-triple "
             "(default: the HV conditioning size N).",
    )
    return parser.parse_args()


def hv_lr_pattern(data_dir, name, n):
    pattern = pd.read_csv(data_dir / f"pattern_{name}.csv")
    order = pd.read_csv(data_dir / f"order_{name}.csv")["ord"].to_numpy()
    Kptr, Kidx = sets_to_csr(pattern["i"], pattern["j"], n)
    return order, Kptr, Kidx, Kptr, Kidx


def nn_pattern(X, target_K):
    [(rho, achieved_K)] = calibrate_rho_values(X, [target_K])
    ret = maxmin_cpp_ancestor(X, np.empty((0, 2)), rho)
    sparsity = np.asarray(ret.sparsity)
    ancestors = np.asarray(ret.ancestor_set_reduced)
    n = X.shape[0]
    Kptr, Kidx = sets_to_csr(sparsity[1], sparsity[0], n)
    Jptr, Jidx = sets_to_csr(ancestors[1], ancestors[0], n)
    print(f"NN-triple: rho = {rho:.4f}, floored mean conditioning size "
          f"{achieved_K}", flush=True)
    return np.asarray(ret.maximin_order), Kptr, Kidx, Jptr, Jidx, rho


def gaussian_scores(z, mean, sd):
    u = (z - mean) / sd
    crps = sd * (u * (2 * norm.cdf(u) - 1) + 2 * norm.pdf(u) - 1 / np.sqrt(np.pi))
    return {
        "rmspe": np.sqrt(np.mean((z - mean) ** 2)),
        "crps": np.mean(crps),
        "logscore": -np.mean(norm.logpdf(z, mean, sd)),
        "coverage90": np.mean(np.abs(u) <= norm.ppf(0.95)),
        # 95% prediction intervals: coverage (PICP) and mean width (MPIW)
        "picp95": np.mean(np.abs(u) <= norm.ppf(0.975)),
        "mpiw95": np.mean(2 * norm.ppf(0.975) * sd),
    }


def main():
    args = parse_args()
    methods = args.methods.split(",")
    data = np.load(args.data_dir / "data.npz")
    with open(args.data_dir / "params.json") as f:
        params = json.load(f)
    side = int(data["side"])
    n = side * side
    Y, split = data["Y"], data["split"]
    n_times = Y.shape[0]
    X = pd.read_csv(args.data_dir / "locs.csv").to_numpy()
    trend = np.column_stack([np.ones(n), X]) @ np.array(params["trend"])
    anomalies = np.where(split == 1, Y - trend, np.nan)
    H = advection_diffusion_matrix(side, params["advection"], params["diffusion"])
    cov0 = (params["variance0"], params["range"], params["smoothness"])
    covQ = (params["varianceQ"], params["range"], params["smoothness"])
    noise = params["noise_variance"]

    hv_order, *hv_sets = hv_lr_pattern(args.data_dir, "hv", n)
    N = int(np.diff(hv_sets[0]).max()) - 1
    target_K = args.conditioning_size or N

    # Keep earlier results of the methods that are not rerun
    records, pattern_info = [], []
    if (args.data_dir / "filter_scores.csv").exists():
        old = pd.read_csv(args.data_dir / "filter_scores.csv")
        records = old[~old["method"].isin(methods)].to_dict("records")
        old = pd.read_csv(args.data_dir / "patterns.csv")
        pattern_info = old[~old["method"].isin(methods)].to_dict("records")
    for method in methods:
        if method == "sparse":
            order, Kptr, Kidx, Jptr, Jidx, rho = nn_pattern(X, target_K)
        else:
            order, Kptr, Kidx, Jptr, Jidx = hv_lr_pattern(
                args.data_dir, method.lower(), n)
            rho = np.nan
        K_sizes, J_sizes = np.diff(Kptr) - 1, np.diff(Jptr)
        pattern_info.append({
            "method": method, "rho": rho, "N": N,
            "K_mean": K_sizes.mean(), "K_max": K_sizes.max(),
            "J_mean": J_sizes.mean(), "J_max": J_sizes.max(),
        })
        print(pattern_info[-1], flush=True)

        # For the short HV and LR rows, the Stage-3 Gram matrices are cheaper
        # from the columns of V, whose product V^T V is dense for LR
        filt = SparseFilter(X[order], H[order][:, order], Kptr, Kidx,
                            Jptr, Jidx, cov0, covQ,
                            gram="product" if method == "sparse" else "columns")
        means = np.empty((n_times, n), dtype=np.float32)
        sds = np.empty((n_times, n), dtype=np.float32)
        for t in range(n_times):
            timings, last = {}, [time.perf_counter()]

            def tick(name):
                now = time.perf_counter()
                timings[f"time_{name}"] = now - last[0]
                last[0] = now

            mean, var = filt.step(anomalies[t, order], noise, timer=tick)
            mean_full, var_full = np.empty(n), np.empty(n)
            mean_full[order], var_full[order] = mean, var
            means[t] = mean_full + trend
            sds[t] = np.sqrt(var_full)
            test = split[t] == 2
            scores = gaussian_scores(Y[t, test], means[t, test],
                                     np.sqrt(var_full[test] + noise))
            records.append({"method": method, "t": t + 1, "day": int(data["days"][t]),
                            "n_train": int((split[t] == 1).sum()),
                            "n_test": int(test.sum()), **scores, **timings,
                            "time_total": sum(timings.values()),
                            "cg_iters": filt.cg_iters,
                            "ic_shift": filt.ic_shift})
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in records[-1].items()}, flush=True)
        np.savez_compressed(args.data_dir / f"filter_{method}.npz",
                            mean=means, sd=sds)
        pd.DataFrame(records).to_csv(args.data_dir / "filter_scores.csv", index=False)
        pd.DataFrame(pattern_info).to_csv(args.data_dir / "patterns.csv", index=False)

    scores = pd.DataFrame(records)
    print(scores.groupby("method", sort=False)[
        ["rmspe", "crps", "logscore", "coverage90", "time_total"]].mean())


if __name__ == "__main__":
    main()
