"""
Section 3.6: filtering at large n on the CPU, in the high-dimensional
advection-diffusion setting of Jurek and Katzfuss (2022, Sec. 6.4): a 300 x
300 grid, diffusion 1e-7, advection 1e-3, Q = Sigma_{1|0} Matern with
variance 0.2, range 0.15 and smoothness 0.5, 10% of the locations observed
at each time, Gaussian noise variance 0.05.

NNV (NN-triple), HV and LR run in the same code (large_n.py); each time step
consists of the forecast (F = H L_{t-1} and the entries of F F^T + Q on the
conditioning sets, Proposition 3), Stage 1, the update and Stages 2-3. Each
method records the time per step, and the log score of its filtering
distribution N(nu_t, L_t L_t^T) at the true state and the RMSPE.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from mirs_sparse import stage1_prior, stage1_forecast, advection_diffusion_matrix
import large_n as ln
from large_n_spatial import warm_up

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, default=300)
    p.add_argument("--models", default=",".join(ln.DATA_MODELS))
    p.add_argument("--n-datasets", type=int, default=5)
    p.add_argument("--n-times", type=int, default=10)
    p.add_argument("--rhos", default="1.0,1.2,1.42,1.8,2.25,2.5")
    p.add_argument("--lr-sizes", default="",
                   help="LR conditioning sizes N, independent of HV (default: "
                        "LR with the N of each HV size).")
    p.add_argument("--hv-sizes", default="5,10,20,35,50,80")
    p.add_argument("--variance", type=float, default=0.2)
    p.add_argument("--range", type=float, default=0.15)
    p.add_argument("--smoothness", type=float, default=0.5)
    p.add_argument("--noise-variance", type=float, default=0.05)
    p.add_argument("--frac-obs", type=float, default=0.1)
    p.add_argument("--diffusion", type=float, default=1e-7)
    p.add_argument("--advection", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "large_n")
    p.add_argument("--pattern-dir", type=Path, default=ROOT / "large_n" / "patterns")
    return p.parse_args()


def simulate(args, H, sampler, n, d, model):
    """True states and data (NaN = unobserved) for T time points."""
    rng = np.random.default_rng([args.seed, d])
    xs, zs = [], []
    x = np.zeros(n)
    n_obs = round(args.frac_obs * n)
    for t in range(args.n_times):
        w, _ = sampler.sample(rng)
        x = w if t == 0 else H @ x + w
        obs = rng.choice(n, n_obs, replace=False)
        rng_m = np.random.default_rng([args.seed, d, t, ln.DATA_MODELS.index(model)])
        z = np.full(n, np.nan)
        z[obs] = ln.simulate_data(model, x[obs], rng_m, args.noise_variance)
        xs.append(x.copy())
        zs.append(z)
    return xs, zs


def main():
    args = parse_args()
    side, n = args.side, args.side ** 2
    X = ln.grid_locations(side)
    H = advection_diffusion_matrix(side, args.advection, args.diffusion)
    cov = (args.variance, args.range, args.smoothness)
    out_csv = args.out_dir / f"filter_side{side}.csv"
    warm_up()

    methods = []
    for rho in map(float, args.rhos.split(",")):
        m = ln.nn_method(X, rho)
        m.info["param"] = rho
        methods.append(m)
    lr_sizes = [int(N) for N in args.lr_sizes.split(",") if N]
    jobs = [("hv", f"hv_m{m}", m) for m in args.hv_sizes.split(",") if m]
    if lr_sizes:
        jobs += [("lr", f"lr_N{N}", N) for N in lr_sizes]
    else:
        jobs += [("lr", f"lr_m{m}", m) for m in args.hv_sizes.split(",") if m]
    for name, fname, size in jobs:
        size = int(size)
        if True:
            m = ln.read_gpvecchia_pattern(
                args.pattern_dir / f"side{side}" / fname, name.upper())
            m.info["param"] = size
            methods.append(m)
    for m in methods:
        m.Xo = np.ascontiguousarray(X[m.order])
        m.Ho = H[m.order][:, m.order].tocsr()

    sampler = ln.GridSampler(side, args.variance, args.range, args.smoothness)
    records = []
    for d in range(1, args.n_datasets + 1):
        for model in args.models.split(","):
            xs, zs = simulate(args, H, sampler, n, d, model)
            for m in methods:
                mean, L = np.zeros(n), None
                for t in range(args.n_times):
                    xo, zo = xs[t][m.order], zs[t][m.order]
                    T, last = {}, [time.perf_counter()]

                    def tick(k):
                        now = time.perf_counter()
                        T[f"time_{k}"] = now - last[0]
                        last[0] = now

                    if L is None:
                        prior_mean = np.zeros(n)
                        u = stage1_prior(m.Kptr, m.Kidx, m.Xo, *cov, m.bounds)
                    else:
                        prior_mean = m.Ho @ mean
                        F = (m.Ho @ L).tocsr()
                        F.sort_indices()
                        u = stage1_forecast(m.Kptr, m.Kidx,
                                            F.indptr.astype(np.int64),
                                            F.indices.astype(np.int64), F.data,
                                            m.Xo, *cov, m.bounds, n)
                    tick("forecast_stage1")
                    mean, V, L, diag = ln.update(m, u, prior_mean, zo, model,
                                                 args.noise_variance, timer=tick)
                    records.append({
                        "side": side, "n": n, "model": model, "dataset": d,
                        "t": t + 1, "method": m.name, "param": m.info["param"],
                        **T, "time_total": sum(T.values()),
                        "K_mean": m.cond_mean, "K_max": m.cond_max,
                        "nnz_per_row": m.nnz_per_row,
                        "ls": ln.neg_log_density_L(xo, mean, L),
                        # NNV + IC0: the Stage-2 distribution N(mean, (V V^T)^{-1})
                        "ls_V": ln.neg_log_density_V(xo, mean, V),
                        "rmspe": np.sqrt(np.mean((mean - xo) ** 2)),
                        "stage3_residual": ln.stage3_residual(V, L), **diag})
                r = pd.DataFrame(records[-args.n_times:])
                print(f"d{d} {model} {m.name} {m.info['param']}: nnz/row "
                      f"{m.nnz_per_row:.1f}, time/step {r.time_total.mean():.2f} s, "
                      f"LS {r.ls.mean():.1f}, RMSPE {r.rmspe.mean():.4f}, "
                      f"Newton {r.newton_iters.mean():.1f}", flush=True)
                pd.DataFrame(records).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
