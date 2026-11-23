"""
Section 3.6: spatial-only comparison at large n on the CPU. NN-triple (NNV),
HV and LR run in the same code (large_n.py) on the same data; the exact
Laplace approximation is added for n <= 10,000 as a check.

For each grid side, data model and data set, every method records the time
of each stage, the log score -log q(x_true | y) of its approximate posterior
N(nu, L L^T) at the true field, and the RMSPE of nu. NNV also records the log
score of its Stage-2 distribution N(nu, (V V^T)^{-1}) ("NNV with exact
Stage 3"), which only needs V.

The data are written to large_n/data/side<S>/ (binary float64: x, then z)
so that src/R/large_n_gpvecchia.R can run GPvecchia on the same data.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla
from mirs_sparse import stage1_prior
import large_n as ln

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, required=True)
    p.add_argument("--models", default=",".join(ln.DATA_MODELS))
    p.add_argument("--n-datasets", type=int, default=10)
    p.add_argument("--rhos", default="1.0,1.1,1.2,1.3,1.42,1.6,1.8,2.0,2.25,2.5")
    p.add_argument("--lr-sizes", default="",
                   help="LR conditioning sizes N, independent of HV (default: "
                        "LR with the N of each HV size).")
    p.add_argument("--hv-sizes", default="5,10,15,20,25,35,50,80,120",
                   help="Requested GPvecchia conditioning sizes for HV/LR.")
    p.add_argument("--exact", action="store_true")
    p.add_argument("--out-name", default=None,
                   help="Output file name (default: spatial_side<S>.csv).")
    p.add_argument("--range", type=float, default=0.15)
    p.add_argument("--smoothness", type=float, default=0.5)
    p.add_argument("--noise-variance", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1988)
    p.add_argument("--out-dir", type=Path, default=ROOT / "large_n")
    p.add_argument("--pattern-dir", type=Path, default=ROOT / "large_n" / "patterns")
    return p.parse_args()


def warm_up():
    """Compile all numba paths on a small problem before timing."""
    X = ln.grid_locations(12)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(X.shape[0])
    m = ln.nn_method(X, 2.0)
    Xo = np.ascontiguousarray(X[m.order])
    u = stage1_prior(m.Kptr, m.Kidx, Xo, 1.0, 0.15, 0.5, m.bounds)
    import warnings
    warnings.simplefilter("ignore")
    for model in ["gaussian", "poisson"]:
        z = ln.simulate_data(model, x, rng, 0.2)
        for exact in [False, True]:
            m.exact_stage3 = exact
            try:
                nu, V, L, _ = ln.update(m, u, np.zeros(x.size), z, model, 0.2)
                ln.neg_log_density_L(x, nu, L)
            except Exception:
                pass
    warnings.resetwarnings()


def exact_laplace(P, logdetP, z, model, noise_variance, x_true,
                  max_iter=50, tol=1e-6):
    """Dense Laplace approximation with prior precision P (zero mean)."""
    loglik, score, hess = ln.likelihood(model, noise_variance)
    y = np.zeros(z.size)

    def log_post(v):
        return np.sum(loglik(v, z)) - 0.5 * v @ (P @ v)

    current = log_post(y)
    for it in range(1, max_iter + 1):
        h = hess(y, z)
        W = P.copy()
        W[np.diag_indices_from(W)] += h
        c = sla.cho_factor(W, lower=True)
        y_new = sla.cho_solve(c, h * y + score(y, z))
        if model == "gaussian":
            y = y_new
            break
        cand = log_post(y_new)
        for _ in range(30):
            if np.isfinite(cand) and cand >= current - 1e-10 * abs(current):
                break
            y_new = (y + y_new) / 2
            cand = log_post(y_new)
        delta = np.max(np.abs(y_new - y))
        y, current = y_new, cand
        if delta < tol:
            break
    W = P.copy()
    W[np.diag_indices_from(W)] += hess(y, z)
    R = np.linalg.cholesky(W)
    r = x_true - y
    ls = -np.sum(np.log(np.diag(R))) + 0.5 * r @ (W @ r) + \
        0.5 * z.size * ln.LOG2PI
    return y, ls


def main():
    args = parse_args()
    side, n = args.side, args.side ** 2
    models = args.models.split(",")
    out_csv = args.out_dir / (args.out_name or f"spatial_side{side}.csv")
    data_dir = args.out_dir / "data" / f"side{side}"
    data_dir.mkdir(parents=True, exist_ok=True)
    X = ln.grid_locations(side)
    warm_up()

    methods = []
    for rho in [float(r) for r in args.rhos.split(",") if r]:
        m = ln.nn_method(X, rho)
        m.info["param"] = rho
        methods.append(m)
    pat = args.pattern_dir / f"side{side}"
    lr_sizes = [int(N) for N in args.lr_sizes.split(",") if N]
    jobs = [("hv", f"hv_m{m}", m) for m in args.hv_sizes.split(",") if m]
    if lr_sizes:
        jobs += [("lr", f"lr_N{N}", N) for N in lr_sizes]
    else:
        jobs += [("lr", f"lr_m{m}", m) for m in args.hv_sizes.split(",") if m]
    for name, fname, size in jobs:
        size = int(size)
        if True:
            m = ln.read_gpvecchia_pattern(pat / fname, name.upper())
            if not ln.check_closed(m):
                raise RuntimeError(f"{name} m={size}: pattern not closed")
            m.info["param"] = size
            methods.append(m)
    for m in methods:
        m.Xo = np.ascontiguousarray(X[m.order])
        print(f"{m.name} {m.info['param']}: K mean {m.cond_mean:.2f}, "
              f"max {m.cond_max}, nnz/row of L {m.nnz_per_row:.1f}, "
              f"pattern {m.pattern_time:.2f} s", flush=True)

    if args.exact:
        Sigma = ln.matern_dense(X, 1.0, args.range, args.smoothness)
        c = sla.cho_factor(Sigma, lower=True)
        P = sla.cho_solve(c, np.eye(n))
        P = (P + P.T) / 2
        del Sigma

    sampler = ln.GridSampler(side, 1.0, args.range, args.smoothness)
    records = []
    for d in range(1, args.n_datasets + 1):
        x, _ = sampler.sample(np.random.default_rng([args.seed, d]))
        for model in models:
            rng = np.random.default_rng([args.seed, d, ln.DATA_MODELS.index(model)])
            z = ln.simulate_data(model, x, rng, args.noise_variance)
            np.concatenate([x, z]).tofile(data_dir / f"{model}_d{d}.bin")
            base = {"side": side, "n": n, "model": model, "dataset": d}
            if args.exact:
                t0 = time.perf_counter()
                nu, ls = exact_laplace(P, None, z, model, args.noise_variance, x)
                records.append({**base, "method": "Exact", "param": np.nan,
                                "time_total": time.perf_counter() - t0,
                                "ls": ls, "rmspe": np.sqrt(np.mean((nu - x) ** 2))})
            for m in methods:
                xo, zo = x[m.order], z[m.order]
                T, last = {}, [time.perf_counter()]

                def tick(k):
                    now = time.perf_counter()
                    T[f"time_{k}"] = now - last[0]
                    last[0] = now

                u = stage1_prior(m.Kptr, m.Kidx, m.Xo, 1.0, args.range, args.smoothness,
                                 m.bounds)
                tick("stage1")
                nu, V, L, diag = ln.update(m, u, np.zeros(n), zo, model,
                                           args.noise_variance, timer=tick)
                rec = {**base, "method": m.name, "param": m.info["param"],
                       **T, "time_total": sum(T.values()),
                       "pattern_time": m.pattern_time,
                       "K_mean": m.cond_mean, "K_max": m.cond_max,
                       "nnz_per_row": m.nnz_per_row,
                       "J_max": int(np.diff(m.Jptr).max()),
                       "ls": ln.neg_log_density_L(xo, nu, L),
                       "ls_V": ln.neg_log_density_V(xo, nu, V),
                       "rmspe": np.sqrt(np.mean((nu - xo) ** 2)),
                       "stage3_residual": ln.stage3_residual(V, L), **diag}
                records.append(rec)
            df = pd.DataFrame(records)
            df.to_csv(out_csv, index=False)
            last_rows = df[(df.dataset == d) & (df.model == model)]
            print(last_rows[["method", "param", "nnz_per_row", "time_total",
                             "ls", "rmspe"]].round(3).to_string(index=False),
                  flush=True)


if __name__ == "__main__":
    main()
