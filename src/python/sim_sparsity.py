"""
Sections 3.2-3.3 at n = 1,600, with all methods matched on the average number
of nonzeros per row of the posterior covariance factor L (instead of the
conditioning-set size), at several sparsity levels. Uses the simulated data
sets of run_spatial_simulation.sh and run_filtering_simulation.sh and the
implementation of large_n.py for NN-triple (NNV), HV and LR; the exact
(Laplace) posterior and Kalman filter are computed with dense matrices.

Spatial-only, for NNV also:
  * NNV with exact Stage 3: log density of N(nu, (V V^T)^{-1});
  * Stage 1 only: exact Laplace approximation with prior precision U U^T.

Scores: dLS = -log q(x_true | y) + log p_exact(x_true | y) and
RRMSPE = RMSPE(method) / RMSPE(exact).
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
from mirs_sparse import stage1_prior, stage1_forecast
import large_n as ln
from large_n_spatial import warm_up

ROOT = Path(__file__).resolve().parents[2]
# Sparsity levels: NN-triple rho, HV requested m and LR N with matched
# nonzeros per row of L (about 20, 40, 60, 80 and 100; src/R/match_hv_nnz.R)
LEVELS = [(1.35, 29, 20), (2.1, 58, 41), (2.8, 87, 60), (3.5, 124, 82),
          (4.0, 150, 99), (4.6, 181, 122)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["spatial", "filter"], required=True)
    p.add_argument("--models", default=",".join(ln.DATA_MODELS))
    p.add_argument("--n-datasets", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results_sparsity")
    p.add_argument("--levels", default="",
                   help="Comma-separated levels (1-based) to run (default: all).")
    p.add_argument("--suffix", default="",
                   help="Suffix of the output files, e.g. _level6.")
    return p.parse_args()


def methods_for(X, pattern_dir, levels=None):
    out = []
    for level, (rho, m_hv, n_lr) in enumerate(LEVELS, start=1):
        if levels and level not in levels:
            continue
        nn = ln.nn_method(X, rho)
        hv = ln.read_gpvecchia_pattern(pattern_dir / f"hv_m{m_hv}", "HV")
        lr = ln.read_gpvecchia_pattern(pattern_dir / f"lr_N{n_lr}", "LR")
        for m, param in [(nn, rho), (hv, m_hv), (lr, n_lr)]:
            m.info.update(level=level, param=param)
            m.Xo = np.ascontiguousarray(X[m.order])
            out.append(m)
    return out


def dense_laplace(P, prior_mean, z, model, noise_variance, max_iter=50, tol=1e-6):
    """Mode and posterior precision for prior N(prior_mean, P^{-1})."""
    loglik, score, hess = ln.likelihood(model, noise_variance)
    obs = ~np.isnan(z)
    z0 = np.where(obs, z, 0.0)
    H = lambda v: np.where(obs, hess(v, z0), 0.0)
    S = lambda v: np.where(obs, score(v, z0), 0.0)

    def log_post(v):
        r = v - prior_mean
        return np.sum(loglik(v, z0)[obs]) - 0.5 * r @ (P @ r)

    Pmu = P @ prior_mean
    y = prior_mean.copy()
    current = log_post(y)
    for _ in range(max_iter):
        h = H(y)
        W = P.copy()
        W[np.diag_indices_from(W)] += h
        y_new = sla.cho_solve(sla.cho_factor(W, lower=True), Pmu + h * y + S(y))
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
    W[np.diag_indices_from(W)] += H(y)
    return y, W


def neg_log_density_W(x, mean, W):
    R = np.linalg.cholesky(W)
    r = x - mean
    return -np.sum(np.log(np.diag(R))) + 0.5 * r @ (W @ r) + \
        0.5 * x.size * ln.LOG2PI


def spatial(args):
    pattern_dir = ROOT / "large_n" / "patterns" / "side40"
    records = []
    for model in args.models.split(","):
        noise = 0.2
        for d in range(1, args.n_datasets + 1):
            df = pd.read_csv(ROOT / f"simulation_{model}" /
                             f"{model}_seed{d}_domain1_dim2_n1600_smooth0.5_range0.15.csv")
            X = df[["x1", "x2"]].to_numpy()
            x, z = df["y"].to_numpy(), df["z"].to_numpy()
            n = x.size
            if d == 1 and model == args.models.split(",")[0]:
                methods = methods_for(X, pattern_dir, args.level_list)
                P = np.linalg.inv(ln.matern_dense(X, 1.0, 0.15, 0.5))
                P = (P + P.T) / 2
            mu_e, W_e = dense_laplace(P, np.zeros(n), z, model, noise)
            ls_e = neg_log_density_W(x, mu_e, W_e)
            rmspe_e = math.sqrt(np.mean((mu_e - x) ** 2))
            base = {"model": model, "dataset": d}
            records.append({**base, "method": "Exact", "dLS": 0.0, "RRMSPE": 1.0})
            for m in methods:
                xo, zo = x[m.order], z[m.order]
                u = stage1_prior(m.Kptr, m.Kidx, m.Xo, 1.0, 0.15, 0.5, m.bounds)
                nu, V, L, diag = ln.update(m, u, np.zeros(n), zo, model, noise)
                rec = {**base, "level": m.info["level"], "param": m.info["param"],
                       "nnz_per_row": m.nnz_per_row, "K_mean": m.cond_mean,
                       "K_max": m.cond_max,
                       "RRMSPE": math.sqrt(np.mean((nu - xo) ** 2)) / rmspe_e,
                       "newton_iters": diag["newton_iters"],
                       "converged": diag["converged"]}
                records.append({**rec, "method": m.name,
                                "dLS": ln.neg_log_density_L(xo, nu, L) - ls_e})
                if m.name == "NNV":
                    records.append({**rec, "method": "NNV-exact-S3",
                                    "dLS": ln.neg_log_density_V(xo, nu, V) - ls_e})
                    U = sp.csc_matrix((u, m.Kidx, m.Kptr), shape=(n, n)).toarray()
                    mu1, W1 = dense_laplace(U @ U.T, np.zeros(n), zo, model, noise)
                    records.append({**rec, "method": "NNV-stage1",
                                    "dLS": neg_log_density_W(xo, mu1, W1) - ls_e,
                                    "RRMSPE": math.sqrt(np.mean((mu1 - xo) ** 2)) / rmspe_e})
            print(model, d, flush=True)
            pd.DataFrame(records).to_csv(args.out_dir / f"spatial{args.suffix}.csv", index=False)


def filtering(args):
    pattern_dir = ROOT / "large_n" / "patterns" / "side40"
    n, T, noise = 1600, 20, 0.25
    records = []
    for model in args.models.split(","):
        data_dir = ROOT / f"filtering_{model}"
        ev = pd.read_csv(data_dir / "evolution_n1600.csv")
        H = sp.csr_matrix((ev["x"], (ev["i"], ev["j"])), shape=(n, n))
        Hd = H.toarray()
        for d in range(1, args.n_datasets + 1):
            df = pd.read_csv(data_dir / f"{model}_seed{d}_n1600_T20_range0.15.csv")
            X = df.loc[df.t == 1, ["x1", "x2"]].to_numpy()
            xs = [df.loc[df.t == t, "y"].to_numpy() for t in range(1, T + 1)]
            zs = [df.loc[df.t == t, "z"].to_numpy() for t in range(1, T + 1)]
            if d == 1 and model == args.models.split(",")[0]:
                methods = methods_for(X, pattern_dir, args.level_list)
                Q = ln.matern_dense(X, 1.0, 0.15, 0.5)
                for m in methods:
                    m.Ho = H[m.order][:, m.order].tocsr()
            # Exact (Laplace) Kalman filter
            exact = []
            mean, C = np.zeros(n), Q.copy()
            for t in range(T):
                if t > 0:
                    mean, C = Hd @ mean, Hd @ C @ Hd.T + Q
                P = np.linalg.inv(C)
                mean, W = dense_laplace((P + P.T) / 2, mean, zs[t], model, noise)
                exact.append((neg_log_density_W(xs[t], mean, W),
                              math.sqrt(np.mean((mean - xs[t]) ** 2))))
                C = np.linalg.inv(W)
                C = (C + C.T) / 2
            for m in methods:
                mean, L = np.zeros(n), None
                for t in range(T):
                    xo, zo = xs[t][m.order], zs[t][m.order]
                    if L is None:
                        prior_mean = np.zeros(n)
                        u = stage1_prior(m.Kptr, m.Kidx, m.Xo, 1.0, 0.15, 0.5, m.bounds)
                    else:
                        prior_mean = m.Ho @ mean
                        F = (m.Ho @ L).tocsr()
                        F.sort_indices()
                        u = stage1_forecast(m.Kptr, m.Kidx, F.indptr.astype(np.int64),
                                            F.indices.astype(np.int64), F.data,
                                            m.Xo, 1.0, 0.15, 0.5, m.bounds, n)
                    mean, V, L, diag = ln.update(m, u, prior_mean, zo, model, noise)
                    records.append({
                        "model": model, "dataset": d, "t": t + 1, "method": m.name,
                        "level": m.info["level"], "param": m.info["param"],
                        "nnz_per_row": m.nnz_per_row,
                        "dLS": ln.neg_log_density_L(xo, mean, L) - exact[t][0],
                        "RRMSPE": math.sqrt(np.mean((mean - xo) ** 2)) / exact[t][1],
                        "newton_iters": diag["newton_iters"],
                        "converged": diag["converged"]})
            print(model, d, flush=True)
            pd.DataFrame(records).to_csv(args.out_dir / f"filter{args.suffix}.csv", index=False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.level_list = [int(v) for v in args.levels.split(",") if v]
    warm_up()
    spatial(args) if args.study == "spatial" else filtering(args)


if __name__ == "__main__":
    main()
