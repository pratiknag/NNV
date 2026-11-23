"""
Section 4: maximum-likelihood estimation of all seven parameters of the
application model with the NNV filtering likelihood of Section 2 (paragraph
"Parameter inference"), on the full 247 x 247 grid (no subwindows):

    sigma_0^2  variance of Sigma_{1|0}           (Matern)
    sigma_Q^2  variance of Q_t                   (Matern, same range/smoothness)
    lambda     shared Matern range (GPvecchia::MaternFun parameterization,
               C(d) = sigma^2 2^{1-nu}/Gamma(nu) (d/lambda)^nu K_nu(d/lambda))
    nu         shared Matern smoothness
    tau^2      noise variance
    alpha      diffusion coefficient of H
    beta       advection coefficient of H

Both Sigma_{1|0} and Q_t are Matern with general smoothness nu. Because all
locations are on the grid, the covariance only depends on the integer
squared offset k = a^2 + b^2 (grid cells), and we tabulate the Matern
correlation exactly (scipy.special.kv) for all k at every evaluation.

Usage (from src/python):
  python mirs_estimate7.py --mode=check     # likelihood vs exact KF, small window
  python mirs_estimate7.py --mode=estimate --rho=3.5 --out-dir=...
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.special as sc
from scipy.optimize import minimize
from numba import njit, prange
import large_n as ln
from mirs_sparse import _stage1_column, advection_diffusion_matrix

ROOT = Path(__file__).resolve().parents[2]
LOG2PI = np.log(2 * np.pi)
SCALE = 247  # grid spacing 1/247 (coordinates = grid indices / 247)
NAMES = ["var0", "varQ", "range", "smoothness", "noise", "diffusion", "advection"]


# ============================================================
# Matern with general smoothness, tabulated on the grid offsets
# ============================================================

def matern_corr(d, range_, nu):
    """Matern correlation (GPvecchia::MaternFun parameterization)."""
    r = np.asarray(d, dtype=np.float64) / range_
    out = np.ones_like(r)
    pos = r > 0
    rp = r[pos]
    if nu == 0.5:
        out[pos] = np.exp(-rp)
    else:
        logc = (1 - nu) * np.log(2) - sc.gammaln(nu)
        # kve(nu, r) = K_nu(r) e^r, stable for large r
        out[pos] = np.exp(logc + nu * np.log(rp) - rp) * sc.kve(nu, rp)
    out[~np.isfinite(out)] = 0.0
    return out


def matern_table(side, range_, nu, scale=SCALE):
    """tab[k] = correlation at distance sqrt(k) / scale, k = a^2 + b^2
    (grid spacing 1 / scale; side = number of cells per grid row)."""
    k = np.arange(2 * (side - 1) ** 2 + 1)
    return matern_corr(np.sqrt(k) / scale, range_, nu)


@njit(parallel=True, cache=True)
def stage1_prior_tab(ptr, idx, G, variance, tab, bounds):
    vals = np.empty(idx.size)
    for c in prange(bounds.size - 1):
        for i in range(bounds[c], bounds[c + 1]):
            K = idx[ptr[i]:ptr[i + 1]]
            m = K.size
            S = np.empty((m, m))
            for a in range(m):
                for b in range(a + 1):
                    da = G[K[a], 0] - G[K[b], 0]
                    db = G[K[a], 1] - G[K[b], 1]
                    S[a, b] = variance * tab[da * da + db * db]
                    S[b, a] = S[a, b]
            _stage1_column(S, vals[ptr[i]:ptr[i + 1]])
    return vals


@njit(parallel=True, cache=True)
def stage1_forecast_tab(ptr, idx, Fptr, Fidx, Fval, G, variance, tab,
                        bounds, n):
    """Vecchia factor of F F^T + Q on the sets (Proposition 4)."""
    vals = np.empty(idx.size)
    for c in prange(bounds.size - 1):
        work = np.zeros(n)
        for i in range(bounds[c], bounds[c + 1]):
            K = idx[ptr[i]:ptr[i + 1]]
            m = K.size
            S = np.empty((m, m))
            for a in range(m):
                ka = K[a]
                for q in range(Fptr[ka], Fptr[ka + 1]):
                    work[Fidx[q]] = Fval[q]
                for b in range(a + 1):
                    kb = K[b]
                    s = 0.0
                    for q in range(Fptr[kb], Fptr[kb + 1]):
                        s += work[Fidx[q]] * Fval[q]
                    da = G[ka, 0] - G[kb, 0]
                    db = G[ka, 1] - G[kb, 1]
                    S[a, b] = s + variance * tab[da * da + db * db]
                    S[b, a] = S[a, b]
                for q in range(Fptr[ka], Fptr[ka + 1]):
                    work[Fidx[q]] = 0.0
            _stage1_column(S, vals[ptr[i]:ptr[i + 1]])
    return vals


# ============================================================
# Data and method
# ============================================================

def load(data_dir, scheme="random", window=None):
    """Training anomalies (NaN = unobserved or held out); optional square
    window (r0, c0, size) of the grid."""
    data = np.load(data_dir / "data.npz")
    side = int(data["side"])
    split = data["split"] if scheme == "random" else data["split_gaps"]
    Y = data["Y"]
    cols, rows = np.meshgrid(np.arange(side), np.arange(side))
    G = np.column_stack([cols.ravel(), rows.ravel()])      # (col, row)
    design = np.column_stack([np.ones(G.shape[0]), G / side])
    train = split == 1
    coef, *_ = np.linalg.lstsq(np.concatenate([design[m] for m in train]),
                               Y[train], rcond=None)
    anomalies = np.where(train, Y - design @ coef, np.nan)
    if window is not None:
        r0, c0, s = window
        keep = ((G[:, 1] >= r0) & (G[:, 1] < r0 + s)
                & (G[:, 0] >= c0) & (G[:, 0] < c0 + s))
        anomalies, G = anomalies[:, keep], G[keep] - [c0, r0]
        side = s
    return side, G, anomalies, coef


def make_nnv(side, G, rho, stage2_uut=True):
    X = G / SCALE
    m = ln.nn_method(np.ascontiguousarray(X), rho)
    if stage2_uut:
        ln.add_stage2_uut_pattern(m)
    m.Go = np.ascontiguousarray(G[m.order]).astype(np.int64)
    return m


def advection_diffusion(side, advection, diffusion, scale=SCALE):
    """advection_diffusion_matrix of mirs_sparse.py (cell k = row * side +
    col) on a side x side (sub)grid with spacing 1 / scale."""
    n = side * side
    dx = 1.0 / scale
    D = diffusion / dx ** 2
    c1 = 1 + 4 * D - 2 * advection / dx
    c2 = -D + advection / dx
    c3 = -D
    below = np.full(n - 1, c2)
    above = np.full(n - 1, c3)
    below[side * np.arange(1, side) - 1] = 0.0
    above[side * np.arange(1, side) - 1] = 0.0
    return sp.diags([np.full(n - side, c2), below, np.full(n, c1), above,
                     np.full(n - side, c3)], [-side, -1, 0, 1, side],
                    shape=(n, n), format="csr")


def evolution(side, G, order, diffusion, advection):
    """H on the grid (cell k = row * side + col), permuted to the ordering."""
    H = advection_diffusion(side, advection, diffusion)
    cell = G[:, 1] * side + G[:, 0]
    return H[cell[order]][:, cell[order]].tocsr()


# ============================================================
# Likelihood
# ============================================================

def nnv_loglik(m, side, G, anomalies, theta, exact_logdet=False, n_times=None):
    """Approximate filtering log-likelihood (Section 2) of one NNV run.
    theta = (var0, varQ, range, nu, tau2, alpha, beta). If exact_logdet, the
    log-determinant of W_t = U U^T + D/tau^2 is also computed exactly by a
    sparse LU (for checking the IC0 term sum log V_ii)."""
    var0, varQ, range_, nu, noise, diffusion, advection = theta
    tab = matern_table(side, range_, nu)
    Ho = evolution(side, G, m.order, diffusion, advection)
    n = m.n
    n_times = n_times or anomalies.shape[0]
    mean, L = np.zeros(n), None
    total, recs = 0.0, []
    for t in range(n_times):
        z = anomalies[t, m.order]
        if L is None:
            prior_mean = np.zeros(n)
            u = stage1_prior_tab(m.Kptr, m.Kidx, m.Go, var0, tab, m.bounds)
        else:
            prior_mean = Ho @ mean
            F = (Ho @ L).tocsr()
            F.sort_indices()
            u = stage1_forecast_tab(m.Kptr, m.Kidx, F.indptr.astype(np.int64),
                                    F.indices.astype(np.int64), F.data,
                                    m.Go, varQ, tab, m.bounds, n)
        mean, V, L, diag = ln.update(m, u, prior_mean, z, "gaussian", noise)
        U = sp.csc_matrix((u, m.Kidx, m.Kptr), shape=(n, n))
        obs = ~np.isnan(z)
        r = U.T @ (mean - prior_mean)
        logdetU = np.sum(np.log(u[m.Kptr[1:] - 1]))
        logdetV = np.sum(np.log(V.data[V.indptr[1:] - 1]))
        base = (-0.5 * np.sum((z[obs] - mean[obs]) ** 2) / noise
                - 0.5 * obs.sum() * (np.log(noise) + LOG2PI)
                + logdetU - 0.5 * r @ r)
        rec = {"t": t + 1, "ll": base - logdetV, "ic_shift": diag["ic_shift"],
               "cg_iters": diag["cg_iters"]}
        if exact_logdet:
            W = (U @ U.T + sp.diags(obs / noise)).tocsc()
            lu = sp.linalg.splu(W, permc_spec="MMD_AT_PLUS_A",
                                options={"SymmetricMode": True})
            rec["logdetV_exact"] = 0.5 * np.sum(np.log(np.abs(lu.U.diagonal())))
            rec["logdetV_ic0"] = logdetV
            rec["ll_exactdet"] = base - rec["logdetV_exact"]
        total += rec["ll"]
        recs.append(rec)
    return total, recs


def exact_kf_loglik(side, G, anomalies, theta, n_times=None):
    """Exact Kalman-filter log-likelihood (dense; small windows only)."""
    var0, varQ, range_, nu, noise, diffusion, advection = theta
    d = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(-1)) / SCALE
    C = matern_corr(d, range_, nu)
    H = evolution(side, G, np.arange(G.shape[0]), diffusion, advection).toarray()
    n_times = n_times or anomalies.shape[0]
    mu, P = np.zeros(G.shape[0]), var0 * C
    total, out = 0.0, []
    for t in range(n_times):
        if t > 0:
            mu, P = H @ mu, H @ P @ H.T + varQ * C
        z = anomalies[t]
        o = ~np.isnan(z)
        S = P[np.ix_(o, o)] + noise * np.eye(o.sum())
        cS = np.linalg.cholesky(S)
        e = np.linalg.solve(cS, z[o] - mu[o])
        ll = -np.sum(np.log(np.diag(cS))) - 0.5 * e @ e - 0.5 * o.sum() * LOG2PI
        K = np.linalg.solve(cS.T, np.linalg.solve(cS, P[o])).T   # P_:o S^{-1}
        mu = mu + K @ (z[o] - mu[o])
        P = P - K @ P[o]
        P = (P + P.T) / 2
        total += ll
        out.append(ll)
    return total, out


# ============================================================
# Modes
# ============================================================

def to_theta(x):
    """Unconstrained x -> theta (advection in units of 1e-3)."""
    return np.array([np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), np.exp(x[3]),
                     np.exp(x[4]), np.exp(x[5]), 1e-3 * x[6]])


def from_theta(th):
    return np.array([np.log(th[0]), np.log(th[1]), np.log(th[2]),
                     np.log(th[3]), np.log(th[4]), np.log(th[5]), th[6] / 1e-3])


def check(args):
    """NNV likelihood vs the exact KF likelihood in a 50 x 50 window."""
    theta = np.array(args.theta) if args.theta else np.array(
        [44.8, 8.16, 0.0155, 1.5, 0.297, 1e-7, 1e-3])
    side, G, an, _ = load(args.data_dir, window=(100, 100, args.window))
    exact, per_t = exact_kf_loglik(side, G, an, theta)
    print(f"window {args.window}x{args.window}, theta {theta}")
    print(f"exact KF loglik {exact:.3f}")
    for rho in args.check_rhos:
        m = make_nnv(side, G, rho, stage2_uut=not args.stage2_srho)
        ll, recs = nnv_loglik(m, side, G, an, theta, exact_logdet=True)
        r = pd.DataFrame(recs)
        print(f"rho {rho}: K mean {m.cond_mean:.1f}: NNV loglik {ll:.3f} "
              f"(diff {ll - exact:+.3f}); with exact Stage-2 logdet "
              f"{r.ll_exactdet.sum():.3f} (diff {r.ll_exactdet.sum() - exact:+.3f})",
              flush=True)


def full_check(args):
    """On the full grid: IC0 log det vs exact (sparse LU) at the start."""
    theta = np.array(args.theta)
    side, G, an, _ = load(args.data_dir)
    m = make_nnv(side, G, args.rho, stage2_uut=not args.stage2_srho)
    t0 = time.perf_counter()
    ll, recs = nnv_loglik(m, side, G, an, theta, exact_logdet=args.exact_logdet,
                          n_times=args.n_times)
    print(f"rho {args.rho}: K mean {m.cond_mean:.1f}, nnz/row L {m.nnz_per_row:.1f}, "
          f"time {time.perf_counter() - t0:.1f} s, loglik {ll:.3f}")
    print(pd.DataFrame(recs).to_string())


def estimate(args):
    side, G, an, coef = load(args.data_dir, scheme=args.scheme)
    m = make_nnv(side, G, args.rho, stage2_uut=not args.stage2_srho)
    print(f"NNV rho {args.rho}: K mean {m.cond_mean:.1f}, nnz/row of L "
          f"{m.nnz_per_row:.1f}, Stage-2 pattern {m.info.get('rho2')}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"estimate7_{args.scheme}.csv"
    evals = []

    def negloglik(x):
        theta = to_theta(x)
        t0 = time.perf_counter()
        try:
            ll, recs = nnv_loglik(m, side, G, an, theta)
            shift = max(r["ic_shift"] for r in recs)
        except Exception as e:
            print("failed:", theta, e, flush=True)
            ll, shift = -np.inf, np.nan
        evals.append({**dict(zip(NAMES, theta)), "loglik": ll, "ic_shift": shift,
                      "seconds": time.perf_counter() - t0})
        pd.DataFrame(evals).to_csv(log_path, index=False)
        print(f"eval {len(evals)}: " + ", ".join(
            f"{k} {v:.5g}" for k, v in zip(NAMES, theta))
            + f" | loglik {ll:.3f} ({evals[-1]['seconds']:.0f} s)", flush=True)
        return -ll if np.isfinite(ll) else 1e12

    start = np.array(args.start)
    x0 = from_theta(start)
    # Initial simplex: +0.3 in each log parameter, +0.5e-3 in the advection
    steps = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.5])
    fit = minimize(negloglik, x0, method="Nelder-Mead",
                   options={"xatol": args.xatol, "fatol": args.fatol,
                            "maxfev": args.max_evals, "adaptive": True,
                            "initial_simplex": x0 + np.vstack(
                                [np.zeros(7), np.diag(steps)])})
    theta = to_theta(fit.x)
    res = {"scheme": args.scheme, "trend": coef.tolist(), "rho": args.rho,
           "stage2": m.info.get("rho2"), "nnz_per_row": m.nnz_per_row,
           **dict(zip(NAMES, theta.tolist())), "loglik": -fit.fun,
           "evaluations": int(fit.nfev), "converged": bool(fit.success),
           "message": fit.message,
           "final_simplex_loglik": (-fit.final_simplex[1]).tolist(),
           "final_simplex": [to_theta(v).tolist() for v in fit.final_simplex[0]]}
    (args.out_dir / f"params7_{args.scheme}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "fullcheck", "estimate"], required=True)
    p.add_argument("--scheme", choices=["random", "gaps"], default="random")
    p.add_argument("--data-dir", type=Path, default=ROOT / "application_mirs")
    p.add_argument("--out-dir", type=Path, default=ROOT / "application_mirs" / "estimate7")
    p.add_argument("--rho", type=float, default=3.5)
    p.add_argument("--stage2-srho", action="store_true",
                   help="Stage 2 on S_rho instead of the pattern of U U^T.")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--check-rhos", type=float, nargs="+", default=[2.5, 3.5, 5.0])
    p.add_argument("--theta", type=float, nargs=7, default=None)
    p.add_argument("--start", type=float, nargs=7,
                   default=[44.8, 8.16, 0.0155, 1.5, 0.297, 1e-7, 1e-3])
    p.add_argument("--exact-logdet", action="store_true")
    p.add_argument("--n-times", type=int, default=None)
    p.add_argument("--max-evals", type=int, default=700)
    p.add_argument("--xatol", type=float, default=1e-3)
    p.add_argument("--fatol", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    {"check": check, "fullcheck": full_check, "estimate": estimate}[a.mode](a)
