"""
Section 4: exact Kalman filter on the full 247 x 247 TPW grid (n = 61,009),
random hold-out scheme, Section-4 model (exponential Sigma_{1|0} and Q_t,
var0 54, varQ 10, range 0.2, tau^2 0.0697, alpha/beta of the Matern fit).

Dense covariance (29.8 GB in float64), updated in place:
  forecast  P <- H P H^T + Q        (numba, one extra n x n buffer)
  update    S = P_oo + tau^2 I = L_S L_S^T     (dpotrf, in place)
            X = P_:o L_S^{-T}                 (dtrsm, in place)
            mu <- mu + X L_S^{-1}(z_o - mu_o),  P <- P - X X^T   (dsyrk)
Peak memory ~60 GB.

Writes <out>/fields_random_exact.npz (mean incl. trend, sd; float32, grid
order) and <out>/exact_random.csv (log-likelihood, times per step).

  python mirs_exact_full.py [--window r0 c0 s]   # window: check vs mirs_speckle
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sl
from scipy.linalg import blas, lapack
from numba import njit, prange
from mirs_estimate7 import load, evolution, SCALE, LOG2PI

ROOT = Path(__file__).resolve().parents[2]
THETA = (54.0, 10.0, 0.2, 0.5, 0.0696949381771818, 4.366135371198055e-08,
         0.0001599539794219)


@njit(parallel=True, cache=True)
def fill_cov(P, X, variance, range_):
    n = X.shape[0]
    for i in prange(n):
        for j in range(n):
            d = np.sqrt((X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2)
            P[i, j] = variance * np.exp(-d / range_)


@njit(parallel=True, cache=True)
def left_mult(Hp, Hi, Hv, P, A):
    """A = H P (H in CSR)."""
    n = P.shape[0]
    for i in prange(n):
        for j in range(n):
            A[i, j] = 0.0
        for q in range(Hp[i], Hp[i + 1]):
            k = Hi[q]
            h = Hv[q]
            for j in range(n):
                A[i, j] += h * P[k, j]


@njit(parallel=True, cache=True)
def right_mult_add_q(Hp, Hi, Hv, A, P, X, varQ, range_):
    """P = A H^T + varQ exp(-d / range)."""
    n = A.shape[0]
    for i in prange(n):
        for j in range(n):
            s = 0.0
            for q in range(Hp[j], Hp[j + 1]):
                s += Hv[q] * A[i, Hi[q]]
            d = np.sqrt((X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2)
            P[i, j] = s + varQ * np.exp(-d / range_)


@njit(parallel=True, cache=True)
def upper_to_lower(P):
    n = P.shape[0]
    for i in prange(n):
        for j in range(i):
            P[i, j] = P[j, i]


@njit(parallel=True, cache=True)
def gather_rows(P, o, out):
    for a in prange(o.size):
        out[a, :] = P[o[a], :]


def exact_filter(side, G, an, theta, log=print):
    var0, varQ, range_, nu, noise, dif, adv = theta
    assert nu == 0.5
    n = G.shape[0]
    X = np.ascontiguousarray(G / SCALE, dtype=np.float64)
    H = evolution(side, G, np.arange(n), dif, adv).tocsr()
    H.sort_indices()
    Hp, Hi, Hv = H.indptr.astype(np.int64), H.indices.astype(np.int64), H.data
    P = np.empty((n, n))
    fill_cov(P, X, var0, range_)
    mu = np.zeros(n)
    means, sds, recs = [], [], []
    for t in range(an.shape[0]):
        t0 = time.perf_counter()
        if t > 0:
            A = np.empty((n, n))
            left_mult(Hp, Hi, Hv, P, A)
            right_mult_add_q(Hp, Hi, Hv, A, P, X, varQ, range_)
            del A
            mu = H @ mu
        t1 = time.perf_counter()
        z = an[t]
        o = np.flatnonzero(~np.isnan(z))
        k = o.size
        if k == 0:
            var = np.diag(P).copy()
            means.append(mu.copy())
            sds.append(np.sqrt(np.maximum(var, 0)))
            recs.append({"t": t + 1, "n_obs": 0, "loglik": 0.0,
                         "min_var": var.min(), "time_forecast": t1 - t0,
                         "time_chol_trsm": 0.0, "time_syrk": 0.0,
                         "time_total": time.perf_counter() - t0})
            log(f"  exact t={t + 1}: no data, forecast {t1 - t0:.0f} s")
            continue
        Pc = np.empty((k, n))
        gather_rows(P, o, Pc)                       # P_o: (C order)
        S = Pc[:, o].copy()                         # symmetric
        S[np.diag_indices(k)] += noise
        c, info = lapack.dpotrf(S.T, lower=1, overwrite_a=1, clean=1)
        assert info == 0, info
        del S
        Xt = Pc.T                                   # P_:o, F order (n x k)
        Xs = blas.dtrsm(1.0, c, Xt, side=1, lower=1, trans_a=1, overwrite_b=1)
        assert np.shares_memory(Xs, Pc)
        e = sl.solve_triangular(c, z[o] - mu[o], lower=True)
        ll = -np.sum(np.log(np.diag(c))) - 0.5 * e @ e - 0.5 * k * LOG2PI
        del c
        mu = mu + Xs @ e
        t2 = time.perf_counter()
        Pf = blas.dsyrk(-1.0, Xs, beta=1.0, c=P.T, trans=0, lower=1,
                        overwrite_c=1)              # lower of P.T = upper of P
        assert np.shares_memory(Pf, P)
        del Xs, Xt, Pc, Pf
        upper_to_lower(P)
        var = np.diag(P).copy()
        t3 = time.perf_counter()
        means.append(mu.copy())
        sds.append(np.sqrt(np.maximum(var, 0)))
        recs.append({"t": t + 1, "n_obs": k, "loglik": ll, "min_var": var.min(),
                     "time_forecast": t1 - t0, "time_chol_trsm": t2 - t1,
                     "time_syrk": t3 - t2, "time_total": t3 - t0})
        log(f"  exact t={t + 1}: n_obs {k}, loglik {ll:.3f}, min var "
            f"{var.min():.3g}, forecast {t1 - t0:.0f} s, chol+trsm {t2 - t1:.0f} s, "
            f"syrk {t3 - t2:.0f} s")
    return np.array(means), np.array(sds), recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "application_mirs")
    p.add_argument("--out", type=Path, default=ROOT / "application_mirs" / "results")
    p.add_argument("--window", type=int, nargs=3, default=None)
    p.add_argument("--n-times", type=int, default=None)
    a = p.parse_args()
    side, G, an, coef = load(a.data_dir, scheme="random",
                             window=tuple(a.window) if a.window else None)
    if a.n_times:
        an = an[:a.n_times]
    log = lambda s: print(s, flush=True)
    means, sds, recs = exact_filter(side, G, an, THETA, log)
    if a.window:
        import mirs_speckle
        ex, _, _ = mirs_speckle.exact_kf(side, G, an, THETA)
        dm = max(np.abs(e["mean"] - m).max() for e, m in zip(ex, means))
        ds = max(np.abs(np.sqrt(e["var"]) - s).max() for e, s in zip(ex, sds))
        print(f"window check vs mirs_speckle.exact_kf: max |mean diff| {dm:.2e}, "
              f"max |sd diff| {ds:.2e}")
        return
    cols, rows = G[:, 0], G[:, 1]
    trend = np.column_stack([np.ones(G.shape[0]), G / side]) @ coef
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "fields_random_exact.npz",
                        mean=(means + trend).astype(np.float32),
                        sd=sds.astype(np.float32))
    pd.DataFrame(recs).to_csv(a.out / "exact_random.csv", index=False)
    print("EXACT DONE", flush=True)


if __name__ == "__main__":
    main()
