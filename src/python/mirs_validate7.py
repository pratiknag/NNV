"""
Section 4: held-out validation of the 7-parameter NNV-likelihood estimates
(mirs_estimate7.py). Uses exactly the NNV model of the likelihood evaluation
(same rho, Stage 2 on pattern(U U^T), general-nu Matern tabulated on the grid,
H rebuilt from the estimated diffusion/advection), runs the filter once per
hold-out scheme and scores the held-out data as in Table 1 (mirs_app.py):
y_test ~ N(trend + nu_t, ||L_t,i||^2 + tau^2).

Parameters: the evaluation with the largest log-likelihood so far in
<est-dir>/estimate7_random.csv (or --theta). Writes <est-dir>/validation/
{scores_<scheme>.csv, params_used.json, table.tex}.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import large_n as ln
from mirs_filter import gaussian_scores
from mirs_estimate7 import (NAMES, LOG2PI, load, make_nnv, matern_table,
                            evolution, stage1_prior_tab, stage1_forecast_tab)

ROOT = Path(__file__).resolve().parents[2]


def run(m, side, G, anomalies, theta, Y, test, trend, fields=None):
    var0, varQ, range_, nu, noise, diffusion, advection = theta
    tab = matern_table(side, range_, nu)
    Ho = evolution(side, G, m.order, diffusion, advection)
    n = m.n
    mean, L, recs = np.zeros(n), None, []
    for t in range(anomalies.shape[0]):
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        # likelihood term (identical to mirs_estimate7.nnv_loglik)
        U = sp.csc_matrix((u, m.Kidx, m.Kptr), shape=(n, n))
        obs = ~np.isnan(z)
        r = U.T @ (mean - prior_mean)
        ll = (-0.5 * np.sum((z[obs] - mean[obs]) ** 2) / noise
              - 0.5 * obs.sum() * (np.log(noise) + LOG2PI)
              + np.sum(np.log(u[m.Kptr[1:] - 1]))
              - np.sum(np.log(V.data[V.indptr[1:] - 1])) - 0.5 * r @ r)
        var_o = np.asarray(L.multiply(L).sum(axis=1)).ravel()
        mu, var = np.empty(n), np.empty(n)
        mu[m.order], var[m.order] = mean, var_o
        if fields is not None:
            fields["mean"].append((mu + trend).astype(np.float32))
            fields["sd"].append(np.sqrt(var).astype(np.float32))
        k = test[t]
        rec = {"t": t + 1, "time_total": elapsed, "loglik": ll,
               "ic_shift": diag["ic_shift"], "cg_iters": diag["cg_iters"],
               "n_test": int(k.sum()),
               **gaussian_scores(Y[t, k], mu[k] + trend[k],
                                 np.sqrt(var[k] + noise))}
        recs.append(rec)
        print(f"  t={t + 1}: RMSPE {rec['rmspe']:.3f} LS {rec['logscore']:.3f} "
              f"PICP {rec['picp95']:.3f} ({elapsed:.1f} s)", flush=True)
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "application_mirs")
    p.add_argument("--est-dir", type=Path,
                   default=ROOT / "application_mirs" / "estimate7")
    p.add_argument("--rho", type=float, default=3.5)
    p.add_argument("--theta", type=float, nargs=7, default=None)
    p.add_argument("--schemes", default="random,gaps")
    p.add_argument("--methods", default="NNV")
    p.add_argument("--save-fields", action="store_true",
                   help="Also save filtering means (+trend) and sds.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output folder (default <est-dir>/validation).")
    p.add_argument("--hv-size", type=int, default=218,
                   help="Requested GPvecchia HV size (m=218: 172.8 nnz/row, "
                        "matching NNV rho 3.5: 172.2).")
    p.add_argument("--lr-size", type=int, default=172,
                   help="LR size N (N=172: 172.8 nnz/row).")
    a = p.parse_args()
    if a.theta:
        theta, info = np.array(a.theta), {"source": "command line"}
    else:
        ev = pd.read_csv(a.est_dir / "estimate7_random.csv")
        best = ev.loglik.idxmax()
        theta = ev.loc[best, NAMES].to_numpy(float)
        info = {"source": "estimate7_random.csv", "eval": int(best + 1),
                "n_evals": len(ev), "loglik": float(ev.loglik[best])}
    print("theta:", dict(zip(NAMES, theta)), info, flush=True)
    out = a.out or a.est_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "params_used.json").write_text(json.dumps(
        {**dict(zip(NAMES, theta.tolist())), **info, "rho": a.rho,
         "stage2": "pattern(UU^T)"}, indent=2))
    data = np.load(a.data_dir / "data.npz")
    rows = []
    for scheme in a.schemes.split(","):
        side, G, an, coef = load(a.data_dir, scheme=scheme)
        design = np.column_stack([np.ones(G.shape[0]), G / side])
        trend = design @ coef
        split = data["split"] if scheme == "random" else data["split_gaps"]
        for name in a.methods.split(","):
            if name == "NNV":
                m = make_nnv(side, G, a.rho)
            else:
                fname = (f"hv_m{a.hv_size}" if name == "HV"
                         else f"lr_N{a.lr_size}")
                m = ln.read_gpvecchia_pattern(a.data_dir / "patterns" / fname, name)
                m.Go = np.ascontiguousarray(G[m.order]).astype(np.int64)
            print(f"{scheme}: {name}, K mean {m.cond_mean:.1f}, nnz/row L "
                  f"{m.nnz_per_row:.1f}", flush=True)
            fields = {"mean": [], "sd": []} if a.save_fields else None
            recs = run(m, side, G, an, theta, data["Y"], split == 2, trend, fields)
            if fields is not None:
                np.savez_compressed(out / f"fields_{scheme}_{name}.npz",
                                    mean=np.stack(fields["mean"]),
                                    sd=np.stack(fields["sd"]))
            df = pd.DataFrame(recs).assign(scheme=scheme, method=name,
                                           nnz_per_row=m.nnz_per_row)
            df.to_csv(out / f"scores_{scheme}_{name}.csv", index=False)
            print(f"{scheme} {name}: loglik {df.loglik.sum():.3f}", flush=True)
            rows.append(df)
    # merge with the methods already scored
    s = pd.concat([pd.read_csv(f) for f in sorted(out.glob("scores_*_*.csv"))])
    g = s.groupby(["scheme", "method"])
    tab = g[["rmspe", "crps", "logscore", "picp95", "mpiw95"]].mean()
    tab["time_total"] = g.time_total.sum()
    order = [(sc, me) for sc in ["random", "gaps"] for me in ["NNV", "HV", "LR"]
             if (sc, me) in tab.index]
    tab = tab.loc[order]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"hold-out & method & RMSPE & CRPS & LS & PICP & MPIW & total time (s) \\",
             r"\midrule"]
    names = {"NNV": r"\NNV{}", "HV": "HV", "LR": "LR"}
    for k, scheme in enumerate(dict.fromkeys(tab.index.get_level_values(0))):
        if k:
            lines.append(r"\midrule")
        for j, (method, v) in enumerate(tab.loc[scheme].iterrows()):
            lines.append(f"{scheme if j == 0 else ''} & {names[method]} & "
                         f"{v.rmspe:.3f} & {v.crps:.3f} & "
                         f"{v.logscore:.3f} & {v.picp95:.3f} & {v.mpiw95:.2f} & "
                         f"{v.time_total:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table.tex").write_text("\n".join(lines) + "\n")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
