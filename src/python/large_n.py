"""
Large-n CPU implementation of the update and filter of Algorithm 1 for
NN-triple (NNV), HV and LR, used for the timing comparisons of Section 3.6.

All three methods run in the same code on the same CPU and differ only in
their ordering and patterns:
  * K_i (Stage-1 conditioning sets, column i of U; Stage 2 runs IC0 on this
    pattern, as in Section 2.3);
  * J_i (Stage-3 pattern, row i of L).
For HV and LR, J_i = K_i and the patterns are closed (k in K_i implies
K_k subset of K_i), so IC0 is the exact Cholesky factor of W and L = V^{-T}
is computed exactly within the pattern by sparse back substitution
(Proposition 4), not by generic SPAI. For NN-triple, J_i is the reduced
ancestor set and Stage 3 is SPAI (eq. 7).

Each update computes Stage 1 once, runs the Newton iterations with
IC0-preconditioned conjugate gradients on W = U U^T + D^{-1} (matrix-vector
products with U and U^T only), and computes Stages 2 and 3 once, at the mode.

Patterns are stored as (ptr, idx): the set of i is idx[ptr[i]:ptr[i+1]],
sorted, with i itself last.
"""
import math
import time
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from mirs_sparse import (chunk_bounds, sets_to_csr, stage1_prior,
                         stage1_forecast, _spai_rows, matern)


# ============================================================
# Simulation on a regular grid by circulant embedding (exact)
# ============================================================

def grid_locations(side):
    """Locations as in expand.grid(seq(0, 1, length = side), ...) in R."""
    g = np.linspace(0.0, 1.0, side)
    x1, x2 = np.meshgrid(g, g, indexing="xy")
    return np.column_stack([x1.ravel(), x2.ravel()])


class GridSampler:
    """Exact samples of a Matern field on the side x side grid of [0, 1]^2."""

    def __init__(self, side, variance, range_param, smoothness):
        h = 1.0 / (side - 1)
        M = 2 * side
        while True:
            k = np.arange(M)
            d1 = np.minimum(k, M - k) * h
            D = np.sqrt(d1[:, None] ** 2 + d1[None, :] ** 2)
            r = D / range_param
            if smoothness == 0.5:
                base = variance * np.exp(-r)
            else:
                base = variance * (1.0 + r) * np.exp(-r)
            lam = np.real(np.fft.fft2(base))
            if lam.min() > -1e-8 * lam.max():
                break
            M *= 2
        self.side, self.M = side, M
        self.sqrt_lam = np.sqrt(np.maximum(lam, 0.0)) / M

    def sample(self, rng):
        """Two independent fields (real and imaginary parts), flattened."""
        e = rng.standard_normal(self.sqrt_lam.shape) + \
            1j * rng.standard_normal(self.sqrt_lam.shape)
        f = np.fft.fft2(self.sqrt_lam * e)[:self.side, :self.side]
        # f[row = x2 index, col = x1 index]; x1 varies fastest
        return f.real.ravel(), f.imag.ravel()


# ============================================================
# Data models
# ============================================================

DATA_MODELS = ["gaussian", "logistic", "poisson", "gamma"]


def simulate_data(model, x, rng, noise_variance, gamma_shape=2.0):
    if model == "gaussian":
        return x + math.sqrt(noise_variance) * rng.standard_normal(x.size)
    if model == "logistic":
        return (rng.random(x.size) < 1 / (1 + np.exp(-x))).astype(float)
    if model == "poisson":
        return rng.poisson(np.exp(x)).astype(float)
    if model == "gamma":
        return rng.gamma(gamma_shape, 1 / np.exp(x))
    raise ValueError(model)


def likelihood(model, noise_variance, gamma_shape=2.0):
    """Log-likelihood, score and negative Hessian, elementwise in x."""
    if model == "gaussian":
        return (lambda x, z: -0.5 * (z - x) ** 2 / noise_variance,
                lambda x, z: (z - x) / noise_variance,
                lambda x, z: np.full_like(x, 1 / noise_variance))
    if model == "logistic":
        return (lambda x, z: z * x - np.logaddexp(0, x),
                lambda x, z: z - 1 / (1 + np.exp(-x)),
                lambda x, z: (lambda p: p * (1 - p))(1 / (1 + np.exp(-x))))
    if model == "poisson":
        return (lambda x, z: z * x - np.exp(x),
                lambda x, z: z - np.exp(x),
                lambda x, z: np.exp(x))
    if model == "gamma":
        return (lambda x, z: gamma_shape * x - z * np.exp(x),
                lambda x, z: gamma_shape - z * np.exp(x),
                lambda x, z: z * np.exp(x))
    raise ValueError(model)


# ============================================================
# Stage 2: IC0 on the pattern K, column by column
# ============================================================

@njit(cache=True)
def _merge_positions(A, B, out):
    """out[a] = position of A[a] in B (sorted arrays), or -1."""
    q = 0
    for a in range(A.size):
        while q < B.size and B[q] < A[a]:
            q += 1
        out[a] = q if (q < B.size and B[q] == A[a]) else -1


@njit(parallel=True, cache=True)
def _w_on_pattern(Kptr, Kidx, Uval, Rptr, Ridx, Rpos, n_chunks):
    """
    Entries of U U^T on the pattern K (column j holds W[K_j, j]):
    W[i, j] = sum over the columns k of U that contain j (row j of U, given
    by Rptr, Ridx and the positions Rpos of j in K_k) of U[i, k] U[j, k].
    Parallel over j, in interleaved chunks for load balance.
    """
    n = Kptr.size - 1
    a = np.zeros(Kidx.size)
    for c in prange(n_chunks):
        pos = np.empty(Kptr.size, dtype=np.int64)
        for j in range(c, n, n_chunks):
            Kj = Kidx[Kptr[j]:Kptr[j + 1]]
            for q in range(Rptr[j], Rptr[j + 1]):
                k = Ridx[q]
                b = Rpos[q]
                K = Kidx[Kptr[k]:Kptr[k] + b + 1]
                u = Uval[Kptr[k]:Kptr[k] + b + 1]
                p = pos[:b + 1]
                _merge_positions(K, Kj, p)
                for cc in range(b + 1):
                    if p[cc] >= 0:
                        a[Kptr[j] + p[cc]] += u[cc] * u[b]
    return a


def w_on_pattern(Kptr, Kidx, Uval):
    n = Kptr.size - 1
    cols = np.repeat(np.arange(n), np.diff(Kptr))
    # Row structure of U: for each row j, the columns k with j in K_k and the
    # position of j within K_k
    posk = np.arange(Kidx.size) - Kptr[cols]
    R = sp.csr_matrix((posk.astype(np.float64) + 1, (Kidx, cols)), shape=(n, n))
    R.sort_indices()
    return _w_on_pattern(Kptr, Kidx, Uval, R.indptr.astype(np.int64),
                         R.indices.astype(np.int64),
                         (R.data - 1).astype(np.int64), 256)


@njit(cache=True)
def _ic0_columns(Kptr, Kidx, Wval, h, shift):
    """
    Upper-triangular V on the pattern K with W ~ V V^T, W = U U^T + diag(h)
    (Wval: U U^T on the pattern), by right-looking IC0 in the reverse
    ordering: column j of V is final once all later columns have been
    processed. Returns the values on (Kptr, Kidx) and a flag that is False
    on breakdown.
    """
    n = Kptr.size - 1
    a = Wval.copy()
    pos = np.empty(Kidx.size, dtype=np.int64)
    for i in range(n):
        a[Kptr[i + 1] - 1] = (a[Kptr[i + 1] - 1] + h[i]) * (1.0 + shift)
    for j in range(n - 1, -1, -1):
        s, e = Kptr[j], Kptr[j + 1]
        d = a[e - 1]
        if d < 1e-10:
            return a, False
        d = math.sqrt(d)
        a[e - 1] = d
        for q in range(s, e - 1):
            a[q] /= d
        K = Kidx[s:e - 1]
        v = a[s:e - 1]
        for b in range(K.size):
            i2 = K[b]
            Ki2 = Kidx[Kptr[i2]:Kptr[i2 + 1]]
            p = pos[:b + 1]
            _merge_positions(K[:b + 1], Ki2, p)
            for c in range(b + 1):
                if p[c] >= 0:
                    a[Kptr[i2] + p[c]] -= v[c] * v[b]
    return a, True


def ic0(Kptr, Kidx, Wval, h, shifts=(0.0, 1e-4, 1e-3, 1e-2, 1e-1)):
    """V (CSC values on K) with U U^T + diag(h) ~ V V^T, given U U^T on the
    pattern (w_on_pattern); diagonal shift on breakdown (Manteuffel, 1980).
    Returns values and the shift used."""
    for alpha in shifts:
        vals, ok = _ic0_columns(Kptr, Kidx, Wval, h, alpha)
        if ok:
            return vals, alpha
    raise RuntimeError("IC0 breakdown for all shifts")


# ============================================================
# Stage 3
# ============================================================

@njit(parallel=True, cache=True)
def _inverse_within_pattern(Kptr, Kidx, Vval, bounds):
    """
    Rows of L = V^{-T} on the closed pattern J_i = K_i (HV, LR): column i of
    V^{-1} by back substitution, using only the short columns of V.
    """
    vals = np.empty(Kidx.size)
    for c in prange(bounds.size - 1):
        for i in range(bounds[c], bounds[c + 1]):
            s, e = Kptr[i], Kptr[i + 1]
            K = Kidx[s:e]
            p = K.size
            r = np.zeros(p)
            r[p - 1] = 1.0
            pos = np.empty(p, dtype=np.int64)
            for b in range(p - 1, -1, -1):
                k = K[b]
                ks, ke = Kptr[k], Kptr[k + 1]
                x = r[b] / Vval[ke - 1]
                r[b] = x
                # r_j -= V_jk x_k for j in K_k, j < k (K_k subset of K_i)
                _merge_positions(Kidx[ks:ke - 1], K[:b], pos)
                for q in range(ke - 1 - ks):
                    if pos[q] >= 0:
                        r[pos[q]] -= Vval[ks + q] * x
            vals[s:e] = r
    return vals


def stage3(V, Jptr, Jidx, bounds, exact):
    """Lower-triangular L (CSR on J) with L ~ V^{-T}."""
    n = V.shape[0]
    if exact:
        Vc = V.tocsc()
        vals = _inverse_within_pattern(Jptr, Jidx, Vc.data, bounds)
    else:
        M = (V.T @ V).tocsr()
        M.sort_indices()
        vals = _spai_rows(Jptr, Jidx, M.indptr.astype(np.int64),
                          M.indices.astype(np.int64), M.data, V.diagonal(),
                          bounds, n)
    return sp.csr_matrix((vals, Jidx, Jptr), shape=(n, n))


# ============================================================
# Triangular solves and PCG
# ============================================================

@njit(cache=True)
def _solve_upper_csc(ptr, idx, val, b):
    """x = V^{-1} b for upper V in CSC with the diagonal last per column."""
    x = b.copy()
    for j in range(ptr.size - 2, -1, -1):
        x[j] /= val[ptr[j + 1] - 1]
        for q in range(ptr[j], ptr[j + 1] - 1):
            x[idx[q]] -= val[q] * x[j]
    return x


@njit(cache=True)
def _solve_lower_from_csc(ptr, idx, val, b):
    """x = V^{-T} b for upper V in CSC (row j of V^T = column j of V)."""
    x = b.copy()
    for j in range(ptr.size - 1):
        s = x[j]
        for q in range(ptr[j], ptr[j + 1] - 1):
            s -= val[q] * x[idx[q]]
        x[j] = s / val[ptr[j + 1] - 1]
    return x


@njit(cache=True)
def _solve_lower_csr(ptr, idx, val, b):
    """x = L^{-1} b for lower L in CSR with the diagonal last per row."""
    x = b.copy()
    for i in range(ptr.size - 1):
        s = x[i]
        for q in range(ptr[i], ptr[i + 1] - 1):
            s -= val[q] * x[idx[q]]
        x[i] = s / val[ptr[i + 1] - 1]
    return x


def pcg(matvec, b, Kptr, Kidx, Vval, x0, tol=1e-10, max_iter=1000):
    """Conjugate gradients for W x = b, preconditioned by (V V^T)^{-1}."""
    def precondition(r):
        # (V V^T)^{-1} r = V^{-T} (V^{-1} r)
        return _solve_lower_from_csc(Kptr, Kidx, Vval,
                                     _solve_upper_csc(Kptr, Kidx, Vval, r))
    x = x0.copy()
    r = b - matvec(x)
    threshold = tol * np.linalg.norm(b)
    if np.linalg.norm(r) <= threshold:
        return x, 0
    z = precondition(r)
    p = z.copy()
    rz = r @ z
    for it in range(1, max_iter + 1):
        Wp = matvec(p)
        alpha = rz / (p @ Wp)
        x += alpha * p
        r -= alpha * Wp
        if np.linalg.norm(r) <= threshold:
            return x, it
        z = precondition(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, max_iter


# ============================================================
# One update (spatial-only problem or one filtering time step)
# ============================================================

class Method:
    """Ordering and patterns of one method."""

    def __init__(self, name, order, Kptr, Kidx, Jptr, Jidx, exact_stage3,
                 pattern_time=np.nan, info=None):
        self.name, self.order = name, np.asarray(order)
        self.Kptr, self.Kidx = Kptr, Kidx
        self.Jptr, self.Jidx = Jptr, Jidx
        self.exact_stage3 = exact_stage3
        self.n = Kptr.size - 1
        self.bounds = chunk_bounds(self.n)
        self.pattern_time = pattern_time
        self.info = info or {}

    @property
    def nnz_per_row(self):
        """Average nonzeros per row of the posterior covariance factor L."""
        return self.Jidx.size / self.n

    @property
    def cond_mean(self):
        return self.Kidx.size / self.n - 1

    @property
    def cond_max(self):
        return int(np.diff(self.Kptr).max()) - 1


def update(method, uvals, prior_mean, z, model, noise_variance,
           gamma_shape=2.0, max_iter=50, tol=1e-6, max_halvings=30,
           timer=None):
    """
    Posterior (filtering) distribution N(nu, L L^T) given the Stage-1 factor
    U (values on K) of the prior N(prior_mean, (U U^T)^{-1}) and data z
    (NaN = unobserved). Returns nu, V (CSC), L (CSR) and diagnostics.
    """
    tick = timer or (lambda name: None)
    n = method.n
    Kptr, Kidx = method.Kptr, method.Kidx
    U = sp.csc_matrix((uvals, Kidx, Kptr), shape=(n, n))
    UT = U.T.tocsr()
    # Optional larger Stage-2 pattern (method.K2ptr, method.K2idx, containing
    # K column by column): IC0 of U U^T + D on it; Stages 1 and 3 unchanged
    K2 = getattr(method, "K2ptr", None) is not None
    if K2:
        uvals = embed_values(Kptr, Kidx, uvals, method.K2ptr, method.K2idx)
        Kptr, Kidx = method.K2ptr, method.K2idx
    loglik, score, hess = likelihood(model, noise_variance, gamma_shape)
    obs = ~np.isnan(z)
    z0 = np.where(obs, z, 0.0)

    def H(x):
        return np.where(obs, hess(x, z0), 0.0)

    def S(x):
        return np.where(obs, score(x, z0), 0.0)

    def log_post(x):
        r = UT @ (x - prior_mean)
        return np.sum(loglik(x, z0)[obs]) - 0.5 * r @ r

    t0 = time.perf_counter()
    Wval = w_on_pattern(Kptr, Kidx, uvals)
    ic0_time = time.perf_counter() - t0
    UUmu = U @ (UT @ prior_mean)
    x = prior_mean.copy()
    linear = model == "gaussian"
    converged, cg_total, shift = False, 0, 0.0
    current = log_post(x)
    it = 0
    for it in range(1, max_iter + 1):
        h = H(x)
        t0 = time.perf_counter()
        Vval, shift = ic0(Kptr, Kidx, Wval, h)
        ic0_time += time.perf_counter() - t0
        rhs = UUmu + h * x + S(x)
        x_new, cg_it = pcg(lambda v: U @ (UT @ v) + h * v, rhs,
                           Kptr, Kidx, Vval, x0=x)
        cg_total += cg_it
        if linear:
            x, converged = x_new, True
            break
        candidate = log_post(x_new)
        for _ in range(max_halvings):
            if np.isfinite(candidate) and candidate >= current - 1e-10 * abs(current):
                break
            x_new = (x + x_new) / 2
            candidate = log_post(x_new)
        delta = np.max(np.abs(x_new - x))
        x, current = x_new, candidate
        if delta < tol:
            converged = True
            break
    tick("update")
    if not linear:
        # Stage 2 at the mode, with the final pseudo-variances
        Vval, shift = ic0(Kptr, Kidx, Wval, H(x))
    V = sp.csc_matrix((Vval, Kidx, Kptr), shape=(n, n))
    tick("stage2")
    L = stage3(V, method.Jptr, method.Jidx, method.bounds,
               method.exact_stage3)
    tick("stage3")
    # "update" includes the IC0 preconditioners of the Newton iterations
    return x, V, L, {"newton_iters": it, "converged": converged,
                     "cg_iters": cg_total, "ic_shift": shift,
                     "ic0_newton_time": ic0_time}


# ============================================================
# Scores
# ============================================================

LOG2PI = math.log(2 * math.pi)


def embed_values(Kptr, Kidx, vals, K2ptr, K2idx):
    """Values on the pattern K placed into the larger pattern K2 (zeros
    elsewhere); each column of K must be contained in that of K2."""
    n = Kptr.size - 1
    out = np.zeros(K2idx.size)
    for j in range(n):
        K2 = K2idx[K2ptr[j]:K2ptr[j + 1]]
        pos = np.searchsorted(K2, Kidx[Kptr[j]:Kptr[j + 1]])
        out[K2ptr[j] + pos] = vals[Kptr[j]:Kptr[j + 1]]
    return out


def add_stage2_pattern(method, X, rho2):
    """Use the nearest-neighbor pattern S_rho2 (rho2 >= rho) on the same
    maximin ordering for Stage 2 of an NNV method: K2_i = {j <= i :
    |x_i - x_j| <= rho2 l_i}, with l_i the distance to the nearest previously
    ordered location (the closest element of K_i), united with K_i."""
    from scipy.spatial import cKDTree
    Xo = np.asarray(X)[method.order]
    n = Xo.shape[0]
    Kptr, Kidx = method.Kptr, method.Kidx
    tree = cKDTree(Xo)
    cols = []
    for i in range(n):
        K = Kidx[Kptr[i]:Kptr[i + 1]]
        if K.size == 1:
            cols.append(K)
            continue
        ell = np.sqrt(((Xo[K[:-1]] - Xo[i]) ** 2).sum(1)).min()
        ball = np.asarray(tree.query_ball_point(Xo[i], rho2 * ell * (1 + 1e-12)))
        cols.append(np.union1d(ball[ball <= i], K))
    K2ptr = np.zeros(n + 1, dtype=np.int64)
    K2ptr[1:] = np.cumsum([c.size for c in cols])
    method.K2ptr, method.K2idx = K2ptr, np.concatenate(cols).astype(np.int64)
    method.info["rho2"] = rho2
    method.info["K2_mean"] = method.K2idx.size / n - 1
    return method


def add_stage2_uut_pattern(method):
    """Use the sparsity pattern of U U^T (upper triangle, column j = rows
    i <= j), which contains K and is contained in S_{2 rho}, for Stage 2 of
    an NNV method (IC0 of W on its own pattern; Schafer et al., Sec. 4.1)."""
    n = method.n
    S = sp.csc_matrix((np.ones(method.Kidx.size), method.Kidx, method.Kptr),
                      shape=(n, n))
    P = sp.triu(S @ S.T, format="csc")
    P.sort_indices()
    method.K2ptr = P.indptr.astype(np.int64)
    method.K2idx = P.indices.astype(np.int64)
    method.info["rho2"] = "UU^T"
    method.info["K2_mean"] = P.nnz / n - 1
    return method


def neg_log_density_L(x, mean, L):
    """-log N(x; mean, L L^T) for lower-triangular L (CSR, diagonal last)."""
    r = _solve_lower_csr(L.indptr.astype(np.int64), L.indices.astype(np.int64),
                         L.data, x - mean)
    diag = L.data[L.indptr[1:] - 1]
    return np.sum(np.log(diag)) + 0.5 * r @ r + 0.5 * x.size * LOG2PI


def neg_log_density_V(x, mean, V):
    """-log N(x; mean, (V V^T)^{-1}) for upper-triangular V (CSC)."""
    r = V.T @ (x - mean)
    diag = V.data[V.indptr[1:] - 1]
    return -np.sum(np.log(diag)) + 0.5 * r @ r + 0.5 * x.size * LOG2PI


def stage3_residual(V, L):
    """epsilon^2 = sum_i (1 - V_ii L_ii) (Proposition 1a)."""
    dv = V.data[V.indptr[1:] - 1]
    dl = L.data[L.indptr[1:] - 1]
    return float(np.sum(1 - dv * dl))


# ============================================================
# Patterns
# ============================================================

def nn_method(X, rho):
    """NN-triple (NNV): maximin ordering, K = S_rho, J = reduced ancestors."""
    from veccs.orderings import maxmin_cpp_ancestor
    t0 = time.perf_counter()
    ret = maxmin_cpp_ancestor(X, np.empty((0, 2)), rho)
    sparsity = np.asarray(ret.sparsity)
    ancestors = np.asarray(ret.ancestor_set_reduced)
    n = X.shape[0]
    Kptr, Kidx = sets_to_csr(sparsity[1], sparsity[0], n)
    Jptr, Jidx = sets_to_csr(ancestors[1], ancestors[0], n)
    return Method("NNV", np.asarray(ret.maximin_order), Kptr, Kidx, Jptr,
                  Jidx, exact_stage3=False,
                  pattern_time=time.perf_counter() - t0, info={"rho": rho})


def read_gpvecchia_pattern(prefix, name):
    """Ordering and conditioning sets written by export_patterns_sweep.R."""
    meta = np.loadtxt(f"{prefix}.meta", dtype=np.int64)
    n, width = int(meta[0]), int(meta[1])
    pattern_time = float(np.loadtxt(f"{prefix}.time"))
    order = np.fromfile(f"{prefix}.ord", dtype=np.int32).astype(np.int64) - 1
    rev = np.fromfile(f"{prefix}.nn", dtype=np.int32).reshape(width, n).T
    # R matrices are column-major; NA (written as -1) pads on the left
    rows = np.repeat(np.arange(n), width)
    cols = rev.ravel().astype(np.int64) - 1
    keep = cols >= 0
    rows, cols = rows[keep], cols[keep]
    off = rows != cols
    Kptr, Kidx = sets_to_csr(rows[off], cols[off], n)
    return Method(name, order, Kptr, Kidx, Kptr, Kidx, exact_stage3=True,
                  pattern_time=pattern_time)


def check_closed(method):
    """True if k in K_i implies K_k subset of K_i (needed for exactness)."""
    Kptr, Kidx = method.Kptr, method.Kidx
    return bool(_check_closed(Kptr, Kidx))


@njit(cache=True)
def _check_closed(Kptr, Kidx):
    n = Kptr.size - 1
    for i in range(n):
        K = Kidx[Kptr[i]:Kptr[i + 1]]
        for k in K[:-1]:
            Kk = Kidx[Kptr[k]:Kptr[k + 1]]
            pos = np.empty(Kk.size, dtype=np.int64)
            _merge_positions(Kk, K, pos)
            if np.any(pos < 0):
                return False
    return True


def matern_dense(X, variance, range_param, smoothness):
    """Dense Matern covariance matrix (only for the exact check)."""
    from scipy.spatial.distance import cdist
    r = cdist(X, X) / range_param
    if smoothness == 0.5:
        return variance * np.exp(-r)
    return variance * (1.0 + r) * np.exp(-r)
