"""
Scalable CPU implementation of the filter of Section 2.7 (Algorithm 1) for
large n. In contrast to sparse_functions.py, no dense n x n matrix is formed:
the forecast covariance is only evaluated on the conditioning sets
(Proposition 4), Stage 2 is a compiled IC0, and the Stage-3 Gram matrices are
read off the sparse matrix V^T V.

The same code runs NN-triple, HV and LR; the methods differ only in their
ordering and in the patterns K_i (Stages 1-2) and J_i (Stage 3). For HV and
LR, J_i = K_i, and Stages 2 and 3 are exact (Proposition 3).

Patterns are stored in CSR-like form (ptr, idx): the set of point i is
idx[ptr[i]:ptr[i + 1]], sorted, with i itself as the last element.
"""
import math
import numpy as np
import scipy.sparse as sp
from numba import njit, prange, get_num_threads


# ============================================================
# Covariance functions (Matern with smoothness 0.5 or 1.5, in the
# parameterization of GPvecchia::MaternFun)
# ============================================================

@njit(cache=True, inline="always")
def matern(d, variance, range_param, smoothness):
    r = d / range_param
    if smoothness == 0.5:
        return variance * math.exp(-r)
    return variance * (1.0 + r) * math.exp(-r)


def matern_matrix(X, variance, range_param, smoothness):
    """Dense covariance matrix, only used for small problems."""
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    r = d / range_param
    if smoothness == 0.5:
        return variance * np.exp(-r)
    return variance * (1.0 + r) * np.exp(-r)


# ============================================================
# Patterns
# ============================================================

def sets_to_csr(rows, cols, n):
    """Pattern (ptr, idx) from pairs (i, j), adding i to its own set."""
    rows = np.concatenate([np.asarray(rows), np.arange(n)])
    cols = np.concatenate([np.asarray(cols), np.arange(n)])
    if np.any(cols > rows):
        raise ValueError("Sets must only contain previously ordered points.")
    M = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    M.sort_indices()
    return M.indptr.astype(np.int64), M.indices.astype(np.int64)


def chunk_bounds(n, n_chunks=None):
    n_chunks = n_chunks or 4 * get_num_threads()
    return np.linspace(0, n, n_chunks + 1).astype(np.int64)


# ============================================================
# Stage 1 on the prior or the forecast covariance
# ============================================================

@njit(cache=True)
def _stage1_column(S, out):
    """out = Sigma_KK^{-1} e_last / sqrt(e_last' Sigma_KK^{-1} e_last)."""
    m = S.shape[0]
    e = np.zeros(m)
    e[m - 1] = 1.0
    v = np.linalg.solve(S, e)
    scale = 1.0 / math.sqrt(v[m - 1])
    for a in range(m):
        out[a] = v[a] * scale


@njit(parallel=True, cache=True)
def stage1_prior(ptr, idx, X, variance, range_param, smoothness, bounds):
    """Vecchia factor U (CSC values on the pattern) of the Matern prior."""
    vals = np.empty(idx.size)
    for c in prange(bounds.size - 1):
        for i in range(bounds[c], bounds[c + 1]):
            K = idx[ptr[i]:ptr[i + 1]]
            m = K.size
            S = np.empty((m, m))
            for a in range(m):
                for b in range(a + 1):
                    d = math.sqrt((X[K[a], 0] - X[K[b], 0]) ** 2
                                  + (X[K[a], 1] - X[K[b], 1]) ** 2)
                    S[a, b] = matern(d, variance, range_param, smoothness)
                    S[b, a] = S[a, b]
            _stage1_column(S, vals[ptr[i]:ptr[i + 1]])
    return vals


@njit(parallel=True, cache=True)
def stage1_forecast(ptr, idx, Fptr, Fidx, Fval, X, variance, range_param,
                    smoothness, bounds, n):
    """
    Vecchia factor U of the forecast covariance F F^T + Q, where only the
    entries [F F^T + Q]_{K_i, K_i} are computed (Proposition 4).
    """
    vals = np.empty(idx.size)
    for c in prange(bounds.size - 1):
        work = np.zeros(n)
        for i in range(bounds[c], bounds[c + 1]):
            K = idx[ptr[i]:ptr[i + 1]]
            m = K.size
            S = np.empty((m, m))
            for a in range(m):
                ka = K[a]
                # Scatter row ka of F, then take sparse inner products
                for q in range(Fptr[ka], Fptr[ka + 1]):
                    work[Fidx[q]] = Fval[q]
                for b in range(a + 1):
                    kb = K[b]
                    s = 0.0
                    for q in range(Fptr[kb], Fptr[kb + 1]):
                        s += work[Fidx[q]] * Fval[q]
                    d = math.sqrt((X[ka, 0] - X[kb, 0]) ** 2
                                  + (X[ka, 1] - X[kb, 1]) ** 2)
                    S[a, b] = s + matern(d, variance, range_param, smoothness)
                    S[b, a] = S[a, b]
                for q in range(Fptr[ka], Fptr[ka + 1]):
                    work[Fidx[q]] = 0.0
            _stage1_column(S, vals[ptr[i]:ptr[i + 1]])
    return vals


# ============================================================
# Stage 2: incomplete Cholesky (IC0)
# ============================================================

@njit(cache=True)
def _ichol0_lower(indptr, indices, vals):
    """
    IC0 of a lower-triangular CSR matrix with sorted column indices and the
    diagonal last in each row (the algorithm of sparse_functions.ichol0_csr).
    """
    n = indptr.size - 1
    for i in range(n):
        for u1 in range(indptr[i], indptr[i + 1]):
            j = indices[u1]
            # sum_{k < j} L_ik L_jk by merging the sorted rows i and j
            l1, l2, u2 = indptr[i], indptr[j], indptr[j + 1] - 1
            dp = 0.0
            while l1 < u1 and l2 < u2:
                if indices[l1] == indices[l2]:
                    dp += vals[l1] * vals[l2]
                    l1 += 1
                    l2 += 1
                elif indices[l1] < indices[l2]:
                    l1 += 1
                else:
                    l2 += 1
            if j < i:
                vals[u1] = (vals[u1] - dp) / vals[indptr[j + 1] - 1]
            else:
                if vals[u1] - dp < 1e-10:
                    raise RuntimeError("IC0 breakdown: nonpositive pivot")
                vals[u1] = math.sqrt(vals[u1] - dp)
    return vals


def ichol0_upper(W, shifts=(0.0, 1e-4, 1e-3, 1e-2, 1e-1)):
    """
    Upper-triangular V with W ~ V V^T, by IC0 on the pattern of W under the
    reverse ordering, as in sparse_functions.ichol0(W, upper=True).

    IC0 can break down for a positive-definite W. In that case, the diagonal
    is scaled by (1 + alpha) for the smallest alpha in shifts that succeeds
    (Manteuffel, 1980). Returns V and alpha.
    """
    n = W.shape[0]
    W = W.tocoo()
    A = sp.csr_matrix((W.data, (n - 1 - W.row, n - 1 - W.col)), shape=(n, n))
    A = sp.tril(A, format="csr")
    A.sum_duplicates()
    A.sort_indices()
    # The diagonal is the last entry of each row
    diagonal = A.indptr[1:] - 1
    for alpha in shifts:
        vals = A.data.copy()
        vals[diagonal] *= 1.0 + alpha
        try:
            vals = _ichol0_lower(A.indptr, A.indices, vals)
            break
        except RuntimeError:
            if alpha == shifts[-1]:
                raise
    L = sp.csr_matrix((vals, A.indices, A.indptr), shape=(n, n)).tocoo()
    V = sp.csr_matrix((L.data, (n - 1 - L.row, n - 1 - L.col)), shape=(n, n))
    return V, alpha


# ============================================================
# Stage 3: SPAI rows from the Gram matrix V^T V
# ============================================================

@njit(parallel=True, cache=True)
def _spai_rows(Jptr, Jidx, Mptr, Midx, Mval, Vdiag, bounds, n):
    vals = np.empty(Jidx.size)
    for c in prange(bounds.size - 1):
        pos = -np.ones(n, dtype=np.int64)
        for i in range(bounds[c], bounds[c + 1]):
            J = Jidx[Jptr[i]:Jptr[i + 1]]
            p = J.size
            for a in range(p):
                pos[J[a]] = a
            G = np.zeros((p, p))
            for a in range(p):
                start, end = Mptr[J[a]], Mptr[J[a] + 1]
                if end - start <= 8 * p:
                    for q in range(start, end):
                        b = pos[Midx[q]]
                        if b >= 0:
                            G[a, b] = Mval[q]
                else:
                    # Long rows (coarse points of HV and LR): binary search
                    row = Midx[start:end]
                    for b in range(p):
                        q = np.searchsorted(row, J[b])
                        if q < row.size and row[q] == J[b]:
                            G[a, b] = Mval[start + q]
            rhs = np.zeros(p)
            rhs[p - 1] = Vdiag[i]
            vals[Jptr[i]:Jptr[i + 1]] = np.linalg.solve(G, rhs)
            for a in range(p):
                pos[J[a]] = -1
    return vals


@njit(parallel=True, cache=True)
def _spai_rows_columns(Jptr, Jidx, Cptr, Cidx, Cval, Vdiag, bounds, n):
    """As _spai_rows, with G computed from the columns (Cptr, Cidx, Cval) of V."""
    vals = np.empty(Jidx.size)
    for c in prange(bounds.size - 1):
        work = np.zeros(n)
        for i in range(bounds[c], bounds[c + 1]):
            J = Jidx[Jptr[i]:Jptr[i + 1]]
            p = J.size
            G = np.empty((p, p))
            for a in range(p):
                for q in range(Cptr[J[a]], Cptr[J[a] + 1]):
                    work[Cidx[q]] = Cval[q]
                for b in range(a + 1):
                    s = 0.0
                    for q in range(Cptr[J[b]], Cptr[J[b] + 1]):
                        s += work[Cidx[q]] * Cval[q]
                    G[a, b] = s
                    G[b, a] = s
                for q in range(Cptr[J[a]], Cptr[J[a] + 1]):
                    work[Cidx[q]] = 0.0
            rhs = np.zeros(p)
            rhs[p - 1] = Vdiag[i]
            vals[Jptr[i]:Jptr[i + 1]] = np.linalg.solve(G, rhs)
    return vals


def spai(V, Jptr, Jidx, bounds, gram="product"):
    """
    Lower-triangular L (CSR on the pattern J) with L ~ V^{-T}. The Gram
    matrices are read off the sparse product V^T V (gram="product"), or are
    computed from the columns of V (gram="columns"), which avoids forming
    V^T V when it is dense, as for LR.
    """
    n = V.shape[0]
    if gram == "product":
        M = (V.T @ V).tocsr()
        M.sort_indices()
        vals = _spai_rows(Jptr, Jidx, M.indptr.astype(np.int64),
                          M.indices.astype(np.int64), M.data, V.diagonal(),
                          bounds, n)
    else:
        C = V.tocsc()
        C.sort_indices()
        vals = _spai_rows_columns(Jptr, Jidx, C.indptr.astype(np.int64),
                                  C.indices.astype(np.int64), C.data,
                                  V.diagonal(), bounds, n)
    return sp.csr_matrix((vals, Jidx, Jptr), shape=(n, n))


# ============================================================
# Update: preconditioned conjugate gradient
# ============================================================

@njit(cache=True)
def _solve_upper(indptr, indices, vals, b):
    n = b.size
    x = b.copy()
    for i in range(n - 1, -1, -1):
        diag = 0.0
        s = x[i]
        for q in range(indptr[i], indptr[i + 1]):
            j = indices[q]
            if j == i:
                diag = vals[q]
            elif j > i:
                s -= vals[q] * x[j]
        x[i] = s / diag
    return x


@njit(cache=True)
def _solve_lower(indptr, indices, vals, b):
    n = b.size
    x = b.copy()
    for i in range(n):
        diag = 0.0
        s = x[i]
        for q in range(indptr[i], indptr[i + 1]):
            j = indices[q]
            if j == i:
                diag = vals[q]
            elif j < i:
                s -= vals[q] * x[j]
        x[i] = s / diag
    return x


def pcg(W, b, V, x0, tol=1e-10, max_iter=1000):
    """Solve W x = b by conjugate gradient, preconditioned by (V V^T)^{-1}."""
    VT = V.T.tocsr()

    def precondition(r):
        y = _solve_upper(V.indptr, V.indices, V.data, r)
        return _solve_lower(VT.indptr, VT.indices, VT.data, y)

    x = x0.copy()
    r = b - W @ x
    z = precondition(r)
    p = z.copy()
    rz = r @ z
    threshold = tol * np.linalg.norm(b)
    for it in range(1, max_iter + 1):
        Wp = W @ p
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
# Evolution
# ============================================================

def advection_diffusion_matrix(side, advection, diffusion):
    """
    Centered finite-difference evolution matrix on a side x side grid with
    spacing 1 / side, as advection_diffusion_matrix in src/R/utils.R (cell
    k = row * side + col).
    """
    n = side * side
    dx = 1.0 / side
    D = diffusion / dx ** 2
    c1 = 1 + 4 * D - 2 * advection / dx
    c2 = -D + advection / dx
    c3 = -D
    below = np.full(n - 1, c2)
    above = np.full(n - 1, c3)
    # No coupling across the edges of the grid rows
    below[side * np.arange(1, side) - 1] = 0.0
    above[side * np.arange(1, side) - 1] = 0.0
    return sp.diags(
        [np.full(n - side, c2), below, np.full(n, c1), above,
         np.full(n - side, c3)],
        [-side, -1, 0, 1, side], shape=(n, n), format="csr",
    )


# ============================================================
# One filter
# ============================================================

class SparseFilter:
    """
    Algorithm 1 for Gaussian data y_t = x_t + noise on a fixed ordering.

    X: ordered locations; H: evolution (CSR) in the same ordering;
    (Kptr, Kidx): Stage-1/2 pattern; (Jptr, Jidx): Stage-3 pattern;
    cov0 and covQ: (variance, range, smoothness) of Sigma_{1|0} and Q_t;
    gram: how Stage 3 computes its Gram matrices (see spai).
    """

    def __init__(self, X, H, Kptr, Kidx, Jptr, Jidx, cov0, covQ,
                 gram="product"):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.H = H.tocsr()
        self.n = X.shape[0]
        self.Kptr, self.Kidx = Kptr, Kidx
        self.Jptr, self.Jidx = Jptr, Jidx
        self.cov0, self.covQ = cov0, covQ
        self.gram = gram
        self.bounds = chunk_bounds(self.n)
        self.mean = None
        self.L = None

    def stage1(self):
        if self.L is None:
            vals = stage1_prior(self.Kptr, self.Kidx, self.X, *self.cov0,
                                self.bounds)
            return np.zeros(self.n), vals
        F = (self.H @ self.L).tocsr()
        F.sort_indices()
        vals = stage1_forecast(self.Kptr, self.Kidx,
                               F.indptr.astype(np.int64),
                               F.indices.astype(np.int64), F.data,
                               self.X, *self.covQ, self.bounds, self.n)
        return self.H @ self.mean, vals

    def step(self, y, noise_variance, timer=None):
        """Forecast (t > 1), Stage 1, update, Stages 2 and 3 for one time."""
        tick = timer or (lambda name: None)
        prior_mean, uvals = self.stage1()
        U = sp.csc_matrix((uvals, self.Kidx, self.Kptr), shape=(self.n, self.n))
        tick("stage1")
        observed = ~np.isnan(y)
        h = np.where(observed, 1.0 / noise_variance, 0.0)
        UUT = (U @ U.T).tocsr()
        W = (UUT + sp.diags(h)).tocsr()
        W.sort_indices()
        V, self.ic_shift = ichol0_upper(W)
        tick("stage2")
        rhs = UUT @ prior_mean + h * np.nan_to_num(y)
        self.mean, self.cg_iters = pcg(W, rhs, V, x0=prior_mean)
        tick("update")
        self.L = spai(V, self.Jptr, self.Jidx, self.bounds, self.gram)
        tick("stage3")
        return self.mean, self.marginal_variance()

    def marginal_variance(self):
        return np.asarray(self.L.multiply(self.L).sum(axis=1)).ravel()
