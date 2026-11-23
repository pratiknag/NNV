import torch
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from veccs.orderings import (
    find_nns_l2_mf,
    maxmin_cpp,
    maxmin_cpp_ancestor,
    maxmin_naive,
    maxmin_pred_cpp,
)

# ============================================================
# 2. Exponential covariance
# ============================================================

def exponential_covariance(
    X,
    variance=1.0,
    range_param=0.1
):
    """
    PyTorch exponential covariance matrix.

        Sigma_ij =
            variance * exp(-||x_i-x_j|| / range_param)

    Parameters
    ----------
    X : torch.Tensor, shape (n,d)

    variance : float
        Marginal variance.

    range_param : float
        Spatial range parameter.

    Returns
    -------
    Sigma : torch.Tensor, shape (n,n)
    """

    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X,dtype=torch.float64)

    diff = (X[:, None, :]- X[None, :, :])
    distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
    Sigma = variance * torch.exp(-distances / range_param)
    return Sigma


def sparse_inverse_cholesky(Sigma, S_rho):
    """
    Batched PyTorch implementation of sparse inverse-Cholesky / Vecchia factor.

    For each i, let K_i be the conditioning set including i. We compute

        Sigma[K_i, K_i] v_i = e_i

        u_ii = sqrt(v_i[i])

        U[K_i, i] = v_i / u_ii

    The Cholesky factorizations are batched by conditioning-set size.

    Parameters
    ----------
    Sigma : torch.Tensor
        Dense covariance matrix, shape (n, n).

    S_rho : torch.Tensor
        Sparse COO tensor containing the Vecchia conditioning sets.
        Column i contains the neighbors K_i.

    Returns
    -------
    U : torch.Tensor
        Sparse COO upper-triangular inverse-Cholesky factor.
    """

    n = Sigma.shape[0]
    device = Sigma.device
    dtype = Sigma.dtype

    if not S_rho.is_sparse:
        S_rho = S_rho.to_sparse_coo()

    S_rho = S_rho.coalesce()

    indices = S_rho.indices()
    rows_S = indices[0]
    cols_S = indices[1]
    K_list = []

    for i in range(n):

        mask = cols_S == i
        K = rows_S[mask]

        # Make sure i is included
        if not torch.any(K == i):
            K = torch.cat([
                K,
                torch.tensor([i], dtype=torch.long, device=device)
            ])

        K = torch.sort(K).values

        K_list.append(K)
    groups = {}

    for i, K in enumerate(K_list):

        m = K.numel()

        if m not in groups:
            groups[m] = []

        groups[m].append(i)
        
    rows_out = []
    cols_out = []
    values_out = []

    for m, i_list in groups.items():

        B = len(i_list)

        # [B, m]
        K_batch = torch.stack([
            K_list[i] for i in i_list
        ])

        Sigma_KK = Sigma[
            K_batch[:, :, None],
            K_batch[:, None, :]
        ]

        L = torch.linalg.cholesky(Sigma_KK)

        e = torch.zeros(
            (B, m),
            dtype=dtype,
            device=device
        )

        # Find position of i inside K_i
        i_tensor = torch.tensor(
            i_list,
            dtype=torch.long,
            device=device
        )

        pos_i = (K_batch == i_tensor[:, None]).nonzero(
            as_tuple=False
        )[:, 1]

        e[
            torch.arange(B, device=device),
            pos_i
        ] = 1.0
        v = torch.cholesky_solve(
            e.unsqueeze(-1),
            L
        ).squeeze(-1)

        v_i = v[
            torch.arange(B, device=device),
            pos_i
        ]

        u_ii = torch.sqrt(v_i)
        u_local = v / u_ii[:, None]

        for b, i in enumerate(i_list):

            rows_out.append(K_batch[b])
            cols_out.append(
                torch.full(
                    (m,),
                    i,
                    dtype=torch.long,
                    device=device
                )
            )

            values_out.append(u_local[b])
    rows = torch.cat(rows_out)
    cols = torch.cat(cols_out)
    values = torch.cat(values_out)

    U = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=(n, n),
        dtype=dtype,
        device=device
    ).coalesce()

    return U

ALMOSTZERO = 1e-10


def dot_prod(l1, u1, l2, u2, row_inds, vals):
    """
    Exact Python equivalent of the C++ dot_prod().

    Computes the dot product of two sparse row segments:
        vals[l1:u1] and vals[l2:u2]

    assuming row_inds[l1:u1] and row_inds[l2:u2] are sorted.
    """

    result = 0.0

    while l1 < u1 and l2 < u2:

        if row_inds[l1] == row_inds[l2]:
            result += vals[l1] * vals[l2]
            l1 += 1
            l2 += 1

        elif row_inds[l1] < row_inds[l2]:
            l1 += 1

        else:
            l2 += 1

    return result


def ichol0_csr(crow_inds, col_inds, vals, nrow):
    """
    Python equivalent of the C++ ichol0().

    Parameters
    ----------
    crow_inds : list[int]
        CSR row pointers.

    col_inds : list[int]
        CSR column indices.

    vals : list[float]
        CSR values. These are modified in-place.

    nrow : int
        Number of rows.

    Returns
    -------
    vals : list[float]
        Values of the incomplete Cholesky factor.
    """

    for i in range(nrow):

        # Equivalent to:
        # if(col_inds[crow_inds[i + 1] - 1] != i)
        if col_inds[crow_inds[i + 1] - 1] != i:
            raise ValueError(
                f"Diagonal {i} should be a non-zero entry\n"
            )

        # Equivalent to:
        # for(int u1 = crow_inds[i]; u1 < crow_inds[i + 1]; ++u1)
        for u1 in range(crow_inds[i], crow_inds[i + 1]):

            j = col_inds[u1]

            # Check lower triangular structure
            if j > i:
                raise ValueError(
                    "The input sparsity is not lower-triangular\n"
                )

            l1 = crow_inds[i]
            l2 = crow_inds[j]

            # C++:
            # int u2 = crow_inds[j + 1] - 1;
            #
            # Notice that u2 is EXCLUSIVE in dot_prod(),
            # so this intentionally excludes the diagonal entry.
            u2 = crow_inds[j + 1] - 1

            dp = dot_prod(l1,u1,l2,u2,col_inds,vals)

            if j < i:

                vals[u1] = (vals[u1] - dp) / vals[crow_inds[j + 1] - 1]
                
            elif j == i:

                if vals[u1] - dp < ALMOSTZERO:
                    raise RuntimeError(
                        f"The {i}-th diagonal coefficient is smaller "
                        f"than {ALMOSTZERO}\n"
                    )

                vals[u1] = math.sqrt(vals[u1] - dp)

    return vals


def ichol0(spMatCOO: torch.Tensor, upper=False):
    """
    Incomplete Cholesky factorization, faithful to the
    supplied C++/pybind11 implementation.

    Parameters
    ----------
    spMatCOO : torch.Tensor
        Sparse COO matrix.

    upper : bool, default=False
        If True, performs the same index reversal used
        in the original implementation.

    Returns
    -------
    torch.Tensor
        Sparse COO representation of the incomplete
        Cholesky factor.
    """

    # Same as C++ wrapper
    if not spMatCOO.is_coalesced():
        spMatCOO = spMatCOO.coalesce()

    nrow = spMatCOO.shape[0]

    indices = spMatCOO.indices()
    vals = spMatCOO.values()

    # Same upper-triangular transformation
    if upper:
        indices = -indices + nrow - 1

    # Keep lower triangular part
    inds_inds_low = indices[0] >= indices[1]

    inds_low = indices[:, inds_inds_low]
    vals_low = vals[inds_inds_low]

    # Reconstruct sparse COO and coalesce
    spMatCOONew = torch.sparse_coo_tensor(
        inds_low,
        vals_low,
        [nrow, nrow]
    ).coalesce()

    inds_low = spMatCOONew.indices()

    # Convert to CSR
    spMatCSR = spMatCOONew.to_sparse_csr()

    # Convert exactly as in original code
    crow_inds = spMatCSR.crow_indices().tolist()
    col_inds = spMatCSR.col_indices().tolist()
    vals = spMatCSR.values().detach().clone().tolist()

    # Python replacement for:
    # rslt = lib.ichol0(crow_inds, col_inds, vals, nrow)
    rslt = ichol0_csr(
        crow_inds,
        col_inds,
        vals,
        nrow
    )

    # Reverse indices back for upper=True
    if upper:
        inds_low = -inds_low + nrow - 1

    return torch.sparse_coo_tensor(
        inds_low,
        torch.tensor(
            rslt,
            dtype=spMatCOONew.values().dtype,
            device=spMatCOONew.values().device
        ),
        [nrow, nrow]
    ).coalesce()

def spai_from_V(
    V,
    S_A,
    batch_size=2048,
    tol=1e-14,
):
    """
    Batched / GPU-parallel SPAI construction.

    For each row i:

        J_i = sparsity pattern from S_A

        V_i = V[:, J_i]

        G_i = V_i.T @ V_i

        b_i = V[i, J_i]

        x_i = G_i^{-1} b_i

    and x_i is inserted into row i of L.

    Parameters
    ----------
    V : torch.Tensor
        Dense or sparse matrix, shape (n, m).

    S_A : torch.Tensor
        Sparse COO sparsity pattern for L.

    batch_size : int
        Number of rows processed simultaneously.

    tol : float
        Entries with abs(value) <= tol are discarded.

    Returns
    -------
    L : torch.Tensor
        Sparse SPAI factor.

    Winv : torch.Tensor
        Sparse L L^T.
    """
    if not V.is_sparse:
        V_dense = V
    else:
        V_dense = V.to_dense()

    if not S_A.is_sparse:
        S_A = S_A.to_sparse_coo()

    S_A = S_A.coalesce()

    n = V_dense.shape[0]

    device = V_dense.device
    dtype = V_dense.dtype

    indices = S_A.indices()

    S_A_rows = indices[0]
    S_A_cols = indices[1]

    row_counts = torch.bincount(
        S_A_rows,
        minlength=n
    )
    all_rows = []
    all_cols = []
    all_values = []
    unique_sizes = torch.unique(row_counts)

    for k_tensor in unique_sizes:

        k = int(k_tensor.item())

        # Rows having exactly k entries
        row_ids = torch.nonzero(
            row_counts == k,
            as_tuple=False
        ).flatten()

        if k == 0:
            continue

        for start in range(0, row_ids.numel(), batch_size):

            rows_batch = row_ids[
                start:start + batch_size
            ]

            B = rows_batch.numel()

            # Since every row in this group has exactly k entries,
            # collect its column indices.
            J_batch = torch.empty(
                (B, k),
                dtype=torch.long,
                device=device
            )

            for b, row in enumerate(rows_batch):

                mask = S_A_rows == row

                J_batch[b] = S_A_cols[mask]

            Vi = V_dense[:, J_batch]
            Vi = Vi.permute(1, 0, 2)
            G = torch.bmm(
                Vi.transpose(1, 2),
                Vi
            )
            b = V_dense[
                rows_batch[:, None],
                J_batch
            ]
            try:

                x = torch.linalg.solve(
                    G,
                    b.unsqueeze(-1)
                ).squeeze(-1)

            except RuntimeError:

                x = torch.linalg.lstsq(
                    G,
                    b.unsqueeze(-1)
                ).solution.squeeze(-1)
            keep = torch.abs(x) > tol

            if torch.any(keep):

                b_idx, local_idx = torch.nonzero(
                    keep,
                    as_tuple=True
                )

                all_rows.append(
                    rows_batch[b_idx]
                )

                all_cols.append(
                    J_batch[b_idx, local_idx]
                )

                all_values.append(
                    x[b_idx, local_idx]
                )
    if len(all_values) == 0:

        L = torch.sparse_coo_tensor(
            torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device
            ),
            torch.empty(
                (0,),
                dtype=dtype,
                device=device
            ),
            (n, n)
        ).coalesce()

    else:

        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        values = torch.cat(all_values)

        L = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            values,
            (n, n),
            dtype=dtype,
            device=device
        ).coalesce()
    L_T = torch.sparse_coo_tensor(
        L.indices().flip(0),
        L.values(),
        L.shape,
        dtype=L.dtype,
        device=L.device
    ).coalesce()
    Winv = torch.sparse.mm(
        L,
        L_T
    ).coalesce()

    return L, Winv

def make_sparse_patterns(
    sparsity,
    ancestor_set_reduced,
    n,
    dtype=torch.float64,
    device=None,
):
    sparsity = torch.as_tensor(
        sparsity,
        dtype=torch.long,
        device=device,
    )

    ancestor_set_reduced = torch.as_tensor(
        ancestor_set_reduced,
        dtype=torch.long,
        device=device,
    )

    if sparsity.shape[0] != 2:
        raise ValueError("sparsity must have shape (2, number_of_pairs).")

    if ancestor_set_reduced.shape[0] != 2:
        raise ValueError(
            "ancestor_set_reduced must have shape (2, number_of_pairs)."
        )

    # sparsity[0] = earlier point j
    # sparsity[1] = current point i
    S_rho = torch.sparse_coo_tensor(
        sparsity,
        torch.ones(
            sparsity.shape[1],
            dtype=dtype,
            device=device,
        ),
        size=(n, n),
        dtype=dtype,
        device=device,
    ).coalesce()

    # ancestor_set_reduced[0] = ancestor j
    # ancestor_set_reduced[1] = current point i
    #
    # spai_from_V expects S_rho_A[i, j],
    # so reverse the two rows.
    S_rho_A_indices = ancestor_set_reduced.flip(0)

    S_rho_A = torch.sparse_coo_tensor(
        S_rho_A_indices,
        torch.ones(
            S_rho_A_indices.shape[1],
            dtype=dtype,
            device=device,
        ),
        size=(n, n),
        dtype=dtype,
        device=device,
    ).coalesce()

    return S_rho, S_rho_A


def sparse_diagonal(A):
    A = A.coalesce()
    indices = A.indices()
    values = A.values()

    diagonal_mask = indices[0] == indices[1]

    diagonal = torch.zeros(
        A.shape[0],
        dtype=A.dtype,
        device=A.device,
    )

    diagonal[indices[0][diagonal_mask]] = values[diagonal_mask]

    return diagonal


def dposterior(y, pred):
    """
    Dense PyTorch Gaussian posterior log-density.

    pred["W"] is the precision matrix.
    pred["V"] is the Cholesky factor satisfying:

        W = V @ V.T
    """
    y = y.reshape(-1)
    mu = pred["mean"].reshape(-1)
    # Convert everything to dense tensors
    W = pred["W"].to_dense() if pred["W"].is_sparse else pred["W"]
    V = pred["V"].to_dense() if pred["V"].is_sparse else pred["V"]

    if y.device != W.device:
        raise ValueError("y and W must be on the same device.")

    if mu.device != W.device:
        raise ValueError("mean and W must be on the same device.")

    if V.device != W.device:
        raise ValueError("V and W must be on the same device.")

    if y.dtype != W.dtype or mu.dtype != W.dtype or V.dtype != W.dtype:
        raise ValueError("y, mean, W, and V must have the same dtype.")

    residual = y - mu
    # W @ (y - mu)
    W_residual = W @ residual
    # -1/2 (y - mu)^T W (y - mu)
    quad_term = -0.5 * (residual @ W_residual)
    # log(det(W)) / 2
    #
    # If W = V @ V.T, then
    # log(det(W)) / 2 = log(det(V))
    diagonal_V = torch.diagonal(V)
    
    if torch.any(diagonal_V <= 0):
        raise ValueError("V must have a positive diagonal.")

    det_term = torch.log(diagonal_V).sum()
    # -n/2 log(2*pi)
    pi_term = -0.5 * y.numel() * torch.log(
        torch.tensor(
            2.0 * torch.pi,
            dtype=W.dtype,
            device=W.device,
        )
    )

    return quad_term + det_term + pi_term
def get_average_conditioning_size(S_rho, n, X):
    """
    Average conditioning-set size for a given S_rho pattern.

    Points with no conditioning neighbors contribute zero to the average.
    """
    S = S_rho.coalesce()
    rows = S.indices()[0]
    cols = S.indices()[1]
    cols_offdiag = cols[rows != cols]
    conditioning_sizes = torch.bincount(cols_offdiag, minlength=n)
    average_K = int(conditioning_sizes.sum().item() // n)
    return average_K, conditioning_sizes


def calibrate_rho_values(locs, target_sizes, iterations=20):
    """Find rho values whose floored mean conditioning sizes match targets."""
    n = locs.shape[0]
    empty_predictions = np.empty((0, locs.shape[1]))

    def get_floor_average(rho):
        ordering = maxmin_cpp_ancestor(locs, empty_predictions, rho)
        pairs = np.unique(ordering.sparsity, axis=1)
        off_diagonal = pairs[0] != pairs[1]
        counts = np.bincount(pairs[1, off_diagonal], minlength=n)
        return int(counts.sum() // n)

    calibrated = []
    for target in target_sizes:
        target = int(target)
        if target < 0 or target >= n:
            raise ValueError(f"target conditioning size must be in [0, {n - 1}].")

        lower = 0.8
        lower_size = get_floor_average(lower)
        if lower_size >= target:
            calibrated.append((lower, lower_size))
            continue

        upper = 1.0
        upper_size = get_floor_average(upper)
        while upper_size < target and upper < 1024.0:
            lower, lower_size = upper, upper_size
            upper *= 2.0
            upper_size = get_floor_average(upper)

        if upper_size < target:
            calibrated.append((upper, upper_size))
            continue

        for _ in range(iterations):
            midpoint = (lower + upper) / 2.0
            midpoint_size = get_floor_average(midpoint)
            if midpoint_size >= target:
                upper, upper_size = midpoint, midpoint_size
            else:
                lower, lower_size = midpoint, midpoint_size

        if abs(lower_size - target) <= abs(upper_size - target):
            calibrated.append((lower, lower_size))
        else:
            calibrated.append((upper, upper_size))

    return calibrated


DATA_MODELS = ["gaussian", "logistic", "poisson", "gamma"]


def get_likelihood(model, noise_variance=0.2, gamma_shape=2.0):
    """
    Score and negative Hessian (diagonal) of log p(z | y) in y,
    matching GPvecchia's likelihood models.

    gamma uses the log-rate parameterization (GPvecchia "gamma_alt"):
        z ~ Gamma(shape=gamma_shape, rate=exp(y)).
    """
    if model == "gaussian":
        score = lambda y, z: (z - y) / noise_variance
        hess = lambda y, z: torch.full_like(y, 1.0 / noise_variance)
    elif model == "logistic":
        score = lambda y, z: z - torch.sigmoid(y)
        hess = lambda y, z: torch.sigmoid(y) * (1.0 - torch.sigmoid(y))
    elif model == "poisson":
        score = lambda y, z: z - torch.exp(y)
        hess = lambda y, z: torch.exp(y)
    elif model == "gamma":
        score = lambda y, z: gamma_shape - z * torch.exp(y)
        hess = lambda y, z: z * torch.exp(y)
    else:
        raise ValueError(f"model must be one of {DATA_MODELS}, got {model!r}")
    return score, hess


def get_log_likelihood(model, noise_variance=0.2, gamma_shape=2.0):
    """
    Elementwise log p(z | y) up to terms that do not depend on y, matching
    get_likelihood.
    """
    if model == "gaussian":
        return lambda y, z: -0.5 * (z - y) ** 2 / noise_variance
    if model == "logistic":
        return lambda y, z: z * y - torch.nn.functional.softplus(y)
    if model == "poisson":
        return lambda y, z: z * y - torch.exp(y)
    if model == "gamma":
        return lambda y, z: gamma_shape * y - z * torch.exp(y)
    raise ValueError(f"model must be one of {DATA_MODELS}, got {model!r}")


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate gradient for A x = b with sparse symmetric positive definite A,
    stopping when ||b - A x|| <= tol * ||b||.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    p = r.clone()
    rs = r @ r
    threshold = (tol * torch.linalg.norm(b)) ** 2
    for _ in range(max_iter or 10 * b.numel()):
        if rs <= threshold:
            break
        Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def newton_laplace(solve, score, hess, Z, max_iter=50, tol=1e-6, linear=False,
                   y_init=None, log_posterior=None, max_halvings=30):
    """
    Newton-Raphson for the Laplace approximation, as in GPvecchia's
    calculate_posterior_VL with a zero prior mean.

    At the current mode y, with h = -d^2/dy^2 log p(z|y) and
    u = d/dy log p(z|y), the update is

        y_new = W^{-1} (h * y + u),    W = Sigma^{-1} + diag(h).

    solve(h, rhs) returns (W^{-1} rhs, W, V) under the chosen approximation.
    If linear (gaussian), a single step is exact. For a nonzero prior mean,
    solve adds Sigma^{-1} mu to rhs and y_init is the prior mean.
    Unobserved locations have h = u = 0. If log_posterior(y) is given, a
    step is halved until it does not decrease the log posterior.
    """
    y = torch.zeros_like(Z) if y_init is None else y_init.clone()
    converged = False
    iters = 0
    for iters in range(1, max_iter + 1):
        h = hess(y, Z)
        if torch.any(h < 0):
            raise RuntimeError("Negative Hessian, check parameters.")
        mean, W, V = solve(h, h * y + score(y, Z))
        if linear:
            return {"mean": mean, "W": W, "V": V}, True, 1
        if not torch.all(torch.isfinite(mean)):
            break
        if log_posterior is not None:
            current = log_posterior(y)
            for _ in range(max_halvings):
                candidate = log_posterior(mean)
                if torch.isfinite(candidate) and (
                    candidate >= current - 1e-10 * torch.abs(current)
                ):
                    break
                mean = (y + mean) / 2
        delta = torch.max(torch.abs(mean - y)).item()
        y = mean
        if delta < tol:
            converged = True
            break
    return {"mean": y, "W": W, "V": V}, converged, iters


def run_experiment(n,
                    df,
                   i,
                    rho=2.0,
                    model="gaussian",
                    range_param=0.15,
                    variance=1.0,
                    noise_variance=0.2,
                    gamma_shape=2.0,
                    seed_X=12344,
                    seed_Y=12345,
                    data_seed=1,
                    data_dir=None,
                    device=None,
                    dtype=torch.float64,
                    use_gpu = True,
                    max_iter=50,
                    tol=1e-6
                ):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
    if data_dir is None:
        data_dir = f"simulation_{model}"
    score, hess = get_likelihood(model, noise_variance, gamma_shape)
    # The Gaussian posterior is linear in z, so one step is exact.
    linear = model == "gaussian"

    row = df[df["seed_off"] == data_seed].iloc[i]
    rrmspe_HV = row["rrmspe_hv"]
    ls_VL = row["ls_hv"]
    rrmspe_LR = row["rrmspe_lr"]
    ls_LR = row["ls_lr"]
    loglik_exact = row["ls_exact"]
    mse_exact = row["mse_exact"]
    time_hv = row["hv_runtime"]
    time_lr = row["lr_runtime"]
    data_path = (
        f"{data_dir}/{model}_seed{data_seed}_domain1_dim2_n{n}"
        f"_smooth0.5_range{range_param}.csv"
    )
    data = pd.read_csv(data_path)
    required_columns = ["x1", "x2", "y", "z"]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(
            f"{data_path} is missing columns: {missing_columns}"
        )
    if len(data) != n:
        raise ValueError(
            f"Expected {n} rows in {data_path}, found {len(data)}"
        )

    X = torch.as_tensor(
        data[["x1", "x2"]].to_numpy(),
        dtype=dtype,
        device=device,
    )
    Y = torch.as_tensor(data["y"].to_numpy(), dtype=dtype, device=device)
    Z = torch.as_tensor(data["z"].to_numpy(), dtype=dtype, device=device)
    n = X.shape[0]

    pred_locs = np.empty((0, 2))
    ret_obj = maxmin_cpp_ancestor(X.cpu().numpy(), pred_locs, rho)
    S_rho, S_rho_A = make_sparse_patterns(
        sparsity=ret_obj.sparsity,
        ancestor_set_reduced=ret_obj.ancestor_set_reduced,
        n=n,
        dtype=X.dtype,
        device=X.device,
    )
    orderObj = torch.as_tensor(
        ret_obj.maximin_order,
        dtype=torch.long,
        device=X.device,
    )

    X = X.index_select(0, orderObj)
    Y = Y.index_select(0, orderObj)
    Z = Z.index_select(0, orderObj)

    Sigma = exponential_covariance(
        X,
        variance=variance,
        range_param=range_param,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_sparse_start = time.perf_counter()

    U = sparse_inverse_cholesky(Sigma, S_rho)

    UT = torch.sparse_coo_tensor(
        U.indices().flip(0),
        U.values(),
        U.shape,
        dtype=U.dtype,
        device=U.device,
    ).coalesce()
    UUT = torch.sparse.mm(U, UT).coalesce()

    def posterior_precision(h):
        Dinv = torch.sparse_coo_tensor(
            torch.arange(n, device=h.device).repeat(2, 1),
            h,
            size=(n, n),
            dtype=dtype,
            device=h.device,
        ).coalesce()
        return (UUT.to(h.device) + Dinv).coalesce()

    # Stage 2 reuses the incomplete Cholesky factor of the sparse method
    # whenever the Hessian is identical (always the case for gaussian).
    ichol0_cache = {}

    def solve_sparse(h, rhs):
        W = posterior_precision(h)
        V = ichol0(W, upper=True)
        ichol0_cache["h"], ichol0_cache["V"] = h.cpu(), V
        L, Winv = spai_from_V(V, S_rho_A)
        mean = torch.sparse.mm(Winv, rhs.unsqueeze(1)).squeeze(1)
        return mean, W, V

    pred_sparse, converged_sparse, iters_sparse = newton_laplace(
        solve_sparse, score, hess, Z, max_iter=max_iter, tol=tol, linear=linear
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_sparse = time.perf_counter() - t_sparse_start

    ## just stage 1, not using icho 0 or the SPAI approximation
    def solve_stage1(h, rhs):
        W = posterior_precision(h)
        V_full = torch.linalg.cholesky(W.to_dense(), upper=True)
        W_inverse_full = torch.linalg.inv(W.to_dense())
        mean = (W_inverse_full @ rhs.unsqueeze(1)).squeeze(1)
        return mean, W, V_full

    ## stage 1 and 2, not using SPAI approximation
    def solve_stage2(h, rhs):
        W = posterior_precision(h)
        if "h" in ichol0_cache and torch.equal(h, ichol0_cache["h"]):
            V = ichol0_cache["V"]
        else:
            V = ichol0(W, upper=True)
        V_ichol0 = V.to_dense() if V.is_sparse else V
        V_ichol0 = V_ichol0.cpu()
        N = V_ichol0.shape[-1]
        I = torch.eye(
            N,
            dtype=V_ichol0.dtype,
            device=V_ichol0.device
        )
        V_inverse = torch.linalg.solve_triangular(
            V_ichol0,
            I,
            upper=True
        )
        W_inverse_stage2 = V_inverse.mH @ V_inverse
        mean = (W_inverse_stage2 @ rhs.unsqueeze(1)).squeeze(1)
        return mean, W, V_ichol0

    Z_cpu = Z.cpu()
    pred_sparse_stage1, converged_stage1, iters_stage1 = newton_laplace(
        solve_stage1, score, hess, Z_cpu, max_iter=max_iter, tol=tol, linear=linear
    )
    pred_sparse_stage2, converged_stage2, iters_stage2 = newton_laplace(
        solve_stage2, score, hess, Z_cpu, max_iter=max_iter, tol=tol, linear=linear
    )
    post_mean_sparse = pred_sparse["mean"]
    post_mean_stage1 = pred_sparse_stage1["mean"]
    post_mean_stage2 = pred_sparse_stage2["mean"]

    average_K, K_i = get_average_conditioning_size(S_rho, n=n, X=X)
    dLS_sparse = -(dposterior(Y, pred_sparse) - loglik_exact)
    dLS_VL = -(ls_VL - loglik_exact)
    dLS_LR = -(ls_LR - loglik_exact)
    rrmspe_sparse = torch.mean((Y - post_mean_sparse) ** 2)/mse_exact

    dLS_sparse_stage1 = -(dposterior(Y.cpu(), pred_sparse_stage1) - loglik_exact)
    dLS_sparse_stage2 = -(dposterior(Y.cpu(), pred_sparse_stage2) - loglik_exact)
    rrmspe_sparse_stage1 = torch.mean((Y.cpu() - post_mean_stage1) ** 2)/mse_exact
    rrmspe_sparse_stage2 = torch.mean((Y.cpu() - post_mean_stage2) ** 2)/mse_exact

    return {
        "model": model,
        "rho": rho,
        "average_K": average_K,
        "N": average_K,
        "rrmspe_sparse": rrmspe_sparse.item(),
        "rrmspe_sparse_stage1": rrmspe_sparse_stage1.item(),
        "rrmspe_sparse_stage2": rrmspe_sparse_stage2.item(),
        "dLS_sparse": dLS_sparse.item(),
        "dLS_sparse_stage1": dLS_sparse_stage1.item(),
        "dLS_sparse_stage2": dLS_sparse_stage2.item(),
        "rrmspe_HV": rrmspe_HV,
        "dLS_HV": dLS_VL,
        "rrmspe_LR": rrmspe_LR,
        "dLS_LR": dLS_LR,
        "time_sparse": time_sparse,
        "time_hv": time_hv,
        "time_lr": time_lr,
        "iters_sparse": iters_sparse,
        "iters_stage1": iters_stage1,
        "iters_stage2": iters_stage2,
        "converged_sparse": converged_sparse,
        "converged_stage1": converged_stage1,
        "converged_stage2": converged_stage2,
        "converged_hv": bool(row["hv_converged"]),
        "converged_lr": bool(row["lr_converged"]),
    }


