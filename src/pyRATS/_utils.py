import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd, svdvals, eigh
from scipy.sparse.linalg import svds
from sklearn.decomposition import KernelPCA
from scipy.sparse import csr_matrix, triu, block_diag, diags
import itertools

from joblib import delayed, Parallel, parallel_backend
import multiprocess as mp_lib

from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import svd_flip
from scipy.sparse.csgraph import (
    minimum_spanning_tree,
    connected_components,
    breadth_first_order,
)
from tqdm import tqdm

import os
import sys
import time
import warnings

_MIN_MEMORY_FLOOR = 64 * 1024 * 1024  # 64 MB — always make some progress
_FALLBACK_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB if psutil is missing


def _cgroup_available_bytes():
    """Linux-only: bytes still available inside the current cgroup, or None.

    Reads cgroup v2 first, then falls back to v1. Returns None on any other
    OS, when no limit is set, or when the files can't be parsed.
    """
    if not sys.platform.startswith("linux"):
        return None
    pairs = (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    for max_path, cur_path in pairs:
        try:
            with open(max_path) as f:
                raw = f.read().strip()
            if not raw or raw == "max":
                return None
            limit = int(raw)
            # cgroup v1 uses a huge sentinel when unconstrained.
            if limit >= (1 << 62):
                return None
            with open(cur_path) as f:
                used = int(f.read().strip())
            return max(0, limit - used)
        except (OSError, ValueError):
            continue
    return None


_MEMORY_CACHE_TTL = 0.25  # seconds — fresh enough to track fit-phase allocations
_memory_cache: tuple[float, int] | None = None  # (timestamp, bytes)


def _available_memory_bytes():
    """Live probe of usable memory in bytes, cached with a short TTL.

    Sources, all combined via min():
      * psutil.virtual_memory().available scaled by 0.75 (or 4GB fallback)
      * cgroup v2/v1 remaining quota on Linux containers (avoids OOM-kill
        before psutil sees pressure)
      * PYRATS_MEMORY_LIMIT env var as a hard cap

    Result is cached for _MEMORY_CACHE_TTL seconds (default 250ms). This
    eliminates psutil and cgroup file-read overhead on hot call sites (e.g.
    iter_eval_ inside best()'s per-point loop) while still reacting to
    large allocations between fit phases — the original "cache forever" bug.
    Floored at 64MB so chunking can always make at least one row of progress.
    """
    global _memory_cache
    now = time.monotonic()
    if _memory_cache is not None and now - _memory_cache[0] < _MEMORY_CACHE_TTL:
        return _memory_cache[1]

    try:
        import psutil
        avail = int(psutil.virtual_memory().available * 0.75)
    except (ImportError, AttributeError):
        avail = _FALLBACK_MEMORY

    cgroup = _cgroup_available_bytes()
    if cgroup is not None:
        avail = min(avail, int(cgroup * 0.75))

    user_cap = os.environ.get("PYRATS_MEMORY_LIMIT")
    if user_cap:
        try:
            avail = min(avail, int(user_cap))
        except ValueError:
            pass

    result = max(avail, _MIN_MEMORY_FLOOR)
    _memory_cache = (now, result)
    return result


# Backward-compatible alias for any external caller pinned to the old name.
def _get_available_memory(n_jobs=1):
    return _available_memory_bytes() // max(1, n_jobs)


def _inner_blas_threads(n_jobs):
    """BLAS threads per outer Parallel worker, sized to avoid oversubscription.

    Apple Accelerate / OpenBLAS otherwise spawn a pool sized to all cores
    inside each outer worker, giving n_jobs * cpu_count software threads on
    cpu_count hardware threads — the n_jobs ~ cpu_count/2 cliff. Pinning to 1
    fixes that but leaves cores idle when n_jobs < cpu_count. cpu_count //
    n_jobs strikes the balance: full utilization, no contention. Floored at 1.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, n_jobs))


def lpca(X, d, U, n_jobs, verbose=False):
    """Fit PCA model on the data X.

    Parameters
    ------
    X : array-like, shape (n_samples, n_features)
        Sample data, in the form of a numpy array of shape (n_samples, n_features).

    d : int
       Intrinsic dimension of the manifold.

    U : array-like, shape (n_samples, n_neighbors)
        Indices of neighboring points for each point point.

    n_jobs : int
        Maximum number of processes to spawn.

    verbose : bool, default=False
        Write logs to stdout.

    Returns
    ------
    param : object
        Param object holding the model which stores the PCA transform.

    Notes
    ------
    PCA transformation is stored in param.Psi;
    demeaning translation is stored in param.mu.
    """

    n, p = X.shape

    param = Param("lpca")
    param.X = X
    param.Psi = np.zeros((n, p, d))
    param.mu = np.zeros((n, p))
    param.var_explained = np.zeros((n, p))
    param.n_pc_dir_chosen = np.zeros(n)

    def target_proc(p_num, chunk_sz):
        start_ind = p_num * chunk_sz
        if p_num == (n_jobs - 1):
            end_ind = n
        else:
            end_ind = (p_num + 1) * chunk_sz

        n_inds = end_ind - start_ind
        Psi = np.zeros((n_inds, p, d))
        mu = np.zeros((n_inds, p))
        var_explained = np.zeros((n_inds, p))
        n_pc_dir_chosen = np.zeros(n_inds)

        for k in range(start_ind, end_ind):
            i = k - start_ind

            U_k = U[k, :].indices
            X_k = X[U_k, :]

            xbar_k = np.mean(X_k, axis=0)[np.newaxis, :]
            X_k = (X_k - xbar_k).T
            d1 = d
            if d in X_k.shape:
                Q_k, Sigma_k, _ = svd(X_k)
            else:
                np.random.seed(42)
                v0 = np.random.uniform(0, 1, np.min(X_k.shape))
                Q_k, Sigma_k, _ = svds(X_k, k=d, which="LM", v0=v0)

            var_explained[i, :d] = Sigma_k**2
            var_explained[i, :d] /= np.sum(var_explained[i, :d])
            n_pc_dir_chosen[i] = d1

            Psi[i, :, :d1] = Q_k[:, :d1]
            mu[i, :] = xbar_k

        return start_ind, end_ind, Psi, mu, var_explained, n_pc_dir_chosen

    chunk_sz = int(n / n_jobs)
    if n_jobs == 1 or n < 1000: # Threshold for Parallel overhead
        results = [target_proc(0, n)]
    else:
        with parallel_backend("loky", inner_max_num_threads=_inner_blas_threads(n_jobs)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(target_proc)(i, chunk_sz)
                for i in tqdm(
                    range(n_jobs), desc="PCA", unit="chunk", leave=False, disable=not verbose
                )
            )

    for i in range(len(results)):
        start_ind, end_ind, Psi_, mu_, var_explained_, n_pc_dir_chosen_ = results[i]
        param.Psi[start_ind:end_ind, :] = Psi_
        param.mu[start_ind:end_ind, :] = mu_
        param.var_explained[start_ind:end_ind, :] = var_explained_
        param.n_pc_dir_chosen[start_ind:end_ind] = n_pc_dir_chosen_

    return param


def kpca(X, d, U, kernel, fit_inverse_transform, n_jobs, verbose=False):
    """Fit Kernel PCA models to the data X. Uses sklearn.decomposition.KernelPCA under the hood.

    Parameters
    ------
    X : array-like, shape (n_samples, n_features)
            Sample data, in the form of a numpy array of shape (n_samples, n_features).

    d : int
       Intrinsic dimension of the manifold.

    U : array-like, shape (n_samples, n_neighbors)
        Indices of neighboring points for each point point.

    kpca_kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} or Callable, default='linear'
        Kernel used for PCA. See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html for more information.

    kpca_fit_inverse_transform : bool, default=False
        When True, computes the inverse of the embedding transformation.
        This flag must be only be enabled when running RATS with using the kpca_kernel.
        If running pca with kpca_kernel=None, kpca_fit_inverse_transform is always True.

    n_jobs : int
        Maximum number of processes to spawn.

    verbose : bool, default-False
        Write logs to stdout.

    Returns
    ------
    param : object
        Param object holding the model which stores the PCA transform.

    Notes
    ------
    Transformations are stored as an sklearn.decomposition.KernelPCA model.
    """

    n, p = X.shape

    local_param = Param("lkpca")
    local_param.X = X
    local_param.model = np.empty(n, dtype=object)
    local_param.zeta = np.zeros(n)

    def target_proc(p_num, chunk_sz):
        start_ind = p_num * chunk_sz
        if p_num == (n_jobs - 1):
            end_ind = n
        else:
            end_ind = (p_num + 1) * chunk_sz

        model_ = np.empty(end_ind - start_ind, dtype=object)
        for k in range(start_ind, end_ind):
            model_[k - start_ind] = KernelPCA(
                n_components=d,
                kernel=kernel,
                eigen_solver="arpack",
                random_state=42,
                fit_inverse_transform=fit_inverse_transform,
            )
            U_k = U[k]
            X_k = X[U_k, :]
            model_[k - start_ind].fit(X_k)
        return start_ind, end_ind, model_

    chunk_sz = int(n / n_jobs)
    with parallel_backend("loky", inner_max_num_threads=_inner_blas_threads(n_jobs)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(target_proc)(p_num, chunk_sz)
            for p_num in tqdm(
                range(n_jobs), desc="KPCA", unit="chunk", leave=False, disable=not verbose
            )
        )

    for start_ind, end_ind, model_ in results:
        local_param.model[start_ind:end_ind] = model_

    return local_param


def sparse_matrix(indices, values):
    """Construct a sparse matrix with entries at neigh_ind and values of neigh_dist

    Parameters
    ------
    indices : array-like or sparse-matrix
        Indices where the sparse matrix should store elements.

    values ; array-like, shape (n_samples, n_values)
        Values of the sparse matrix.

    Returns
    ------
    Sparse matrix of shape (n_samples, n_samples)
    """
    if indices.dtype == "object":
        row_inds = []
        col_inds = []
        data = []
        for k in range(indices.shape[0]):
            row_inds.append(np.repeat(k, indices[k].shape[0]).tolist())
            col_inds.append(indices[k].tolist())
            data.append(values[k].tolist())
        row_inds = list(itertools.chain.from_iterable(row_inds))
        col_inds = list(itertools.chain.from_iterable(col_inds))
        data = list(itertools.chain.from_iterable(data))
    else:
        row_inds = np.repeat(np.arange(values.shape[0]), values.shape[1])
        col_inds = indices.flatten()
        data = values.flatten()
    return csr_matrix((data, (row_inds, col_inds)))


def batched_pdist(x):
    """Computes all pairwise euclidean distances of elements in x per batch.

    Parameter
    ------
    x : array-like, shape (n, k, d)
        n-batches of k-vectors of length d

    Returns
    ------
    Numpy array of euclidean distances between all pairs of vectors within a batch, shape (n, k*(k-1)/2).

    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    n, k, _ = x.shape
    # Precompute all upper-triangle pair indices (same order as the original loop)
    pair_i, pair_j = np.triu_indices(k, k=1)          # each shape (num_pairs,)
    diff = x[:, pair_i, :] - x[:, pair_j, :]          # (n, num_pairs, d)
    return np.sqrt(np.sum(diff * diff, axis=-1))        # (n, num_pairs)


def batched_procrustes_cost(X, Y):
    """Computes the cost of alignment between elements in X and Y

    Parameters
    ------
    X : array-like, shape (b, n, d)

    Y : array-like, shape (b, n, d)

    Returns
    ------
    Array of costs of procrustes alignment between X and Y, shape (b)

    Notes
    ------
    Supports broadcasting of the batch dimension of either X or Y.
    """

    muX = X.mean(1)
    muY = Y.mean(1)

    X0 = X - muX[:, None, ...]
    Y0 = Y - muY[:, None, ...]

    A = np.matmul(X0.transpose(0, 2, 1), Y0)
    S = np.linalg.svdvals(A)

    c = np.sum(X0**2, axis=(1, 2)) + np.sum(Y0**2, axis=(1, 2)) - 2 * np.sum(S, axis=1)
    return c


def custom_procrustes_batched(X, Y):
    """Computes the cost of alignment between elements in X and Y

    Parameters
    ------
    X : array-like, shape (b, n, d)

    Y : array-like, shape (b, n, d)

    Returns
    ------
    Array of costs of procrustes alignment between X and Y, shape (b)

    Notes
    ------
    Supports broadcasting of the batch dimension of either X or Y.
    """
    muX = X.mean(1)
    muY = Y.mean(1)

    X0 = X - muX[:, None, ...]
    Y0 = Y - muY[:, None, ...]

    A = np.matmul(X0.transpose(0, 2, 1), Y0)
    _, S, _ = np.linalg.svd(A, full_matrices=False)

    c = np.sum(X0**2, axis=(1, 2)) + np.sum(Y0**2, axis=(1, 2)) - 2 * np.sum(S, axis=1)
    return c


def nearest_neighbors(X, k, metric, sort_results=True, n_jobs=-1):
    """Fitting the neighborhood graph.

    Parameters
    ------
    X : array shape (n_samples, n_features)
        A 2d array containing data representing a manifold.

    k : int
        Number of neighbors of each point.

    metric : str
        Distance metric

    sort_results: bool, default=True
        Sorts neighbors by index in ascending order.

    n_jobs : int, default=-1
        Maximum number of processes to use

    Returns
    ------
    neigh_dist: array-like, shape (n_samples, k)
        Distances between neighbors

    neigh_ind: array-like, shape (n_samples, k)
        Indices of neighbors

    """

    n = len(X)
    if k > 1:
        neigh = NearestNeighbors(n_neighbors=k - 1, metric=metric, n_jobs=n_jobs)
        neigh.fit(X)
        neigh_dist, neigh_ind = neigh.kneighbors()
        neigh_dist = np.insert(neigh_dist, 0, np.zeros(n), axis=1)
        neigh_ind = np.insert(neigh_ind, 0, np.arange(n), axis=1)
        if sort_results:  # switch on for determinism
            inds = np.argsort(neigh_dist, axis=-1)
            for i in range(neigh_ind.shape[0]):
                neigh_ind[i, :] = neigh_ind[i, inds[i, :]]
                neigh_dist[i, :] = neigh_dist[i, inds[i, :]]
    else:
        neigh_dist = np.zeros((n, 1))
        neigh_ind = np.arange(n).reshape((n, 1)).astype("int")

    return neigh_dist, neigh_ind


def cost_of_moving_distortion(
    k, d_e, neigh_ind_k, U_k, local_param, c, n_C, Utilde, eta_min, eta_max, n_jobs=1
):
    """Computes the minimum cost and destination cluster possible
    when merging k with its neighboring clusters.

    Parameters
    ------
    k : int
        Index of point

    d_e : sparse matrix, shape (n_samples, n_samples)
        Distance matrix between neighbors

    neigh_ind_k : array-like, shape (n_neighbors)
        List of neighbor indices of point k.

    U_k : array-like, shape (n_neighbors)
        List of neighbor indices of point k.

    param : Param object
        Stores the parameterizations for each point.

    c : array-like (n_samples)
        Array of indices that map from datapoint to cluster.

    n_C : array-like, shape (n_samples)
        Array of indices that map from cluster to number of points per cluster

    Utilde : list
        List of neighbor indices for each sample.

    eta_min : int
        Minimum allowed size of the clusters underlying the intermediate views.
        The values must be >= 1.

    eta_max : int
        Maximum allowed size of the clusters underlying the intermediate views.
        The value must be > eta_min.

    Returns
    ------
    cost_k : float
        Minmum cost of moving from k to its neighbor's cluster

    dest_k : int
        Destination cluster index

    """

    c_k = c[k]
    # Compute |C_{c_k}|
    n_C_c_k = n_C[c_k]

    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1

    # Compute neighboring clusters c_{U_k} of x_k
    c_U_k = c[neigh_ind_k]
    c_U_k_uniq = np.unique(c_U_k).tolist()
    cost_x_k_to = np.zeros(len(c_U_k_uniq)) + np.inf

    # Reconstruct the set from neigh_ind_k inside this process: pickling and
    # unpickling a set rebuilds the hash table with a different bucket layout,
    # so list(U_k) iterates in a different order in the parent vs. a loky
    # worker. Building the set from the same numpy array in each process gives
    # the same iteration order everywhere.
    U_k_list = list(set(neigh_ind_k))

    # Iterate over all m in c_{U_k}
    i = 0
    for m in c_U_k_uniq:
        if m == c_k:
            i += 1
            continue

        # Compute |C_{m}|
        n_C_m = n_C[m]
        # Check if |C_{m}| < eta_{max}. If not
        # then mth cluster has reached the max
        # allowed size of the cluster. Move on.
        if n_C_m >= eta_max:
            i += 1
            continue

        # Check if |C_{m}| >= |C_{c_k}|. If yes, then
        # mth cluster satisfies all required conditions
        # and is a candidate cluster to move x_k in.
        if n_C_m >= n_C_c_k:
            # Compute union of Utilde_m U_k
            U_k_U_Utilde_m = list(U_k.union(Utilde[m]))
            # Compute the cost of moving x_k to mth cluster,
            # that is cost_{x_k \rightarrow m}
            cost_x_k_to[i] = compute_zeta(
                d_e[np.ix_(U_k_U_Utilde_m, U_k_U_Utilde_m)],
                local_param.eval_(m, U_k_U_Utilde_m),
            )

        i += 1

    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
    else:
        dest_k = c_U_k_uniq[dest_k]

    return cost_k, dest_k


def compute_zeta(d_e_mask0, Psi_k_mask):
    """Computes the local distortion zeta for a given parameterization and neighborhood distances.
    
    Parameters
    ----------
    d_e_mask0 : sparse matrix or array-like
        Pairwise distances between neighbors in the original space.
    Psi_k_mask : array-like
        Embedded coordinates of the neighbors.
    """
    # For small k (typical in clustering), dense operations are MUCH faster than
    # sparse triu/nonzero lookups.
    d_e_mask = d_e_mask0.toarray() if hasattr(d_e_mask0, "toarray") else d_e_mask0
    if d_e_mask.shape[0] <= 1:
        return 1
    
    # Use squareform/pdist for efficient pair extraction on small dense matrices
    d_e_mask_ = squareform(d_e_mask)
    mask = d_e_mask_ != 0
    if not np.any(mask):
        return 1
    
    d_e_vals = d_e_mask_[mask]
    dist_embedded = pdist(Psi_k_mask)[mask]
    
    disc_lip_const = dist_embedded / d_e_vals
    return np.max(disc_lip_const) / (np.min(disc_lip_const) + 1e-12)


def cost_of_moving_alignment_error(
    k, d_e, neigh_ind_k, U_k, param, c, n_C, Utilde, eta_min, eta_max, n_jobs=1
):
    """Computes the minimum cost and destination cluster possible
    when merging k with its neighboring clusters.

    Parameters
    ------
    k : int
        Index of point

    d_e : sparse matrix, shape (n_samples, n_samples)
        Distance matrix between neighbors

    neigh_ind_k : array-like, shape (n_neighbors)
        List of neighbor indices of point k.

    U_k : array-like, shape (n_neighbors)
        List of neighbor indices of point k.

    param : Param object
        Stores the parameterizations for each point.

    c : array-like (n_samples)
        Array of indices that map from datapoint to cluster.

    n_C : array-like, shape (n_samples)
        Array of indices that map from cluster to number of points per cluster

    Utilde : list
        List of neighbor indices for each sample.

    eta_min : int
        Minimum allowed size of the clusters underlying the intermediate views.
        The values must be >= 1.

    eta_max : int
        Maximum allowed size of the clusters underlying the intermediate views.
        The value must be > eta_min.

    Returns
    ------
    cost_k : float
        Minmum cost of moving from k to its neighbor's cluster

    dest_k : int
        Destination cluster index

    """

    c_k = c[k]
    # Compute |C_{c_k}|
    n_C_c_k = n_C[c_k]

    # Check if |C_{c_k}| < eta_{min}
    # If not then c_k is already
    # an intermediate cluster
    if n_C_c_k >= eta_min:
        return np.inf, -1

    # Compute neighboring clusters c_{U_k} of x_k
    c_U_k = c[neigh_ind_k]
    c_U_k_uniq = np.unique(c_U_k)
    cost_x_k_to = np.zeros(len(c_U_k_uniq)) + np.inf

    # Reconstruct the set from neigh_ind_k inside this process: pickling and
    # unpickling a set rebuilds the hash table with a different bucket layout,
    # so list(U_k) iterates in a different order in the parent vs. a loky
    # worker. Building the set from the same numpy array in each process gives
    # the same iteration order everywhere.
    U_k_list = list(set(neigh_ind_k))

    neighbor_mask = (
        (n_C[c_U_k_uniq] < eta_max) & (n_C[c_U_k_uniq] >= n_C_c_k) & (c_U_k_uniq != c_k)
    )

    # c_U_k_uniq_k = np.union1d(c_U_k_uniq[neighbor_mask], c_k)
    c_U_k_uniq_k = np.union1d(c_U_k_uniq[neighbor_mask], k)
    evals = param.batched_eval_(
        c_U_k_uniq_k,
        np.broadcast_to(U_k_list, [len(c_U_k_uniq_k), len(U_k_list)]),
        n_jobs=n_jobs
    )

    m = c_U_k_uniq_k == k
    cost_x_k_to[neighbor_mask] = custom_procrustes_batched(
        evals[~m] if len(c_U_k_uniq_k) > neighbor_mask.sum() else evals, evals[m]
    )

    # find the cluster with minimum cost
    # to move x_k in.
    dest_k = np.argmin(cost_x_k_to)
    cost_k = cost_x_k_to[dest_k]
    if cost_k == np.inf:
        dest_k = -1
    else:
        dest_k = c_U_k_uniq[dest_k]

    return cost_k, dest_k


def best(d_e, U, param, eta_min, eta_max, cost_fn, verbose, n_jobs):
    """Greedy merging points to form clusters.

    Parameters
    ------
    d_e : sparse-matrix
        Symmetric matrix of distances between neighboring points

    U : sparse-matrix
        Matrix of indices of neighboring points

    param: Param object
        Stores the parameterizations for each point.

    eta_min : int, default=5
        Minimum number of points per cluster. A cluster is formed by merging local embeddings that minimize cost_fn_name.
        Larger clusters improve the runtime, and regularise noisy manifolds.
        The value must be >= 1.

    eta_max : int, default=25
        Maximum allowed size of the clusters.
        The value must be > eta_min.

    cost_fn_name : {'alignment', 'distortion'}, default='alignment'
        'alignment': Alignment error should be prefered when runtime matters or dealing with noisy manifolds.
        'distortion': Distortion is slow, but works well on low noise manifolds.

    verbose : bool
        If True, print logs to stdout.

    n_jobs : int
        Maximum number of of processes allowed.

    Returns
    ______

    c : array-like, shape (n_samples)
        The cluster index for each datapoint.

    n_C: array-like, shape (n_clusters)
        The number of datapoints per cluster.
    """

    assert cost_fn in [
        "alignment",
        "distortion",
    ], f"{cost_fn} is not implemented. Choose between 'alignment' and 'distortion'."
    cost_of_moving = (
        cost_of_moving_alignment_error
        if cost_fn == "alignment"
        else cost_of_moving_distortion
    )

    n = d_e.shape[0]
    c = np.arange(n)
    n_C = np.zeros(n) + 1
    Clstr = list(map(set, np.arange(n).reshape((n, 1)).tolist()))
    indices = U.indices
    indptr = U.indptr
    Utilde = []
    U_ = []
    neigh_ind = []
    for i in range(n):
        col_inds = indices[indptr[i] : indptr[i + 1]]
        Utilde.append(set(col_inds))
        U_.append(set(col_inds))
        neigh_ind.append(col_inds)

    neigh_ind = np.array(neigh_ind)

    cost = np.zeros(n) + np.inf
    dest = np.zeros(n, dtype="int") - 1
    U_csc = U.tocsc()

    # Vary eta from 2 to eta_{min}
    for eta in tqdm(
        range(2, eta_min + 1), desc="Intermediate views", disable=not verbose
    ):
            # tqdm.write(
            #     "#non-empty views with sz < %d = %d"
            #     % (eta, np.sum((n_C > 0) * (n_C < eta)))
            # )
            # tqdm.write("#nodes in views with sz < %d = %d" % (eta, np.sum(n_C[c] < eta)))

        def target_proc(p_num, chunk_sz, n_, Utilde, n_C, c):
            start_ind = p_num * chunk_sz
            if p_num == (n_jobs - 1):
                end_ind = n_
            else:
                end_ind = (p_num + 1) * chunk_sz

            cost_ = np.zeros(end_ind - start_ind) + np.inf
            dest_ = np.zeros(end_ind - start_ind, dtype="int") - 1
            for i, k in enumerate(range(start_ind, end_ind)):
                cost_[i], dest_[i] = cost_of_moving(
                    k, d_e, neigh_ind[k], U_[k], param, c, n_C, Utilde, eta, eta_max, n_jobs=n_jobs
                )
            return start_ind, end_ind, cost_, dest_

        chunk_sz = int(n / n_jobs)
        if n_jobs == 1 or n < 1000:
            results = [target_proc(0, n, n, Utilde, n_C, c)]
        else:
            # Use multiprocess.Process (fork) rather than joblib/loky (spawn).
            # cost_of_moving reads Utilde[m], a Python set whose iteration
            # order depends on its hash-table bucket layout. Spawn re-pickles
            # the set and rebuilds the table with a different layout in each
            # worker, so list(Utilde[m]) iterates in a different order than
            # in the parent — permuting the rows/cols passed to
            # local_param.eval_ and d_e, producing float-LSB cost diffs that
            # flip argmin tie-breaks and yield divergent cluster
            # assignments. Fork shares the parent's address space via
            # copy-on-write, so the children see the exact same set layout
            # and produce bit-identical costs to a serial run.
            q = mp_lib.Queue()
            def _worker(p_num):
                q.put(target_proc(p_num, chunk_sz, n, Utilde, n_C, c))
            procs = [
                mp_lib.Process(target=_worker, args=(p_num,), daemon=True)
                for p_num in range(n_jobs)
            ]
            for p in procs:
                p.start()
            results = [q.get() for _ in range(n_jobs)]
            for p in procs:
                p.join()
        for start_ind, end_ind, cost_, dest_ in results:
            cost[start_ind:end_ind] = cost_
            dest[start_ind:end_ind] = dest_

        # Compute point with minimum cost
        # Compute k and cost^*
        k = np.argmin(cost)
        cost_star = cost[k]

        # if verbose:
        #     tqdm.write("Costs computed when eta = %d." % eta)

        # Loop until minimum cost is inf
        pbar_merge = None
        if verbose:
            pbar_merge = tqdm(
                total=np.sum(n_C[c] < eta),
                desc="Merging",
                unit="pts",
                leave=False,
            )

        total_len_S = 0
        ctr = 0
        while cost_star < np.inf:
            # Move x_k from cluster s to
            # dest_k and update variables
            s = c[k]
            dest_k = dest[k]
            c[k] = dest_k
            n_C[s] -= 1
            n_C[dest_k] += 1
            Clstr[s].remove(k)
            Clstr[dest_k].add(k)
            Utilde[dest_k] = U_[k].union(Utilde[dest_k])
            Utilde[s] = set(itertools.chain.from_iterable(neigh_ind[list(Clstr[s])]))

            # Compute the set of points S for which
            # cost of moving needs to be recomputed
            if n_C[s] > 0:
                S_ = (
                    (c == dest_k)
                    | (dest == dest_k)
                    | np.array(U_csc[:, list(Clstr[s])].sum(1), dtype=bool).flatten()
                )
            else:
                S_ = (c == dest_k) | (dest == dest_k) | (dest == s)
            S = np.where(S_)[0].tolist()
            len_S = len(S)
            total_len_S += len_S
            ctr += 1

            for k in S:
                cost[k], dest[k] = cost_of_moving(
                    k, d_e, neigh_ind[k], U_[k], param, c, n_C, Utilde, eta, eta_max, n_jobs=1
                )

            if verbose:
                pbar_merge.update(1)

            k = np.argmin(cost)
            cost_star = cost[k]

        if verbose:
            pbar_merge.close()
            # tqdm.write(
            #     "ctr=%d, total_len_S=%d, avg_len_S=%0.3f"
            #     % (ctr, total_len_S, total_len_S / (ctr + 1e-12))
            # )
            # tqdm.write(
            #     "Remaining #nodes in views with sz < %d = %d"
            #     % (eta, np.sum(n_C[c] < eta))
            # )
            # tqdm.write("Done with eta = %d." % eta)

    return c, n_C


def compute_seq_of_views(
    d,  # embedding dimension
    i_mat,  # incidence matrix |#views| x |#points|
    overlap,  # size of overlap between views |#views| x |#views|
    param,  # d-dimensional parameterization of points in each view
    n_forced_clusters,
    verbose,
    n_jobs,
):
    """Connect intermediate views to a tree structure.

    Parameters
    ------
    d : int
        Embedding dimension

    i_mat : sparse-matrix, shape (n_samples, n_intermed_views)
        Incidence matrix

    overlap : sparse-matrix, shape (n_intermed_views, n_intermed_views)
        Size of overlap between intermediate views

    param : Param object
        Stores parameterizations for each point

    n_forced_clusters : int
        Minimum no. of clusters to force in the embeddings.

    verbose : bool
        If True, print logs to stdout.

    n_jobs : int
        Maximum number of processes to use.

    Return
    ------
    seq_of_views_in_cluster : array-like, shape (n_samples)

    parents_of_views_in_cluster : array-like, shape (n_samples)

    cluster_of_view : array-like, shape (n_samples)

    """

    n_views = i_mat.shape[0]

    # W_{mm'} = W_{m'm} measures the ambiguity between
    # the two embeddings of the points on the overlap
    # between mth and m'th intermediate views
    W_rows, W_cols = triu(overlap).nonzero()
    n_elem = W_rows.shape[0]
    overlap_svals = np.zeros((n_elem, d))
    chunk_sz = int(n_elem / n_jobs)

    def target_proc(p_num):
        start_ind = p_num * chunk_sz
        if p_num == (n_jobs - 1):
            end_ind = n_elem
        else:
            end_ind = (p_num + 1) * chunk_sz
        overlap_svals_ = np.zeros((end_ind - start_ind, d))
        for i in range(start_ind, end_ind):
            m = W_rows[i]
            mpp = W_cols[i]
            mask = i_mat[m, :].multiply(i_mat[mpp, :]).nonzero()[1]
            V_mmp = param.eval_(m, mask)
            V_mpm = param.eval_(mpp, mask)
            Vbar_mmp = V_mmp - np.mean(V_mmp, 0)[np.newaxis, :]
            Vbar_mpm = V_mpm - np.mean(V_mpm, 0)[np.newaxis, :]
            # Compute ambiguity of the overlaps captured by singular values
            svdvals_ = svdvals(np.dot(Vbar_mmp.T, Vbar_mpm))
            # overlap_svals_[i-start_ind] = svdvals_[-1]
            overlap_svals_[i - start_ind, :] = svdvals_
        return (start_ind, end_ind, overlap_svals_)

    if n_jobs == 1 or n_elem < 1000:
        res = [target_proc(p_num) for p_num in range(n_jobs)]
    else:
        with parallel_backend("loky", inner_max_num_threads=_inner_blas_threads(n_jobs)):
            res = list(Parallel(n_jobs=n_jobs,
                                return_as="generator_unordered")(
                delayed(target_proc)(p_num) for p_num in range(n_jobs)
            ))
    for value in res:
        start_ind, end_ind, overlap_svals_ = value
        overlap_svals[start_ind:end_ind, :] = overlap_svals_

    # In most cases, when almost all overlaps have rank d, 
    # W_data = overlap_svals[:,-1], the d-th singular value of the overlap
    # is sufficient to determine the priority of the overlaps.
    # overlap_svals[overlap_svals<tol] = 0
    W_data = overlap_svals[:, -1]
    # But in cases when a low-dimensional object is embedded in a high-dimensional space,
    # or when d is higher than the local intrinsic dimension at several points,
    # the d-th singular value will be zero at all the points with less than d local intrinsic dimension.
    # In such cases, we look at the (d-1)-th singular value (and so on) to determine the priority of the overlaps.
    for i in range(overlap_svals.shape[1] - 2, -1, -1):
        mask = W_data == 0
        n_zero_elem = np.sum(mask)
        if n_zero_elem == 0:
            break
        temp = overlap_svals[mask, i]
        if n_zero_elem < len(W_data):
            temp2 = 0.5 * W_data[~mask]
            W_data[mask] = temp * (np.min(temp2) / (np.max(temp) + 1e-12))
        else:
            W_data[mask] = temp

    W = csr_matrix(
        (W_data, (W_rows, W_cols)), shape=(n_views, n_views)
    )  # strict upper triangular
    W = W + W.T

    n_comp, comp_labels = connected_components(W, directed=False, return_labels=True)

    # Remove edges to force clusters if desired
    if n_forced_clusters > n_comp:
        inds = np.argsort(W.data)[-(n_forced_clusters - n_comp) :]
        W.data[inds] = 0
        W.eliminate_zeros()
        n_comp, comp_labels = connected_components(
            W, directed=False, return_labels=True
        )

    if verbose:
        print("No. of connected components (manifolds):", n_comp)

    # Create a sequence of views for each cluster representing a manifold
    seq_of_views_in_cluster = []
    parents_of_views_in_cluster = []
    cluster_of_view = np.zeros(n_views, dtype=int)

    for i in range(n_comp):
        views_in_this_comp = np.where(comp_labels == i)[0]
        n_views_in_this_comp = len(views_in_this_comp)
        W_ = W[views_in_this_comp, :][:, views_in_this_comp].copy()
        if n_views_in_this_comp > 1:
            # Compute maximum spanning tree/forest of W
            T = minimum_spanning_tree(-W_)

            # center_i = np.argmax(n_C[views_in_this_comp])
            center_i = center_of_tree(T)

            seq, rho_ = breadth_first_order(
                T, center_i, directed=False
            )  # (ignores edge weights)
            seq = views_in_this_comp[seq]
            mask = rho_ > 0
            rho_[mask] = views_in_this_comp[rho_[mask]]
            rho = np.zeros(n_views, dtype=int) - 9999
            rho[views_in_this_comp] = rho_
        else:
            seq = views_in_this_comp
            rho = np.zeros(n_views, dtype=int) - 9999

        seq_of_views_in_cluster.append(seq)
        parents_of_views_in_cluster.append(rho)
        cluster_of_view[seq] = i

    return seq_of_views_in_cluster, parents_of_views_in_cluster, cluster_of_view


def center_of_tree(T):
    """Compute the center node of a tree T"""

    s1, pred1 = breadth_first_order(T, 0, directed=False)
    s2, pred2 = breadth_first_order(T, s1[-1], directed=False)
    nodes_on_longest_path = [s2[-1]]
    pred = 0
    while pred >= 0:
        pred = pred2[nodes_on_longest_path[-1]]
        nodes_on_longest_path.append(pred)
    n = len(nodes_on_longest_path)
    return nodes_on_longest_path[n // 2]


def compute_init_embedding(
    d,
    Utilde,
    param,
    seq_of_intermed_views_in_cluster,
    parents_of_intermed_views_in_cluster,
    C,
    verbose,
):
    """Initial alignment of datapoints in embedding via Procrustes alignment.

    Parameter
    ------
    d : int
        Embedding dimension

    seq_of_intermed_views_in_cluster : array-like, shape (n_samples)

    parents_of_intermed_views_in_cluster : array-like, shape (n_sampels)

    C : sparse-matrix, shape (n_samples, n_samples)

    verbose : bool
        If True, print logs to stdout.

    Returns
    ------
    y : array-like, shape (n_samples, d)
        Points in new space after initial alignment

    """

    M, n = Utilde.shape

    param.T = np.tile(np.eye(d), [M, 1, 1])
    param.v = np.zeros((M, d))
    y = np.zeros((n, d))

    n_clusters = len(seq_of_intermed_views_in_cluster)

    # Boolean array to keep track of already visited views
    is_visited_view = np.zeros(M, dtype=bool)

    for i in range(n_clusters):
        # First view global embedding is same as intermediate embedding
        seq = seq_of_intermed_views_in_cluster[i]
        rho = parents_of_intermed_views_in_cluster[i]
        seq_0 = seq[0]
        is_visited_view[seq_0] = True
        y[C[seq_0, :].indices, :] = param.eval_(
            seq_0, C[seq_0, :].indices
        )
        y, is_visited_view = procrustes_init(
            seq, rho, y, is_visited_view, d, Utilde, C, param
        )

    if verbose:
        err = compute_alignment_err(d, Utilde, param, verbose)
        print("Alignment error: %0.3f" % (err / Utilde.nnz), flush=True)

    return y


def compute_incidence_matrix_in_embedding(y, C, k, nu, metric="euclidean"):
    """

    Parameters
    ------
    y : array-like, shape (n_samples, d)
        Data in embedding

    C : sparse-matrix, shape (n_clusters, n_samples)

    k : int
        Number of neighbors

    nu : int
        The ratio of the size of local views in the embedding against those
        in the data.

    metric : str, default='euclidean'
        Distance metric in the embedding space.

    Returns
    ------
    Utildeg : sparse-matrix, shape (n_clusters, n_clusters)

    """

    M, n = C.shape
    k_ = min(int(k * nu), n - 1)
    _, neigh_indg = nearest_neighbors(y, k_, metric)
    Ug = sparse_matrix(neigh_indg, np.ones(neigh_indg.shape, dtype=bool))
    Utildeg = C.dot(Ug).astype(bool)
    return Utildeg


def compute_final_embedding(
    y,
    d,
    Utilde,
    C,
    param,
    to_tear,
    patience,
    max_iter,
    max_internal_iter,
    tol,
    nu,
    k,
    metric,
    alpha,
    verbose,
):
    """Align clusters via Riemannian Gradient Descent.

        Parameters
        ------
        y : array-like, shape (n_samples, d)
            Initial embedding of data.

        d : int
            Embedding dimension

        Utilde : sparse-matrix, shape (n_clusters, n_samples)
            Bipartite graph

        C : sparse-matrix, shape (n_clusters, n_samples)


        param : Param object
            Holds parameterisations for each point.

        to_tear : bool
            Whether to tear the manifold.

        patience : int
            Patience epochs

        max_iter : int
            Number of iterations to refine the global embedding for.
            In total Riemannian gradient descent is run for max_iter * max_internal_iter iterations.
            For every iter in max_iter, the alignment of points in the embedding is recomputed.

        max_internal_iter : int
            The number of internal iterations used by Riemannian Gradient Descent.

        tol : float
            The tolerance level for the relative change in the alignment error and the
            relative change in the size of the tear.

        nu : int
            The ratio of the size of local views in the embedding against those
            in the data.

        k : int
            Neighborhood size for local view fitting.

        metric : str, default='euclidean'
            Metric assumed on the embedding. Currently only euclidean is supported.

        alpha : float
            The step size used during Riemannian gradient descent.

        verbose : bool
            If True, print logs to stdout.

        Returns
        ------
        y : array-like, shape (n_samples, d)
            Aligned points in the embedded space
    s
    """

    np.random.seed(42)  # for reproducbility

    patience_ctr = patience
    prev_err = None
    prev_edges = None
    Utilde_t = Utilde.copy()

    # Refine global embedding y
    for it0 in tqdm(
        range(max_iter), desc="RGD alignment", unit="iter", disable=not verbose
    ):

        if to_tear:
            Utildeg = compute_incidence_matrix_in_embedding(y, C, k, nu, metric)
            Utilde_t = Utildeg.multiply(Utilde)
            Utilde_t.eliminate_zeros()

        y = rgd_alignment(d, Utilde_t, param, max_internal_iter, alpha, verbose)
        if (
            patience_ctr < max_iter or verbose
        ):  # If it makes sense to compute the error or verbose
            err = compute_alignment_err(d, Utilde_t, param, verbose)
            E_Gamma_t = Utilde_t.nnz
            err = err / E_Gamma_t
            if prev_err is not None:
                if (np.abs(err - prev_err) / (prev_err + 1e-12) < tol) and (
                    np.abs(E_Gamma_t - prev_edges) / (prev_edges + 1e-12) < tol
                ):
                    patience_ctr -= 1
                else:
                    patience_ctr = patience
            prev_err = err
            prev_edges = E_Gamma_t

            if patience_ctr <= 0:
                break

    if to_tear:
        Utildeg = compute_incidence_matrix_in_embedding(y, C, k, nu, metric)
        return y, Utildeg

    return y, None


def rgd_alignment(d, Utilde, param, max_internal_iter, alpha, verbose):
    """Riemannian Gradient Descent (RGD)

    Parameters
    ------
    d : int
        Embedding dimension

    Utilde : sparse-matrix, shape (n_clusters, n_samples)
        Bipartite graph

    param : Param object
        Stores parameterisations for all points

    max_internal_iter : int
        RGD alignment iterations

    alpha : float
        Step size

    verbose : bool
        If True, print logs to stdout.

    Returns
    ------
    Geometric mean of points after alignment.

    """

    def unique_qr(A):
        Q, R = np.linalg.qr(A)
        signs = 2 * np.diagonal(R >= 0, axis1=1, axis2=2) - 1
        Q *= signs[:, None, :]
        return Q

    def update(alpha, max_iter, O, CC, M, d):
        O_ = np.reshape(np.asarray(O), (d, d, M), order="F").transpose(2, 0, 1)
        CC_ = np.reshape(np.asarray(CC), (d, M, d, M), order="F")

        for _ in range(max_iter):
            # xi_ = np.einsum('imj,jikb->bkm', O_, CC_)
            xi_ = np.tensordot(O_, CC_, axes=([0, 2], [1, 0])).transpose(2, 1, 0)
            h = O_ @ xi_
            h_skew = h.transpose(0, -1, -2) - h
            O_ = unique_qr(O_ - alpha * h_skew @ O_)

        return O_.transpose(1, 2, 0).reshape(O.shape, order="F")

    CC, Lpinv_BT, _, _ = build_ortho_optim(d, Utilde, param, verbose)
    M, n = Utilde.shape

    Tstar = update(
        alpha, max_internal_iter, np.tile(np.eye(d), (1, M)), CC, M, d
    )  # At each iteration compute a new S starting with S = I

    Zstar = Tstar.dot(Lpinv_BT.transpose())  # update y_k's

    Tstar_ = Tstar.reshape((len(Tstar), d, -1), order="F").transpose(2, 1, 0)
    param.T = np.matmul(param.T, Tstar_)  # update transformations of individual points
    param.v = np.matmul(param.v[:, np.newaxis, :], Tstar_)[:,0,:] + Zstar[:, n:].T

    return Zstar[:, :n].T


def compute_Lpinv_helpers(W):
    M, n = W.shape
    B_ = W.transpose().tocsr().astype("float")
    D_1 = np.asarray(B_.sum(axis=1))
    D_2 = np.asarray(B_.sum(axis=0))
    D_1_inv_sqrt = np.sqrt(1 / D_1).flatten()
    D_2_inv_sqrt = np.sqrt(1 / D_2).flatten()
    
    #B_tilde = B_.multiply(D_2_inv_sqrt).multiply(D_1_inv_sqrt)
    # Create sparse diagonal matrices
    D1_diag = diags(D_1_inv_sqrt)
    D2_diag = diags(D_2_inv_sqrt)
    
    # Sparse matrix multiplication is MUCH faster than .multiply()
    B_tilde = D1_diag @ B_ @ D2_diag

    # U12, SS, VT = svd(B_tilde.todense(), full_matrices=False)
    # --- OPTIMIZED SVD: Gram Matrix Approach ---
    if n >= M:
        # B_tilde is (n, M). C becomes a tiny (M, M) dense matrix.
        # Sparse matrix multiplication here is incredibly fast.
        C = (B_tilde.T @ B_tilde).todense()
        S2, V = eigh(C)
        
        # eigh returns ascending order; reverse to match standard SVD output
        idx = np.argsort(S2)[::-1]
        S2 = S2[idx]
        V = V[:, idx]
        
        # Recover singular values and right singular vectors
        SS = np.sqrt(np.maximum(S2, 0))
        VT = V.T
        
        # Recover left singular vectors: U = B_tilde @ V @ diag(1/SS)
        with np.errstate(divide='ignore', invalid='ignore'):
            SS_inv = np.where(SS > 1e-10, 1.0 / SS, 0.0)
            
        U12 = (B_tilde @ V) * SS_inv[np.newaxis, :]
    else:
        # Fallback if M > n: Compute (n, n) Gram matrix instead
        C = (B_tilde @ B_tilde.T).todense()
        S2, U12 = eigh(C)
        
        idx = np.argsort(S2)[::-1]
        S2 = S2[idx]
        U12 = U12[:, idx]
        
        SS = np.sqrt(np.maximum(S2, 0))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            SS_inv = np.where(SS > 1e-10, 1.0 / SS, 0.0)
            
        VT = (U12.T @ B_tilde) * SS_inv[:, np.newaxis]
    # -------------------------------------------
    U12, VT = svd_flip(U12, VT)

    V = VT.T
    mask = np.abs(SS - 1) < 1e-6
    m_1 = np.sum(mask)
    Sigma = np.expand_dims(SS[m_1:], 1)
    Sigma_1 = 1 / (1 - Sigma**2)
    Sigma_2 = Sigma * Sigma_1
    U1 = U12[:, :m_1]
    U2 = U12[:, m_1:]
    V1 = V[:, :m_1]
    V2 = V[:, m_1:]
    return [D_1_inv_sqrt[:,None], D_2_inv_sqrt[None,:], U1, U2, V1, V2, Sigma_1, Sigma_2]


# Ngoc-Diep Ho, Paul Van Dooren, On the pseudo-inverse of the Laplacian of a bipartite graph
def compute_Lpinv_MT(Lpinv_helpers, B):
    D_1_inv_sqrt, D_2_inv_sqrt, U1, U2, V1, V2, Sigma_1, Sigma_2 = Lpinv_helpers
    n = D_1_inv_sqrt.shape[0]
    Md = B.shape[0]
    M = B.shape[1] - n

    B_mean = np.array(B.mean(axis=1))
    if len(B_mean.shape) == 1:
        B_mean = B_mean[:, None]

    B1 = B[:, :n]
    B2 = B[:, n:]

    # Optimized matrix-vector products using identities to avoid full dense B_n
    # Identity: U^T (diag(D) (B - mu 1^T))^T = (U^T diag(D)) B^T - (U^T diag(D) 1) mu^T
    
    # Compute U^T * B1T terms
    U1T_D1 = (U1 * D_1_inv_sqrt).T  # (m1, n)
    U1TB1T = (U1T_D1 @ B1.T) - (U1T_D1.sum(axis=1)[:, None] @ B_mean.T)

    U2T_D1 = (U2 * D_1_inv_sqrt).T  # (M-m1, n)
    U2TB1T = (U2T_D1 @ B1.T) - (U2T_D1.sum(axis=1)[:, None] @ B_mean.T)

    # Compute V^T * B2T terms
    # D_2_inv_sqrt is (1, M)
    V1T_D2 = (V1 * D_2_inv_sqrt.T).T  # (m1, M)
    V1TB2T = (V1T_D2 @ B2.T) - (V1T_D2.sum(axis=1)[:, None] @ B_mean.T)

    V2T_D2 = (V2 * D_2_inv_sqrt.T).T  # (M-m1, M)
    V2TB2T = (V2T_D2 @ B2.T) - (V2T_D2.sum(axis=1)[:, None] @ B_mean.T)

    # B1T and B2T are needed for the final sum, but we only materialize them once
    # B1T = D_1_inv_sqrt * (B - B_mean)[:, :n].T
    B1T = (B1.T.multiply(D_1_inv_sqrt)).toarray() - (D_1_inv_sqrt @ B_mean.T)

    temp1 = (
        -0.75 * (U1 @ U1TB1T)
        - 0.25 * (U1 @ V1TB2T)
        + (U2 @ ((Sigma_1 - 1) * U2TB1T))
        + (U2 @ (Sigma_2 * V2TB2T))
        + B1T
    )
    temp1 = temp1 * D_1_inv_sqrt

    temp2 = (
        -0.25 * (V1 @ U1TB1T)
        + 0.25 * (V1 @ V1TB2T)
        + (V2 @ (Sigma_2 * U2TB1T))
        + (V2 @ (Sigma_1 * V2TB2T))
    )
    temp2 = temp2 * D_2_inv_sqrt.T

    temp = np.concatenate((temp1, temp2), axis=0)
    temp = temp - np.mean(temp, axis=0, keepdims=True)
    return temp


def compute_CC(D, B, Lpinv_BT):
    CC = D - B.dot(Lpinv_BT)
    return 0.5 * (CC + CC.T)


def build_ortho_optim(d, Utilde, param, verbose):
    """Compute the Graph-Laplacian's inverse times B^\top."""
    M, n = Utilde.shape
    W = Utilde.astype(float)
    W_vals_all = W.data
    
    # Vectorized construction of D and B components
    D_list = []
    
    # We still loop over M to build B values, but we use pre-allocated arrays
    # or better, compute B values in blocks.
    B_data_vals = []
    B_cluster_vals = []
    B_cols = []
    
    for i in range(M):
        Utilde_i = Utilde[i, :].indices
        X_ = param.eval_(i, Utilde_i)
        weights_i = W_vals_all[W.indptr[i]:W.indptr[i+1]]
        sqrt_p_ki = np.sqrt(weights_i[:, None])
        
        # Weighted embeddings for D: sqrt(W) * X
        X_weighted = sqrt_p_ki * X_
        D_list.append(X_weighted.T @ X_weighted)

        # Weighted embeddings for B: W * X
        X_B = weights_i[:, None] * X_
        B_data_vals.append(X_B.T.flatten())
        B_cluster_vals.append(np.sum(-X_B.T, axis=1))
        B_cols.append(Utilde_i)

    D = block_diag(D_list, format="csr")
    
    # Efficiently construct B
    B_data_vals = np.concatenate(B_data_vals)
    B_cluster_vals = np.concatenate(B_cluster_vals)
    
    B_cols_data = []
    for i in range(M):
        B_cols_data.append(np.tile(B_cols[i], d))
    B_cols_data = np.concatenate(B_cols_data)
    
    # Row indices for data values
    counts = np.diff(W.indptr)
    B_rows_data = np.repeat(np.arange(M) * d, counts * d) + np.tile(np.arange(d), len(B_cols_data) // d)
    # Wait, the above tiling is not quite right if counts vary.
    # Correct rows: repeat each row index (i*d + offset)
    B_rows_data = []
    for i in range(M):
        r = np.arange(i * d, (i + 1) * d)
        B_rows_data.append(np.repeat(r, counts[i]))
    B_rows_data = np.concatenate(B_rows_data)

    # B indices for cluster nodes
    B_rows_cluster = np.arange(M * d)
    B_cols_cluster = np.repeat(np.arange(n, n + M), d)
    
    # Combine everything
    B_row_all = np.concatenate([B_rows_cluster, B_rows_data])
    B_col_all = np.concatenate([B_cols_cluster, B_cols_data])
    B_val_all = np.concatenate([B_cluster_vals, B_data_vals])
    
    B = csr_matrix((B_val_all, (B_row_all, B_col_all)), shape=(M * d, n + M))

    Lpinv_helpers = compute_Lpinv_helpers(W)
    Lpinv_BT = compute_Lpinv_MT(Lpinv_helpers, B)
    CC = compute_CC(D, B, Lpinv_BT)

    return CC, Lpinv_BT, D, B


# unscaled alignment error
def compute_alignment_err(d, Utilde, intermed_param, verbose):
    """Compute alignment error of data in new space.

    Parameters
    ------
    d : int
        Embedding dimension

    Utilde : sparse-matrix, shape (n_clusters, n_samples)
        Bipartite graph

    param : Param object
        Stores parameterisations for all points
    """

    CC, _, _, _ = build_ortho_optim(d, Utilde, intermed_param, verbose)
    M, n = Utilde.shape

    CC_mask = np.tile(np.eye(d, dtype=bool), (M, M))
    return np.sum(CC[CC_mask])


def custom_procrustes(X, Y, compute_cost=False):
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    A = np.dot(X0.T, Y0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    v = muX - np.dot(muY, T)

    if compute_cost:
        c = np.sum(X0**2) + np.sum(Y0**2) - 2 * np.sum(S)
        return T, v, c
    else:
        return T, v


# Solves for T, v s.t. T, v = argmin_{R,w)||AR + w - B||_F^2
# Here A and B have same shape n x d, T is d x d and v is 1 x d
def procrustes(A, B):
    T, v = custom_procrustes(B, A)
    return T, v


def procrustes_init(seq, rho, y, is_visited_view, d, Utilde, C, param):
    n = Utilde.shape[1]
    # Traverse views from 2nd view
    for m in range(1, seq.shape[0]):
        s = seq[m]
        # pth view is the parent of sth view
        p = rho[s]
        Utilde_s = Utilde[s, :]

        Z_s = [p]
        # Compute centroid mu_s
        # n_Utilde_s_Z_s[k] = #views in Z_s which contain
        # kth point if kth point is in the sth view, else zero
        n_Utilde_s_Z_s = np.zeros(n, dtype=int)
        mu_s = np.zeros((n, d))
        for mp in Z_s:
            Utilde_s_mp = Utilde_s.multiply(Utilde[mp, :]).nonzero()[1]
            n_Utilde_s_Z_s[Utilde_s_mp] += 1
            mu_s[Utilde_s_mp, :] += param.eval_(
                mp, Utilde_s_mp
            )

        # Compute T_s and v_s by aligning the embedding of the overlap
        # between sth view and the views in Z_s, with the centroid mu_s
        temp = n_Utilde_s_Z_s > 0
        mu_s = mu_s[temp, :] / n_Utilde_s_Z_s[temp, np.newaxis]
        V_s_Z_s = param.eval_(s, temp)

        T_s, v_s = procrustes(V_s_Z_s, mu_s)

        # Update T_s, v_
        param.T[s, :, :] = np.matmul(param.T[s, :, :], T_s)
        param.v[s, :] = np.matmul(param.v[s, :][np.newaxis, :], T_s) + v_s

        # Mark sth view as visited
        is_visited_view[s] = True

        # Compute global embedding of point in sth cluster
        C_s = C[s, :].indices
        y[C_s, :] = param.eval_(s, C_s)
    return y, is_visited_view





class Param:
    """Model to transform each point to its embedding.

    Parameters
    ------
    algo: {'lpca', 'kpca'}, default='lpca'
        'lpca': Use linear PCA for local fitting.
        'kpca': Use kernel PCA.
    """

    def __init__(self, algo="lpca", **kwargs):
        self.algo = algo
        self.T = None
        self.v = None
        self.b = None

        # Following variables are
        # initialized externally
        # i.e. by the caller
        self.zeta = None
        self.noise_seed = None
        self.noise_var = 0
        self.noise = None

        # For LPCA and its variants
        self.Psi = None
        self.mu = None
        self.X = None
        self.y = None

        # For KPCA etc
        self.model = None

        self.add_dim = False
        self.standardize = False

    def iter_eval_(self, view_index, data_mask, peak_bytes_per_row=0):
        """Stream the local-(K-)PCA embedding in memory-bounded chunks.

        Yields ``(slice, chunk)`` pairs where ``chunk`` is the embedded
        sub-array for ``view_index[slice]`` / ``data_mask[slice]``. Lets
        callers reduce per chunk (e.g. compute pdists, take a max) without
        ever materializing the full ``(n_eval, k, d)`` output.

        Parameters
        ------
        view_index : array-like, shape (n_points,)
        data_mask : array-like, shape (n_points, n_neighbors)
        peak_bytes_per_row : int, default=0
            Caller-provided estimate of additional bytes the downstream
            pipeline will allocate per row (e.g. batched_pdist's diff
            buffer). Folded into the chunk-size budget so the *whole*
            pipeline stays within available memory, not just this method.

        On MemoryError the chunk size is halved and the slice is retried;
        a one-time RuntimeWarning is emitted so the user knows they're at
        the limit.
        """
        ks = np.asarray(view_index) if not isinstance(view_index, np.ndarray) else view_index
        masks = data_mask
        n_eval = len(ks)
        if n_eval == 0:
            return

        if self.algo == "lpca":
            n_neighbors = masks.shape[1]
            n_features = self.X.shape[1]
            d = self.Psi.shape[2]
            itemsize = self.X.dtype.itemsize
            # Per-row peak inside this method: input gather + output buffer.
            own_per_row = (n_neighbors * n_features + n_neighbors * d) * itemsize
            per_row = max(1, own_per_row + max(0, int(peak_bytes_per_row)))

            chunk_size = max(1, _available_memory_bytes() // per_row)
            chunk_size = min(chunk_size, n_eval)

            warned = False
            start = 0
            while start < n_eval:
                end = min(start + chunk_size, n_eval)
                sl = slice(start, end)
                ks_chunk = ks[sl]
                masks_chunk = masks[sl]
                try:
                    chunk = np.matmul(
                        self.X[masks_chunk] - self.mu[ks_chunk][:, np.newaxis, :],
                        self.Psi[ks_chunk],
                    )
                except MemoryError:
                    if chunk_size <= 1:
                        raise
                    chunk_size = max(1, chunk_size // 2)
                    if not warned:
                        warnings.warn(
                            "pyRATS: hit MemoryError during local embedding; "
                            f"halving chunk size to {chunk_size} and retrying. "
                            "Consider lowering n_neighbors or setting "
                            "PYRATS_MEMORY_LIMIT.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        warned = True
                    continue
                chunk = self._apply_post_(ks_chunk, masks_chunk, chunk)
                yield sl, chunk
                start = end
        else:
            # KPCA path is already row-by-row; yield singletons so callers
            # have a uniform streaming interface.
            for i, k in enumerate(ks):
                X_ = self.X[masks[i], :]
                if self.standardize:
                    X_ = X_ - np.mean(X_, axis=0)[None, :]
                    X_ = X_ / (np.std(X_, axis=0, ddof=1)[None, :] + 1e-12)
                chunk = self.model[k].transform(X_)[None, ...]
                ks_chunk = ks[i:i + 1]
                masks_chunk = masks[i:i + 1]
                chunk = self._apply_post_(ks_chunk, masks_chunk, chunk)
                yield slice(i, i + 1), chunk

    def _apply_post_(self, ks, masks, temp):
        """Apply noise / add_dim / b / T / v to a chunk of embedded points.

        Factored out of batched_eval_ so iter_eval_ can apply the same
        post-processing per chunk. Semantics match the original inline code.
        """
        if self.noise_var:
            np.random.seed(self.noise_seed[0])
            temp2 = np.random.normal(0, self.noise_var, (self.X.shape[0], temp.shape[2]))
            temp = temp + temp2[masks, :]
        if self.noise is not None:
            temp = temp + self.noise[masks]
        if self.add_dim:
            temp = np.concatenate([temp, np.zeros((*temp.shape[:2], 1))], axis=2)
        if self.b is not None:
            temp = temp * self.b[ks][:, None, None]
            if self.T is not None:
                temp = np.dot(temp, self.T[ks, :, :])
            if self.v is not None:
                temp = temp + self.v[[ks], :]
        return temp

    def batched_eval_(self, view_index, data_mask, n_jobs=1):
        """Materialize the full local-(K-)PCA embedding.

        Thin wrapper around iter_eval_ for callers that need the whole
        ``(n_eval, n_neighbors, d)`` array (e.g. clustering's procrustes
        cost). Memory-bounded internally via iter_eval_'s chunking and
        MemoryError backoff.

        ``n_jobs`` is accepted for backward compatibility and ignored;
        the live memory probe in iter_eval_ already reflects whatever
        sibling workers have allocated.
        """
        ks = view_index
        masks = data_mask
        n_eval = len(ks)
        if n_eval == 0:
            d = self.Psi.shape[2] if self.algo == "lpca" else 0
            return np.zeros((0, masks.shape[1] if hasattr(masks, "shape") else 0, d))

        it = self.iter_eval_(ks, masks)
        first_sl, first_chunk = next(it)
        if first_sl.stop == n_eval:
            # Common path: everything fit in one chunk — return directly,
            # no extra allocation or copy.
            return first_chunk

        # Multi-chunk path: accumulate into a pre-allocated output array.
        out = np.empty((n_eval, first_chunk.shape[1], first_chunk.shape[2]),
                       dtype=first_chunk.dtype)
        out[first_sl] = first_chunk
        for sl, chunk in it:
            out[sl] = chunk
        return out

    def eval_(self, view_index, data_mask, apply_b=True):
        """Maps points to the new space through their local (K-)PCA prameterizations.

        Parameters
        ------
        view_index : int
            The index of the parameterization to use.

        data_mask : array-like, shape (n_neighbors)
            List of points to map to the embedding dimension.

        apply_b : bool, default=True
            Whether to apply the scale transformation b[k].
        """

        k = view_index
        mask = data_mask

        if self.algo == "lpca":
            temp = np.dot(
                self.X[mask, :] - self.mu[k, :][np.newaxis, :], self.Psi[k, :, :]
            )
            n = self.X.shape[0]
        else:
            X_ = self.X[mask, :]
            if self.standardize:
                X_ = X_ - np.mean(X_, axis=0)[None, :]
                X_ = X_ / (np.std(X_, axis=0, ddof=1)[None, :] + 1e-12)
            temp = self.model[k].transform(X_)

        if self.noise_var:
            np.random.seed(self.noise_seed[k])
            temp2 = np.random.normal(0, self.noise_var, (n, temp.shape[1]))
            temp = temp + temp2[mask, :]

        if self.noise is not None:
            temp = temp + self.noise[k, mask, :]

        if self.add_dim:
            temp = np.concatenate([temp, np.zeros((temp.shape[0], 1))], axis=1)

        if self.b is None or not apply_b:
            return temp
        else:
            temp = self.b[k] * temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k, :, :])
            if self.v is not None:
                temp = temp + self.v[[k], :]
            return temp

    def replace_(self, new_param_ind):
        """Re-orders the parameterizations.

        Parameters
        ------
        new_param_ind : array-like, shape (n_samples)
            Indices of new ordering of parameterizations.

        """
        if self.algo in ["lpca"]:
            self.Psi = self.Psi[new_param_ind, :]
            self.mu = self.mu[new_param_ind, :]
        else:  # ISOMAP, LKPCA
            self.model = self.model[new_param_ind]

    def reconstruct_(self, view_index, embeddings):
        k = view_index
        y_ = embeddings
        if self.algo == "LPCA":
            temp = (
                np.dot(
                    np.dot(y_ - self.v[[k], :], self.T[k, :, :].T), self.Psi[k, :, :].T
                )
                + self.mu[k, :][np.newaxis, :]
            )
        else:
            temp = self.model[k].inverse_transform(y_)
        return temp
