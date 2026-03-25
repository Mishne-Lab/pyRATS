import numpy as np
import warnings
from scipy.sparse import csr_matrix

from multiprocessing import cpu_count
from joblib import Parallel, delayed

from pyRATS._utils import (
    nearest_neighbors,
    sparse_matrix,
    kpca,
    lpca,
    best,
    compute_seq_of_views,
    compute_init_embedding,
    compute_final_embedding,
    batched_pdist,
)
from pyRATS._tear_coloring import compute_color_of_pts_on_tear


def _postprocess_col_range(
    start,
    end,
    connectivity_matrix,
    reconsider_mask,
    param,
    future_use_phi_of,
    original_dists,
    n,
):
    """Process a contiguous range of neighbor-columns [start, end) for _postprocess.

    Returns the element-wise minimum distortion and the corresponding column
    index across the processed range, both as length-n arrays.
    """
    local_zeta_min = np.full(n, np.inf)
    local_best_col = np.zeros(n, dtype=int)

    for i in range(start, end):
        col_mask = reconsider_mask[:, i]
        use_phi_of = connectivity_matrix[:, i][col_mask]
        if len(use_phi_of) == 0:
            continue

        neighbors_to_reconsider = connectivity_matrix[col_mask]
        points_to_reconsider = np.where(col_mask)[0]

        batch_embedded_datapoints = param.batched_eval_(
            future_use_phi_of[use_phi_of], neighbors_to_reconsider
        )
        batch_embedded_dists = batched_pdist(batch_embedded_datapoints)
        batch_original_dists = original_dists[points_to_reconsider]

        disc_lip_const = np.divide(
            batch_embedded_dists,
            batch_original_dists,
            out=np.full_like(batch_embedded_dists, np.nan),
            where=batch_original_dists != 0,
        )
        col_zeta = np.full(n, np.inf)
        col_zeta[points_to_reconsider] = np.nanmax(disc_lip_const, axis=1) / (
            np.nanmin(disc_lip_const, axis=1) + 1e-12
        )

        improved = col_zeta < local_zeta_min
        local_best_col[improved] = i
        local_zeta_min = np.minimum(local_zeta_min, col_zeta)

    return local_zeta_min, local_best_col


class RATS:
    """Riemannian Alignment of Tangent Spaces

    Parameters
    ----------
    n_components : int, default=2
        Intrinsic dimension of the manifold (target embedding dimension).

    n_neighbors : int, default=28
        Neighborhood size for local view fitting.

    postprocess : bool, default=True
        If True, searches for local embeddings that minimize the local distortion.
        Useful for noisy manifolds.

    kernel : {None, 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} or Callable, default=None
        Kernel used for PCA. None uses standard linear PCA (lpca). Any other value
        uses sklearn.decomposition.KernelPCA. See
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
        for more information.

    cost_function : {'alignment', 'distortion'}, default='alignment'
        'alignment': Alignment error should be preferred when runtime matters or dealing with noisy manifolds.
        'distortion': Distortion is slow, but works well on low noise manifolds.

    min_cluster_size : int, default=5
        Minimum number of points per cluster. A cluster is formed by merging local
        embeddings that minimise cost_function. Larger clusters improve runtime and
        regularise noisy manifolds. The value must be >= 1.

    max_cluster_size : int, default=25
        Maximum allowed size of the clusters. The value must be > min_cluster_size.

    tear : bool, default=True
        Whether to tear the manifold.

    nu : int, default=3
        The ratio of the size of local views in the embedding against those in the data.

    n_iter : int, default=20
        Number of outer iterations to refine the global embedding.
        Riemannian gradient descent runs for n_iter * n_iter_inner total steps.
        For every iteration the alignment of points in the embedding is recomputed
        and the tear is re-evaluated if tear=True.

    n_iter_inner : int, default=100
        Number of internal iterations used by Riemannian Gradient Descent per outer step.

    alpha : float, default=0.3
        Step size used in the Riemannian gradient descent.

    n_iter_without_progress : int, default=5
        The number of outer iterations to wait after the relative change in alignment
        error and tear size both drop below *tol* before stopping early.

    tol : float, default=1e-2
        Tolerance level for the relative change in the alignment error and the
        relative change in the size of the tear.

    metric : str, default='euclidean'
        To be added in future releases. Metric assumed on the embedding.
        Currently only 'euclidean' is supported.

    fit_inverse_transform : bool, default=False
        To be added in future releases. If True, computes the inverse of the embedding
        transformation. Must only be enabled together with a non-None kernel.
        When kernel=None (linear PCA) the inverse is always available.

    verbose : bool, default=False
        If True, print logs to stdout.

    n_jobs : int, default=-1
        The number of CPU-cores to use. If -1, uses all available cores.

    """

    # Mapping from deprecated parameter names to their new equivalents.
    _DEPRECATED_PARAMS = {
        "d": "n_components",
        "k": "n_neighbors",
        "max_iter": "n_iter",
        "max_internal_iter": "n_iter_inner",
        "patience": "n_iter_without_progress",
        "cost_fn_name": "cost_function",
        "eta_min": "min_cluster_size",
        "eta_max": "max_cluster_size",
        "kpca_kernel": "kernel",
        "kpca_fit_inverse_transform": "fit_inverse_transform",
        "to_postprocess": "postprocess",
        "to_tear": "tear",
    }

    def __init__(
        self,
        n_components=2,
        kernel=None,
        fit_inverse_transform=False,
        n_neighbors=28,
        cost_function="alignment",
        metric="euclidean",
        postprocess=True,
        min_cluster_size=5,
        max_cluster_size=25,
        tear=True,
        nu=3,
        n_iter=20,
        n_iter_inner=100,
        alpha=0.3,
        eps=1e-8,
        n_iter_without_progress=5,
        tol=1e-2,
        n_forced_clusters=1,
        verbose=False,
        n_jobs=-1,
        **kwargs,
    ):
        # ------------------------------------------------------------------
        # Backward-compatibility: accept deprecated parameter names and emit
        # a DeprecationWarning so old code keeps working for one release.
        # ------------------------------------------------------------------
        for old, new in self._DEPRECATED_PARAMS.items():
            if old in kwargs:
                warnings.warn(
                    f"The parameter '{old}' is deprecated and will be removed in a "
                    f"future release. Use '{new}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Only override the new-name value if the caller did not also
                # supply the new name explicitly.
                locals_snapshot = {
                    "n_components": n_components,
                    "kernel": kernel,
                    "fit_inverse_transform": fit_inverse_transform,
                    "n_neighbors": n_neighbors,
                    "cost_function": cost_function,
                    "postprocess": postprocess,
                    "min_cluster_size": min_cluster_size,
                    "max_cluster_size": max_cluster_size,
                    "tear": tear,
                    "n_iter": n_iter,
                    "n_iter_inner": n_iter_inner,
                    "n_iter_without_progress": n_iter_without_progress,
                }
                if old == "d":
                    n_components = kwargs.pop(old)
                elif old == "k":
                    n_neighbors = kwargs.pop(old)
                elif old == "max_iter":
                    n_iter = kwargs.pop(old)
                elif old == "max_internal_iter":
                    n_iter_inner = kwargs.pop(old)
                elif old == "patience":
                    n_iter_without_progress = kwargs.pop(old)
                elif old == "cost_fn_name":
                    cost_function = kwargs.pop(old)
                elif old == "eta_min":
                    min_cluster_size = kwargs.pop(old)
                elif old == "eta_max":
                    max_cluster_size = kwargs.pop(old)
                elif old == "kpca_kernel":
                    kernel = kwargs.pop(old)
                elif old == "kpca_fit_inverse_transform":
                    fit_inverse_transform = kwargs.pop(old)
                elif old == "to_postprocess":
                    postprocess = kwargs.pop(old)
                elif old == "to_tear":
                    tear = kwargs.pop(old)
            else:
                kwargs.pop(old, None)

        if kwargs:
            raise TypeError(
                f"RATS.__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        if cost_function not in ["alignment", "distortion"]:
            raise ValueError(
                f"cost_function must be 'alignment' or 'distortion', got {cost_function!r}."
            )
        if max_cluster_size <= min_cluster_size:
            raise ValueError(
                f"max_cluster_size ({max_cluster_size}) must be greater than "
                f"min_cluster_size ({min_cluster_size})."
            )
        if metric != "euclidean":
            warnings.warn(
                f"metric={metric!r} is not yet fully supported. Only 'euclidean' is "
                "used in the embedding step. This parameter will be extended in a "
                "future release.",
                FutureWarning,
                stacklevel=2,
            )
        if fit_inverse_transform:
            warnings.warn(
                "fit_inverse_transform=True is not yet supported and will be "
                "ignored. Inverse transform support will be added in a future release.",
                FutureWarning,
                stacklevel=2,
            )

        self.verbose = verbose
        self.n_jobs = n_jobs

        self.d = n_components
        self.kpca_kernel = kernel
        self.kpca_fit_inverse_transform = fit_inverse_transform
        self.k = n_neighbors
        if cost_function == "distortion":
            self.k_nn0 = max(n_neighbors, max_cluster_size * n_neighbors)
        else:
            self.k_nn0 = n_neighbors
        self.cost_fn = cost_function
        self.to_postprocess = postprocess
        self.eta_min, self.eta_max = min_cluster_size, max_cluster_size
        self.to_tear = tear
        self.nu = nu
        self.max_iter, self.max_internal_iter = n_iter, n_iter_inner
        self.alpha, self.eps = alpha, eps
        self.patience, self.tol = n_iter_without_progress, tol
        self.metric = metric
        self.n_forced_clusters = n_forced_clusters

        if self.patience is None:
            self.patience = self.max_iter

        if n_jobs == -1:
            self.n_jobs = cpu_count()

    def fit_transform(self, X):
        """Fit the model on the data in X, and transform X.

        Parameters
        ---------
        X : array-like, shape (n_samples, n_features)
            Sample data, in the form of a numpy array of shape (n_samples, n_features).

        Returns
        -------
        y : array-like, shape (n_samples, d)
            X transformed in the new space.
        """

        self._fit_nbrhd_graph(X)

        # Construct low dimensional local views
        self._fit_local_views(X)
        if self.to_postprocess:
            self._postprocess()

        # Construct intermediate views
        c, n_C = self._fit_intermediate_views()

        # Construct Global views
        y = self._fit_global_views(c, n_C)

        return y

    def compute_color_of_pts_on_tear(self, y, tear_color_eig_inds=[0, 1, 2]):
        """Compute glueing instructions for the tear.

        Parameters
        ------
        y: array-like, shape (n_samples, d)
            X tranformed in the new space.

        tear_color_eig_inds:

        Returns
        ------
            Color of points on tear or None if no tear could be detected.

        """
        if not self.to_tear:
            print(
                "Manifold is not torn. Gluing instructions can only be provided for torn manifolds."
            )
            return None

        return compute_color_of_pts_on_tear(
            y,
            self.Utilde,
            self.C,
            self.n_Utilde_Utilde,
            tear_color_eig_inds,
            self.k,
            self.nu,
            self.metric,
            self.verbose,
            self.n_jobs,
            self.Utildeg,
        )

    def fit(self, X):
        """Fit the model on X without returning the embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample data.

        Returns
        -------
        self
        """
        raise NotImplementedError(
            "fit() is not yet implemented. Use fit_transform() instead."
        )

    def transform(self, X):
        """Apply the fitted embedding to new data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to embed.

        Returns
        -------
        y : array-like, shape (n_samples, d)
            X transformed in the new space.
        """
        raise NotImplementedError(
            "transform() is not yet implemented. Use fit_transform() instead."
        )

    def inverse_transform(self, y):
        """Map points from the embedding space back to the original space.

        Parameters
        ----------
        y : array-like, shape (n_samples, d)
            Points in the embedding space.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Points in the original space.
        """
        raise NotImplementedError(
            "inverse_transform() is not yet implemented."
        )


    def _fit_nbrhd_graph(self, X, sort_results=True):
        """Fitting the neighborhood graph.

        Parameters
        ----------
        X : array shape (n_samples, n_features)
            A 2d array containing data representing a manifold.

        sort_results: bool, default=True
            If True, sorts neighbors by index in ascending order for deterministic behavior.

        """

        self.neigh_dist, self.neigh_ind = nearest_neighbors(
            X, self.k_nn0, self.metric, sort_results, self.n_jobs
        )

        self.U = sparse_matrix(  # needed for distortion cost_fn
            self.neigh_ind[:, : self.k], np.ones((len(X), self.k), dtype=bool)
        )

        self.neighbor_graph_idcs = sparse_matrix(
            self.neigh_ind, np.ones((len(X), self.k_nn0), dtype=bool)
        )  # self.U
        self.neighborhood_graph = sparse_matrix(self.neigh_ind, self.neigh_dist)
        self.neighborhood_graph_idcs_sym = self.neighbor_graph_idcs.maximum(
            self.neighbor_graph_idcs.transpose()
        )
        self.neighborhood_graph_sym = self.neighborhood_graph.maximum(
            self.neighborhood_graph.transpose()
        )

    def _fit_local_views(self, X):
        """Project local neighborhoods via (Kernel-) PCA to the intrinsic dimension.

        Parameters
        ----------
        X : array shape (n_samples, n_features)
            A 2d array containing data representing a manifold.
        """

        n = len(X)

        # Compute local views
        if self.kpca_kernel:
            self.param = kpca(
                X,
                self.d,
                sparse_matrix(
                    self.neigh_ind[:, : self.k], np.ones((n, self.k), dtype=bool)
                ),
                self.kpca_kernel,
                self.kpca_fit_inverse_transform,
                self.n_jobs,
                verbose=self.verbose,
            )
        else:
            self.param = lpca(
                X,
                self.d,
                sparse_matrix(
                    self.neigh_ind[:, : self.k], np.ones((n, self.k), dtype=bool)
                ),
                self.n_jobs,
                verbose=self.verbose,
            )

        self.param.b = np.ones(n)

    def _fit_intermediate_views(self):
        """Cluster local views.

        Returns
        -------
        c : array-like, shape (n_samples)
            Holds the cluster index for each datapoint.

        n_C: array-like, shape (n_clusters)
            Holds the number of datapoints per cluster.

        Notes
        -------
        This algorithm is based on the following rules:
        1) A point can only move into one of its neighbors' clusters.
        2) A point cannot move into a cluster that is smaller than the cluster it currently belongs to.
        3) A point always moves to the cluster whose parameterization induces the lowest cost.

        """

        n = self.neighborhood_graph_sym.shape[0]
        if self.eta_min > 1:
            c, n_C = best(
                self.neighborhood_graph_sym,
                self.U,
                self.param,
                self.eta_min,
                self.eta_max,
                self.cost_fn,
                self.verbose,
                self.n_jobs,
            )
            if self.verbose:
                print("Pruning and cleaning up.")
            # Prune empty clusters
            non_empty_C = n_C > 0
            M = np.sum(non_empty_C)
            old_to_new_map = np.arange(n)
            old_to_new_map[non_empty_C] = np.arange(M)
            c = old_to_new_map[c]
            n_C = n_C[non_empty_C]

            # Construct a boolean array C s.t. C[m,i] = 1 if c_i == m, 0 otherwise
            C = csr_matrix((np.ones(n), (c, np.arange(n))), shape=(M, n), dtype=bool)

            # Compute intermediate views
            if self.param.b is not None:
                self.param.b = self.param.b[non_empty_C]
            if self.param.Psi is not None:
                self.param.Psi = self.param.Psi[non_empty_C, :]
            if self.param.mu is not None:
                self.param.mu = self.param.mu[non_empty_C, :]
            if self.param.model is not None:
                self.param.model = self.param.model[non_empty_C]

            Utilde = C.dot(self.U)

            np.random.seed(42)
            self.param.noise_seed = np.random.randint(M * M, size=M)
            if self.verbose:
                print("Done.")
        else:
            c = np.arange(n, dtype=int)
            C = csr_matrix((np.ones(n), (c, np.arange(n))), shape=(n, n), dtype=bool)
            n_C = np.ones(n, dtype=int)
            Utilde = self.U.copy()
            np.random.seed(42)
            self.param.noise_seed = np.random.randint(n * n, size=n)

        self.Utilde = Utilde
        self.C = C
        self.c = c

        return c, n_C

    def _fit_global_views(self, c, n_C):
        """Align intermediate clusters via Riemannian gradient descent.

        Parameters
        ----------
        c : array-like, shape (n_samples)
            Holds the cluster index for each datapoint.

        n_C: array-like, shape (n_clusters)
            Holds the number of datapoints per cluster.

        Returns
        -------
        y : array-like, shape (n_samples, d)
            Representation of X in the new space.
        """

        # Compute |Utilde_{mm'}|
        n_Utilde_Utilde = self.Utilde.dot(self.Utilde.transpose())
        n_Utilde_Utilde.setdiag(0)
        self.n_Utilde_Utilde = n_Utilde_Utilde

        # Compute sequence of intermedieate views
        seq_of_intermed_views_in_cluster, parents_of_intermed_views_in_cluster, _ = (
            compute_seq_of_views(
                self.d,
                self.Utilde,
                n_Utilde_Utilde,
                self.param,
                self.n_forced_clusters,
                self.verbose,
                self.n_jobs,
            )
        )

        # Compute initial embedding
        y_init = compute_init_embedding(
            self.d,
            self.Utilde,
            self.param,
            seq_of_intermed_views_in_cluster,
            parents_of_intermed_views_in_cluster,
            self.C,
            self.verbose,
        )

        # apply RGD
        y_final, Utildeg = compute_final_embedding(
            y_init,
            self.d,
            self.Utilde,
            self.C,
            self.param,
            self.to_tear,
            self.patience,
            self.max_iter,
            self.max_internal_iter,
            self.tol,
            self.nu,
            self.k,
            self.metric,
            self.alpha,
            self.verbose,
        )
        self.Utildeg = Utildeg

        return y_final

    def _postprocess(self):
        """Replace high local distortion incuring parameterizations by those of neighboring points."""

        n = self.U.shape[0]

        # Vectorised: gather all pairwise neighbour distances at once.
        # neigh_inds[i] are the k neighbour indices of point i (same as
        # connectivity_matrix used below, extracted once here for clarity).
        neigh_inds = self.U.indices.reshape(n, self.k)  # (n, k)
        pair_i, pair_j = np.triu_indices(self.k, k=1)  # all upper-triangle pairs
        row_idx = neigh_inds[:, pair_i].ravel()  # (n * num_pairs,)
        col_idx = neigh_inds[:, pair_j].ravel()  # (n * num_pairs,)
        num_pairs = len(pair_i)
        original_dists = np.asarray(
            self.neighborhood_graph_sym[row_idx, col_idx]
        ).reshape(n, num_pairs)

        embedded_datapoints = self.param.batched_eval_(
            range(n), self.U.indices.reshape(n, self.k)
        )  # tested and they are equivalent
        embedded_dists = batched_pdist(embedded_datapoints)
        disc_lip_const = np.divide(
            embedded_dists,
            original_dists,
            out=np.full_like(embedded_dists, np.nan),
            where=original_dists != 0,
        )
        zeta = np.nanmax(disc_lip_const, axis=1) / (
            np.nanmin(disc_lip_const, axis=1) + 1e-12
        )

        self.param.zeta = zeta

        if self.verbose:
            print(
                r"Maximum local distortion before postprocessing is: %0.4f"
                % np.max(zeta)
            )

        connectivity_matrix = self.U.indices.reshape(n, -1)
        param_changed = np.arange(n)
        future_use_phi_of = np.arange(n, dtype=int)

        while len(param_changed) > 0:
            changed_mask = np.zeros(n, dtype=bool)
            changed_mask[param_changed] = True
            reconsider_mask = changed_mask[connectivity_matrix] & (
                connectivity_matrix != np.arange(n)[:, None]
            )

            # Split the k columns into n_jobs chunks and process each chunk in
            # a separate worker.  Each worker returns its local (zeta_min,
            # best_col) for the columns it owns; we then do a final
            # min-reduction here.  This mirrors the target_proc pattern used
            # throughout _utils.py (lpca, best, …) while keeping memory
            # bounded to one chunk of columns per worker at a time.
            k_cols = connectivity_matrix.shape[1]
            chunk_sz = max(1, k_cols // self.n_jobs)
            ranges = [(p * chunk_sz, (p + 1) * chunk_sz) for p in range(self.n_jobs)]
            ranges[-1] = (ranges[-1][0], k_cols)  # extend last chunk to cover remainder
            ranges = [(s, e) for s, e in ranges if s < k_cols]  # drop empty ranges

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_postprocess_col_range)(
                    s, e,
                    connectivity_matrix,
                    reconsider_mask,
                    self.param,
                    future_use_phi_of,
                    original_dists,
                    n,
                )
                for s, e in ranges
            )

            # Reduce chunk results into a single (zeta_min, best_col).
            zeta_min = np.full(n, np.inf)
            best_col = np.zeros(n, dtype=int)
            for chunk_zeta_min, chunk_best_col in results:
                improved = chunk_zeta_min < zeta_min
                best_col[improved] = chunk_best_col[improved]
                zeta_min = np.minimum(zeta_min, chunk_zeta_min)

            param_changed = np.where(zeta > zeta_min)[0]
            future_use_phi_of[param_changed] = future_use_phi_of[
                connectivity_matrix[param_changed, best_col[param_changed]]
            ]
            zeta = np.minimum(zeta, zeta_min)

            if self.verbose:
                print(
                    "#Param replaced: %d, max distortion: %f"
                    % (len(param_changed), np.max(zeta))
                )

        self.param.zeta = zeta.copy()

        self.param.replace_(future_use_phi_of)
        if self.verbose:
            print(
                f"Maximum local distortion after postprocessing is: %0.4f"
                % (np.max(self.param.zeta))
            )
