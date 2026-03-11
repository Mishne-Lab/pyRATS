import pdb
import time
import numpy as np
import itertools
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path, breadth_first_order, dijkstra
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
import multiprocess as mp
import os
import pickle

from joblib import Parallel, delayed


def sub_dict(x, keys):
    y = {}
    for k in keys:
        y[k] = x[k]
    return y


def path_exists(path):
    return os.path.exists(path) or os.path.islink(path)


def makedirs(dirpath):
    if path_exists(dirpath):
        return
    os.makedirs(dirpath)


def read(fpath, verbose=True):
    if not path_exists(fpath):
        if verbose:
            print(fpath, "does not exist.")
        return None
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    if verbose:
        print("Read data from", fpath, flush=True)
    return data


def save(dirpath, fname, data, verbose=True):
    if not path_exists(dirpath):
        os.makedirs(dirpath)
    fpath = dirpath + "/" + fname
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print("Saved data in", fpath, flush=True)


def create_mask(inds, n):
    temp = np.zeros(n, dtype=bool)
    temp[inds] = 1
    return temp


def get_ranks(x, small_val_better_rank=True):
    temp = x.argsort()
    if not small_val_better_rank:
        temp = temp[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(temp))
    return ranks


def shortest_paths(
    X,
    n_nbrs=None,
    radius=None,
    metric="euclidean",
    return_predecessors=False,
    indices=None,
):
    assert (n_nbrs is not None) or (radius is not None)
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, radius=radius, metric=metric).fit(X)
    if n_nbrs is not None:
        knn_graph = nbrs.kneighbors_graph(mode="distance")
    else:
        knn_graph = nbrs.radius_neighbors_graph(mode="distance")
    return shortest_path(
        knn_graph,
        return_predecessors=return_predecessors,
        directed=False,
        indices=indices,
    )


def center_of_tree(T):
    s1, pred1 = breadth_first_order(T, 0, directed=False)
    s2, pred2 = breadth_first_order(T, s1[-1], directed=False)
    nodes_on_longest_path = [s2[-1]]
    pred = 0
    while pred >= 0:
        pred = pred2[nodes_on_longest_path[-1]]
        nodes_on_longest_path.append(pred)
    n = len(nodes_on_longest_path)
    return nodes_on_longest_path[n // 2]


def compute_farthest_points(nbrhd_graph, s=0, n_points=20, return_min_max=False):
    if issparse(nbrhd_graph):
        d_e = nbrhd_graph
    else:
        d_e = nbrhd_graph.sparse_matrix()
    far_off_points = [s]
    dist_from_far_off_points = np.zeros(d_e.shape[0]) + np.inf
    if return_min_max:
        min_max_dist = np.inf
        min_max = None
    while len(far_off_points) < n_points:
        temp = dijkstra(d_e, directed=False, indices=far_off_points[-1])
        if return_min_max:
            max_dist = np.max(temp)
            if min_max_dist > max_dist:
                min_max_dist = max_dist
                min_max = far_off_points[-1]
        dist_from_far_off_points = np.minimum(dist_from_far_off_points, temp)
        non_inf_indices = np.where(~np.isinf(dist_from_far_off_points))[0]
        far_off_points.append(
            non_inf_indices[np.argmax(dist_from_far_off_points[non_inf_indices])]
        )
    if return_min_max:
        return np.array(far_off_points), min_max
    else:
        return np.array(far_off_points)


# Computes views in the embedding space
def compute_incidence_matrix_in_embedding(y, C, k, nu, metric="euclidean"):
    M, n = C.shape
    k_ = min(int(k * nu), n - 1)
    _, neigh_indg = nearest_neighbors(y, k_, metric)
    Ug = sparse_matrix(neigh_indg, np.ones(neigh_indg.shape, dtype=bool))
    Utildeg = C.dot(Ug)
    return Utildeg


def print_log(s, log_time, local_start_time, global_start_time):
    print(s)
    if log_time:
        print("##############################")
        print(
            "Time elapsed from last time log: %0.1f seconds"
            % (time.perf_counter() - local_start_time)
        )
        print(
            "Total time elapsed: %0.1f seconds"
            % (time.perf_counter() - global_start_time)
        )
        print("##############################")
    return time.perf_counter()


class Param:
    def __init__(self, algo="LPCA", **kwargs):
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
        self.X = None
        self.y = None

        self.add_dim = False
        self.standardize = False

    def get_state(self):
        state = {
            "algo": self.algo,
            "T": self.T,
            "v": self.v,
            "b": self.b,
            "zeta": self.zeta,
            "noise_seed": self.noise_seed,
            "noise_var": self.noise_var,
            "noise": self.noise,
            "Psi": self.Psi,
            "mu": self.mu,
            "X": self.X,
            "y": self.y,
            "model": self.model,
            "add_dim": self.add_dim,
            "standardize": self.standardize,
        }
        return state

    def set_state(self, state):
        self.algo = state["algo"]
        self.T = state["T"]
        self.v = state["v"]
        self.b = state["b"]
        self.zeta = state["zeta"]
        self.noise_seed = state["noise_seed"]
        self.noise_var = state["noise_var"]
        self.noise = state["noise"]
        # self.Psi_gamma = state['Psi_gamma']
        # self.Psi_i = state['Psi_i']
        # self.phi = state['phi']
        # self.gamma = state['gamma']
        # self.w = state['w']
        self.Psi = state["Psi"]
        self.mu = state["mu"]
        self.X = state["X"]
        # self.y = state['y']
        # self.model = state['model']
        # self.add_dim = state['add_dim']
        # self.standardize = state['standardize']

    def eval_(self, opts):
        k = opts["view_index"]
        mask = opts["data_mask"]

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

        if self.b is None:
            return temp
        else:
            temp = self.b[k] * temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k, :, :])
            if self.v is not None:
                temp = temp + self.v[[k], :]
            return temp

    def compute_local_distortion_(self, nbrhd_graph):
        n = nbrhd_graph.get_num_nodes()
        d_e = nbrhd_graph.sparse_matrix(symmetrize=True)
        self.zeta = np.zeros(n)
        for k in range(n):
            U_k = nbrhd_graph.get_nbr_inds(k)
            d_e_k = d_e[np.ix_(U_k, U_k)]
            self.zeta[k] = compute_zeta(
                d_e_k, self.eval_({"view_index": k, "data_mask": U_k})
            )

    def replace_(self, new_param_ind):
        if self.algo in ["lpca"]:
            self.Psi = self.Psi[new_param_ind, :]
            self.mu = self.mu[new_param_ind, :]
        else:  # ISOMAP, LKPCA
            self.model = self.model[new_param_ind]

    def reconstruct_(self, opts):
        k = opts["view_index"]
        y_ = opts["embeddings"]
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

    def out_of_sample_eval_(self, opts):
        k = opts["view_index"]
        X_ = opts["out_of_samples"]

        if self.algo == "LPCA":
            temp = np.dot(X_ - self.mu[k, :][np.newaxis, :], self.Psi[k, :, :])
            n = self.X.shape[0]
        else:
            temp = self.isomap[k].transform(X_)

        if self.add_dim:
            temp = np.concatenate([temp, np.zeros((temp.shape[0], 1))], axis=1)

        if self.b is None:
            return temp
        else:
            temp = self.b[k] * temp
            if self.T is not None:
                temp = np.dot(temp, self.T[k, :, :])
            if self.v is not None:
                temp = temp + self.v[[k], :]
            return temp

    def alignment_wts(self, opts):
        beta = opts["beta"]
        if beta is None:
            return None
        k = opts["view_index"]
        mask = opts["data_mask"]
        mu = np.mean(self.X[mask, :], axis=0)
        temp = self.X[mask, :] - mu[None, :]
        w = -np.linalg.norm(temp, 1, axis=1) / beta
        return w
        # p = np.exp(w - np.max(w))
        # p *= (temp.shape[0]/np.sum(p))
        # return p

    def repulsion_wts(self, opts):
        beta = opts["beta"]
        if beta is None:
            return None
        k = opts["pt_index"]
        far_off_pts = opts["repelling_pts_indices"]
        if self.y is not None:
            temp = self.y[far_off_pts, :] - self.y[k, :][None, :]
            w = np.linalg.norm(temp, 2, axis=1) ** 2
            # temp0 = self.X[far_off_pts,:] - self.X[k,:][None,:]
            # w0 = np.linalg.norm(temp0, 2, axis=1)**2
            # p = 1.0*((w-w0)<0)
            p = 1 / (w + 1e-12)
        else:
            p = np.ones(len(far_off_pts))
        return p


# includes self as the first neighbor
# data is either X or distance matrix d_e
def nearest_neighbors(data, k_nn, metric, n_jobs=-1, sort_results=True):
    n = data.shape[0]
    if k_nn > 1:
        neigh = NearestNeighbors(n_neighbors=k_nn - 1, metric=metric, n_jobs=n_jobs)
        neigh.fit(data)
        neigh_dist, neigh_ind = neigh.kneighbors()
        neigh_dist = np.insert(neigh_dist, 0, np.zeros(n), axis=1)
        neigh_ind = np.insert(neigh_ind, 0, np.arange(n), axis=1)
        if sort_results:
            inds = np.argsort(neigh_dist, axis=-1)
            for i in range(neigh_ind.shape[0]):
                neigh_ind[i, :] = neigh_ind[i, inds[i, :]]
                neigh_dist[i, :] = neigh_dist[i, inds[i, :]]
    else:
        neigh_dist = np.zeros((n, 1))
        neigh_ind = np.arange(n).reshape((n, 1)).astype("int")

    return neigh_dist, neigh_ind


# includes self as the first neighbor
# data is either X or distance matrix d_e
def radial_nearest_neighbors(data, radius, metric, n_jobs=-1, sort_results=True):
    n = data.shape[0]
    neigh = NearestNeighbors(radius=radius, metric=metric, n_jobs=n_jobs)
    neigh.fit(data)
    neigh_dist, neigh_ind = neigh.radius_neighbors()
    for i in range(len(neigh_dist)):
        neigh_dist[i] = np.insert(neigh_dist[i], 0, np.zeros(1))
        neigh_ind[i] = np.insert(neigh_ind[i], 0, np.array([i]))
        if sort_results:
            inds = np.argsort(neigh_dist[i], axis=-1)
            neigh_ind[i] = neigh_ind[i][inds]
            neigh_dist[i] = neigh_dist[i][inds]

    return neigh_dist, neigh_ind


# |E| x |V|
def incidence_matrix(neigh_ind, neigh_dist):
    n_edges = np.prod(neigh_dist.shape)
    n_nodes = neigh_ind.shape[0]
    row_inds = np.arange(n_edges)
    row_inds = np.concatenate([row_inds, row_inds])
    col_inds_1 = np.repeat(np.arange(n_nodes), neigh_ind.shape[1])
    col_inds_2 = neigh_ind.flatten()
    col_inds = np.concatenate([col_inds_1, col_inds_2])
    data = neigh_dist.flatten()
    data = np.concatenate([-data, data]) + np.finfo(np.float64).eps
    imat = csr_matrix((data, (row_inds, col_inds)))
    imat.eliminate_zeros()
    return imat


def sparse_matrix(neigh_ind, neigh_dist):
    if neigh_ind.dtype == "object":
        row_inds = []
        col_inds = []
        data = []
        for k in range(neigh_ind.shape[0]):
            row_inds.append(np.repeat(k, neigh_ind[k].shape[0]).tolist())
            col_inds.append(neigh_ind[k].tolist())
            data.append(neigh_dist[k].tolist())
        row_inds = list(itertools.chain.from_iterable(row_inds))
        col_inds = list(itertools.chain.from_iterable(col_inds))
        data = list(itertools.chain.from_iterable(data))
    else:
        row_inds = np.repeat(np.arange(neigh_dist.shape[0]), neigh_dist.shape[1])
        col_inds = neigh_ind.flatten()
        data = neigh_dist.flatten()
    return csr_matrix((data, (row_inds, col_inds)))


def to_dense(x):
    if issparse(x):
        return x.toarray()
    else:
        return x


def compute_zeta(d_e_mask0, Psi_k_mask):
    d_e_mask = to_dense(d_e_mask0)
    if d_e_mask.shape[0] == 1:
        return 1
    d_e_mask_ = squareform(d_e_mask)
    mask = d_e_mask_ != 0
    d_e_mask_ = d_e_mask_[mask]
    disc_lip_const = pdist(Psi_k_mask)[mask] / d_e_mask_
    return np.max(disc_lip_const) / (np.min(disc_lip_const) + 1e-12)


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


def procrustes_cost(A, B):
    _, _, c = custom_procrustes(B, A, compute_cost=True)
    return c


def ixmax(x, k=0, idx=None):
    col = x[idx, k] if idx is not None else x[:, k]
    z = np.where(col == col.max())[0]
    return z if idx is None else idx[z]


def lexargmax(x):
    idx = None
    for k in range(x.shape[1]):
        idx = ixmax(x, k, idx)
        if len(idx) < 2:
            break
    return idx[0]


def compute_distortion_at(y_d_e, s_d_e):
    scale_factors = (y_d_e + 1e-12) / (s_d_e + 1e-12)
    mask = np.ones(scale_factors.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_distortion = np.max(scale_factors[mask]) / np.min(scale_factors[mask])
    print("Max distortion is:", max_distortion, flush=True)
    n = y_d_e.shape[0]
    distortion_at = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = 0
        distortion_at[i] = np.max(scale_factors[i, mask]) / np.min(
            scale_factors[i, mask]
        )
        mask[i] = 1
    return distortion_at, max_distortion


def compute_holed_distortion_at(y_d_e, s_d_e, hole_prctile=50):
    hole_dist = np.percentile(y_d_e, hole_prctile, axis=1)
    mask = y_d_e < hole_dist[:, None]
    scale_factors = (y_d_e * mask + 1e-12) / (s_d_e * mask + 1e-12)
    mask = np.ones(scale_factors.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_distortion = np.max(scale_factors[mask]) / np.min(scale_factors[mask])
    print("Max distortion is:", max_distortion, flush=True)
    n = y_d_e.shape[0]
    distortion_at = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = 0
        distortion_at[i] = np.max(scale_factors[i, mask]) / np.min(
            scale_factors[i, mask]
        )
        mask[i] = 1
    return distortion_at, max_distortion


def compute_distortion_at_from_data(Y, X, n_nbrs=10):
    s_d_e = shortest_paths(X, n_nbrs=n_nbrs, return_predecessors=False)
    y_d_e = shortest_paths(Y, n_nbrs=n_nbrs, return_predecessors=False)
    return compute_distortion_at(y_d_e, s_d_e)


def compute_quantile_distortion_at(y_d_e, s_d_e, quantile=0.9):
    scale_factors = (y_d_e + 1e-12) / (s_d_e + 1e-12)
    mask = np.ones(scale_factors.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_distortion = np.quantile(scale_factors[mask], quantile) / np.percentile(
        scale_factors[mask], 1 - quantile
    )
    print("Max distortion is:", max_distortion, flush=True)
    n = y_d_e.shape[0]
    distortion_at = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = 0
        distortion_at[i] = np.quantile(scale_factors[i, mask], quantile) / np.quantile(
            scale_factors[i, mask], 1 - quantile
        )
        mask[i] = 1
    return distortion_at, max_distortion


def get_path_lengths_in_embedding_space(pred, y_d_e, n_proc=256):
    n = pred.shape[0]
    y_d_e2 = np.zeros((n, n))

    def target_proc(pairs_to_proc, start_ind, end_ind, q_, y_d_e, pred):
        def get_path_length(i, j):
            path_length = 0
            k = j
            while pred[i, k] != -9999:
                path_length += y_d_e[k, pred[i, k]]
                k = pred[i, k]
            return path_length

        my_data = np.zeros(end_ind - start_ind)
        for ind in range(start_ind, end_ind):
            i, j = pairs_to_proc[ind]
            my_data[ind - start_ind] = get_path_length(i, j)

        q_.put((start_ind, end_ind, my_data))

    pairs_to_proc = list(itertools.combinations(np.arange(n), 2))
    q_ = mp.Queue()
    chunk_sz = len(pairs_to_proc) // n_proc
    proc = []
    start_ind = 0
    end_ind = 1
    for p_num in range(n_proc):
        if p_num == n_proc - 1:
            end_ind = len(pairs_to_proc)
        else:
            end_ind = (p_num + 1) * chunk_sz

        proc.append(
            mp.Process(
                target=target_proc,
                args=(pairs_to_proc, start_ind, end_ind, q_, y_d_e, pred),
                daemon=True,
            )
        )
        proc[-1].start()
        start_ind = end_ind

    print("All processes started", flush=True)
    for p_num in range(n_proc):
        start_ind, end_ind, y_d_e2_ = q_.get()
        for ind in range(start_ind, end_ind):
            i, j = pairs_to_proc[ind]
            y_d_e2[i, j] = y_d_e2_[ind - start_ind]
            y_d_e2[j, i] = y_d_e2[i, j]

    q_.close()
    for p_num in range(n_proc):
        proc[p_num].join()

    return y_d_e2


def reconstruct_(self, opts):
    k = opts["view_index"]
    y_ = opts["embeddings"]
    if self.algo == "LPCA":
        temp = (
            np.dot(np.dot(y_ - self.v[[k], :], self.T[k, :, :].T), self.Psi[k, :, :].T)
            + self.mu[k, :][np.newaxis, :]
        )
    return temp


def reconstruct_data(p, buml_obj, y, is_init=False, averaging=True):
    if averaging:
        Utildeg = buml_obj.GlobalViews.compute_incidence_matrix_in_embedding(
            y, buml_obj.IntermedViews.C, buml_obj.global_opts
        )
        Utilde = buml_obj.IntermedViews.Utilde.multiply(Utildeg)
        Utilde.eliminate_zeros()
    else:
        Utilde = buml_obj.IntermedViews.C
    if is_init:
        intermed_param = buml_obj.GlobalViews.intermed_param_init
    else:
        intermed_param = buml_obj.IntermedViews.intermed_param
    m, n = Utilde.shape
    X = np.zeros((n, p))
    for k in range(n):
        views = Utilde[:, k].nonzero()[0].tolist()
        temp = []
        embedding = y[k, :][None, :]
        for i in views:
            temp.append(
                reconstruct_(intermed_param, {"view_index": i, "embeddings": embedding})
            )
        temp = np.array(temp)
        X[k, :] = np.mean(temp, axis=0)
    return X


def compute_global_distortions(
    X, y, n_nbr=10, buml_obj_path="", read_dir_root="", save_dir_root="", n_proc=32
):
    # Shortest paths in the data
    s_d_e_path = read_dir_root + "/s_d_e.dat"
    pred_path = read_dir_root + "/pred.dat"
    save0 = read_dir_root != ""
    if (not path_exists(s_d_e_path)) or (not path_exists(pred_path)):
        s_d_e, pred = shortest_paths(X, n_nbr)
        if save0:
            save(read_dir_root, "s_d_e.dat", s_d_e)
            save(read_dir_root, "pred.dat", pred)
    else:
        s_d_e = read(s_d_e_path)
        pred = read(pred_path)

    # Shortest paths in the embedding
    save1 = save_dir_root != ""
    y_s_d_e_path = save_dir_root + "/y_s_d_e.dat"
    if not path_exists(y_s_d_e_path):
        y_s_d_e, _ = shortest_paths(y, n_nbr)

        if buml_obj_path:
            all_data = read(buml_obj_path)
            X, labelsMat, buml_obj, gv_info, ex_name = all_data
            if buml_obj.global_opts["to_tear"]:
                intermed_param = buml_obj.IntermedViews.intermed_param
                Utilde = buml_obj.IntermedViews.Utilde
                C = buml_obj.IntermedViews.C
                global_opts = buml_obj.global_opts
                y_s_d_e = gv_info["gv"].compute_pwise_dist_in_embedding(
                    intermed_param, Utilde, C, global_opts, dist=y_s_d_e, y=y
                )
        if save1:
            save(save_dir_root, "y_s_d_e.dat", y_s_d_e)
    else:
        y_s_d_e = read(y_s_d_e_path)

    # Lengths of the embeddings of the shortest paths in the data
    save2 = save_dir_root != ""
    y_d_e2_path = save_dir_root + "/y_d_e2.dat"
    if not path_exists(y_d_e2_path):
        y_d_e2 = get_path_lengths_in_embedding_space(pred, y_s_d_e, n_proc=n_proc)
        if save2:
            save(save_dir_root, "y_d_e2.dat", y_d_e2)
    else:
        y_d_e2 = read(y_d_e2_path)

    sd_at, max_sd = compute_distortion_at(y_s_d_e, s_d_e)
    wd_at, max_wd = compute_distortion_at(y_d_e2, s_d_e)
    return sd_at, max_sd, wd_at, max_wd


def compute_points_across_tear_graph(
    views_across_tear_graph, i_mat, partition, n_batches=64, approx_bipartite_tear=True
):
    _, n = i_mat.shape
    views_across_tear_graph_row, views_across_tear_graph_col = (
        views_across_tear_graph.nonzero()
    )
    n_pairs = len(views_across_tear_graph_row)
    print("#pairs of partitions/views across tear:", n_pairs, flush=True)

    # Define per-pair processing function
    def process_pairs(ij_pairs):
        rows = []
        cols = []
        pts = []
        empty_lists = True
        for i, j in zip(ij_pairs[0], ij_pairs[1]):
            part_i = partition[i, :]
            part_j = partition[j, :]
            imat_j = i_mat[j, :]
            imat_i = i_mat[i, :]

            T_ij = imat_j.multiply(part_i).nonzero()[1]
            T_ji = imat_i.multiply(part_j).nonzero()[1]
            n_T_ij = len(T_ij)
            n_T_ji = len(T_ji)
            if (n_T_ij == 0) or (n_T_ji == 0):
                continue

            rows.append(np.repeat(T_ij, n_T_ji))
            cols.append(np.tile(T_ji, n_T_ij))
            if not approx_bipartite_tear:
                rows.append(np.repeat(T_ij, n_T_ij))
                cols.append(np.tile(T_ij, n_T_ij))

            # rows.append(
            #     np.concatenate([
            #         np.repeat(T_ij, n_T_ji),
            #         np.repeat(T_ji, n_T_ij)
            #     ])
            # )
            # if not approx_bipartite_tear:
            #     rows.append(
            #         np.concatenate([
            #             np.repeat(T_ij, n_T_ij),
            #             np.repeat(T_ji, n_T_ji)
            #         ])
            #     )
            # cols.append(
            #     np.concatenate([
            #         np.tile(T_ji, n_T_ij),
            #         np.tile(T_ij, n_T_ji)
            #     ])
            # )
            # if not approx_bipartite_tear:
            #     cols.append(
            #         np.concatenate([
            #             np.tile(T_ij, n_T_ij),
            #             np.tile(T_ji, n_T_ji)
            #         ])
            #     )
            pts.append(np.concatenate([T_ij, T_ji]).tolist())
            empty_lists = False
        if empty_lists:
            temp = np.array([]).astype(int)
            return temp, temp, temp
        else:
            return np.concatenate(rows), np.concatenate(cols), np.concatenate(pts)

    if n_batches < n_pairs:
        chunk_sz = int(n_pairs / n_batches)
        ij_pairs_batches = []
        for i in range(n_batches):
            start_ind = i * chunk_sz
            if i < n_batches:
                end_ind = start_ind + chunk_sz
            else:
                end_ind = n_pairs
            ij_pairs_batches.append(
                (
                    views_across_tear_graph_row[start_ind:end_ind],
                    views_across_tear_graph_col[start_ind:end_ind],
                )
            )
    else:
        ij_pairs_batches = [(views_across_tear_graph_row, views_across_tear_graph_col)]

    # Run in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_pairs)(ij_pairs) for ij_pairs in ij_pairs_batches
    )

    # Aggregate results
    tear_graph_row_inds = np.concatenate([r for r, _, _ in results if len(r)])
    tear_graph_col_inds = np.concatenate([c for _, c, _ in results if len(c)])

    pts_across_tear_mask = np.zeros(n, dtype=bool)
    for _, _, pts in results:
        pts_across_tear_mask[pts] = True

    print("Computing tear graph.", flush=True)
    # Build sparse tear graph
    tear_graph = csr_matrix(
        (
            np.ones(len(tear_graph_row_inds), dtype=bool),
            (tear_graph_row_inds, tear_graph_col_inds),
        ),
        shape=(n, n),
        dtype=bool,
    )
    # Symmetrize
    tear_graph = tear_graph + tear_graph.T
    # Restrict to points on the tear
    pts_across_tear = np.where(pts_across_tear_mask)[0]
    tear_graph = tear_graph[np.ix_(pts_across_tear, pts_across_tear)]
    return pts_across_tear, tear_graph


def compute_tear_graph(
    y,  # embedding |#points| x embedding dimension
    i_mat,  # incidence matrix |#views| x |#points|
    partition,  # partition matrix |#partion| x |#points| (#partition = #views)
    opts,
    overlap,  # size of overlap between views |#views| x |#views|
    i_mat_in_emb=None,  # incidence matrix in the embedding |#views| x |#points|
    approx_bipartite_tear=True,
):
    k = opts.k
    nu = opts.nu
    metric = opts.metric

    _, n = i_mat.shape
    # If i_mat_in_emb if not provided
    if i_mat_in_emb is None:
        print("Metric used for computing incidence matrix in embedding:", opts.metric)
        i_mat_in_emb = compute_incidence_matrix_in_embedding(
            y, partition, k, nu, metric
        )

    # compute the tear graph: a graph between partitions/views where ith partition
    # is connected to jth partition if they are across the tear i.e.
    # if the corresponding views are overlapping in the
    # ambient space but not in the embedding space
    overlap_in_emb = i_mat_in_emb.dot(i_mat_in_emb.T)
    overlap_in_emb.setdiag(False)
    overlap.setdiag(False)
    views_across_tear_graph = overlap - overlap.multiply(overlap_in_emb)
    views_across_tear_graph.eliminate_zeros()
    # If no partitions/views are across the tear then there is no tear
    if len(views_across_tear_graph.data) == 0:
        print("No views across the tear detected.")
        return None

    print(
        "total #pairs of overlapping partitions/views:",
        overlap.count_nonzero(),
        flush=True,
    )
    pts_across_tear, tear_graph = compute_points_across_tear_graph(
        views_across_tear_graph,
        i_mat,
        partition,
        n_batches=opts.n_jobs,
        approx_bipartite_tear=approx_bipartite_tear,
    )
    print("#vertices in tear graph =", tear_graph.shape[0], flush=True)
    print("#edges in tear graph =", len(tear_graph.data), flush=True)
    return pts_across_tear, tear_graph


def add_weights_to_tear_graph(
    param, tear_graph, pts_across_tear, cluster_label, n_batches=8
):
    tear_graph = tear_graph.copy().tocoo()
    tear_graph_row_inds = tear_graph.row
    tear_graph_col_inds = tear_graph.col
    n_edges = len(tear_graph_row_inds)
    dist_of_pts_across_tear = np.zeros(n_edges)

    def process_(start_ind, end_ind):
        temp = np.zeros(end_ind - start_ind) + np.inf
        for k in range(start_ind, end_ind):
            r_i = tear_graph_row_inds[k]
            c_i = tear_graph_col_inds[k]
            edge_i = pts_across_tear[r_i]
            edge_j = pts_across_tear[c_i]
            # for view_ind in view_cont_pts_across_tear[(edge_i, edge_j)]:
            for view_ind in [cluster_label[edge_i], cluster_label[edge_j]]:
                local_coords = param.eval_(
                    {
                        "data_mask": [edge_i, edge_j],
                        "view_index": view_ind,
                    },
                    apply_b=False,
                )
                temp[k - start_ind] = min(
                    temp[k - start_ind],
                    np.linalg.norm(local_coords[0, :] - local_coords[1, :]),
                )
        return temp

    chunk_sz = n_edges // n_batches
    start_end_inds = []
    for i in range(n_batches):
        start_ind = i * chunk_sz
        if i < n_batches - 1:
            end_ind = start_ind + chunk_sz
        else:
            end_ind = n_edges
        start_end_inds.append((start_ind, end_ind))

    results = Parallel(n_jobs=-1)(delayed(process_)(a, b) for a, b in start_end_inds)
    dist_of_pts_across_tear = np.concatenate(results)

    return csr_matrix(
        (dist_of_pts_across_tear, (tear_graph_row_inds, tear_graph_col_inds)),
        shape=tear_graph.shape,
    )


# dist = y_d_e
# y can either be the embedding of size (n_samples, d)
# or a dense distance matrix of size (n_samples, n_samples).
# The latter can either represent Euclidean distances of
def compute_tear_aware_shortest_path_distances(
    y,
    intermed_param,
    Utilde,
    C,
    c,
    global_opts,
    max_crossings=5,
    tol=1e-6,
    dist=None,
    n_batches=None,
    debug=False,
    device_num=0,
):
    M, n = Utilde.shape
    if n_batches is None:
        n_batches = global_opts.n_jobs
    if dist is None:
        # dist = squareform(pdist(y))
        print("n_nbrs used for naive shortest path distances:", global_opts.k)
        print("Metric used for naive shortest path distances:", global_opts.metric)
        dist = shortest_paths(
            y,
            n_nbrs=global_opts.k,
            metric=global_opts.metric,
            return_predqecessors=False,
        )

    n_Utilde_Utilde = Utilde.dot(Utilde.transpose())
    n_Utilde_Utilde.setdiag(False)

    output = compute_tear_graph(
        y,
        Utilde,
        C,
        global_opts,
        n_Utilde_Utilde,
        i_mat_in_emb=None,
    )
    if output is None:
        return dist

    pts_across_tear, tear_graph = output
    n_pts_across_tear = len(pts_across_tear)
    tear_graph = tear_graph.tocoo()
    tear_graph_row_inds = tear_graph.row
    tear_graph_col_inds = tear_graph.col
    tear_graph_with_weights = add_weights_to_tear_graph(
        intermed_param, tear_graph, pts_across_tear, c
    )
    dist_of_pts_across_tear = tear_graph_with_weights.data
    tear_graph_row_inds = pts_across_tear[tear_graph_row_inds]
    tear_graph_col_inds = pts_across_tear[tear_graph_col_inds]

    # make shortcuts in dist
    for k in range(len(tear_graph_row_inds)):
        dist[tear_graph_row_inds[k], tear_graph_col_inds[k]] = dist_of_pts_across_tear[
            k
        ]

    # dist_of_pts_across_tear = csr_matrix(
    #     (dist_of_pts_across_tear, (tear_graph_row_inds, tear_graph_col_inds)),
    #     shape=(n_pts_across_tear, n_pts_across_tear)
    # )

    # def process_(dist_on_subset, dist_from_pts_on_tear1, dist_from_pts_on_tear2):
    #     for k in range(len(dist_of_pts_across_tear)):
    #         r_i = pts_across_tear[tear_graph_row_inds[k]]
    #         c_i = pts_across_tear[tear_graph_col_inds[k]]
    #         val = dist_of_pts_across_tear[k]
    #         dist_on_subset = np.minimum(
    #             dist_on_subset,
    #             dist_from_pts_on_tear1[:,r_i:r_i+1] + val + dist_from_pts_on_tear2[c_i:c_i+1,:]
    #         )

    # for i_crossing in range(max_crossings):
    #     old_dist = dist.copy()
    # dist_from_tear = dist[:,pts_across_tear]
    # def process_(dist_i, dist_from_tear_i):
    #     temp = dist_from_tear_i[:,None,None] + dist_of_pts_across_tear[:,:,None] + (dist_from_tear.T)[None,:,:]
    #     dist_i = np.minimum(dist_i, np.min(temp, axis=(0,1)))
    #     return dist_i
    # for i in range(n):
    #     print(i, flush=True)
    #     dist_i = dist[i,:]
    #     dist_from_tear_i = dist_from_tear[i,:]
    #     dist_i = process_(dist_i, dist_from_tear_i)

    print("Computing tear aware distance...")
    return recompute_dist_using_tear_numpy(
        dist,
        pts_across_tear,
        max_crossings=max_crossings,
        n_batches=n_batches,
        tol=tol,
        debug=debug,
        # device_num=device_num
    )


def recompute_dist_using_tear_numpy(
    dist, pts_across_tear, max_crossings=20, n_batches=32, tol=1e-6, debug=False
):
    n = dist.shape[0]
    if debug:
        dists = [dist.copy()]
    for i_crossing in range(max_crossings):
        old_dist = dist.copy()
        dist_from_pts_across_tear = dist[pts_across_tear, :].T  # n x n_pts_across_tear

        def process_(dist, start_ind, end_ind):
            dist_ = np.zeros((end_ind - start_ind, n))
            for i_ in range(start_ind, end_ind):
                i = i_ - start_ind
                # dist_[i-start_ind,:] = np.min((dist[i,tear_graph_row_inds]+dist_of_pts_across_tear)[:,None]+dist[tear_graph_col_inds,:], axis=0)
                dist_[i, :] = np.minimum(
                    dist[i, :],
                    np.minimum(
                        dist[i, :],
                        np.min(
                            dist_from_pts_across_tear[i_, :][:, None]
                            + dist_from_pts_across_tear.T,
                            axis=0,
                        ),
                    ),
                )
            return dist_

        start_end_inds = []
        chunk_sz = n // n_batches
        for j in range(n_batches):
            start_ind = j * chunk_sz
            if j < n_batches - 1:
                end_ind = start_ind + chunk_sz
            else:
                end_ind = n
            start_end_inds.append((start_ind, end_ind))

        results = Parallel(n_jobs=-1)(
            delayed(process_)(dist.copy()[a:b, :], a, b) for a, b in start_end_inds
        )

        for i in range(len(results)):
            start_ind, end_ind = start_end_inds[i]
            dist[start_ind:end_ind, :] = results[i]

        if debug:
            dists.append(dist.copy())

        mean_abs_rel_diff = np.ma.masked_invalid(
            np.abs(dist - old_dist) / (old_dist + 1e-12)
        ).mean()
        print(
            "Mean absolute relative difference in distances:",
            mean_abs_rel_diff,
            flush=True,
        )
        if mean_abs_rel_diff < tol:
            break

    if debug:
        return dist, dists
    else:
        return dist


def recompute_dist_using_tear_cupy(
    dist,
    pts_across_tear,
    max_crossings=20,
    n_batches=32,
    tol=1e-6,
    debug=False,
    device_num=1,
):
    raise NotImplementedError()
    # cp.cuda.Device(device_num).use()
    # n = dist.shape[0]
    # dist = cp.asarray(dist.astype('float32'))
    # pts_across_tear = cp.asarray(pts_across_tear)
    # if debug:
    #     dists = [cp.asnumpy(dist.copy())]

    # start_end_inds = []
    # chunk_sz = n//n_batches
    # for j in range(n_batches):
    #     start_ind = j*chunk_sz
    #     if j < n_batches-1:
    #         end_ind = start_ind+chunk_sz
    #     else:
    #         end_ind = n
    #     start_end_inds.append((start_ind, end_ind))

    # for i_crossing in range(max_crossings):
    #     old_dist = dist.copy()
    #     dist_from_pts_across_tear = dist[pts_across_tear,:].T.copy() # n x n_pts_across_tear

    #     # Process each batch on GPU
    #     for start_ind, end_ind in start_end_inds:
    #         rows = dist[start_ind:end_ind, :]  # (batch_size, n)
    #         cross_rows = dist_from_pts_across_tear[start_ind:end_ind,:]  # (batch_size, n_pts_across_tear)

    #         # Compute batched min
    #         temp = cross_rows[:, :, None] + dist_from_pts_across_tear.T[None, :, :]  # shape: (batch, n_pts_across_tear, n)
    #         min_update = cp.min(temp, axis=1)  # shape: (batch, n)

    #         # Element-wise min with original
    #         rows = cp.minimum(rows, min_update)

    #         # Write back
    #         dist[start_ind:end_ind, :] = rows

    #     if debug:
    #         dists.append(cp.asnumpy(dist.copy()))

    #     diff = cp.abs(dist - old_dist) / (old_dist + 1e-12)
    #     mean_abs_rel_diff = np.ma.masked_invalid(cp.asnumpy(diff)).mean()
    #     print('Mean absolute relative difference in distances:', mean_abs_rel_diff, flush=True)

    #     if mean_abs_rel_diff < tol:
    #         break

    # if debug:
    #     return cp.asnumpy(dist), dists
    # else:
    #     return cp.asnumpy(dist)
