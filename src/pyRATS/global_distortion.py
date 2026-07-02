from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from pyRATS._tear_coloring import compute_tear_graph
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


def _shortest_paths(
    X,
    k,
    metric="euclidean",
):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    knn_graph = nbrs.kneighbors_graph(mode="distance")

    return shortest_path(
        knn_graph,
        return_predecessors=False,
        directed=False,
    )


def _compute_tear_aware_shortest_path_distances(
    param,
    y,
    Utilde,
    n_Utilde_Utilde,
    Utildeg,
    C,
    c,
    k,
    nu,
    metric="euclidean",
    max_crossings=5,
    tol=1e-6,
    dist=None,
    n_batches=None,
    debug=False,
    n_jobs=-1,
):
    if n_batches is None:
        n_batches = n_jobs
    if dist is None:
        dist = _shortest_paths(y, n_nbrs=k, metric=metric, return_predqecessors=False)

    pts_across_tear, tear_graph = compute_tear_graph(
        y,
        Utilde,
        C,
        n_Utilde_Utilde,
        k,
        nu,
        metric,
        False,
        n_jobs,
        i_mat_in_emb=Utildeg,
    )
    if tear_graph is None:
        return dist
        
    tear_graph = tear_graph.tocoo()
    tear_graph_row_inds = tear_graph.row
    tear_graph_col_inds = tear_graph.col
    tear_graph_with_weights = _add_weights_to_tear_graph(
        param, tear_graph, pts_across_tear, c
    )
    dist_of_pts_across_tear = tear_graph_with_weights.data
    tear_graph_row_inds = pts_across_tear[tear_graph_row_inds]
    tear_graph_col_inds = pts_across_tear[tear_graph_col_inds]

    # make shortcuts in dist
    for k in range(len(tear_graph_row_inds)):
        dist[tear_graph_row_inds[k], tear_graph_col_inds[k]] = dist_of_pts_across_tear[
            k
        ]

    final_dist = _recompute_dist_using_tear(
        dist,
        pts_across_tear,
        max_crossings=max_crossings,
        n_batches=n_batches,
        tol=tol,
        debug=debug,
    )

    return final_dist


def _add_weights_to_tear_graph(
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
                    data_mask=[edge_i, edge_j], view_index=view_ind
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

    results = Parallel(n_jobs=n_batches)(
        delayed(process_)(a, b) for a, b in start_end_inds
    )
    dist_of_pts_across_tear = np.concatenate(results)

    return csr_matrix(
        (dist_of_pts_across_tear, (tear_graph_row_inds, tear_graph_col_inds)),
        shape=tear_graph.shape,
    )


def _recompute_dist_using_tear(
    dist, pts_across_tear, max_crossings=20, n_batches=32, tol=1e-6, debug=False
):
    if n_batches < 1:
        n_batches = 5

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
        results = Parallel(n_jobs=n_batches)(
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

        if mean_abs_rel_diff < tol:
            break

    return dist


def _compute_distortion_at(y_d_e, s_d_e):
    scale_factors = (y_d_e + 1e-12) / (s_d_e + 1e-12)
    mask = np.ones(scale_factors.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_distortion = np.max(scale_factors[mask]) / np.min(scale_factors[mask])
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


def get_n_nbrs_for_distortion_analysis(
    X,
    n_nbrs_list=list(range(5, 16)),
    tol=1e-3,
    persistence=3,
    metric="euclidean",
):

    gt_dist_list = [_shortest_paths(X, k, metric=metric) for k in n_nbrs_list]

    spd_rel_mean_abs_diff = []
    for i in range(1, len(gt_dist_list)):
        temp = np.mean(
            np.abs(
                (gt_dist_list[i] - gt_dist_list[i - 1]) / (gt_dist_list[i - 1] + 1e-12)
            )
        )
        spd_rel_mean_abs_diff.append(temp)

    while True:
        temp = np.array(spd_rel_mean_abs_diff) < tol
        if np.sum(temp) == 0:
            tol = 2 * tol
        else:
            temp2 = np.zeros(len(temp) - persistence + 1, dtype=bool)
            for i in range(len(temp) - persistence + 1):
                if np.prod(temp[i : i + persistence]):
                    temp2[i] = 1
            if np.sum(temp2) == 0:
                tol = 2 * tol
            else:
                n_nbrs_ind = np.argmax(temp2)
                break

    return n_nbrs_list[n_nbrs_ind]


def compute_global_distortion(
    X,
    y,
    n_nbrs,
    model,
    metric="euclidean",
    max_crossings=20,
):

    gt_dist = _shortest_paths(X, n_nbrs, metric=metric)

    emb_dist = _shortest_paths(y, n_nbrs)

    if hasattr(model, "to_tear") and model.to_tear:
        emb_dist = _compute_tear_aware_shortest_path_distances(
            model.param,
            y,
            model.Utilde,
            model.n_Utilde_Utilde,
            model.Utildeg,
            model.C,
            model.c,
            n_nbrs,
            model.nu,
            metric,
            max_crossings=max_crossings,
            dist=emb_dist,
            n_jobs=model.n_jobs,
        )

    return _compute_distortion_at(emb_dist, gt_dist)

def find_best_hyperparams(
    dist_ats,
):

    best_hyp_param = {}
    for algo in dist_ats:
        hyp_params = list(dist_ats[algo].keys())
        if len(hyp_params) == 0:
            continue
        max_dist_at = np.max(np.array(list(dist_ats[algo].values())), axis=1)
        if len(max_dist_at) == 0:
            print(
                "Max of distortion distributions are possibly infinity across hyperparameters for",
                algo,
            )
            continue
        i = np.nanargmin(max_dist_at)
        best_hyp_param[algo] = hyp_params[i]

    dist_dict = {}
    for algo in best_hyp_param:
        dist_dict[algo] = {algo: dist_ats[algo][best_hyp_param[algo]]}

    return best_hyp_param
