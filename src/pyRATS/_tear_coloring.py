import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
import scipy

from joblib import delayed, Parallel
from tqdm import tqdm

from pyRATS import _utils


def compute_color_of_pts_on_tear(
    y,
    i_mat,
    partition,
    overlap,
    tear_color_eig_inds,
    k,
    nu,
    metric,
    verbose,
    n_jobs,
    i_mat_in_emb=None,
):

    return compute_spectral_color_of_pts_on_tear(
        y,
        i_mat,
        partition,
        overlap,
        tear_color_eig_inds,
        k,
        nu,
        metric,
        verbose,
        n_jobs,
        i_mat_in_emb=i_mat_in_emb,
    )


def compute_spectral_color_of_pts_on_tear(
    y,  # embedding |#points| x embedding dimension
    i_mat,  # incidence matrix |#views| x |#points|
    partition,  # partition matrix |#partion| x |#points| (#partition = #views)
    overlap,  # size of overlap between views |#views| x |#views|
    tear_color_eig_inds,
    k,
    nu,
    metric,
    verbose,
    n_jobs,
    color_cutoff_frac=0.001,
    color_cutoff_frac2=0.0,
    i_mat_in_emb=None,  # incidence matrix in the embedding |#views| x |#points|
    return_tear_graph_info=False,
):

    pts_across_tear, tear_graph = compute_tear_graph(
        y,
        i_mat,
        partition,
        overlap,
        k,
        nu,
        metric,
        verbose,
        n_jobs,
        i_mat_in_emb=i_mat_in_emb,
    )
    if pts_across_tear is None:
        if verbose:
            tqdm.write("No tear detected.")
        return None
    n_pts_across_tear = len(pts_across_tear)
    n_comp, labels = connected_components(
        tear_graph, directed=False, return_labels=True
    )
    if verbose:
        tqdm.write(f"Number of components in the tear graph: {n_comp}")

    n_points_in_comp = []
    for i in range(n_comp):
        comp_i = labels == i
        n_points_in_comp.append(np.sum(comp_i))

    _, n = i_mat.shape
    max_diversity = np.max(tear_color_eig_inds) + 1
    color_of_pts_on_tear = np.zeros((n, max_diversity)) + np.nan
    offset = np.zeros(max_diversity)

    # iterate from large to small components in tear graph
    for i in np.flip(np.argsort(n_points_in_comp)).tolist():
        comp_i = labels == i
        n_comp_i = np.sum(comp_i)
        scale = n_comp_i / n_pts_across_tear
        if verbose:
            tqdm.write(f"#points in the tear component no. {i} are: {n_comp_i}")

        # If the component is very small then assign the constant color
        if n_comp_i <= max(3, int(color_cutoff_frac * n)):
            if n_comp_i <= int(color_cutoff_frac2 * n):
                continue
            color_of_pts_on_tear[pts_across_tear[comp_i], :] = offset + scale / 2
            offset += scale
            continue

        # Construct laplacian on the i-th component of the tear graph
        tear_graph_comp_i = tear_graph[np.ix_(comp_i, comp_i)]
        tear_graph_comp_i = laplacian(tear_graph_comp_i.astype("float"))

        # Compute color of points on the points in the i-th component of the tear graph
        np.random.seed(42)
        v0 = np.random.uniform(0, 1, tear_graph_comp_i.shape[0])
        n_eigs = min(n_comp_i - 1, max_diversity)
        _, colors_ = scipy.sparse.linalg.eigsh(
            tear_graph_comp_i, v0=v0, k=n_eigs, sigma=-1e-3
        )
        colors_max = np.max(colors_, axis=0)[None, :]
        colors_min = np.min(colors_, axis=0)[None, :]
        colors_ = (colors_ - colors_min) / (
            colors_max - colors_min + 1e-12
        )  # scale to [0,1]
        colors_ = offset[None, :n_eigs] + colors_ * scale

        # repeat the last color (max_diversity-n_eigs) times
        if max_diversity - n_eigs > 0:
            colors_ = np.concatenate(
                [colors_, colors_[:, -1] * np.ones((1, max_diversity - n_eigs))], axis=1
            )

        # color_of_pts_on_tear[np.ix_(pts_on_tear[comp_i],np.arange(n_eigs))] = colors_
        color_of_pts_on_tear[pts_across_tear[comp_i], :] = colors_

        offset += scale

    if return_tear_graph_info:
        return color_of_pts_on_tear, [labels]
    else:
        return color_of_pts_on_tear


def compute_tear_graph(
    y,  # embedding |#points| x embedding dimension
    i_mat,  # incidence matrix |#views| x |#points|
    partition,  # partition matrix |#partion| x |#points| (#partition = #views)
    overlap,  # size of overlap between views |#views| x |#views|
    k,
    nu,
    metric,
    verbose,
    n_jobs,
    i_mat_in_emb=None,  # incidence matrix in the embedding |#views| x |#points|
    approx_bipartite_tear=True,
):

    _, n = i_mat.shape
    # If i_mat_in_emb if not provided
    if i_mat_in_emb is None:
        if verbose:
            tqdm.write(f"Metric used for computing incidence matrix in embedding: {metric}")
        i_mat_in_emb = _utils.compute_incidence_matrix_in_embedding(
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
        if verbose:
            tqdm.write("No views across the tear detected.")
        return None, None

    if verbose:
        tqdm.write(
            f"total #pairs of overlapping partitions/views: {overlap.count_nonzero()}"
        )
    pts_across_tear, tear_graph = compute_points_across_tear_graph(
        views_across_tear_graph,
        i_mat,
        partition,
        approx_bipartite_tear=approx_bipartite_tear,
        n_jobs=n_jobs,
    )
    if verbose:
        tqdm.write(f"#vertices in tear graph = {tear_graph.shape[0]}")
        tqdm.write(f"#edges in tear graph = {len(tear_graph.data)}")
    return pts_across_tear, tear_graph


def compute_points_across_tear_graph(
    views_across_tear_graph,
    i_mat,
    partition,
    n_batches=64,
    approx_bipartite_tear=True,
    verbose=False,
    n_jobs=-1,
):
    _, n = i_mat.shape
    views_across_tear_graph_row, views_across_tear_graph_col = (
        views_across_tear_graph.nonzero()
    )
    n_pairs = len(views_across_tear_graph_row)
    if verbose:
        tqdm.write(f"#pairs of partitions/views across tear: {n_pairs}")

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
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pairs)(ij_pairs) for ij_pairs in ij_pairs_batches
    )

    # Aggregate results
    tear_graph_row_inds = np.concatenate([r for r, _, _ in results if len(r)])
    tear_graph_col_inds = np.concatenate([c for _, c, _ in results if len(c)])

    pts_across_tear_mask = np.zeros(n, dtype=bool)
    for _, _, pts in results:
        pts_across_tear_mask[pts] = True
    if verbose:
        tqdm.write("Computing tear graph.")
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
