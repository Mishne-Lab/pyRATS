import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

"""
References:
Landa, B., Coifman, R. R., & Kluger, Y. (2021). 
Doubly stochastic normalization of the gaussian kernel is robust to heteroskedastic noise. 
SIAM journal on mathematics of data science, 3(1), 388-413.

Landa, B., & Cheng, X. (2023). 
Robust inference of manifold density and geometry by doubly stochastic scaling. 
SIAM Journal on Mathematics of Data Science, 5(3), 589-614.

Cheng, X., & Landa, B. (2024). 
Bi-stochastically normalized graph Laplacian: convergence to manifold Laplacian and robustness to outlier noise. 
Information and Inference: A Journal of the IMA, 13(4), iaae026.
"""
def sinkhorn(
        K,
        maxiter=10000,
        delta=1e-12,
        boundC = 1e-8,
        print_freq=1000
    ):
    n = K.shape[0]
    r = np.ones((n,1))
    u = np.ones((n,1))
    v = r/(K.dot(u))
    x = np.sqrt(u*v)
    assert np.min(x) > boundC, 'assert min(x) > boundC failed.'
    for tau in range(maxiter):
        error =  np.max(np.abs(u*(K.dot(v)) - r))
        if tau%print_freq:
            print('Error:', error, flush=True)
        
        if error < delta:
            print('Sinkhorn converged at iter:', tau)
            break

        u = r/(K.dot(v))
        v = r/(K.dot(u))
        x = np.sqrt(u*v)
        if np.sum(x<boundC) > 0:
            print('boundC not satisfied at iter:', tau)
            x[x < boundC] = boundC
        
        u=x
        v=x
    x = x.flatten()
    K.data = K.data*x[K.row]*x[K.col]
    return K

def graph_laplacian(
    neigh_ind,
    neigh_dist, # assume no self-loops
    which='unnorm',
    tuning='self',
    k_tune=7,
    kernel='gaussian',
    ds_max_iter=0,
    return_diag=False,
    use_out_degree=True
):
    n = neigh_ind.shape[0]
    k_nn = neigh_ind.shape[1]
    assert k_nn >= k_tune, 'k_nn=%d < k_tune=%d' % (k_nn, k_tune)

    row_inds = np.repeat(np.arange(n), k_nn)
    col_inds = neigh_ind.flatten()
    data = neigh_dist.flatten()

    if tuning is not None:
        # Compute local scale
        if tuning in ['self', 'solo']:
            sigma = neigh_dist[:,k_tune-1]
            if tuning=='self': # scaling depends on sigma_i and sigma_j
                autotune = sigma[row_inds]*sigma[col_inds]
            else:# tuning=='solo': # scaling depends on sigma_i only
                autotune = sigma[row_inds]**2
        elif tuning=='median': # scaling is fixed across data points
            autotune = np.median(sigma)**2
        else:
            raise NotImplementedError("tuning=%s not implemented." % tuning)

    eps = np.finfo(np.float64).eps
    if kernel=='binary':
        K = np.ones(row_inds)
        autotune = None
    elif kernel=='gaussian':
        K = np.exp(-data**2/autotune) + eps
    elif kernel=='laplacian':
        K = np.exp(-data/np.sqrt(autotune)) + eps

    K = csr_matrix(
        (K, (row_inds, col_inds)),
        shape=(n,n)
    )
    ones_like_K = csr_matrix(
        (np.ones(row_inds.shape[0]), (row_inds, col_inds)),
        shape=(n,n)
    )
    # average symmetrization
    K = K + K.T
    ones_like_K = ones_like_K + ones_like_K.T
    K.data /= ones_like_K.data

    if ds_max_iter:
        K = sinkhorn(K.tocoo(), maxiter=ds_max_iter)
        
    if which=='diffusion':
        Dinv = 1/(K.sum(axis=1).reshape((n,1)))
        K = K.multiply(Dinv).multiply(Dinv.transpose())
        which = 'symnorm'

    if which=='symnorm':
        normed=True
    else:
        normed=False

    L = laplacian(
        K,
        normed=normed,
        return_diag=return_diag,
        use_out_degree=use_out_degree
    )
    return L


def spectrum_of_laplacian_from_neighbors(
    neigh_ind,
    neigh_dist, # assumes no self-loops
    opts = {}
):
    default_opts = {
        'which': 'unnorm',
        'tuning': 'self',
        'k_tune': 7,
        'kernel': 'gaussian',
        'ds_max_iter': 0,
        'n_eig': 10,
        'n_ignore': 1
    }
    default_opts.update(opts)
    opts = default_opts
    which = opts['which']
    tuning = opts['tuning']
    k_tune = opts['k_tune']
    kernel = opts['kernel']
    ds_max_iter = opts['ds_max_iter']
    n_eig = opts['n_eig']
    n_ignore = opts['n_ignore']

    n = neigh_ind.shape[0]
    
    np.random.seed(42)
    v0 = np.ones(n)/np.sqrt(n)
    if which in ['unnorm', 'symnorm']:
        L = graph_laplacian(
            neigh_ind, neigh_dist,
            which=which, tuning=tuning, k_tune=k_tune,
            kernel=kernel, ds_max_iter=ds_max_iter
        )
        lmbda, phi = eigsh(L, k=n_eig+n_ignore, v0=v0, sigma=-1e-3)
    elif which in ['random_walk', 'diffusion']:
        if which == 'random_walk':
            which = 'symnorm'
        L, sqrt_D = graph_laplacian(
            neigh_ind, neigh_dist,
            which=which, tuning=tuning, k_tune=k_tune,
            kernel=kernel, ds_max_iter=ds_max_iter,
            return_diag=True
        )
        lmbda, phi = eigsh(L, k=n_eig+n_ignore, v0=v0, sigma=-1e-3)
        
        # Uncomment and return if the Laplacian is needed
        # L = L.multiply(1/sqrt_D[:,np.newaxis]).multiply(sqrt_D[np.newaxis,:])
        phi = phi/sqrt_D[:,np.newaxis]
        
        #TODO: Is this normalization needed?
        # phi = phi/(np.linalg.norm(phi,axis=0)[np.newaxis,:])

    return lmbda[n_ignore:], phi[:,n_ignore:]