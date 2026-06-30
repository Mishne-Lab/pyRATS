# RATS: Riemannian Alignment of Tangent Spaces

RATS follows a bottom-up approach to manifold learning, integrating local representations of data into a globally consistent embedding.   
This guide provides practical instructions for preprocessing data and tuning the key hyperparameters in RATS: `min_cluster_size`, `n_neighbors`, and the `tearing` flag.

---

## Table of Contents

1. [Computational Complexity](https://docs.google.com/document/d/1uWP-hmgtU0tC-d6lGc99Urnz1oW7vLdJnXJmwUUa7cU/edit#computational-complexity)  
2. [Hyperparameter Guidelines](https://docs.google.com/document/d/1uWP-hmgtU0tC-d6lGc99Urnz1oW7vLdJnXJmwUUa7cU/edit#hyperparameter-guidelines)  
3. [Preprocessing & Scaling](https://docs.google.com/document/d/1uWP-hmgtU0tC-d6lGc99Urnz1oW7vLdJnXJmwUUa7cU/edit#preprocessing--scaling)  
4. [Manifold Tearing](https://docs.google.com/document/d/1uWP-hmgtU0tC-d6lGc99Urnz1oW7vLdJnXJmwUUa7cU/edit#manifold-tearing)  
5. [Future Roadmap](https://docs.google.com/document/d/1uWP-hmgtU0tC-d6lGc99Urnz1oW7vLdJnXJmwUUa7cU/edit#future-roadmap)

---

## Computational Complexity

To understand how to tune RATS, it is helpful to understand the underlying algorithmic time complexities. Let $ n$ be the number of samples, $ m$ be the number of intermediate views, $ k$ be `n_neighbors`, $ D$ be the input dimension, and $ d$ be the intrinsic dimension.

| Operation | Time Complexity | Notes |
| :---- | :---- | :---- |
| **Nearest Neighbor Graph** | $\\mathcal{O}(n \\log n \\cdot D)$ | Standard tree-based construction. |
| **Local Linear PCA** | $\\mathcal{O}(n \\cdot k \\cdot D \\cdot d)$ | Assumes truncated SVD for intrinsic dims. |
| **Local Kernel PCA** | $\\mathcal{O}(n \\cdot (k^3 \+ k^2 \\cdot D))$ | Kernel matrix formulation \+ eigendecomposition. |
| **Intermediate Views** | $\\mathcal{O}(n \\cdot k \\cdot \\eta\_{\\min}(k^2 \+ k D d))$ | Where $\\eta\_{\\min}$ is `min_cluster_size`. |
| **View Post-Processing** | $\\mathcal{O}(n k (k^2 \+ kDd))$ |  |
| **Laplacian Pseudo-Inverse** | $\\mathcal{O}(n m^2)$ | Linear in $ n$, quadratic in $ m$. |

---

## Hyperparameter Guidelines

### 1\. `min_cluster_size` ($\\eta\_{\\min}$)

To align intermediate views, RATS solves a non-convex optimization problem by computing the pseudo-inverse of a combinatorial Laplacian. This bipartite graph captures the correspondence between $m$ intermediate views and $n$ points. Because the number of views $m$ is typically on the order of $n / \\eta\_{\\min}$, therefore $\\eta\_{\\min}$ should be larger for lower computational time and robust alignment. But too large $\\eta\_{\\min}$ may result in bigger intermediate views which may result in high global distortion. 

* **Recommended Range** for datasets on the order of several thousands of points: `[3, 10].`

### 2\. `n_neighbors` ($k$)

Because the runtime for intermediate view construction scales cubically with the number of neighbors, this value must be kept small, while too small of a value may result in smaller overlaps between views leading to inaccurate alignment. 

* **Recommended Range:** for datasets on the order of several thousands of points: `[15, 50]`

### 3\. `metric`

Metric to find the nearest neighbors i.e. local neighborhoods in the data space.   
Typical options: `'euclidean','cosine'` etc.

### 4\. `kernel`

The kernel used in local kernel PCA during the local view construction.  
Typical options: `'linear'/None, 'cosine','rbf'` etc.

---

## Preprocessing & Scaling

### Handling High Input Dimensions ($D \> 100$)

As shown in the complexity table, the input dimension ($ D$) plays a compounding role in nearest neighbor construction, local PCA, and post-processing.

* **Recommendation:** If your dataset has a high input dimension ($D \> 100$), perform **global PCA** on the entire dataset to reduce the dimension to $D \< 100$ before running RATS. This will prevent severe computational bottlenecks.

### Handling Large Datasets ($n \\gg 10^4$)

The current pseudo-inverse computation restricts standard execution times (10–30 seconds) to $n \\approx 10^4$.

* **Recommendation:** If your dataset is orders of magnitude larger than $10,000$, we highly recommend subsampling it prior to embedding. Use **uniform subsampling** or **topological subsampling** ([Kloke, J. & Carlsson, G. Topological De-Noising: Strengthening the Topological Signal.](http://paperpile.com/b/BLBzQS/XNRkN)).

---

## Manifold Tearing

Unlike many existing manifold learning algorithms, RATS is capable of tearing closed or non-orientable manifolds. This is achieved by enabling the `tearing` flag. 

* **If `tearing = False`:** The Laplacian pseudo-inverse is computed once.  
* **If `tearing = True`:** The combinatorial Laplacian and its pseudo-inverse must be recomputed at each "outer" iteration of the alignment procedure, resulting in longer run-times. 

---

## Future Roadmap

As RATS matures, we are actively developing the following computational optimizations:

1. **Approximate Pseudo-Inverse Solvers:** We plan to implement approximate Laplacian solvers (Spielman and Teng, 2014\) that scale linearly with matrix dimensions, drastically reducing base alignment time.  
2. **Overlap Tracking for Tearing:** We will track whether the overlapping embedding structure has meaningfully changed between iterations. If the tear has stabilized, we can safely bypass redundant pseudo-inverse updates.  
3. **Woodbury Matrix Identity Updates:** We aim to model the Laplacian of the new overlapping structure as a mathematical perturbation of the previous state. By employing Woodbury pseudo-inverse formulas, we can efficiently update the inverse instead of recomputing it from scratch.
