from joblib import Parallel, delayed

import os
import sys
import pickle
import itertools
import argparse

# Allow importing examples.datasets
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyRATS import rats
from examples import datasets


def save(dirpath, fname, data, verbose=True):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    fpath = os.path.join(dirpath, fname)
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print("Saved data in", fpath, flush=True)


def load_dataset(name):
    if name == "curvedtorus3d":
        X, labels, ddX, _ = datasets.Datasets().curvedtorus3d(Rmax=0.3, n=5000)
    elif name == "kleinbottle":
        X, labels, _ = datasets.Datasets().kleinbottle4d(n=5000)
    elif name == "barbell":
        X, labels, ddX = datasets.Datasets().barbell()
    elif name == "small_noisyswissroll":
        X, labels, ddX = datasets.Datasets().noisyswissroll(RES=50, noise=0.0075)
    elif name == "small_spherewithhole":
        X, labels, ddX = datasets.Datasets().spherewithhole(n=2000)
    elif name == "swissrollwithhole":
        X, labels, ddX = datasets.Datasets().swissrollwithholegrid(RES=50)
    elif name == "squarewithtwoholes":
        X, labels, ddX = datasets.Datasets().squarewithtwoholesgrid(RES=70)
    else:
        raise NotImplementedError()

    return X, labels


def run_model(k, eta_min, cost_fn_name, dataset_name, dirpath, force_compute=False):
    fname = f"dataset={dataset_name}_k={k}_eta_min={eta_min}_cost-fn={cost_fn_name}.res"
    path = os.path.join(dirpath, fname)
    
    if os.path.exists(path) and not force_compute:
        return

    X, _ = load_dataset(dataset_name)

    model = rats.RATS(
        d=2,
        k=k,
        cost_fn_name=cost_fn_name,
        eta_min=eta_min,
        max_iter=3,
        nu=4,
        verbose=True,
    )
    y = model.fit_transform(X=X)

    color_of_pts_on_tear = model.compute_color_of_pts_on_tear(y, [1])
    d = {
        "y": y,
        "color_of_pts_on_tear": color_of_pts_on_tear,
        "Utilde": model.Utilde,
        "C": model.C,
        "n_Utilde_Utilde": model.n_Utilde_Utilde,
        "c": model.c,
        "Utildeg": model.Utildeg,
    }

    save(
        dirpath=dirpath,
        fname=fname,
        data=d,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate_snapshots", description="Run toy examples to generate baselines.")
    parser.add_argument(
        "--path-to-results-dir",
        type=str,
        required=True,
        help="Results will be stored in this directory.",
    )
    parser.add_argument(
        "--force-compute", action="store_true", help="Re-compute if results already exist."
    )
    parser.add_argument(
        "--fast-mode", action="store_true", help="Run a minimal test subset for fast CI execution."
    )
    args = parser.parse_args()

    force_compute = args.force_compute
    dirpath = args.path_to_results_dir.rstrip("/")
    
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    if args.fast_mode:
        dataset_names = ["small_spherewithhole"]
        ks = [14]
        eta_mins = [5]
        cost_fn_names = ["distortion"]
    else:
        dataset_names = [
            "curvedtorus3d",
            "kleinbottle",
            "barbell",
            "small_noisyswissroll",
            "small_spherewithhole",
            "swissrollwithhole",
            "squarewithtwoholes",
        ]
        ks = [14, 21]
        eta_mins = [5, 10]
        cost_fn_names = ["distortion", "alignment"]

    argslist = list(itertools.product(ks, eta_mins, cost_fn_names, dataset_names))

    Parallel(n_jobs=-1)(
        delayed(run_model)(*args, dirpath, force_compute) for args in argslist
    )
