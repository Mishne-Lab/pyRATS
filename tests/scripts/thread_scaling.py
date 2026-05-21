"""Measure how RATS.fit_transform scales with n_jobs on a kleinbottle.

Usage:
    python tests/scripts/thread_scaling.py [--n N] [--jobs 1 2 4 8] [--repeats R]

Prints a small table of wall times and speedup vs. the n_jobs=1 baseline.
"""

import argparse
import os
import sys
import time

# Allow importing src and examples
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyRATS import rats
from examples import datasets


def run_once(X, n_jobs):
    model = rats.RATS(
        n_components=2,
        n_neighbors=14,
        cost_function="distortion",
        min_cluster_size=5,
        n_iter=3,
        nu=4,
        verbose=False,
        n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    model.fit_transform(X=X)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000,
                        help="Klein bottle sample count (default 2000).")
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="List of n_jobs values to benchmark.")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Repeats per n_jobs (best time kept).")
    args = parser.parse_args()

    print(f"Loading kleinbottle4d with n={args.n}...")
    ds = datasets.Datasets()
    X, _, _ = ds.kleinbottle4d(n=args.n)

    # Warmup so JIT / first-touch costs don't pollute the first measurement.
    print("Warmup (n_jobs=1)...")
    run_once(X, 1)

    print(f"\n| n_jobs | best (s) | speedup | all runs (s)            |")
    print(f"|--------|----------|---------|-------------------------|")
    baseline = None
    for nj in args.jobs:
        runs = [run_once(X, nj) for _ in range(args.repeats)]
        best = min(runs)
        if baseline is None:
            baseline = best
        speedup = baseline / best
        runs_str = ", ".join(f"{r:.2f}" for r in runs)
        print(f"| {nj:6d} | {best:8.3f} | {speedup:7.2f} | {runs_str:23s} |")


if __name__ == "__main__":
    main()
