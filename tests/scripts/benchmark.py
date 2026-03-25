import os
import sys
import time
import json
import argparse

# Allow importing src and examples
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyRATS import rats
from examples import datasets

def run_benchmark(n_samples, n_neighbors=14, min_cluster_size=5, cost_function="distortion"):
    sys.stderr.write(f"Running benchmark for n={n_samples}, n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}...\n")
    sys.stderr.flush()
    
    # Load dataset
    ds = datasets.Datasets()
    X, _, _ = ds.kleinbottle4d(n=n_samples)
    
    model = rats.RATS(
        n_components=2,
        n_neighbors=n_neighbors,
        cost_function=cost_function,
        min_cluster_size=min_cluster_size,
        n_iter=3,
        nu=4,
        verbose=False, # Set to False for cleaner benchmark output
    )
    
    start_time = time.perf_counter()
    model.fit_transform(X=X)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    sys.stderr.write(f"Completed in {duration:.4f}s\n")
    sys.stderr.flush()
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RATS performance benchmarks.")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--fast", action="store_true", help="Run only a small subset for testing")
    args = parser.parse_args()

    if args.fast:
        sample_sizes = [400, 400, 500]
    else:
        sample_sizes = [500 , 500, 1000, 2000, 5000, 10_000, 20_000]
    
    results = []

    print("\n| Sample Size | Duration (s) |")
    print("|-------------|--------------|")
    for n in sample_sizes:
        duration = run_benchmark(n)
        results.append({
            "name": f"Klein Bottle {n} samples",
            "unit": "s",
            "value": duration
        })
        print(f"| {n:11} | {duration:12.4f} |")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
