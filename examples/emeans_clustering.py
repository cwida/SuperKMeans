import numpy as np
import sys
import time
from sklearn.datasets import make_blobs
from superkmeans import SuperEMeans


def print_usage(program_name):
    print(f"Usage: {program_name} [n] [d] [k]")
    print("  n: Number of vectors (default: 1000000)")
    print("  d: Dimensionality (default: 768)")
    print("  k: Number of clusters (default: 1000)")
    print("Example:")
    print(f"  {program_name} 500000 512 100")


def main():
    n = 1000000
    d = 768
    k = 1000

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage(sys.argv[0])
            return 0
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        d = int(sys.argv[2])
    if len(sys.argv) > 3:
        k = int(sys.argv[3])

    print(f"Parameters: n={n}, d={d}, k={k}")
    print(f"Generating {n} vectors with {d} dimensions...")
    X, _ = make_blobs(n_samples=n, n_features=d, centers=100, random_state=42)
    data = X.astype(np.float32)

    emeans = SuperEMeans(
        n_clusters=k,
        dimensionality=d,
    )

    print("Generating centroids...")
    start_time = time.time_ns()
    centroids = emeans.train(data)
    end_time = time.time_ns()
    print(f"Index built in {(end_time - start_time) / 1000000:.2f} milliseconds")

    print("Assigning points to clusters...")
    assignments = emeans.assign_training_points(data, centroids)

    print("Done!")



if __name__ == "__main__":
    main()
