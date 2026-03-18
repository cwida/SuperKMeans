import os
import sys

# Add parent directory to path for bench_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["BLIS_NUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

import mlpack
import numpy as np
import time
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

VALID_ALGORITHMS = {
    "naive": "NaiveKMeans",
    "elkan": "ElkanKMeans",
    "hamerly": "HamerlyKMeans",
    "pelleg-moore": "PellegMooreKMeans",
    "dualtree": "DualTreeKMeans",
}

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove200"
    variant = sys.argv[2] if len(sys.argv) > 2 else "naive"
    experiment_name = sys.argv[3] if len(sys.argv) > 3 else "end_to_end"

    if variant not in VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm variant '{variant}'. "
            f"Choose from {list(VALID_ALGORITHMS.keys())}"
        )
    algorithm = f"mlpack_{variant}"

    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = 5 # get_default_n_clusters(num_vectors)
    n_iter = 10 # MAX_ITERS
    threads = threads

    print(f"=== Running algorithm: {algorithm} ({VALID_ALGORITHMS[variant]}) ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")

    filename = get_data_path(dataset)
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)
    if dataset in ANGULAR_DATASETS:
        print(f"\nWARNING: Dataset '{dataset}' should use spherical k-means, "
              f"but mlpack does not support this. Results may be suboptimal.")

    # mlpack expects data as (num_dimensions, num_vectors) — column-major
    data_transposed = np.ascontiguousarray(data.T, dtype=np.float64)

    with Timer() as timer:
        result = mlpack.kmeans(
            input_=data_transposed,
            clusters=num_centroids,
            algorithm=variant,
            max_iterations=n_iter,
            seed=42,
            labels_only=True,
        )
    construction_time_ms = timer.get_milliseconds()

    # Extract centroids: mlpack returns (d, k) column-major -> transpose to (k, d) row-major
    centroids = result['centroid'].astype(np.float32)
    # Extract assignments: mlpack returns a (1, num_vectors) row of cluster labels
    assignments = result['output'].flatten().astype(np.int64)

    # mlpack does not report actual iterations; assume max_iterations were run
    actual_iterations = n_iter

    # Compute final objective (inertia): sum of squared distances to assigned centroids
    diffs = data - centroids[assignments]
    final_objective = float(np.sum(diffs ** 2))

    print(f"\nTraining completed in {construction_time_ms:.2f} ms")
    print(f"Actual iterations: {actual_iterations} (requested: {n_iter})")
    print(f"Final objective (inertia): {final_objective}")

    gt_filename = get_ground_truth_path(dataset)
    queries_filename = get_query_path(dataset)
    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")
        gt_dict = load_ground_truth(gt_filename)
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = N_QUERIES
        queries = queries[:n_queries * num_dimensions].reshape(n_queries, num_dimensions)
        print(f"Using {n_queries} queries (loaded {len(gt_dict)} from ground truth)")

        results_knn_10 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 10)
        print_recall_results(results_knn_10, 10)
        results_knn_100 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 100)
        print_recall_results(results_knn_100, 100)

        config_dict = {
            "algorithm": variant,
            "algorithm_class": VALID_ALGORITHMS[variant],
            "max_iterations": str(n_iter),
            "seed": "42",
        }

        write_results_to_csv(
            experiment_name, algorithm, dataset, n_iter, actual_iterations,
            num_dimensions, num_vectors, num_centroids, construction_time_ms,
            threads, final_objective, config_dict,
            results_knn_10, results_knn_100
        )
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
        print("Skipping CSV output (recall computation requires ground truth)")
