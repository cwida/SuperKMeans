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

import numpy as np
import time
import evoc
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    algorithm = "evoc"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove200"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "end_to_end"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    n_iter = MAX_ITERS
    threads = threads

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"threads={threads}, n_iter={n_iter}")

    filename = get_data_path(dataset)
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)
    if dataset in ANGULAR_DATASETS:
        print(
            f"\nWARNING: Dataset '{dataset}' should use spherical k-means, "
            f"but EVoC (like scikit-learn) is not explicitly spherical. "
            f"Results may be suboptimal."
        )

    # Configure EVoC. We construct it with default args and pass
    # algorithmic parameters to fit(), as expected by the installed version.
    clusterer = evoc.EVoC()

    print(f"Fitting data with EVoC")
    with Timer() as timer:
        clusterer.fit(
            data,
            # base_min_cluster_size=100,
            approx_n_clusters=num_centroids,
            n_epochs=10,
            random_state=42,
            max_layers=1,
        )
    construction_time_ms = timer.get_milliseconds()
    print(f"\nTraining completed in {construction_time_ms:.2f} ms")
    exit()

    # Derive cluster assignments from the chosen EVoC layer.
    # With max_layers=1, clusterer.labels_ already corresponds to that layer.
    raw_labels = np.asarray(clusterer.labels_, dtype=int)

    # Remap labels to ensure they are consecutive integers in [0, num_centroids-1]
    # and handle noise points (-1) by assigning them to a dedicated "noise" cluster
    # at the end, so compute_recall can safely use them.
    unique_labels = np.unique(raw_labels)
    positive_labels = unique_labels[unique_labels >= 0]

    label_map = {}
    for new_idx, old_label in enumerate(positive_labels):
        label_map[int(old_label)] = int(new_idx)

    has_noise = np.any(raw_labels < 0)
    noise_cluster_id = len(positive_labels) if has_noise else None

    assignments = np.empty_like(raw_labels)
    for i, lab in enumerate(raw_labels):
        if lab >= 0:
            assignments[i] = label_map[int(lab)]
        else:
            # Map all noise points to a single extra cluster
            assignments[i] = noise_cluster_id

    num_centroids = int(assignments.max() + 1)

    # Compute centroids for each cluster as the mean of assigned points.
    centroids = np.zeros((num_centroids, num_dimensions), dtype=data.dtype)
    for c in range(num_centroids):
        mask = assignments == c
        if np.any(mask):
            centroids[c] = data[mask].mean(axis=0)

    print(f"Total clusters (including noise cluster if present): {num_centroids}")

    # EVoC does not expose a standard k-means objective or iteration count.
    # For CSV compatibility we record iterations as 0 and objective as NaN.
    actual_iterations = 0
    final_objective = float("nan")

    gt_filename = get_ground_truth_path(dataset)
    queries_filename = get_query_path(dataset)
    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")
        gt_dict = load_ground_truth(gt_filename)
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = N_QUERIES
        queries = queries[: n_queries * num_dimensions].reshape(
            n_queries, num_dimensions
        )
        print(f"Using {n_queries} queries (loaded {len(gt_dict)} from ground truth)")

        results_knn_10 = compute_recall(
            gt_dict, assignments, queries, centroids, num_centroids, 10
        )
        print_recall_results(results_knn_10, 10)
        results_knn_100 = compute_recall(
            gt_dict, assignments, queries, centroids, num_centroids, 100
        )
        print_recall_results(results_knn_100, 100)

        # Record basic configuration for reproducibility.
        # Use getattr with defaults to support multiple EVoC versions.
        config_dict = {
            "noise_level": str(getattr(clusterer, "noise_level", "NA")),
            "base_min_cluster_size": str(
                getattr(clusterer, "base_min_cluster_size", "NA")
            ),
            "base_n_clusters": str(getattr(clusterer, "base_n_clusters", "NA")),
            "approx_n_clusters": str(getattr(clusterer, "approx_n_clusters", "NA")),
            "n_neighbors": str(getattr(clusterer, "n_neighbors", "NA")),
            "min_samples": str(getattr(clusterer, "min_samples", "NA")),
            "n_epochs": str(getattr(clusterer, "n_epochs", "NA")),
            "node_embedding_init": str(
                getattr(clusterer, "node_embedding_init", "NA")
            ),
            "symmetrize_graph": str(
                getattr(clusterer, "symmetrize_graph", "NA")
            ).lower(),
            "node_embedding_dim": str(getattr(clusterer, "node_embedding_dim", "NA")),
            "neighbor_scale": str(getattr(clusterer, "neighbor_scale", "NA")),
            "random_state": str(getattr(clusterer, "random_state", "NA")),
            "min_similarity_threshold": str(
                getattr(clusterer, "min_similarity_threshold", "NA")
            ),
            "max_layers": str(getattr(clusterer, "max_layers", "NA")),
            "n_label_prop_iter": str(getattr(clusterer, "n_label_prop_iter", "NA")),
        }

        write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            n_iter,
            actual_iterations,
            num_dimensions,
            num_vectors,
            num_centroids,
            construction_time_ms,
            threads,
            final_objective,
            config_dict,
            results_knn_10,
            results_knn_100,
        )
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
        print("Skipping CSV output (recall computation requires ground truth)")