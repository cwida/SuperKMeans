import os
import sys

# Add parent directory to path for bench_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, Timer, write_results_to_csv,
                         N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

N_EPOCHS = 2
SEED = 42

if __name__ == "__main__":
    algorithm = "faiss_emeans"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mxbai"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "end_to_end"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    threads = faiss.omp_get_max_threads()
    spherical = dataset in ANGULAR_DATASETS

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_epochs={N_EPOCHS}")
    print(f"faiss {faiss.__version__} [{faiss.get_compile_options()}]")

    filename = get_data_path(dataset)
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)
    if spherical:
        print(f"Using spherical e-means for dataset: {dataset}")

    emeans = NumpyEmeans(num_dimensions, num_centroids, nepochs=N_EPOCHS, seed=SEED, spherical=spherical)

    with Timer() as timer:
        centroids, _ = emeans.train(data)
    construction_time_ms = timer.get_milliseconds()
    actual_iterations = len(emeans.iteration_stats)
    final_objective = emeans.iteration_stats[-1]["obj"]

    print(f"\nTraining completed in {construction_time_ms:.2f} ms")
    print(f"Actual epochs: {actual_iterations} (requested: {N_EPOCHS})")
    print(f"Final objective: {final_objective}")

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
        _, assignments = emeans.assign(data)

        results_knn_10 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 10)
        print_recall_results(results_knn_10, 10)
        results_knn_100 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 100)
        print_recall_results(results_knn_100, 100)

        config_dict = {
            "nepochs": str(emeans.nepochs),
            "seed": str(emeans.seed),
            "spherical": str(emeans.spherical).lower(),
            "batch_size": str(emeans.recipe["batch_size"]),
            "alpha0": str(emeans.recipe["alpha0"]),
            "faiss_version": faiss.__version__
        }

        write_results_to_csv(
            experiment_name, algorithm, dataset, N_EPOCHS, actual_iterations,
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
