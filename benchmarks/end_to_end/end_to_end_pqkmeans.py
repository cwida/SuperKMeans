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

import pqkmeans
import numpy as np
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    algorithm = "pqkmeans"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove200"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "end_to_end"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    n_iter = 10

    # PQ parameters
    num_subdim = 8                                      # Number of subspaces
    Ks = 256                                            # Codebook size per subspace (8-bit PQ codes)
    encoder_train_size = min(num_vectors, 100_000)      # Vectors used for encoder training

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")
    print(f"PQ params: num_subdim={num_subdim}, Ks={Ks}, encoder_train_size={encoder_train_size}")

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
              f"but pqkmeans does not support this. Results may be suboptimal.")

    # pqkmeans requires float64
    data_f64 = data.astype(np.float64)

    with Timer() as timer:
        # Step 1: Train PQ encoder on a subset
        encoder = pqkmeans.encoder.PQEncoder(num_subdim=num_subdim, Ks=Ks)
        encoder.fit(data_f64[:encoder_train_size])

        # Step 2: Encode all data to PQ codes
        X_pqcode = encoder.transform(data_f64)

        # Step 3: Run PQ k-means clustering
        kmeans = pqkmeans.clustering.PQKMeans(
            encoder=encoder, k=num_centroids, iteration=n_iter, verbose=False
        )
        assignments = kmeans.fit_predict(X_pqcode)

    construction_time_ms = timer.get_milliseconds()
    assignments = np.array(assignments, dtype=np.int64)

    # Reconstruct approximate float centroids from PQ centroid codes
    centroid_pqcodes = np.array(kmeans.cluster_centers_, dtype=encoder.code_dtype)
    centroids = encoder.inverse_transform(centroid_pqcodes).astype(np.float32)

    # Reconstruct exact f32 centroids as mean of assigned vectors
    # centroids = np.zeros((num_centroids, num_dimensions), dtype=np.float32)
    # for c in range(num_centroids):
    #     mask = assignments == c
    #     if np.any(mask):
    #         centroids[c] = data[mask].mean(axis=0)

    # pqkmeans does not report actual iterations or inertia — compute manually
    actual_iterations = n_iter
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
            "num_subdim": str(num_subdim),
            "Ks": str(Ks),
            "encoder_train_size": str(encoder_train_size),
            "iteration": str(n_iter),
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
