"""Utility functions for K-means benchmarking."""

import csv
import json
import math
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
import h5py


def get_default_n_clusters(n):
    """Compute the default number of clusters for a dataset.

    Uses the heuristic: n_clusters = max(1, sqrt(n) * 4)

    Args:
        n: Number of data points

    Returns:
        Default number of clusters
    """
    return max(1, int(math.sqrt(n) * 4))

# Path constants for benchmark data
BENCHMARKS_ROOT = Path(__file__).parent
DATA_DIR = BENCHMARKS_ROOT / 'data'
GROUND_TRUTH_DIR = BENCHMARKS_ROOT / 'ground_truth'

def get_data_path(dataset):
    return DATA_DIR / f'data_{dataset}.bin'

def get_query_path(dataset):
    return DATA_DIR / f'data_{dataset}_test.bin'

def get_ground_truth_path(dataset):
    return GROUND_TRUTH_DIR / f'{dataset}.json'


def l2_normalize(x, eps=1e-12):
    """Normalize vectors to unit length using L2 norm.

    Args:
        x: Input array of shape (n, d)
        eps: Small epsilon to avoid division by zero

    Returns:
        L2-normalized array of same shape
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def read_hdf5_data(raw_data_path, dataset_hdf5_name):
    """Read train and test data from HDF5 file.

    Args:
        raw_data_path: Path to directory containing HDF5 files
        dataset_hdf5_name: Name of HDF5 file (without .hdf5 extension)

    Returns:
        Tuple of (train, test) numpy arrays as float32
    """
    hdf5_file_name = os.path.join(raw_data_path, dataset_hdf5_name + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)


# Dataset configurations: name -> (num_vectors, num_dimensions)
DATASET_PARAMS = {
    "fmnist": (60_000, 784),
    "yi": (187_843, 128),
    "llama": (256_921, 128),
    "sift": (1_000_000, 128),
    "yahoo": (677_305, 384),
    "yandex": (1_000_000, 200),
    "glove200": (1_183_514, 200),
    "clip": (1_281_167, 512),
    "mxbai": (769_382, 1024),
    "wiki": (260_372, 3072),
    "contriever": (990_000, 768),
    "gist": (1_000_000, 960),
    "openai": (999_000, 1536),
    "arxiv": (2_253_000, 768),
    "cohere": (10_000_000, 1024),
    "cohere50m": (50_000_000, 1024)
}

# Datasets that should use angular/spherical k-means
ANGULAR_DATASETS = [
    "yandex", "glove200", "glove100", "glove50", "llama", "yi", "contriever"
]

# Mapping from dataset identifier to HDF5 filename (without .hdf5 extension)
DATASET_HDF5_NAMES = {
    "glove200": "glove-200-angular",
    "clip": "imagenet-clip-512-normalized",
    "yahoo": "yahoo-minilm-384-normalized",
    "llama": "llama-128-ip",
    "yi": "yi-128-ip",
    "yandex": "yandex-200-cosine",
    "mxbai": "agnews-mxbai-1024-euclidean",
    "openai": "openai-1536-angular",
    "arxiv": "instructorxl-arxiv-768",
    "sift": "sift-128-euclidean",
    "fmnist": "fashion-mnist-784-euclidean",
    "gist": "gist-960-euclidean",
    "contriever": "contriever-768",
    "wiki": "simplewiki-openai-3072-normalized",
}

# Standard exploration fractions for recall computation
EXPLORE_FRACTIONS = [
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275,
    0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0475, 0.0500,
    0.1
]

# Fixed absolute centroid counts to always evaluate (nprobe=1, nprobe=2)
ABSOLUTE_EXPLORE_COUNTS = [1, 2]

# KNN values to test
KNN_VALUES = [10, 100]

# Benchmark configuration
MAX_ITERS = 25
N_QUERIES = 1000

# Early termination benchmark configuration
RECALL_TOL_VALUES = [0.03, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001]
FAISS_EARLY_TERM_ITERS = [10]
SCIKIT_EARLY_TERM_MAX_ITERS = 300
SCIKIT_EARLY_TERM_TOL = 1e-8

# Sampling fraction values for sampling experiment
SAMPLING_FRACTION_VALUES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

# Iteration values for pareto experiment (grid search)
PARETO_ITERS_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# n_clusters values for varying_k experiment
VARYING_K_VALUES = [100, 1000, 10000, 100000]

def load_ground_truth(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def compute_recall(gt_dict, assignments, queries, centroids, num_centroids, knn):
    """Compute recall@K for different exploration fractions.

    Args:
        gt_dict: Ground truth dictionary (query_idx -> vector_ids)
        assignments: Cluster assignments for all data vectors
        queries: Query vectors (n_queries, d)
        centroids: Centroid vectors (num_centroids, d)
        num_centroids: Number of centroids
        knn: Number of ground truth neighbors to consider

    Returns:
        List of tuples (centroids_to_explore, explore_fraction, recall_mean, recall_std, avg_vectors_to_visit)
    """
    n_queries = queries.shape[0]

    # Count cluster sizes to compute vectors to visit
    cluster_sizes = np.bincount(assignments, minlength=num_centroids)

    # Compute distances from queries to centroids
    # Using L2 distance: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
    query_norms = np.sum(queries ** 2, axis=1, keepdims=True)  # (n_queries, 1)
    centroid_norms = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, num_centroids)
    dot_products = queries @ centroids.T  # (n_queries, num_centroids)
    distances = query_norms + centroid_norms - 2 * dot_products  # (n_queries, num_centroids)

    # Build list of (centroids_to_explore, explore_frac) configs:
    # absolute counts first (nprobe=1, nprobe=2), then fraction-based
    explore_configs = []
    for count in ABSOLUTE_EXPLORE_COUNTS:
        c = min(count, num_centroids)
        explore_configs.append((c, c / num_centroids))
    for frac in EXPLORE_FRACTIONS:
        explore_configs.append((max(1, int(num_centroids * frac)), frac))

    results = []
    for centroids_to_explore, explore_frac in explore_configs:
        # Find top-N nearest centroids for each query
        top_centroid_indices = np.argsort(distances, axis=1)[:, :centroids_to_explore]

        # Compute recall and vectors to visit for each query
        query_recalls = []
        total_vectors_to_visit = 0
        for query_idx in range(n_queries):
            query_key = str(query_idx)
            if query_key not in gt_dict:
                continue

            gt_vector_ids = gt_dict[query_key][:knn]  # List of ground truth vector IDs
            top_centroids_list = top_centroid_indices[query_idx]  # Top N centroids for this query
            top_centroids = set(top_centroids_list)

            # Count vectors to visit for this query
            vectors_to_visit = np.sum(cluster_sizes[top_centroids_list])
            total_vectors_to_visit += vectors_to_visit

            # Count how many ground truth vectors have their assigned centroid in the top N
            found = 0
            for vector_id in gt_vector_ids:
                assigned_centroid = assignments[vector_id]  # Which centroid this vector belongs to
                if assigned_centroid in top_centroids:
                    found += 1

            query_recall = found / len(gt_vector_ids)
            query_recalls.append(query_recall)

        # Compute mean and standard deviation
        average_recall = np.mean(query_recalls)
        std_recall = np.std(query_recalls, ddof=1) if len(query_recalls) > 1 else 0.0
        avg_vectors_to_visit = total_vectors_to_visit / n_queries
        results.append((centroids_to_explore, explore_frac, average_recall, std_recall, avg_vectors_to_visit))

    return results


def print_recall_results(results, knn):
    print(f"\n--- Recall@{knn} ---")
    for centroids_to_explore, explore_frac, recall, std_recall, avg_vectors in results:
        print(
            f"Recall@{centroids_to_explore:4d} ({explore_frac * 100:5.2f}% centroids, {avg_vectors:8.0f} avg vectors): {recall:.4f} ± {std_recall:.4f}")


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

    def get_milliseconds(self):
        return self.elapsed_ms


def write_results_to_csv(
        experiment_name,
        algorithm,
        dataset,
        n_iters,
        actual_iterations,
        dimensionality,
        data_size,
        n_clusters,
        construction_time_ms,
        threads,
        final_objective,
        config_dict,
        results_knn_10,
        results_knn_100,
        balance_stats_json="",
        iteration_stats_json=""
):
    """Write results to CSV file.

    Args:
        experiment_name: Name of the experiment (e.g., "end_to_end")
        algorithm: Name of the algorithm (e.g., "scikit")
        dataset: Dataset name
        n_iters: Number of iterations (max requested)
        actual_iterations: Actual iterations performed (may be less if early termination)
        dimensionality: Data dimensionality
        data_size: Number of data points
        n_clusters: Number of clusters
        construction_time_ms: Construction time in milliseconds
        threads: Number of threads used
        final_objective: Final k-means objective value
        config_dict: Dictionary with algorithm-specific configuration (will be serialized to JSON)
        results_knn_10: Results for KNN=10
        results_knn_100: Results for KNN=100
        balance_stats_json: JSON string with cluster balance statistics (including cluster_sizes)
    """
    arch = os.environ.get('SKM_ARCH', 'default')
    benchmarks_dir = Path(__file__).parent
    results_dir = benchmarks_dir / 'results' / arch
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f'{experiment_name}.csv'

    file_exists = csv_path.exists()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    has_recall_data = len(results_knn_10) > 0 or len(results_knn_100) > 0
    header = ['timestamp', 'algorithm', 'dataset', 'n_iters', 'actual_iterations', 'dimensionality',
              'data_size', 'n_clusters', 'construction_time_ms', 'threads', 'final_objective']
    if has_recall_data:
        for knn in KNN_VALUES:
            for count in ABSOLUTE_EXPLORE_COUNTS:
                header.append(f'recall@{knn}@nprobe{count}')
                header.append(f'recall_std@{knn}@nprobe{count}')
                header.append(f'centroids_explored@{knn}@nprobe{count}')
                header.append(f'vectors_explored@{knn}@nprobe{count}')
            for explore_frac in EXPLORE_FRACTIONS:
                header.append(f'recall@{knn}@{explore_frac * 100:.2f}')
                header.append(f'recall_std@{knn}@{explore_frac * 100:.2f}')
                header.append(f'centroids_explored@{knn}@{explore_frac * 100:.2f}')
                header.append(f'vectors_explored@{knn}@{explore_frac * 100:.2f}')
    header.append('balance_stats')
    header.append('iteration_stats')
    header.append('config')
    row = [
        timestamp,
        algorithm,
        dataset,
        n_iters,
        actual_iterations,
        dimensionality,
        data_size,
        n_clusters,
        f'{construction_time_ms:.2f}',
        threads,
        f'{final_objective:.6f}'
    ]

    if has_recall_data:
        # Add KNN=10 results
        for centroids_to_explore, explore_frac, recall, std_recall, avg_vectors in results_knn_10:
            row.append(f'{recall:.6f}')
            row.append(f'{std_recall:.6f}')
            row.append(str(centroids_to_explore))
            row.append(f'{avg_vectors:.2f}')
        # Add KNN=100 results
        for centroids_to_explore, explore_frac, recall, std_recall, avg_vectors in results_knn_100:
            row.append(f'{recall:.6f}')
            row.append(f'{std_recall:.6f}')
            row.append(str(centroids_to_explore))
            row.append(f'{avg_vectors:.2f}')
    row.append(balance_stats_json)
    row.append(iteration_stats_json)
    config_json = json.dumps(config_dict)
    row.append(config_json)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Results written to: {csv_path}")


def _build_recall_stats_dict(results_knn_10, results_knn_100):
    """Build a flat dict of recall stats from result tuples.

    Returns dict with keys like:
        recall@10@nprobe1, recall_std@10@nprobe1, centroids_explored@10@nprobe1, vectors_explored@10@nprobe1,
        recall@10@0.10, recall_std@10@0.10, ...
    """
    stats = {}
    for knn, results in [(10, results_knn_10), (100, results_knn_100)]:
        if not results:
            continue
        abs_idx = 0
        for centroids_to_explore, explore_frac, recall, std_recall, avg_vectors in results:
            if abs_idx < len(ABSOLUTE_EXPLORE_COUNTS):
                count = ABSOLUTE_EXPLORE_COUNTS[abs_idx]
                suffix = f"nprobe{count}"
                abs_idx += 1
            else:
                suffix = f"{explore_frac * 100:.2f}"
            stats[f"recall@{knn}@{suffix}"] = round(recall, 6)
            stats[f"recall_std@{knn}@{suffix}"] = round(std_recall, 6)
            stats[f"centroids_explored@{knn}@{suffix}"] = centroids_to_explore
            stats[f"vectors_explored@{knn}@{suffix}"] = round(avg_vectors, 2)
    return stats


def write_results_to_csv_v2(
        experiment_name,
        algorithm,
        dataset,
        n_iters,
        actual_iterations,
        dimensionality,
        data_size,
        n_clusters,
        construction_time_ms,
        threads,
        final_objective,
        config_dict,
        assign_results_knn_10,
        assign_results_knn_100,
        quantized_assign_results_knn_10=None,
        quantized_assign_results_knn_100=None,
        balance_stats_json="",
        iteration_stats_json=""
):
    """Write results to CSV with recall stats packed into a single JSON column.

    Same core columns as write_results_to_csv, but instead of one column per
    recall measurement, all recall/std/centroids/vectors stats are stored in a
    single ``clustering_quality_stats`` JSON column with the structure::

        {
            "assign": {
                "recall@10@nprobe1": 0.50,
                "recall_std@10@nprobe1": 0.26,
                "centroids_explored@10@nprobe1": 1,
                "vectors_explored@10@nprobe1": 78.07,
                ...
            },
            "quantized_assign": { ... }   // only present when provided
        }

    Args:
        experiment_name: Name of the experiment (used as CSV filename)
        algorithm: Algorithm name
        dataset: Dataset name
        n_iters: Requested iterations
        actual_iterations: Actual iterations performed
        dimensionality: Data dimensionality
        data_size: Number of data points
        n_clusters: Number of clusters
        construction_time_ms: Construction time in milliseconds
        threads: Number of threads used
        final_objective: Final k-means objective value
        config_dict: Algorithm-specific config (serialized to JSON)
        assign_results_knn_10: Recall results for KNN=10 (float assignments)
        assign_results_knn_100: Recall results for KNN=100 (float assignments)
        quantized_assign_results_knn_10: Recall results for KNN=10 (quantized assignments), or None
        quantized_assign_results_knn_100: Recall results for KNN=100 (quantized assignments), or None
        balance_stats_json: JSON string with cluster balance statistics
        iteration_stats_json: JSON string with per-iteration statistics
    """
    arch = os.environ.get('SKM_ARCH', 'default')
    benchmarks_dir = Path(__file__).parent
    results_dir = benchmarks_dir / 'results' / arch
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f'{experiment_name}.csv'

    file_exists = csv_path.exists()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    header = [
        'timestamp', 'algorithm', 'dataset', 'n_iters', 'actual_iterations',
        'dimensionality', 'data_size', 'n_clusters', 'construction_time_ms',
        'threads', 'final_objective',
        'clustering_quality_stats',
        'balance_stats', 'iteration_stats', 'config'
    ]

    # Build the clustering_quality_stats JSON
    quality_stats = {}
    assign_stats = _build_recall_stats_dict(assign_results_knn_10, assign_results_knn_100)
    if assign_stats:
        quality_stats["assign"] = assign_stats

    if quantized_assign_results_knn_10 is not None or quantized_assign_results_knn_100 is not None:
        qa_stats = _build_recall_stats_dict(
            quantized_assign_results_knn_10 or [],
            quantized_assign_results_knn_100 or []
        )
        if qa_stats:
            quality_stats["quantized_assign"] = qa_stats

    row = [
        timestamp,
        algorithm,
        dataset,
        n_iters,
        actual_iterations,
        dimensionality,
        data_size,
        n_clusters,
        f'{construction_time_ms:.2f}',
        threads,
        f'{final_objective:.6f}',
        json.dumps(quality_stats) if quality_stats else "",
        balance_stats_json,
        iteration_stats_json,
        json.dumps(config_dict)
    ]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Results written to: {csv_path}")
