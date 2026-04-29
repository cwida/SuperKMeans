#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/superkmeans.h"

/**
 * @brief Compute component-wise median centroids from cluster assignments.
 *
 * For each cluster, collects all assigned data points and computes the
 * median of each dimension independently.
 */
static void ComputeMedianCentroids(
    const float* data,
    const uint32_t* assignments,
    float* out_centroids,
    size_t n,
    size_t d,
    size_t n_clusters
) {
    // Build per-cluster point indices
    std::vector<std::vector<uint32_t>> cluster_members(n_clusters);
    for (size_t i = 0; i < n; ++i) {
        cluster_members[assignments[i]].push_back(static_cast<uint32_t>(i));
    }

    #pragma omp parallel
    {
        std::vector<float> dim_values; // reused across clusters/dims

        #pragma omp for schedule(dynamic)
        for (size_t c = 0; c < n_clusters; ++c) {
            const auto& members = cluster_members[c];
            if (members.empty()) {
                std::memset(out_centroids + c * d, 0, d * sizeof(float));
                continue;
            }
            dim_values.resize(members.size());
            for (size_t j = 0; j < d; ++j) {
                for (size_t m = 0; m < members.size(); ++m) {
                    dim_values[m] = data[members[m] * d + j];
                }
                size_t mid = members.size() / 2;
                std::nth_element(dim_values.begin(), dim_values.begin() + mid, dim_values.end());
                if (members.size() % 2 == 1) {
                    out_centroids[c * d + j] = dim_values[mid];
                } else {
                    // Even count: average of two middle elements
                    float upper = dim_values[mid];
                    float lower = *std::max_element(dim_values.begin(), dim_values.begin() + mid);
                    out_centroids[c * d + j] = (lower + upper) * 0.5f;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    const std::string algorithm = "superkmeans_ivn";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n) * 1;
    int n_iters = 10;
    float sampling_fraction = 1.0f;
    std::string filename = bench_utils::get_data_path(dataset);
    std::string filename_queries = bench_utils::get_query_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";

    // ── Load data and queries ──
    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) { std::cerr << "Failed to open " << filename << std::endl; return 1; }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(filename_queries, std::ios::binary);
        if (!f) { std::cerr << "Failed to open " << filename_queries << std::endl; return 1; }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    // ── Train ──
    using SKM = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>;

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.verbose_detail = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = false;
    config.tol = 1e-3f;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans = SKM(n_clusters, d, config);

    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans.iteration_stats.size());

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;

    // ── Assign all data points ──
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    double wcss = SKM::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d);
    std::cout << "WCSS (original centroids): " << std::fixed << std::setprecision(2) << wcss
              << std::endl;

    auto balance = SKM::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance.print();

    // ── Find Inverted Voronoi Neighbors (IVN): closest point to each centroid ──
    std::cout << "\n--- Computing IVN centroids (FindKNearestNeighbors) ---" << std::endl;

    timer.Reset();
    timer.Tic();

    using f32_batch = skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;

    constexpr size_t topk = 5;

    // Compute norms for centroids (queries) and data (references)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        centroids_mat(centroids.data(), n_clusters, d);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_mat(data.data(), n, d);

    std::vector<float> centroid_norms(n_clusters);
    std::vector<float> data_norms(n);
    Eigen::Map<Eigen::VectorXf>(centroid_norms.data(), n_clusters) =
        centroids_mat.rowwise().squaredNorm();
    Eigen::Map<Eigen::VectorXf>(data_norms.data(), n) =
        data_mat.rowwise().squaredNorm();

    // out_knn[c * topk + k] = index of k-th nearest data point to centroid c
    std::vector<uint32_t> topk_indices(n_clusters * topk);
    std::vector<float> topk_distances(n_clusters * topk);
    std::unique_ptr<float[]> tmp_buf(
        new float[skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE]
    );

    f32_batch::FindKNearestNeighbors(
        centroids.data(),    // x = centroids (queries)
        data.data(),         // y = data points (references)
        n_clusters, n, d,
        centroid_norms.data(),
        data_norms.data(),
        topk,
        topk_indices.data(),
        topk_distances.data(),
        tmp_buf.get()
    );

    timer.Toc();
    std::cout << "IVN computation: " << timer.GetMilliseconds() << " ms" << std::endl;

    // Print top-5 for the first 3 clusters
    for (size_t c = 0; c < std::min<size_t>(3, n_clusters); ++c) {
        std::cout << "  Cluster " << c << " (size=" << balance.mean << " avg): top-" << topk
                  << " nearest data points:" << std::endl;
        for (size_t k = 0; k < topk; ++k) {
            uint32_t idx = topk_indices[c * topk + k];
            float dist = topk_distances[c * topk + k];
            std::cout << "    [" << k << "] point " << idx
                      << ", L2²=" << std::fixed << std::setprecision(6) << dist << std::endl;
        }
    }

    // Build IVN centroids: use the top-1 (closest) data point per centroid
    std::vector<float> ivn_centroids(n_clusters * d);
    for (size_t c = 0; c < n_clusters; ++c) {
        uint32_t nearest_idx = topk_indices[c * topk]; // top-1
        std::memcpy(
            ivn_centroids.data() + c * d,
            data.data() + nearest_idx * d,
            d * sizeof(float)
        );
    }

    // Print IVN distance stats
    float avg_ivn_dist = 0.0f;
    float max_ivn_dist = 0.0f;
    size_t empty_clusters = 0;
    for (size_t c = 0; c < n_clusters; ++c) {
        float dist = topk_distances[c * topk];
        if (dist >= std::numeric_limits<float>::max()) {
            empty_clusters++;
            continue;
        }
        avg_ivn_dist += dist;
        max_ivn_dist = std::max(max_ivn_dist, dist);
    }
    if (n_clusters > empty_clusters) {
        avg_ivn_dist /= static_cast<float>(n_clusters - empty_clusters);
    }
    std::cout << "Avg IVN L2² distance: " << avg_ivn_dist << std::endl;
    std::cout << "Max IVN L2² distance: " << max_ivn_dist << std::endl;
    if (empty_clusters > 0) {
        std::cout << "Warning: " << empty_clusters << " empty clusters" << std::endl;
    }

    // ── Re-assign with IVN centroids ──
    auto ivn_assignments = kmeans.Assign(data.data(), ivn_centroids.data(), n, n_clusters);

    double wcss_ivn = SKM::ComputeWCSS(data.data(), ivn_centroids.data(), ivn_assignments.data(), n, d);
    std::cout << "\nWCSS (IVN centroids): " << std::fixed << std::setprecision(2) << wcss_ivn
              << std::endl;

    auto ivn_balance = SKM::GetClustersBalanceStats(ivn_assignments.data(), n, n_clusters);
    ivn_balance.print();

    // ── Compute median centroids ──
    std::cout << "\n--- Computing median centroids ---" << std::endl;

    timer.Reset();
    timer.Tic();

    std::vector<float> median_centroids(n_clusters * d);
    ComputeMedianCentroids(
        data.data(), assignments.data(), median_centroids.data(), n, d, n_clusters
    );

    timer.Toc();
    std::cout << "Median computation: " << timer.GetMilliseconds() << " ms" << std::endl;

    auto median_assignments = kmeans.Assign(data.data(), median_centroids.data(), n, n_clusters);

    double wcss_median = SKM::ComputeWCSS(
        data.data(), median_centroids.data(), median_assignments.data(), n, d
    );
    std::cout << "WCSS (median centroids): " << std::fixed << std::setprecision(2) << wcss_median
              << std::endl;

    auto median_balance = SKM::GetClustersBalanceStats(median_assignments.data(), n, n_clusters);
    median_balance.print();

    // ── Recall comparison ──
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "\n--- Recall@10 comparison ---" << std::endl;
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        // Recall with original centroids
        std::cout << "\n  [Original centroids]" << std::endl;
        auto recall_original = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(),
            n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(recall_original, 10);

        // Recall with IVN centroids
        std::cout << "\n  [IVN centroids]" << std::endl;
        auto recall_ivn = bench_utils::compute_recall(
            gt_map, ivn_assignments, queries.data(), ivn_centroids.data(),
            n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(recall_ivn, 10);

        // Recall with median centroids
        std::cout << "\n  [Median centroids]" << std::endl;
        auto recall_median = bench_utils::compute_recall(
            gt_map, median_assignments, queries.data(), median_centroids.data(),
            n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(recall_median, 10);
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file_check.good()) {
            std::cout << "Queries file not found: " << filename_queries << std::endl;
        }
    }
}
