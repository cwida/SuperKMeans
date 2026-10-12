#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

template <skmeans::Quantization Q>
int RunBenchmark(const std::string& dataset, const bool blas_only) {
    const std::string algorithm = std::string("superkmeans_") + skmeans::QuantizationName(Q);

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    int n_iters = 10;
    float sampling_fraction = 1.0f;
    std::string filename = bench_utils::GetDataPath(dataset);
    std::string filename_queries = bench_utils::GetQueryPath(dataset);
    const size_t THREADS = skmeans::ResolveNumThreads(0);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<float> data;
    std::vector<float> queries;
    try {
        data.reserve(n * d);
        queries.reserve(n_queries * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << filename_queries << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    file_queries.close();

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.tol = 1e-3f;
    config.use_blas_only = blas_only;
    config.quantized_centroid_update = true;
    if (blas_only) {
        std::cout << "BLAS-only mode (no pruning)" << std::endl;
    }

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans = skmeans::SuperKMeans<Q>(n_clusters, d, config);
    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans.iteration_stats.size());
    double final_objective = kmeans.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective (quantized): " << final_objective << std::endl;

    // Compute assignments with Assign() and QuantizedAssign()
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    auto q_assignments = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    using SKM = skmeans::SuperKMeans<Q>;
    double wcss_assign = SKM::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d);
    double wcss_q_assign =
        SKM::ComputeWCSS(data.data(), centroids.data(), q_assignments.data(), n, d);
    std::cout << "WCSS (f32, Assign):          " << std::fixed << std::setprecision(2)
              << wcss_assign << std::endl;
    std::cout << "WCSS (f32, QuantizedAssign): " << std::fixed << std::setprecision(2)
              << wcss_q_assign << std::endl;

    std::cout << "\n--- Assign() cluster balance ---" << std::endl;
    auto balance_stats = SKM::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance_stats.print();

    std::cout << "--- QuantizedAssign() cluster balance ---" << std::endl;
    auto q_balance_stats = SKM::GetClustersBalanceStats(q_assignments.data(), n, n_clusters);
    q_balance_stats.print();

    // Compute recall if ground truth file exists
    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << filename_queries << std::endl;

        auto gt_map = bench_utils::ParseGroundTruthJson(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        // Recall with Assign()
        std::cout << "\n  [Assign()]" << std::endl;
        auto results_knn_10 = bench_utils::ComputeRecall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::PrintRecallResults(results_knn_10, 10);

        auto results_knn_100 = bench_utils::ComputeRecall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::PrintRecallResults(results_knn_100, 100);

        // Recall with QuantizedAssign()
        std::cout << "\n  [QuantizedAssign()]" << std::endl;
        auto q_results_knn_10 = bench_utils::ComputeRecall(
            gt_map, q_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::PrintRecallResults(q_results_knn_10, 10);

        auto q_results_knn_100 = bench_utils::ComputeRecall(
            gt_map, q_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::PrintRecallResults(q_results_knn_100, 100);
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file_check.good()) {
            std::cout << "Queries file not found: " << filename_queries << std::endl;
        }
        std::cout << "Skipping CSV output (recall computation requires ground truth)" << std::endl;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");
    std::string quantizer = (argc > 2) ? std::string(argv[2]) : std::string("sq8");
    bool blas_only = !(argc > 3 && std::string(argv[3]) == "pruning");

    if (quantizer == "sq8") {
        return RunBenchmark<skmeans::Quantization::sq8>(dataset, blas_only);
    }
    if (quantizer == "lvq4") {
        return RunBenchmark<skmeans::Quantization::lvq4>(dataset, blas_only);
    }
    if (quantizer == "rabitq") {
        return RunBenchmark<skmeans::Quantization::rabitq>(dataset, blas_only);
    }
    std::cerr << "Unknown quantizer '" << quantizer << "' (use sq8|lvq4|rabitq)\n";
    return 1;
}
