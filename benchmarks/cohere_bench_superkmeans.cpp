#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

// #ifndef SMALLER_TEST
// #define SMALLER_TEST = true
// #endif

#include <fstream>
#include <iostream>
#include <omp.h>
#include <unordered_map>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/hierarchical_superkmeans.h"

int main() {
    const std::string experiment_name = "blog_post_1";
    const std::string algorithm = "hierarchical_superkmeans";
#ifdef SMALLER_TEST
    const std::string dataset = "openai";
    const size_t n = 999000;
    const size_t n_queries = 1000;
    const size_t d = 1536;
    const size_t n_clusters = 10000;
    const std::string filename = "./data/data_openai.bin";
    const std::string filename_queries = "./data/data_openai_test.bin";
#else
    const std::string dataset = "cohere";
    const size_t n = 10000000;
    const size_t n_queries = 1000;
    const size_t d = 1024;
    const size_t n_clusters = 40000;
    const std::string filename = "./data/data_cohere.bin";
    const std::string filename_queries = "./data/data_cohere_test.bin";
#endif

    std::vector<float> data;
    std::vector<float> queries;
    data.reserve(n * d);
    queries.reserve(n_queries * d);

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

    skmeans::HierarchicalSuperKMeansConfig config;
    config.verbose = true;
    config.sampling_fraction = 1.0;
    config.use_blas_only = false;
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 5;
    config.iters_refinement = 0;

    std::cout << "=== Running SuperKMeans on Cohere 10M dataset ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " sampling_fraction=" << config.sampling_fraction
              << "\n";

    auto kmeans_state =
        skmeans::HierarchicalSuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config
        );

    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans_state.Train(data.data(), n);
    timer.Toc();

    double construction_time_ms = timer.GetMilliseconds();
    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;

    timer.Reset();
    timer.Tic();
    auto assignments_brute = kmeans_state.Assign(data.data(), centroids.data(), n, n_clusters);
    timer.Toc();
    double assignment_time_ms_brute = timer.GetMilliseconds();
    std::cout << "Brute Force Assignment completed in " << assignment_time_ms_brute << " ms"
              << std::endl;

    timer.Reset();
    timer.Tic();
    auto assignments =
        kmeans_state.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    timer.Toc();
    double assignment_time_ms = timer.GetMilliseconds();
    std::cout << "Fast Assignment completed in " << assignment_time_ms << " ms" << std::endl;

    auto balance_stats = skmeans::HierarchicalSuperKMeans<
        skmeans::Quantization::f32,
        skmeans::DistanceFunction::l2>::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance_stats.print();

    // Compute recall if ground truth file exists
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << filename_queries << std::endl;

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);

        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        std::unordered_map<std::string, std::string> config_map;
        config_map["sampling_fraction"] = std::to_string(config.sampling_fraction);
        config_map["iters_mesoclustering"] = std::to_string(config.iters_mesoclustering);
        config_map["iters_fineclustering"] = std::to_string(config.iters_fineclustering);
        config_map["iters_refinement"] = std::to_string(config.iters_refinement);
        config_map["use_blas_only"] = config.use_blas_only ? "true" : "false";
        config_map["verbose"] = config.verbose ? "true" : "false";
        config_map["training_time"] = std::to_string(construction_time_ms);
        config_map["assignment_time"] = std::to_string(assignment_time_ms);
        config_map["assignment_time_naive"] = std::to_string(assignment_time_ms_brute);

        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            config.iters_mesoclustering + config.iters_fineclustering + config.iters_refinement,
            config.iters_mesoclustering + config.iters_fineclustering + config.iters_refinement,
            static_cast<int>(d),
            n,
            static_cast<int>(n_clusters),
            construction_time_ms + assignment_time_ms,
            omp_get_max_threads(),
            0.0,
            config_map,
            results_knn_10,
            results_knn_100,
            balance_stats.to_json()
        );
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file_check.good()) {
            std::cout << "Queries file not found: " << filename_queries << std::endl;
        }
        std::cout << "Skipping CSV output (recall computation requires ground truth)" << std::endl;
    }
}
