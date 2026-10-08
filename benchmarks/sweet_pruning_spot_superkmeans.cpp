#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    const std::string experiment_name = "sweet_pruning_spot";
    const std::string algorithm = "superkmeans";
    const int n_iters = 10;
    const float sampling_fraction = 1.0f;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t THREADS = skmeans::ResolveNumThreads(0);
    const float PRUNING_PCT_WINDOW_SIZE = 0.005f;

    std::cout << "=== Running sweet_pruning_spot benchmark ===" << std::endl;
    std::cout << "Fixed parameters: iters=" << n_iters
              << ", sampling_fraction=" << sampling_fraction << std::endl;
    std::cout << "Threads: " << THREADS << std::endl;
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    // Generate pruning parameter pairs: [min, max] from [0.0, 0.005] to [0.10, 0.105]
    std::vector<std::pair<float, float>> pruning_params;
    pruning_params.push_back({0.03f, 0.05f}); // First dummy iteration
    for (float min_pct = 0.0f; min_pct <= 0.10f; min_pct += PRUNING_PCT_WINDOW_SIZE) {
        float max_pct = min_pct + PRUNING_PCT_WINDOW_SIZE;
        pruning_params.push_back({min_pct, max_pct});
    }

    std::vector<float> adjustment_factors = {0.10f, 0.15f, 0.20f, 0.25f, 0.30f};
    size_t total_combinations = pruning_params.size() * adjustment_factors.size();
    std::cout << "Trying " << total_combinations << " parameter combinations "
              << "(" << pruning_params.size() << " pruning pairs × " << adjustment_factors.size()
              << " adjustment factors)" << std::endl;

    for (const auto& dataset_entry : bench_utils::DATASET_PARAMS) {
        const std::string& dataset = dataset_entry.first;
        const size_t n = dataset_entry.second.first;
        const size_t d = dataset_entry.second.second;
        const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
        std::string filename = bench_utils::GetDataPath(dataset);
        std::string filename_queries = bench_utils::GetQueryPath(dataset);
        std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;
        std::cout << "n_clusters=" << n_clusters << std::endl;
        std::cout << "========================================" << std::endl;

        std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
        std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;
        try {
            data.reserve(n * d);
            queries.reserve(n_queries * d);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                      << "\n";
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open data file: " << filename << std::endl;
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
        file.close();

        std::ifstream file_queries(filename_queries, std::ios::binary);
        if (!file_queries) {
            std::cerr << "Failed to open queries file: " << filename_queries << std::endl;
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }
        file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
        file_queries.close();

        std::ifstream gt_file(gt_filename);
        if (!gt_file.good()) {
            std::cerr << "Ground truth file not found: " << gt_filename << std::endl;
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }
        gt_file.close();
        auto gt_map = bench_utils::ParseGroundTruthJson(gt_filename);
        std::cout << "Loaded ground truth with " << gt_map.size() << " queries" << std::endl;

        for (float adjustment_factor : adjustment_factors) {
            std::cout << "\n=== Testing adjustment_factor: " << adjustment_factor
                      << " ===" << std::endl;
            for (const auto& params : pruning_params) {
                float min_not_pruned_pct = params.first;
                float max_not_pruned_pct = params.second;

                std::cout << "\n--- Testing pruning params: [" << min_not_pruned_pct << ", "
                          << max_not_pruned_pct << "], adjustment: " << adjustment_factor << " ---"
                          << std::endl;

                skmeans::SuperKMeansConfig config;
                config.iters = n_iters;
                config.verbose = true;
                config.n_threads = THREADS;
                config.objective_k = 100;
                config.ann_explore_fraction = 0.01f;
                config.unrotate_centroids = true;
                config.early_termination = false;
                config.sampling_fraction = sampling_fraction;
                config.use_blas_only = false;
                config.min_not_pruned_pct = min_not_pruned_pct;
                config.max_not_pruned_pct = max_not_pruned_pct;
                config.adjustment_factor_for_partial_d = adjustment_factor;
                auto is_angular = std::find(
                    bench_utils::ANGULAR_DATASETS.begin(),
                    bench_utils::ANGULAR_DATASETS.end(),
                    dataset
                );
                if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
                    std::cout << "Using spherical k-means" << std::endl;
                    config.angular = true;
                }

                auto kmeans_state = skmeans::SuperKMeans(n_clusters, d, config);

                bench_utils::TicToc timer;
                timer.Tic();
                std::vector<float> centroids = kmeans_state.Train(data.data(), n);
                timer.Toc();

                double construction_time_ms = timer.GetMilliseconds();
                int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
                double final_objective = kmeans_state.iteration_stats.back().objective;

                std::cout << "Training: " << construction_time_ms << " ms, "
                          << "Objective: " << final_objective << ", "
                          << "Iterations: " << actual_iterations << std::endl;

                // Skip assignment and recall computation for this benchmark
                std::vector<std::tuple<int, float, float, float, float>> results_knn_10;
                std::vector<std::tuple<int, float, float, float, float>> results_knn_100;

                std::unordered_map<std::string, std::string> config_map;
                config_map["iters"] = std::to_string(config.iters);
                config_map["sampling_fraction"] = std::to_string(config.sampling_fraction);
                config_map["n_threads"] = std::to_string(config.n_threads);
                config_map["seed"] = std::to_string(config.seed);
                config_map["use_blas_only"] = config.use_blas_only ? "true" : "false";
                config_map["tol"] = std::to_string(config.tol);
                config_map["recall_tol"] = std::to_string(config.recall_tol);
                config_map["early_termination"] = config.early_termination ? "true" : "false";
                config_map["sample_queries"] = config.sample_queries ? "true" : "false";
                config_map["objective_k"] = std::to_string(config.objective_k);
                config_map["ann_explore_fraction"] = std::to_string(config.ann_explore_fraction);
                config_map["unrotate_centroids"] = config.unrotate_centroids ? "true" : "false";
                config_map["verbose"] = config.verbose ? "true" : "false";
                config_map["min_not_pruned_pct"] = std::to_string(config.min_not_pruned_pct);
                config_map["max_not_pruned_pct"] = std::to_string(config.max_not_pruned_pct);
                config_map["adjustment_factor_for_partial_d"] =
                    std::to_string(config.adjustment_factor_for_partial_d);
                config_map["not_pruned_pct_window_size"] = std::to_string(PRUNING_PCT_WINDOW_SIZE);

                bench_utils::WriteResultsToCsv(
                    experiment_name,
                    algorithm,
                    dataset,
                    n_iters,
                    actual_iterations,
                    static_cast<int>(d),
                    n,
                    static_cast<int>(n_clusters),
                    construction_time_ms,
                    static_cast<int>(THREADS),
                    final_objective,
                    config_map,
                    results_knn_10,
                    results_knn_100
                );
            }
        }
    }
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "Results written to: results/sweet_pruning_spot.csv" << std::endl;
    std::cout << "========================================" << std::endl;
}
