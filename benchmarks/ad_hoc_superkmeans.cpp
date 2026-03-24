#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

#include "superkmeans/distance_computers/kernels.cuh"

int main(int argc, char* argv[]) {
#ifdef USE_CUDA
    // Trigger GPU Initialization
    // We need to do this before benchmarking, as the GPU will only initialize
    // when the first kernel is launched. Therefore we now launch a bogus kernel first.
    printf("Trigger GPU initialization.\n");
    skmeans::kernels::trigger_gpu_initialization();
    printf("Triggered GPU initialization.\n");
#endif

    // Experiment configuration
    const std::string algorithm = "superkmeans";

    // Choose dataset by name. You can also pass the dataset name as the first CLI argument.
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("llama");

    // Experiment name can be passed as second argument (default: "end_to_end")
    std::string experiment_name = (argc > 2) ? std::string(argv[2]) : std::string("end_to_end");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters =  //10000;
          std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = bench_utils::MAX_ITERS;
    float sampling_fraction = 1.0;
    std::string filename = bench_utils::get_data_path(dataset);
    std::string filename_queries = bench_utils::get_query_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;
    try {
        data.resize(n * d);
        queries.resize(n_queries * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    file_queries.close();

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.objective_k = 100;
    config.ann_explore_fraction = 0.01f;
    config.unrotate_centroids = true;
    config.perform_assignments = false;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = false;

    auto is_angular = std::find(bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset);
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }


    // Time the training
    bench_utils::TicToc timer;
    timer.Tic();

    auto kmeans_state =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config
        );
    std::vector<float> centroids = kmeans_state.Train(
        data.data(), n //,
                       // queries.data(), n_queries
    );
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();

    // Get actual iterations and final objective
    int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
    double final_objective = kmeans_state.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective: " << final_objective << std::endl;

    // return 0;

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

        // Load ground truth
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        // Assign each data point to its nearest centroid using SuperKMeans::Assign()
        auto assignments = kmeans_state.Assign(data.data(), centroids.data(), n, n_clusters);

        // Compute recall for both KNN values
        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);

        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        // COMMENTED OUT: CSV writing for ad-hoc testing
        /*
        // Create comprehensive config dictionary with all parameters
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
        config_map["perform_assignments"] = config.perform_assignments ? "true" : "false";
        config_map["verbose"] = config.verbose ? "true" : "false";

        // Write results to CSV
        bench_utils::write_results_to_csv(
            experiment_name, algorithm, dataset, n_iters, actual_iterations,
            static_cast<int>(d), n, static_cast<int>(n_clusters), construction_time_ms,
            static_cast<int>(THREADS), final_objective, config_map,
            results_knn_10, results_knn_100
        );
        */
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
