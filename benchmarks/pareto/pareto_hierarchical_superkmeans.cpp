#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"

int main(int argc, char* argv[]) {
    const std::string experiment_name = "pareto_hierarchical";
    const std::string algorithm = "hierarchical_superkmeans";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("openai");
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    std::string filename = bench_utils::GetDataPath(dataset);
    std::string filename_queries = bench_utils::GetQueryPath(dataset);
    const size_t THREADS = skmeans::ResolveNumThreads(0);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << "\n";
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;
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
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    file_queries.close();

    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);

    // Triple nested loop for hyperparameter grid search
    for (int iters_meso : bench_utils::HIERARCHICAL_PARETO_MESOCLUSTERING_ITERS) {
        std::cout << "\n#################################################" << std::endl;
        std::cout << "# Mesoclustering iterations = " << iters_meso << std::endl;
        std::cout << "#################################################" << std::endl;

        for (int iters_fine : bench_utils::HIERARCHICAL_PARETO_FINECLUSTERING_ITERS) {
            std::cout << "\n============================================" << std::endl;
            std::cout << "## Fineclustering iterations = " << iters_fine << std::endl;
            std::cout << "============================================" << std::endl;

            for (int iters_refine : bench_utils::HIERARCHICAL_PARETO_REFINEMENT_ITERS) {
                std::cout << "\n========================================" << std::endl;
                std::cout << "Running with iters_meso=" << iters_meso
                          << ", iters_fine=" << iters_fine << ", iters_refine=" << iters_refine
                          << std::endl;
                std::cout << "========================================" << std::endl;

                skmeans::HierarchicalSuperKMeansConfig config;
                // Base SuperKMeans config parameters
                config.verbose = false;
                config.n_threads = THREADS;
                config.objective_k = 100;
                config.ann_explore_fraction = 0.01f;
                config.unrotate_centroids = true;
                config.early_termination = false;
                config.sampling_fraction = 1.0f;
                config.use_blas_only = false;

                // Hierarchical SuperKMeans specific parameters
                config.iters_mesoclustering = iters_meso;
                config.iters_fineclustering = iters_fine;
                config.iters_refinement = iters_refine;

                auto is_angular = std::find(
                    bench_utils::ANGULAR_DATASETS.begin(),
                    bench_utils::ANGULAR_DATASETS.end(),
                    dataset
                );
                if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
                    config.angular = true;
                }

                auto kmeans_state = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);

                bench_utils::TicToc timer;
                timer.Tic();
                std::vector<float> centroids = kmeans_state.Train(data.data(), n);
                timer.Toc();

                double construction_time_ms = timer.GetMilliseconds();

                // For hierarchical superkmeans, total iterations = sum of all three phases
                int actual_iterations = iters_meso + iters_fine + iters_refine;

                // Compute final objective (get last refinement objective if available)
                double final_objective = 0.0;
                if (!kmeans_state.hierarchical_iteration_stats.refinement_iteration_stats.empty()) {
                    final_objective =
                        kmeans_state.hierarchical_iteration_stats.refinement_iteration_stats.back()
                            .objective;
                } else if (!kmeans_state.hierarchical_iteration_stats.fineclustering_iteration_stats
                                .empty()) {
                    final_objective =
                        kmeans_state.hierarchical_iteration_stats.fineclustering_iteration_stats
                            .back()
                            .objective;
                } else if (!kmeans_state.hierarchical_iteration_stats.mesoclustering_iteration_stats
                                .empty()) {
                    final_objective =
                        kmeans_state.hierarchical_iteration_stats.mesoclustering_iteration_stats
                            .back()
                            .objective;
                }

                std::cout << "\nTraining completed in " << construction_time_ms << " ms"
                          << std::endl;
                std::cout << "Total iterations: " << actual_iterations << std::endl;
                std::cout << "Final objective: " << final_objective << std::endl;

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
                    auto assignments =
                        kmeans_state.Assign(data.data(), centroids.data(), n, n_clusters);

                    // Compute cluster balance statistics
                    auto balance_stats =
                        skmeans::HierarchicalSuperKMeans<>::GetClustersBalanceStats(
                            assignments.data(), n, n_clusters
                        );
                    balance_stats.print();

                    auto results_knn_10 = bench_utils::ComputeRecall(
                        gt_map,
                        assignments,
                        queries.data(),
                        centroids.data(),
                        n_queries,
                        n_clusters,
                        d,
                        10
                    );
                    bench_utils::PrintRecallResults(results_knn_10, 10);
                    auto results_knn_100 = bench_utils::ComputeRecall(
                        gt_map,
                        assignments,
                        queries.data(),
                        centroids.data(),
                        n_queries,
                        n_clusters,
                        d,
                        100
                    );
                    bench_utils::PrintRecallResults(results_knn_100, 100);

                    std::unordered_map<std::string, std::string> config_map;
                    config_map["iters_mesoclustering"] =
                        std::to_string(config.iters_mesoclustering);
                    config_map["iters_fineclustering"] =
                        std::to_string(config.iters_fineclustering);
                    config_map["iters_refinement"] = std::to_string(config.iters_refinement);
                    config_map["sampling_fraction"] = std::to_string(config.sampling_fraction);
                    config_map["n_threads"] = std::to_string(config.n_threads);
                    config_map["seed"] = std::to_string(config.seed);
                    config_map["use_blas_only"] = config.use_blas_only ? "true" : "false";
                    config_map["tol"] = std::to_string(config.tol);
                    config_map["recall_tol"] = std::to_string(config.recall_tol);
                    config_map["early_termination"] = config.early_termination ? "true" : "false";
                    config_map["sample_queries"] = config.sample_queries ? "true" : "false";
                    config_map["objective_k"] = std::to_string(config.objective_k);
                    config_map["ann_explore_fraction"] =
                        std::to_string(config.ann_explore_fraction);
                    config_map["unrotate_centroids"] = config.unrotate_centroids ? "true" : "false";
                    config_map["verbose"] = config.verbose ? "true" : "false";

                    bench_utils::WriteResultsToCsv(
                        experiment_name,
                        algorithm,
                        dataset,
                        actual_iterations, // Total iterations
                        actual_iterations, // Actual = requested for hierarchical
                        static_cast<int>(d),
                        n,
                        static_cast<int>(n_clusters),
                        construction_time_ms,
                        static_cast<int>(THREADS),
                        final_objective,
                        config_map,
                        results_knn_10,
                        results_knn_100,
                        balance_stats.ToJson()
                    );
                } else {
                    if (!gt_file.good()) {
                        std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
                    }
                    if (!queries_file_check.good()) {
                        std::cout << "Queries file not found: " << filename_queries << std::endl;
                    }
                    std::cout << "Skipping CSV output (recall computation requires ground truth)"
                              << std::endl;
                }
            }
        }
    }
}
