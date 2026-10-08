#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

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
    const std::string algorithm = "superkmeans";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");
    bool blas_only = !(argc > 2 && std::string(argv[2]) == "pruning");
    bool in_place = (argc > 3 && std::string(argv[3]) == "inplace");

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
              << " sampling_fraction=" << sampling_fraction
              << " training=" << (in_place ? "in-place" : "out-of-place") << "\n";
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

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.objective_k = 100;
    config.ann_explore_fraction = 0.01f;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = blas_only;
    config.tol = 1e-3f;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans_state = skmeans::SuperKMeans(n_clusters, d, config);
    const double rss_after_load = bench_utils::PeakRSSGiB();
    std::cout << "Peak RSS after loading data: " << std::fixed << std::setprecision(2)
              << rss_after_load << " GiB" << std::endl;

    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids =
        in_place ? kmeans_state.TrainInPlace(data.data(), n) : kmeans_state.Train(data.data(), n);
    timer.Toc();

    const double rss_after_train = bench_utils::PeakRSSGiB();
    std::cout << "Peak RSS after training:     " << std::fixed << std::setprecision(2)
              << rss_after_train << " GiB  (training added " << rss_after_train - rss_after_load
              << " GiB)" << std::endl;

    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
    double final_objective = kmeans_state.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective: " << final_objective << std::endl;

    // Compute assignments and cluster balance statistics
    auto assignments =
        kmeans_state.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);

    using SKM = skmeans::SuperKMeans<>;
    double wcss_f32 = SKM::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d);
    std::cout << "WCSS (f32): " << std::fixed << std::setprecision(2) << wcss_f32 << std::endl;

    auto balance_stats =
        skmeans::SuperKMeans<>::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance_stats.print();

    // Compute top-k distances for a random sample of points
    std::string topk_output = bench_utils::BENCHMARKS_ROOT + "/results/topk_distances_" + dataset +
                              "_f32" + (in_place ? "_inplace" : "") + ".json";
    bench_utils::ComputeAndStoreTopkDistances(
        data.data(),
        centroids.data(),
        n,
        n_clusters,
        d,
        /*k=*/100,
        /*sample_size=*/1000,
        topk_output
    );

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

        // TrainInPlace leaves the data buffer and the centroids in the rotated domain, so the
        // queries must be rotated too before being compared against the centroids.
        std::vector<float> rotated_queries;
        const float* queries_p = queries.data();
        if (in_place) {
            skmeans::ADSamplingPruner pruner(d, skmeans::PRUNER_INITIAL_THRESHOLD, config.seed);
            rotated_queries.resize(n_queries * d);
            skmeans::ExecutorScope executor_scope(nullptr, THREADS);
            pruner.Rotate(executor_scope.Get(), queries.data(), rotated_queries.data(), n_queries);
            queries_p = rotated_queries.data();
        }

        auto results_knn_10 = bench_utils::ComputeRecall(
            gt_map, assignments, queries_p, centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::PrintRecallResults(results_knn_10, 10);

        auto results_knn_100 = bench_utils::ComputeRecall(
            gt_map, assignments, queries_p, centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::PrintRecallResults(results_knn_100, 100);
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
