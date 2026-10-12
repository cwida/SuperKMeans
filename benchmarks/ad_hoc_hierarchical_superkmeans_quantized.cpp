#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"

template <skmeans::Quantization Q>
void RunBenchmark(const std::string& dataset, bool blas_only) {
    using HSKM = skmeans::HierarchicalSuperKMeans<Q>;
    const std::string quantizer_name = skmeans::QuantizationName(Q);

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;

    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    std::string filename = bench_utils::GetDataPath(dataset);
    std::string filename_queries = bench_utils::GetQueryPath(dataset);
    const size_t THREADS = skmeans::ResolveNumThreads(0);

    const std::string algorithm = "hierarchical_superkmeans_" + quantizer_name;
    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " blas_only=" << blas_only << "\n";

    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(filename_queries, std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open " << filename_queries << std::endl;
            return;
        }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    skmeans::HierarchicalSuperKMeansConfig config;
    config.verbose = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.tol = 1e-3f;
    config.use_blas_only = blas_only;
    config.quantized_centroid_update = true;
    config.full_precision_final_centroids = false;
    // Hierarchical-specific
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 5;
    config.iters_refinement = 0;

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

    auto kmeans = HSKM(n_clusters, d, config);
    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Iteration config: meso=" << config.iters_mesoclustering
              << ", fine=" << config.iters_fineclustering << ", refine=" << config.iters_refinement
              << "\n";

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    auto q_assignments = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    double wcss_assign = HSKM::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d);
    double wcss_q_assign =
        HSKM::ComputeWCSS(data.data(), centroids.data(), q_assignments.data(), n, d);
    std::cout << "WCSS (f32, Assign):          " << std::fixed << std::setprecision(2)
              << wcss_assign << std::endl;
    std::cout << "WCSS (f32, QuantizedAssign): " << std::fixed << std::setprecision(2)
              << wcss_q_assign << std::endl;

    std::cout << "\n--- Assign() cluster balance ---" << std::endl;
    HSKM::GetClustersBalanceStats(assignments.data(), n, n_clusters).print();

    std::cout << "--- QuantizedAssign() cluster balance ---" << std::endl;
    HSKM::GetClustersBalanceStats(q_assignments.data(), n, n_clusters).print();

    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        auto gt_map = bench_utils::ParseGroundTruthJson(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        std::cout << "\n  [Assign()]" << std::endl;
        bench_utils::PrintRecallResults(
            bench_utils::ComputeRecall(
                gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
            ),
            10
        );
        bench_utils::PrintRecallResults(
            bench_utils::ComputeRecall(
                gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
            ),
            100
        );

        std::cout << "\n  [QuantizedAssign()]" << std::endl;
        bench_utils::PrintRecallResults(
            bench_utils::ComputeRecall(
                gt_map,
                q_assignments,
                queries.data(),
                centroids.data(),
                n_queries,
                n_clusters,
                d,
                10
            ),
            10
        );
        bench_utils::PrintRecallResults(
            bench_utils::ComputeRecall(
                gt_map,
                q_assignments,
                queries.data(),
                centroids.data(),
                n_queries,
                n_clusters,
                d,
                100
            ),
            100
        );
    } else {
        if (!gt_file.good())
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        if (!queries_file_check.good())
            std::cout << "Queries file not found: " << filename_queries << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <sq8|lvq4|rabitq> [pruning]\n";
        return 1;
    }

    const std::string dataset = argv[1];
    const std::string quantizer = argv[2];

    bool blas_only = true;
    if (argc > 3)
        blas_only = std::string(argv[3]) != "pruning";

    if (quantizer == "sq8") {
        RunBenchmark<skmeans::Quantization::sq8>(dataset, blas_only);
    } else if (quantizer == "lvq4") {
        RunBenchmark<skmeans::Quantization::lvq4>(dataset, blas_only);
    } else if (quantizer == "rabitq") {
        RunBenchmark<skmeans::Quantization::rabitq>(dataset, blas_only);
    } else {
        std::cerr << "Invalid quantizer: " << quantizer << " (expected: sq8, lvq4, rabitq)\n";
        return 1;
    }

    return 0;
}
