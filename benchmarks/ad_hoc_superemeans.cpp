#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/superemeans.h"

template <skmeans::Quantization Q>
int RunBenchmark(const std::string& dataset, const bool blas_only, const uint32_t n_epochs) {
    const std::string algorithm = std::string("superemeans_") + skmeans::QuantizationName(Q);

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    std::string filename = bench_utils::GetDataPath(dataset);
    std::string filename_queries = bench_utils::GetQueryPath(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_epochs=" << n_epochs
              << (blas_only ? " (BLAS-only)" : " (pruning)") << "\n";

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

    skmeans::SuperEMeansConfig config;
    config.iters = n_epochs;
    config.verbose = true;
    config.n_threads = skmeans::ResolveNumThreads(0);
    config.unrotate_centroids = true;
    config.use_blas_only = blas_only;
    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical e-means" << std::endl;
        config.angular = true;
    }

    auto emeans = skmeans::SuperEMeans<Q>(n_clusters, d, config);
    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = emeans.Train(data.data(), n);
    timer.Toc();
    std::cout << "\nTraining completed in " << timer.GetMilliseconds() << " ms" << std::endl;
    std::cout << "Batch size: " << emeans.GetBatchSize()
              << " | Initial step: " << emeans.GetInitialStep() << std::endl;

    std::vector<uint32_t> last_pass(emeans.assignments.get(), emeans.assignments.get() + n);
    timer.Reset();
    timer.Tic();
    auto assignments = emeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    timer.Toc();
    std::cout << "AssignTrainingPoints completed in " << timer.GetMilliseconds() << " ms"
              << std::endl;

    using SEM = skmeans::SuperEMeans<Q>;
    std::cout << "WCSS (f32, last pass):            " << std::fixed << std::setprecision(2)
              << SEM::ComputeWCSS(data.data(), centroids.data(), last_pass.data(), n, d)
              << std::endl;
    std::cout << "WCSS (f32, AssignTrainingPoints): "
              << SEM::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d)
              << std::endl;

    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    if (!std::ifstream(gt_filename).good()) {
        std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        return 0;
    }
    auto gt_map = bench_utils::ParseGroundTruthJson(gt_filename);
    for (const auto& [name, labels] :
         {std::pair{"last pass", &last_pass}, std::pair{"AssignTrainingPoints", &assignments}}) {
        std::cout << "\n  [" << name << "]" << std::endl;
        for (int knn : {10, 100}) {
            bench_utils::PrintRecallResults(
                bench_utils::ComputeRecall(
                    gt_map, *labels, queries.data(), centroids.data(), n_queries, n_clusters, d, knn
                ),
                knn
            );
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");
    std::string quantizer = (argc > 2) ? std::string(argv[2]) : std::string("f32");
    bool blas_only = argc > 3 && std::string(argv[3]) == "blas";
    uint32_t n_epochs =
        (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : skmeans::EMEANS_DEFAULT_EPOCHS;

    if (quantizer == "f32") {
        return RunBenchmark<skmeans::Quantization::f32>(dataset, blas_only, n_epochs);
    }
    if (quantizer == "sq8") {
        return RunBenchmark<skmeans::Quantization::sq8>(dataset, blas_only, n_epochs);
    }
    if (quantizer == "lvq4") {
        return RunBenchmark<skmeans::Quantization::lvq4>(dataset, blas_only, n_epochs);
    }
    if (quantizer == "rabitq") {
        return RunBenchmark<skmeans::Quantization::rabitq>(dataset, blas_only, n_epochs);
    }
    std::cerr << "Unknown quantizer '" << quantizer << "' (use f32|sq8|lvq4|rabitq)\n";
    return 1;
}
