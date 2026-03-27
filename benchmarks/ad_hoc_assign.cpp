#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = 40000; // bench_utils::get_default_n_clusters(n);
    int n_iters = 5;
    float sampling_fraction = 1.0;
    std::string filename = bench_utils::get_data_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== Assign Benchmark ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";

    std::vector<float> data;
    try {
        data.reserve(n * d);
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

    // --- Training ---
    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 5;
    config.iters_refinement = 0;
    config.verbose = false;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = false;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans =
        skmeans::HierarchicalSuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config
        );

    bench_utils::TicToc timer_train;
    timer_train.Tic();
    auto centroids = kmeans.Train(data.data(), n);
    timer_train.Toc();
    std::cout << "\nTraining completed in " << timer_train.GetMilliseconds() << " ms" << std::endl;

    // --- AssignTrainingPoints (GEMM+PRUNING fast path) ---
    bench_utils::TicToc timer_fast;
    timer_fast.Tic();
    auto assignments_fast =
        kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    timer_fast.Toc();
    double fast_ms = timer_fast.GetMilliseconds();
    std::cout << "\nAssignTrainingPoints: " << fast_ms << " ms" << std::endl;

    // --- Assign (brute force) ---
    bench_utils::TicToc timer_brute;
    timer_brute.Tic();
    auto assignments_brute = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    timer_brute.Toc();
    double brute_ms = timer_brute.GetMilliseconds();
    std::cout << "Assign:     " << brute_ms << " ms" << std::endl;

    // --- Compare ---
    double speedup = brute_ms / fast_ms;
    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (assignments_fast[i] == assignments_brute[i]) {
            ++matches;
        }
    }
    double match_pct = 100.0 * static_cast<double>(matches) / static_cast<double>(n);

    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;
    std::cout << "Agreement: " << match_pct << "% (" << matches << "/" << n << ")" << std::endl;
}
