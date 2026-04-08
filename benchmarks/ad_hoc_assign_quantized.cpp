#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
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
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    int n_iters = 10;
    float sampling_fraction = 1.0f;
    std::string filename = bench_utils::get_data_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== F32 vs SQ8 Assignment Comparison ===" << std::endl;
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

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );

    // --- Train F32 ---
    std::cout << "\n--- Training F32 ---" << std::endl;
    skmeans::SuperKMeansConfig config_f32;
    config_f32.iters = n_iters;
    config_f32.verbose = true;
    config_f32.n_threads = THREADS;
    config_f32.unrotate_centroids = true;
    config_f32.early_termination = false;
    config_f32.sampling_fraction = sampling_fraction;
    config_f32.use_blas_only = false;
    config_f32.tol = 1e-3f;
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        config_f32.angular = true;
    }

    auto kmeans_f32 =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config_f32
        );
    bench_utils::TicToc timer_f32;
    timer_f32.Tic();
    auto centroids_f32 = kmeans_f32.Train(data.data(), n);
    timer_f32.Toc();
    std::cout << "F32 training: " << timer_f32.GetMilliseconds() << " ms" << std::endl;

    // --- Train SQ8 ---
    std::cout << "\n--- Training SQ8 ---" << std::endl;
    skmeans::SuperKMeansConfig config_u8;
    config_u8.iters = n_iters;
    config_u8.verbose = true;
    config_u8.n_threads = THREADS;
    config_u8.unrotate_centroids = true;
    config_u8.early_termination = false;
    config_u8.sampling_fraction = sampling_fraction;
    config_u8.tol = 1e-3f;
    config_u8.quantizer_type = skmeans::QuantizerType::sq8;
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        config_u8.angular = true;
    }

    auto kmeans_u8 =
        skmeans::SuperKMeans<skmeans::Quantization::u8, skmeans::DistanceFunction::l2>(
            n_clusters, d, config_u8
        );
    bench_utils::TicToc timer_u8;
    timer_u8.Tic();
    auto centroids_u8 = kmeans_u8.Train(data.data(), n);
    timer_u8.Toc();
    std::cout << "SQ8 training: " << timer_u8.GetMilliseconds() << " ms" << std::endl;

    // --- Assign with F32 centroids ---
    std::cout << "\n--- Assigning with F32 centroids ---" << std::endl;
    bench_utils::TicToc timer_assign_f32;
    timer_assign_f32.Tic();
    auto assignments_f32 =
        kmeans_f32.Assign(data.data(), centroids_f32.data(), n, n_clusters);
    timer_assign_f32.Toc();
    std::cout << "F32 assign: " << timer_assign_f32.GetMilliseconds() << " ms" << std::endl;

    // --- Assign with SQ8 centroids ---
    std::cout << "\n--- Assigning with SQ8 centroids ---" << std::endl;
    bench_utils::TicToc timer_assign_u8;
    timer_assign_u8.Tic();
    auto assignments_u8 =
        kmeans_u8.Assign(data.data(), centroids_u8.data(), n, n_clusters);
    timer_assign_u8.Toc();
    std::cout << "SQ8 assign: " << timer_assign_u8.GetMilliseconds() << " ms" << std::endl;

    // --- QuantizedAssign with SQ8 centroids ---
    std::cout << "\n--- QuantizedAssign with SQ8 centroids ---" << std::endl;
    bench_utils::TicToc timer_qassign_u8;
    timer_qassign_u8.Tic();
    auto assignments_qu8 =
        kmeans_u8.QuantizedAssign(data.data(), centroids_u8.data(), n, n_clusters);
    timer_qassign_u8.Toc();
    std::cout << "SQ8 QuantizedAssign: " << timer_qassign_u8.GetMilliseconds() << " ms" << std::endl;

    // --- Cross-assign: use F32 brute-force to assign to BOTH centroid sets ---
    // This gives a fair comparison of centroid quality independent of assignment method
    auto kmeans_f32_ref =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, skmeans::SuperKMeansConfig{}
        );
    auto assign_f32_to_f32 =
        kmeans_f32_ref.Assign(data.data(), centroids_f32.data(), n, n_clusters);
    auto assign_f32_to_u8 =
        kmeans_f32_ref.Assign(data.data(), centroids_u8.data(), n, n_clusters);

    // --- Compare centroid quality via WCSS ---
    auto compute_wcss = [&](const std::vector<uint32_t>& assignments,
                            const std::vector<float>& ctrs) {
        double wcss = 0.0;
#pragma omp parallel for reduction(+ : wcss) num_threads(THREADS)
        for (size_t i = 0; i < n; ++i) {
            uint32_t c = assignments[i];
            double local = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = data[i * d + j] - ctrs[c * d + j];
                local += diff * diff;
            }
            wcss += local;
        }
        return wcss;
    };

    double wcss_f32 = compute_wcss(assign_f32_to_f32, centroids_f32);
    double wcss_u8 = compute_wcss(assign_f32_to_u8, centroids_u8);

    // --- Also cross-assign with QuantizedAssign to SQ8 centroids ---
    auto assign_qu8_to_u8 =
        kmeans_u8.QuantizedAssign(data.data(), centroids_u8.data(), n, n_clusters);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Training time  - F32: " << timer_f32.GetMilliseconds()
              << " ms, SQ8: " << timer_u8.GetMilliseconds() << " ms"
              << " (speedup: " << timer_f32.GetMilliseconds() / timer_u8.GetMilliseconds() << "x)"
              << std::endl;
    std::cout << "Assign time    - F32: " << timer_assign_f32.GetMilliseconds()
              << " ms, SQ8 Assign: " << timer_assign_u8.GetMilliseconds()
              << " ms, SQ8 QuantizedAssign: " << timer_qassign_u8.GetMilliseconds()
              << " ms" << std::endl;
    std::cout << "WCSS (f32 ref) - F32: " << wcss_f32 << ", SQ8: " << wcss_u8
              << " (ratio: " << wcss_u8 / wcss_f32 << ")" << std::endl;

    double wcss_qu8 = compute_wcss(assign_qu8_to_u8, centroids_u8);
    std::cout << "WCSS (QuantizedAssign to SQ8 centroids): " << wcss_qu8
              << " (ratio vs f32: " << wcss_qu8 / wcss_f32 << ")" << std::endl;

    // --- Compare assignments: how many vectors get the same cluster? ---
    auto count_matches = [&](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
        size_t m = 0;
        for (size_t i = 0; i < n; ++i) {
            if (a[i] == b[i]) ++m;
        }
        return m;
    };

    size_t matches_f32_vs_u8 = count_matches(assign_f32_to_f32, assign_f32_to_u8);
    size_t matches_f32_vs_qu8 = count_matches(assign_f32_to_f32, assign_qu8_to_u8);
    size_t matches_u8_vs_qu8 = count_matches(assign_f32_to_u8, assign_qu8_to_u8);

    auto pct = [&](size_t m) { return 100.0 * static_cast<double>(m) / static_cast<double>(n); };

    std::cout << "\nAssignment agreement:" << std::endl;
    std::cout << "  F32 vs SQ8 (f32-assigned):          " << pct(matches_f32_vs_u8)
              << "% (" << matches_f32_vs_u8 << "/" << n << ")" << std::endl;
    std::cout << "  F32 vs SQ8 (QuantizedAssign):       " << pct(matches_f32_vs_qu8)
              << "% (" << matches_f32_vs_qu8 << "/" << n << ")" << std::endl;
    std::cout << "  SQ8 f32-assigned vs QuantizedAssign: " << pct(matches_u8_vs_qu8)
              << "% (" << matches_u8_vs_qu8 << "/" << n << ")" << std::endl;

    // Balance stats
    std::cout << "\n--- F32 cluster balance ---" << std::endl;
    auto balance_f32 =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::
            GetClustersBalanceStats(assign_f32_to_f32.data(), n, n_clusters);
    balance_f32.print();

    std::cout << "--- SQ8 cluster balance ---" << std::endl;
    auto balance_u8 =
        skmeans::SuperKMeans<skmeans::Quantization::u8, skmeans::DistanceFunction::l2>::
            GetClustersBalanceStats(assign_f32_to_u8.data(), n, n_clusters);
    balance_u8.print();
}
