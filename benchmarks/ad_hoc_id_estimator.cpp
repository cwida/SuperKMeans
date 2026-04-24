#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/pdx/utils.h"

// Number of points to sample for ID estimation
constexpr size_t N_SAMPLE = 50000;
// k values to evaluate for MLE and LID
constexpr int K_VALUES[] = {5, 10, 20, 50, 100};
constexpr size_t N_K_VALUES = sizeof(K_VALUES) / sizeof(K_VALUES[0]);
// Maximum k needed (for kNN precomputation).
// We request K_MAX+1 from BatchComputer to account for self-matches.
constexpr int K_MAX = 100;
constexpr int K_QUERY = K_MAX + 1;

using L2BatchComputer = skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;

/**
 * @brief Two-NN estimator (Facco et al. 2017).
 *
 * For each point, compute mu = r2/r1 (ratio of 2nd to 1st NN distance).
 * Global ID estimate: d_hat = n / sum(log(mu_i))
 */
static double EstimateTwoNN(const std::vector<float>& knn_dists, size_t n, int k) {
    double sum_log_mu = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float r1 = knn_dists[i * k + 0];
        float r2 = knn_dists[i * k + 1];
        if (r1 > 0.0f) {
            sum_log_mu += std::log(static_cast<double>(r2) / static_cast<double>(r1));
            ++valid;
        }
    }
    if (sum_log_mu <= 0.0) return 0.0;
    return static_cast<double>(valid) / sum_log_mu;
}

/**
 * @brief MLE estimator (Levina & Bickel 2004).
 *
 * For each point i with kNN distances r_{i,1} <= ... <= r_{i,k}:
 *   m_k(x_i) = [ 1/(k-1) * sum_{j=1}^{k-1} log(r_{i,k} / r_{i,j}) ]^{-1}
 *
 * Global estimate: ID = (1/n) * sum m_k(x_i)
 */
static double EstimateMLE(
    const std::vector<float>& knn_dists, size_t n, int k_max, int k
) {
    double sum_id = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float rk = knn_dists[i * k_max + (k - 1)];
        if (rk <= 0.0f) continue;
        double sum_log = 0.0;
        for (int j = 0; j < k - 1; ++j) {
            float rj = knn_dists[i * k_max + j];
            if (rj <= 0.0f) continue;
            sum_log += std::log(static_cast<double>(rk) / static_cast<double>(rj));
        }
        if (sum_log > 0.0) {
            sum_id += static_cast<double>(k - 1) / sum_log;
            ++valid;
        }
    }
    return (valid > 0) ? sum_id / static_cast<double>(valid) : 0.0;
}

/**
 * @brief LID MLE estimator (Amsaleg et al. 2015 / Houle 2017).
 *
 * For each point i with kNN distances r_{i,1} <= ... <= r_{i,k}:
 *   LID(x_i) = -k / sum_{j=1}^{k} log(r_{i,j} / r_{i,k})
 *
 * Global estimate: median of per-point LID values.
 */
static double EstimateLID(
    const std::vector<float>& knn_dists, size_t n, int k_max, int k
) {
    std::vector<double> lid_values;
    lid_values.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float rk = knn_dists[i * k_max + (k - 1)];
        if (rk <= 0.0f) continue;
        double sum_log = 0.0;
        int count = 0;
        for (int j = 0; j < k - 1; ++j) {
            float rj = knn_dists[i * k_max + j];
            if (rj <= 0.0f) continue;
            sum_log += std::log(static_cast<double>(rj) / static_cast<double>(rk));
            ++count;
        }
        // sum_log is negative (rj < rk), so -count/sum_log is positive
        if (sum_log < 0.0) {
            lid_values.push_back(-static_cast<double>(count) / sum_log);
        }
    }
    if (lid_values.empty()) return 0.0;

    std::sort(lid_values.begin(), lid_values.end());
    return lid_values[lid_values.size() / 2]; // median
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>" << std::endl;
        return 1;
    }
    const std::string dataset = argv[1];

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << std::endl;
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;

    // Load data
    std::string data_path = bench_utils::get_data_path(dataset);
    std::vector<float> data(n * d);
    {
        std::ifstream file(data_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }

    // Sample points
    const size_t n_sample = std::min(N_SAMPLE, n);
    std::vector<float> sampled(n_sample * d);
    {
        std::mt19937 rng(42);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < n_sample; ++i) {
            std::memcpy(
                sampled.data() + i * d,
                data.data() + indices[i] * d,
                d * sizeof(float)
            );
        }
    }
    std::cout << "Sampled " << n_sample << " points for ID estimation" << std::endl;

    // Pre-compute squared L2 norms
    auto norms = skmeans::ComputeNorms(sampled.data(), n_sample, d);

    // Allocate scratch buffer for BatchComputer
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

    // Compute (K_MAX+1)-NN via BLAS-accelerated BatchComputer (self-kNN includes self-match)
    std::cout << "Computing " << K_QUERY << "-NN distances (BLAS)..." << std::flush;
    bench_utils::TicToc timer;
    timer.Tic();

    std::vector<uint32_t> knn_indices(n_sample * K_QUERY);
    std::vector<float> knn_dists_raw(n_sample * K_QUERY);
    L2BatchComputer::FindKNearestNeighbors(
        sampled.data(), sampled.data(),
        n_sample, n_sample, d,
        norms.data(), norms.data(),
        K_QUERY,
        knn_indices.data(), knn_dists_raw.data(),
        tmp_buf.data()
    );

    // Post-process: strip self-matches and convert squared distances to L2 distances.
    // BatchComputer returns squared L2 distances. Each point's nearest neighbor list
    // may include itself (distance ~0). We skip self-matches and keep K_MAX neighbors.
    std::vector<float> knn_dists(n_sample * K_MAX);
    for (size_t i = 0; i < n_sample; ++i) {
        int out_idx = 0;
        for (int j = 0; j < K_QUERY && out_idx < K_MAX; ++j) {
            if (knn_indices[i * K_QUERY + j] == static_cast<uint32_t>(i)) continue;
            knn_dists[i * K_MAX + out_idx] = std::sqrt(knn_dists_raw[i * K_QUERY + j]);
            ++out_idx;
        }
    }

    timer.Toc();
    std::cout << " done (" << std::fixed << std::setprecision(1)
              << timer.GetMilliseconds() << " ms)" << std::endl;

    // ── Two-NN ──
    double two_nn = EstimateTwoNN(knn_dists, n_sample, K_MAX);
    std::cout << "\n=== Intrinsic Dimensionality Estimates ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTwo-NN (Facco et al. 2017):  " << two_nn << std::endl;

    // ── MLE and LID for various k values ──
    std::cout << "\n" << std::setw(8) << "k"
              << std::setw(16) << "MLE (L&B)"
              << std::setw(16) << "LID (median)"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    for (size_t ki = 0; ki < N_K_VALUES; ++ki) {
        int k = K_VALUES[ki];
        double mle = EstimateMLE(knn_dists, n_sample, K_MAX, k);
        double lid = EstimateLID(knn_dists, n_sample, K_MAX, k);
        std::cout << std::setw(8) << k
                  << std::setw(16) << mle
                  << std::setw(16) << lid
                  << std::endl;
    }

    // ── Suggested TARGET_D ──
    // Use MLE at k=20 as the primary estimate, round up to next multiple of 64
    double mle_k20 = EstimateMLE(knn_dists, n_sample, K_MAX, 20);
    size_t suggested = ((static_cast<size_t>(std::ceil(mle_k20)) + 63) / 64) * 64;
    suggested = std::min(suggested, d);
    std::cout << "\nSuggested TARGET_D (MLE@k=20 rounded to 64): " << suggested << std::endl;

    return 0;
}
