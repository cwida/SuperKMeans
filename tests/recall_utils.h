#pragma once

// Shared helpers for the IVF-recall ground-truth tests. The recall metric treats
// the trained centroids as IVF entry points and measures average recall@knn when
// probing the nearest `frac` of clusters. It is computed in float space on the
// final (unrotated) centroids, so it is identical machinery for every quantizer.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

namespace skm_test {

inline constexpr size_t RECALL_N = 10000;
inline constexpr size_t RECALL_D = 1024;
inline constexpr size_t RECALL_N_CLUSTERS = 100;
inline constexpr size_t RECALL_N_QUERIES = 100;
inline constexpr int RECALL_KNN = 10;
inline constexpr float RECALL_FRAC = 0.01f;
inline constexpr float RECALL_TOL = 0.04f;
inline constexpr float RECALL_PRUNE_TOL = 0.02f;
inline constexpr int RECALL_ITERS = 10;
inline constexpr unsigned int RECALL_SEED = 42;

// Regenerate with generate_recall_ground_truth.out
inline const std::unordered_map<std::string, float> RECALL_GROUND_TRUTH = {
    {"f32", 0.601f},
    {"sq8", 0.612f},
    {"lvq4", 0.615f},
    {"rabitq", 0.601f},
    {"hierarchical_f32", 0.589f},
    {"hierarchical_sq8", 0.591f},
    {"hierarchical_lvq4", 0.592f},
    {"hierarchical_rabitq", 0.587f},
};

// Load raw float32 data (n x max_d, row-major, no header) and copy the
// first `d` dimensions of each row.
inline std::vector<float> LoadTestDataSubdim(
    const std::string& path,
    size_t n,
    size_t max_d,
    size_t d
) {
    if (d > max_d)
        throw std::runtime_error("Requested d exceeds max_d");
    std::vector<float> full(n * max_d);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Could not open " + path + ". Run generate_wcss_ground_truth.out first."
        );
    }
    in.read(reinterpret_cast<char*>(full.data()), full.size() * sizeof(float));
    in.close();

    std::vector<float> data(n * d);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(&data[i * d], &full[i * max_d], d * sizeof(float));
    }
    return data;
}

// Exact top-k nearest neighbours (brute force, L2) of the first `n_queries` rows
// of `data` over the whole `data` set. Query row q is included in its own list
// (distance 0), matching the standard recall convention.
inline std::unordered_map<int, std::vector<int>> BuildGroundTruthNN(
    const float* data,
    size_t n,
    size_t d,
    size_t n_queries,
    int k
) {
    std::unordered_map<int, std::vector<int>> gt;
    const size_t kk = std::min(static_cast<size_t>(k), n);
    for (size_t q = 0; q < n_queries; ++q) {
        const float* query = data + q * d;
        std::vector<std::pair<float, int>> dists(n);
        for (size_t i = 0; i < n; ++i) {
            const float* v = data + i * d;
            float s = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                const float diff = query[j] - v[j];
                s += diff * diff;
            }
            dists[i] = {s, static_cast<int>(i)};
        }
        std::partial_sort(dists.begin(), dists.begin() + kk, dists.end());
        std::vector<int> nn(kk);
        for (size_t i = 0; i < kk; ++i)
            nn[i] = dists[i].second;
        gt[static_cast<int>(q)] = std::move(nn);
    }
    return gt;
}

// Average recall@knn at the exploration fraction closest to `frac`, extracted
// from bench_utils::ComputeRecall's fraction sweep. Returns -1 if not found.
inline float RecallAtFraction(
    const std::unordered_map<int, std::vector<int>>& gt_map,
    const std::vector<uint32_t>& assignments,
    const float* queries,
    const float* centroids,
    size_t n_queries,
    size_t n_clusters,
    size_t d,
    int knn,
    float frac
) {
    auto results = bench_utils::ComputeRecall(
        gt_map, assignments, queries, centroids, n_queries, n_clusters, d, knn
    );
    for (const auto& r : results) {
        if (std::abs(std::get<1>(r) - frac) < 1e-4f)
            return std::get<2>(r);
    }
    return -1.0f;
}

// Train n_clusters centroids on the first RECALL_N rows of data_path, then return
// average recall@RECALL_KNN at RECALL_FRAC. Defaults reproduce the ground-truth config.
//
// With IN_PLACE, training rotates the data buffer in place and leaves both data and centroids in
// the rotated domain. Rotation is orthonormal, so recall is comparable to the default path.
template <skmeans::Quantization q, bool IN_PLACE = false>
inline float ClusteringRecall(
    const std::string& data_path,
    size_t n_clusters = RECALL_N_CLUSTERS,
    size_t d = RECALL_D,
    bool pruning = false
) {
    auto data = LoadTestDataSubdim(data_path, RECALL_N, RECALL_D, d);
    auto gt = BuildGroundTruthNN(data.data(), RECALL_N, d, RECALL_N_QUERIES, RECALL_KNN);

    skmeans::SuperKMeansConfig config;
    config.iters = RECALL_ITERS;
    config.verbose = false;
    config.seed = RECALL_SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.max_points_per_cluster = 99999;
    config.n_threads = 1;
    config.use_blas_only = !pruning;

    auto kmeans = skmeans::SuperKMeans<q>(n_clusters, d, config);
    std::vector<skmeans::skmeans_centroid_value_t> centroids;
    if constexpr (IN_PLACE) {
        centroids = kmeans.TrainInPlace(data.data(), RECALL_N);
    } else {
        centroids = kmeans.Train(data.data(), RECALL_N);
    }
    auto assign = kmeans.Assign(data.data(), centroids.data(), RECALL_N, n_clusters);
    return RecallAtFraction(
        gt,
        assign,
        data.data(),
        centroids.data(),
        RECALL_N_QUERIES,
        n_clusters,
        d,
        RECALL_KNN,
        RECALL_FRAC
    );
}

template <skmeans::Quantization q, bool IN_PLACE = false>
inline float HierarchicalClusteringRecall(const std::string& data_path) {
    auto data = LoadTestDataSubdim(data_path, RECALL_N, RECALL_D, RECALL_D);
    auto gt = BuildGroundTruthNN(data.data(), RECALL_N, RECALL_D, RECALL_N_QUERIES, RECALL_KNN);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.verbose = false;
    config.seed = RECALL_SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.max_points_per_cluster = 99999;
    config.n_threads = 1;
    auto kmeans = skmeans::HierarchicalSuperKMeans<q>(RECALL_N_CLUSTERS, RECALL_D, config);
    std::vector<skmeans::skmeans_centroid_value_t> centroids;
    if constexpr (IN_PLACE) {
        centroids = kmeans.TrainInPlace(data.data(), RECALL_N);
    } else {
        centroids = kmeans.Train(data.data(), RECALL_N);
    }
    auto assign = kmeans.Assign(data.data(), centroids.data(), RECALL_N, RECALL_N_CLUSTERS);
    return RecallAtFraction(
        gt,
        assign,
        data.data(),
        centroids.data(),
        RECALL_N_QUERIES,
        RECALL_N_CLUSTERS,
        RECALL_D,
        RECALL_KNN,
        RECALL_FRAC
    );
}

struct AssignMethodRecall {
    float assign_training_points;
    float quantized_assign;
};

// Train once, then measure IVF recall of the training points under two assignment
// methods: AssignTrainingPoints (reuses trained state) vs QuantizedAssign (standalone
// re-encode). Used to check both yield comparable end-to-end recall.
template <skmeans::Quantization q>
inline AssignMethodRecall ClusteringRecallAssignMethods(
    const std::string& data_path,
    size_t n_clusters,
    size_t d
) {
    auto data = LoadTestDataSubdim(data_path, RECALL_N, RECALL_D, d);
    auto gt = BuildGroundTruthNN(data.data(), RECALL_N, d, RECALL_N_QUERIES, RECALL_KNN);

    skmeans::SuperKMeansConfig config;
    config.iters = RECALL_ITERS;
    config.verbose = false;
    config.seed = RECALL_SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.max_points_per_cluster = 99999;
    config.n_threads = 1;
    config.use_blas_only = false;

    auto kmeans = skmeans::SuperKMeans<q>(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), RECALL_N);

    auto atp = kmeans.AssignTrainingPoints(data.data(), centroids.data(), RECALL_N, n_clusters);
    auto qa = kmeans.QuantizedAssign(data.data(), centroids.data(), RECALL_N, n_clusters);

    return {
        RecallAtFraction(
            gt,
            atp,
            data.data(),
            centroids.data(),
            RECALL_N_QUERIES,
            n_clusters,
            d,
            RECALL_KNN,
            RECALL_FRAC
        ),
        RecallAtFraction(
            gt,
            qa,
            data.data(),
            centroids.data(),
            RECALL_N_QUERIES,
            n_clusters,
            d,
            RECALL_KNN,
            RECALL_FRAC
        ),
    };
}

} // namespace skm_test
