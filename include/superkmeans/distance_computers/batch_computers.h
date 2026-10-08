#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/executor.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
#include <Eigen/Dense>

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::f32> {

    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<Quantization::f32>;
    using norms_t = skmeans_value_t<Quantization::f32>;
    using knn_candidate_t = KNNCandidate;
    using layout_t = PDXLayout<Quantization::f32>;
    using MatrixR = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  private:
    /**
     * @brief Performs BLAS matrix multiplication: distances = x * y^T
     *
     * Computes the dot product matrix between X and Y
     * Can optionally use only the first partial_d dimensions for partial distance computation.
     *
     * @param batch_x_p Pointer to query vectors batch (batch_n_x × d)
     * @param batch_y_p Pointer to reference vectors batch (batch_n_y × d)
     * @param batch_n_x Number of query vectors in batch
     * @param batch_n_y Number of reference vectors in batch
     * @param d Full dimensionality
     * @param partial_d Number of dimensions to use (if 0, uses all dimensions)
     * @param tmp_distances_buf Output buffer for distance matrix (batch_n_x × batch_n_y)
     */
    static void BlasMatrixMultiplication(
        const data_t* SKM_RESTRICT batch_x_p,
        const data_t* SKM_RESTRICT batch_y_p,
        const size_t batch_n_x,
        const size_t batch_n_y,
        const size_t d,
        const size_t partial_d,
        float* SKM_RESTRICT tmp_distances_buf
    ) {
        const char trans_a = 'T';
        const char trans_b = 'N';

        int m = static_cast<int>(batch_n_y);
        int n = static_cast<int>(batch_n_x);
        int k = static_cast<int>(partial_d > 0 && partial_d < d ? partial_d : d);
        float alpha = 1.0f;
        float beta = 0.0f;
        int lda = static_cast<int>(d);         // d of y (row stride in row-major)
        int ldb = static_cast<int>(d);         // d of x (row stride in row-major)
        int ldc = static_cast<int>(batch_n_y); // Leading dimension of tmp_distances_buf

        Sgemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            batch_y_p,
            lda,
            batch_x_p,
            ldb,
            beta,
            tmp_distances_buf,
            ldc
        );
    }

  public:
    /// Floats of scratch the nearest-neighbor searches need: one mini-batch distance tile per
    /// worker, so worker w writes to [w * MINI_BATCH_SIZE * Y_BATCH_SIZE, ...).
    static size_t ScratchSize(const ParallelExecutor& executor) {
        return executor.NumWorkers() * MINI_BATCH_SIZE * Y_BATCH_SIZE;
    }

    /**
     * @brief Finds the top-1 nearest neighbor for each query vector.
     *
     * Computes L2 distances between X abd Y
     * using the identity: ||x-y||² = ||x||² + ||y||² - 2*x·y
     *
     * @param x Query vectors in row-major layout (n_x × d)
     * @param y Reference vectors in row-major layout (n_y × d)
     * @param n_x Number of query vectors
     * @param n_y Number of reference vectors
     * @param d Dimensionality
     * @param norms_x Pre-computed squared L2 norms of query vectors
     * @param norms_y Pre-computed squared L2 norms of reference vectors
     * @param out_knn Output: index of nearest neighbor for each query
     * @param out_distances Output: distance to nearest neighbor for each query
     * @param tmp_distances_buf Scratch of ScratchSize(executor) floats
     *
     * One parallel region over mini-batches of MINI_BATCH_SIZE rows of X: each worker multiplies
     * its mini-batch by every Y tile in turn (single-threaded GEMM) and keeps the running top-1.
     */
    static void FindNearestNeighbor(
        ParallelExecutor& executor,
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT tmp_distances_buf
    ) {
        SKM_PROFILE_SCOPE("search");
        const size_t n_blocks = (n_x + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
        executor.ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t worker) {
            float* buffer = tmp_distances_buf + worker * MINI_BATCH_SIZE * Y_BATCH_SIZE;
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t i = block * MINI_BATCH_SIZE;
                const size_t rows = std::min(MINI_BATCH_SIZE, n_x - i);
                std::fill_n(out_distances + i, rows, std::numeric_limits<distance_t>::max());
                for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                    const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);
                    BlasMatrixMultiplication(x + i * d, y + j * d, rows, batch_n_y, d, 0, buffer);
                    for (size_t r = 0; r < rows; ++r) {
                        const auto i_idx = i + r;
                        const float norm_x_i = norms_x[i_idx];
                        float* row_p = buffer + r * batch_n_y;
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                        }
                        Eigen::Map<Eigen::Matrix<distance_t, 1, Eigen::Dynamic>> row(
                            row_p, batch_n_y
                        );
                        uint32_t knn_idx;
                        auto batch_top_1 = row.minCoeff(&knn_idx);
                        if (batch_top_1 < out_distances[i_idx]) {
                            out_distances[i_idx] = std::max(0.0f, batch_top_1);
                            out_knn[i_idx] = j + knn_idx;
                        }
                    }
                }
            }
        });
    }

    /**
     * @brief Finds the top-k nearest neighbors for each query vector.
     *
     * Similar to FindNearestNeighbor but maintains top-k candidates per query.
     * Results are merged across Y batches using partial sort.
     *
     * @param x Query vectors in row-major layout (n_x × d)
     * @param y Reference vectors in row-major layout (n_y × d)
     * @param n_x Number of query vectors
     * @param n_y Number of reference vectors
     * @param d Dimensionality
     * @param norms_x Pre-computed squared L2 norms of query vectors
     * @param norms_y Pre-computed squared L2 norms of reference vectors
     * @param k Number of nearest neighbors to find
     * @param out_knn Output: indices of k nearest neighbors for each query (size: n_x × k)
     * @param out_distances Output: distances to k nearest neighbors (size: n_x × k)
     * @param tmp_distances_buf Scratch of ScratchSize(executor) floats
     */
    static void FindKNearestNeighbors(
        ParallelExecutor& executor,
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        const size_t k,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT tmp_distances_buf
    ) {
        std::fill_n(out_distances, n_x * k, std::numeric_limits<distance_t>::max());
        std::fill_n(out_knn, n_x * k, static_cast<uint32_t>(-1));

        const size_t max_candidates = k + Y_BATCH_SIZE;
        const size_t n_blocks = (n_x + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
        executor.ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t worker) {
            float* buffer = tmp_distances_buf + worker * MINI_BATCH_SIZE * Y_BATCH_SIZE;
            // Per-worker candidates, allocated once to keep the hot loop free of heap allocation
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(max_candidates);
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t i = block * MINI_BATCH_SIZE;
                const size_t rows = std::min(MINI_BATCH_SIZE, n_x - i);
                for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                    const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);
                    BlasMatrixMultiplication(x + i * d, y + j * d, rows, batch_n_y, d, 0, buffer);
                    for (size_t r = 0; r < rows; ++r) {
                        const auto i_idx = i + r;
                        const float norm_x_i = norms_x[i_idx];
                        float* row_p = buffer + r * batch_n_y;
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                        }

                        // TODO(@lkuffo, low): I feel this can be improved
                        candidates.clear();
                        // Add previous top-k
                        for (size_t ki = 0; ki < k; ++ki) {
                            if (out_distances[i_idx * k + ki] <
                                std::numeric_limits<distance_t>::max()) {
                                candidates.emplace_back(
                                    out_distances[i_idx * k + ki], out_knn[i_idx * k + ki]
                                );
                            }
                        }
                        // Add current batch candidates
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            candidates.emplace_back(row_p[c], static_cast<uint32_t>(j + c));
                        }
                        size_t actual_k = std::min(k, candidates.size());
                        std::partial_sort(
                            candidates.begin(), candidates.begin() + actual_k, candidates.end()
                        );
                        for (size_t ki = 0; ki < actual_k; ++ki) {
                            out_distances[i_idx * k + ki] = std::max(0.0f, candidates[ki].first);
                            out_knn[i_idx * k + ki] = candidates[ki].second;
                        }
                        for (size_t ki = actual_k; ki < k; ++ki) {
                            out_distances[i_idx * k + ki] = std::numeric_limits<distance_t>::max();
                            out_knn[i_idx * k + ki] = static_cast<uint32_t>(-1);
                        }
                    }
                }
            }
        });
    }

    /**
     * @brief Finds nearest neighbors using partial GEMM+PRUNING.
     *
     * Hybrid approach that computes partial distances (first partial_d dimensions)
     * via GEMM, then uses ADSampling+PDX pruning to skip full distance computation
     * for unlikely candidates.
     *
     * @param x Query vectors in row-major layout (n_x × d)
     * @param y Reference vectors in row-major layout (n_y × d)
     * @param n_x Number of query vectors
     * @param n_y Number of reference vectors (centroids)
     * @param d Full dimensionality
     * @param norms_x Pre-computed partial squared L2 norms of queries (first partial_d dims)
     * @param norms_y Pre-computed partial squared L2 norms of references (first partial_d dims)
     * @param out_knn Input/Output: current assignment indices (updated with better assignments)
     * @param out_distances Input/Output: current distances (updated with better distances)
     * @param tmp_distances_buf Scratch of ScratchSize(executor) floats
     * @param pdx_centroids PDX layout containing centroids and searcher for pruned search
     * @param partial_d Number of dimensions used for initial BLAS computation
     * @param out_not_pruned_counts count of non-pruned vectors per query (for tuning d')
     *
     * Same schedule as FindNearestNeighbor: each worker runs its mini-batches through every Y
     * tile, the partial-d GEMM followed right away by the PDX pruned search.
     */
    static void FindNearestNeighborWithPruning(
        ParallelExecutor& executor,
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT tmp_distances_buf,
        const layout_t& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) {
        SKM_PROFILE_SCOPE("search");
        const size_t n_blocks = (n_x + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
        executor.ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t worker) {
            float* buffer = tmp_distances_buf + worker * MINI_BATCH_SIZE * Y_BATCH_SIZE;
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t i = block * MINI_BATCH_SIZE;
                const size_t rows = std::min(MINI_BATCH_SIZE, n_x - i);
                for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                    const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);
                    BlasMatrixMultiplication(
                        x + i * d, y + j * d, rows, batch_n_y, d, partial_d, buffer
                    );
                    for (size_t r = 0; r < rows; ++r) {
                        const auto i_idx = i + r;

                        // Norms: convert dot products to squared L2 distances
                        const float norm_x_i = norms_x[i_idx];
                        float* row_p = buffer + r * batch_n_y;
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                        }

                        // PDX pruned search per vector
                        auto data_p = x + (i_idx * d);
                        const auto prev_assignment = out_knn[i_idx];
                        distance_t dist_to_prev_centroid;
                        if (j == 0) {
                            // We get the assignment from the previous iteration
                            // To prune as soon as possible
                            dist_to_prev_centroid =
                                DistanceComputer<DistanceFunction::l2, Quantization::f32>::
                                    Horizontal(y + (prev_assignment * d), data_p, d);
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        size_t local_not_pruned = 0;
                        knn_candidate_t assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    data_p,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    row_p,
                                    partial_d,
                                    j / VECTOR_CHUNK_SIZE, // start cluster_idx
                                    (j + Y_BATCH_SIZE) /
                                        VECTOR_CHUNK_SIZE, // end cluster_idx; We use
                                                           // Y_BATCH_SIZE and not batch_n_y
                                                           // because otherwise we would not
                                                           // go up until incomplete clusters
                                    local_not_pruned
                                );
                        out_not_pruned_counts[i_idx] += local_not_pruned;
                        auto [assignment_idx, assignment_distance] = assignment;
                        out_knn[i_idx] = assignment_idx;
                        out_distances[i_idx] = assignment_distance;
                    }
                }
            }
        });
    }
};

} // namespace skmeans
