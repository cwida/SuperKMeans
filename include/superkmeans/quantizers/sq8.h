#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/gemms.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq_common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace skmeans {

/**
 * @brief 8-bit scalar quantizer with ruy GEMM backend.
 *
 * Global min/max quantization: q[i] = round((val[i] - base) * scale), clamped to [0, MAX_VALUE].
 * For L2 distance the base cancels: ||x-y||² = inv_scale² * Σ(x_q - y_q)².
 *
 * Code layout per vector:
 *   [d u8 bytes]
 *   CodeSize(d) = d
 * base and scale are global (see GetParams()), so codes carry no per-vector metadata.
 */
class SQ8Quantizer : public IQuantizer<Quantization::sq8> {
  public:
    using quantized_t = IQuantizer::quantized_t;
    using u8_computer = DistanceComputer<DistanceFunction::l2, Quantization::sq8>;

    static constexpr uint8_t MAX_VALUE = 255;

    SQ8Quantizer() : has_amx(DetectAMX()) {}

    /**
     * @brief u8×u8→u32 dot product GEMM, dispatching between NumKong and ruy.
     *
     * - ARM: always ruy (best for thin/wide matrices on NEON).
     * - x86 with AMX: always NumKong (leverages AMX tiles).
     * - x86 without AMX (e.g. Zen 5): NumKong when k > THIN_MATRIX_THRESHOLD,
     *   ruy otherwise (NumKong is subpar for small k).
     *
     * Handles OMP parallelization internally (backends need different threading).
     */
    void MatrixMultiplication(
        const quantized_t* a,
        const quantized_t* b,
        uint32_t* out,
        size_t m,
        size_t n,
        size_t k,
        size_t a_stride,
        size_t b_stride
    ) const {
        const bool use_numkong = !IS_ARM && (has_amx || k > THIN_MATRIX_THRESHOLD);
        U8Gemm(a, b, out, m, n, k, a_stride, b_stride, use_numkong, centroids_nk_packed_buf, true);
    }

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        const size_t total_elements = n * d;
        params = ComputeScalarQuantizationParams(
            embeddings, total_elements, static_cast<float>(MAX_VALUE)
        );
        if (!tmp_dots_buf) {
            tmp_dots_buf.reset(new uint32_t[X_BATCH_SIZE * Y_BATCH_SIZE]);
        }
        centroids_nk_packed_buf.resize(nk_dots_packed_size_u8(Y_BATCH_SIZE, d));
        fitted = true;
    }

    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("encoding");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float quantization_scale = params.quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const float* embedding = embeddings + row * d;
            quantized_t* output_quantized_embedding = output_quantized_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                const int rounded = static_cast<int>(
                    std::round((embedding[i] - quantization_base) * quantization_scale)
                );
                if (SKM_UNLIKELY(rounded > MAX_VALUE)) {
                    output_quantized_embedding[i] = MAX_VALUE;
                } else if (SKM_UNLIKELY(rounded < 0)) {
                    output_quantized_embedding[i] = 0;
                } else {
                    output_quantized_embedding[i] = static_cast<uint8_t>(rounded);
                }
            }
        }
    }

    void Decode(
        const quantized_t* quantized_embeddings,
        float* output_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("decoding");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const quantized_t* quantized_embedding = quantized_embeddings + row * d;
            float* output_embedding = output_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                output_embedding[i] =
                    static_cast<float>(quantized_embedding[i]) * inv_quantization_scale +
                    quantization_base;
            }
        }
    }

    /**
     * @brief Compute float L2 squared norms of quantized vectors.
     *
     * Since base cancels in L2 distance:
     *   norm[i] = inv_scale² * Σ q[i][dim]²
     */
    void ComputeNorms(const quantized_t* quantized_embeddings, size_t n, size_t d, float* out_norms)
        const override {
        assert(fitted);
        const float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const quantized_t* row = quantized_embeddings + i * d;
            uint32_t sum_sq = 0;
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                uint32_t v = row[j];
                sum_sq += v * v;
            }
            out_norms[i] = inv_scale_sq * static_cast<float>(sum_sq);
        }
    }

    /**
     * @brief Find top-1 nearest neighbor using ruy u8 dot product GEMM.
     *
     * Computes dot products via ruy, then converts to L2² using pre-computed
     * norms: L2²(x,y) = ||x||² + ||y||² - 2·dot(x,y).
     */
    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/1st_blas");
        assert(fitted);
        (void) x_float;
        (void) y_float;
        const float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;
        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                MatrixMultiplication(
                    x + i * d, y + j * d, tmp_dots_buf.get(), batch_n_x, batch_n_y, d, d, d
                );

                // L2²(x,y) = ||x||² + ||y||² - 2·inv_scale²·dot(x,y)
                // TODO(@lkuffo, low): I believe we can just avoid the inv_scale term
                using MatrixR =
                    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<MatrixR> dists_matrix(tmp_buf, batch_n_x, batch_n_y);
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dots_row = tmp_dots_buf.get() + r * batch_n_y;
                    float* dists_row = tmp_buf + r * batch_n_y;
                    const float nx = norms_x[idx];

                    SKM_VECTORIZE_LOOP
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        dists_row[c] = nx + norms_y[j + c] -
                                       2.0f * inv_scale_sq * static_cast<float>(dots_row[c]);
                    }

                    uint32_t knn_idx;
                    float batch_top_1 = dists_matrix.row(r).minCoeff(&knn_idx);
                    if (batch_top_1 < out_distances[idx]) {
                        out_distances[idx] = std::max(0.0f, batch_top_1);
                        out_knn[idx] = static_cast<uint32_t>(j + knn_idx);
                    }
                }
            }
        }
    }

    void CacheDataPartialNorms(const quantized_t* data, size_t n, size_t d, uint32_t partial_d)
        override {
        CachePartialNorms(data, n, d, partial_d, cached_data_partial_norms);
        cached_data_partial_d_ = partial_d;
    }

    void CacheCentroidPartialNorms(
        const quantized_t* centroids,
        size_t n,
        size_t d,
        uint32_t partial_d
    ) override {
        CachePartialNorms(centroids, n, d, partial_d, cached_centroid_partial_norms);
    }

    /**
     * @brief Find top-1 nearest neighbor with PDX pruning for u8.
     *
     * Hybrid approach that computes partial distances (first partial_d dimensions)
     * via GEMM, then uses ADSampling+PDX pruning to skip full distance computation
     * for unlikely candidates.
     * Final distances are float (converted inside PDXearch::SetBestCandidate).
     * Partial norms must be cached via CacheDataPartialNorms / CacheCentroidPartialNorms.
     */
    void FindNearestNeighborWithPruning(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<Quantization::sq8>& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("search");
        (void) x_float;
        (void) y_float;

        if (cached_data_partial_norms.size() != n_x || cached_data_partial_d_ != partial_d) {
            CachePartialNorms(x, n_x, d, partial_d, cached_data_partial_norms);
            cached_data_partial_d_ = partial_d;
        }

        const float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;

        pdx_centroids.index->quantization_scale_squared =
            params.quantization_scale * params.quantization_scale;
        pdx_centroids.index->inverse_scale_factor_squared = inv_scale_sq;

        // Buffers for batched dot products (uint32_t)
        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("search/blas");
                    MatrixMultiplication(
                        x + i * d,
                        y + j * d,
                        tmp_dots_buf.get(),
                        batch_n_x,
                        batch_n_y,
                        partial_d,
                        d,
                        d
                    );
                }

                {
                    SKM_PROFILE_SCOPE("search/pdx");
                    // Convert dots to L2² and run PDXearch per query vector
#if defined(__clang__)
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
#else
#pragma omp parallel for num_threads(g_n_threads)
#endif
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const size_t i_idx = i + r;

                        // Norms: convert dot products to squared L2 distances
                        const uint32_t norm_x_i = cached_data_partial_norms[i_idx];
                        uint32_t* partial_distances_p = tmp_dots_buf.get() + r * batch_n_y;
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            partial_distances_p[c] = norm_x_i +
                                                     cached_centroid_partial_norms[j + c] -
                                                     2 * partial_distances_p[c];
                        }

                        // PDX pruned search per vector
                        auto data_p = x + (i_idx * d);
                        const auto prev_assignment = out_knn[i_idx];
                        float dist_to_prev_centroid;
                        if (j == 0) {
                            uint32_t u8_dist =
                                u8_computer::Horizontal(data_p, y + prev_assignment * d, d);
                            dist_to_prev_centroid = static_cast<float>(u8_dist) * inv_scale_sq;
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        // PDXearch with uint32_t partial distances
                        size_t local_not_pruned = 0;
                        auto assignment = pdx_centroids.searcher
                                              ->Top1PartialSearchWithThresholdAndPartialDistances(
                                                  data_p,
                                                  dist_to_prev_centroid,
                                                  prev_assignment,
                                                  partial_distances_p,
                                                  partial_d,
                                                  j / VECTOR_CHUNK_SIZE,
                                                  (j + Y_BATCH_SIZE) / VECTOR_CHUNK_SIZE,
                                                  local_not_pruned
                                              );
                        out_not_pruned_counts[i_idx] += local_not_pruned;
                        out_knn[i_idx] = assignment.index;
                        out_distances[i_idx] = assignment.distance;
                    }
                }
            }
        }
    }

    void ResetCentroidAccumulators(size_t n_clusters, size_t d) override {
        const size_t total = n_clusters * d;
        if (centroid_accumulators.size() < total)
            centroid_accumulators.resize(total);
        std::fill_n(centroid_accumulators.data(), total, 0u);
    }

    void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* /*centroid_accumulators_float*/,
        uint32_t* cluster_sizes,
        size_t n,
        size_t n_clusters,
        size_t d,
        uint32_t n_threads
    ) const override {
        SKM_PROFILE_SCOPE("SQ8::UpdateCentroids");
        assert(fitted);
#pragma omp parallel if (n_threads > 1) num_threads(n_threads)
        {
            uint32_t nt = n_threads;
            uint32_t rank = omp_get_thread_num();
            size_t c0 = (n_clusters * rank) / nt;
            size_t c1 = (n_clusters * (rank + 1)) / nt;
            for (size_t i = 0; i < n; ++i) {
                uint32_t ci = assignments[i];
                if (ci >= c0 && ci < c1) {
                    cluster_sizes[ci] += 1;
                    const auto* vec = encoded_data + i * d;
                    auto* acc = centroid_accumulators.data() + ci * d;
                    SKM_VECTORIZE_LOOP
                    for (size_t j = 0; j < d; ++j) {
                        acc[j] += vec[j];
                    }
                }
            }
        }
    }

    void FinalizeCentroids(
        float* centroids,
        const uint32_t* cluster_sizes,
        size_t n_clusters,
        size_t d
    ) const override {
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        // TODO(@lkuffo, low): large clusters may overflow the uint32 accumulator
        for (size_t i = 0; i < n_clusters; ++i) {
            if (cluster_sizes[i] == 0)
                continue;
            const uint32_t* acc = centroid_accumulators.data() + i * d;
            float* row = centroids + i * d;
            const uint32_t half = cluster_sizes[i] / 2;
            const float inv_size = 1.0f / static_cast<float>(cluster_sizes[i]);
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                uint8_t qval = static_cast<uint8_t>(static_cast<float>(acc[j] + half) * inv_size);
                row[j] = static_cast<float>(qval) * inv_quantization_scale + quantization_base;
            }
        }
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return true; }
    const ScalarQuantizationParams& GetParams() const { return params; }

  private:
    void CachePartialNorms(
        const quantized_t* vecs,
        size_t n,
        size_t d,
        uint32_t partial_d,
        std::vector<uint32_t>& out
    ) const {
        out.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = vecs + idx * d;
            SKM_VECTORIZE_LOOP
            for (size_t dim = 0; dim < partial_d; ++dim) {
                uint32_t v = row[dim];
                sum += v * v;
            }
            out[idx] = sum;
        }
    }

    ScalarQuantizationParams params{};
    bool fitted = false;
    bool has_amx = false;
    mutable std::vector<uint32_t> cached_data_partial_norms;
    mutable uint32_t cached_data_partial_d_ = 0;
    std::vector<uint32_t> cached_centroid_partial_norms;
    mutable std::vector<uint32_t> centroid_accumulators;
    mutable std::unique_ptr<uint32_t[]> tmp_dots_buf;
    mutable std::vector<char> centroids_nk_packed_buf;
};

} // namespace skmeans
