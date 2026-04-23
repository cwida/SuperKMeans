#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq8.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <utility>
#include <vector>

#include <numkong/numkong.h>

namespace skmeans {

/**
 * @brief 4-bit scalar quantizer with NumKong u4 GEMM backend.
 *
 * Global min/max quantization: q[i] = round((val[i] - base) * scale), clamped to [0, 15].
 * For L2 distance the base cancels: ||x-y||² = inv_scale² * Σ(x_q - y_q)².
 *
 * Encoded data is stored in nk_u4x2_t packed format (two nibbles per byte),
 * so CodeSize(d) = d/2. The NumKong u4 kernels operate directly on this format.
 *
 * Requires d to be even.
 */
class SQ4Quantizer : public IQuantizer<Quantization::u4> {
  public:
    using quantized_t = IQuantizer::quantized_t;
    using utils = UtilsComputer<Quantization::u4>;

    static constexpr uint8_t MAX_VALUE = 15;

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        if (d % 2 != 0) {
            throw std::invalid_argument(
                "SQ4Quantizer requires even dimensionality (got " + std::to_string(d) + ")"
            );
        }
        const size_t total_elements = n * d;
        params = ComputeQuantizationParams(embeddings, total_elements);
        // Pre-allocate scratch buffers (avoids expensive per-call allocation)
        nn_dists_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        pruning_dots_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        packed_buf.resize(nk_dots_packed_size_u4(Y_BATCH_SIZE, d));
        fitted = true;
    }

    static ScalarQuantizationParams ComputeQuantizationParams(
        const float* embeddings,
        const size_t total_elements
    ) {
        float global_min = std::numeric_limits<float>::max();
        float global_max = std::numeric_limits<float>::lowest();

#pragma omp parallel for reduction(min : global_min) reduction(max : global_max)                   \
    num_threads(g_n_threads)
        for (size_t i = 0; i < total_elements; ++i) {
            global_min = std::min(global_min, embeddings[i]);
            global_max = std::max(global_max, embeddings[i]);
        }

        const float range = global_max - global_min;
        const float scale = (range > 0) ? static_cast<float>(MAX_VALUE) / range : 1.0f;
        return {global_min, scale, 1.0f / scale};
    }

    /**
     * @brief Encode float data to packed u4x2 format.
     *
     * Quantizes each float to [0,15], then packs adjacent pairs into single bytes
     * using SIMD-accelerated PackU8ToU4x2. Output is d/2 bytes per vector.
     */
    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("encoding");
        assert(fitted);
        assert(d % 2 == 0);
        const float quantization_base = params.quantization_base;
        const float quantization_scale = params.quantization_scale;
        const size_t d_packed = d / 2;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const float* embedding = embeddings + row * d;
            quantized_t* output_row = output_quantized_embeddings + row * d_packed;

            thread_local uint8_t tmp_u8[SKM_MAX_DIMS];

            for (size_t i = 0; i < d; ++i) {
                const int rounded = static_cast<int>(
                    std::round((embedding[i] - quantization_base) * quantization_scale)
                );
                if (SKM_UNLIKELY(rounded > MAX_VALUE)) {
                    tmp_u8[i] = MAX_VALUE;
                } else if (SKM_UNLIKELY(rounded < 0)) {
                    tmp_u8[i] = 0;
                } else {
                    tmp_u8[i] = static_cast<uint8_t>(rounded);
                }
            }

            utils::PackU8ToU4x2(tmp_u8, output_row, d);
        }
    }

    /**
     * @brief Decode packed u4x2 data back to float.
     *
     * Unpacks nibbles and dequantizes: val = nibble * inv_scale + base.
     */
    void Decode(
        const quantized_t* quantized_embeddings,
        float* output_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("decoding");
        assert(fitted);
        assert(d % 2 == 0);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;
        const size_t d_packed = d / 2;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const quantized_t* packed_row = quantized_embeddings + row * d_packed;
            float* output_embedding = output_embeddings + row * d;
            SKM_VECTORIZE_LOOP
            for (size_t k = 0; k < d_packed; ++k) {
                uint8_t lo = packed_row[k] & 0x0F;
                uint8_t hi = (packed_row[k] >> 4) & 0x0F;
                output_embedding[2 * k] =
                    static_cast<float>(lo) * inv_quantization_scale + quantization_base;
                output_embedding[2 * k + 1] =
                    static_cast<float>(hi) * inv_quantization_scale + quantization_base;
            }
        }
    }

    /**
     * @brief Compute float L2 squared norms of packed u4x2 vectors.
     *
     * Since base cancels in L2 distance:
     *   norm[i] = inv_scale² * Σ q[i][dim]²
     */
    void ComputeNorms(
        const quantized_t* quantized_embeddings, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted);
        assert(d % 2 == 0);
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        const size_t d_packed = d / 2;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const quantized_t* row = quantized_embeddings + i * d_packed;
            uint32_t sum_sq = 0;
            SKM_VECTORIZE_LOOP
            for (size_t k = 0; k < d_packed; ++k) {
                uint32_t lo = row[k] & 0x0F;
                uint32_t hi = (row[k] >> 4) & 0x0F;
                sum_sq += lo * lo + hi * hi;
            }
            out_norms[i] = inv_scale_sq * static_cast<float>(sum_sq);
        }
    }

    /**
     * @brief Find top-1 nearest neighbor using NumKong u4 Euclidean kernel.
     *
     * Data is already in u4x2 packed format (d/2 bytes per vector),
     * so no conversion is needed. Centroids (y) are packed for NumKong
     * via nk_dots_pack_u4 per batch.
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
        assert(d % 2 == 0);
        (void)x_float;
        (void)y_float;
        (void)norms_x;
        (void)norms_y;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        const size_t d_packed = d / 2;

        const auto* x_u4 = reinterpret_cast<const nk_u4x2_t*>(x);
        const auto* y_u4 = reinterpret_cast<const nk_u4x2_t*>(y);

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                // Pack Y batch for NumKong
                const size_t pack_size = nk_dots_packed_size_u4(batch_n_y, d);
                if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
                nk_dots_pack_u4(y_u4 + j * d_packed, batch_n_y, d, d_packed, packed_buf.data());

                const size_t batch_r_stride = batch_n_y * sizeof(float);

#pragma omp parallel num_threads(g_n_threads)
                {
                    nk_configure_thread(nk_capabilities());
                    int tid = omp_get_thread_num();
                    int nt = omp_get_num_threads();
                    size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                    size_t start = tid * rows_per_t;
                    size_t count = std::min(rows_per_t, batch_n_x - start);
                    if (start < batch_n_x && count > 0) {
                        nk_euclideans_packed_u4(
                            x_u4 + (i + start) * d_packed,
                            packed_buf.data(),
                            nn_dists_buf.data() + start * batch_n_y,
                            count,
                            batch_n_y,
                            d,
                            d_packed,
                            batch_r_stride
                        );
                    }
                }

#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const float* dists_row = nn_dists_buf.data() + r * batch_n_y;

                    for (size_t c = 0; c < batch_n_y; ++c) {
                        if (dists_row[c] < out_distances[idx]) {
                            out_distances[idx] = dists_row[c];
                            out_knn[idx] = static_cast<uint32_t>(j + c);
                        }
                    }
                }
            }
        }

        // Convert best distances from quantized L2 to true L2²:
        // nk_euclideans_packed_u4 outputs sqrt(Σ(x_q-y_q)²);
        // true L2² = inv_scale² × Σ(x_q-y_q)²
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            out_distances[i] = inv_scale_sq * out_distances[i] * out_distances[i];
        }
    }

    size_t DefaultRerankK() const override { return 0; }

    /**
     * @brief Quantized coarse top-k search followed by exact f32 reranking.
     *
     * 1. Find top rerank_k candidates per query in quantized space (NumKong u4)
     * 2. Compute exact f32 L2² to those candidates using original float data
     * 3. Write the best to out_knn / out_distances
     */
    void FindNearestNeighborWithReranking(
        const quantized_t* x_quantized,
        const quantized_t* y_quantized,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        size_t rerank_k,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/rerank");
        assert(fitted);
        assert(d % 2 == 0);
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        using f32_computer = DistanceComputer<DistanceFunction::l2, Quantization::f32>;

        const size_t d_packed = d / 2;

        const auto* x_u4 = reinterpret_cast<const nk_u4x2_t*>(x_quantized);
        const auto* y_u4 = reinterpret_cast<const nk_u4x2_t*>(y_quantized);

        // Per-thread candidate buffers for top-k merge
        const size_t max_candidates = rerank_k + Y_BATCH_SIZE;
        const uint32_t num_threads = g_n_threads;
        std::vector<std::vector<std::pair<float, uint32_t>>> thread_candidates(num_threads);
        for (auto& tc : thread_candidates) {
            tc.reserve(max_candidates);
        }

        // Per-query top-k: (distance, index) for each query × rerank_k
        std::vector<float> topk_distances(n_x * rerank_k, std::numeric_limits<float>::max());
        std::vector<uint32_t> topk_indices(n_x * rerank_k, static_cast<uint32_t>(-1));

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                // Pack Y batch for NumKong
                const size_t pack_size = nk_dots_packed_size_u4(batch_n_y, d);
                if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
                nk_dots_pack_u4(y_u4 + j * d_packed, batch_n_y, d, d_packed, packed_buf.data());

                const size_t batch_r_stride = batch_n_y * sizeof(float);

#pragma omp parallel num_threads(g_n_threads)
                {
                    nk_configure_thread(nk_capabilities());
                    int tid = omp_get_thread_num();
                    int nt = omp_get_num_threads();
                    size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                    size_t start = tid * rows_per_t;
                    size_t count = std::min(rows_per_t, batch_n_x - start);
                    if (start < batch_n_x && count > 0) {
                        nk_euclideans_packed_u4(
                            x_u4 + (i + start) * d_packed,
                            packed_buf.data(),
                            nn_dists_buf.data() + start * batch_n_y,
                            count,
                            batch_n_y,
                            d,
                            d_packed,
                            batch_r_stride
                        );
                    }
                }

                // Merge candidates into per-query top-k
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const float* dists_row = nn_dists_buf.data() + r * batch_n_y;

                    auto& candidates = thread_candidates[omp_get_thread_num()];
                    candidates.clear();

                    // Add previous top-k
                    for (size_t ki = 0; ki < rerank_k; ++ki) {
                        if (topk_distances[idx * rerank_k + ki] <
                            std::numeric_limits<float>::max()) {
                            candidates.emplace_back(
                                topk_distances[idx * rerank_k + ki],
                                topk_indices[idx * rerank_k + ki]
                            );
                        }
                    }
                    // Add current batch candidates
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        candidates.emplace_back(dists_row[c], static_cast<uint32_t>(j + c));
                    }

                    size_t actual_k = std::min(rerank_k, candidates.size());
                    std::partial_sort(
                        candidates.begin(), candidates.begin() + actual_k, candidates.end()
                    );

                    for (size_t ki = 0; ki < actual_k; ++ki) {
                        topk_distances[idx * rerank_k + ki] = candidates[ki].first;
                        topk_indices[idx * rerank_k + ki] = candidates[ki].second;
                    }
                    for (size_t ki = actual_k; ki < rerank_k; ++ki) {
                        topk_distances[idx * rerank_k + ki] =
                            std::numeric_limits<float>::max();
                        topk_indices[idx * rerank_k + ki] = static_cast<uint32_t>(-1);
                    }
                }
            }
        }

        // F32 reranking: compute exact L2² to top-k candidates, keep best
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_idx = 0;

            for (size_t ki = 0; ki < rerank_k; ++ki) {
                const uint32_t cand_idx = topk_indices[i * rerank_k + ki];
                if (cand_idx == static_cast<uint32_t>(-1)) break;

                const float dist =
                    f32_computer::Horizontal(x_float + i * d, y_float + cand_idx * d, d);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = cand_idx;
                }
            }

            out_knn[i] = best_idx;
            out_distances[i] = best_dist;
        }
    }

    bool IsFitted() const override { return fitted; }

    bool SupportsPruning() const override { return true; }

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t d, uint32_t partial_d
    ) override {
        const size_t d_packed = d / 2;
        const size_t partial_bytes = partial_d / 2;
        cached_data_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = data + idx * d_packed;
            for (size_t k = 0; k < partial_bytes; ++k) {
                uint32_t lo = row[k] & 0x0F;
                uint32_t hi = (row[k] >> 4) & 0x0F;
                sum += lo * lo + hi * hi;
            }
            cached_data_partial_norms[idx] = sum;
        }
    }

    void CacheCentroidPartialNorms(
        const quantized_t* centroids, size_t n, size_t d, uint32_t partial_d
    ) override {
        const size_t d_packed = d / 2;
        const size_t partial_bytes = partial_d / 2;
        cached_centroid_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = centroids + idx * d_packed;
            for (size_t k = 0; k < partial_bytes; ++k) {
                uint32_t lo = row[k] & 0x0F;
                uint32_t hi = (row[k] >> 4) & 0x0F;
                sum += lo * lo + hi * hi;
            }
            cached_centroid_partial_norms[idx] = sum;
        }
    }

    /**
     * @brief Find top-1 nearest neighbor with PDX pruning for u4.
     *
     * Uses partial u4 dot products via NumKong on the first partial_d dimensions,
     * converts to uint32_t L2² via norm expansion, then PDXearch prunes the rest.
     * Partial norms must be cached via CacheDataPartialNorms / CacheCentroidPartialNorms.
     *
     * Key difference from SQ8: data is packed u4x2 (d/2 bytes per vector).
     * NumKong functions use real dim count (partial_d), but PDX operates in packed
     * byte units (partial_d_packed = partial_d / 2).
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
        PDXLayout<Quantization::u4, DistanceFunction::l2>& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("search");
        assert(fitted);
        assert(d % 2 == 0);
        assert(partial_d % 2 == 0);
        (void)x_float;
        (void)y_float;
        assert(!cached_data_partial_norms.empty() && "CacheDataPartialNorms must be called first");
        assert(
            !cached_centroid_partial_norms.empty() &&
            "CacheCentroidPartialNorms must be called first"
        );

        using u4_computer = DistanceComputer<DistanceFunction::l2, Quantization::u4>;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        const size_t d_packed = d / 2;
        const size_t partial_d_packed = partial_d / 2;

        const auto* x_u4 = reinterpret_cast<const nk_u4x2_t*>(x);
        const auto* y_u4 = reinterpret_cast<const nk_u4x2_t*>(y);

        // Set scale factors on the PDX index for threshold conversion
        pdx_centroids.index->quantization_scale_squared =
            params.quantization_scale * params.quantization_scale;
        pdx_centroids.index->inverse_scale_factor_squared = inv_scale_sq;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("search/blas");
                    // Pack y for partial_d dimensions (stride = d_packed for packed rows)
                    const size_t packed_size = nk_dots_packed_size_u4(batch_n_y, partial_d);
                    if (packed_size > packed_buf.size()) packed_buf.resize(packed_size);
                    nk_dots_pack_u4(
                        y_u4 + j * d_packed, batch_n_y, partial_d, d_packed, packed_buf.data()
                    );

                    const size_t c_stride = batch_n_y * sizeof(uint32_t);

                    // Compute partial u4 dot products via NumKong
#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_n_x - start);
                        if (start < batch_n_x && count > 0) {
                            nk_dots_packed_u4(
                                x_u4 + (i + start) * d_packed,
                                packed_buf.data(),
                                pruning_dots_buf.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                partial_d,
                                d_packed,
                                c_stride
                            );
                        }
                    }
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
                        uint32_t* dots_row = pruning_dots_buf.data() + r * batch_n_y;

                        // Convert dot products to L2²: l2 = norm_x + norm_y - 2*dot
                        const uint32_t nx = cached_data_partial_norms[i_idx];
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            dots_row[c] =
                                nx + cached_centroid_partial_norms[j + c] - 2 * dots_row[c];
                        }

                        // Initial threshold: compute u4 packed L2² to previous assignment
                        const auto prev_assignment = out_knn[i_idx];
                        float dist_to_prev_centroid;
                        if (j == 0) {
                            uint32_t u4_dist = u4_computer::Horizontal(
                                x + i_idx * d_packed, y + prev_assignment * d_packed, d_packed
                            );
                            dist_to_prev_centroid = static_cast<float>(u4_dist) * inv_scale_sq;
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        // PDXearch with uint32_t partial distances
                        // PDX operates in packed byte units: partial_d_packed
                        size_t local_not_pruned = 0;
                        auto assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    x + i_idx * d_packed,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    dots_row,
                                    static_cast<uint32_t>(partial_d_packed),
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

    void AverageCentroids(
        const uint32_t* accumulators,
        const uint32_t* cluster_sizes,
        quantized_t* out,
        size_t n_clusters,
        size_t d
    ) const override {
        assert(fitted);
        assert(d % 2 == 0);
        const size_t d_packed = d / 2;
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_clusters; ++i) {
            if (cluster_sizes[i] == 0) continue;
            const uint32_t* acc = accumulators + i * d;
            quantized_t* row = out + i * d_packed;
            const uint32_t half = cluster_sizes[i] / 2;
            const float inv_size = 1.0f / static_cast<float>(cluster_sizes[i]);
            SKM_VECTORIZE_LOOP
            for (size_t k = 0; k < d_packed; ++k) {
                uint8_t lo = static_cast<uint8_t>(static_cast<float>(acc[2 * k] + half) * inv_size);
                uint8_t hi = static_cast<uint8_t>(static_cast<float>(acc[2 * k + 1] + half) * inv_size);
                row[k] = lo | (hi << 4);
            }
        }
    }

    size_t CodeSize(size_t d) const override {
        assert(d % 2 == 0);
        return d / 2;
    }

    const ScalarQuantizationParams& GetParams() const { return params; }

  private:
    ScalarQuantizationParams params{};
    bool fitted = false;
    std::vector<uint32_t> cached_data_partial_norms;
    std::vector<uint32_t> cached_centroid_partial_norms;
    mutable std::vector<float> nn_dists_buf;
    mutable std::vector<uint32_t> pruning_dots_buf;
    mutable std::vector<char> packed_buf;
};

} // namespace skmeans
