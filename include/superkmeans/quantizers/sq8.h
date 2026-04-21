#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/quantizers/quantizer.h"

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

struct ScalarQuantizationParams {
    float quantization_base;
    float quantization_scale;
    float inv_quantization_scale;
};

/**
 * @brief 8-bit scalar quantizer with NumKong GEMM backend.
 *
 * Global min/max quantization: q[i] = round((val[i] - base) * scale), clamped to [0, MAX_VALUE].
 * For L2 distance the base cancels: ||x-y||² = inv_scale² * Σ(x_q - y_q)².
 */
template <Quantization q>
class SQ8Quantizer : public IQuantizer<q> {
    static_assert(q == Quantization::u8, "SQ8Quantizer only supports u8");

  public:
    using quantized_t = typename IQuantizer<q>::quantized_t;

    static constexpr uint8_t MAX_VALUE = 255;

    void Fit(const float* embeddings, size_t n, size_t d) override {
        const size_t total_elements = n * d;
        params = ComputeQuantizationParams(embeddings, total_elements);
        // Pre-allocate scratch buffers (avoids expensive per-call allocation)
        nn_dists_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        pruning_dots_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        packed_buf.resize(nk_dots_packed_size_u8(Y_BATCH_SIZE, d));
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

    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
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
    void ComputeNorms(
        const quantized_t* quantized_embeddings, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted);
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

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
     * @brief Find top-1 nearest neighbor using NumKong u8 Euclidean kernel.
     *
     * Packs centroids (y) with nk_dots_pack_u8 (norms stored in footer),
     * then nk_euclideans_packed_u8 computes L2 distances directly via the
     * inner-product expansion: sqrt(Σx_q² + Σy_q² - 2·dot(x_q, y_q)).
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
        (void)x_float;
        (void)y_float;
        (void)norms_x; // nk_euclideans_packed_u8 computes norms internally
        (void)norms_y;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                const size_t packed_size = nk_dots_packed_size_u8(batch_n_y, d);
                if (packed_size > packed_buf.size()) packed_buf.resize(packed_size);
                nk_dots_pack_u8(y + j * d, batch_n_y, d, d, packed_buf.data());

                const size_t batch_r_stride = batch_n_y * sizeof(float);

                // Compute u8 Euclidean distances via NumKong
#pragma omp parallel num_threads(g_n_threads)
                {
                    nk_configure_thread(nk_capabilities());
                    int tid = omp_get_thread_num();
                    int nt = omp_get_num_threads();
                    size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                    size_t start = tid * rows_per_t;
                    size_t count = std::min(rows_per_t, batch_n_x - start);
                    if (start < batch_n_x && count > 0) {
                        nk_euclideans_packed_u8(
                            x + (i + start) * d,
                            packed_buf.data(),
                            nn_dists_buf.data() + start * batch_n_y,
                            count,
                            batch_n_y,
                            d,
                            d,
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
        // nk_euclideans_packed_u8 outputs sqrt(Σ(x_q-y_q)²);
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
     * 1. Find top rerank_k candidates per query in quantized space (NumKong u8)
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
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        using f32_computer = DistanceComputer<DistanceFunction::l2, Quantization::f32>;

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

                const size_t packed_size = nk_dots_packed_size_u8(batch_n_y, d);
                if (packed_size > packed_buf.size()) packed_buf.resize(packed_size);
                nk_dots_pack_u8(y_quantized + j * d, batch_n_y, d, d, packed_buf.data());

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
                        nk_euclideans_packed_u8(
                            x_quantized + (i + start) * d,
                            packed_buf.data(),
                            nn_dists_buf.data() + start * batch_n_y,
                            count,
                            batch_n_y,
                            d,
                            d,
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
        cached_data_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = data + idx * d;
            SKM_VECTORIZE_LOOP
            for (size_t dim = 0; dim < partial_d; ++dim) {
                uint32_t v = row[dim];
                sum += v * v;
            }
            cached_data_partial_norms[idx] = sum;
        }
    }

    void CacheCentroidPartialNorms(
        const quantized_t* centroids, size_t n, size_t d, uint32_t partial_d
    ) override {
        cached_centroid_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = centroids + idx * d;
            SKM_VECTORIZE_LOOP
            for (size_t dim = 0; dim < partial_d; ++dim) {
                uint32_t v = row[dim];
                sum += v * v;
            }
            cached_centroid_partial_norms[idx] = sum;
        }
    }

    /**
     * @brief Find top-1 nearest neighbor with PDX pruning for u8.
     *
     * Uses partial u8 dot products via NumKong on the first partial_d dimensions,
     * converts to uint32_t L2² via norm expansion, then PDXearch prunes the rest.
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
        PDXLayout<q, DistanceFunction::l2>& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("search");
        assert(fitted);
        (void) x_float;
        (void) y_float;
        assert(!cached_data_partial_norms.empty() && "CacheDataPartialNorms must be called first");
        assert(
            !cached_centroid_partial_norms.empty() &&
            "CacheCentroidPartialNorms must be called first"
        );

        using u8_computer = DistanceComputer<DistanceFunction::l2, q>;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

        // Set scale factors on the PDX index for threshold conversion
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
                    // Pack y for partial_d dimensions (stride = d for full vectors)
                    const size_t packed_size = nk_dots_packed_size_u8(batch_n_y, partial_d);
                    if (packed_size > packed_buf.size()) packed_buf.resize(packed_size);
                    nk_dots_pack_u8(y + j * d, batch_n_y, partial_d, d, packed_buf.data());

                    const size_t c_stride = batch_n_y * sizeof(uint32_t);

                    // Compute partial u8 dot products via NumKong
#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_n_x - start);
                        if (start < batch_n_x && count > 0) {
                            nk_dots_packed_u8(
                                x + (i + start) * d,
                                packed_buf.data(),
                                pruning_dots_buf.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                partial_d,
                                d,
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

                        // Initial threshold: compute u8 L2² to previous assignment, scale to float
                        const auto prev_assignment = out_knn[i_idx];
                        float dist_to_prev_centroid;
                        if (j == 0) {
                            uint32_t u8_dist = u8_computer::Horizontal(
                                x + i_idx * d, y + prev_assignment * d, d
                            );
                            dist_to_prev_centroid = static_cast<float>(u8_dist) * inv_scale_sq;
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        // PDXearch with uint32_t partial distances
                        size_t local_not_pruned = 0;
                        auto assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    x + i_idx * d,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    dots_row,
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
