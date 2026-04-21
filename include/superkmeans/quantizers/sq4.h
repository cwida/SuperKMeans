#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
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
 * Internally, the IQuantizer interface stores one u8 per dimension (values in [0,15]).
 * The NumKong u4 kernels expect nk_u4x2_t (two nibbles per byte), so we pack on the fly
 * before calling the distance kernels.
 *
 * Requires d to be even.
 */
template <Quantization q>
class SQ4Quantizer : public IQuantizer<q> {
    static_assert(q == Quantization::u8, "SQ4Quantizer only supports u8");

  public:
    using quantized_t = typename IQuantizer<q>::quantized_t;

    static constexpr uint8_t MAX_VALUE = 15;

    void Fit(const float* embeddings, size_t n, size_t d) override {
        assert(d % 2 == 0 && "SQ4Quantizer requires even dimensionality");
        const size_t total_elements = n * d;
        params = ComputeQuantizationParams(embeddings, total_elements);
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
     * @brief Find top-1 nearest neighbor using NumKong u4 Euclidean kernel.
     *
     * Converts u8 data ([0,15], one per byte) to nk_u4x2_t (two per byte),
     * packs centroids (y) with nk_dots_pack_u4, then nk_euclideans_packed_u4
     * computes L2 distances via inner-product expansion.
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
        assert(fitted);
        assert(d % 2 == 0);
        (void)x_float;
        (void)y_float;
        (void)norms_x;
        (void)norms_y;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        const size_t d_packed = d / 2; // bytes per vector in u4x2 format

        // Bounded distance buffer: X_BATCH_SIZE × Y_BATCH_SIZE floats
        std::vector<float> dists_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

        // u4x2 conversion buffers
        std::vector<nk_u4x2_t> x_u4(X_BATCH_SIZE * d_packed);
        std::vector<nk_u4x2_t> y_u4(Y_BATCH_SIZE * d_packed);

        // Pack buffer — reused per Y-batch
        const size_t max_packed_size = nk_dots_packed_size_u4(Y_BATCH_SIZE, d);
        std::vector<char> y_packed(max_packed_size);

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            // Convert X batch to u4x2
            PackToU4x2(x + i * d, x_u4.data(), batch_n_x, d);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                // Convert Y batch to u4x2 and pack for NumKong
                PackToU4x2(y + j * d, y_u4.data(), batch_n_y, d);

                const size_t packed_size = nk_dots_packed_size_u4(batch_n_y, d);
                if (packed_size > y_packed.size()) y_packed.resize(packed_size);
                nk_dots_pack_u4(y_u4.data(), batch_n_y, d, d_packed, y_packed.data());

                const size_t batch_r_stride = batch_n_y * sizeof(float);

                // Compute u4 Euclidean distances via NumKong
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
                            x_u4.data() + start * d_packed,
                            y_packed.data(),
                            dists_buf.data() + start * batch_n_y,
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
                    const float* dists_row = dists_buf.data() + r * batch_n_y;

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
        assert(fitted);
        assert(d % 2 == 0);
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        using f32_computer = DistanceComputer<DistanceFunction::l2, Quantization::f32>;

        const size_t d_packed = d / 2;

        // Bounded distance buffer: X_BATCH_SIZE × Y_BATCH_SIZE floats
        std::vector<float> dists_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

        // u4x2 conversion buffers
        std::vector<nk_u4x2_t> x_u4(X_BATCH_SIZE * d_packed);
        std::vector<nk_u4x2_t> y_u4(Y_BATCH_SIZE * d_packed);

        // Pack buffer — reused per Y-batch
        const size_t max_packed_size = nk_dots_packed_size_u4(Y_BATCH_SIZE, d);
        std::vector<char> y_packed(max_packed_size);

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

            // Convert X batch to u4x2
            PackToU4x2(x_quantized + i * d, x_u4.data(), batch_n_x, d);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                // Convert Y batch to u4x2 and pack for NumKong
                PackToU4x2(y_quantized + j * d, y_u4.data(), batch_n_y, d);

                const size_t packed_size = nk_dots_packed_size_u4(batch_n_y, d);
                if (packed_size > y_packed.size()) y_packed.resize(packed_size);
                nk_dots_pack_u4(y_u4.data(), batch_n_y, d, d_packed, y_packed.data());

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
                            x_u4.data() + start * d_packed,
                            y_packed.data(),
                            dists_buf.data() + start * batch_n_y,
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
                    const float* dists_row = dists_buf.data() + r * batch_n_y;

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

    const ScalarQuantizationParams& GetParams() const { return params; }

    /**
     * @brief Convert u8 array (1 value per byte, [0,15]) to nk_u4x2_t (2 nibbles per byte).
     *
     * Layout: byte[k] = (src[2k] & 0x0F) | ((src[2k+1] & 0x0F) << 4)
     * i.e. even-indexed dim → low nibble, odd-indexed dim → high nibble.
     */
    static void PackToU4x2(
        const quantized_t* src, nk_u4x2_t* dst, size_t n, size_t d
    ) {
        const size_t d_packed = d / 2;
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const quantized_t* src_row = src + row * d;
            nk_u4x2_t* dst_row = dst + row * d_packed;
            for (size_t k = 0; k < d_packed; ++k) {
                dst_row[k] = static_cast<nk_u4x2_t>(
                    (src_row[2 * k] & 0x0F) | ((src_row[2 * k + 1] & 0x0F) << 4)
                );
            }
        }
    }

  private:
    ScalarQuantizationParams params{};
    bool fitted = false;
};

} // namespace skmeans
