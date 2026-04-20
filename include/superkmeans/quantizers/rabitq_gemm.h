#pragma once

// #ifdef HAS_FAISS

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/impl/RaBitQuantizer.h>
#include <numkong/numkong.h>

namespace skmeans {

/**
 * @brief RaBitQ quantizer with NumKong binary GEMM distance kernel.
 *
 * Uses FAISS RaBitQuantizer for Fit/Encode/Decode.
 * Replaces FAISS's one-at-a-time distance_to_code with batched binary GEMM
 * via NumKong nk_dots_packed_u1, following the SQ4 batching pattern.
 *
 * For k-means: centroids (K, small) are SQ-quantized to qb bits and
 * bit-transposed once. Data points (N, large) stay as binary codes.
 * The binary GEMM computes popcount(code AND centroid_plane) for all N×K
 * pairs, then applies the RaBitQ correction formula.
 */
template <Quantization q>
class RaBitQGemmQuantizer : public IQuantizer<q> {
    static_assert(q == Quantization::u8, "RaBitQGemmQuantizer only supports u8");

  public:
    using quantized_t = typename IQuantizer<q>::quantized_t;

    void Fit(const float* data, size_t n, size_t d) override {
        d_ = d;
        binary_bytes_ = (d + 7) / 8;
        centroid_.resize(d, 0.0f);

        // Compute dataset mean as the centering centroid for RaBitQ
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0;
            for (size_t i = 0; i < n; ++i) {
                sum += data[i * d + j];
            }
            centroid_[j] = static_cast<float>(sum / static_cast<double>(n));
        }

        faiss_quantizer_ = std::make_unique<faiss::RaBitQuantizer>(d, faiss::METRIC_L2);
        faiss_quantizer_->centroid = centroid_.data();
        faiss_code_size_ = faiss_quantizer_->code_size;
        assert(faiss_code_size_ == CodeSize(d));
        fitted_ = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->compute_codes(in, reinterpret_cast<uint8_t*>(out), n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->decode(reinterpret_cast<const uint8_t*>(in), out, n);
    }

    void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code =
                reinterpret_cast<const uint8_t*>(data) + i * faiss_code_size_;
            const auto* factors =
                reinterpret_cast<const RaBitQFactors*>(code + binary_bytes_);
            out_norms[i] = factors->or_minus_c_l2sqr;
        }
    }

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
        assert(fitted_);
        (void)y;
        (void)x_float;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);

        // Precompute per-code factors
        std::vector<uint32_t> sum_q(n_x);
        std::vector<float> or_c_l2sqr(n_x);
        std::vector<float> dp_mult(n_x);
        PrecomputeCodeFactors(x_codes, n_x, sum_q.data(), or_c_l2sqr.data(), dp_mult.data());

        // Quantize centroids → bit planes + correction factors
        std::vector<uint8_t> query_planes(qb_ * n_y * binary_bytes_, 0);
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        QuantizeCentroids(
            y_float, n_y, d,
            query_planes.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data()
        );

        // Batched binary GEMM
        std::vector<uint32_t> dot_qo_buf(X_BATCH_SIZE * Y_BATCH_SIZE, 0);
        std::vector<uint32_t> partial_buf(X_BATCH_SIZE * Y_BATCH_SIZE, 0);
        const size_t max_packed_size = nk_dots_packed_size_u1(Y_BATCH_SIZE, d);
        std::vector<char> y_packed(max_packed_size);

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_nx = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_ny = std::min(Y_BATCH_SIZE, n_y - j);
                const size_t batch_c_stride = batch_ny * sizeof(uint32_t);

                // Clear accumulated dot product buffer
                std::memset(dot_qo_buf.data(), 0, batch_nx * batch_ny * sizeof(uint32_t));

                for (int b = 0; b < qb_; ++b) {
                    // Pack centroid bit plane b for this Y batch
                    const uint8_t* plane_b = query_planes.data() +
                        b * n_y * binary_bytes_ + j * binary_bytes_;

                    const size_t packed_size = nk_dots_packed_size_u1(batch_ny, d);
                    if (packed_size > y_packed.size()) y_packed.resize(packed_size);
                    nk_dots_pack_u1(
                        reinterpret_cast<const nk_u1x8_t*>(plane_b),
                        batch_ny, d, binary_bytes_, y_packed.data()
                    );

                    // Binary GEMM: partial[r,c] = popcount(code[i+r] AND plane_b[j+c])
#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_nx + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_nx - start);
                        if (start < batch_nx && count > 0) {
                            nk_dots_packed_u1(
                                reinterpret_cast<const nk_u1x8_t*>(
                                    x_codes + (i + start) * faiss_code_size_),
                                y_packed.data(),
                                partial_buf.data() + start * batch_ny,
                                count,
                                batch_ny,
                                d,
                                faiss_code_size_,
                                batch_c_stride
                            );
                        }
                    }

                    // Accumulate: dot_qo += partial << b
                    const size_t total = batch_nx * batch_ny;
#pragma omp parallel for num_threads(g_n_threads)
                    for (size_t k = 0; k < total; ++k) {
                        dot_qo_buf[k] += partial_buf[k] << b;
                    }
                }

                // Apply RaBitQ corrections + update argmin
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_nx; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dq_row = dot_qo_buf.data() + r * batch_ny;

                    for (size_t c = 0; c < batch_ny; ++c) {
                        const size_t cidx = j + c;
                        const float final_dot =
                            c1[cidx] * static_cast<float>(dq_row[c]) +
                            c2[cidx] * static_cast<float>(sum_q[idx]) -
                            c34[cidx];
                        const float dist =
                            or_c_l2sqr[idx] + qr_to_c_l2sqr[cidx] -
                            2.0f * dp_mult[idx] * final_dot;

                        if (dist < out_distances[idx]) {
                            out_distances[idx] = dist;
                            out_knn[idx] = static_cast<uint32_t>(cidx);
                        }
                    }
                }
            }
        }
    }

    size_t DefaultRerankK() const override { return 0; }

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
        assert(fitted_);
        (void)y_quantized;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        using f32_computer = DistanceComputer<DistanceFunction::l2, Quantization::f32>;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x_quantized);

        // Precompute per-code factors
        std::vector<uint32_t> sum_q(n_x);
        std::vector<float> or_c_l2sqr(n_x);
        std::vector<float> dp_mult(n_x);
        PrecomputeCodeFactors(x_codes, n_x, sum_q.data(), or_c_l2sqr.data(), dp_mult.data());

        // Quantize centroids
        std::vector<uint8_t> query_planes(qb_ * n_y * binary_bytes_, 0);
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        QuantizeCentroids(
            y_float, n_y, d,
            query_planes.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data()
        );

        // Buffers
        std::vector<uint32_t> dot_qo_buf(X_BATCH_SIZE * Y_BATCH_SIZE, 0);
        std::vector<uint32_t> partial_buf(X_BATCH_SIZE * Y_BATCH_SIZE, 0);
        const size_t max_packed_size = nk_dots_packed_size_u1(Y_BATCH_SIZE, d);
        std::vector<char> y_packed(max_packed_size);

        // Per-thread candidate buffers for top-k merge
        const uint32_t num_threads = g_n_threads;
        const size_t max_candidates = rerank_k + Y_BATCH_SIZE;
        std::vector<std::vector<std::pair<float, uint32_t>>> thread_candidates(num_threads);
        for (auto& tc : thread_candidates) {
            tc.reserve(max_candidates);
        }

        // Per-query top-k
        std::vector<float> topk_distances(n_x * rerank_k, std::numeric_limits<float>::max());
        std::vector<uint32_t> topk_indices(n_x * rerank_k, static_cast<uint32_t>(-1));

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_nx = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_ny = std::min(Y_BATCH_SIZE, n_y - j);
                const size_t batch_c_stride = batch_ny * sizeof(uint32_t);

                std::memset(dot_qo_buf.data(), 0, batch_nx * batch_ny * sizeof(uint32_t));

                for (int b = 0; b < qb_; ++b) {
                    const uint8_t* plane_b = query_planes.data() +
                        b * n_y * binary_bytes_ + j * binary_bytes_;

                    const size_t packed_size = nk_dots_packed_size_u1(batch_ny, d);
                    if (packed_size > y_packed.size()) y_packed.resize(packed_size);
                    nk_dots_pack_u1(
                        reinterpret_cast<const nk_u1x8_t*>(plane_b),
                        batch_ny, d, binary_bytes_, y_packed.data()
                    );

#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_nx + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_nx - start);
                        if (start < batch_nx && count > 0) {
                            nk_dots_packed_u1(
                                reinterpret_cast<const nk_u1x8_t*>(
                                    x_codes + (i + start) * faiss_code_size_),
                                y_packed.data(),
                                partial_buf.data() + start * batch_ny,
                                count,
                                batch_ny,
                                d,
                                faiss_code_size_,
                                batch_c_stride
                            );
                        }
                    }

                    const size_t total = batch_nx * batch_ny;
#pragma omp parallel for num_threads(g_n_threads)
                    for (size_t k = 0; k < total; ++k) {
                        dot_qo_buf[k] += partial_buf[k] << b;
                    }
                }

                // Compute distances and merge into per-query top-k
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_nx; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dq_row = dot_qo_buf.data() + r * batch_ny;

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
                    for (size_t c = 0; c < batch_ny; ++c) {
                        const size_t cidx = j + c;
                        const float final_dot =
                            c1[cidx] * static_cast<float>(dq_row[c]) +
                            c2[cidx] * static_cast<float>(sum_q[idx]) -
                            c34[cidx];
                        const float dist =
                            or_c_l2sqr[idx] + qr_to_c_l2sqr[cidx] -
                            2.0f * dp_mult[idx] * final_dot;
                        candidates.emplace_back(dist, static_cast<uint32_t>(cidx));
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

        // F32 reranking
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

    size_t CodeSize(size_t d) const override {
        return (d + 7) / 8 + sizeof(RaBitQFactors);
    }

    bool IsFitted() const override { return fitted_; }

  private:
    /// Extract per-code metadata: popcount, or_c_l2sqr, dp_multiplier.
    void PrecomputeCodeFactors(
        const uint8_t* codes, size_t n,
        uint32_t* sum_q, float* or_c_l2sqr, float* dp_mult
    ) const {
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * faiss_code_size_;

            // Popcount of the binary part
            uint32_t pc = 0;
            size_t b = 0;
            for (; b + 8 <= binary_bytes_; b += 8) {
                uint64_t word;
                std::memcpy(&word, code + b, 8);
                pc += static_cast<uint32_t>(__builtin_popcountll(word));
            }
            for (; b < binary_bytes_; ++b) {
                pc += static_cast<uint32_t>(__builtin_popcount(code[b]));
            }
            sum_q[i] = pc;

            const auto* fac = reinterpret_cast<const RaBitQFactors*>(code + binary_bytes_);
            or_c_l2sqr[i] = fac->or_minus_c_l2sqr;
            dp_mult[i] = fac->dp_multiplier;
        }
    }

    /// Batch SQ-quantize centroids to qb bits and bit-transpose into planes.
    void QuantizeCentroids(
        const float* y_float, size_t n_y, size_t d,
        uint8_t* query_planes,
        float* c1, float* c2, float* c34, float* qr_to_c_l2sqr
    ) const {
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float max_val = static_cast<float>((1 << qb_) - 1);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < n_y; ++j) {
            std::vector<float> rotated(d);
            float v_min = std::numeric_limits<float>::max();
            float v_max = std::numeric_limits<float>::lowest();
            float norm_sq = 0;

            for (size_t dim = 0; dim < d; ++dim) {
                rotated[dim] = y_float[j * d + dim] - centroid_[dim];
                v_min = std::min(v_min, rotated[dim]);
                v_max = std::max(v_max, rotated[dim]);
                norm_sq += rotated[dim] * rotated[dim];
            }
            qr_to_c_l2sqr[j] = norm_sq;

            float delta = (v_max - v_min) / max_val;
            if (delta < std::numeric_limits<float>::epsilon()) delta = 1.0f;
            const float inv_delta = 1.0f / delta;
            float sum_qq = 0;

            std::vector<uint8_t> quantized(d);
            for (size_t dim = 0; dim < d; ++dim) {
                int v = static_cast<int>(std::lround((rotated[dim] - v_min) * inv_delta));
                v = std::max(0, std::min(v, static_cast<int>(max_val)));
                quantized[dim] = static_cast<uint8_t>(v);
                sum_qq += static_cast<float>(v);
            }

            c1[j] = 2.0f * delta * inv_sqrt_d;
            c2[j] = 2.0f * v_min * inv_sqrt_d;
            c34[j] = inv_sqrt_d * (delta * sum_qq + static_cast<float>(d) * v_min);

            // Bit-transpose into qb planes
            for (int b = 0; b < qb_; ++b) {
                uint8_t* plane = query_planes + (b * n_y + j) * binary_bytes_;
                std::memset(plane, 0, binary_bytes_);
                for (size_t dim = 0; dim < d; ++dim) {
                    if ((quantized[dim] >> b) & 1) {
                        plane[dim / 8] |= static_cast<uint8_t>(1 << (dim % 8));
                    }
                }
            }
        }
    }

    int qb_ = 4;
    size_t d_ = 0;
    size_t binary_bytes_ = 0;
    size_t faiss_code_size_ = 0;
    std::vector<float> centroid_;
    std::unique_ptr<faiss::RaBitQuantizer> faiss_quantizer_;
    bool fitted_ = false;
};

} // namespace skmeans

// #endif // HAS_FAISS
