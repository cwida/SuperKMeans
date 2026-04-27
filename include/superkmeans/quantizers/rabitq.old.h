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

namespace skmeans {

/**
 * @brief RaBitQ quantizer with SIMD-accelerated binary distance kernel.
 *
 * Uses FAISS RaBitQuantizer for Fit/Encode/Decode.
 * Distance computation uses DistanceComputer<l2, b8> (popcount-based)
 * for 1:1 binary inner products between data codes and centroid bit planes.
 *
 * For k-means: centroids (K, small) are SQ-quantized to qb bits and
 * bit-transposed once. Data points (N, large) stay as binary codes.
 * For each (data, centroid) pair, computes weighted popcount across qb
 * bit planes, then applies the RaBitQ correction formula.
 */
class RaBitQQuantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    void Fit(const float* data, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("RaBitQ::Fit");
        assert(d % 8 == 0 && "RaBitQ-GEMM requires dimensionality divisible by 8");
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
        SKM_PROFILE_SCOPE("RaBitQ::Encode");
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->compute_codes(in, reinterpret_cast<uint8_t*>(out), n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("RaBitQ::Decode");
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->decode(reinterpret_cast<const uint8_t*>(in), out, n);
    }

    void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const override {
        SKM_PROFILE_SCOPE("RaBitQ::ComputeNorms");

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

        using b8_computer = DistanceComputer<DistanceFunction::l2, Quantization::b8>;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        {
            SKM_PROFILE_SCOPE("RaBitQ::BinaryDistance");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t i = 0; i < n_x; ++i) {
                const uint8_t* code = x_codes + i * faiss_code_size_;
                float best_dist = std::numeric_limits<float>::max();
                uint32_t best_idx = 0;

                for (size_t j = 0; j < n_y; ++j) {
                    uint32_t dot_qo = 0;
                    for (int b = 0; b < qb_; ++b) {
                        const uint8_t* plane =
                            query_planes.data() + (b * n_y + j) * binary_bytes_;
                        dot_qo += b8_computer::Horizontal(code, plane, binary_bytes_) << b;
                    }
                    const float final_dot =
                        c1[j] * static_cast<float>(dot_qo) +
                        c2[j] * static_cast<float>(sum_q[i]) -
                        c34[j];
                    const float dist =
                        or_c_l2sqr[i] + qr_to_c_l2sqr[j] -
                        2.0f * dp_mult[i] * final_dot;

                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx = static_cast<uint32_t>(j);
                    }
                }
                out_distances[i] = best_dist;
                out_knn[i] = best_idx;
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

        using b8_computer = DistanceComputer<DistanceFunction::l2, Quantization::b8>;

        // Per-query top-k from quantized distances
        std::vector<float> topk_distances(n_x * rerank_k, std::numeric_limits<float>::max());
        std::vector<uint32_t> topk_indices(n_x * rerank_k, static_cast<uint32_t>(-1));

        {
            SKM_PROFILE_SCOPE("RaBitQ::BinaryDistance");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t i = 0; i < n_x; ++i) {
                const uint8_t* code = x_codes + i * faiss_code_size_;
                std::vector<std::pair<float, uint32_t>> candidates(n_y);

                for (size_t j = 0; j < n_y; ++j) {
                    uint32_t dot_qo = 0;
                    for (int b = 0; b < qb_; ++b) {
                        const uint8_t* plane =
                            query_planes.data() + (b * n_y + j) * binary_bytes_;
                        dot_qo += b8_computer::Horizontal(code, plane, binary_bytes_) << b;
                    }
                    const float final_dot =
                        c1[j] * static_cast<float>(dot_qo) +
                        c2[j] * static_cast<float>(sum_q[i]) -
                        c34[j];
                    const float dist =
                        or_c_l2sqr[i] + qr_to_c_l2sqr[j] -
                        2.0f * dp_mult[i] * final_dot;
                    candidates[j] = {dist, static_cast<uint32_t>(j)};
                }

                size_t actual_k = std::min(rerank_k, n_y);
                std::partial_sort(
                    candidates.begin(), candidates.begin() + actual_k, candidates.end()
                );
                for (size_t ki = 0; ki < actual_k; ++ki) {
                    topk_distances[i * rerank_k + ki] = candidates[ki].first;
                    topk_indices[i * rerank_k + ki] = candidates[ki].second;
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
        SKM_PROFILE_SCOPE("RaBitQ::PrecomputeCodeFactors");
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
        SKM_PROFILE_SCOPE("RaBitQ::QuantizeCentroids");
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
