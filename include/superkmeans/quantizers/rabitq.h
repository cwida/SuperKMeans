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

struct RaBitQFactors {
    float or_minus_c_l2sqr; // ||original - centroid||²  (for L2 metric)
    float dp_multiplier;    // scaling factor for dot-product estimation
};
static_assert(sizeof(RaBitQFactors) == 8, "RaBitQFactors must match FAISS FactorsData");

/**
 * @brief RaBitQ quantizer with FastScan-accelerated distance kernel.
 *
 * Uses FAISS RaBitQuantizer for Fit/Encode/Decode.
 * Distance computation uses a custom FastScan kernel (VPSHUFB/TBL lookups)
 * that processes 32 data points simultaneously per centroid.
 *
 * For k-means: centroids (K, small) are SQ-quantized to qb bits once,
 * then LUTs are built from the quantized values. Data points (N, large)
 * are byte-transposed into blocks of 32 for SIMD lookup.
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

        // Quantize centroids and build LUTs
        const size_t n_sub = 2 * binary_bytes_; // 2 sub-quantizers per byte position
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        QuantizeCentroidsAndBuildLUTs(
            y_float, n_y, d,
            all_luts.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data()
        );

        const size_t lut_stride = n_sub * 16;
        const size_t n_blocks = (n_x + FastScanComputer::kBlockSize - 1) / FastScanComputer::kBlockSize;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        {
            SKM_PROFILE_SCOPE("RaBitQ::FastScanDistance");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t blk = 0; blk < n_blocks; ++blk) {
                const size_t blk_start = blk * FastScanComputer::kBlockSize;
                const size_t blk_count = std::min(FastScanComputer::kBlockSize, n_x - blk_start);

                // Byte-transpose this block's binary codes
                std::unique_ptr<uint8_t[]> packed(new uint8_t[FastScanComputer::kBlockSize * binary_bytes_]);
                TransposeBlock(x_codes, blk_start, blk_count, packed.get());

                float best_dist[FastScanComputer::kBlockSize];
                uint32_t best_idx[FastScanComputer::kBlockSize];
                std::fill_n(best_dist, FastScanComputer::kBlockSize, std::numeric_limits<float>::max());
                std::fill_n(best_idx, FastScanComputer::kBlockSize, 0u);

                for (size_t j = 0; j < n_y; ++j) {
                    uint16_t dot_qo[FastScanComputer::kBlockSize];
                    FastScanComputer::ScanBlock(
                        packed.get(), all_luts.data() + j * lut_stride,
                        binary_bytes_, dot_qo, blk_count
                    );

                    for (size_t k = 0; k < blk_count; ++k) {
                        const size_t i = blk_start + k;
                        const float final_dot =
                            c1[j] * static_cast<float>(dot_qo[k]) +
                            c2[j] * static_cast<float>(sum_q[i]) -
                            c34[j];
                        const float dist =
                            or_c_l2sqr[i] + qr_to_c_l2sqr[j] -
                            2.0f * dp_mult[i] * final_dot;

                        if (dist < best_dist[k]) {
                            best_dist[k] = dist;
                            best_idx[k] = static_cast<uint32_t>(j);
                        }
                    }
                }

                for (size_t k = 0; k < blk_count; ++k) {
                    out_distances[blk_start + k] = best_dist[k];
                    out_knn[blk_start + k] = best_idx[k];
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

        // Quantize centroids and build LUTs
        const size_t n_sub = 2 * binary_bytes_;
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        QuantizeCentroidsAndBuildLUTs(
            y_float, n_y, d,
            all_luts.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data()
        );

        const size_t lut_stride = n_sub * 16;
        const size_t n_blocks = (n_x + FastScanComputer::kBlockSize - 1) / FastScanComputer::kBlockSize;

        // Per-query top-k from quantized distances
        std::vector<float> topk_distances(n_x * rerank_k, std::numeric_limits<float>::max());
        std::vector<uint32_t> topk_indices(n_x * rerank_k, static_cast<uint32_t>(-1));

        {
            SKM_PROFILE_SCOPE("RaBitQ::FastScanDistance");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t blk = 0; blk < n_blocks; ++blk) {
                const size_t blk_start = blk * FastScanComputer::kBlockSize;
                const size_t blk_count = std::min(FastScanComputer::kBlockSize, n_x - blk_start);

                std::unique_ptr<uint8_t[]> packed(new uint8_t[FastScanComputer::kBlockSize * binary_bytes_]);
                TransposeBlock(x_codes, blk_start, blk_count, packed.get());

                // Per-point candidate heaps (small vectors on stack)
                std::vector<std::pair<float, uint32_t>> candidates[FastScanComputer::kBlockSize];
                for (size_t k = 0; k < blk_count; ++k) {
                    candidates[k].reserve(n_y);
                }

                for (size_t j = 0; j < n_y; ++j) {
                    uint16_t dot_qo[FastScanComputer::kBlockSize];
                    FastScanComputer::ScanBlock(
                        packed.get(), all_luts.data() + j * lut_stride,
                        binary_bytes_, dot_qo, blk_count
                    );

                    for (size_t k = 0; k < blk_count; ++k) {
                        const size_t i = blk_start + k;
                        const float final_dot =
                            c1[j] * static_cast<float>(dot_qo[k]) +
                            c2[j] * static_cast<float>(sum_q[i]) -
                            c34[j];
                        const float dist =
                            or_c_l2sqr[i] + qr_to_c_l2sqr[j] -
                            2.0f * dp_mult[i] * final_dot;
                        candidates[k].push_back({dist, static_cast<uint32_t>(j)});
                    }
                }

                for (size_t k = 0; k < blk_count; ++k) {
                    const size_t i = blk_start + k;
                    size_t actual_k = std::min(rerank_k, n_y);
                    std::partial_sort(
                        candidates[k].begin(),
                        candidates[k].begin() + actual_k,
                        candidates[k].end()
                    );
                    for (size_t ki = 0; ki < actual_k; ++ki) {
                        topk_distances[i * rerank_k + ki] = candidates[k][ki].first;
                        topk_indices[i * rerank_k + ki] = candidates[k][ki].second;
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

    /// SQ-quantize centroids to qb bits and build 16-entry LUTs for FastScan.
    ///
    /// Each centroid produces d SQ-quantized uint8 values.
    /// Sub-quantizer m corresponds to 4 consecutive dimensions (4m..4m+3).
    /// LUT[m][c] = sum of quantized values at dimensions where bit k is set in c.
    ///
    /// Layout: lut[j * n_sub * 16 + m * 16 + c] for centroid j, sub-quantizer m, code c.
    /// Sub-quantizers are ordered: byte 0 low nibble, byte 0 high nibble,
    ///                             byte 1 low nibble, byte 1 high nibble, ...
    void QuantizeCentroidsAndBuildLUTs(
        const float* y_float, size_t n_y, size_t d,
        uint8_t* all_luts,
        float* c1, float* c2, float* c34, float* qr_to_c_l2sqr
    ) const {
        SKM_PROFILE_SCOPE("RaBitQ::QuantizeCentroidsAndBuildLUTs");
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float max_val = static_cast<float>((1 << qb_) - 1);
        const size_t n_sub = 2 * binary_bytes_;

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

            // Build LUTs for this centroid
            // Each byte of the binary code maps to 2 sub-quantizers (4 dims each).
            // Sub-quantizer 2b handles dims 8b..8b+3 (low nibble of byte b)
            // Sub-quantizer 2b+1 handles dims 8b+4..8b+7 (high nibble of byte b)
            uint8_t* lut_j = all_luts + j * n_sub * 16;
            for (size_t b = 0; b < binary_bytes_; ++b) {
                uint8_t* lut_lo = lut_j + (2 * b) * 16;
                uint8_t* lut_hi = lut_j + (2 * b + 1) * 16;

                // Get the 8 SQ values for dimensions 8b..8b+7
                uint8_t sq[8] = {0};
                for (int k = 0; k < 8 && (8 * b + k) < d; ++k) {
                    sq[k] = quantized[8 * b + k];
                }

                // Low nibble LUT: dims 8b..8b+3
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[0];
                    if (c & 2) val += sq[1];
                    if (c & 4) val += sq[2];
                    if (c & 8) val += sq[3];
                    lut_lo[c] = val;
                }

                // High nibble LUT: dims 8b+4..8b+7
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[4];
                    if (c & 2) val += sq[5];
                    if (c & 4) val += sq[6];
                    if (c & 8) val += sq[7];
                    lut_hi[c] = val;
                }
            }
        }
    }

    /// Byte-level matrix transpose: extract binary codes from n points
    /// and write them in column-major order for SIMD scanning.
    ///
    /// Input:  codes[i] has binary_bytes_ bytes at offset i * faiss_code_size_
    /// Output: packed[b * FastScanComputer::kBlockSize + k] = codes[blk_start + k][b]
    void TransposeBlock(
        const uint8_t* codes, size_t blk_start, size_t blk_count,
        uint8_t* packed
    ) const {
        // Zero the entire block (handles partial blocks where blk_count < 32)
        std::memset(packed, 0, binary_bytes_ * FastScanComputer::kBlockSize);

        for (size_t k = 0; k < blk_count; ++k) {
            const uint8_t* src = codes + (blk_start + k) * faiss_code_size_;
            for (size_t b = 0; b < binary_bytes_; ++b) {
                packed[b * FastScanComputer::kBlockSize + k] = src[b];
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
