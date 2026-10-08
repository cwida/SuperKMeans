#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/gemms.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace skmeans {

inline void AccumulateNibbles(const uint8_t* code, size_t nbytes, uint32_t& sum, uint32_t& sum_sq) {
    for (size_t b = 0; b < nbytes; ++b) {
        uint8_t lo = code[b] & 0x0F;
        uint8_t hi = code[b] >> 4;
        sum += lo + hi;
        sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
    }
}

/**
 * @brief LVQ4 (Locally-adaptive Vector Quantization, 4-bit) quantizer.
 *
 * Per-vector adaptive scalar quantization: each vector stores its own
 * scale and bias, capturing local dynamic range. Encoding:
 *   code[dim] = round((x[dim] - bias) / scale), clamped to [0,15]
 * where bias = min(x), scale = (max(x) - min(x)) / 15.
 *
 * Code layout per vector:
 *   [d/2 packed u4x2 bytes] [float scale: 4B] [float bias: 4B]
 *   CodeSize(d) = d/2 + 8
 * scale and bias are per-vector, so there are no global parameters and hence no GetParams().
 *
 * Distance formula between LVQ4 vectors x_i, y_j:
 *   L2(x_i, y_j) = s_i*Scx + s_j*Scy - 2*s_i*s_j*<cx,cy>
 *                    + 2*db*(s_i*Scx - s_j*Scy) + d*db
 * where db = b_i - b_j, and <cx,cy> is the integer dot product of nibbles.
 *
 * Uses NumKong u4 GEMM (nk_dots_packed_u4) for bulk dot products and
 * u4 Horizontal SIMD kernels for per-pair distance in pruning survivors.
 */
class LVQ4Quantizer : public IQuantizer<Quantization::lvq4> {
  public:
    using quantized_t = IQuantizer::quantized_t;
    using u4_computer = DistanceComputer<DistanceFunction::l2, Quantization::sq4>;
    using u4_utils = UtilsComputer<Quantization::sq4>;
    using f32_utils = UtilsComputer<Quantization::f32>;

    LVQ4Quantizer() : has_amx(DetectAMX()) {}

    void InvalidateCaches() override {
        cached_x_ptr_ = nullptr;
        cached_n_x_ = 0;
        cached_partial_d_ = 0;
    }

    void Fit(ParallelExecutor& /*executor*/, const float* /*data*/, size_t /*n*/, size_t d)
        override {
        SKM_PROFILE_SCOPE("LVQ4::Fit");
        if (d % 2 != 0) {
            throw std::invalid_argument(
                "LVQ4 requires even dimensionality (got " + std::to_string(d) + ")"
            );
        }
        d_ = d;
        nibble_bytes_ = d / 2;
        code_size_ = nibble_bytes_ + 8;
        if (!dots_buf_) {
            dots_buf_.reset(new uint32_t[X_BATCH_SIZE * Y_BATCH_SIZE]);
        }
        packed_buf_.resize(nk_dots_packed_size_u4(Y_BATCH_SIZE, d));
        fitted_ = true;
    }

    static void UnpackU4x2ToU8(
        ParallelExecutor& executor,
        const quantized_t* src,
        uint8_t* dst,
        size_t n_rows,
        size_t k,
        size_t row_stride
    ) {
        executor.ParallelFor(n_rows, [&](size_t row_begin, size_t row_end, size_t) {
            for (size_t row = row_begin; row < row_end; ++row) {
                u4_utils::UnpackU4x2ToU8(src + row * row_stride, dst + row * k, k);
            }
        });
    }

    void MatrixMultiplication(
        ParallelExecutor& executor,
        const quantized_t* a,
        const quantized_t* b,
        uint32_t* out,
        size_t m,
        size_t n,
        size_t k,
        size_t a_stride,
        size_t b_stride,
        bool a_changed = true,
        bool b_changed = true
    ) const {
        const bool decode_to_u8 = has_amx || IS_ARM || (k <= THIN_MATRIX_THRESHOLD);

        if (decode_to_u8) {
            {
                SKM_PROFILE_SCOPE("search/unpack");
                if (a_changed) {
                    const size_t a_u8_size = m * k;
                    if (decoded_a_buf.size() < a_u8_size)
                        decoded_a_buf.resize(a_u8_size);
                    UnpackU4x2ToU8(executor, a, decoded_a_buf.data(), m, k, a_stride);
                }
                if (b_changed) {
                    const size_t b_u8_size = n * k;
                    if (decoded_b_buf.size() < b_u8_size)
                        decoded_b_buf.resize(b_u8_size);
                    UnpackU4x2ToU8(executor, b, decoded_b_buf.data(), n, k, b_stride);
                }
            }

            const bool use_numkong = !IS_ARM && (has_amx || k > THIN_MATRIX_THRESHOLD);
            U8Gemm(
                executor,
                decoded_a_buf.data(),
                decoded_b_buf.data(),
                out,
                m,
                n,
                k,
                k,
                k,
                use_numkong,
                packed_buf_,
                b_changed
            );
            return;
        }

        // Native u4 NumKong path: x86 without AMX, wide matrices
        U4Gemm(executor, a, b, out, m, n, k, a_stride, b_stride, packed_buf_, b_changed);
    }

    void Encode(ParallelExecutor& executor, const float* in, quantized_t* out, size_t n, size_t d)
        const override {
        SKM_PROFILE_SCOPE("LVQ4::Encode");
        assert(fitted_ && d == d_);

        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                LVQ4Codec::EncodeOne(in + i * d, out + i * code_size_, d, nibble_bytes_);
            }
        });
    }

    void Decode(ParallelExecutor& executor, const quantized_t* in, float* out, size_t n, size_t d)
        const override {
        SKM_PROFILE_SCOPE("LVQ4::Decode");
        assert(fitted_ && d == d_);

        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                LVQ4Codec::DecodeOne(in + i * code_size_, out + i * d, d, nibble_bytes_);
            }
        });
    }

    void ComputeNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        size_t d,
        float* out_norms
    ) const override {
        SKM_PROFILE_SCOPE("LVQ4::ComputeNorms");
        assert(fitted_ && d == d_);

        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = data + i * code_size_;
                float s, b;
                std::memcpy(&s, code + nibble_bytes_, sizeof(float));
                std::memcpy(&b, code + nibble_bytes_ + sizeof(float), sizeof(float));

                uint32_t sum_c = 0, sum_c_sq = 0;
                AccumulateNibbles(code, nibble_bytes_, sum_c, sum_c_sq);
                out_norms[i] = s * s * static_cast<float>(sum_c_sq) +
                               2.0f * s * b * static_cast<float>(sum_c) +
                               static_cast<float>(d) * b * b;
            }
        });
    }

    void FindNearestNeighbor(
        ParallelExecutor& executor,
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* /*norms_x*/,
        const float* /*norms_y*/,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/1st_blas");
        assert(fitted_);

        EnsureCodeFactorsCache(executor, reinterpret_cast<const uint8_t*>(x), n_x);

        CentroidFactors cf;
        ExtractCentroidFactors(executor, reinterpret_cast<const uint8_t*>(y), n_y, 0, 0, cf);

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                const bool a_changed = (j == 0);
                const bool b_changed = true;

                MatrixMultiplication(
                    executor,
                    x + i * code_size_,
                    y + j * code_size_,
                    dots_buf_.get(),
                    batch_n_x,
                    batch_n_y,
                    d,
                    code_size_,
                    code_size_,
                    a_changed,
                    b_changed
                );

                // Apply LVQ4 distance formula and find nearest per row
                using MatrixR =
                    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<MatrixR> dists_matrix(tmp_buf, batch_n_x, batch_n_y);
                executor.ParallelFor(batch_n_x, [&](size_t r_begin, size_t r_end, size_t) {
                    for (size_t r = r_begin; r < r_end; ++r) {
                        const size_t idx = i + r;
                        const uint32_t* dots_row = dots_buf_.get() + r * batch_n_y;
                        float* dists_row = tmp_buf + r * batch_n_y;
                        const float si = cached_scales_[idx];
                        const float bi = cached_biases_[idx];
                        const float norm_x_i = cached_norm_x_full_[idx];
                        const float two_si = 2.0f * si;
                        const float two_A_x_i = 2.0f * cached_A_x_full_[idx];
                        const float two_bi = 2.0f * bi;

                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            const size_t j_idx = j + c;
                            dists_row[c] =
                                norm_x_i + cf.norm_y_full[j_idx] -
                                two_si * cf.scales[j_idx] * static_cast<float>(dots_row[c]) -
                                two_A_x_i * cf.biases[j_idx] - two_bi * cf.sj_sum_cy_full[j_idx];
                        }

                        uint32_t knn_idx;
                        float batch_top_1 = dists_matrix.row(r).minCoeff(&knn_idx);
                        if (batch_top_1 < out_distances[idx]) {
                            out_distances[idx] = batch_top_1;
                            out_knn[idx] = static_cast<uint32_t>(j + knn_idx);
                        }
                    }
                });
            }
        }
    }

    void CacheDataPartialNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        size_t /*d*/,
        uint32_t partial_d
    ) override {
        ComputeDataPartialNorms(executor, data, n, partial_d);
    }

    void ComputeDataPartialNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        uint32_t partial_d
    ) const {
        SKM_PROFILE_SCOPE("LVQ4::CacheDataPartialNorms");
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(data);
        const size_t front_bytes = partial_d / 2;
        const size_t mid_bytes = d_ / 8; // d/4, in packed bytes
        const float front_d_f = static_cast<float>(front_bytes * 2);
        const float mid_d_f = static_cast<float>(mid_bytes * 2);

        cached_sum_cx_sq_front_.resize(n);
        cached_sum_cx_sq_mid_.resize(n);
        cached_norm_x_front_.resize(n);
        cached_A_x_front_.resize(n);
        cached_norm_x_mid_.resize(n);
        cached_A_x_mid_.resize(n);
        cached_partial_d_ = partial_d;

        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = codes + i * code_size_;
                float si, bi;
                std::memcpy(&si, code + nibble_bytes_, sizeof(float));
                std::memcpy(&bi, code + nibble_bytes_ + sizeof(float), sizeof(float));

                uint32_t sum = 0, sum_sq = 0;
                const size_t min_fb = std::min(front_bytes, mid_bytes);
                const size_t max_fb = std::max(front_bytes, mid_bytes);

                AccumulateNibbles(code, min_fb, sum, sum_sq);

                if (front_bytes <= mid_bytes) {
                    cached_sum_cx_sq_front_[i] = sum_sq;
                    float sf = static_cast<float>(sum), sqf = static_cast<float>(sum_sq);
                    cached_norm_x_front_[i] =
                        si * si * sqf + 2.0f * bi * si * sf + front_d_f * bi * bi;
                    cached_A_x_front_[i] = si * sf + front_d_f * bi;
                    AccumulateNibbles(code + min_fb, max_fb - min_fb, sum, sum_sq);
                    cached_sum_cx_sq_mid_[i] = sum_sq;
                    sf = static_cast<float>(sum);
                    sqf = static_cast<float>(sum_sq);
                    cached_norm_x_mid_[i] = si * si * sqf + 2.0f * bi * si * sf + mid_d_f * bi * bi;
                    cached_A_x_mid_[i] = si * sf + mid_d_f * bi;
                } else {
                    cached_sum_cx_sq_mid_[i] = sum_sq;
                    float sf = static_cast<float>(sum), sqf = static_cast<float>(sum_sq);
                    cached_norm_x_mid_[i] = si * si * sqf + 2.0f * bi * si * sf + mid_d_f * bi * bi;
                    cached_A_x_mid_[i] = si * sf + mid_d_f * bi;
                    AccumulateNibbles(code + min_fb, max_fb - min_fb, sum, sum_sq);
                    cached_sum_cx_sq_front_[i] = sum_sq;
                    sf = static_cast<float>(sum);
                    sqf = static_cast<float>(sum_sq);
                    cached_norm_x_front_[i] =
                        si * si * sqf + 2.0f * bi * si * sf + front_d_f * bi * bi;
                    cached_A_x_front_[i] = si * sf + front_d_f * bi;
                }
            }
        });
    }

    void CacheCentroidPartialNorms(
        ParallelExecutor& /*executor*/,
        const quantized_t* /*centroids*/,
        size_t /*n*/,
        size_t /*d*/,
        uint32_t /*partial_d*/
    ) override {
        // No-op: centroid partials computed in ExtractCentroidFactors.
    }

    void FindNearestNeighborWithPruning(
        ParallelExecutor& executor,
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<Quantization::lvq4>& /*pdx_centroids*/,
        uint32_t partial_d,
        size_t* out_not_pruned_counts,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("search");
        assert(fitted_);

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);
        const uint8_t* y_codes = reinterpret_cast<const uint8_t*>(y);

        EnsureCodeFactorsCache(executor, x_codes, n_x);

        if (cached_partial_d_ != partial_d || cached_norm_x_front_.size() != n_x) {
            ComputeDataPartialNorms(executor, x, n_x, partial_d);
        }

        // Pruning geometry
        const size_t front_bytes = partial_d / 2;
        const size_t front_d = front_bytes * 2;
        const size_t mid_bytes = d / 8; // d/4, in packed bytes
        const size_t mid_d = mid_bytes * 2;
        const bool use_mid = (front_bytes < mid_bytes) && (mid_bytes < nibble_bytes_);

        // ADSampling ratios
        const float ad_ratio_front = ComputeADSamplingRatio(front_d, d, PRUNER_INITIAL_THRESHOLD);
        const float ad_ratio_mid =
            use_mid ? ComputeADSamplingRatio(mid_d, d, PRUNER_INITIAL_THRESHOLD) : 1.0f;

        CentroidFactors cf;
        ExtractCentroidFactors(executor, y_codes, n_y, front_d, mid_d, cf);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("search/blas");
                    const bool a_changed = (j == 0);
                    const bool b_changed = true;
                    MatrixMultiplication(
                        executor,
                        x + i * code_size_,
                        y + j * code_size_,
                        dots_buf_.get(),
                        batch_n_x,
                        batch_n_y,
                        partial_d,
                        code_size_,
                        code_size_,
                        a_changed,
                        b_changed
                    );
                }

                {
                    SKM_PROFILE_SCOPE("search/pdx");
                    executor.ParallelFor(batch_n_x, [&](size_t r_begin, size_t r_end, size_t) {
                        for (size_t r = r_begin; r < r_end; ++r) {
                            const size_t i_idx = i + r;
                            const uint32_t* dots_row = dots_buf_.get() + r * batch_n_y;
                            const uint8_t* x_code = x_codes + i_idx * code_size_;

                            const float si = cached_scales_[i_idx];
                            const float bi = cached_biases_[i_idx];
                            const float two_si = 2.0f * si;
                            const float two_bi = 2.0f * bi;
                            const float norm_x_front_i = cached_norm_x_front_[i_idx];
                            const float two_A_x_front_i = 2.0f * cached_A_x_front_[i_idx];
                            const float norm_x_mid_i = cached_norm_x_mid_[i_idx];
                            const float two_A_x_mid_i = 2.0f * cached_A_x_mid_[i_idx];
                            const float norm_x_full_i = cached_norm_x_full_[i_idx];
                            const float two_A_x_full_i = 2.0f * cached_A_x_full_[i_idx];

                            // Phase 1: threshold from previous assignment
                            float best_dist;
                            uint32_t best_idx;
                            if (j == 0) {
                                const uint32_t prev_j = out_knn[i_idx];
                                best_idx = prev_j;
                                best_dist = ComputeFullDistance(
                                    x_code,
                                    y_codes + prev_j * code_size_,
                                    si,
                                    bi,
                                    cached_sum_cx_sq_[i_idx],
                                    norm_x_full_i,
                                    cached_A_x_full_[i_idx],
                                    cf.scales[prev_j],
                                    cf.biases[prev_j],
                                    cf.sum_cy_sq[prev_j],
                                    cf.norm_y_full[prev_j],
                                    cf.sj_sum_cy_full[prev_j]
                                );
                                out_not_pruned_counts[i_idx] = 0;
                            } else {
                                best_dist = out_distances[i_idx];
                                best_idx = out_knn[i_idx];
                            }

                            // Phase 2: vectorized front partial distances
                            thread_local float partial_dists[Y_BATCH_SIZE];
                            thread_local uint32_t survivor_positions[Y_BATCH_SIZE];

                            SKM_VECTORIZE_LOOP
                            for (size_t c = 0; c < batch_n_y; ++c) {
                                const size_t j_idx = j + c;
                                partial_dists[c] =
                                    norm_x_front_i + cf.norm_y_front[j_idx] -
                                    two_si * cf.scales[j_idx] * static_cast<float>(dots_row[c]) -
                                    two_A_x_front_i * cf.biases[j_idx] -
                                    two_bi * cf.sj_sum_cy_front_f[j_idx];
                            }

                            // Compact survivor indices via SIMD
                            size_t n_survivors = 0;
                            const float front_threshold = best_dist * ad_ratio_front;
                            f32_utils::InitPositionsArray(
                                batch_n_y,
                                n_survivors,
                                survivor_positions,
                                front_threshold,
                                partial_dists
                            );
                            out_not_pruned_counts[i_idx] += n_survivors;

                            // Phase 3+: resolve survivors
                            thread_local uint32_t mid_dots[Y_BATCH_SIZE];
                            thread_local uint32_t mid_survivor_positions[Y_BATCH_SIZE];

                            // Pointers/counts for the final rest+full loop
                            const uint32_t* rest_positions = survivor_positions;
                            const uint32_t* rest_dots = nullptr; // use dots_row[c] directly
                            size_t n_rest = n_survivors;
                            size_t rest_start_byte = front_bytes;
                            uint32_t sum_cx_sq_at_rest = cached_sum_cx_sq_front_[i_idx];

                            if (use_mid) {
                                // Phase 3a: compute mid gap dots for all front survivors
                                const uint32_t sum_cx_sq_gap =
                                    cached_sum_cx_sq_mid_[i_idx] - cached_sum_cx_sq_front_[i_idx];

                                for (size_t s = 0; s < n_survivors; ++s) {
                                    const size_t c = survivor_positions[s];
                                    const size_t j_idx = j + c;
                                    const uint8_t* y_code = y_codes + j_idx * code_size_;

                                    uint32_t gap_l2_int = u4_computer::Horizontal(
                                        (const nk_u4x2_t*) (x_code + front_bytes),
                                        (const nk_u4x2_t*) (y_code + front_bytes),
                                        mid_bytes - front_bytes
                                    );
                                    uint32_t sum_cy_sq_gap =
                                        cf.sum_cy_sq_mid[j_idx] - cf.sum_cy_sq_front[j_idx];
                                    uint32_t gap_dot =
                                        (sum_cx_sq_gap + sum_cy_sq_gap - gap_l2_int) / 2;
                                    mid_dots[s] = dots_row[c] + gap_dot;
                                }

                                // Phase 3b: vectorized mid distance computation
                                SKM_VECTORIZE_LOOP
                                for (size_t s = 0; s < n_survivors; ++s) {
                                    const size_t j_idx = j + survivor_positions[s];
                                    partial_dists[s] =
                                        norm_x_mid_i + cf.norm_y_mid[j_idx] -
                                        two_si * cf.scales[j_idx] * static_cast<float>(mid_dots[s]) -
                                        two_A_x_mid_i * cf.biases[j_idx] -
                                        two_bi * cf.sj_sum_cy_mid_f[j_idx];
                                }

                                // Phase 3c: compact mid survivors
                                size_t n_mid_survivors = 0;
                                const float mid_threshold = best_dist * ad_ratio_mid;
                                f32_utils::InitPositionsArray(
                                    n_survivors,
                                    n_mid_survivors,
                                    mid_survivor_positions,
                                    mid_threshold,
                                    partial_dists
                                );

                                rest_positions = mid_survivor_positions;
                                rest_dots = mid_dots;
                                n_rest = n_mid_survivors;
                                rest_start_byte = mid_bytes;
                                sum_cx_sq_at_rest = cached_sum_cx_sq_mid_[i_idx];
                            }

                            // Phase 4: rest + full for surviving candidates
                            const uint32_t sum_cx_sq_rest_i =
                                cached_sum_cx_sq_[i_idx] - sum_cx_sq_at_rest;

                            for (size_t rs = 0; rs < n_rest; ++rs) {
                                const size_t s = rest_positions[rs];
                                const size_t c = use_mid ? survivor_positions[s] : s;
                                const size_t j_idx = j + c;
                                const float sj = cf.scales[j_idx];
                                const float bj = cf.biases[j_idx];
                                const uint8_t* y_code = y_codes + j_idx * code_size_;

                                uint32_t dot_accumulated = use_mid ? rest_dots[s] : dots_row[c];

                                uint32_t rest_l2_int = u4_computer::Horizontal(
                                    (const nk_u4x2_t*) (x_code + rest_start_byte),
                                    (const nk_u4x2_t*) (y_code + rest_start_byte),
                                    nibble_bytes_ - rest_start_byte
                                );
                                uint32_t sum_cy_sq_at_start =
                                    use_mid ? cf.sum_cy_sq_mid[j_idx] : cf.sum_cy_sq_front[j_idx];
                                uint32_t sum_cy_sq_rest = cf.sum_cy_sq[j_idx] - sum_cy_sq_at_start;
                                uint32_t rest_dot =
                                    (sum_cx_sq_rest_i + sum_cy_sq_rest - rest_l2_int) / 2;
                                uint32_t full_dot = dot_accumulated + rest_dot;

                                float full_l2 =
                                    norm_x_full_i + cf.norm_y_full[j_idx] -
                                    two_si * sj * static_cast<float>(full_dot) -
                                    two_A_x_full_i * bj - two_bi * cf.sj_sum_cy_full[j_idx];

                                if (full_l2 < best_dist) {
                                    best_dist = full_l2;
                                    best_idx = static_cast<uint32_t>(j_idx);
                                }
                            }

                            out_distances[i_idx] = best_dist;
                            out_knn[i_idx] = best_idx;
                        }
                    });
                }
            }
        }
    }

    size_t CodeSize(size_t d) const override { return d / 2 + 8; }
    bool IsFitted() const override { return fitted_; }
    bool SupportsPruning() const override { return true; }
    bool NeedsPDXLayout() const override { return false; }

    /// ~12.5% of d, aligned to 8 for byte-aligned code boundaries.
    uint32_t InitialPartialD(uint32_t vertical_d) const override {
        return std::max<uint32_t>(MIN_PARTIAL_D, ((vertical_d / 8) + 7) & ~7u);
    }

    /// Round up to multiple of 8 after tuning adjustment.
    uint32_t AlignPartialD(uint32_t partial_d, uint32_t vertical_d) const override {
        return std::min((partial_d + 7) & ~7u, vertical_d);
    }
    void UpdateCentroids(
        ParallelExecutor& executor,
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
        uint32_t* cluster_sizes,
        size_t n,
        size_t n_clusters,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("LVQ4::UpdateCentroids");
        assert(fitted_ && d == d_);
        // One rank per centroid range; the executor runs every rank exactly once.
        const size_t nt = executor.NumWorkers();
        executor.ParallelFor(nt, [&](size_t rank_begin, size_t rank_end, size_t) {
            std::unique_ptr<float[]> decode_buf(new float[d]);
            for (size_t rank = rank_begin; rank < rank_end; ++rank) {
                const size_t c0 = (n_clusters * rank) / nt;
                const size_t c1 = (n_clusters * (rank + 1)) / nt;
                for (size_t i = 0; i < n; ++i) {
                    uint32_t ci = assignments[i];
                    if (ci >= c0 && ci < c1) {
                        LVQ4Codec::DecodeOne(
                            encoded_data + i * code_size_, decode_buf.get(), d, nibble_bytes_
                        );
                        cluster_sizes[ci] += 1;
                        float* acc = centroid_accumulators + ci * d;
                        SKM_VECTORIZE_LOOP
                        for (size_t j = 0; j < d; ++j) {
                            acc[j] += decode_buf[j];
                        }
                    }
                }
            }
        });
    }

  private:
    size_t d_ = 0;
    size_t nibble_bytes_ = 0; // d/2
    size_t code_size_ = 0;    // d/2 + 8
    bool fitted_ = false;
    bool has_amx = false;

    // Per-data caches (keyed by pointer + count for invalidation)
    mutable const uint8_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<float> cached_scales_, cached_biases_;
    mutable std::vector<uint32_t> cached_sum_cx_sq_;

    // GEMM buffers
    mutable std::unique_ptr<uint32_t[]> dots_buf_;
    mutable std::vector<char> packed_buf_;
    mutable std::vector<uint8_t> decoded_a_buf;
    mutable std::vector<uint8_t> decoded_b_buf;

    // Pruning checkpoint caches
    mutable std::vector<uint32_t> cached_sum_cx_sq_front_;
    mutable std::vector<uint32_t> cached_sum_cx_sq_mid_;
    mutable uint32_t cached_partial_d_ = 0;

    // Precomputed float caches: norm_x = si²·Σcx² + 2·bi·si·Σcx + D·bi²
    //                           A_x    = si·Σcx + D·bi
    mutable std::vector<float> cached_norm_x_full_, cached_A_x_full_;
    mutable std::vector<float> cached_norm_x_front_, cached_A_x_front_;
    mutable std::vector<float> cached_norm_x_mid_, cached_A_x_mid_;

    struct CentroidFactors {
        std::vector<float> scales, biases;
        std::vector<uint32_t> sum_cy, sum_cy_sq;
        std::vector<uint32_t> sum_cy_front, sum_cy_sq_front;
        std::vector<uint32_t> sum_cy_mid, sum_cy_sq_mid;
        // Precomputed: norm_y = sj²·Σcy² + 2·bj·sj·Σcy + D·bj²
        //              sj_sum_cy = sj·Σcy
        std::vector<float> norm_y_full, sj_sum_cy_full;
        std::vector<float> norm_y_front, sj_sum_cy_front_f;
        std::vector<float> norm_y_mid, sj_sum_cy_mid_f;
    };

    void EnsureCodeFactorsCache(ParallelExecutor& executor, const uint8_t* x_codes, size_t n_x)
        const {
        if (cached_x_ptr_ == x_codes && cached_n_x_ == n_x)
            return;
        SKM_PROFILE_SCOPE("LVQ4::EnsureCodeFactorsCache");

        cached_scales_.resize(n_x);
        cached_biases_.resize(n_x);
        cached_sum_cx_sq_.resize(n_x);
        cached_norm_x_full_.resize(n_x);
        cached_A_x_full_.resize(n_x);

        const float d_f = static_cast<float>(d_);

        executor.ParallelFor(n_x, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = x_codes + i * code_size_;
                float si, bi;
                std::memcpy(&si, code + nibble_bytes_, sizeof(float));
                std::memcpy(&bi, code + nibble_bytes_ + sizeof(float), sizeof(float));
                cached_scales_[i] = si;
                cached_biases_[i] = bi;

                uint32_t sum = 0, sum_sq = 0;
                AccumulateNibbles(code, nibble_bytes_, sum, sum_sq);
                cached_sum_cx_sq_[i] = sum_sq;

                const float sum_f = static_cast<float>(sum);
                const float sum_sq_f = static_cast<float>(sum_sq);
                cached_norm_x_full_[i] =
                    si * si * sum_sq_f + 2.0f * bi * si * sum_f + d_f * bi * bi;
                cached_A_x_full_[i] = si * sum_f + d_f * bi;
            }
        });

        cached_x_ptr_ = x_codes;
        cached_n_x_ = n_x;
    }

    void ExtractCentroidFactors(
        ParallelExecutor& executor,
        const uint8_t* y_codes,
        size_t n_y,
        size_t front_d,
        size_t mid_d,
        CentroidFactors& cf
    ) const {
        SKM_PROFILE_SCOPE("LVQ4::ExtractCentroidFactors");
        const size_t front_bytes = front_d / 2;
        const size_t mid_bytes = mid_d / 2;

        cf.scales.resize(n_y);
        cf.biases.resize(n_y);
        cf.sum_cy.resize(n_y);
        cf.sum_cy_sq.resize(n_y);
        cf.sum_cy_front.resize(n_y);
        cf.sum_cy_sq_front.resize(n_y);
        cf.sum_cy_mid.resize(n_y);
        cf.sum_cy_sq_mid.resize(n_y);
        cf.norm_y_full.resize(n_y);
        cf.sj_sum_cy_full.resize(n_y);
        cf.norm_y_front.resize(n_y);
        cf.sj_sum_cy_front_f.resize(n_y);
        cf.norm_y_mid.resize(n_y);
        cf.sj_sum_cy_mid_f.resize(n_y);

        const float d_f = static_cast<float>(d_);
        const float front_d_f = static_cast<float>(front_d);
        const float mid_d_f = static_cast<float>(mid_d);

        executor.ParallelFor(n_y, [&](size_t begin, size_t end, size_t) {
            for (size_t j = begin; j < end; ++j) {
                const uint8_t* code_j = y_codes + j * code_size_;
                std::memcpy(&cf.scales[j], code_j + nibble_bytes_, sizeof(float));
                std::memcpy(&cf.biases[j], code_j + nibble_bytes_ + sizeof(float), sizeof(float));

                const size_t fb = std::min(front_bytes, nibble_bytes_);
                const size_t mb = std::min(mid_bytes, nibble_bytes_);
                const size_t min_fb = std::min(fb, mb);
                const size_t max_fb = std::max(fb, mb);

                uint32_t sum = 0, sum_sq = 0;
                AccumulateNibbles(code_j, min_fb, sum, sum_sq);
                // NOLINTNEXTLINE(bugprone-branch-clone)
                if (fb <= mb) {
                    cf.sum_cy_front[j] = sum;
                    cf.sum_cy_sq_front[j] = sum_sq;
                    AccumulateNibbles(code_j + min_fb, max_fb - min_fb, sum, sum_sq);
                    cf.sum_cy_mid[j] = sum;
                    cf.sum_cy_sq_mid[j] = sum_sq;
                } else {
                    cf.sum_cy_mid[j] = sum;
                    cf.sum_cy_sq_mid[j] = sum_sq;
                    AccumulateNibbles(code_j + min_fb, max_fb - min_fb, sum, sum_sq);
                    cf.sum_cy_front[j] = sum;
                    cf.sum_cy_sq_front[j] = sum_sq;
                }
                AccumulateNibbles(code_j + max_fb, nibble_bytes_ - max_fb, sum, sum_sq);
                cf.sum_cy[j] = sum;
                cf.sum_cy_sq[j] = sum_sq;

                // Precompute float values for optimized distance formula
                const float sj = cf.scales[j];
                const float bj = cf.biases[j];

                float sc_f = static_cast<float>(cf.sum_cy[j]);
                float scsq_f = static_cast<float>(cf.sum_cy_sq[j]);
                cf.norm_y_full[j] = sj * sj * scsq_f + 2.0f * bj * sj * sc_f + d_f * bj * bj;
                cf.sj_sum_cy_full[j] = sj * sc_f;

                sc_f = static_cast<float>(cf.sum_cy_front[j]);
                scsq_f = static_cast<float>(cf.sum_cy_sq_front[j]);
                cf.norm_y_front[j] =
                    sj * sj * scsq_f + 2.0f * bj * sj * sc_f + front_d_f * bj * bj;
                cf.sj_sum_cy_front_f[j] = sj * sc_f;

                sc_f = static_cast<float>(cf.sum_cy_mid[j]);
                scsq_f = static_cast<float>(cf.sum_cy_sq_mid[j]);
                cf.norm_y_mid[j] = sj * sj * scsq_f + 2.0f * bj * sj * sc_f + mid_d_f * bj * bj;
                cf.sj_sum_cy_mid_f[j] = sj * sc_f;
            }
        });
    }

    float ComputeFullDistance(
        const uint8_t* x_code,
        const uint8_t* y_code,
        float si,
        float bi,
        uint32_t sum_cx_sq,
        float norm_x_full,
        float A_x_full,
        float sj,
        float bj,
        uint32_t sum_cy_sq,
        float norm_y_full,
        float sj_sum_cy_full
    ) const {
        uint32_t l2_int = u4_computer::Horizontal(
            (const nk_u4x2_t*) x_code, (const nk_u4x2_t*) y_code, nibble_bytes_
        );
        uint32_t int_dot = (sum_cx_sq + sum_cy_sq - l2_int) / 2;
        return norm_x_full + norm_y_full - 2.0f * si * sj * static_cast<float>(int_dot) -
               2.0f * A_x_full * bj - 2.0f * bi * sj_sum_cy_full;
    }
};

} // namespace skmeans
