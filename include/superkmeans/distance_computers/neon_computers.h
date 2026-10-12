#pragma once

#include <cassert>
#include <cstring>
#include <iostream>

#include "arm_neon.h"
#include "superkmeans/common.h"
#include "superkmeans/distance_computers/scalar_computers.h"

namespace skmeans {

// Equivalent of vdotq_u32(acc, a, a) for squared accumulation.
static inline uint32x4_t SquaredDotAccumulate(uint32x4_t acc, uint8x16_t a) {
#ifdef __ARM_FEATURE_DOTPROD
    return vdotq_u32(acc, a, a);
#else
    uint16x8_t sq_lo = vmull_u8(vget_low_u8(a), vget_low_u8(a));
    uint16x8_t sq_hi = vmull_u8(vget_high_u8(a), vget_high_u8(a));
    uint32x4_t partial_lo = vpaddlq_u16(sq_lo);
    uint32x4_t partial_hi = vpaddlq_u16(sq_hi);
    return vaddq_u32(acc, vpaddq_u32(partial_lo, partial_hi));
#endif
}

template <DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<DistanceFunction::l2, Quantization::sq8> {
  public:
    using distance_t = pdx_distance_t<Quantization::sq8>;
    using value_t = skmeans_value_t<Quantization::sq8>;

    /**
     * @brief Computes the L2 distance between two uint8 vectors using NEON.
     * Taken from SimSimd library: https://github.com/ashvardanian/SimSIMD
     * @param vector1 Input vector 1
     * @param vector2 Input vector 2
     * @param num_dimensions Number of dimensions
     * @return L2 distance between the two vectors
     */
    static distance_t Horizontal(
        const value_t* SKM_RESTRICT vector1,
        const value_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        uint32x4_t sum_vec = vdupq_n_u32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
            sum_vec = SquaredDotAccumulate(sum_vec, d_vec);
        }
        distance_t distance = vaddvq_u32(sum_vec);
        for (; i < num_dimensions; ++i) {
            int n = static_cast<int>(vector1[i]) - vector2[i];
            distance += n * n;
        }
        return distance;
    };
};

template <>
class SIMDComputer<DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<Quantization::f32>;

    /**
     * @brief Computes the L2 distance between two float vectors using NEON.
     * Taken from SimSimd library: https://github.com/ashvardanian/SimSIMD
     * @param vector1 Input vector 1
     * @param vector2 Input vector 2
     * @param num_dimensions Number of dimensions
     * @return L2 distance between the two vectors
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
#if defined(__APPLE__)
        distance_t distance = 0.0f;
        SKM_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            float diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
#else
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }
        distance_t distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            float diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
#endif
    };
};

template <>
class SIMDComputer<DistanceFunction::l2, Quantization::sq4> {
  public:
    using distance_t = pdx_distance_t<Quantization::sq4>;
    using data_t = skmeans_value_t<Quantization::sq4>;

    /**
     * @brief Computes L2² distance between two packed u4x2 vectors using NEON.
     * @param vector1 Packed u4x2 input vector 1
     * @param vector2 Packed u4x2 input vector 2
     * @param num_packed_bytes Number of packed bytes (each byte = 2 dims)
     * @return L2² distance as uint32_t
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_packed_bytes
    ) {
        uint32x4_t sum_vec = vdupq_n_u32(0);
        const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
        size_t i = 0;
        for (; i + 16 <= num_packed_bytes; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            // Extract low nibbles
            uint8x16_t a_lo = vandq_u8(a_vec, nibble_mask);
            uint8x16_t b_lo = vandq_u8(b_vec, nibble_mask);
            // Extract high nibbles
            uint8x16_t a_hi = vshrq_n_u8(a_vec, 4);
            uint8x16_t b_hi = vshrq_n_u8(b_vec, 4);
            // Absolute difference
            uint8x16_t diff_lo = vabdq_u8(a_lo, b_lo);
            uint8x16_t diff_hi = vabdq_u8(a_hi, b_hi);
            // Square and accumulate
            sum_vec = SquaredDotAccumulate(sum_vec, diff_lo);
            sum_vec = SquaredDotAccumulate(sum_vec, diff_hi);
        }
        distance_t distance = vaddvq_u32(sum_vec);
        // Scalar tail
        for (; i < num_packed_bytes; ++i) {
            int32_t a_lo = vector1[i] & 0x0F;
            int32_t b_lo = vector2[i] & 0x0F;
            int32_t a_hi = (vector1[i] >> 4) & 0x0F;
            int32_t b_hi = (vector2[i] >> 4) & 0x0F;
            int32_t d_lo = a_lo - b_lo, d_hi = a_hi - b_hi;
            distance += static_cast<uint32_t>(d_lo * d_lo + d_hi * d_hi);
        }
        return distance;
    };
};

template <>
class SIMDComputer<DistanceFunction::l2, Quantization::rabitq> {
  public:
    using distance_t = pdx_distance_t<Quantization::rabitq>;
    using data_t = skmeans_value_t<Quantization::rabitq>;

    /**
     * @brief Computes popcount(a AND b) — binary inner product using NEON.
     * Uses vcntq_u8 for per-byte popcount.
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_bytes
    ) {
        uint32_t count = 0;
        size_t i = 0;
        for (; i + 16 <= num_bytes; i += 16) {
            uint8x16_t va = vld1q_u8(vector1 + i);
            uint8x16_t vb = vld1q_u8(vector2 + i);
            uint8x16_t cnt = vcntq_u8(vandq_u8(va, vb));
            count += vaddvq_u8(cnt);
        }
        for (; i < num_bytes; ++i) {
            count += static_cast<uint32_t>(SKM_POPCOUNT(vector1[i] & vector2[i]));
        }
        return count;
    };

    static uint32_t HorizontalMultiPlane(
        const data_t* SKM_RESTRICT data,
        const data_t* planes_interleaved,
        size_t num_bytes,
        int qb
    ) {
        uint64x2_t acc = vdupq_n_u64(0);
        size_t i = 0;
        for (; i + 16 <= num_bytes; i += 16) {
            uint8x16_t x = vld1q_u8(data + i);
            const uint8_t* chunk = planes_interleaved + i * qb;
            for (int bp = 0; bp < qb; ++bp) {
                uint8x16_t p = vld1q_u8(chunk + static_cast<size_t>(bp) * 16);
                uint8x16_t cnt = vcntq_u8(vandq_u8(x, p));
                uint64x2_t popcnt64 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt)));
                acc = vaddq_u64(acc, vshlq_u64(popcnt64, vdupq_n_s64(bp)));
            }
        }
        uint32_t result = static_cast<uint32_t>(vaddvq_u64(acc));
        for (; i < num_bytes; ++i) {
            size_t chunk_idx = i / 16;
            size_t byte_in_chunk = i % 16;
            for (int bp = 0; bp < qb; ++bp) {
                result +=
                    static_cast<uint32_t>(SKM_POPCOUNT(
                        data[i] &
                        planes_interleaved
                            [chunk_idx * qb * 16 + static_cast<size_t>(bp) * 16 + byte_in_chunk]
                    ))
                    << bp;
            }
        }
        return result;
    }
};

template <Quantization q>
class SIMDUtilsComputer {
  public:
    using data_t = skmeans_value_t<q>;
    using pdx_dist_t = pdx_distance_t<q>;

    static void FlipSign(const data_t*, data_t*, const uint32_t*, size_t) {
        assert(false && "FlipSign not supported");
    }

    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        pdx_dist_t pruning_threshold,
        const pdx_dist_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        size_t vector_idx = 0;
        constexpr size_t k_simd_width = 4;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        uint32x4_t threshold_vec = vdupq_n_u32(pruning_threshold);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            uint32x4_t distances = vld1q_u32(pruning_distances + vector_idx);
            uint32x4_t cmp_result = vcltq_u32(distances, threshold_vec);
            uint32_t any_passed = vmaxvq_u32(cmp_result);
            if (SKM_UNLIKELY(any_passed)) {
                uint32_t mask[4];
                vst1q_u32(mask, cmp_result);
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask[i] != 0);
                }
            }
        }
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    static void PackU8ToU4x2(const uint8_t*, uint8_t*, size_t) {
        assert(false && "PackU8ToU4x2 not applicable");
    }
    static void UnpackU4x2ToU8(const uint8_t*, uint8_t*, size_t) {
        assert(false && "UnpackU4x2ToU8 not applicable");
    }
};

template <>
class SIMDUtilsComputer<Quantization::f32> {
  public:
    using data_t = skmeans_value_t<Quantization::f32>;

    /**
     * @brief Flip sign of floats based on a mask using NEON.
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t vec = vld1q_f32(data + j);
            const uint32x4_t mask = vld1q_u32(masks + j);
            vec = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec), mask));
            vst1q_f32(out + j, vec);
        }
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        for (; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors using NEON.
     *
     * Optimized for cases where only ~2% of vectors pass the threshold test.
     * This version is only slightly faster than a scalar kernel
     *
     * @param n_vectors Number of vectors to process
     * @param n_vectors_not_pruned Output: count of vectors passing threshold (updated)
     * @param pruning_positions Output array of indices that passed (compacted)
     * @param pruning_threshold Threshold value for comparison
     * @param pruning_distances Input array of distances to compare
     */
    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        data_t pruning_threshold,
        const data_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        size_t vector_idx = 0;
        constexpr size_t k_simd_width = 4;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        float32x4_t threshold_vec = vdupq_n_f32(pruning_threshold);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            float32x4_t distances = vld1q_f32(pruning_distances + vector_idx);
            uint32x4_t cmp_result = vcltq_f32(distances, threshold_vec);
            uint32_t any_passed = vmaxvq_u32(cmp_result);
            if (SKM_UNLIKELY(any_passed)) {
                uint32_t mask[4];
                vst1q_u32(mask, cmp_result);
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask[i] != 0);
                }
            }
        }
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    static void PackU8ToU4x2(const uint8_t*, uint8_t*, size_t) {
        assert(false && "PackU8ToU4x2 not applicable for f32");
    }
    static void UnpackU4x2ToU8(const uint8_t*, uint8_t*, size_t) {
        assert(false && "UnpackU4x2ToU8 not applicable for f32");
    }
};

template <>
class SIMDUtilsComputer<Quantization::sq4> {
  public:
    using data_t = skmeans_value_t<Quantization::sq4>;
    using pdx_dist_t = pdx_distance_t<Quantization::sq4>;

    static void FlipSign(const data_t*, data_t*, const uint32_t*, size_t) {
        assert(false && "FlipSign not supported for u4");
    }

    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        pdx_dist_t pruning_threshold,
        const pdx_dist_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        size_t vector_idx = 0;
        constexpr size_t k_simd_width = 4;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        uint32x4_t threshold_vec = vdupq_n_u32(pruning_threshold);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            uint32x4_t distances = vld1q_u32(pruning_distances + vector_idx);
            uint32x4_t cmp_result = vcltq_u32(distances, threshold_vec);
            uint32_t any_passed = vmaxvq_u32(cmp_result);
            if (SKM_UNLIKELY(any_passed)) {
                uint32_t mask[4];
                vst1q_u32(mask, cmp_result);
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask[i] != 0);
                }
            }
        }
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    /**
     * @brief Pack u8 values [0,15] into u4x2 format using NEON.
     *
     * Uses vuzp to deinterleave even/odd bytes, shifts odd by 4, ORs them.
     * Processes 16 input bytes (8 output bytes) per iteration.
     */
    static void PackU8ToU4x2(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            uint8x16_t v = vld1q_u8(src + i);
            uint8x8x2_t pairs = vuzp_u8(vget_low_u8(v), vget_high_u8(v));
            uint8x8_t packed = vorr_u8(pairs.val[0], vshl_n_u8(pairs.val[1], 4));
            vst1_u8(dst + i / 2, packed);
        }
        for (; i + 2 <= count; i += 2) {
            dst[i / 2] = (src[i] & 0x0F) | ((src[i + 1] & 0x0F) << 4);
        }
    }

    /**
     * @brief Unpack u4x2 packed bytes to individual u8 values using NEON.
     *
     * Masks low/high nibbles and interleaves with vzipq.
     * Processes 16 packed bytes (32 output) per iteration.
     */
    static void UnpackU4x2ToU8(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        size_t i = 0;
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        for (; i + 16 <= n_packed; i += 16) {
            uint8x16_t v = vld1q_u8(src + i);
            uint8x16_t lo = vandq_u8(v, mask);
            uint8x16_t hi = vshrq_n_u8(v, 4);
            uint8x16x2_t result = vzipq_u8(lo, hi);
            vst1q_u8(dst + i * 2, result.val[0]);
            vst1q_u8(dst + i * 2 + 16, result.val[1]);
        }
        for (; i < n_packed; ++i) {
            dst[2 * i] = src[i] & 0x0F;
            dst[2 * i + 1] = (src[i] >> 4) & 0x0F;
        }
    }
};

class SIMDFastScanComputer {
  public:
    static constexpr size_t kBlockSize = 32;

    /**
     * @brief NEON-accelerated RaBitQ partial L2 for a 32-point block.
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     * Processes 4 floats per NEON iteration (8 iterations for kBlockSize=32).
     */
    /**
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     * @param sum_q_f32 Pre-converted float array (caller converts uint32_t→float once per block).
     */
    template <bool U32Dot = false>
    static void RabitQCorrection(
        const void* partial_dot,
        float c1j,
        float c2j,
        float c34j,
        float qr_j,
        const float* sum_q_f32,
        const float* or_c_l2sqr,
        const float* dp_mult,
        float* out_partial_l2,
        size_t blk_count
    ) {
        const float32x4_t v_c1j = vdupq_n_f32(c1j);
        const float32x4_t v_c2j = vdupq_n_f32(c2j);
        const float32x4_t v_c34j = vdupq_n_f32(c34j);
        const float32x4_t v_qr_j = vdupq_n_f32(qr_j);
        const float32x4_t v_neg2 = vdupq_n_f32(-2.0f);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        size_t k = 0;
        for (; k + 4 <= blk_count; k += 4) {
            float32x4_t v_pd;
            if constexpr (U32Dot) {
                v_pd = vcvtq_f32_u32(vld1q_u32(pd_u32 + k));
            } else {
                v_pd = vcvtq_f32_u32(vmovl_u16(vld1_u16(pd_u16 + k)));
            }

            float32x4_t v_sq = vld1q_f32(sum_q_f32 + k);

            float32x4_t fdt = vmlaq_f32(vmlaq_f32(vnegq_f32(v_c34j), v_c1j, v_pd), v_c2j, v_sq);

            float32x4_t v_or = vld1q_f32(or_c_l2sqr + k);
            float32x4_t v_dp = vld1q_f32(dp_mult + k);
            float32x4_t or_plus_qr = vaddq_f32(v_or, v_qr_j);
            float32x4_t result = vmlaq_f32(or_plus_qr, v_neg2, vmulq_f32(v_dp, fdt));

            vst1q_f32(out_partial_l2 + k, result);
        }
        for (; k < blk_count; ++k) {
            float dot_f;
            if constexpr (U32Dot) {
                dot_f = static_cast<float>(pd_u32[k]);
            } else {
                dot_f = static_cast<float>(pd_u16[k]);
            }
            const float fdt = c1j * dot_f + c2j * sum_q_f32[k] - c34j;
            out_partial_l2[k] = or_c_l2sqr[k] + qr_j - 2.0f * dp_mult[k] * fdt;
        }
    }

    /// Fused correction + compaction: no intermediate buffer store/load.
    template <bool U32Dot = false>
    static void RabitQCorrectionAndCompact(
        const void* partial_dot,
        float c1j,
        float c34j,
        float qr_j,
        float neg2_c2j,
        const float* or_c_l2sqr,
        const float* neg2_dp,
        const float* dp_sum_q,
        const float* threshold,
        uint32_t* survivor_positions,
        size_t& n_survivors,
        size_t blk_count
    ) {
        const float32x4_t v_c1j = vdupq_n_f32(c1j);
        const float32x4_t v_c34j = vdupq_n_f32(c34j);
        const float32x4_t v_qr_j = vdupq_n_f32(qr_j);
        const float32x4_t v_neg2c2j = vdupq_n_f32(neg2_c2j);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        n_survivors = 0;
        size_t k = 0;
        for (; k + 4 <= blk_count; k += 4) {
            float32x4_t v_pd;
            if constexpr (U32Dot) {
                v_pd = vcvtq_f32_u32(vld1q_u32(pd_u32 + k));
            } else {
                v_pd = vcvtq_f32_u32(vmovl_u16(vld1_u16(pd_u16 + k)));
            }

            // Chain A: base = or + qr + neg2_c2j * dp_sum_q
            float32x4_t base = vmlaq_f32(
                vaddq_f32(vld1q_f32(or_c_l2sqr + k), v_qr_j), v_neg2c2j, vld1q_f32(dp_sum_q + k)
            );

            // Chain B: shifted = c1j * dot - c34j
            float32x4_t shifted = vmlaq_f32(vnegq_f32(v_c34j), v_c1j, v_pd);

            // Merge: dist = neg2_dp * shifted + base
            float32x4_t result = vmlaq_f32(base, vld1q_f32(neg2_dp + k), shifted);

            float32x4_t thresh = vld1q_f32(threshold + k);
            uint32x4_t cmp_result = vcleq_f32(result, thresh);
            uint32_t any_passed = vmaxvq_u32(cmp_result);
            if (SKM_UNLIKELY(any_passed)) {
                uint32_t mask[4];
                vst1q_u32(mask, cmp_result);
                for (size_t i = 0; i < 4; ++i) {
                    survivor_positions[n_survivors] = static_cast<uint32_t>(k + i);
                    n_survivors += (mask[i] != 0);
                }
            }
        }
        for (; k < blk_count; ++k) {
            float dot_f;
            if constexpr (U32Dot) {
                dot_f = static_cast<float>(pd_u32[k]);
            } else {
                dot_f = static_cast<float>(pd_u16[k]);
            }
            const float base = or_c_l2sqr[k] + qr_j + neg2_c2j * dp_sum_q[k];
            const float shifted = c1j * dot_f - c34j;
            const float dist = neg2_dp[k] * shifted + base;
            survivor_positions[n_survivors] = static_cast<uint32_t>(k);
            n_survivors += dist <= threshold[k];
        }
    }

    template <bool WideAdd = false>
    static void ScanBlock(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot,
        size_t blk_count
    ) {
        if (blk_count == kBlockSize) {
            ScanBlockNeon(packed, lut, binary_bytes, out_dot);
            return;
        }
        ScalarFastScanComputer::ScanBlock<WideAdd>(packed, lut, binary_bytes, out_dot, blk_count);
    }

    /// Multi-block ScanBlock: delegates to per-block (NEON could be optimized later).
    template <int NBlocks>
    static void ScanBlockMulti(
        const uint8_t* const* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* const* out_dot
    ) {
        for (int b = 0; b < NBlocks; ++b) {
            ScanBlock(packed[b], lut, binary_bytes, out_dot[b], kBlockSize);
        }
    }

  private:
    /**
     * @brief NEON FastScan for nibble-split kPerm0-packed data.
     *
     * Loads lo/hi codes and LUTs as 4 separate 16B loads per byte position.
     * Combines lo+hi per-vector as u8, widens to u16 and accumulates.
     * Final vuzp deinterleave from kPerm0 to natural order.
     */
    static void ScanBlockNeon(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot
    ) {
        const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

        // 4 accumulators: A = lo-nibble extract (vecs kPerm0[j]),
        //                 B = hi-nibble extract (vecs kPerm0[j]+16)
        uint16x8_t acc_A_lo = vdupq_n_u16(0);
        uint16x8_t acc_A_hi = vdupq_n_u16(0);
        uint16x8_t acc_B_lo = vdupq_n_u16(0);
        uint16x8_t acc_B_hi = vdupq_n_u16(0);

        for (size_t b = 0; b < binary_bytes; ++b) {
            uint8x16_t lo_lut_vec = vld1q_u8(lut + b * 32);
            uint8x16_t hi_lut_vec = vld1q_u8(lut + b * 32 + 16);

            uint8x16_t lo_codes = vld1q_u8(packed + b * kBlockSize);
            uint8x16_t hi_codes = vld1q_u8(packed + b * kBlockSize + 16);

            // Extract indices: lo nibble = vecA, hi nibble = vecB
            uint8x16_t lo_idx_A = vandq_u8(lo_codes, mask_0f);
            uint8x16_t lo_idx_B = vshrq_n_u8(lo_codes, 4);
            uint8x16_t hi_idx_A = vandq_u8(hi_codes, mask_0f);
            uint8x16_t hi_idx_B = vshrq_n_u8(hi_codes, 4);

            // Lookups
            uint8x16_t res_lo_A = vqtbl1q_u8(lo_lut_vec, lo_idx_A);
            uint8x16_t res_hi_A = vqtbl1q_u8(hi_lut_vec, hi_idx_A);
            uint8x16_t res_lo_B = vqtbl1q_u8(lo_lut_vec, lo_idx_B);
            uint8x16_t res_hi_B = vqtbl1q_u8(hi_lut_vec, hi_idx_B);

            // Combine lo+hi for each vector set, widen and accumulate
            uint8x16_t total_A = vaddq_u8(res_lo_A, res_hi_A);
            uint8x16_t total_B = vaddq_u8(res_lo_B, res_hi_B);

            acc_A_lo = vaddw_u8(acc_A_lo, vget_low_u8(total_A));
            acc_A_hi = vaddw_u8(acc_A_hi, vget_high_u8(total_A));
            acc_B_lo = vaddw_u8(acc_B_lo, vget_low_u8(total_B));
            acc_B_hi = vaddw_u8(acc_B_hi, vget_high_u8(total_B));
        }

        // Deinterleave from kPerm0 order to natural order
        // acc_A_lo = {v0,v8,v1,v9,v2,v10,v3,v11}, acc_A_hi = {v4,v12,v5,v13,v6,v14,v7,v15}
        // vuzp1 takes even indices, vuzp2 takes odd indices
        uint16x8_t out_0_7 = vuzp1q_u16(acc_A_lo, acc_A_hi);   // {v0,v1,v2,v3,v4,v5,v6,v7}
        uint16x8_t out_8_15 = vuzp2q_u16(acc_A_lo, acc_A_hi);  // {v8,v9,...,v15}
        uint16x8_t out_16_23 = vuzp1q_u16(acc_B_lo, acc_B_hi); // {v16,...,v23}
        uint16x8_t out_24_31 = vuzp2q_u16(acc_B_lo, acc_B_hi); // {v24,...,v31}

        vst1q_u16(out_dot, out_0_7);
        vst1q_u16(out_dot + 8, out_8_15);
        vst1q_u16(out_dot + 16, out_16_23);
        vst1q_u16(out_dot + 24, out_24_31);
    }
};

class SIMDRaBitQCodec {
  public:
    /**
     * @brief NEON encode: 4 floats per iteration.
     * vcgtq_f32 → extract sign bits manually via shifts and ORs.
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t binary_bytes,
        const float* SKM_RESTRICT centroid
    ) {
        std::memset(code, 0, binary_bytes);

        float32x4_t norm_acc = vdupq_n_f32(0.0f);
        float32x4_t abs_acc = vdupq_n_f32(0.0f);
        const float32x4_t zero = vdupq_n_f32(0.0f);

        size_t j = 0;
        // Process 8 dims (2 × 4 NEON) per byte
        for (; j + 8 <= d; j += 8) {
            float32x4_t xv0 = vld1q_f32(x + j);
            float32x4_t cv0 = vld1q_f32(centroid + j);
            float32x4_t res0 = vsubq_f32(xv0, cv0);
            float32x4_t xv1 = vld1q_f32(x + j + 4);
            float32x4_t cv1 = vld1q_f32(centroid + j + 4);
            float32x4_t res1 = vsubq_f32(xv1, cv1);

            norm_acc = vfmaq_f32(norm_acc, res0, res0);
            norm_acc = vfmaq_f32(norm_acc, res1, res1);
            abs_acc = vaddq_f32(abs_acc, vabsq_f32(res0));
            abs_acc = vaddq_f32(abs_acc, vabsq_f32(res1));

            // Extract 4 sign bits from each group
            uint32x4_t cmp0 = vcgtq_f32(res0, zero);
            uint32x4_t cmp1 = vcgtq_f32(res1, zero);

            // Narrow to 16-bit, then 8-bit to pack bits
            uint16x4_t n0 = vmovn_u32(cmp0);
            uint16x4_t n1 = vmovn_u32(cmp1);
            uint8x8_t n8 = vmovn_u16(vcombine_u16(n0, n1));
            // n8 has 8 bytes, each 0x00 or 0xFF. Extract MSBs.
            static const uint8_t shift_vals[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            uint8x8_t shifts = vld1_u8(shift_vals);
            // Convert 0xFF→1, 0x00→0, then shift each to its bit position
            uint8x8_t bits = vshr_n_u8(n8, 7); // 0xFF→1, 0x00→0
            uint8x8_t shifted = vshl_u8(bits, vreinterpret_s8_u8(shifts));
            // Horizontal OR to combine 8 bits into one byte
            uint8_t byte_val = shifted[0] | shifted[1] | shifted[2] | shifted[3] | shifted[4] |
                               shifted[5] | shifted[6] | shifted[7];
            code[j / 8] = byte_val;
        }

        // Scalar tail
        float norm_tail = 0.0f, abs_tail = 0.0f;
        for (; j < d; ++j) {
            const float res = x[j] - centroid[j];
            norm_tail += res * res;
            abs_tail += std::abs(res);
            if (res > 0.0f) {
                code[j / 8] |= static_cast<uint8_t>(1 << (j % 8));
            }
        }

        const float norm_L2sqr = vaddvq_f32(norm_acc) + norm_tail;
        const float dp_oO = vaddvq_f32(abs_acc) + abs_tail;
        const float sqrt_d = std::sqrt(static_cast<float>(d));

        const float or_minus_c_l2sqr = norm_L2sqr;
        const float dp_multiplier = dp_oO > 0.0f ? norm_L2sqr * sqrt_d / dp_oO : 0.0f;
        std::memcpy(code + binary_bytes, &or_minus_c_l2sqr, sizeof(float));
        std::memcpy(code + binary_bytes + sizeof(float), &dp_multiplier, sizeof(float));
    }

    /**
     * @brief NEON decode: expand 4 sign bits to mask, blend ±0.5*scale, add centroid.
     */
    static void DecodeOne(
        const uint8_t* SKM_RESTRICT code,
        float* SKM_RESTRICT x,
        size_t d,
        size_t binary_bytes,
        const float* SKM_RESTRICT centroid
    ) {
        float dp_multiplier;
        std::memcpy(&dp_multiplier, code + binary_bytes + sizeof(float), sizeof(float));
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float scale = dp_multiplier * 2.0f * inv_sqrt_d;

        const float32x4_t pos = vdupq_n_f32(+0.5f * scale);
        const float32x4_t neg = vdupq_n_f32(-0.5f * scale);

        size_t j = 0;
        for (; j + 4 <= d; j += 4) {
            // Extract 4 bits starting at bit position j
            size_t byte_idx = j / 8;
            size_t bit_off = j % 8;
            uint32_t bits;
            if (bit_off + 4 <= 8) {
                bits = (code[byte_idx] >> bit_off) & 0x0F;
            } else {
                uint16_t two_bytes = *(const uint16_t*) (code + byte_idx);
                bits = (two_bytes >> bit_off) & 0x0F;
            }

            // Expand 4 bits to 4 × uint32 masks
            uint32_t m0 = (bits & 1) ? 0xFFFFFFFF : 0;
            uint32_t m1 = (bits & 2) ? 0xFFFFFFFF : 0;
            uint32_t m2 = (bits & 4) ? 0xFFFFFFFF : 0;
            uint32_t m3 = (bits & 8) ? 0xFFFFFFFF : 0;
            uint32x4_t mask = {m0, m1, m2, m3};

            float32x4_t val = vbslq_f32(mask, pos, neg);
            float32x4_t cv = vld1q_f32(centroid + j);
            vst1q_f32(x + j, vaddq_f32(val, cv));
        }

        // Scalar tail
        for (; j < d; ++j) {
            const float bit = ((code[j / 8] >> (j % 8)) & 1) ? 1.0f : 0.0f;
            x[j] = (bit - 0.5f) * scale + centroid[j];
        }
    }
};

class SIMDLVQ4Codec {
  public:
    /**
     * @brief NEON LVQ4 encode: min/max reduction 4 floats/iter, quantize 8 floats/iter, pack via
     * vuzp.
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t nibble_bytes
    ) {
        // Min/max reduction: 4 floats per iteration
        float32x4_t v_min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
        float32x4_t v_max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
        size_t j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t v = vld1q_f32(x + j);
            v_min_vec = vminq_f32(v_min_vec, v);
            v_max_vec = vmaxq_f32(v_max_vec, v);
        }
        float v_min = vminvq_f32(v_min_vec);
        float v_max = vmaxvq_f32(v_max_vec);
        for (; j < d; ++j) {
            v_min = std::min(v_min, x[j]);
            v_max = std::max(v_max, x[j]);
        }

        float range = v_max - v_min;
        if (range < 1e-30f)
            range = 1e-30f;
        const float scale = range / 15.0f;
        const float inv_scale = 1.0f / scale;
        const float bias = v_min;

        const float32x4_t v_bias_vec = vdupq_n_f32(bias);
        const float32x4_t v_inv_scale_vec = vdupq_n_f32(inv_scale);
        const int32x4_t v_zero4 = vdupq_n_s32(0);
        const int32x4_t v_fifteen4 = vdupq_n_s32(15);

        j = 0;
        size_t out_off = 0;
        for (; j + 8 <= d; j += 8, out_off += 4) {
            float32x4_t v0 = vld1q_f32(x + j);
            float32x4_t v1 = vld1q_f32(x + j + 4);
            float32x4_t q0 = vmulq_f32(vsubq_f32(v0, v_bias_vec), v_inv_scale_vec);
            float32x4_t q1 = vmulq_f32(vsubq_f32(v1, v_bias_vec), v_inv_scale_vec);

            // Round to nearest and convert to int
            int32x4_t qi0 = vcvtnq_s32_f32(q0);
            int32x4_t qi1 = vcvtnq_s32_f32(q1);
            qi0 = vmaxq_s32(qi0, v_zero4);
            qi0 = vminq_s32(qi0, v_fifteen4);
            qi1 = vmaxq_s32(qi1, v_zero4);
            qi1 = vminq_s32(qi1, v_fifteen4);

            // Narrow 8 × int32 → 8 × uint8
            uint16x4_t n0 = vmovn_u32(vreinterpretq_u32_s32(qi0));
            uint16x4_t n1 = vmovn_u32(vreinterpretq_u32_s32(qi1));
            uint8x8_t n8 = vmovn_u16(vcombine_u16(n0, n1));

            // Pack pairs via vuzp: even→lo nibbles, odd→hi nibbles
            uint8x8x2_t pairs = vuzp_u8(n8, vdup_n_u8(0));
            uint8x8_t packed = vorr_u8(pairs.val[0], vshl_n_u8(pairs.val[1], 4));
            vst1_lane_u32((uint32_t*) (code + out_off), vreinterpret_u32_u8(packed), 0);
        }
        // Scalar tail
        for (; j + 2 <= d; j += 2, ++out_off) {
            int q_lo = static_cast<int>(std::lround((x[j] - bias) * inv_scale));
            int q_hi = static_cast<int>(std::lround((x[j + 1] - bias) * inv_scale));
            q_lo = std::max(0, std::min(15, q_lo));
            q_hi = std::max(0, std::min(15, q_hi));
            code[out_off] = static_cast<uint8_t>(q_lo | (q_hi << 4));
        }

        std::memcpy(code + nibble_bytes, &scale, sizeof(float));
        std::memcpy(code + nibble_bytes + sizeof(float), &bias, sizeof(float));
    }

    /**
     * @brief NEON LVQ4 decode: 4 packed bytes → 8 floats per iteration via vzip interleave.
     */
    static void DecodeOne(
        const uint8_t* SKM_RESTRICT code,
        float* SKM_RESTRICT x,
        size_t d,
        size_t nibble_bytes
    ) {
        float scale, bias;
        std::memcpy(&scale, code + nibble_bytes, sizeof(float));
        std::memcpy(&bias, code + nibble_bytes + sizeof(float), sizeof(float));

        const float32x4_t v_scale = vdupq_n_f32(scale);
        const float32x4_t v_bias = vdupq_n_f32(bias);

        size_t b = 0;
        for (; b + 4 <= nibble_bytes; b += 4) {
            uint32_t four_bytes = *(const uint32_t*) (code + b);
            uint8x8_t packed = vreinterpret_u8_u32(vdup_n_u32(four_bytes));

            uint8x8_t lo = vand_u8(packed, vdup_n_u8(0x0F));
            uint8x8_t hi = vshr_n_u8(packed, 4);

            // Interleave: [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]
            uint8x8x2_t zipped = vzip_u8(lo, hi);
            uint16x8_t wide16 = vmovl_u8(zipped.val[0]);

            uint32x4_t wide32_lo = vmovl_u16(vget_low_u16(wide16));
            float32x4_t floats_lo = vcvtq_f32_u32(wide32_lo);
            vst1q_f32(x + b * 2, vmlaq_f32(v_bias, floats_lo, v_scale));

            uint32x4_t wide32_hi = vmovl_u16(vget_high_u16(wide16));
            float32x4_t floats_hi = vcvtq_f32_u32(wide32_hi);
            vst1q_f32(x + b * 2 + 4, vmlaq_f32(v_bias, floats_hi, v_scale));
        }
        // Scalar tail
        for (; b < nibble_bytes; ++b) {
            x[2 * b] = scale * static_cast<float>(code[b] & 0x0F) + bias;
            x[2 * b + 1] = scale * static_cast<float>(code[b] >> 4) + bias;
        }
    }
};

} // namespace skmeans
