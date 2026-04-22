#pragma once

#include <cassert>
#include <iostream>

#include "arm_neon.h"
#include "superkmeans/common.h"

namespace skmeans {

// Equivalent of vdotq_u32(acc, a, a) for squared accumulation.
static inline uint32x4_t squared_dot_accumulate(uint32x4_t acc, uint8x16_t a) {
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
class SIMDComputer<DistanceFunction::l2, Quantization::u8> {
  public:
    using distance_t = pdx_distance_t<Quantization::u8>;
    using value_t = skmeans_value_t<Quantization::u8>;

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
            sum_vec = squared_dot_accumulate(sum_vec, d_vec);
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
    using distance_t = skmeans_distance_t<Quantization::f32>;
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
class SIMDComputer<DistanceFunction::l2, Quantization::u4> {
  public:
    using distance_t = pdx_distance_t<Quantization::u4>;
    using data_t = skmeans_value_t<Quantization::u4>;

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
            sum_vec = squared_dot_accumulate(sum_vec, diff_lo);
            sum_vec = squared_dot_accumulate(sum_vec, diff_hi);
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

template <Quantization q>
class SIMDUtilsComputer {};

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
};

template <>
class SIMDUtilsComputer<Quantization::u8> {
  public:
    using data_t = skmeans_value_t<Quantization::u8>;
    using pdx_dist_t = pdx_distance_t<Quantization::u8>;

    static void FlipSign(const data_t*, data_t*, const uint32_t*, size_t) {
        assert(false && "FlipSign not supported for u8");
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
        assert(false && "PackU8ToU4x2 not applicable for u8");
    }
};

template <>
class SIMDUtilsComputer<Quantization::u4> {
  public:
    using data_t = skmeans_value_t<Quantization::u4>;
    using pdx_dist_t = pdx_distance_t<Quantization::u4>;

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
};

} // namespace skmeans
