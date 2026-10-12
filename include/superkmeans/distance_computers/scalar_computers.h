#pragma once

#include <cassert>
#include <cstring>

#include "superkmeans/common.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class ScalarComputer {};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::sq8> {

  public:
    using distance_t = pdx_distance_t<Quantization::sq8>;
    using data_t = skmeans_value_t<Quantization::sq8>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0;
        SKM_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            int diff = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
            distance += diff * diff;
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<Quantization::f32>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0.0;
        SKM_VECTORIZE_LOOP
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance_t to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::dp, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<Quantization::f32>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0.0;
        SKM_VECTORIZE_LOOP
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::sq4> {
  public:
    using distance_t = pdx_distance_t<Quantization::sq4>;
    using data_t = skmeans_value_t<Quantization::sq4>;

    /**
     * @brief Computes L2² distance between two packed u4x2 vectors (scalar).
     * Adapted from nk_sqeuclidean_u4_serial in NumKong.
     * @param vector1 Packed u4x2 input vector 1
     * @param vector2 Packed u4x2 input vector 2
     * @param num_packed_bytes Number of packed bytes to process (each byte = 2 dims)
     * @return L2² distance as uint32_t
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_packed_bytes
    ) {
        distance_t distance = 0;
        for (size_t i = 0; i < num_packed_bytes; ++i) {
            int32_t a_lo = vector1[i] & 0x0F;
            int32_t b_lo = vector2[i] & 0x0F;
            int32_t a_hi = (vector1[i] >> 4) & 0x0F;
            int32_t b_hi = (vector2[i] >> 4) & 0x0F;
            int32_t diff_lo = a_lo - b_lo;
            int32_t diff_hi = a_hi - b_hi;
            distance += static_cast<uint32_t>(diff_lo * diff_lo + diff_hi * diff_hi);
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::rabitq> {
  public:
    using distance_t = pdx_distance_t<Quantization::rabitq>;
    using data_t = skmeans_value_t<Quantization::rabitq>;

    /**
     * @brief Computes popcount(a AND b) — binary inner product.
     * @param vector1 Binary vector 1 (packed bytes)
     * @param vector2 Binary vector 2 (packed bytes)
     * @param num_bytes Number of bytes to process (d/8 for d-bit vectors)
     * @return popcount of bitwise AND
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_bytes
    ) {
        uint32_t count = 0;
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(vector1);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(vector2);
        size_t n_words = num_bytes / 8;
        for (size_t i = 0; i < n_words; ++i) {
            count += static_cast<uint32_t>(SKM_POPCOUNT64(a64[i] & b64[i]));
        }
        for (size_t i = n_words * 8; i < num_bytes; ++i) {
            count += static_cast<uint32_t>(SKM_POPCOUNT(vector1[i] & vector2[i]));
        }
        return count;
    };

    /**
     * @brief Fused multi-bitplane popcount on chunk-interleaved layout.
     * Layout: for each 16B data chunk, qb×16B of bitplane data contiguous.
     */
    static uint32_t HorizontalMultiPlane(
        const data_t* SKM_RESTRICT data,
        const data_t* planes_interleaved,
        size_t num_bytes,
        int qb
    ) {
        uint32_t result = 0;
        size_t i = 0;
        // Process 16-byte chunks using uint64 words
        for (; i + 16 <= num_bytes; i += 16) {
            const uint64_t* d64 = reinterpret_cast<const uint64_t*>(data + i);
            for (int w = 0; w < 2; ++w) {
                uint64_t x = d64[w];
                for (int bp = 0; bp < qb; ++bp) {
                    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(
                        planes_interleaved + i * qb + static_cast<size_t>(bp) * 16
                    );
                    result += static_cast<uint32_t>(SKM_POPCOUNT64(x & p64[w])) << bp;
                }
            }
        }
        // Byte tail
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
class ScalarUtilsComputer {
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
        for (size_t i = 0; i < n_vectors; ++i) {
            pruning_positions[n_vectors_not_pruned] = i;
            n_vectors_not_pruned += pruning_distances[i] < pruning_threshold;
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
class ScalarUtilsComputer<Quantization::f32> {
  public:
    using data_t = skmeans_value_t<Quantization::f32>;

    /**
     * @brief Flip sign of floats based on a mask (single vector).
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        SKM_VECTORIZE_LOOP
        for (size_t j = 0; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors (scalar fallback).
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
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
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
class ScalarUtilsComputer<Quantization::sq4> {
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
        for (size_t i = 0; i < n_vectors; ++i) {
            pruning_positions[n_vectors_not_pruned] = i;
            n_vectors_not_pruned += pruning_distances[i] < pruning_threshold;
        }
    }

    /**
     * @brief Pack u8 values [0,15] into u4x2 format (two nibbles per byte).
     *
     * dst[k] = (src[2k] & 0x0F) | ((src[2k+1] & 0x0F) << 4)
     *
     * @param src Input u8 array (count elements, each in [0,15])
     * @param dst Output u4x2 array (count/2 bytes)
     * @param count Number of input u8 elements (must be even)
     */
    static void PackU8ToU4x2(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        SKM_VECTORIZE_LOOP
        for (size_t k = 0; k < n_packed; ++k) {
            dst[k] = (src[2 * k] & 0x0F) | ((src[2 * k + 1] & 0x0F) << 4);
        }
    }

    static void UnpackU4x2ToU8(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        for (size_t k = 0; k < n_packed; ++k) {
            dst[2 * k] = src[k] & 0x0F;
            dst[2 * k + 1] = (src[k] >> 4) & 0x0F;
        }
    }
};

class ScalarFastScanComputer {
  public:
    static constexpr size_t kBlockSize = 32;

    /**
     * @brief Compute RaBitQ partial L2 distances for a block of points against one centroid.
     *
     * For each point k in [0, blk_count):
     *   fdt = c1j * float(partial_dot[k]) + c2j * float(sum_q[k]) - c34j
     *   out[k] = or_c_l2sqr[k] + qr_j - 2 * dp_mult[k] * fdt
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     */
    /**
     * @brief Compute RaBitQ partial L2 distances for a block of points against one centroid.
     *
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
        for (size_t k = 0; k < blk_count; ++k) {
            float dot_f;
            if constexpr (U32Dot) {
                dot_f = static_cast<float>(static_cast<const uint32_t*>(partial_dot)[k]);
            } else {
                dot_f = static_cast<float>(static_cast<const uint16_t*>(partial_dot)[k]);
            }
            const float fdt = c1j * dot_f + c2j * sum_q_f32[k] - c34j;
            out_partial_l2[k] = or_c_l2sqr[k] + qr_j - 2.0f * dp_mult[k] * fdt;
        }
    }

    /**
     * @brief Fused RaBitQ correction + survivor compaction (scalar fallback).
     *
     * Computes partial L2 distances and immediately compacts survivors whose
     * distance <= threshold, avoiding the intermediate partial_l2 buffer.
     *
     * Uses refactored formula with precomputed centroid-independent values:
     *   base[k]    = or_c_l2sqr[k] + qr_j + neg2_c2j * dp_sum_q[k]   (dot-independent)
     *   shifted[k] = c1j * dot[k] - c34j                               (dot-dependent)
     *   dist[k]    = neg2_dp[k] * shifted[k] + base[k]
     *
     * @param neg2_c2j  Precomputed -2 * c2j (per-centroid scalar)
     * @param neg2_dp   Precomputed -2 * dp_mult[k] (per-element, centroid-independent)
     * @param dp_sum_q  Precomputed dp_mult[k] * sum_q[k] (per-element, centroid-independent)
     */
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
        n_survivors = 0;
        for (size_t k = 0; k < blk_count; ++k) {
            float dot_f;
            if constexpr (U32Dot) {
                dot_f = static_cast<float>(static_cast<const uint32_t*>(partial_dot)[k]);
            } else {
                dot_f = static_cast<float>(static_cast<const uint16_t*>(partial_dot)[k]);
            }
            const float base = or_c_l2sqr[k] + qr_j + neg2_c2j * dp_sum_q[k];
            const float shifted = c1j * dot_f - c34j;
            const float dist = neg2_dp[k] * shifted + base;
            if (dist <= threshold[k]) {
                survivor_positions[n_survivors++] = static_cast<uint32_t>(k);
            }
        }
    }

    /// kPerm0 interleaving used by nibble-split TransposeBlock.
    static constexpr int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    template <bool WideAdd = false>
    static void ScanBlock(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot,
        size_t blk_count
    ) {
        (void) sizeof(WideAdd);
        (void) blk_count;
        std::memset(out_dot, 0, kBlockSize * sizeof(uint16_t));
        for (size_t b = 0; b < binary_bytes; ++b) {
            const uint8_t* lut_lo = lut + b * 32;
            const uint8_t* lut_hi = lut + b * 32 + 16;
            const uint8_t* row = packed + b * kBlockSize;
            for (int j = 0; j < 16; ++j) {
                const uint8_t lo_byte = row[j];
                const uint8_t hi_byte = row[j + 16];
                const int vA = kPerm0[j];
                const int vB = kPerm0[j] + 16;
                out_dot[vA] += static_cast<uint16_t>(lut_lo[lo_byte & 0x0F]) +
                               static_cast<uint16_t>(lut_hi[hi_byte & 0x0F]);
                out_dot[vB] += static_cast<uint16_t>(lut_lo[lo_byte >> 4]) +
                               static_cast<uint16_t>(lut_hi[hi_byte >> 4]);
            }
        }
    }

    /// Scalar fallback: just call ScanBlock per block.
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
};

class ScalarRaBitQCodec {
  public:
    /**
     * @brief Encode one float vector into RaBitQ 1-bit binary code + factors.
     *
     * Code layout: [(d+7)/8 sign bytes][or_minus_c_l2sqr: 4B][dp_multiplier: 4B]
     * sign_bit[j] = 1 iff (x[j] - centroid[j]) > 0
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t binary_bytes,
        const float* SKM_RESTRICT centroid
    ) {
        std::memset(code, 0, binary_bytes);

        float norm_L2sqr = 0.0f;
        float dp_oO = 0.0f;

        for (size_t j = 0; j < d; ++j) {
            const float res = x[j] - centroid[j];
            norm_L2sqr += res * res;
            dp_oO += std::abs(res);
            if (res > 0.0f) {
                code[j / 8] |= static_cast<uint8_t>(1 << (j % 8));
            }
        }

        const float sqrt_d = std::sqrt(static_cast<float>(d));
        const float or_minus_c_l2sqr = norm_L2sqr;
        const float dp_multiplier = dp_oO > 0.0f ? norm_L2sqr * sqrt_d / dp_oO : 0.0f;
        std::memcpy(code + binary_bytes, &or_minus_c_l2sqr, sizeof(float));
        std::memcpy(code + binary_bytes + sizeof(float), &dp_multiplier, sizeof(float));
    }

    /**
     * @brief Decode one RaBitQ code back to float vector.
     *
     * x[j] = (sign_bit[j] ? +0.5 : -0.5) * dp_multiplier * 2 / sqrt(d) + centroid[j]
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

        for (size_t j = 0; j < d; ++j) {
            const float bit = ((code[j / 8] >> (j % 8)) & 1) ? 1.0f : 0.0f;
            x[j] = (bit - 0.5f) * scale + centroid[j];
        }
    }
};

class ScalarLVQ4Codec {
  public:
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t nibble_bytes
    ) {
        float v_min = x[0], v_max = x[0];
        for (size_t dim = 1; dim < d; ++dim) {
            v_min = std::min(v_min, x[dim]);
            v_max = std::max(v_max, x[dim]);
        }

        float range = v_max - v_min;
        if (range < 1e-30f)
            range = 1e-30f;
        float scale = range / 15.0f;
        float inv_scale = 1.0f / scale;
        float bias = v_min;

        for (size_t dim = 0; dim < d; dim += 2) {
            int q_lo = static_cast<int>(std::lround((x[dim] - bias) * inv_scale));
            int q_hi = static_cast<int>(std::lround((x[dim + 1] - bias) * inv_scale));
            q_lo = std::max(0, std::min(15, q_lo));
            q_hi = std::max(0, std::min(15, q_hi));
            code[dim / 2] = static_cast<uint8_t>(q_lo | (q_hi << 4));
        }

        std::memcpy(code + nibble_bytes, &scale, sizeof(float));
        std::memcpy(code + nibble_bytes + sizeof(float), &bias, sizeof(float));
    }

    static void DecodeOne(
        const uint8_t* SKM_RESTRICT code,
        float* SKM_RESTRICT x,
        size_t d,
        size_t nibble_bytes
    ) {
        float scale, bias;
        std::memcpy(&scale, code + nibble_bytes, sizeof(float));
        std::memcpy(&bias, code + nibble_bytes + sizeof(float), sizeof(float));

        for (size_t b = 0; b < nibble_bytes; ++b) {
            x[2 * b] = scale * static_cast<float>(code[b] & 0x0F) + bias;
            x[2 * b + 1] = scale * static_cast<float>(code[b] >> 4) + bias;
        }
    }
};

} // namespace skmeans
