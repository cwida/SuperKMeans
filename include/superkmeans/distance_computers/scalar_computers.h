#pragma once

#include <cassert>

#include "superkmeans/common.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class ScalarComputer {};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::u8> {

  public:
    using distance_t = pdx_distance_t<Quantization::u8>;
    using data_t = skmeans_value_t<Quantization::u8>;

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
    using distance_t = skmeans_distance_t<Quantization::f32>;
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
    using distance_t = skmeans_distance_t<Quantization::f32>;
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
class ScalarComputer<DistanceFunction::l2, Quantization::u4> {
  public:
    using distance_t = pdx_distance_t<Quantization::u4>;
    using data_t = skmeans_value_t<Quantization::u4>;

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

template <Quantization q>
class ScalarUtilsComputer {};

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
};

template <>
class ScalarUtilsComputer<Quantization::u8> {
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
        for (size_t i = 0; i < n_vectors; ++i) {
            pruning_positions[n_vectors_not_pruned] = i;
            n_vectors_not_pruned += pruning_distances[i] < pruning_threshold;
        }
    }

    static void PackU8ToU4x2(const uint8_t*, uint8_t*, size_t) {
        assert(false && "PackU8ToU4x2 not applicable for u8");
    }
};

template <>
class ScalarUtilsComputer<Quantization::u4> {
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
};

} // namespace skmeans
