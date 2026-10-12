#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/scalar_computers.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::sq8> {
  public:
    using distance_t = pdx_distance_t<skmeans::Quantization::sq8>;
    using data_t = skmeans_value_t<skmeans::Quantization::sq8>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m256i d2_vec = _mm256_setzero_si256();
        __m256i zeros = _mm256_setzero_si256();
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector1 + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector2 + i));
            __m256i diff =
                _mm256_or_si256(_mm256_subs_epu8(a_vec, b_vec), _mm256_subs_epu8(b_vec, a_vec));
            __m256i lo16 = _mm256_unpacklo_epi8(diff, zeros);
            __m256i hi16 = _mm256_unpackhi_epi8(diff, zeros);
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(lo16, lo16));
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(hi16, hi16));
        }
        // Reduce 8 x i32 to scalar (simsimd_reduce_i32x8_haswell)
        __m128i lo = _mm256_castsi256_si128(d2_vec);
        __m128i hi = _mm256_extracti128_si256(d2_vec, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        distance_t distance = _mm_cvtsi128_si32(sum128);
        // Scalar tail.
        for (; i < num_dimensions; ++i) {
            int n = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
            distance += n * n;
        }
        return distance;
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<skmeans::Quantization::f32>;
    using scalar_computer =
        ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;

    /**
     * @brief Computes the L2 distance between two float vectors using AVX2.
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
        __m256 d2_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= num_dimensions; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(vector1 + i);
            __m256 b_vec = _mm256_loadu_ps(vector2 + i);
            __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
            d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
        }

        // _simsimd_reduce_f32x8_haswell
        // Convert the lower and higher 128-bit lanes of the input vector to double precision
        __m128 low_f32 = _mm256_castps256_ps128(d2_vec);
        __m128 high_f32 = _mm256_extractf128_ps(d2_vec, 1);

        // Convert single-precision (float) vectors to double-precision (double) vectors
        __m256d low_f64 = _mm256_cvtps_pd(low_f32);
        __m256d high_f64 = _mm256_cvtps_pd(high_f32);

        // Perform the addition in double-precision
        __m256d sum = _mm256_add_pd(low_f64, high_f64);

        // Reduce the double-precision vector to a scalar
        // Horizontal add the first and second double-precision values, and third and fourth
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum128 = _mm_add_pd(sum_low, sum_high);

        // Horizontal add again to accumulate all four values into one
        sum128 = _mm_hadd_pd(sum128, sum128);

        // Convert the final sum to a scalar double-precision value and return
        double d2 = _mm_cvtsd_f64(sum128);

        SKM_VECTORIZE_LOOP
        for (; i < num_dimensions; ++i) {
            float d = vector1[i] - vector2[i];
            d2 += d * d;
        }

        return static_cast<distance_t>(d2); // NOLINT(bugprone-narrowing-conversions)
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::dp, skmeans::Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<skmeans::Quantization::f32>;

    /**
     * @brief Computes the Dot Product of two float vectors using AVX2.
     * Taken from: https://github.com/ashvardanian/SimSIMD
     * @param vector1 Input vector 1
     * @param vector2 Input vector 2
     * @param num_dimensions Number of dimensions
     * @return Dot Product between the two vectors
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m256 d2_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= num_dimensions; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(vector1 + i);
            __m256 b_vec = _mm256_loadu_ps(vector2 + i);
            d2_vec = _mm256_fmadd_ps(a_vec, b_vec, d2_vec);
        }

        // _simsimd_reduce_f32x8_haswell
        // Convert the lower and higher 128-bit lanes of the input vector to double precision
        __m128 low_f32 = _mm256_castps256_ps128(d2_vec);
        __m128 high_f32 = _mm256_extractf128_ps(d2_vec, 1);

        // Convert single-precision (float) vectors to double-precision (double) vectors
        __m256d low_f64 = _mm256_cvtps_pd(low_f32);
        __m256d high_f64 = _mm256_cvtps_pd(high_f32);

        // Perform the addition in double-precision
        __m256d sum = _mm256_add_pd(low_f64, high_f64);

        // Reduce the double-precision vector to a scalar
        // Horizontal add the first and second double-precision values, and third and fourth
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum128 = _mm_add_pd(sum_low, sum_high);

        // Horizontal add again to accumulate all four values into one
        sum128 = _mm_hadd_pd(sum128, sum128);

        // Convert the final sum to a scalar double-precision value and return
        double d2 = _mm_cvtsd_f64(sum128);

        for (; i < num_dimensions; ++i) {
            d2 += vector1[i] * vector2[i];
        }
        return static_cast<distance_t>(d2); // NOLINT(bugprone-narrowing-conversions)
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::sq4> {
  public:
    using distance_t = pdx_distance_t<skmeans::Quantization::sq4>;
    using data_t = skmeans_value_t<skmeans::Quantization::sq4>;

    /**
     * @brief Computes L2² distance between two packed u4x2 vectors using AVX2.
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
        __m256i d2_vec = _mm256_setzero_si256();
        const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
        const __m256i ones_16 = _mm256_set1_epi16(1);
        size_t i = 0;
        for (; i + 32 <= num_packed_bytes; i += 32) {
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector1 + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector2 + i));
            // Extract low nibbles
            __m256i a_lo = _mm256_and_si256(a_vec, nibble_mask);
            __m256i b_lo = _mm256_and_si256(b_vec, nibble_mask);
            // Extract high nibbles
            __m256i a_hi = _mm256_and_si256(_mm256_srli_epi16(a_vec, 4), nibble_mask);
            __m256i b_hi = _mm256_and_si256(_mm256_srli_epi16(b_vec, 4), nibble_mask);
            // Absolute diff via saturating subtraction: |a-b| = (a⊖b) | (b⊖a)
            __m256i diff_lo =
                _mm256_or_si256(_mm256_subs_epu8(a_lo, b_lo), _mm256_subs_epu8(b_lo, a_lo));
            __m256i diff_hi =
                _mm256_or_si256(_mm256_subs_epu8(a_hi, b_hi), _mm256_subs_epu8(b_hi, a_hi));
            // Square: maddubs treats first arg as unsigned, second as signed.
            // Since diff values are in [0,15], signed interpretation is fine.
            __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo); // u16 sums of pairs
            __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);
            // Accumulate u16 → u32
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(sq_lo, ones_16));
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(sq_hi, ones_16));
        }
        // Horizontal reduce 8 × i32
        __m128i lo = _mm256_castsi256_si128(d2_vec);
        __m128i hi = _mm256_extracti128_si256(d2_vec, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        distance_t distance = static_cast<distance_t>(_mm_cvtsi128_si32(sum128));
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
class SIMDComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::rabitq> {
  public:
    using distance_t = pdx_distance_t<skmeans::Quantization::rabitq>;
    using data_t = skmeans_value_t<skmeans::Quantization::rabitq>;

    /**
     * @brief Computes popcount(a AND b) — binary inner product using AVX2.
     * Uses VPSHUFB nibble lookup table for byte-level popcount.
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_bytes
    ) {
        // clang-format off
        const __m256i lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        // clang-format on
        const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
        __m256i acc = _mm256_setzero_si256();
        size_t i = 0;
        for (; i + 32 <= num_bytes; i += 32) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector1 + i));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector2 + i));
            __m256i v = _mm256_and_si256(va, vb);
            __m256i lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(v, nibble_mask));
            __m256i hi =
                _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble_mask));
            __m256i byte_cnt = _mm256_add_epi8(lo, hi);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(byte_cnt, _mm256_setzero_si256()));
        }
        // Reduce 4 × u64 → scalar
        __m128i lo128 = _mm256_castsi256_si128(acc);
        __m128i hi128 = _mm256_extracti128_si256(acc, 1);
        __m128i sum128 = _mm_add_epi64(lo128, hi128);
        __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
        uint32_t count = static_cast<uint32_t>(_mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)));
        // Scalar tail
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
        // clang-format off
        const __m256i lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        // clang-format on
        const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
        const __m256i zero = _mm256_setzero_si256();
        __m256i acc = zero;
        size_t i = 0;
        // Process 16 bytes at a time: broadcast x, load 2 bitplanes per 256-bit
        for (; i + 16 <= num_bytes; i += 16) {
            __m128i x128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
            __m256i x = _mm256_broadcastsi128_si256(x128);
            const uint8_t* chunk = planes_interleaved + i * qb;
            // bp0+bp1 in one 256-bit load
            __m256i p01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chunk));
            __m256i v01 = _mm256_and_si256(x, p01);
            __m256i lo01 = _mm256_shuffle_epi8(lookup, _mm256_and_si256(v01, nibble_mask));
            __m256i hi01 = _mm256_shuffle_epi8(
                lookup, _mm256_and_si256(_mm256_srli_epi16(v01, 4), nibble_mask)
            );
            __m256i cnt01 = _mm256_sad_epu8(_mm256_add_epi8(lo01, hi01), zero);
            // lane0,1 = bp0 popcounts (shift 0), lane2,3 = bp1 popcounts (shift 1)
            __m256i shift01 = _mm256_set_epi64x(1, 1, 0, 0);
            acc = _mm256_add_epi64(acc, _mm256_sllv_epi64(cnt01, shift01));
            // bp2+bp3
            __m256i p23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chunk + 32));
            __m256i v23 = _mm256_and_si256(x, p23);
            __m256i lo23 = _mm256_shuffle_epi8(lookup, _mm256_and_si256(v23, nibble_mask));
            __m256i hi23 = _mm256_shuffle_epi8(
                lookup, _mm256_and_si256(_mm256_srli_epi16(v23, 4), nibble_mask)
            );
            __m256i cnt23 = _mm256_sad_epu8(_mm256_add_epi8(lo23, hi23), zero);
            __m256i shift23 = _mm256_set_epi64x(3, 3, 2, 2);
            acc = _mm256_add_epi64(acc, _mm256_sllv_epi64(cnt23, shift23));
        }
        __m128i lo128 = _mm256_castsi256_si128(acc);
        __m128i hi128 = _mm256_extracti128_si256(acc, 1);
        __m128i sum128 = _mm_add_epi64(lo128, hi128);
        __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
        uint32_t result = static_cast<uint32_t>(_mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)));
        // Scalar tail
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
        constexpr size_t k_simd_width = 8;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        const __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
        __m256i threshold_vec =
            _mm256_sub_epi32(_mm256_set1_epi32(static_cast<int32_t>(pruning_threshold)), bias);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m256i distances =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pruning_distances + vector_idx)
                );
            __m256i biased = _mm256_sub_epi32(distances, bias);
            __m256i cmp_result = _mm256_cmpgt_epi32(threshold_vec, biased);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));
            if (SKM_UNLIKELY(mask)) {
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask >> i) & 1;
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
class SIMDUtilsComputer<skmeans::Quantization::f32> {
  public:
    using data_t = skmeans_value_t<skmeans::Quantization::f32>;

    /**
     * @brief Flip sign of floats based on a mask using AVX2.
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 vec = _mm256_loadu_ps(data + j);
            __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(masks + j));
            __m256i vec_i = _mm256_castps_si256(vec);
            vec_i = _mm256_xor_si256(vec_i, mask);
            _mm256_storeu_ps(out + j, _mm256_castsi256_ps(vec_i));
        }
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        for (; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors using AVX2.
     *
     * Optimized for cases where only ~2% of vectors pass the threshold test.
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
        constexpr size_t k_simd_width = 8;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        __m256 threshold_vec = _mm256_set1_ps(pruning_threshold);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m256 distances = _mm256_loadu_ps(pruning_distances + vector_idx);
            __m256 cmp_result = _mm256_cmp_ps(distances, threshold_vec, _CMP_LT_OQ);
            int mask = _mm256_movemask_ps(cmp_result);
            if (SKM_UNLIKELY(mask)) {
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask >> i) & 1;
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
class SIMDUtilsComputer<skmeans::Quantization::sq4> {
  public:
    using data_t = skmeans_value_t<skmeans::Quantization::sq4>;
    using pdx_dist_t = pdx_distance_t<skmeans::Quantization::sq4>;

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
        constexpr size_t k_simd_width = 8;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        const __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
        __m256i threshold_vec =
            _mm256_sub_epi32(_mm256_set1_epi32(static_cast<int32_t>(pruning_threshold)), bias);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m256i distances =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pruning_distances + vector_idx)
                );
            __m256i biased = _mm256_sub_epi32(distances, bias);
            __m256i cmp_result = _mm256_cmpgt_epi32(threshold_vec, biased);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));
            if (SKM_UNLIKELY(mask)) {
                for (size_t i = 0; i < k_simd_width; ++i) {
                    pruning_positions[n_vectors_not_pruned] = vector_idx + i;
                    n_vectors_not_pruned += (mask >> i) & 1;
                }
            }
        }
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    /**
     * @brief Pack u8 values [0,15] into u4x2 format using AVX2.
     *
     * Uses _mm256_maddubs_epi16 with [1, 16] multiplier to compute
     * src[2k] + src[2k+1]*16 == src[2k] | (src[2k+1] << 4) when values in [0,15].
     * Processes 32 input bytes (16 output bytes) per iteration.
     */
    static void PackU8ToU4x2(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        size_t i = 0;
        const __m256i mul = _mm256_set1_epi16(0x1001);
        for (; i + 32 <= count; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i sum16 = _mm256_maddubs_epi16(v, mul);
            __m256i packed = _mm256_packus_epi16(sum16, _mm256_setzero_si256());
            packed = _mm256_permute4x64_epi64(packed, 0b00001000);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst + i / 2), _mm256_castsi256_si128(packed)
            );
        }
        for (; i + 2 <= count; i += 2) {
            dst[i / 2] = (src[i] & 0x0F) | ((src[i + 1] & 0x0F) << 4);
        }
    }

    /**
     * @brief Unpack u4x2 packed bytes to individual u8 values using AVX2.
     *
     * Uses vpmovzxbw (cvtepu8_epi16) to widen each packed byte to 16 bits,
     * splits nibbles, recombines as adjacent bytes in each 16-bit slot.
     * Processes 16 packed bytes (32 output) per iteration.
     */
    static void UnpackU4x2ToU8(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        size_t i = 0;
        for (; i + 16 <= n_packed; i += 16) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            __m256i v16 = _mm256_cvtepu8_epi16(v);
            __m256i lo = _mm256_and_si256(v16, _mm256_set1_epi16(0x000F));
            __m256i hi = _mm256_srli_epi16(v16, 4);
            __m256i result = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i * 2), result);
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
     * @brief AVX2-accelerated RaBitQ partial L2 for a 32-point block.
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     * Processes 8 floats per AVX2 iteration (4 iterations for kBlockSize=32).
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
        const __m256 v_c1j = _mm256_set1_ps(c1j);
        const __m256 v_c2j = _mm256_set1_ps(c2j);
        const __m256 v_c34j = _mm256_set1_ps(c34j);
        const __m256 v_qr_j = _mm256_set1_ps(qr_j);
        const __m256 v_neg2 = _mm256_set1_ps(-2.0f);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        size_t k = 0;
        for (; k + 8 <= blk_count; k += 8) {
            __m256 v_pd;
            if constexpr (U32Dot) {
                __m256i u32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u32 + k));
                v_pd = _mm256_cvtepi32_ps(u32);
            } else {
                __m128i u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pd_u16 + k));
                v_pd = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(u16));
            }

            __m256 v_sq = _mm256_loadu_ps(sum_q_f32 + k);

            __m256 fdt = _mm256_fmadd_ps(v_c2j, v_sq, _mm256_fmsub_ps(v_c1j, v_pd, v_c34j));

            __m256 v_or = _mm256_loadu_ps(or_c_l2sqr + k);
            __m256 v_dp = _mm256_loadu_ps(dp_mult + k);

            __m256 or_plus_qr = _mm256_add_ps(v_or, v_qr_j);
            __m256 result = _mm256_fmadd_ps(v_neg2, _mm256_mul_ps(v_dp, fdt), or_plus_qr);

            _mm256_storeu_ps(out_partial_l2 + k, result);
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
        const __m256 v_c1j = _mm256_set1_ps(c1j);
        const __m256 v_c34j = _mm256_set1_ps(c34j);
        const __m256 v_qr_j = _mm256_set1_ps(qr_j);
        const __m256 v_neg2c2j = _mm256_set1_ps(neg2_c2j);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        n_survivors = 0;
        size_t k = 0;
        for (; k + 8 <= blk_count; k += 8) {
            __m256 v_pd;
            if constexpr (U32Dot) {
                __m256i u32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u32 + k));
                v_pd = _mm256_cvtepi32_ps(u32);
            } else {
                __m128i u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pd_u16 + k));
                v_pd = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(u16));
            }

            // Chain A: base = or + qr + neg2_c2j * dp_sum_q
            __m256 base = _mm256_fmadd_ps(
                v_neg2c2j,
                _mm256_loadu_ps(dp_sum_q + k),
                _mm256_add_ps(_mm256_loadu_ps(or_c_l2sqr + k), v_qr_j)
            );

            // Chain B: shifted = c1j * dot - c34j
            __m256 shifted = _mm256_fmsub_ps(v_c1j, v_pd, v_c34j);

            // Merge: dist = neg2_dp * shifted + base
            __m256 result = _mm256_fmadd_ps(_mm256_loadu_ps(neg2_dp + k), shifted, base);

            __m256 thresh = _mm256_loadu_ps(threshold + k);
            __m256 cmp = _mm256_cmp_ps(result, thresh, _CMP_LE_OQ);
            int mask = _mm256_movemask_ps(cmp);
            if (SKM_UNLIKELY(mask)) {
                for (int i = 0; i < 8; ++i) {
                    survivor_positions[n_survivors] = static_cast<uint32_t>(k + i);
                    n_survivors += (mask >> i) & 1;
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
            ScanBlockAVX2(packed, lut, binary_bytes, out_dot);
            return;
        }
        ScalarFastScanComputer::ScanBlock<WideAdd>(packed, lut, binary_bytes, out_dot, blk_count);
    }

    /// Multi-block ScanBlock: delegates to scalar (AVX2 could be optimized later).
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
     * @brief AVX2 FastScan for nibble-split kPerm0-packed data.
     *
     * Single 256-bit LUT load per byte position (no broadcasts).
     * Interleaved u16 accumulation trick avoids per-iteration widening.
     * Output: 32 uint16 in natural vector order 0..31.
     */
    static void ScanBlockAVX2(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot
    ) {
        const __m256i lo_mask = _mm256_set1_epi8(0x0F);

        __m256i accu0 = _mm256_setzero_si256();
        __m256i accu1 = _mm256_setzero_si256();
        __m256i accu2 = _mm256_setzero_si256();
        __m256i accu3 = _mm256_setzero_si256();

        for (size_t b = 0; b < binary_bytes; ++b) {
            __m256i c =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + b * kBlockSize));
            __m256i tab = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + b * 32));

            __m256i lo = _mm256_and_si256(c, lo_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), lo_mask);

            __m256i res_lo = _mm256_shuffle_epi8(tab, lo);
            __m256i res_hi = _mm256_shuffle_epi8(tab, hi);

            accu0 = _mm256_add_epi16(accu0, res_lo);
            accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
            accu2 = _mm256_add_epi16(accu2, res_hi);
            accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
        }

        // Fix up: remove odd-byte contamination
        accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
        accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));

        // Cross-lane reduction: sum lo+hi nibble contributions
        // accu0 lane0=lo_vecs_0_7, lane1=hi_vecs_0_7 → sum = total_vecs_0_7
        // accu1 lane0=lo_vecs_8_15, lane1=hi_vecs_8_15 → sum = total_vecs_8_15
        __m256i dis0 = _mm256_add_epi16(
            _mm256_permute2f128_si256(accu0, accu1, 0x21), _mm256_blend_epi32(accu0, accu1, 0xF0)
        );
        __m256i dis1 = _mm256_add_epi16(
            _mm256_permute2f128_si256(accu2, accu3, 0x21), _mm256_blend_epi32(accu2, accu3, 0xF0)
        );

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_dot), dis0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_dot + 16), dis1);
    }
};

class SIMDRaBitQCodec {
  public:
    /**
     * @brief AVX2 encode: 8 floats per iteration.
     * _mm256_movemask_ps gives 8 sign bits as a byte — one store per iter.
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t binary_bytes,
        const float* SKM_RESTRICT centroid
    ) {
        std::memset(code, 0, binary_bytes);

        __m256 norm_acc = _mm256_setzero_ps();
        __m256 abs_acc = _mm256_setzero_ps();
        const __m256 zero = _mm256_setzero_ps();
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);

        size_t j = 0;
        size_t byte_off = 0;
        for (; j + 8 <= d; j += 8, byte_off += 1) {
            __m256 xv = _mm256_loadu_ps(x + j);
            __m256 cv = _mm256_loadu_ps(centroid + j);
            __m256 res = _mm256_sub_ps(xv, cv);

            norm_acc = _mm256_fmadd_ps(res, res, norm_acc);
            abs_acc = _mm256_add_ps(abs_acc, _mm256_andnot_ps(sign_mask, res));

            __m256 cmp = _mm256_cmp_ps(res, zero, _CMP_GT_OQ);
            code[byte_off] = static_cast<uint8_t>(_mm256_movemask_ps(cmp));
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

        // Horizontal sum: 8 floats → scalar
        __m128 hi128 = _mm256_extractf128_ps(norm_acc, 1);
        __m128 lo128 = _mm256_castps256_ps128(norm_acc);
        __m128 sum4 = _mm_add_ps(lo128, hi128);
        __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        __m128 sum1 = _mm_add_ss(sum2, _mm_movehdup_ps(sum2));
        const float norm_L2sqr = _mm_cvtss_f32(sum1) + norm_tail;

        hi128 = _mm256_extractf128_ps(abs_acc, 1);
        lo128 = _mm256_castps256_ps128(abs_acc);
        sum4 = _mm_add_ps(lo128, hi128);
        sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        sum1 = _mm_add_ss(sum2, _mm_movehdup_ps(sum2));
        const float dp_oO = _mm_cvtss_f32(sum1) + abs_tail;

        const float sqrt_d = std::sqrt(static_cast<float>(d));

        const float or_minus_c_l2sqr = norm_L2sqr;
        const float dp_multiplier = dp_oO > 0.0f ? norm_L2sqr * sqrt_d / dp_oO : 0.0f;
        std::memcpy(code + binary_bytes, &or_minus_c_l2sqr, sizeof(float));
        std::memcpy(code + binary_bytes + sizeof(float), &dp_multiplier, sizeof(float));
    }

    /**
     * @brief AVX2 decode: expand 8 sign bits to mask, blend ±0.5*scale, add centroid.
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

        const __m256 pos = _mm256_set1_ps(+0.5f * scale);
        const __m256 neg = _mm256_set1_ps(-0.5f * scale);
        const __m256i bit_masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

        size_t j = 0;
        size_t byte_off = 0;
        for (; j + 8 <= d; j += 8, byte_off += 1) {
            // Broadcast byte to all 8 lanes, AND with bit masks, compare != 0
            __m256i byte_bcast = _mm256_set1_epi32(code[byte_off]);
            __m256i masked = _mm256_and_si256(byte_bcast, bit_masks);
            __m256i cmp = _mm256_cmpeq_epi32(masked, bit_masks);
            __m256 mask = _mm256_castsi256_ps(cmp);

            __m256 val = _mm256_blendv_ps(neg, pos, mask);
            __m256 cv = _mm256_loadu_ps(centroid + j);
            _mm256_storeu_ps(x + j, _mm256_add_ps(val, cv));
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
     * @brief AVX2 LVQ4 encode: min/max reduction 8 floats/iter, quantize, pack nibbles.
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t nibble_bytes
    ) {
        // Min/max reduction: 8 floats per iteration
        __m256 v_min_vec = _mm256_set1_ps(std::numeric_limits<float>::max());
        __m256 v_max_vec = _mm256_set1_ps(std::numeric_limits<float>::lowest());
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 v = _mm256_loadu_ps(x + j);
            v_min_vec = _mm256_min_ps(v_min_vec, v);
            v_max_vec = _mm256_max_ps(v_max_vec, v);
        }
        // Horizontal reduce 8 → scalar
        __m128 min_lo = _mm256_castps256_ps128(v_min_vec);
        __m128 min_hi = _mm256_extractf128_ps(v_min_vec, 1);
        __m128 min4 = _mm_min_ps(min_lo, min_hi);
        min4 = _mm_min_ps(min4, _mm_movehl_ps(min4, min4));
        min4 = _mm_min_ss(min4, _mm_movehdup_ps(min4));
        float v_min = _mm_cvtss_f32(min4);

        __m128 max_lo = _mm256_castps256_ps128(v_max_vec);
        __m128 max_hi = _mm256_extractf128_ps(v_max_vec, 1);
        __m128 max4 = _mm_max_ps(max_lo, max_hi);
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float v_max = _mm_cvtss_f32(max4);

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

        const __m256 v_bias = _mm256_set1_ps(bias);
        const __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        const __m256i v_zero = _mm256_setzero_si256();
        const __m256i v_fifteen = _mm256_set1_epi32(15);
        const __m128i v_mul = _mm_set1_epi16(0x1001);

        j = 0;
        size_t out_off = 0;
        for (; j + 8 <= d; j += 8, out_off += 4) {
            __m256 v = _mm256_loadu_ps(x + j);
            __m256 q = _mm256_mul_ps(_mm256_sub_ps(v, v_bias), v_inv_scale);
            __m256i qi = _mm256_cvtps_epi32(q);
            qi = _mm256_max_epi32(qi, v_zero);
            qi = _mm256_min_epi32(qi, v_fifteen);

            // Narrow 8 × epi32 → 8 × epi8 (cross-lane safe)
            __m128i lo = _mm256_castsi256_si128(qi);
            __m128i hi = _mm256_extracti128_si256(qi, 1);
            __m128i packed16 = _mm_packs_epi32(lo, hi);                        // 8 × int16
            __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128()); // 8 × uint8

            // Pack nibble pairs: maddubs → 4 packed bytes
            __m128i nibbles16 = _mm_maddubs_epi16(packed8, v_mul);
            __m128i nibbles8 = _mm_packus_epi16(nibbles16, _mm_setzero_si128());
            *(uint32_t*) (code + out_off) = static_cast<uint32_t>(_mm_cvtsi128_si32(nibbles8));
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
     * @brief AVX2 LVQ4 decode: 4 packed bytes → 8 floats per iteration via unpack interleave.
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

        const __m256 v_scale = _mm256_set1_ps(scale);
        const __m256 v_bias = _mm256_set1_ps(bias);

        size_t b = 0;
        for (; b + 4 <= nibble_bytes; b += 4) {
            __m128i packed = _mm_cvtsi32_si128(*(const int32_t*) (code + b));
            __m128i wide = _mm_cvtepu8_epi32(packed); // 4 × epi32

            __m128i lo = _mm_and_si128(wide, _mm_set1_epi32(0x0F));
            __m128i hi = _mm_srli_epi32(wide, 4);

            // Interleave: [lo0,hi0,lo1,hi1] and [lo2,hi2,lo3,hi3]
            __m128i first = _mm_unpacklo_epi32(lo, hi);
            __m128i second = _mm_unpackhi_epi32(lo, hi);
            __m256i interleaved = _mm256_inserti128_si256(_mm256_castsi128_si256(first), second, 1);

            __m256 floats = _mm256_cvtepi32_ps(interleaved);
            __m256 result = _mm256_fmadd_ps(floats, v_scale, v_bias);
            _mm256_storeu_ps(x + b * 2, result);
        }
        // Scalar tail
        for (; b < nibble_bytes; ++b) {
            x[2 * b] = scale * static_cast<float>(code[b] & 0x0F) + bias;
            x[2 * b + 1] = scale * static_cast<float>(code[b] >> 4) + bias;
        }
    }
};

} // namespace skmeans
