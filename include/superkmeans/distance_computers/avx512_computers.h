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
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::sq8> {
  public:
    using distance_t = pdx_distance_t<Quantization::sq8>;
    using data_t = skmeans_value_t<Quantization::sq8>;

    /**
     * @brief Computes the squared L2 distance between two uint8 vectors using AVX-512.
     *
     * Adapted from NumKong nk_sqeuclidean_u8_icelake. Widens u8 absolute differences
     * to i16 before squaring via _mm512_dpwssd_epi32 (VNNI i16×i16→i32).
     * This avoids the dpbusds signed-interpretation bug where abs differences > 127
     * would be misinterpreted as negative in the second operand.
     *
     * Processes 64 bytes per iteration using two i16 accumulators (low/high halves).
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m512i d2_low_i32 = _mm512_setzero_si512();
        __m512i d2_high_i32 = _mm512_setzero_si512();
        __m512i const zeros = _mm512_setzero_si512();
        __m512i a_u8, b_u8, diff_u8, diff_low_i16, diff_high_i16;

    nk_sqeuclidean_u8_ice_cycle:
        if (num_dimensions < 64) {
            __mmask64 mask = static_cast<__mmask64>(_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions));
            a_u8 = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8 = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        } else {
            a_u8 = _mm512_loadu_si512(vector1);
            b_u8 = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Absolute difference via saturating subtraction
        diff_u8 = _mm512_or_si512(_mm512_subs_epu8(a_u8, b_u8), _mm512_subs_epu8(b_u8, a_u8));
        // Widen u8 -> i16 (zero-extend) to avoid signed misinterpretation
        diff_low_i16 = _mm512_unpacklo_epi8(diff_u8, zeros);
        diff_high_i16 = _mm512_unpackhi_epi8(diff_u8, zeros);
        // Square and accumulate at i16 level into i32
        d2_low_i32 = _mm512_dpwssd_epi32(d2_low_i32, diff_low_i16, diff_low_i16);
        d2_high_i32 = _mm512_dpwssd_epi32(d2_high_i32, diff_high_i16, diff_high_i16);
        if (num_dimensions)
            goto nk_sqeuclidean_u8_ice_cycle;

        return _mm512_reduce_add_epi32(_mm512_add_epi32(d2_low_i32, d2_high_i32));
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t;
    using data_t = skmeans_value_t<Quantization::f32>;
    using scalar_computer = ScalarComputer<skmeans::DistanceFunction::l2, Quantization::f32>;

    /**
     * @brief Computes the L2 distance between two float vectors using AVX-512.
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
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;
    simsimd_l2sq_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = (__mmask16) _bzhi_u32(0xFFFFFFFF, num_dimensions);
            a_vec = _mm512_maskz_loadu_ps(mask, vector1);
            b_vec = _mm512_maskz_loadu_ps(mask, vector2);
            num_dimensions = 0;
        } else {
            a_vec = _mm512_loadu_ps(vector1);
            b_vec = _mm512_loadu_ps(vector2);
            vector1 += 16, vector2 += 16, num_dimensions -= 16;
        }
        __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
        d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
        if (num_dimensions)
            goto simsimd_l2sq_f32_skylake_cycle;

        // _simsimd_reduce_f32x16_skylake
        __m512 x =
            _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(
            _mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1)))
        );
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::sq4> {
  public:
    using distance_t = pdx_distance_t<Quantization::sq4>;
    using data_t = skmeans_value_t<Quantization::sq4>;

    /**
     * @brief Computes L2² distance between two packed u4x2 vectors using AVX-512.
     * Adapted from nk_sqeuclidean_u4_icelake in NumKong.
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
        const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
        __m512i d2_i32x16 = _mm512_setzero_si512();
        __m512i a_vec, b_vec;
        __m512i a_lo, a_hi, b_lo, b_hi, diff_lo, diff_hi;

    simsimd_l2sq_u4_ice_cycle:
        if (num_packed_bytes < 64) {
            const __mmask64 mask =
                static_cast<__mmask64>(_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_packed_bytes));
            a_vec = _mm512_maskz_loadu_epi8(mask, vector1);
            b_vec = _mm512_maskz_loadu_epi8(mask, vector2);
            num_packed_bytes = 0;
        } else {
            a_vec = _mm512_loadu_si512(vector1);
            b_vec = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_packed_bytes -= 64;
        }
        // Extract nibbles
        a_lo = _mm512_and_si512(a_vec, nibble_mask);
        a_hi = _mm512_and_si512(_mm512_srli_epi16(a_vec, 4), nibble_mask);
        b_lo = _mm512_and_si512(b_vec, nibble_mask);
        b_hi = _mm512_and_si512(_mm512_srli_epi16(b_vec, 4), nibble_mask);
        // Absolute diff via saturating sub: |a-b| = (a⊖b) | (b⊖a)
        diff_lo = _mm512_or_si512(_mm512_subs_epu8(a_lo, b_lo), _mm512_subs_epu8(b_lo, a_lo));
        diff_hi = _mm512_or_si512(_mm512_subs_epu8(a_hi, b_hi), _mm512_subs_epu8(b_hi, a_hi));
        // Square and accumulate using DPBUSD (VNNI)
        d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_lo, diff_lo);
        d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_hi, diff_hi);
        if (num_packed_bytes)
            goto simsimd_l2sq_u4_ice_cycle;

        return static_cast<distance_t>(_mm512_reduce_add_epi32(d2_i32x16));
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::rabitq> {
  public:
    using distance_t = pdx_distance_t<Quantization::rabitq>;
    using data_t = skmeans_value_t<Quantization::rabitq>;

    /**
     * @brief Computes popcount(a AND b) — binary inner product using AVX-512.
     * Uses VPOPCNTQ when available, otherwise falls back to scalar.
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_bytes
    ) {
#ifdef __AVX512VPOPCNTDQ__
        __m512i acc = _mm512_setzero_si512();
        size_t i = 0;
        for (; i + 64 <= num_bytes; i += 64) {
            __m512i va = _mm512_loadu_si512(vector1 + i);
            __m512i vb = _mm512_loadu_si512(vector2 + i);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        }
        uint32_t count = static_cast<uint32_t>(_mm512_reduce_add_epi64(acc));
        for (; i < num_bytes; ++i) {
            count += static_cast<uint32_t>(SKM_POPCOUNT(vector1[i] & vector2[i]));
        }
        return count;
#else
        return ScalarComputer<DistanceFunction::l2, Quantization::rabitq>::Horizontal(
            vector1, vector2, num_bytes
        );
#endif
    };

    /**
     * @brief Fused multi-bitplane popcount on chunk-interleaved layout.
     * Layout: for each 16B data chunk, qb×16B of bitplane data contiguous.
     * AVX-512: broadcast 16B of data, single 512-bit load for all 4 bitplanes,
     * AND, VPOPCNTQ, VPSLLVQ by [0,0,1,1,2,2,3,3], reduce.
     */
    static uint32_t HorizontalMultiPlane(
        const data_t* SKM_RESTRICT data,
        const data_t* planes_interleaved,
        size_t num_bytes,
        int qb
    ) {
#ifdef __AVX512VPOPCNTDQ__
        const __m512i shift_vec = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
        __m512i acc = _mm512_setzero_si512();
        size_t i = 0;
        for (; i + 16 <= num_bytes; i += 16) {
            __m512i x =
                _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i)));
            __m512i bp = _mm512_loadu_si512(planes_interleaved + i * qb);
            __m512i popcnt = _mm512_popcnt_epi64(_mm512_and_si512(x, bp));
            acc = _mm512_add_epi64(acc, _mm512_sllv_epi64(popcnt, shift_vec));
        }
        // 256-bit tail: process 8 bytes with _mm256_popcnt_epi64
        // After the 512-bit loop, i is a multiple of 16 so byte_in_chunk=0.
        // Gather 8B from each bitplane at stride-16 (all within one cache line).
        if (i + 8 <= num_bytes) {
            const data_t* chunk_ptr = planes_interleaved + (i / 16) * qb * 16;
            const __m256i shift_vec_256 = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i x = _mm256_set1_epi64x(*reinterpret_cast<const int64_t*>(data + i));
            __m256i bp = _mm256_set_epi64x(
                *reinterpret_cast<const int64_t*>(chunk_ptr + static_cast<size_t>(3) * 16),
                *reinterpret_cast<const int64_t*>(chunk_ptr + static_cast<size_t>(2) * 16),
                *reinterpret_cast<const int64_t*>(chunk_ptr + static_cast<size_t>(1) * 16),
                *reinterpret_cast<const int64_t*>(chunk_ptr)
            );
            __m256i popcnt = _mm256_popcnt_epi64(_mm256_and_si256(x, bp));
            __m256i shifted = _mm256_sllv_epi64(popcnt, shift_vec_256);
            // Zero-extend to 512-bit and accumulate
            acc = _mm512_add_epi64(acc, _mm512_inserti64x4(_mm512_setzero_si512(), shifted, 0));
            i += 8;
        }
        // Single reduce after both SIMD paths
        uint32_t result = static_cast<uint32_t>(_mm512_reduce_add_epi64(acc));
        // Scalar tail for remaining < 8 bytes
        for (; i < num_bytes; ++i) {
            size_t chunk = i / 16;
            size_t byte_in_chunk = i % 16;
            for (int bp = 0; bp < qb; ++bp) {
                result += static_cast<uint32_t>(SKM_POPCOUNT(
                              data[i] &
                              planes_interleaved
                                  [chunk * qb * 16 + static_cast<size_t>(bp) * 16 + byte_in_chunk]
                          ))
                          << bp;
            }
        }
        return result;
#else
        return ScalarComputer<DistanceFunction::l2, Quantization::rabitq>::HorizontalMultiPlane(
            data, planes_interleaved, num_bytes, qb
        );
#endif
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
        constexpr size_t k_simd_width = 16;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        __m512i threshold_vec = _mm512_set1_epi32(static_cast<int32_t>(pruning_threshold));
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m512i distances = _mm512_loadu_si512(pruning_distances + vector_idx);
            __mmask16 cmp_mask = _mm512_cmplt_epu32_mask(distances, threshold_vec);
            if (SKM_UNLIKELY(cmp_mask)) {
                __m512i indices = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int>(vector_idx)),
                    _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
                );
                _mm512_mask_compressstoreu_epi32(
                    pruning_positions + n_vectors_not_pruned, cmp_mask, indices
                );
                n_vectors_not_pruned += _mm_popcnt_u32(cmp_mask);
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
     * @brief Flip sign of floats based on a mask using AVX-512.
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 vec = _mm512_loadu_ps(data + j);
            __m512i mask = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(masks + j));
            __m512i vec_i = _mm512_castps_si512(vec);
            vec_i = _mm512_xor_si512(vec_i, mask);
            _mm512_storeu_ps(out + j, _mm512_castsi512_ps(vec_i));
        }
        for (; j + 8 <= d; j += 8) {
            __m256 vec = _mm256_loadu_ps(data + j);
            __m256i mask_avx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(masks + j));
            __m256i vec_i = _mm256_castps_si256(vec);
            vec_i = _mm256_xor_si256(vec_i, mask_avx);
            _mm256_storeu_ps(out + j, _mm256_castsi256_ps(vec_i));
        }
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        for (; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors using AVX-512.
     *
     * Optimized for cases where only ~2% of vectors pass the threshold test.
     * Processes 16 floats at a time and uses vpcompressd for efficient scatter.
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
        constexpr size_t k_simd_width = 16;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        __m512 threshold_vec = _mm512_set1_ps(pruning_threshold);
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m512 distances = _mm512_loadu_ps(pruning_distances + vector_idx);
            __mmask16 cmp_mask = _mm512_cmp_ps_mask(distances, threshold_vec, _CMP_LT_OQ);
            if (SKM_UNLIKELY(cmp_mask)) {
                __m512i indices = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int>(vector_idx)),
                    _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
                );
                _mm512_mask_compressstoreu_epi32(
                    pruning_positions + n_vectors_not_pruned, cmp_mask, indices
                );
                n_vectors_not_pruned += _mm_popcnt_u32(cmp_mask);
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
        constexpr size_t k_simd_width = 16;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        __m512i threshold_vec = _mm512_set1_epi32(static_cast<int32_t>(pruning_threshold));
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m512i distances = _mm512_loadu_si512(pruning_distances + vector_idx);
            __mmask16 cmp_mask = _mm512_cmplt_epu32_mask(distances, threshold_vec);
            if (SKM_UNLIKELY(cmp_mask)) {
                __m512i indices = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int>(vector_idx)),
                    _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
                );
                _mm512_mask_compressstoreu_epi32(
                    pruning_positions + n_vectors_not_pruned, cmp_mask, indices
                );
                n_vectors_not_pruned += _mm_popcnt_u32(cmp_mask);
            }
        }
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    /**
     * @brief Pack u8 values [0,15] into u4x2 format using AVX-512.
     *
     * Same maddubs approach as AVX2 but 512-bit wide.
     * Processes 64 input bytes (32 output bytes) per iteration.
     */
    static void PackU8ToU4x2(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        size_t i = 0;
        const __m512i mul = _mm512_set1_epi16(0x1001);
        for (; i + 64 <= count; i += 64) {
            __m512i v = _mm512_loadu_si512(src + i);
            __m512i sum16 = _mm512_maddubs_epi16(v, mul);
            __m512i packed = _mm512_packus_epi16(sum16, _mm512_setzero_si512());
            packed = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst + i / 2), _mm512_castsi512_si256(packed)
            );
        }
        const __m256i mul256 = _mm256_set1_epi16(0x1001);
        for (; i + 32 <= count; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i sum16 = _mm256_maddubs_epi16(v, mul256);
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
     * @brief Unpack u4x2 packed bytes to individual u8 values using AVX-512.
     *
     * Reverse of PackU8ToU4x2. Uses vpmovzxbw (cvtepu8_epi16) to widen each
     * packed byte to 16 bits, splits nibbles, recombines as adjacent bytes,
     * and stores. No lane-crossing permutes needed.
     * Processes 32 packed bytes (64 output) per AVX-512 iteration.
     */
    static void UnpackU4x2ToU8(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        size_t i = 0;
        for (; i + 32 <= n_packed; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m512i v16 = _mm512_cvtepu8_epi16(v);
            __m512i lo = _mm512_and_si512(v16, _mm512_set1_epi16(0x000F));
            __m512i hi = _mm512_srli_epi16(v16, 4);
            __m512i result = _mm512_or_si512(lo, _mm512_slli_epi16(hi, 8));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i * 2), result);
        }
        const __m128i mask8 = _mm_set1_epi8(0x0F);
        for (; i + 16 <= n_packed; i += 16) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            __m128i lo = _mm_and_si128(v, mask8);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * 2), _mm_unpacklo_epi8(lo, hi));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst + i * 2 + 16), _mm_unpackhi_epi8(lo, hi)
            );
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
     * @brief AVX-512-accelerated RaBitQ partial L2 for a 32-point block.
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     * Processes 16 floats per AVX-512 iteration (2 iterations for kBlockSize=32).
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
        const __m512 v_c1j = _mm512_set1_ps(c1j);
        const __m512 v_c2j = _mm512_set1_ps(c2j);
        const __m512 v_c34j = _mm512_set1_ps(c34j);
        const __m512 v_qr_j = _mm512_set1_ps(qr_j);
        const __m512 v_neg2 = _mm512_set1_ps(-2.0f);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        size_t k = 0;
        for (; k + 16 <= blk_count; k += 16) {
            __m512 v_pd;
            if constexpr (U32Dot) {
                __m512i u32 = _mm512_loadu_si512(pd_u32 + k);
                v_pd = _mm512_cvtepi32_ps(u32);
            } else {
                __m256i u16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u16 + k));
                v_pd = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16));
            }

            __m512 v_sq = _mm512_loadu_ps(sum_q_f32 + k);

            __m512 fdt = _mm512_fmadd_ps(v_c2j, v_sq, _mm512_fmsub_ps(v_c1j, v_pd, v_c34j));

            __m512 v_or = _mm512_loadu_ps(or_c_l2sqr + k);
            __m512 v_dp = _mm512_loadu_ps(dp_mult + k);

            __m512 or_plus_qr = _mm512_add_ps(v_or, v_qr_j);
            __m512 result = _mm512_fmadd_ps(v_neg2, _mm512_mul_ps(v_dp, fdt), or_plus_qr);

            _mm512_storeu_ps(out_partial_l2 + k, result);
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

    /**
     * @brief Fused correction + compaction: computes partial L2 and compacts
     * survivors in a single pass, eliminating the intermediate store/load.
     *
     * The L2 result stays in registers and is compared directly against the
     * threshold, feeding into vpcompressd without a memory round-trip.
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
        const __m512 v_c1j = _mm512_set1_ps(c1j);
        const __m512 v_c34j = _mm512_set1_ps(c34j);
        const __m512 v_qr_j = _mm512_set1_ps(qr_j);
        const __m512 v_neg2c2j = _mm512_set1_ps(neg2_c2j);

        const auto* pd_u16 = static_cast<const uint16_t*>(partial_dot);
        const auto* pd_u32 = static_cast<const uint32_t*>(partial_dot);

        n_survivors = 0;

        // Fast path: blk_count == 32 → manually unrolled 2×16 with interleaved ILP
        if (blk_count == 32) {
            const __m512i offsets_lo =
                _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            const __m512i offsets_hi =
                _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);

            // Load dot products for both halves
            __m512 v_pd_lo, v_pd_hi;
            if constexpr (U32Dot) {
                v_pd_lo = _mm512_cvtepi32_ps(_mm512_loadu_si512(pd_u32));
                v_pd_hi = _mm512_cvtepi32_ps(_mm512_loadu_si512(pd_u32 + 16));
            } else {
                v_pd_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u16))
                ));
                v_pd_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u16 + 16))
                ));
            }

            // Chain A (dot-independent): base = or + qr + neg2_c2j * dp_sum_q
            __m512 base_lo = _mm512_fmadd_ps(
                v_neg2c2j,
                _mm512_loadu_ps(dp_sum_q),
                _mm512_add_ps(_mm512_loadu_ps(or_c_l2sqr), v_qr_j)
            );
            __m512 base_hi = _mm512_fmadd_ps(
                v_neg2c2j,
                _mm512_loadu_ps(dp_sum_q + 16),
                _mm512_add_ps(_mm512_loadu_ps(or_c_l2sqr + 16), v_qr_j)
            );

            // Chain B (dot-dependent): shifted = c1j * dot - c34j
            __m512 shifted_lo = _mm512_fmsub_ps(v_c1j, v_pd_lo, v_c34j);
            __m512 shifted_hi = _mm512_fmsub_ps(v_c1j, v_pd_hi, v_c34j);

            // Merge: dist = neg2_dp * shifted + base
            __m512 dist_lo = _mm512_fmadd_ps(_mm512_loadu_ps(neg2_dp), shifted_lo, base_lo);
            __m512 dist_hi = _mm512_fmadd_ps(_mm512_loadu_ps(neg2_dp + 16), shifted_hi, base_hi);

            // Compare + compact
            __mmask16 mask_lo = _mm512_cmp_ps_mask(dist_lo, _mm512_loadu_ps(threshold), _CMP_LE_OQ);
            __mmask16 mask_hi =
                _mm512_cmp_ps_mask(dist_hi, _mm512_loadu_ps(threshold + 16), _CMP_LE_OQ);

            if (SKM_UNLIKELY(mask_lo)) {
                _mm512_mask_compressstoreu_epi32(survivor_positions, mask_lo, offsets_lo);
                n_survivors = _mm_popcnt_u32(mask_lo);
            }
            if (SKM_UNLIKELY(mask_hi)) {
                _mm512_mask_compressstoreu_epi32(
                    survivor_positions + n_survivors, mask_hi, offsets_hi
                );
                n_survivors += _mm_popcnt_u32(mask_hi);
            }
            return;
        }

        //  General path: loop 16 at a time
        const __m512i offsets =
            _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        size_t k = 0;
        for (; k + 16 <= blk_count; k += 16) {
            __m512 v_pd;
            if constexpr (U32Dot) {
                v_pd = _mm512_cvtepi32_ps(_mm512_loadu_si512(pd_u32 + k));
            } else {
                v_pd = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pd_u16 + k))
                ));
            }

            // Chain A: base = or + qr + neg2_c2j * dp_sum_q
            __m512 base = _mm512_fmadd_ps(
                v_neg2c2j,
                _mm512_loadu_ps(dp_sum_q + k),
                _mm512_add_ps(_mm512_loadu_ps(or_c_l2sqr + k), v_qr_j)
            );

            // Chain B: shifted = c1j * dot - c34j
            __m512 shifted = _mm512_fmsub_ps(v_c1j, v_pd, v_c34j);

            // Merge: dist = neg2_dp * shifted + base
            __m512 result = _mm512_fmadd_ps(_mm512_loadu_ps(neg2_dp + k), shifted, base);

            __m512 thresh = _mm512_loadu_ps(threshold + k);
            __mmask16 cmp_mask = _mm512_cmp_ps_mask(result, thresh, _CMP_LE_OQ);
            if (SKM_UNLIKELY(cmp_mask)) {
                __m512i indices = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(k)), offsets);
                _mm512_mask_compressstoreu_epi32(
                    survivor_positions + n_survivors, cmp_mask, indices
                );
                n_survivors += _mm_popcnt_u32(cmp_mask);
            }
        }
        // Scalar tail
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
            ScanBlockAVX512(packed, lut, binary_bytes, out_dot);
            return;
        }
        ScalarFastScanComputer::ScanBlock<WideAdd>(packed, lut, binary_bytes, out_dot, blk_count);
    }

    /**
     * @brief Multi-block ScanBlock: process NBlocks data blocks against one shared LUT.
     * Amortizes LUT loads across multiple blocks of 32 X points.
     */
    template <int NBlocks>
    static void ScanBlockMulti(
        const uint8_t* const* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* const* out_dot
    ) {
        ScanBlockAVX512Multi<NBlocks>(packed, lut, binary_bytes, out_dot);
    }

  private:
    /**
     * @brief AVX-512 FastScan for nibble-split kPerm0-packed data.
     *
     * Processes 2 byte positions per iteration via 512-bit loads.
     * Uses interleaved u16 accumulation trick (no per-iteration widening).
     * Output: 32 uint16 in natural vector order 0..31.
     */
    static void ScanBlockAVX512(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot
    ) {
        const __m512i lo_mask = _mm512_set1_epi8(0x0F);
        __m512i accu0 = _mm512_setzero_si512();
        __m512i accu1 = _mm512_setzero_si512();
        __m512i accu2 = _mm512_setzero_si512();
        __m512i accu3 = _mm512_setzero_si512();

        // Process 2 byte positions per iteration (64B packed + 64B LUT)
        size_t b = 0;
        for (; b + 2 <= binary_bytes; b += 2) {
            __m512i c = _mm512_loadu_si512(packed + b * kBlockSize);
            __m512i tab = _mm512_loadu_si512(lut + b * 32);

            __m512i lo = _mm512_and_si512(c, lo_mask);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

            __m512i res_lo = _mm512_shuffle_epi8(tab, lo);
            __m512i res_hi = _mm512_shuffle_epi8(tab, hi);

            accu0 = _mm512_add_epi16(accu0, res_lo);
            accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
            accu2 = _mm512_add_epi16(accu2, res_hi);
            accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
        }

        // Handle odd trailing byte position with 256-bit
        if (b < binary_bytes) {
            __m256i c256 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + b * kBlockSize));
            __m256i tab256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + b * 32));
            __m256i lo_mask_256 = _mm256_set1_epi8(0x0F);

            __m256i lo = _mm256_and_si256(c256, lo_mask_256);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c256, 4), lo_mask_256);

            __m256i res_lo = _mm256_shuffle_epi8(tab256, lo);
            __m256i res_hi = _mm256_shuffle_epi8(tab256, hi);

            // Zero-extend to 512-bit before accumulating
            accu0 = _mm512_add_epi16(accu0, _mm512_inserti64x4(_mm512_setzero_si512(), res_lo, 0));
            accu1 = _mm512_add_epi16(
                accu1, _mm512_inserti64x4(_mm512_setzero_si512(), _mm256_srli_epi16(res_lo, 8), 0)
            );
            accu2 = _mm512_add_epi16(accu2, _mm512_inserti64x4(_mm512_setzero_si512(), res_hi, 0));
            accu3 = _mm512_add_epi16(
                accu3, _mm512_inserti64x4(_mm512_setzero_si512(), _mm256_srli_epi16(res_hi, 8), 0)
            );
        }

        // Fix up: remove odd-byte contamination
        accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
        accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

        // Reduce 4 lanes → 1 lane per accumulator, then combine
        __m512i ret1 = _mm512_add_epi16(
            _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
            _mm512_shuffle_i64x2(accu0, accu1, 0b01001110)
        );
        __m512i ret2 = _mm512_add_epi16(
            _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
            _mm512_shuffle_i64x2(accu2, accu3, 0b01001110)
        );

        __m512i ret = _mm512_add_epi16(
            _mm512_shuffle_i64x2(ret1, ret2, 0b10001000),
            _mm512_shuffle_i64x2(ret1, ret2, 0b11011101)
        );

        _mm512_storeu_si512(out_dot, ret);
    }

    /// Reduce 4 interleaved accumulators → 32 uint16 and store.
    SKM_ALWAYS_INLINE static void ReduceAndStore(
        __m512i a0,
        __m512i a1,
        __m512i a2,
        __m512i a3,
        uint16_t* out
    ) {
        a0 = _mm512_sub_epi16(a0, _mm512_slli_epi16(a1, 8));
        a2 = _mm512_sub_epi16(a2, _mm512_slli_epi16(a3, 8));
        __m512i r1 = _mm512_add_epi16(
            _mm512_mask_blend_epi64(0b11110000, a0, a1), _mm512_shuffle_i64x2(a0, a1, 0b01001110)
        );
        __m512i r2 = _mm512_add_epi16(
            _mm512_mask_blend_epi64(0b11110000, a2, a3), _mm512_shuffle_i64x2(a2, a3, 0b01001110)
        );
        _mm512_storeu_si512(
            out,
            _mm512_add_epi16(
                _mm512_shuffle_i64x2(r1, r2, 0b10001000), _mm512_shuffle_i64x2(r1, r2, 0b11011101)
            )
        );
    }

    /**
     * @brief Multi-block AVX-512 FastScan: load LUT once, apply to NBlocks data blocks.
     * NBlocks is a compile-time constant (1-4) so the compiler fully unrolls the block loop,
     * keeping all 4×NBlocks accumulators in registers.
     */
    template <int NBlocks>
    static void ScanBlockAVX512Multi(
        const uint8_t* const* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* const* out_dot
    ) {
        static_assert(NBlocks >= 1 && NBlocks <= 8);
        const __m512i lo_mask = _mm512_set1_epi8(0x0F);

        // 4 accumulators per block (interleaved trick)
        __m512i accu[NBlocks][4];
        for (int blk = 0; blk < NBlocks; ++blk)
            for (int a = 0; a < 4; ++a)
                accu[blk][a] = _mm512_setzero_si512();

        // Main loop: 2 byte positions per iteration, LUT loaded once
        size_t b = 0;
        for (; b + 2 <= binary_bytes; b += 2) {
            __m512i tab = _mm512_loadu_si512(lut + b * 32);

            for (int blk = 0; blk < NBlocks; ++blk) {
                __m512i c = _mm512_loadu_si512(packed[blk] + b * kBlockSize);
                __m512i lo = _mm512_and_si512(c, lo_mask);
                __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);
                __m512i res_lo = _mm512_shuffle_epi8(tab, lo);
                __m512i res_hi = _mm512_shuffle_epi8(tab, hi);
                accu[blk][0] = _mm512_add_epi16(accu[blk][0], res_lo);
                accu[blk][1] = _mm512_add_epi16(accu[blk][1], _mm512_srli_epi16(res_lo, 8));
                accu[blk][2] = _mm512_add_epi16(accu[blk][2], res_hi);
                accu[blk][3] = _mm512_add_epi16(accu[blk][3], _mm512_srli_epi16(res_hi, 8));
            }
        }

        // Odd trailing byte
        if (b < binary_bytes) {
            __m256i tab256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + b * 32));
            __m256i lo_mask_256 = _mm256_set1_epi8(0x0F);

            for (int blk = 0; blk < NBlocks; ++blk) {
                __m256i c256 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(packed[blk] + b * kBlockSize)
                );
                __m256i lo = _mm256_and_si256(c256, lo_mask_256);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c256, 4), lo_mask_256);
                __m256i res_lo = _mm256_shuffle_epi8(tab256, lo);
                __m256i res_hi = _mm256_shuffle_epi8(tab256, hi);
                __m512i zero = _mm512_setzero_si512();
                accu[blk][0] = _mm512_add_epi16(accu[blk][0], _mm512_inserti64x4(zero, res_lo, 0));
                accu[blk][1] = _mm512_add_epi16(
                    accu[blk][1], _mm512_inserti64x4(zero, _mm256_srli_epi16(res_lo, 8), 0)
                );
                accu[blk][2] = _mm512_add_epi16(accu[blk][2], _mm512_inserti64x4(zero, res_hi, 0));
                accu[blk][3] = _mm512_add_epi16(
                    accu[blk][3], _mm512_inserti64x4(zero, _mm256_srli_epi16(res_hi, 8), 0)
                );
            }
        }

        // Reduce and store each block
        for (int blk = 0; blk < NBlocks; ++blk) {
            ReduceAndStore(accu[blk][0], accu[blk][1], accu[blk][2], accu[blk][3], out_dot[blk]);
        }
    }
};

class SIMDRaBitQCodec {
  public:
    /**
     * @brief AVX-512 encode: compute residual, accumulate norm/abs_sum, pack sign bits.
     *
     * 16 floats per iteration. _mm512_cmpgt_ps_mask gives 16 sign bits directly
     * as __mmask16, written as uint16_t — zero bit shuffling.
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t binary_bytes,
        const float* SKM_RESTRICT centroid
    ) {
        std::memset(code, 0, binary_bytes);

        __m512 norm_acc = _mm512_setzero_ps();
        __m512 abs_acc = _mm512_setzero_ps();
        const __m512 zero = _mm512_setzero_ps();
        const __m512 sign_mask = _mm512_set1_ps(-0.0f);

        size_t j = 0;
        size_t byte_off = 0;
        for (; j + 16 <= d; j += 16, byte_off += 2) {
            __m512 xv = _mm512_loadu_ps(x + j);
            __m512 cv = _mm512_loadu_ps(centroid + j);
            __m512 res = _mm512_sub_ps(xv, cv);

            norm_acc = _mm512_fmadd_ps(res, res, norm_acc);
            abs_acc = _mm512_add_ps(abs_acc, _mm512_andnot_ps(sign_mask, res));

            __mmask16 signs = _mm512_cmp_ps_mask(res, zero, _CMP_GT_OQ);
            *(uint16_t*) (code + byte_off) = static_cast<uint16_t>(signs);
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

        const float norm_L2sqr = _mm512_reduce_add_ps(norm_acc) + norm_tail;
        const float dp_oO = _mm512_reduce_add_ps(abs_acc) + abs_tail;
        const float sqrt_d = std::sqrt(static_cast<float>(d));

        const float or_minus_c_l2sqr = norm_L2sqr;
        const float dp_multiplier = dp_oO > 0.0f ? norm_L2sqr * sqrt_d / dp_oO : 0.0f;
        std::memcpy(code + binary_bytes, &or_minus_c_l2sqr, sizeof(float));
        std::memcpy(code + binary_bytes + sizeof(float), &dp_multiplier, sizeof(float));
    }

    /**
     * @brief AVX-512 decode: expand sign bits via mask blend, add centroid.
     *
     * 16 dims per iteration. __mmask16 loaded directly from 2 code bytes,
     * used with _mm512_mask_blend_ps for branchless ±0.5 selection.
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

        const __m512 pos = _mm512_set1_ps(+0.5f * scale);
        const __m512 neg = _mm512_set1_ps(-0.5f * scale);

        size_t j = 0;
        size_t byte_off = 0;
        for (; j + 16 <= d; j += 16, byte_off += 2) {
            __mmask16 bits = static_cast<__mmask16>(*(const uint16_t*) (code + byte_off));
            __m512 val = _mm512_mask_blend_ps(bits, neg, pos);
            __m512 cv = _mm512_loadu_ps(centroid + j);
            _mm512_storeu_ps(x + j, _mm512_add_ps(val, cv));
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
     * @brief AVX-512 LVQ4 encode: min/max reduction, quantize 16 floats/iter, pack nibbles.
     * @TODO(lkuffo, low): This could be optimized further
     */
    static void EncodeOne(
        const float* SKM_RESTRICT x,
        uint8_t* SKM_RESTRICT code,
        size_t d,
        size_t nibble_bytes
    ) {
        // Min/max reduction: 16 floats per iteration
        __m512 v_min_vec = _mm512_set1_ps(std::numeric_limits<float>::max());
        __m512 v_max_vec = _mm512_set1_ps(std::numeric_limits<float>::lowest());
        size_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 v = _mm512_loadu_ps(x + j);
            v_min_vec = _mm512_min_ps(v_min_vec, v);
            v_max_vec = _mm512_max_ps(v_max_vec, v);
        }
        float v_min = _mm512_reduce_min_ps(v_min_vec);
        float v_max = _mm512_reduce_max_ps(v_max_vec);
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

        const __m512 v_bias = _mm512_set1_ps(bias);
        const __m512 v_inv_scale = _mm512_set1_ps(inv_scale);
        const __m512i v_zero = _mm512_setzero_si512();
        const __m512i v_fifteen = _mm512_set1_epi32(15);
        const __m128i v_mul = _mm_set1_epi16(0x1001);

        j = 0;
        size_t out_off = 0;
        for (; j + 16 <= d; j += 16, out_off += 8) {
            __m512 v = _mm512_loadu_ps(x + j);
            __m512 q = _mm512_mul_ps(_mm512_sub_ps(v, v_bias), v_inv_scale);
            __m512i qi = _mm512_cvtps_epi32(q);
            qi = _mm512_max_epi32(qi, v_zero);
            qi = _mm512_min_epi32(qi, v_fifteen);

            // Narrow 16 × epi32 → 16 × epi8
            __m128i narrow = _mm512_cvtepi32_epi8(qi);
            // Pack nibble pairs: maddubs with [1, 16] → val[2k] + val[2k+1]*16
            __m128i packed16 = _mm_maddubs_epi16(narrow, v_mul);
            __m128i packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
            _mm_storel_epi64((__m128i*) (code + out_off), packed8);
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
     * @brief AVX-512 LVQ4 decode: 8 packed bytes → 16 floats per iteration via permutex2var
     * interleave.
     * @TODO(lkuffo, low): This could be optimized further
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

        const __m512 v_scale = _mm512_set1_ps(scale);
        const __m512 v_bias = _mm512_set1_ps(bias);
        // Interleave index: lo[0], hi[0], lo[1], hi[1], ..., lo[7], hi[7]
        // lo indices 0-7 from src1, hi indices 16-23 from src2
        const __m512i interleave_idx =
            _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

        size_t b = 0;
        for (; b + 8 <= nibble_bytes; b += 8) {
            __m128i packed = _mm_loadl_epi64((const __m128i*) (code + b));
            __m256i wide = _mm256_cvtepu8_epi32(packed);

            __m256i lo = _mm256_and_si256(wide, _mm256_set1_epi32(0x0F));
            __m256i hi = _mm256_srli_epi32(wide, 4);

            __m512i lo_512 = _mm512_castsi256_si512(lo);
            __m512i hi_512 = _mm512_castsi256_si512(hi);
            __m512i interleaved = _mm512_permutex2var_epi32(lo_512, interleave_idx, hi_512);

            __m512 floats = _mm512_cvtepi32_ps(interleaved);
            __m512 result = _mm512_fmadd_ps(floats, v_scale, v_bias);
            _mm512_storeu_ps(x + b * 2, result);
        }
        // Scalar tail
        for (; b < nibble_bytes; ++b) {
            x[2 * b] = scale * static_cast<float>(code[b] & 0x0F) + bias;
            x[2 * b + 1] = scale * static_cast<float>(code[b] >> 4) + bias;
        }
    }
};

} // namespace skmeans
