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
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::u8> {
  public:
    using distance_t = pdx_distance_t<Quantization::u8>;
    using data_t = skmeans_value_t<Quantization::u8>;

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
            __mmask64 mask =
                static_cast<__mmask64>(_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions));
            a_u8 = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8 = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        } else {
            a_u8 = _mm512_loadu_si512(vector1);
            b_u8 = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Absolute difference via saturating subtraction
        diff_u8 = _mm512_or_si512(
            _mm512_subs_epu8(a_u8, b_u8), _mm512_subs_epu8(b_u8, a_u8)
        );
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

    /**
     * @brief Asymmetric squared L2 using VNNI dpbusds.
     *
     * Only correct when absolute differences fit in 7 bits (max 127), e.g. when
     * one operand's range is restricted. Uses saturating u8 subtraction for abs diff,
     * then VNNI dot-product to square and accumulate.
     * Adapted from SimSIMD: https://github.com/ashvardanian/SimSIMD
     */
    static distance_t HorizontalAsymmetric(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m512i d2_i32_vec = _mm512_setzero_si512();
        __m512i a_u8_vec, b_u8_vec;

    simsimd_l2sq_u8_ice_cycle:
        if (num_dimensions < 64) {
            const __mmask64 mask =
                static_cast<__mmask64>(_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions));
            a_u8_vec = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8_vec = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        } else {
            a_u8_vec = _mm512_loadu_si512(vector1);
            b_u8_vec = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Subtracting unsigned vectors via saturating subtraction:
        __m512i d_u8_vec = _mm512_or_si512(
            _mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec)
        );

        // Multiply and accumulate — second operand interpreted as signed int8,
        // so only correct when abs differences <= 127:
        d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, d_u8_vec, d_u8_vec);
        if (num_dimensions)
            goto simsimd_l2sq_u8_ice_cycle;
        return _mm512_reduce_add_epi32(d2_i32_vec);
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t<Quantization::f32>;
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
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::u4> {
  public:
    using distance_t = pdx_distance_t<Quantization::u4>;
    using data_t = skmeans_value_t<Quantization::u4>;

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
        diff_lo = _mm512_or_si512(
            _mm512_subs_epu8(a_lo, b_lo), _mm512_subs_epu8(b_lo, a_lo)
        );
        diff_hi = _mm512_or_si512(
            _mm512_subs_epu8(a_hi, b_hi), _mm512_subs_epu8(b_hi, a_hi)
        );
        // Square and accumulate using DPBUSD (VNNI)
        d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_lo, diff_lo);
        d2_i32x16 = _mm512_dpbusd_epi32(d2_i32x16, diff_hi, diff_hi);
        if (num_packed_bytes)
            goto simsimd_l2sq_u4_ice_cycle;

        return static_cast<distance_t>(_mm512_reduce_add_epi32(d2_i32x16));
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::b8> {
  public:
    using distance_t = pdx_distance_t<Quantization::b8>;
    using data_t = skmeans_value_t<Quantization::b8>;

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
            count += static_cast<uint32_t>(__builtin_popcount(vector1[i] & vector2[i]));
        }
        return count;
#else
        return ScalarComputer<DistanceFunction::l2, Quantization::b8>::Horizontal(
            vector1, vector2, num_bytes
        );
#endif
    };
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
                    _mm512_set1_epi32(vector_idx),
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
                    _mm512_set1_epi32(vector_idx),
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
        constexpr size_t k_simd_width = 16;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;
        __m512i threshold_vec = _mm512_set1_epi32(static_cast<int32_t>(pruning_threshold));
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m512i distances = _mm512_loadu_si512(pruning_distances + vector_idx);
            __mmask16 cmp_mask = _mm512_cmplt_epu32_mask(distances, threshold_vec);
            if (SKM_UNLIKELY(cmp_mask)) {
                __m512i indices = _mm512_add_epi32(
                    _mm512_set1_epi32(vector_idx),
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
            packed = _mm512_permutexvar_epi64(
                _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed
            );
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst + i / 2),
                _mm512_castsi512_si256(packed)
            );
        }
        const __m256i mul256 = _mm256_set1_epi16(0x1001);
        for (; i + 32 <= count; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i sum16 = _mm256_maddubs_epi16(v, mul256);
            __m256i packed = _mm256_packus_epi16(sum16, _mm256_setzero_si256());
            packed = _mm256_permute4x64_epi64(packed, 0b00001000);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(dst + i / 2),
                _mm256_castsi256_si128(packed)
            );
        }
        for (; i + 2 <= count; i += 2) {
            dst[i / 2] = (src[i] & 0x0F) | ((src[i + 1] & 0x0F) << 4);
        }
    }
};

class SIMDFastScanComputer {
  public:
    static constexpr size_t kBlockSize = 32;

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
        ScalarFastScanComputer::ScanBlock(packed, lut, binary_bytes, out_dot, blk_count);
    }

  private:
    // AVX-512 includes AVX2; use the 256-bit path for the 32-point block.
    static void ScanBlockAVX2(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot
    ) {
        const __m256i mask_0f = _mm256_set1_epi8(0x0F);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        for (size_t b = 0; b < binary_bytes; ++b) {
            __m128i lut_lo_128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(lut + (2 * b) * 16));
            __m128i lut_hi_128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(lut + (2 * b + 1) * 16));
            __m256i lut_lo_vec = _mm256_broadcastsi128_si256(lut_lo_128);
            __m256i lut_hi_vec = _mm256_broadcastsi128_si256(lut_hi_128);

            __m256i data = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(packed + b * kBlockSize));

            __m256i lo_idx = _mm256_and_si256(data, mask_0f);
            __m256i hi_idx = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask_0f);

            __m256i res_lo = _mm256_shuffle_epi8(lut_lo_vec, lo_idx);
            __m256i res_hi = _mm256_shuffle_epi8(lut_hi_vec, hi_idx);

            __m256i partial = _mm256_add_epi8(res_lo, res_hi);

            __m256i lo16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(partial));
            __m256i hi16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(partial, 1));

            acc0 = _mm256_add_epi16(acc0, lo16);
            acc1 = _mm256_add_epi16(acc1, hi16);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_dot), acc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_dot + 16), acc1);
    }
};

} // namespace skmeans
