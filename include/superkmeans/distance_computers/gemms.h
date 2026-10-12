#pragma once

#include "superkmeans/common.h"
#include "superkmeans/executor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#if !defined(__EMSCRIPTEN__)
#include "ruy/ruy.h"
#endif
#include <numkong/numkong.h>

namespace skmeans {

/**
 * @brief u8×u8→u32 dot-product GEMM leaf, dispatching between NumKong and ruy.
 *
 * Shared by SQ8 (native u8 codes) and LVQ4 (u4 codes decoded to u8). The caller
 * decides the backend via `use_numkong` (Wasm builds always use NumKong: they have no ruy);
 * the NumKong path packs `b` into `packed_buf` (skipped when `pack_b` is false, i.e. b is
 * unchanged across calls). Rows are split across the executor's workers.
 */
inline void U8Gemm(
    ParallelExecutor& executor,
    const uint8_t* a,
    const uint8_t* b,
    uint32_t* out,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride,
    size_t b_stride,
    bool use_numkong,
    std::vector<char>& packed_buf,
    bool pack_b
) {
    if (use_numkong || IS_WASM) {
        if (pack_b) {
            const size_t pack_size = nk_dots_packed_size_u8(n, k);
            if (pack_size > packed_buf.size())
                packed_buf.resize(pack_size);
            nk_dots_pack_u8(b, n, k, b_stride, packed_buf.data());
        }

        const size_t c_stride = n * sizeof(uint32_t);
        executor.ParallelFor(m, [&](size_t row_begin, size_t row_end, size_t) {
            nk_configure_thread(nk_capabilities());
            nk_dots_packed_u8(
                a + row_begin * a_stride,
                packed_buf.data(),
                out + row_begin * n,
                row_end - row_begin,
                n,
                k,
                a_stride,
                c_stride
            );
        });
        return;
    }

#if !defined(__EMSCRIPTEN__)
    executor.ParallelFor(m, [&](size_t row_start, size_t row_end, size_t) {
        const size_t local_rows = row_end - row_start;

        thread_local ruy::Context ctx;
        ctx.set_max_num_threads(1);

        ruy::Matrix<std::uint8_t> lhs;
        lhs.mutable_layout()->set_rows(static_cast<int>(local_rows));
        lhs.mutable_layout()->set_cols(static_cast<int>(k));
        lhs.mutable_layout()->set_order(ruy::Order::kRowMajor);
        lhs.mutable_layout()->set_stride(static_cast<int>(a_stride));
        lhs.set_data(a + row_start * a_stride);

        ruy::Matrix<std::uint8_t> rhs;
        rhs.mutable_layout()->set_rows(static_cast<int>(k));
        rhs.mutable_layout()->set_cols(static_cast<int>(n));
        rhs.mutable_layout()->set_order(ruy::Order::kColMajor);
        rhs.mutable_layout()->set_stride(static_cast<int>(b_stride));
        rhs.set_data(b);

        ruy::Matrix<std::int32_t> dst;
        dst.mutable_layout()->set_rows(static_cast<int>(local_rows));
        dst.mutable_layout()->set_cols(static_cast<int>(n));
        dst.mutable_layout()->set_order(ruy::Order::kRowMajor);
        dst.mutable_layout()->set_stride(static_cast<int>(n));
        dst.set_data(reinterpret_cast<std::int32_t*>(out + row_start * n));

        ruy::MulParams<std::int32_t, std::int32_t> mul_params;
        ruy::Mul(lhs, rhs, mul_params, &ctx, &dst);
    });
#endif
}

/**
 * @brief u4×u4→u32 dot-product GEMM leaf via NumKong (no ruy fallback).
 *
 * Used by LVQ4 on x86 without AMX for wide matrices, where the packed u4 codes
 * are fed directly to NumKong. `a`/`b` point to packed u4x2 bytes; the NumKong
 * path packs `b` into `packed_buf` (skipped when `pack_b` is false) and
 * parallelizes the row dots. Strides are in nk_u4x2_t units.
 */
inline void U4Gemm(
    ParallelExecutor& executor,
    const uint8_t* a,
    const uint8_t* b,
    uint32_t* out,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride,
    size_t b_stride,
    std::vector<char>& packed_buf,
    bool pack_b
) {
    const auto* a_u4 = reinterpret_cast<const nk_u4x2_t*>(a);
    const auto* b_u4 = reinterpret_cast<const nk_u4x2_t*>(b);

    if (pack_b) {
        const size_t pack_size = nk_dots_packed_size_u4(n, k);
        if (pack_size > packed_buf.size())
            packed_buf.resize(pack_size);
        nk_dots_pack_u4(b_u4, n, k, b_stride, packed_buf.data());
    }

    const size_t c_stride = n * sizeof(uint32_t);
    executor.ParallelFor(m, [&](size_t row_begin, size_t row_end, size_t) {
        nk_configure_thread(nk_capabilities());
        nk_dots_packed_u4(
            a_u4 + row_begin * a_stride,
            packed_buf.data(),
            out + row_begin * n,
            row_end - row_begin,
            n,
            k,
            a_stride,
            c_stride
        );
    });
}

} // namespace skmeans
