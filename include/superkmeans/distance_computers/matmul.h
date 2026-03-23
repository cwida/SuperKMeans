#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"

#include <Eigen/Dense>
#include <pthreadpool.h>
#include <xnnpack.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// Eigen already declares sgemm_, so we don't need to redeclare it
extern "C" {}

namespace skmeans {
namespace matmul {

inline bool UseXnnpack() {
    static const bool enabled = (std::getenv("USE_XNNPACK") != nullptr);
    return enabled;
}

inline bool InitXnnpack() {
    static const bool ok = (xnn_initialize(nullptr) == xnn_status_success);
    return ok;
}

/**
 * @brief Packs kernel rows into a contiguous partial-dimension buffer for XNNPACK.
 *
 * When partial_d < d, XNNPACK requires the kernel to be contiguous N × partial_d.
 * This copies the first partial_d elements from each of the n_rows rows (stride d)
 * into a thread-local contiguous buffer and returns a pointer to it.
 * When partial_d == d, returns the original pointer with no copy.
 *
 * @return Pointer to contiguous kernel data and the effective dimension count.
 */
template <typename T>
inline std::pair<const T*, size_t> PackKernelForPartialD(
    const T* kernel, size_t n_rows, size_t d, size_t partial_d
) {
    const size_t partial_d_ = (partial_d > 0 && partial_d < d) ? partial_d : d;
    if (partial_d_ == d) {
        return {kernel, d};
    }
    thread_local std::vector<char> buf;
    buf.resize(n_rows * partial_d_ * sizeof(T));
    auto* dst = reinterpret_cast<T*>(buf.data());
    for (size_t i = 0; i < n_rows; ++i) {
        std::memcpy(dst + i * partial_d_, kernel + i * d, partial_d_ * sizeof(T));
    }
    return {dst, partial_d_};
}

/**
 * @brief Performs matrix multiplication using XNNPACK's fully-connected operator: C = X * Y^T
 *
 * Uses XNNPACK's NC F32 fully-connected operator where X is the "input" (batch_n_x × d)
 * and Y is the "kernel" (batch_n_y × d). Supports partial-dimension computation via partial_d:
 * the input (X) uses XNNPACK's input_stride to skip trailing dimensions without copying,
 * while the kernel (Y) rows are packed into a contiguous buffer when partial_d < d.
 *
 * Thread-local pthreadpool with g_n_threads threads is used. Not safe to call from OMP parallel regions.
 */
inline void XnnpackMatmulF32(
    const float* SKM_RESTRICT batch_x_p,
    const float* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    float* SKM_RESTRICT tmp_distances_buf
) {
    SKM_PROFILE_SCOPE("xnnpack_f32");
    const float* kernel_p;
    size_t partial_d_;
    {
        SKM_PROFILE_SCOPE("xnnpack_f32/pack");
        std::tie(kernel_p, partial_d_) = PackKernelForPartialD(batch_y_p, batch_n_y, d, partial_d);
    }

    {
        SKM_PROFILE_SCOPE("xnnpack_f32/matmul");
        thread_local pthreadpool_t tp = pthreadpool_create(omp_get_max_threads());
        xnn_operator_t op = nullptr;
        xnn_create_fully_connected_nc_f32(
            partial_d_, batch_n_y, d, batch_n_y,
            kernel_p, nullptr,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            0, nullptr, &op
        );
        xnn_reshape_fully_connected_nc_f32(op, batch_n_x, tp);
        xnn_setup_fully_connected_nc_f32(op, batch_x_p, tmp_distances_buf);
        xnn_run_operator(op, tp);
        xnn_delete_operator(op);
    }
}

/**
 * @brief Performs int8→float32 matrix multiplication using XNNPACK: C = X * Y^T
 *
 * Uses XNNPACK's dynamic-quantized int8 fully-connected operator (qd8_f32_qc8w).
 * Identity quantization (scale=1.0, zero_point=0) so the result is the integer dot product
 * cast to float. Supports partial-dimension computation via partial_d.
 *
 * Thread-local pthreadpool with g_n_threads threads is used. Not safe to call from OMP parallel regions.
 */
inline void XnnpackMatmulI8F32(
    const int8_t* SKM_RESTRICT batch_x_p,
    const int8_t* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    float* SKM_RESTRICT tmp_distances_buf
) {
    SKM_PROFILE_SCOPE("xnnpack_i8f32");
    const int8_t* kernel_p;
    size_t partial_d_;
    {
        SKM_PROFILE_SCOPE("xnnpack_i8f32/pack");
        std::tie(kernel_p, partial_d_) = PackKernelForPartialD(batch_y_p, batch_n_y, d, partial_d);
    }

    {
        SKM_PROFILE_SCOPE("xnnpack_i8f32/matmul");
        thread_local std::vector<float> kernel_scale;
        kernel_scale.assign(batch_n_y, 1.0f);

        thread_local std::vector<xnn_quantization_params> qparams;
        qparams.assign(batch_n_x, {0, 1.0f});

        thread_local pthreadpool_t tp_i8 = pthreadpool_create(omp_get_max_threads());
        xnn_operator_t op = nullptr;
        xnn_create_fully_connected_nc_qd8_f32_qc8w(
            partial_d_, batch_n_y, d, batch_n_y,
            kernel_scale.data(), kernel_p, nullptr,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            0, nullptr, &op
        );
        size_t ws_size = 0;
        xnn_reshape_fully_connected_nc_qd8_f32_qc8w(op, batch_n_x, &ws_size, tp_i8);
        thread_local std::vector<char> ws;
        ws.resize(ws_size);
        xnn_setup_fully_connected_nc_qd8_f32_qc8w(
            op, batch_x_p, tmp_distances_buf, ws.data(), qparams.data()
        );
        xnn_run_operator(op, tp_i8);
        xnn_delete_operator(op);
    }
}

#if defined(__aarch64__)
/**
 * @brief Performs int8 matrix multiplication using NEON SDOT: C = X * Y^T
 *
 * Hand-written kernel using vdotq_s32 (SDOT) instructions for int8 dot products.
 * Both X and Y are row-major with stride d; only the first partial_d dimensions are used.
 * Output is int32 accumulation of the dot products.
 */
inline void NeonSdotMatmulI8(
    const int8_t* SKM_RESTRICT batch_x_p,
    const int8_t* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    int32_t* SKM_RESTRICT tmp_distances_buf
) {
    const size_t partial_d_ = (partial_d > 0 && partial_d < d) ? partial_d : d;
    const size_t K16 = partial_d_ & ~(size_t)15;

#pragma omp parallel for schedule(static) num_threads(g_n_threads)
    for (size_t m = 0; m < batch_n_x; m++) {
        const int8_t* a_row = batch_x_p + m * d;
        int32_t* c_row = tmp_distances_buf + m * batch_n_y;

        size_t n = 0;
        for (; n + 4 <= batch_n_y; n += 4) {
            const int8_t* b0 = batch_y_p + (n + 0) * d;
            const int8_t* b1 = batch_y_p + (n + 1) * d;
            const int8_t* b2 = batch_y_p + (n + 2) * d;
            const int8_t* b3 = batch_y_p + (n + 3) * d;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            for (size_t k = 0; k < K16; k += 16) {
                int8x16_t va = vld1q_s8(a_row + k);
                acc0 = vdotq_s32(acc0, va, vld1q_s8(b0 + k));
                acc1 = vdotq_s32(acc1, va, vld1q_s8(b1 + k));
                acc2 = vdotq_s32(acc2, va, vld1q_s8(b2 + k));
                acc3 = vdotq_s32(acc3, va, vld1q_s8(b3 + k));
            }

            int32_t s0 = vaddvq_s32(acc0), s1 = vaddvq_s32(acc1);
            int32_t s2 = vaddvq_s32(acc2), s3 = vaddvq_s32(acc3);

            for (size_t k = K16; k < partial_d_; k++) {
                int32_t a_val = a_row[k];
                s0 += a_val * static_cast<int32_t>(b0[k]);
                s1 += a_val * static_cast<int32_t>(b1[k]);
                s2 += a_val * static_cast<int32_t>(b2[k]);
                s3 += a_val * static_cast<int32_t>(b3[k]);
            }

            c_row[n + 0] = s0;
            c_row[n + 1] = s1;
            c_row[n + 2] = s2;
            c_row[n + 3] = s3;
        }

        for (; n < batch_n_y; n++) {
            const int8_t* b_row = batch_y_p + n * d;
            int32x4_t acc = vdupq_n_s32(0);
            size_t k = 0;
            for (; k < K16; k += 16)
                acc = vdotq_s32(acc, vld1q_s8(a_row + k), vld1q_s8(b_row + k));
            int32_t sum = vaddvq_s32(acc);
            for (; k < partial_d_; k++)
                sum += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_row[k]);
            c_row[n] = sum;
        }
    }
}
#endif

/**
 * @brief Performs matrix multiplication via BLAS sgemm: C = X * Y^T
 *
 * Pure sgemm call with partial-dimension support via the k parameter.
 * Both X (batch_n_x × d) and Y (batch_n_y × d) are row-major; sgemm handles
 * the transpose by using the leading dimension as stride.
 */
inline void BlasMatrixMultiplication(
    const float* SKM_RESTRICT batch_x_p,
    const float* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    float* SKM_RESTRICT tmp_distances_buf
) {
    const char trans_a = 'T';
    const char trans_b = 'N';

    int m = static_cast<int>(batch_n_y);
    int n = static_cast<int>(batch_n_x);
    int k = static_cast<int>(partial_d > 0 && partial_d < d ? partial_d : d);
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = static_cast<int>(d);
    int ldb = static_cast<int>(d);
    int ldc = static_cast<int>(batch_n_y);

    sgemm_(
        &trans_a,
        &trans_b,
        &m,
        &n,
        &k,
        &alpha,
        batch_y_p,
        &lda,
        batch_x_p,
        &ldb,
        &beta,
        tmp_distances_buf,
        &ldc
    );
}

/**
 * @brief Generic matrix multiplication dispatcher: C = X * Y^T
 *
 * Routes to XNNPACK (if USE_XNNPACK env var is set) or falls back to BLAS sgemm.
 * Supports partial-dimension computation via partial_d.
 */
inline void MatrixMultiplication(
    const float* SKM_RESTRICT batch_x_p,
    const float* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    float* SKM_RESTRICT tmp_distances_buf
) {
    if (UseXnnpack() && InitXnnpack()) {
        XnnpackMatmulF32(
            batch_x_p, batch_y_p, batch_n_x, batch_n_y, d, partial_d, tmp_distances_buf
        );
        return;
    }
    BlasMatrixMultiplication(
        batch_x_p, batch_y_p, batch_n_x, batch_n_y, d, partial_d, tmp_distances_buf
    );
}

} // namespace matmul
} // namespace skmeans
