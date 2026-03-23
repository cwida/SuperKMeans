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
 * XNNPack does not support performing matrix multiplication on an arbitrary subset of dimensions
 * Based on our benchmarks - we believe it is sufficiently performant to just re-copy the subset of dimensions that we care about
 *
 * When partial_d < d, XNNPACK requires the kernel to be contiguous N × partial_d.
 * This copies the first partial_d elements from each of the n_rows rows (stride d)
 * into a thread-local contiguous buffer and returns a pointer to it.
 * If partial_d == d, returns the original pointer with no copy.
 */
template <typename T>
inline std::pair<const T*, size_t> PackKernelForPartialD(
    const T* kernel, size_t n_rows, size_t d, size_t partial_d
) {
    const size_t partial_d_ = (partial_d > 0 && partial_d < d) ? partial_d : d;
    if (partial_d_ == d) {
        return {kernel, d};
    }
    // Note: this file uses threadlocals to avoid repeated allocations
    thread_local std::vector<char> buf;
    buf.resize(n_rows * partial_d_ * sizeof(T));
    auto* dst = reinterpret_cast<T*>(buf.data());
    for (size_t i = 0; i < n_rows; ++i) {
        std::memcpy(dst + i * partial_d_, kernel + i * d, partial_d_ * sizeof(T));
    }
    return {dst, partial_d_};
}

/**
 * Uses XNNPACK's NC F32 fully-connected operator where X is the "input" (batch_n_x × d)
 * and Y is the "kernel" (batch_n_y × d). Supports partial-dimension computation via partial_d:
 * the input (X) uses XNNPACK's input_stride to skip trailing dimensions without copying,
 * while the kernel (Y) rows are packed into a contiguous buffer when partial_d < d.
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
        // Only Y needs repacking: XNNPACK supports strided access for X (via input_stride),
        // but requires Y to be contiguous N × partial_d.
        std::tie(kernel_p, partial_d_) = PackKernelForPartialD(batch_y_p, batch_n_y, d, partial_d);
    }

    {
        SKM_PROFILE_SCOPE("xnnpack_f32/matmul");
        thread_local pthreadpool_t tp = pthreadpool_create(omp_get_max_threads());
        xnn_operator_t op = nullptr;
        // Create the fully-connected operator. XNNPACK's API is designed for neural network layers,
        // so we have to pass several parameters (bias, clamping, flags) that we don't use.
        xnn_create_fully_connected_nc_f32(
            // input_channels: number of dimensions to dot-product over
            partial_d_,
            // output_channels: number of rows in Y (columns in the output matrix)
            batch_n_y,
            // input_stride: full row stride in X, even if we only use partial_d_ dims
            d,
            // output_stride
            batch_n_y,
            // the Y matrix
            kernel_p,
            // Do not set a "bias" since we want raw dot products. The "bias" is a vector that is added to the output.
            nullptr,
            // no output clamping (we should be able to use the full range of f32 values)
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            // Don't set any special flags
            0,
            // Do not set a cache; we recreate the operator each call
            nullptr,
            &op
        );
        // Allocate internal buffers for batch_n_x input rows
        xnn_reshape_fully_connected_nc_f32(op, batch_n_x, tp);
        // Bind X as input and tmp_distances_buf as output
        xnn_setup_fully_connected_nc_f32(op, batch_x_p, tmp_distances_buf);
        // Execute the matmul using the thread pool
        xnn_run_operator(op, tp);
        // Free the operator (the thread pool is reused across calls)
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
    const int8_t* kernel_p;
    size_t partial_d_;
    {
        SKM_PROFILE_SCOPE("i8_knn/pack");
        // Only Y needs repacking: XNNPACK supports strided access for X (via input_stride),
        // but requires Y to be contiguous N × partial_d.
        std::tie(kernel_p, partial_d_) = PackKernelForPartialD(batch_y_p, batch_n_y, d, partial_d);
    }

    {
        SKM_PROFILE_SCOPE("i8_knn/matmul");
        thread_local std::vector<float> kernel_scale;
        kernel_scale.assign(batch_n_y, 1.0f);

        thread_local std::vector<xnn_quantization_params> qparams;
        qparams.assign(batch_n_x, {0, 1.0f});

        thread_local pthreadpool_t tp_i8 = pthreadpool_create(omp_get_max_threads());
        xnn_operator_t op = nullptr;
        xnn_create_fully_connected_nc_qd8_f32_qc8w(
            // input_channels: number of dimensions to dot-product over
            partial_d_,
            // output_channels: number of rows in Y (columns in the output matrix)
            batch_n_y,
            // input_stride: full row stride in X, even if we only use partial_d_ dims
            d,
            // output_stride
            batch_n_y,
            // per-row scale for Y; set to 1.0 for identity quantization
            kernel_scale.data(),
            // the Y matrix (XNNPACK calls this "kernel")
            kernel_p,
            // Do not set a "bias" since we want raw dot products. The "bias" is a vector that is added to the output.
            nullptr,
            // no output clamping (we should be able to use the full range of f32 values)
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            // Don't set any special flags
            0,
            // Do not set a cache; we recreate the operator each call
            nullptr,
            &op
        );
        // Allocate internal buffers for batch_n_x input rows; also reports workspace size needed
        size_t ws_size = 0;
        xnn_reshape_fully_connected_nc_qd8_f32_qc8w(op, batch_n_x, &ws_size, tp_i8);
        thread_local std::vector<char> ws;
        ws.resize(ws_size);
        // Bind X as input, tmp_distances_buf as output, plus workspace and quantization params
        xnn_setup_fully_connected_nc_qd8_f32_qc8w(
            op, batch_x_p, tmp_distances_buf, ws.data(), qparams.data()
        );
        // Execute the matmul using the thread pool
        xnn_run_operator(op, tp_i8);
        // Free the operator (the thread pool is reused across calls)
        xnn_delete_operator(op);
    }
}

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
    // While we do support XnnpackMatmulF32 - we don't use it in any e2e benchmarks as of now. 
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
