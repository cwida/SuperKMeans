#pragma once

#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/executor.h"
#include "superkmeans/pdx/utils.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>
#ifdef HAS_FFTW
#include <fftw3.h>
#endif

namespace skmeans {

/**
 * @brief Compute the ADSampling ratio on the fly.
 *
 * @param visited_dimensions Number of dimensions already visited
 * @param num_dimensions Total number of dimensions
 * @param epsilon0 Pruning threshold parameter
 * @return The computed sampling ratio
 */
inline float ComputeADSamplingRatio(
    size_t visited_dimensions,
    size_t num_dimensions,
    float epsilon0
) {
    if (visited_dimensions == 0 || visited_dimensions >= num_dimensions)
        return 1.0f;
    const double eps0 = static_cast<double>(epsilon0);
    const double ratio = static_cast<double>(visited_dimensions) /
                         static_cast<double>(num_dimensions) *
                         (1.0 + eps0 / std::sqrt(static_cast<double>(visited_dimensions))) *
                         (1.0 + eps0 / std::sqrt(static_cast<double>(visited_dimensions)));
    return static_cast<float>(ratio);
}

/**
 * @brief ADSampling pruner
 *
 * Implements Adaptive Dimension Sampling (ADSampling) which enables early termination
 * during distance computations by predicting whether a candidate can be pruned based
 * on partial distance calculations. Uses a random rotation matrix to ensure dimensions
 * contribute equally to the distance.
 * Reference: https://dl.acm.org/doi/abs/10.1145/3589282
 *
 * For high-dimensional data (>= D_THRESHOLD_FOR_DCT_ROTATION), uses DCT-based rotation
 * which is more efficient than full matrix multiplication.
 *
 * Rotations run on the bound executor (SetExecutor), serially when unbound. Rotate and Unrotate
 * release its threads when they return.
 */
class ADSamplingPruner : public ExecutorHolder {
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  public:
    uint32_t num_dimensions;
    std::vector<float> ratios; // Precomputed pruning threshold ratios

    /**
     * @brief Constructor
     *
     * @param num_dimensions_
     * @param epsilon0 Pruning threshold parameter (higher = more aggressive pruning, less accuracy)
     * @param seed Random seed for reproducible rotation matrix generation
     * @param data_already_rotated The caller rotates the data itself: no rotation is built, and
     * Rotate/Unrotate warn and copy the vectors unchanged
     */
    ADSamplingPruner(
        uint32_t num_dimensions_,
        float epsilon0,
        uint32_t seed = 42,
        bool data_already_rotated = false
    )
        : num_dimensions(num_dimensions_), epsilon0(epsilon0) {
        InitializeRatios();
        if (data_already_rotated) {
            return;
        }
        std::mt19937 gen(seed); // NOLINT(bugprone-narrowing-conversions)
        bool matrix_created = false;
#ifdef HAS_FFTW
#ifdef __AVX2__
        // x86 machines don't behave well with FFTW with non-power-of-2 dimensions
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION && IsPowerOf2(num_dimensions)) {
#else
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
#endif
            matrix.resize(1, num_dimensions);
            std::uniform_int_distribution<int> dist(0, 1);
            for (size_t i = 0; i < num_dimensions; ++i) {
                matrix(i) = dist(gen) ? 1.0f : -1.0f;
            }
            flip_masks.resize(num_dimensions);
            for (size_t i = 0; i < num_dimensions; ++i) {
                // Use matrix(i) which has the random +1/-1 values, not flip_masks[i] which is
                // uninitialized
                flip_masks[i] = (matrix(i) < 0.0f ? 0x80000000u : 0u);
            }
            matrix_created = true;
        }
#endif
        if (!matrix_created) {
            matrix.resize(num_dimensions, num_dimensions);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (uint32_t i = 0; i < num_dimensions; ++i) {
                for (uint32_t j = 0; j < num_dimensions; ++j) {
                    matrix(i, j) = dist(gen);
                }
            }
            const Eigen::HouseholderQR<MatrixR> qr(matrix);
            matrix = qr.householderQ() * MatrixR::Identity(num_dimensions, num_dimensions);
        }
    }

    /**
     * @brief Pre-computes pruning threshold ratios for all dimension indices.
     *
     * Called during construction and when epsilon0 changes.
     */
    void InitializeRatios() {
        // + 1 to be able to map n_dims to 1.0f, avoiding a branch in GetPruningThreshold
        ratios.resize(num_dimensions + 1);
        for (size_t i = 0; i < num_dimensions + 1; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    /**
     * @brief Updates the pruning threshold parameter and recalculates ratios.
     * @param eps0 New epsilon0 value
     */
    void SetEpsilon0(float eps0) {
        epsilon0 = eps0;
        InitializeRatios();
    }

    /**
     * @brief Computes the pruning threshold for a given number of visited dimensions.
     *
     * @param best_candidate Current best candidate (threshold)
     * @param current_dimension_idx Number of dimensions visited so far
     * @return Pruning threshold - vectors with partial distance above this can be pruned
     */
    skmeans_distance_t GetPruningThreshold(
        const KNNCandidate& best_candidate,
        const uint32_t current_dimension_idx
    ) const {
        return best_candidate.distance * ratios[current_dimension_idx];
    }

    /**
     * @brief Applies sign flipping for DCT-based rotation (FFTW path).
     *
     * @param data Input vectors (row-major, n × num_dimensions)
     * @param out Output vectors (row-major, n × num_dimensions)
     * @param n Number of vectors
     */
    void FlipSign(const float* data, float* out, const size_t n) const {
        GetExecutor().ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const size_t offset = i * num_dimensions;
                UtilsComputer<Quantization::f32>::FlipSign(
                    data + offset, out + offset, flip_masks.data(), num_dimensions
                );
            }
        });
    }

    /**
     * @brief Rotates vectors
     *
     * Transforms vectors to a rotated space where dimensions contribute more equally
     * to the total distance, enabling effective early termination.
     *
     * For DCT path: applies sign flipping followed by DCT-II transform.
     * For matrix path: computes out = vectors * matrix^T.
     *
     * With IN_PLACE, callers pass the same pointer for vectors and out_buffer. The DCT path
     * already transforms out_buffer in place. The matrix path cannot alias its GEMM operands,
     * so it walks blocks of INPLACE_ROTATION_BLOCK_ROWS through a scratch buffer.
     *
     * @tparam IN_PLACE Whether vectors and out_buffer are the same buffer
     * @param vectors Input vectors (row-major, n × num_dimensions)
     * @param out_buffer Output buffer for rotated vectors (n × num_dimensions)
     * @param n Number of vectors to rotate
     */
    template <bool IN_PLACE = false>
    void Rotate(const float* vectors, float* out_buffer, const uint32_t n) const {
        ParallelSection parallel_section(GetExecutor());
        if (WarnIfNoRotation(vectors, out_buffer, n)) {
            return;
        }
        Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
#ifdef HAS_FFTW
#ifdef __AVX2__
        // x86 machines don't behave well with FFTW with non-power-of-2 dimensions
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION && IsPowerOf2(num_dimensions)) {
#else
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
#endif
            FlipSign(vectors, out_buffer, n);
            ParallelDCT(FFTW_REDFT10, out_buffer, n);
            const float s0 = std::sqrt(1.0f / (4.0f * num_dimensions));
            const float s = std::sqrt(1.0f / (2.0f * num_dimensions));
            out.col(0) *= s0;
            out.rightCols(num_dimensions - 1) *= s;
            return;
        }
#endif
        if constexpr (IN_PLACE) {
            const size_t n_block_rows = std::min<size_t>(INPLACE_ROTATION_BLOCK_ROWS, n);
            std::unique_ptr<float[]> tmp_block(new float[n_block_rows * num_dimensions]);
            for (size_t i = 0; i < n; i += n_block_rows) {
                const size_t n_rows = std::min(n_block_rows, static_cast<size_t>(n) - i);
                float* dst = out_buffer + i * num_dimensions;
                std::memcpy(tmp_block.get(), dst, n_rows * num_dimensions * sizeof(float));
                RotateImpl(tmp_block.get(), dst, static_cast<uint32_t>(n_rows));
            }
        } else {
            RotateImpl(vectors, out_buffer, n);
        }
    }

    /**
     * @brief Computes out = vectors * matrix^T. Operands must not alias.
     *
     * Single-threaded GEMMs over blocks of MINI_BATCH_SIZE rows, run in parallel.
     *
     * @param vectors Input vectors (row-major, n × num_dimensions)
     * @param out_buffer Output buffer for rotated vectors (n × num_dimensions)
     * @param n Number of vectors to rotate
     */
    void RotateImpl(
        const float* SKM_RESTRICT vectors,
        float* SKM_RESTRICT out_buffer,
        const uint32_t n
    ) const {
        const char trans_a = 'T';
        const char trans_b = 'N';
        int m = static_cast<int>(num_dimensions);
        int k = static_cast<int>(num_dimensions);
        float alpha = 1.0f;
        float beta = 0.0f;
        int lda = static_cast<int>(num_dimensions);
        int ldb = static_cast<int>(num_dimensions);
        int ldc = static_cast<int>(num_dimensions);
        const size_t n_blocks = (static_cast<size_t>(n) + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
        GetExecutor().ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t) {
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t row = block * MINI_BATCH_SIZE;
                int n_rows =
                    static_cast<int>(std::min(MINI_BATCH_SIZE, static_cast<size_t>(n) - row));
                Sgemm(
                    trans_a,
                    trans_b,
                    m,
                    n_rows,
                    k,
                    alpha,
                    matrix.data(),
                    lda,
                    vectors + row * num_dimensions,
                    ldb,
                    beta,
                    out_buffer + row * num_dimensions,
                    ldc
                );
            }
        });
    }

    /**
     * @brief Unrotate vectors (inverse of Rotate).
     * For orthonormal matrix Q: Rotate does out = vectors * Q^T, so Unrotate does out = vectors * Q
     * For DCT: applies inverse scaling, then inverse DCT (DCT-III), then FlipSign
     * Like Rotate, it runs in parallel over blocks of rows.
     */
    void Unrotate(
        const float* SKM_RESTRICT rotated_vectors,
        float* SKM_RESTRICT out_buffer,
        const uint32_t n
    ) const {
        ParallelSection parallel_section(GetExecutor());
        if (WarnIfNoRotation(rotated_vectors, out_buffer, n)) {
            return;
        }
        Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
#ifdef HAS_FFTW
#ifdef __AVX2__
        // x86 machines don't behave well with FFTW with non-power-of-2 dimensions
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION && IsPowerOf2(num_dimensions)) {
#else
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
#endif
            // Copy input to output buffer for in-place transform
            std::memcpy(
                out_buffer, rotated_vectors, static_cast<size_t>(n) * num_dimensions * sizeof(float)
            );

            // Undo scaling (inverse of forward scaling)
            const float inv_s0 = std::sqrt(4.0f * num_dimensions);
            const float inv_s = std::sqrt(2.0f * num_dimensions);
            out.col(0) *= inv_s0;
            out.rightCols(num_dimensions - 1) *= inv_s;

            // Apply inverse DCT (DCT-III = FFTW_REDFT01)
            ParallelDCT(FFTW_REDFT01, out_buffer, n);

            // FFTW's DCT-III needs normalization by 1/(2*n)
            out *= (1.0f / (2.0f * num_dimensions));

            // Undo FlipSign (FlipSign is its own inverse)
            FlipSign(out_buffer, out_buffer, n);
            return;
        }
#endif
        // For orthonormal matrix: Q^{-1} = Q^T, and Rotate does v * Q^T, so Unrotate does v * Q
        const char trans_a = 'N';
        const char trans_b = 'N';
        int m = static_cast<int>(num_dimensions);
        int k = static_cast<int>(num_dimensions);
        float alpha = 1.0f;
        float beta = 0.0f;
        int lda = static_cast<int>(num_dimensions);
        int ldb = static_cast<int>(num_dimensions);
        int ldc = static_cast<int>(num_dimensions);
        const size_t n_blocks = (static_cast<size_t>(n) + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;
        GetExecutor().ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t) {
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t row = block * MINI_BATCH_SIZE;
                int n_rows =
                    static_cast<int>(std::min(MINI_BATCH_SIZE, static_cast<size_t>(n) - row));
                Sgemm(
                    trans_a,
                    trans_b,
                    m,
                    n_rows,
                    k,
                    alpha,
                    matrix.data(),
                    lda,
                    rotated_vectors + row * num_dimensions,
                    ldb,
                    beta,
                    out_buffer + row * num_dimensions,
                    ldc
                );
            }
        });
    }

  private:
    float epsilon0 = 1.5f;            // Pruning aggressiveness parameter
    MatrixR matrix;                   // Rotation matrix (or sign vector for DCT)
    std::vector<uint32_t> flip_masks; // Sign flip masks for DCT-based rotation

#ifdef HAS_FFTW
    /**
     * @brief Runs a DCT of the given kind in place on every row of out (n × num_dimensions).
     *
     * One single-threaded plan per block size, made on scratch (FFTW_MEASURE overwrites the
     * arrays it plans on) by the calling thread (the planner is not thread-safe), executed on
     * blocks of MINI_BATCH_SIZE rows in parallel (fftwf_execute_r2r is thread-safe).
     */
    void ParallelDCT(fftw_r2r_kind kind, float* out, size_t n) const {
        if (n == 0) {
            return;
        }
        const int n0 = static_cast<int>(num_dimensions);
        const unsigned flag =
            (IsPowerOf2(num_dimensions) ? FFTW_ESTIMATE : FFTW_MEASURE) | FFTW_UNALIGNED;
        const size_t block_rows = std::min(MINI_BATCH_SIZE, n);
        const size_t tail_rows = n % block_rows;
        std::unique_ptr<float[]> scratch(new float[block_rows * num_dimensions]);
        auto make_plan = [&](size_t rows) {
            const int howmany = static_cast<int>(rows);
            return fftwf_plan_many_r2r(
                1, &n0, howmany, scratch.get(), NULL, 1, n0, scratch.get(), NULL, 1, n0, &kind, flag
            );
        };
        fftwf_plan block_plan = make_plan(block_rows);
        fftwf_plan tail_plan = tail_rows > 0 ? make_plan(tail_rows) : nullptr;
        const size_t n_blocks = (n + block_rows - 1) / block_rows;
        GetExecutor().ParallelFor(n_blocks, [&](size_t block_begin, size_t block_end, size_t) {
            for (size_t block = block_begin; block < block_end; ++block) {
                const size_t row = block * block_rows;
                float* rows_p = out + row * num_dimensions;
                fftwf_execute_r2r(row + block_rows <= n ? block_plan : tail_plan, rows_p, rows_p);
            }
        });
        fftwf_destroy_plan(block_plan);
        if (tail_plan != nullptr) {
            fftwf_destroy_plan(tail_plan);
        }
    }
#endif

    /**
     * @brief Computes the pruning ratio for a given number of visited dimensions.
     *
     * Based on the ADSampling paper, the ratio accounts for the expected contribution
     * of remaining dimensions to the total distance.
     *
     * @param visited_dimensions Number of dimensions visited so far
     * @return Ratio to multiply with best distance to get pruning threshold
     */
    float GetRatio(const size_t visited_dimensions) {
        return ComputeADSamplingRatio(visited_dimensions, num_dimensions, epsilon0);
    }

    /**
     * @brief Warns when the pruner was built without a rotation (data_already_rotated), and
     * copies the vectors unchanged.
     *
     * @return Whether there is no rotation (the Rotate/Unrotate call is then done)
     */
    bool WarnIfNoRotation(const float* vectors, float* out_buffer, const uint32_t n) const {
        if (matrix.size() != 0) {
            return false;
        }
        std::cout << "WARNING: the ADSampling pruner was built for already rotated data and has no "
                     "rotation, the vectors are copied unchanged"
                  << std::endl;
        if (vectors != out_buffer) {
            std::memcpy(
                out_buffer, vectors, static_cast<size_t>(n) * num_dimensions * sizeof(float)
            );
        }
        return true;
    }
};

} // namespace skmeans
