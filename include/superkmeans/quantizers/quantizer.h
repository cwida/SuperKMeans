#pragma once

#include "superkmeans/common.h"
#include <cstddef>
#include <cstdint>

namespace skmeans {

/**
 * @brief Abstract quantizer interface.
 *
 * Each quantizer implements encoding/decoding AND its own distance
 * computation kernel. This lets different quantizers (e.g. product
 * quantization) ship a custom GEMM without touching BatchComputer.
 *
 * @tparam q Quantization enum that determines the quantized data type
 */
template <Quantization q>
class IQuantizer {
  public:
    using quantized_t = skmeans_value_t<q>;

    virtual ~IQuantizer() = default;

    /**
     * @brief Compute quantization parameters from float data.
     *
     * @param data Row-major float matrix (n × d)
     * @param n Number of vectors
     * @param d Dimensionality
     */
    virtual void Fit(const float* data, size_t n, size_t d) = 0;

    /**
     * @brief Quantize a batch of float vectors.
     */
    virtual void Encode(const float* in, quantized_t* out, size_t n, size_t d) const = 0;

    /**
     * @brief Dequantize back to float.
     */
    virtual void Decode(const quantized_t* in, float* out, size_t n, size_t d) const = 0;

    /**
     * @brief Compute float L2 squared norms of quantized vectors.
     *
     * The returned norms must be in the original float distance space
     * so that L2(x,y) = norm(x) + norm(y) - 2 * dot_scaled(x,y) holds.
     */
    virtual void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const = 0;

    /**
     * @brief Find top-1 nearest neighbor for each query vector.
     *
     * @param x Quantized query vectors (n_x × code_size, row-major)
     * @param y Quantized reference vectors (n_y × code_size, row-major)
     * @param x_float Original float query vectors (n_x × d, row-major)
     * @param y_float Original float reference vectors (n_y × d, row-major)
     * @param n_x Number of queries
     * @param n_y Number of references
     * @param d Dimensionality
     * @param norms_x Pre-computed float norms for x (length n_x)
     * @param norms_y Pre-computed float norms for y (length n_y)
     * @param out_knn Output: nearest reference index per query (length n_x)
     * @param out_distances Output: L2 squared distance to nearest (length n_x)
     * @param tmp_buf Scratch space (at least X_BATCH_SIZE * Y_BATCH_SIZE floats)
     */
    virtual void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const = 0;

    /**
     * @brief Default number of reranking candidates for this quantizer.
     *
     * Return 0 if no reranking is needed (quantized top-1 is sufficient).
     * Subclasses with lower precision (e.g. sq4) may return a higher value.
     */
    virtual size_t DefaultRerankK() const = 0;

    /**
     * @brief Quantized coarse top-k search followed by exact f32 reranking.
     *
     * 1. Find top rerank_k candidates per query in quantized space
     * 2. Compute exact f32 L2² to those candidates using original float data
     * 3. Write the best to out_knn / out_distances
     *
     * @param x_quantized Quantized query vectors (n_x × d, row-major)
     * @param y_quantized Quantized reference vectors (n_y × d, row-major)
     * @param x_float     Original float query vectors for reranking (n_x × d)
     * @param y_float     Original float reference vectors for reranking (n_y × d)
     * @param n_x Number of queries
     * @param n_y Number of references
     * @param d Dimensionality
     * @param norms_x Pre-computed float norms for x (length n_x)
     * @param norms_y Pre-computed float norms for y (length n_y)
     * @param rerank_k Number of coarse candidates to rerank per query
     * @param out_knn Output: top-1 index per query (length n_x)
     * @param out_distances Output: top-1 L2² distance (length n_x)
     * @param tmp_buf Scratch space (at least X_BATCH_SIZE * Y_BATCH_SIZE floats)
     */
    virtual void FindNearestNeighborWithReranking(
        const quantized_t* x_quantized,
        const quantized_t* y_quantized,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        size_t rerank_k,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const = 0;

    /**
     * @brief Bytes per encoded vector. Default = d (one byte per dimension).
     * Quantizers with variable-length codes (e.g. RaBitQ) override this.
     */
    virtual size_t CodeSize(size_t d) const { return d; }

    /**
     * @brief Whether the quantizer has been fitted.
     */
    virtual bool IsFitted() const = 0;
};

} // namespace skmeans
