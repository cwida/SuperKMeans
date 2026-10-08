#pragma once

#include "superkmeans/common.h"
#include "superkmeans/executor.h"
#include <cstddef>
#include <cstdint>

namespace skmeans {

// Forward declaration for pruning interface
template <Quantization q>
class PDXLayout;

/**
 * @brief Abstract quantizer interface.
 *
 * Each quantizer implements encoding/decoding AND its own distance
 * computation kernel. This lets different quantizers (e.g. product
 * quantization) ship a custom GEMM without touching BatchComputer.
 *
 * @tparam q Quantizer type that determines the quantized data type
 */
template <Quantization q>
class IQuantizer : public ExecutorHolder {
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
    virtual void ComputeNorms(const quantized_t* data, size_t n, size_t d, float* out_norms)
        const = 0;

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
     * @param tmp_buf Scratch space: X_BATCH_SIZE * Y_BATCH_SIZE floats, or at least
     * BatchComputer::ScratchSize of the bound executor when larger (f32: one tile per worker)
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
     * @brief Bytes per encoded vector. Default = d (one byte per dimension).
     * Quantizers with variable-length codes (e.g. RaBitQ) override this.
     */
    virtual size_t CodeSize(size_t d) const { return d; }

    /**
     * @brief Whether the quantizer has been fitted.
     */
    virtual bool IsFitted() const = 0;

    /**
     * @brief Whether this quantizer supports PDX pruning.
     */
    virtual bool SupportsPruning() const { return false; }

    /**
     * @brief Whether this quantizer needs a PDXLayout for pruning.
     *
     * SQ quantizers return true (use PDXearch). RaBitQ returns false
     * (uses a custom pruning loop with ADSampling). Default: same as
     * SupportsPruning().
     */
    virtual bool NeedsPDXLayout() const { return SupportsPruning(); }

    /**
     * @brief Compute the initial partial_d for a given vertical_d.
     * Default: max(MIN_PARTIAL_D, vertical_d / 2), clamped to vertical_d.
     */
    virtual uint32_t InitialPartialD(uint32_t vertical_d) const {
        return std::min(std::max<uint32_t>(MIN_PARTIAL_D, vertical_d / 2), vertical_d);
    }

    /**
     * @brief Align partial_d after tuning (enforce quantizer-specific constraints).
     * Default: clamp to vertical_d.
     * Quantizers with fixed partial_d (e.g. RaBitQ) override to always return
     * the fixed value, making tuning a no-op.
     */
    virtual uint32_t AlignPartialD(uint32_t partial_d, uint32_t vertical_d) const {
        return std::min(partial_d, vertical_d);
    }

    /**
     * @brief Invalidate any internal caches keyed on data pointers.
     *
     * Called when the underlying data buffer is reused with different content
     * (e.g. between mesocluster fine-clustering phases in HierarchicalSuperKMeans).
     * Quantizers with pointer-based caches (e.g. RaBitQ) override this.
     */
    virtual void InvalidateCaches() {}

    /**
     * @brief Reset quantizer-internal centroid accumulators.
     *
     * Called before each assignment pass. SQ quantizers override to zero
     * their internal uint32 accumulator buffer. Default: no-op.
     */
    virtual void ResetCentroidAccumulators(size_t n_clusters, size_t d) {
        (void) n_clusters;
        (void) d;
    }

    /**
     * @brief Accumulate encoded vectors into centroid state.
     *
     * Float-accumulating quantizers (f32, RaBitQ, LVQ4) write directly to
     * centroid_accumulators. SQ quantizers accumulate into internal uint32 buffers.
     * Threading partitions by centroid range.
     *
     * @param encoded_data Encoded vectors (n × code_size)
     * @param assignments Cluster assignment per vector (length n)
     * @param centroid_accumulators Float centroid sums [n_clusters × d], pre-zeroed
     * @param cluster_sizes Counts per cluster [n_clusters], pre-zeroed
     * @param n Number of data vectors
     * @param n_clusters Number of centroids
     * @param d Dimensionality
     */
    virtual void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
        uint32_t* cluster_sizes,
        size_t n,
        size_t n_clusters,
        size_t d
    ) const {
        (void) encoded_data;
        (void) assignments;
        (void) centroid_accumulators;
        (void) cluster_sizes;
        (void) n;
        (void) n_clusters;
        (void) d;
        assert(false && "UpdateCentroids not supported by this quantizer");
    }

    /**
     * @brief Produce final float centroids from accumulated state.
     *
     * Default: divide float centroid_accumulators by cluster_sizes in-place.
     * SQ overrides: integer-divide internal uint32 → quantized → decode → float.
     */
    virtual void FinalizeCentroids(
        float* centroids,
        const uint32_t* cluster_sizes,
        size_t n_clusters,
        size_t d
    ) const {
        for (size_t i = 0; i < n_clusters; ++i) {
            if (cluster_sizes[i] == 0)
                continue;
            float mult = 1.0f / static_cast<float>(cluster_sizes[i]);
            float* row = centroids + i * d;
            for (size_t j = 0; j < d; ++j) {
                row[j] *= mult;
            }
        }
    }

    /**
     * @brief Cache partial L2 squared norms for data vectors.
     *
     * Precomputes norms over the first partial_d dimensions of the data.
     * These are used internally by FindNearestNeighborWithPruning.
     * Must be called before pruning, and again whenever partial_d changes.
     *
     * @param data Encoded data vectors (n × code_size, row-major)
     * @param n Number of vectors
     * @param d Full dimensionality
     * @param partial_d Number of leading dimensions to compute norms over
     */
    virtual void CacheDataPartialNorms(
        const quantized_t* data,
        size_t n,
        size_t d,
        uint32_t partial_d
    ) {
        (void) data;
        (void) n;
        (void) d;
        (void) partial_d;
        assert(false && "CacheDataPartialNorms not supported by this quantizer");
    }

    /**
     * @brief Cache partial L2 squared norms for centroid vectors.
     *
     * Same as CacheDataPartialNorms but for centroids.
     * Must be called before each pruning iteration (centroids change every iteration).
     */
    virtual void CacheCentroidPartialNorms(
        const quantized_t* centroids,
        size_t n,
        size_t d,
        uint32_t partial_d
    ) {
        (void) centroids;
        (void) n;
        (void) d;
        (void) partial_d;
        assert(false && "CacheCentroidPartialNorms not supported by this quantizer");
    }

    /**
     * @brief Find top-1 nearest neighbor with PDX pruning.
     *
     * Combines partial GEMM (on first partial_d dimensions) with PDX pruned search
     * on the remaining dimensions. Final distances are float.
     * Partial norms must be cached via CacheDataPartialNorms / CacheCentroidPartialNorms
     * before calling this method.
     *
     * @param x Quantized query vectors (n_x × code_size, row-major)
     * @param y Quantized reference vectors (n_y × code_size, row-major)
     * @param x_float Original float query vectors (n_x × d, row-major)
     * @param y_float Original float reference vectors (n_y × d, row-major)
     * @param n_x Number of queries
     * @param n_y Number of references
     * @param d Dimensionality
     * @param out_knn Output: nearest reference index per query (length n_x)
     * @param out_distances Output: L2 squared distance to nearest (length n_x, float)
     * @param pdx_centroids PDXLayout holding the PDXified centroid data
     * @param partial_d Number of dimensions covered by partial GEMM
     * @param out_not_pruned_counts Output: count of non-pruned vectors per query (length n_x)
     * @param tmp_buf Scratch space: X_BATCH_SIZE * Y_BATCH_SIZE floats, or at least
     * BatchComputer::ScratchSize of the bound executor when larger (f32: one tile per worker)
     */
    virtual void FindNearestNeighborWithPruning(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<q>& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts,
        float* tmp_buf
    ) const {
        (void) x;
        (void) y;
        (void) x_float;
        (void) y_float;
        (void) n_x;
        (void) n_y;
        (void) d;
        (void) out_knn;
        (void) out_distances;
        (void) pdx_centroids;
        (void) partial_d;
        (void) out_not_pruned_counts;
        (void) tmp_buf;
        assert(false && "FindNearestNeighborWithPruning not supported by this quantizer");
    }
};

} // namespace skmeans
