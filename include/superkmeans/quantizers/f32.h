#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/quantizers/quantizer.h"

#include <Eigen/Dense>
#include <cassert>
#include <cstring>
#include <vector>

namespace skmeans {

/**
 * @brief Identity quantizer for float32 data.
 *
 * Wraps BatchComputer<l2, f32> static methods in the IQuantizer interface,
 * enabling a unified code path for f32 and quantized types in SuperKMeans.
 * Encode/Decode are identity operations (memcpy when src != dst).
 */
template <Quantization q>
class F32Quantizer : public IQuantizer<q> {
    static_assert(q == Quantization::f32, "F32Quantizer only supports f32");

  public:
    using quantized_t = typename IQuantizer<q>::quantized_t; // float

    using batch_computer = BatchComputer<DistanceFunction::l2, Quantization::f32>;
    using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

    void Fit(const float* /*data*/, size_t /*n*/, size_t d) override {
        dim = d;
        pruning_tmp_distances.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        fitted = true;
    }

    void Encode(const float* in, float* out, size_t n, size_t d) const override {
        assert(fitted);
        if (in != out) {
            memcpy(out, in, n * d * sizeof(float));
        }
    }

    void Decode(const float* in, float* out, size_t n, size_t d) const override {
        assert(fitted);
        if (in != out) {
            memcpy(out, in, n * d * sizeof(float));
        }
    }

    void ComputeNorms(
        const float* data, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted);
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(out_norms, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
    }

    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        assert(fitted);
        batch_computer::FindNearestNeighbor(
            x, y, n_x, n_y, d, norms_x, norms_y, out_knn, out_distances, tmp_buf
        );
    }

    size_t DefaultRerankK() const override { return 0; }

    void FindNearestNeighborWithReranking(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        size_t /*rerank_k*/,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        // f32 doesn't need reranking — delegate to exact search
        FindNearestNeighbor(
            x, y, nullptr, nullptr, n_x, n_y, d, norms_x, norms_y, out_knn, out_distances, tmp_buf
        );
    }

    bool IsFitted() const override { return fitted; }

    bool SupportsPruning() const override { return true; }

    size_t CodeSize(size_t d) const override { return d; }

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t d, uint32_t partial_d
    ) override {
        cached_data_partial_norms.resize(n);
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(cached_data_partial_norms.data(), n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    void CacheCentroidPartialNorms(
        const quantized_t* centroids, size_t n, size_t d, uint32_t partial_d
    ) override {
        cached_centroid_partial_norms.resize(n);
        Eigen::Map<const MatrixR> e_data(centroids, n, d);
        Eigen::Map<VectorR> e_norms(cached_centroid_partial_norms.data(), n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    void FindNearestNeighborWithPruning(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        layout_t& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        assert(fitted);
        assert(!cached_data_partial_norms.empty() && "CacheDataPartialNorms must be called first");
        assert(
            !cached_centroid_partial_norms.empty() && "CacheCentroidPartialNorms must be called first"
        );

        batch_computer::FindNearestNeighborWithPruning(
            x,
            y,
            n_x,
            n_y,
            d,
            cached_data_partial_norms.data(),
            cached_centroid_partial_norms.data(),
            out_knn,
            out_distances,
            pruning_tmp_distances.data(),
            pdx_centroids,
            partial_d,
            out_not_pruned_counts
        );
    }

  private:
    bool fitted = false;
    size_t dim = 0;
    std::vector<float> cached_data_partial_norms;
    std::vector<float> cached_centroid_partial_norms;
    mutable std::vector<float> pruning_tmp_distances;
};

} // namespace skmeans
