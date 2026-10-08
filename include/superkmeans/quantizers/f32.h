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
class F32Quantizer : public IQuantizer<Quantization::f32> {
  public:
    using quantized_t = IQuantizer::quantized_t; // float

    using batch_computer = BatchComputer<DistanceFunction::l2, Quantization::f32>;
    using layout_t = PDXLayout<Quantization::f32>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

    void Fit(ParallelExecutor& /*executor*/, const float* /*data*/, size_t /*n*/, size_t d)
        override {
        dim = d;
        fitted = true;
    }

    void Encode(ParallelExecutor& /*executor*/, const float* in, float* out, size_t n, size_t d)
        const override {
        assert(fitted);
        if (in != out) {
            memcpy(out, in, n * d * sizeof(float));
        }
    }

    void Decode(ParallelExecutor& /*executor*/, const float* in, float* out, size_t n, size_t d)
        const override {
        assert(fitted);
        if (in != out) {
            memcpy(out, in, n * d * sizeof(float));
        }
    }

    void ComputeNorms(
        ParallelExecutor& /*executor*/,
        const float* data,
        size_t n,
        size_t d,
        float* out_norms
    ) const override {
        assert(fitted);
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(out_norms, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
    }

    void FindNearestNeighbor(
        ParallelExecutor& executor,
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
            executor, x, y, n_x, n_y, d, norms_x, norms_y, out_knn, out_distances, tmp_buf
        );
    }

    void CacheDataPartialNorms(
        ParallelExecutor& /*executor*/,
        const quantized_t* data,
        size_t n,
        size_t d,
        uint32_t partial_d
    ) override {
        cached_data_partial_norms.resize(n);
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(cached_data_partial_norms.data(), n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    void CacheCentroidPartialNorms(
        ParallelExecutor& /*executor*/,
        const quantized_t* centroids,
        size_t n,
        size_t d,
        uint32_t partial_d
    ) override {
        cached_centroid_partial_norms.resize(n);
        Eigen::Map<const MatrixR> e_data(centroids, n, d);
        Eigen::Map<VectorR> e_norms(cached_centroid_partial_norms.data(), n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    void FindNearestNeighborWithPruning(
        ParallelExecutor& executor,
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
        size_t* out_not_pruned_counts,
        float* tmp_buf
    ) const override {
        assert(fitted);
        assert(!cached_data_partial_norms.empty() && "CacheDataPartialNorms must be called first");
        assert(
            !cached_centroid_partial_norms.empty() &&
            "CacheCentroidPartialNorms must be called first"
        );

        batch_computer::FindNearestNeighborWithPruning(
            executor,
            x,
            y,
            n_x,
            n_y,
            d,
            cached_data_partial_norms.data(),
            cached_centroid_partial_norms.data(),
            out_knn,
            out_distances,
            tmp_buf,
            pdx_centroids,
            partial_d,
            out_not_pruned_counts
        );
    }

    void UpdateCentroids(
        ParallelExecutor& executor,
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
        uint32_t* cluster_sizes,
        size_t n,
        size_t n_clusters,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("F32::UpdateCentroids");
        // One rank per centroid range; the executor runs every rank exactly once.
        const size_t nt = executor.NumWorkers();
        executor.ParallelFor(nt, [&](size_t rank_begin, size_t rank_end, size_t) {
            for (size_t rank = rank_begin; rank < rank_end; ++rank) {
                const size_t c0 = (n_clusters * rank) / nt;
                const size_t c1 = (n_clusters * (rank + 1)) / nt;
                for (size_t i = 0; i < n; ++i) {
                    uint32_t ci = assignments[i];
                    if (ci >= c0 && ci < c1) {
                        const float* vec = encoded_data + i * d;
                        float* acc = centroid_accumulators + ci * d;
                        cluster_sizes[ci] += 1;
                        SKM_VECTORIZE_LOOP
                        for (size_t j = 0; j < d; ++j) {
                            acc[j] += vec[j];
                        }
                    }
                }
            }
        });
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return true; }
    size_t CodeSize(size_t d) const override { return d; }

  private:
    bool fitted = false;
    size_t dim = 0;
    std::vector<float> cached_data_partial_norms;
    std::vector<float> cached_centroid_partial_norms;
};

} // namespace skmeans
