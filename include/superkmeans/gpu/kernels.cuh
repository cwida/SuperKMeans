#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include <memory>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
// #include "superkmeans/profiler.h"

#include "superkmeans/gpu/utils.h"

namespace skmeans {
namespace kernels {

using distance_t = skmeans_distance_t<Quantization::f32>;
using data_t = skmeans_value_t<Quantization::f32>;
using norms_t = skmeans_value_t<Quantization::f32>;
using knn_candidate_t = KNNCandidate<Quantization::f32>;
using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;

void trigger_gpu_initialization();

void first_blas(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    distance_t* out_distances,
    uint32_t* out_knn,
    const cudaStream_t stream
);

void norms(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    const cudaStream_t stream
);
struct ConstantPruneDataView {
    uint32_t num_dimensions;
    uint32_t num_horizontal_dimensions;
    uint32_t num_vertical_dimensions;
    float* ratios;
};

class CPUConstantPruneData {
  public:
    size_t num_dimensions;
    size_t num_horizontal_dimensions;
    size_t num_vertical_dimensions;
    std::unique_ptr<float[]> ratios;

    CPUConstantPruneData(const layout_t& pdx_centroids) {
        num_dimensions = pdx_centroids.searcher->pdx_data.num_dimensions;
        num_horizontal_dimensions = pdx_centroids.searcher->pdx_data.num_horizontal_dimensions;
        num_vertical_dimensions = pdx_centroids.searcher->pdx_data.num_vertical_dimensions;
        ratios = std::make_unique<float[]>(pdx_centroids.searcher->pruner.ratios.size());
        std::copy(
            pdx_centroids.searcher->pruner.ratios.begin(),
            pdx_centroids.searcher->pruner.ratios.end(),
            ratios.get()
        );
    }

    ConstantPruneDataView as_view() const {
        return ConstantPruneDataView{
            static_cast<uint32_t>(num_dimensions),
            static_cast<uint32_t>(num_horizontal_dimensions),
            static_cast<uint32_t>(num_vertical_dimensions),
            ratios.get(),
        };
    }
};

class ConstantPruneData {
  public:
    uint32_t num_dimensions;
    uint32_t num_horizontal_dimensions;
    uint32_t num_vertical_dimensions;
    gpu::DeviceBuffer<float> ratios;

    ConstantPruneData(const layout_t& pdx_centroids, const cudaStream_t stream)
        : ratios(
              gpu::compute_buffer_size<float>(pdx_centroids.searcher->pruner.ratios.size()),
              stream
          ) {
        num_dimensions = pdx_centroids.searcher->pdx_data.num_dimensions;
        num_horizontal_dimensions = pdx_centroids.searcher->pdx_data.num_horizontal_dimensions;
        num_vertical_dimensions = pdx_centroids.searcher->pdx_data.num_vertical_dimensions;
        ratios.copy_to_device(pdx_centroids.searcher->pruner.ratios.data());
    }

    ConstantPruneDataView as_view() const {
        return ConstantPruneDataView{
            num_dimensions,
            num_horizontal_dimensions,
            num_vertical_dimensions,
            ratios.get(),
        };
    }
};

struct ClusterPruneDataView {
    skmeans_value_t<Quantization::f32>* data;
    size_t n_vectors;
    uint32_t* vector_indices;
    skmeans_value_t<Quantization::f32>* aux_vertical_dimensions_in_horizontal_layout;
};

class CPUClusterPruneData {
  public:
    std::unique_ptr<skmeans_value_t<Quantization::f32>[]> data;
    size_t n_vectors;
    std::unique_ptr<uint32_t[]> vector_indices;
    std::unique_ptr<skmeans_value_t<Quantization::f32>[]>
        aux_vertical_dimensions_in_horizontal_layout;

    CPUClusterPruneData(
        Cluster<Quantization::f32> y_batch_cluster,
        const ConstantPruneDataView& constant_prune_data,
        const size_t n_centroids
    ) {
        const auto data_buffer_size =
            y_batch_cluster.num_embeddings * constant_prune_data.num_dimensions;
        data = std::make_unique<skmeans_value_t<Quantization::f32>[]>(data_buffer_size);
        std::copy(y_batch_cluster.data, y_batch_cluster.data + data_buffer_size, data.get());

        n_vectors = y_batch_cluster.num_embeddings;
        vector_indices = std::make_unique<uint32_t[]>(y_batch_cluster.num_embeddings);
        std::copy(
            y_batch_cluster.indices,
            y_batch_cluster.indices + y_batch_cluster.num_embeddings,
            vector_indices.get()
        );

        const auto aux_size =
            constant_prune_data.num_vertical_dimensions * y_batch_cluster.num_embeddings;
        aux_vertical_dimensions_in_horizontal_layout =
            std::make_unique<skmeans_value_t<Quantization::f32>[]>(aux_size);

        std::copy(
            y_batch_cluster.aux_vertical_dimensions_in_horizontal_layout,
            y_batch_cluster.aux_vertical_dimensions_in_horizontal_layout + aux_size,
            aux_vertical_dimensions_in_horizontal_layout.get()
        );
    }

    ClusterPruneDataView as_view() const {
        return ClusterPruneDataView{
            data.get(),
            n_vectors,
            vector_indices.get(),
            aux_vertical_dimensions_in_horizontal_layout.get()
        };
    }
};

class ClusterPruneData {
  public:
    gpu::DeviceBuffer<skmeans_value_t<Quantization::f32>> data;
    size_t n_vectors;
    gpu::DeviceBuffer<uint32_t> vector_indices;
    gpu::DeviceBuffer<skmeans_value_t<Quantization::f32>>
        aux_vertical_dimensions_in_horizontal_layout;

    ClusterPruneData(
        Cluster<Quantization::f32> y_batch_cluster,
        const ConstantPruneDataView& constant_prune_data,
        const size_t n_centroids,
        const cudaStream_t stream
    )
        : data(

              gpu::compute_buffer_size<skmeans_value_t<Quantization::f32>>(
                  y_batch_cluster.num_embeddings,
                  constant_prune_data.num_dimensions
              ),
              stream
          ),
          n_vectors(y_batch_cluster.num_embeddings),
          vector_indices(
              gpu::compute_buffer_size<uint32_t>(y_batch_cluster.num_embeddings),
              stream
          ),
          aux_vertical_dimensions_in_horizontal_layout(
              gpu::compute_buffer_size<skmeans_value_t<Quantization::f32>>(
                  constant_prune_data.num_vertical_dimensions * y_batch_cluster.num_embeddings
              ),
              stream

          )

    {
        data.copy_to_device(y_batch_cluster.data);
        vector_indices.copy_to_device(y_batch_cluster.indices);
        aux_vertical_dimensions_in_horizontal_layout.copy_to_device(
            y_batch_cluster.aux_vertical_dimensions_in_horizontal_layout
        );
    }

    ClusterPruneDataView as_view() const {
        return ClusterPruneDataView{
            data.get(),
            n_vectors,
            vector_indices.get(),
            aux_vertical_dimensions_in_horizontal_layout.get()
        };
    }
};

void GPUCalculateDistanceToCurrentCentroids(
    const uint32_t n_x,
    const uint32_t n_y,
    const uint32_t d,
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    const cudaStream_t stream
);

void GPUSearchPDX(
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const uint32_t partial_d,
    const data_t* SKM_RESTRICT x,
    const ConstantPruneDataView constant_prune_data,
    const ClusterPruneDataView cluster_prune_data,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    size_t* SKM_RESTRICT out_not_pruned_counts,
    float* SKM_RESTRICT all_distances_buf,
    const cudaStream_t stream
);

void GPUUpdateCentroids(
    const data_t* SKM_RESTRICT data,
    const size_t n_clusters,
    const size_t n_samples,
    const uint32_t* SKM_RESTRICT assignments,
    uint32_t* SKM_RESTRICT cluster_sizes,
    data_t* SKM_RESTRICT horizontal_centroids,
    const size_t d,
    const cudaStream_t stream
);

} // namespace kernels
} // namespace skmeans
