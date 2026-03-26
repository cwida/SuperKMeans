#pragma once

#include <cstdint>
#include <cstdio>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"

#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/gpu/utils.h"
#include "superkmeans/gpu/kernels.cuh"

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Eigen already declares sgemm_, so we don't need to redeclare it
// Use Eigen's BLAS declarations from Eigen/src/Core/util/BlasUtil.h

namespace skmeans {

namespace gpu {

template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::u8> {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::f32> {
    using distance_t = skmeans_distance_t<Quantization::f32>;
    using data_t = skmeans_value_t<Quantization::f32>;
    using norms_t = skmeans_value_t<Quantization::f32>;
    using knn_candidate_t = KNNCandidate<Quantization::f32>;
    using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;
    using MatrixR = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  public:
    /**
     * @brief Finds the nearest neighbor for each query vector using batched BLAS.
     *
     * Computes L2 distances between all query vectors (X) and reference vectors (Y)
     * using the identity: ||x-y||² = ||x||² + ||y||² - 2*x·y
     * The dot products are computed efficiently via matrix multiplication.
     *
     * @param x Query vectors in row-major layout (n_x × d)
     * @param y Reference vectors in row-major layout (n_y × d)
     * @param n_x Number of query vectors
     * @param n_y Number of reference vectors
     * @param d Dimensionality
     * @param norms_x Pre-computed squared L2 norms of query vectors
     * @param norms_y Pre-computed squared L2 norms of reference vectors
     * @param out_knn Output: index of nearest neighbor for each query
     * @param out_distances Output: distance to nearest neighbor for each query
     * @param all_distances_buf Scratch buffer for batch distance computation (size: X_BATCH_SIZE ×
     * Y_BATCH_SIZE)
     */
    static void FindNearestNeighbor(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf
    ) {
        SKM_PROFILE_SCOPE("1st_blas");

        std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());

        auto stream = gpu::ManagedCudaStream();
        auto y_dev =
            gpu::DeviceBuffer<data_t>(gpu::compute_buffer_size<data_t>(n_y, d), stream.get());
        y_dev.copy_to_device(y);
        auto multiplier = gpu::BatchedMatrixMultiplier(stream.get());

        constexpr int32_t N_BATCH_STREAMS = 4;
        auto batch_streams = std::vector<gpu::ManagedCudaStream>(N_BATCH_STREAMS);
        auto batch_multipliers = std::vector<gpu::BatchedMatrixMultiplier>();
        auto batch_x_buffers_dev = std::vector<gpu::DeviceBuffer<data_t>>();
        auto batch_all_distances_buffers_dev = std::vector<gpu::DeviceBuffer<float>>();

        batch_multipliers.reserve(N_BATCH_STREAMS);
        batch_all_distances_buffers_dev.reserve(N_BATCH_STREAMS);

        for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
            batch_multipliers.emplace_back(batch_streams[i].get());
            batch_x_buffers_dev.emplace_back(
                gpu::compute_buffer_size<float>(X_BATCH_SIZE, d), batch_streams[i].get()
            );
            batch_all_distances_buffers_dev.emplace_back(
                gpu::compute_buffer_size<float>(X_BATCH_SIZE, Y_BATCH_SIZE), batch_streams[i].get()
            );
        }

        auto norms_x_dev =
            gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_x), stream.get());
        auto norms_y_dev =
            gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_y), stream.get());
        auto out_distances_dev =
            gpu::DeviceBuffer<distance_t>(gpu::compute_buffer_size<distance_t>(n_x), stream.get());
        auto out_knn_dev =
            gpu::DeviceBuffer<uint32_t>(gpu::compute_buffer_size<uint32_t>(n_x), stream.get());
        norms_x_dev.copy_to_device(norms_x);
        norms_y_dev.copy_to_device(norms_y);
        // all_distances_buf_dev.copy_to_device(all_distances_buf);
        out_distances_dev.copy_to_device(out_distances);
        out_knn_dev.copy_to_device(out_knn);
        stream.synchronize();

        size_t iteration_count = 0;
        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = x + (i * d);
            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }

            auto batch_stream_index = iteration_count % N_BATCH_STREAMS;
            ++iteration_count;

            batch_x_buffers_dev[batch_stream_index].copy_to_device(
                batch_x_p, gpu::compute_buffer_size<data_t>(batch_n_x, d)
            );
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                auto batch_y_p = y_dev.get() + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }
                batch_multipliers[batch_stream_index].multiply(
                    batch_x_buffers_dev[batch_stream_index].get(),
                    batch_y_p,
                    batch_n_x,
                    batch_n_y,
                    d,
                    0,
                    batch_all_distances_buffers_dev[batch_stream_index].get()
                );

                kernels::first_blas(
                    batch_n_x,
                    batch_n_y,
                    i,
                    j,
                    norms_x_dev.get(),
                    norms_y_dev.get(),
                    batch_all_distances_buffers_dev[batch_stream_index].get(),
                    out_distances_dev.get(),
                    out_knn_dev.get(),
                    batch_streams[batch_stream_index].get()
                );
            }
        }
        stream.synchronize();
        for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
            batch_streams[i].synchronize();
        }
        out_distances_dev.copy_to_host(out_distances);
        out_knn_dev.copy_to_host(out_knn);
        stream.synchronize();
    }

    static void FindNearestNeighborWithDeviceContext(
        gpu::GPUDeviceContext<data_t, norms_t, distance_t>& gpu_device_context,
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf
    ) {
        SKM_PROFILE_SCOPE("1st_blas");

        std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());

        auto stream = gpu::ManagedCudaStream();

        gpu_device_context.y.copy_to_device(y);
        auto multiplier = gpu::BatchedMatrixMultiplier(stream.get());

        constexpr int32_t N_BATCH_STREAMS = 4;
        auto batch_streams = std::vector<gpu::ManagedCudaStream>(N_BATCH_STREAMS);
        auto batch_multipliers = std::vector<gpu::BatchedMatrixMultiplier>();
        auto batch_x_buffers_dev = std::vector<gpu::DeviceBuffer<data_t>>();
        auto batch_all_distances_buffers_dev = std::vector<gpu::DeviceBuffer<float>>();

        batch_multipliers.reserve(N_BATCH_STREAMS);
        batch_all_distances_buffers_dev.reserve(N_BATCH_STREAMS);

        for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
            batch_multipliers.emplace_back(batch_streams[i].get());
            batch_x_buffers_dev.emplace_back(
                gpu::compute_buffer_size<float>(X_BATCH_SIZE, d), batch_streams[i].get()
            );
            batch_all_distances_buffers_dev.emplace_back(
                gpu::compute_buffer_size<float>(X_BATCH_SIZE, Y_BATCH_SIZE), batch_streams[i].get()
            );
        }

        gpu_device_context.norms_x.copy_to_device(norms_x);
        gpu_device_context.norms_y.copy_to_device(norms_y);
        gpu_device_context.out_distances.copy_to_device(out_distances);
        gpu_device_context.out_knn.copy_to_device(out_knn);
        stream.synchronize();

        size_t iteration_count = 0;
        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = x + (i * d);
            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }

            auto batch_stream_index = iteration_count % N_BATCH_STREAMS;
            ++iteration_count;

            if (!gpu_device_context.loaded_x_data) {
                gpu_device_context.x.copy_to_device_at_offset(
                    batch_x_p,
                    gpu::compute_buffer_size<data_t>(batch_n_x, d),
                    i * d * sizeof(data_t)
                );
            }
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                auto batch_y_p = gpu_device_context.y.get() + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }
                batch_multipliers[batch_stream_index].multiply(
                    gpu_device_context.x.get() + i * d,
                    batch_y_p,
                    batch_n_x,
                    batch_n_y,
                    d,
                    0,
                    batch_all_distances_buffers_dev[batch_stream_index].get()
                );

                kernels::first_blas(
                    batch_n_x,
                    batch_n_y,
                    i,
                    j,
                    gpu_device_context.norms_x.get(),
                    gpu_device_context.norms_y.get(),
                    batch_all_distances_buffers_dev[batch_stream_index].get(),
                    gpu_device_context.out_distances.get(),
                    gpu_device_context.out_knn.get(),
                    batch_streams[batch_stream_index].get()
                );
            }
        }
        stream.synchronize();
        for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
            batch_streams[i].synchronize();
        }
        gpu_device_context.out_distances.copy_to_host(out_distances);
        gpu_device_context.out_knn.copy_to_host(out_knn);
        stream.synchronize();
        gpu_device_context.loaded_x_data = true;
    }

    /**
     * @brief Finds nearest neighbors using partial BLAS computation with PDX pruning.
     *
     * Hybrid approach that computes partial distances (first partial_d dimensions)
     * via BLAS, then uses ADSampling pruning to skip full distance computation
     * for unlikely candidates. Significantly faster than full BLAS when pruning
     * is effective. To prune even better, we get a threshold from the previously assigned centroid.
     *
     * @param x Query vectors in row-major layout (n_x × d)
     * @param y Reference vectors in row-major layout (n_y × d)
     * @param n_x Number of query vectors
     * @param n_y Number of reference vectors (centroids)
     * @param d Full dimensionality
     * @param norms_x Pre-computed partial squared L2 norms of queries (first partial_d dims)
     * @param norms_y Pre-computed partial squared L2 norms of references (first partial_d dims)
     * @param out_knn Input/Output: current assignment indices (updated with better assignments)
     * @param out_distances Input/Output: current distances (updated with better distances)
     * @param all_distances_buf Scratch buffer for batch distance computation
     * @param pdx_centroids PDX layout containing centroids and searcher for pruned search
     * @param partial_d Number of dimensions used for initial BLAS computation
     * @param out_not_pruned_counts Optional output: count of non-pruned vectors per query (for
     * tuning)
     */

    static void FindNearestNeighborWithPruning(
        gpu::GPUDeviceContext<data_t, norms_t, distance_t>& gpu_device_context,
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf,
        const layout_t& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts = nullptr
    ) {
        SKM_PROFILE_SCOPE("search");

        // gpu_device_context.sw.start("Set up");

        auto& stream = gpu_device_context.main_stream;

        gpu_device_context.y.copy_to_device(y);
        gpu_device_context.norms_x.copy_to_device(norms_x);
        gpu_device_context.norms_y.copy_to_device(norms_y);

        gpu_device_context.out_knn.copy_to_device(out_knn);
        gpu_device_context.out_not_pruned_counts.copy_to_device(out_not_pruned_counts);

        auto constant_prune_data = kernels::ConstantPruneData(pdx_centroids, stream.get());

        const size_t n_y_clusters = (n_y + Y_BATCH_SIZE - 1) / Y_BATCH_SIZE;
        auto cluster_data = std::vector<kernels::ClusterPruneData>();
        cluster_data.reserve(n_y_clusters);

        for (size_t i = 0; i < n_y_clusters; ++i) {
            auto y_batch_cluster = pdx_centroids.searcher->pdx_data.clusters[i];
            cluster_data.emplace_back(
                y_batch_cluster, constant_prune_data.as_view(), n_y, stream.get()
            );
        }

        stream.synchronize();
        // gpu_device_context.sw.stop("Set up");
        // gpu_device_context.sw.start("Main");

        size_t iteration_count = 0;
        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = gpu_device_context.x.get() + (i * d);

            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }

            auto current_stream = iteration_count % gpu_device_context.stream_pool.size();
            iteration_count += 1;

            // batch_stream_buffers[current_stream].batch_x_buffer_dev.copy_to_device(
            //     batch_x_p, gpu::compute_buffer_size<data_t>(batch_n_x, d)
            // );

            kernels::GPUCalculateDistanceToCurrentCentroids(
                batch_n_x,
                n_y,
                d,
                batch_x_p, // batch_stream_buffers[current_stream].batch_x_buffer_dev.get(),
                gpu_device_context.y.get(),                 // const
                gpu_device_context.out_knn.get() + i,       // const
                gpu_device_context.out_distances.get() + i, // mutated
                gpu_device_context.stream_buffers[current_stream].stream
            );

            MatrixR materialize_x_left_cols;
            size_t current_y_batch = 0;
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                // auto batch_y_p = y + (j * d);
                auto batch_y_p = gpu_device_context.y.get() + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }

                gpu_device_context.stream_buffers[current_stream].multiplier.multiply(
                    batch_x_p, // batch_stream_buffers[current_stream].batch_x_buffer_dev.get(), //
                               // const
                    batch_y_p, // const
                    batch_n_x,
                    batch_n_y,
                    d,
                    partial_d,
                    gpu_device_context.stream_buffers[current_stream]
                        .all_distances_buf_dev.get() // mutated
                );
                kernels::norms(
                    batch_n_x,
                    batch_n_y,
                    i,
                    j,
                    gpu_device_context.norms_x.get(), // const
                    gpu_device_context.norms_y.get(), // const
                    gpu_device_context.stream_buffers[current_stream]
                        .all_distances_buf_dev.get(), // mutated
                    gpu_device_context.stream_buffers[current_stream].stream
                );

                kernels::GPUSearchPDX(
                    batch_n_x,
                    batch_n_y,
                    d,
                    partial_d,
                    batch_x_p, // batch_stream_buffers[current_stream].batch_x_buffer_dev.get(), //
                               // const
                    constant_prune_data.as_view(),                      // const
                    cluster_data[current_y_batch].as_view(),            // const
                    gpu_device_context.out_knn.get() + i,               // mutated
                    gpu_device_context.out_distances.get() + i,         // mutated
                    gpu_device_context.out_not_pruned_counts.get() + i, // mutated
                    gpu_device_context.stream_buffers[current_stream]
                        .all_distances_buf_dev.get(), // mutated
                    gpu_device_context.stream_buffers[current_stream].stream
                );

                current_y_batch += 1;
            }
        }
        gpu_device_context.stream_pool.synchronize();
        // gpu_device_context.sw.stop("Main");
        // gpu_device_context.sw.start("Clean up");
        gpu_device_context.out_knn.copy_to_host(out_knn);
        gpu_device_context.out_distances.copy_to_host(out_distances);
        gpu_device_context.out_not_pruned_counts.copy_to_host(out_not_pruned_counts);
        stream.synchronize();
        // gpu_device_context.sw.stop("Clean up");
    }
};

} // namespace gpu
} // namespace skmeans
