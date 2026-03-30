#pragma once

#ifdef SKMEANS_ENABLE_GPU

#include <Eigen/Dense>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/profiler.h"

#include "superkmeans/gpu/gpu_batch_computers.h"
#include "utils.h"

namespace skmeans {

template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class GPUSuperKMeans : public skmeans::SuperKMeans<q, alpha> {
  public:
    virtual ~GPUSuperKMeans() = default;

  protected:
    // These are pulled in from the base class

    using MatrixR = typename skmeans::SuperKMeans<q, alpha>::MatrixR;
    using VectorR = typename skmeans::SuperKMeans<q, alpha>::VectorR;
    using centroid_value_t = typename skmeans::SuperKMeans<q, alpha>::centroid_value_t;
    using distance_t = typename skmeans::SuperKMeans<q, alpha>::distance_t;
    using layout_t = typename skmeans::SuperKMeans<q, alpha>::layout_t;
    using pruner_t = typename skmeans::SuperKMeans<q, alpha>::pruner_t;
    using vector_value_t = typename skmeans::SuperKMeans<q, alpha>::vector_value_t;

    using skmeans::SuperKMeans<q, alpha>::ComputeCost;
    using skmeans::SuperKMeans<q, alpha>::ComputeRecall;
    using skmeans::SuperKMeans<q, alpha>::ComputeShift;
    using skmeans::SuperKMeans<q, alpha>::ConsolidateCentroids;
    using skmeans::SuperKMeans<q, alpha>::GenerateCentroids;
    using skmeans::SuperKMeans<q, alpha>::GetGTAssignmentsAndDistances;
    using skmeans::SuperKMeans<q, alpha>::GetL2NormsRowMajor;
    using skmeans::SuperKMeans<q, alpha>::GetNVectorsToSample;
    using skmeans::SuperKMeans<q, alpha>::GetOutputCentroids;
    using skmeans::SuperKMeans<q, alpha>::GetPartialL2NormsRowMajor;
    using skmeans::SuperKMeans<q, alpha>::RotateOrCopy;
    using skmeans::SuperKMeans<q, alpha>::SampleAndRotateVectors;
    using skmeans::SuperKMeans<q, alpha>::ShouldStopEarly;
    using skmeans::SuperKMeans<q, alpha>::TunePartialD;
    using skmeans::SuperKMeans<q, alpha>::RunIteration;

    using gpu_batch_computer = skmeans::gpu::BatchComputer<alpha, q>;
    using cpu_batch_computer = skmeans::BatchComputer<alpha, q>;

  public:
    GPUSuperKMeans(size_t n_clusters, size_t dimensionality, const SuperKMeansConfig& config)
        : skmeans::SuperKMeans<q, alpha>(n_clusters, dimensionality, config) {}

    GPUSuperKMeans(size_t n_clusters, size_t dimensionality)
        : GPUSuperKMeans(n_clusters, dimensionality, SuperKMeansConfig{}) {}

    /**
     * @brief Run k-means clustering to determine centroids
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors (ignored if queries is nullptr and sample_queries is
     * false)
     *
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) override {
        SKMEANS_ENSURE_POSITIVE(n);
        if (trained) {
            throw std::runtime_error("The clustering has already been trained");
        }
        iteration_stats.clear();
        if (n < n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        if (n_queries > 0 && queries == nullptr && !config.sample_queries) {
            throw std::invalid_argument(
                "Queries must be provided if n_queries > 0 and sample_queries is false"
            );
        }

        const vector_value_t* SKM_RESTRICT data_p = data;
        n_samples = GetNVectorsToSample(n, n_clusters);

        // We use n, not n_samples for when fast assign is called afterwards
        gpu_device_context =
            gpu::GPUDeviceContext<skmeans_value_t<q>, skmeans_value_t<q>, distance_t>(
                n_samples, n_clusters, d, GPU_STREAM_POOL_SIZE
            );
        if (n_samples < n_clusters) {
            throw std::runtime_error(
                "Not enough samples to train. Try increasing the sampling_fraction or "
                "max_points_per_cluster"
            );
        }
        {
            SKM_PROFILE_SCOPE("allocator");
            centroids.reset(new centroid_value_t[n_clusters * d]);
            horizontal_centroids.reset(new centroid_value_t[n_clusters * d]);
            prev_centroids.reset(new centroid_value_t[n_clusters * d]);
            cluster_sizes.reset(new uint32_t[n_clusters]);
            assignments.reset(new uint32_t[n]);
            distances.reset(new distance_t[n]);
            data_norms.reset(new vector_value_t[n_samples]);
            centroid_norms.reset(new vector_value_t[n_clusters]);
        }
        std::vector<vector_value_t> centroids_partial_norms;
        centroids_partial_norms.reserve(n_clusters);
        std::vector<size_t> not_pruned_counts;
        not_pruned_counts.reserve(n_samples);
        std::vector<distance_t> tmp_distances_buf;
        tmp_distances_buf.reserve(X_BATCH_SIZE * Y_BATCH_SIZE);
        vertical_d = PDXLayout<q, alpha>::GetDimensionSplit(d).vertical_d;
        partial_horizontal_centroids.reset(new centroid_value_t[n_clusters * vertical_d]);

        // Set partial_d (d') dynamically as half of vertical_d (around 12% of d)
        partial_d = std::max<uint32_t>(MIN_PARTIAL_D, vertical_d / 2);
        if (partial_d > vertical_d) {
            partial_d = vertical_d;
        }
        if (config.verbose) {
            std::cout << "Front dimensions (d') = " << partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << d - vertical_d << std::endl;
        }

        auto centroids_pdx_wrapper =
            GenerateCentroids(data_p, n_samples, n_clusters, !config.data_already_rotated);
        if (config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::vector<vector_value_t> data_samples_buffer;
        data_samples_buffer.reserve(n_samples * d);
        auto data_to_cluster = SampleAndRotateVectors(
            data_p, data_samples_buffer.data(), n, n_samples, !config.data_already_rotated
        );

        RotateOrCopy(
            horizontal_centroids.get(),
            prev_centroids.get(),
            n_clusters,
            !config.data_already_rotated
        );

        GetL2NormsRowMajor(data_to_cluster, n_samples, data_norms.get());
        GetL2NormsRowMajor(prev_centroids.get(), n_clusters, centroid_norms.get());

        std::vector<vector_value_t> rotated_queries;
        if (n_queries) {
            centroids_to_explore =
                std::max<size_t>(static_cast<size_t>(n_clusters * config.ann_explore_fraction), 1);
            if (config.verbose) {
                std::cout << "Centroids to explore: " << centroids_to_explore << " ("
                          << config.ann_explore_fraction * 100.0f << "% of " << n_clusters << ")"
                          << std::endl;
            }
            {
                SKM_PROFILE_SCOPE("allocator");
                gt_assignments.reset(new uint32_t[n_queries * config.objective_k]);
                gt_distances.reset(new distance_t[n_queries * config.objective_k]);
                tmp_distances_buffer.reset(new distance_t[X_BATCH_SIZE * Y_BATCH_SIZE]);
                promising_centroids.reset(new uint32_t[n_queries * centroids_to_explore]);
                recall_distances.reset(new distance_t[n_queries * centroids_to_explore]);
                query_norms.reset(new distance_t[n_queries]);
            }
            rotated_queries.reserve(n_queries * d);
            if (config.sample_queries) {
                std::cout << "Sampling queries from data..." << std::endl;
                SampleAndRotateVectors(
                    data_to_cluster, rotated_queries.data(), n_samples, n_queries, false
                );
            } else {
                RotateOrCopy(
                    queries, rotated_queries.data(), n_queries, !config.data_already_rotated
                );
            }
            GetL2NormsRowMajor(rotated_queries.data(), n_queries, query_norms.get());
            GetGTAssignmentsAndDistances(data_to_cluster, rotated_queries.data(), n_queries);
        }

        bool always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING || config.use_blas_only ||
                                n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        bool partial_norms_computed = false;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;

        for (size_t iter_idx = 0; iter_idx < config.iters; ++iter_idx) {
            bool use_gemm_only = (iter_idx == 0) || always_gemm_only;
            if (!use_gemm_only && !partial_norms_computed) {
                GetPartialL2NormsRowMajor(data_to_cluster, n_samples, data_norms.get(), partial_d);
                partial_norms_computed = true;
            }

            if (use_gemm_only) {
                this->template RunIteration<true>(
                    data_to_cluster,
                    tmp_distances_buf.data(),
                    centroids_pdx_wrapper,
                    centroids_partial_norms,
                    not_pruned_counts,
                    rotated_queries.data(),
                    n_queries,
                    n_samples,
                    n_clusters,
                    iter_idx,
                    iter_idx == 0,
                    iteration_stats
                );
            } else {
                this->template RunIteration<false>(
                    data_to_cluster,
                    tmp_distances_buf.data(),
                    centroids_pdx_wrapper,
                    centroids_partial_norms,
                    not_pruned_counts,
                    rotated_queries.data(),
                    n_queries,
                    n_samples,
                    n_clusters,
                    iter_idx,
                    false,
                    iteration_stats
                );
            }
            if (config.early_termination &&
                ShouldStopEarly(n_queries > 0, best_recall, iters_without_improvement, iter_idx)) {
                break;
            }
        }

        trained = true;

        auto output_centroids = GetOutputCentroids(config.unrotate_centroids);
        if (config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /**
     * @brief Assign vectors to their nearest centroid using brute force search.
     *
     * The vectors and centroids are assumed to be in the same domain
     * (no rotation/transformation needed).
     *
     * @param vectors The data matrix (row-major, n_vectors x d)
     * @param centroids The centroids matrix (row-major, n_centroids x d)
     * @param n_vectors Number of vectors
     * @param n_centroids Number of centroids
     * @return std::vector<uint32_t> Assignment for each vector (index of nearest centroid)
     */
    [[nodiscard]] std::vector<uint32_t> Assign(
        const vector_value_t* SKM_RESTRICT vectors,
        const vector_value_t* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) override {
        SKM_PROFILE_SCOPE("assign");
        std::vector<uint32_t> result_assignments(n_vectors);
        std::unique_ptr<distance_t[]> tmp_distances_buf(
            new distance_t[X_BATCH_SIZE * Y_BATCH_SIZE]
        );
        std::vector<vector_value_t> vector_norms(n_vectors);
        std::vector<vector_value_t> centroid_norms_local(n_centroids);
        std::vector<distance_t> result_distances(n_vectors);

        Eigen::Map<const MatrixR> vectors_mat(vectors, n_vectors, d);
        Eigen::Map<VectorR> v_norms(vector_norms.data(), n_vectors);
        v_norms.noalias() = vectors_mat.rowwise().squaredNorm();

        Eigen::Map<const MatrixR> centroids_mat(centroids, n_centroids, d);
        Eigen::Map<VectorR> c_norms(centroid_norms_local.data(), n_centroids);
        c_norms.noalias() = centroids_mat.rowwise().squaredNorm();

        gpu_batch_computer::FindNearestNeighbor(
            vectors,
            centroids,
            n_vectors,
            n_centroids,
            d,
            vector_norms.data(),
            centroid_norms_local.data(),
            result_assignments.data(),
            result_distances.data(),
            tmp_distances_buf.get()
        );

        return result_assignments;
    }

    /**
     * @brief Fast assignment using GEMM+PRUNING with trained state.
     *
     * Assumes that the vectors sent here are the same as those used in .Train().
     * Leverages the assignments from the training for a faster
     * assignment than brute force Assign().
     *
     * @param vectors The data matrix (row-major, n_vectors x d)
     * @param centroids The centroids matrix (row-major, n_centroids x d)
     * @param n_vectors Number of vectors
     * @param n_centroids Number of centroids
     * @return std::vector<uint32_t> Assignment for each vector (index of nearest centroid)
     */
    [[nodiscard]] std::vector<uint32_t> FastAssign(
        const vector_value_t* SKM_RESTRICT vectors,
        const vector_value_t* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) override {
        SKM_PROFILE_SCOPE("fast_assign");
        if (!trained) {
            throw std::runtime_error("FastAssign requires SuperKMeans to be trained first");
        }

        if (config.use_blas_only || d < DIMENSION_THRESHOLD_FOR_PRUNING ||
            n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
            if (!config.suppress_warnings) {
                std::cout
                    << "WARNING: FastAssign cannot be used, falling back to brute force Assign"
                    << std::endl;
            }
            return Assign(vectors, centroids, n_vectors, n_centroids);
        }
        if (config.verbose) {
            Profiler::Get().Reset();
        }

        std::vector<uint32_t> result_assignments(n_vectors);
        std::vector<distance_t> tmp_distances_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

        partial_d = std::max<uint32_t>(MIN_PARTIAL_D, vertical_d / 2);

        std::vector<size_t> not_pruned_counts;
        not_pruned_counts.reserve(n_vectors);
        std::fill(not_pruned_counts.data(), not_pruned_counts.data() + n_vectors, 0);
        std::vector<vector_value_t> data_buffer;
        const vector_value_t* data_p;
        if (config.data_already_rotated) {
            // Data is already rotated: use original pointer directly (avoid redundant memcpy)
            data_p = vectors;
        } else {
            data_buffer.reserve(n_vectors * d);
            RotateOrCopy(vectors, data_buffer.data(), n_vectors, true);
            data_p = data_buffer.data();
        }
        GetPartialL2NormsRowMajor(
            horizontal_centroids.get(), n_centroids, centroid_norms.get(), partial_d
        );

        // Consolidate was called at the end of RunIteration<true>, so we don't need to call it here
        // All the centroid-related pointers are updated with the final centroids
        auto pdx_centroids = PDXLayout<q, alpha>(
            this->centroids.get(), *pruner, n_clusters, d, partial_horizontal_centroids.get()
        );

        // If nothing was sampled, then we just go ahead with GEMM+PRUNING
        if (config.sampling_fraction == 1.0f) {
            // Recompute data norms defensively (data_p is independently rotated)
            GetPartialL2NormsRowMajor(data_p, n_vectors, data_norms.get(), partial_d);
            // We do it on the GPU, as all vectors are there
            gpu_batch_computer::FindNearestNeighborWithPruning(
                gpu_device_context,
                data_p,
                horizontal_centroids.get(),
                n_vectors,
                n_clusters,
                d,
                data_norms.get(),
                centroid_norms.get(),
                assignments.get(),
                distances.get(),
                tmp_distances_buf.data(),
                pdx_centroids,
                partial_d,
                not_pruned_counts.data()
            );
            memcpy(result_assignments.data(), assignments.get(), n_vectors * sizeof(uint32_t));
            return result_assignments;
        } else if (config.sampling_fraction > 0.8f) {
            // Dereference the current assignments from the sampled_indices
            size_t cur_vector_idx = 0;
            for (; cur_vector_idx < n_samples; ++cur_vector_idx) {
                result_assignments[sampled_indices[cur_vector_idx]] = assignments[cur_vector_idx];
            }
            // Seed remaining vectors with a cluster drawn proportionally to cluster size
            std::mt19937 rng(config.seed + 1);
            std::discrete_distribution<uint32_t> cluster_dist(
                cluster_sizes.get(), cluster_sizes.get() + n_clusters
            );
            for (; cur_vector_idx < n_vectors; ++cur_vector_idx) {
                result_assignments[sampled_indices[cur_vector_idx]] = cluster_dist(rng);
            }

            // data_norms was allocated for n_samples in Train(), reallocate for n_vectors
            data_norms.reset(new vector_value_t[n_vectors]);
            GetPartialL2NormsRowMajor(data_p, n_vectors, data_norms.get(), partial_d);

            // We do it on the CPU, as only the sampled vectors are loaded on the GPU
            cpu_batch_computer::FindNearestNeighborWithPruning(
                data_p,
                horizontal_centroids.get(),
                n_vectors,
                n_clusters,
                d,
                data_norms.get(),
                centroid_norms.get(),
                result_assignments.data(),
                distances.get(),
                tmp_distances_buf.data(),
                pdx_centroids,
                partial_d,
                not_pruned_counts.data()
            );
            return result_assignments;
        } else {
            // When sampling_fraction is very low we don't have good initial assignments.
            // We obtain a good initial assignment by clustering the given centroids into
            // sqrt(n_centroids) meso-clusters, then map each vector's meso-assignment
            // back to a representative original centroid for seeding.
            SuperKMeansConfig tmp_config;
            tmp_config.iters = 10;
            tmp_config.sampling_fraction = 1.0f;
            tmp_config.use_blas_only = false;
            tmp_config.verbose = config.verbose;
            tmp_config.suppress_warnings = config.suppress_warnings;
            tmp_config.seed = config.seed;
            tmp_config.angular = config.angular;
            tmp_config.data_already_rotated = config.data_already_rotated;
            auto new_n_centroids = static_cast<size_t>(std::sqrt(n_centroids));
            SuperKMeans tmp_kmeans(new_n_centroids, d, tmp_config);
            auto meso_centroids = tmp_kmeans.Train(centroids, n_centroids);
            auto meso_assignments =
                tmp_kmeans.Assign(vectors, meso_centroids.data(), n_vectors, new_n_centroids);

            // Map each meso-centroid to a single representative original centroid
            auto centroids_to_meso =
                tmp_kmeans.Assign(centroids, meso_centroids.data(), n_centroids, new_n_centroids);
            std::vector<uint32_t> meso_to_original(new_n_centroids, 0);
            for (size_t c = 0; c < n_centroids; ++c) {
                meso_to_original[centroids_to_meso[c]] = static_cast<uint32_t>(c);
            }

            // Seed sampled vectors from training assignments
            size_t cur_vector_idx = 0;
            for (; cur_vector_idx < n_samples; ++cur_vector_idx) {
                result_assignments[sampled_indices[cur_vector_idx]] = assignments[cur_vector_idx];
            }
            // Seed non-sampled vectors: map their meso-assignment to an original centroid
            for (; cur_vector_idx < n_vectors; ++cur_vector_idx) {
                size_t orig_idx = sampled_indices[cur_vector_idx];
                result_assignments[orig_idx] = meso_to_original[meso_assignments[orig_idx]];
            }

            data_norms.reset(new vector_value_t[n_vectors]);
            GetPartialL2NormsRowMajor(data_p, n_vectors, data_norms.get(), partial_d);

            // We do it on the CPU, as only the sampled vectors are loaded on the GPU
            cpu_batch_computer::FindNearestNeighborWithPruning(
                data_p,
                horizontal_centroids.get(),
                n_vectors,
                n_clusters,
                d,
                data_norms.get(),
                centroid_norms.get(),
                result_assignments.data(),
                distances.get(),
                tmp_distances_buf.data(),
                pdx_centroids,
                partial_d,
                not_pruned_counts.data()
            );
            return result_assignments;
        }
    }

  protected:
    /**
     * @brief Performs first assignment and centroid update using FULL GEMM.
     *
     * Used for the first iteration where full distance computation via GEMM is used
     * (no pruning). Assigns each data point to its nearest centroid, then updates
     * centroid positions.
     *
     * @param data Data matrix (row-major, n_samples × d)
     * @param rotated_initial_centroids Initial centroids (row-major, n_clusters × d)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param n_samples Number of vectors in the data
     * @param n_clusters Number of centroids
     */
    void FirstAssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT rotated_initial_centroids,
        distance_t* SKM_RESTRICT tmp_distances_buf,
        const size_t n_samples,
        const size_t n_clusters
    ) override {
        gpu_batch_computer::FindNearestNeighborWithDeviceContext(
            gpu_device_context,
            data,
            rotated_initial_centroids,
            n_samples,
            n_clusters,
            d,
            data_norms.get(),
            centroid_norms.get(),
            assignments.get(),
            distances.get(),
            tmp_distances_buf
        );
        {
            SKM_PROFILE_SCOPE("fill");
            std::fill(
                horizontal_centroids.get(), horizontal_centroids.get() + (n_clusters * d), 0.0
            );
            std::fill(cluster_sizes.get(), cluster_sizes.get() + n_clusters, 0);
        }
    }

    /**
     * @brief Performs assignment and centroid update using GEMM+PRUNING.
     *
     * Uses GEMM for partial distance computation (first partial_d dimensions),
     * then PRUNING for completing distances for remaining candidates.
     *
     * @param data Data matrix (row-major, n_samples × d)
     * @param centroids Centroids to use for GEMM distance computation (row-major)
     * @param partial_centroid_norms Partial norms of centroids (first partial_d dims)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param pdx_centroids PDX-layout centroids for PRUNING
     * @param out_not_pruned_counts Output for pruning statistics
     */
    void AssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT centroids,
        const vector_value_t* SKM_RESTRICT partial_centroid_norms,
        distance_t* SKM_RESTRICT tmp_distances_buf,
        const layout_t& pdx_centroids,
        size_t* out_not_pruned_counts,
        const size_t n_samples,
        const size_t n_clusters
    ) override {
        gpu_batch_computer::FindNearestNeighborWithPruning(
            gpu_device_context,
            data,
            centroids,
            n_samples,
            n_clusters,
            d,
            data_norms.get(),
            partial_centroid_norms,
            assignments.get(),
            distances.get(),
            tmp_distances_buf,
            pdx_centroids,
            partial_d,
            out_not_pruned_counts
        );
        {
            SKM_PROFILE_SCOPE("fill");
            std::fill(
                horizontal_centroids.get(), horizontal_centroids.get() + (n_clusters * d), 0.0
            );
            std::fill(cluster_sizes.get(), cluster_sizes.get() + n_clusters, 0);
        }
    }

    void UpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n_samples,
        const size_t n_clusters
    ) override {
        gpu::DeviceBuffer<uint32_t> cluster_sizes_dev(
            gpu::compute_buffer_size<uint32_t>(n_clusters), gpu_device_context.main_stream.get()
        );
        gpu::DeviceBuffer<centroid_value_t> horizontal_centroids_dev(
            gpu::compute_buffer_size<centroid_value_t>(n_clusters * d),
            gpu_device_context.main_stream.get()
        );
        kernels::GPUUpdateCentroids(
            gpu_device_context.x.get(),
            n_clusters,
            n_samples,
            gpu_device_context.out_knn.get(),
            cluster_sizes_dev.get(),
            horizontal_centroids_dev.get(),
            d,
            gpu_device_context.main_stream.get()
        );

        cluster_sizes_dev.copy_to_host(cluster_sizes.get());
        horizontal_centroids_dev.copy_to_host(horizontal_centroids.get());
        gpu_device_context.main_stream.synchronize();
    }

    gpu::GPUDeviceContext<skmeans_value_t<q>, skmeans_value_t<q>, distance_t> gpu_device_context;

    // Bring in members of the base class
    using skmeans::SuperKMeans<q, alpha>::d;
    using skmeans::SuperKMeans<q, alpha>::n_clusters;
    using skmeans::SuperKMeans<q, alpha>::config;
    using skmeans::SuperKMeans<q, alpha>::n_threads;
    using skmeans::SuperKMeans<q, alpha>::n_samples;
    using skmeans::SuperKMeans<q, alpha>::partial_d;
    using skmeans::SuperKMeans<q, alpha>::trained;
    using skmeans::SuperKMeans<q, alpha>::n_split;
    using skmeans::SuperKMeans<q, alpha>::centroids_to_explore;
    using skmeans::SuperKMeans<q, alpha>::vertical_d;
    using skmeans::SuperKMeans<q, alpha>::prev_cost;
    using skmeans::SuperKMeans<q, alpha>::cost;
    using skmeans::SuperKMeans<q, alpha>::shift;
    using skmeans::SuperKMeans<q, alpha>::recall;
    using skmeans::SuperKMeans<q, alpha>::pruner;
    using skmeans::SuperKMeans<q, alpha>::centroids;
    using skmeans::SuperKMeans<q, alpha>::horizontal_centroids;
    using skmeans::SuperKMeans<q, alpha>::prev_centroids;
    using skmeans::SuperKMeans<q, alpha>::partial_horizontal_centroids;
    using skmeans::SuperKMeans<q, alpha>::distances;
    using skmeans::SuperKMeans<q, alpha>::cluster_sizes;
    using skmeans::SuperKMeans<q, alpha>::data_norms;
    using skmeans::SuperKMeans<q, alpha>::centroid_norms;
    using skmeans::SuperKMeans<q, alpha>::sampled_indices;
    using skmeans::SuperKMeans<q, alpha>::gt_assignments;
    using skmeans::SuperKMeans<q, alpha>::gt_distances;
    using skmeans::SuperKMeans<q, alpha>::query_norms;
    using skmeans::SuperKMeans<q, alpha>::tmp_distances_buffer;
    using skmeans::SuperKMeans<q, alpha>::promising_centroids;
    using skmeans::SuperKMeans<q, alpha>::recall_distances;

  public:
    using skmeans::SuperKMeans<q, alpha>::assignments;
    using skmeans::SuperKMeans<q, alpha>::iteration_stats;
};

} // namespace skmeans

#endif
