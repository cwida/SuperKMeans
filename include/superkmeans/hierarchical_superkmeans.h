#pragma once

#include <atomic>

#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace skmeans {

/**
 * @brief Configuration parameters for Hierarchical SuperKMeans clustering.
 */
struct HierarchicalSuperKMeansConfig : SuperKMeansConfig {
    uint32_t iters_mesoclustering = 3;
    uint32_t iters_fineclustering = 5;
    uint32_t iters_refinement = 0; // Refinement iteration is not needed to achieve good recall

    HierarchicalSuperKMeansConfig() { sampling_fraction = 1.0f; }
};

/**
 * @brief Statistics for HierarchicalSuperKMeans clustering.
 */
struct HierarchicalSuperKMeansIterationStats {
    std::vector<SuperKMeansIterationStats> mesoclustering_iteration_stats;
    std::vector<SuperKMeansIterationStats> refinement_iteration_stats;
    std::vector<SuperKMeansIterationStats> fineclustering_iteration_stats;
};

/**
 * @brief Calculate the number of mesoclusters for a given number of clusters
 *
 * @param n_clusters Total number of clusters
 * @return Number of mesoclusters
 */
inline size_t GetNMesoclusters(const size_t n_clusters) {
    return static_cast<size_t>(std::round(std::sqrt(n_clusters)));
}

/**
 * @brief Arrange fine clusters proportionally to mesocluster sizes
 *
 * Allocates the total number of clusters across mesoclusters proportionally
 * to their sizes, ensuring balanced distribution.
 *
 * @param n_clusters Total number of fine clusters to distribute
 * @param n_mesoclusters Number of mesoclusters
 * @param n_samples Total number of samples
 * @param mesocluster_sizes Sizes of each mesocluster
 * @return Vector of fine cluster counts per mesocluster
 */
inline std::vector<size_t> ArrangeFineClusters(
    size_t n_clusters,
    size_t n_mesoclusters,
    size_t n_samples,
    const std::vector<uint32_t>& mesocluster_sizes
) {
    SKM_PROFILE_SCOPE("arrange_fine_clusters");
    std::vector<size_t> fine_clusters_nums(n_mesoclusters);

    size_t n_clusters_remaining = n_clusters;
    size_t n_nonempty_mesoclusters_remaining = 0;
    size_t last_nonempty_mesocluster = 0;
    for (size_t i = 0; i < n_mesoclusters; ++i) {
        if (mesocluster_sizes[i] > 0) {
            n_nonempty_mesoclusters_remaining++;
            last_nonempty_mesocluster = i;
        }
    }

    size_t n_samples_remaining = n_samples;
    for (size_t i = 0; i < n_mesoclusters; ++i) {
        if (mesocluster_sizes[i] == 0) {
            fine_clusters_nums[i] = 0;
        } else if (i == last_nonempty_mesocluster) {
            // Last non-empty mesocluster gets all remaining clusters
            fine_clusters_nums[i] = n_clusters_remaining;
        } else {
            n_nonempty_mesoclusters_remaining--;
            double proportion = static_cast<double>(n_clusters_remaining * mesocluster_sizes[i]) /
                                static_cast<double>(n_samples_remaining);
            size_t allocated = static_cast<size_t>(std::lround(proportion));
            allocated =
                std::min(allocated, n_clusters_remaining - n_nonempty_mesoclusters_remaining);
            fine_clusters_nums[i] = std::max(allocated, size_t{1});
        }
        n_clusters_remaining -= fine_clusters_nums[i];
        n_samples_remaining -= mesocluster_sizes[i];
    }

    return fine_clusters_nums;
}

inline std::vector<size_t> GroupPointsByMesocluster(
    const uint32_t* SKM_RESTRICT labels,
    const size_t n_points,
    const std::vector<uint32_t>& mesoclusters_sizes,
    size_t* SKM_RESTRICT mesocluster_indices
) {
    SKM_PROFILE_SCOPE("compact_indices");
    std::vector<size_t> mesocluster_offsets(mesoclusters_sizes.size() + 1, 0);
    for (size_t k = 0; k < mesoclusters_sizes.size(); ++k) {
        mesocluster_offsets[k + 1] = mesocluster_offsets[k] + mesoclusters_sizes[k];
    }
    std::vector<size_t> next_to_write_index = mesocluster_offsets;
    for (size_t i = 0; i < n_points; ++i) {
        mesocluster_indices[next_to_write_index[labels[i]]++] = i;
    }
    return mesocluster_offsets;
}

template <Quantization q>
struct FineclusteringContext {
    const float* data_to_cluster;
    const skmeans_value_t<q>* encoded_data;
    const float* data_norms;
    const size_t* mesocluster_indices;
    const size_t* mesocluster_offsets;
    const uint32_t* mesoclusters_sizes;
    const size_t* fine_clusters_nums;
    const size_t* fineclusters_offsets;
    uint32_t initial_partial_d;
    uint32_t* final_assignments;
    skmeans_distance_t* final_distances;
    uint32_t* final_cluster_sizes;
    skmeans_centroid_value_t* final_centroids;
    std::vector<SuperKMeansIterationStats>* fineclustering_stats;
};

inline std::vector<size_t> OrderMesoclustersBySize(
    const std::vector<uint32_t>& mesoclusters_sizes,
    const std::vector<size_t>& fine_clusters_nums
) {
    std::vector<size_t> mesocluster_order;
    for (size_t k = 0; k < fine_clusters_nums.size(); ++k) {
        if (fine_clusters_nums[k] > 0) {
            mesocluster_order.push_back(k);
        }
    }
    std::stable_sort(
        mesocluster_order.begin(),
        mesocluster_order.end(),
        [&](size_t a, size_t b) { return mesoclusters_sizes[a] > mesoclusters_sizes[b]; }
    );
    return mesocluster_order;
}

template <Quantization q = Quantization::f32>
class HierarchicalSuperKMeans : public SuperKMeans<q> {
    using typename SuperKMeans<q>::centroid_value_t;
    using typename SuperKMeans<q>::vector_value_t;
    using typename SuperKMeans<q>::distance_t;
    using typename SuperKMeans<q>::MatrixR;
    using typename SuperKMeans<q>::VectorR;
    using typename SuperKMeans<q>::pruner_t;
    using typename SuperKMeans<q>::layout_t;
    using typename SuperKMeans<q>::batch_computer;

  public:
    /**
     * @brief Constructor with custom configuration
     */
    HierarchicalSuperKMeans(
        size_t n_clusters,
        size_t dimensionality,
        const HierarchicalSuperKMeansConfig& config
    )
        : SuperKMeans<q>(n_clusters, dimensionality, config), hierarchical_config(config) {
        this->pruner = std::make_unique<pruner_t>(
            dimensionality,
            HIERARCHICAL_PRUNER_INITIAL_THRESHOLD,
            this->config.seed,
            this->config.data_already_rotated,
            &this->GetAllocator()
        );
        this->pruner->SetExecutor(&this->GetExecutor());
        SKMEANS_ENSURE_POSITIVE(config.iters_mesoclustering);
        SKMEANS_ENSURE_POSITIVE(config.iters_fineclustering);

        static_cast<SuperKMeansConfig&>(hierarchical_config) = this->config;
        hierarchical_config.sampling_fraction = config.sampling_fraction;

        if (n_clusters <= 128) {
            if (!this->hierarchical_config.suppress_warnings) {
                std::cout
                    << "WARNING: n_clusters <= 128 is not recommended for HierarchicalSuperKMeans. "
                       "Consider using at least 128 clusters."
                    << std::endl;
            }
        }
    }

    /**
     * @brief Default constructor
     */
    HierarchicalSuperKMeans(size_t n_clusters, size_t dimensionality)
        : HierarchicalSuperKMeans(n_clusters, dimensionality, HierarchicalSuperKMeansConfig{}) {}

    /**
     * @brief Run hierarchical k-means clustering to determine centroids
     * We don't support Early Termination by Recall here.
     * queries and n_queries are ignored. But we keep the function signature for compatibility.
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     *
     * @return std::vector<skmeans_centroid_value_t> Trained centroids
     */
    std::vector<skmeans_centroid_value_t> Train(
        const float* SKM_RESTRICT data,
        const size_t n,
        const float* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        SKMEANS_ENSURE_POSITIVE(n);
        if (this->state.trained) {
            throw std::runtime_error("The clustering has already been trained");
        }
        if (n_queries > 0) {
            if (!this->hierarchical_config.suppress_warnings) {
                std::cout << "WARNING: Early Termination by Recall is not supported in "
                             "HierarchicalSuperKMeans"
                          << std::endl;
            }
        }
        n_mesoclusters = GetNMesoclusters(this->n_clusters);
        hierarchical_iteration_stats.fineclustering_iteration_stats.clear();
        hierarchical_iteration_stats.mesoclustering_iteration_stats.clear();
        hierarchical_iteration_stats.refinement_iteration_stats.clear();
        if (n < this->n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        ParallelSection parallel_section(this->GetExecutor());
        const float* SKM_RESTRICT data_p = data;
        this->n_samples = this->GetNVectorsToSample(n, this->n_clusters);
        this->n_train = n;
        this->state.training_data_rotated = this->hierarchical_config.data_already_rotated;
        this->state.rotator = this->pruner.get();
        if (this->n_samples < this->n_clusters) {
            throw std::runtime_error(
                "Not enough samples to train. Try increasing the sampling_fraction or "
                "max_points_per_cluster"
            );
        }
        {
            SKM_PROFILE_SCOPE("allocator");
            // Buffers to concatenate each fineclustering assignments and centroids
            auto& allocator = this->GetAllocator();
            this->final_assignments = AllocateBuffer<uint32_t>(allocator, this->n_samples);
            this->final_centroids =
                AllocateBuffer<centroid_value_t>(allocator, this->n_clusters * this->d);
            // These buffers are reused for all three phases
            this->centroids =
                AllocateBuffer<centroid_value_t>(allocator, this->n_clusters * this->d);
            this->horizontal_centroids =
                AllocateBuffer<centroid_value_t>(allocator, this->n_clusters * this->d);
            this->prev_centroids =
                AllocateBuffer<centroid_value_t>(allocator, this->n_clusters * this->d);
            this->cluster_sizes = AllocateBuffer<uint32_t>(allocator, this->n_clusters);
            this->assignments = AllocateBuffer<uint32_t>(allocator, n);
            this->distances = AllocateBuffer<distance_t>(allocator, n);
            this->data_norms = AllocateBuffer<float>(allocator, this->n_samples);
            this->centroid_norms = AllocateBuffer<float>(allocator, this->n_clusters);
        }
        this->EnsureTmpDistancesBuffer();
        this->vertical_d = PDXLayout<q>::GetDimensionSplit(this->PDXDim(this->d)).vertical_d;
        this->partial_horizontal_centroids = AllocateBuffer<centroid_value_t>(
            this->GetAllocator(), this->n_clusters * this->vertical_d
        );

        //
        // MESOCLUSTERING
        //
        TicToc timer_mesoclustering;
        timer_mesoclustering.Tic();
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 1: MESOCLUSTERING (k=" << n_mesoclusters
                      << " clusters) ===" << std::endl;
        }
        auto centroids_pdx_wrapper = this->GenerateCentroids(
            data_p, this->n_samples, n_mesoclusters, !this->hierarchical_config.data_already_rotated
        );
        if (this->hierarchical_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }
        // Samples for both mesoclustering and fineclustering
        AllocatedBuffer<float> data_samples_buffer;
        if (this->n_samples < n || !this->hierarchical_config.data_already_rotated) {
            data_samples_buffer =
                AllocateBuffer<float>(this->GetAllocator(), this->n_samples * this->d);
        }
        auto data_to_cluster = this->SampleAndRotateVectors(
            data_p,
            data_samples_buffer.get(),
            n,
            this->n_samples,
            !this->hierarchical_config.data_already_rotated
        );
        this->RotateOrCopy(
            this->horizontal_centroids.get(),
            this->prev_centroids.get(),
            n_mesoclusters,
            !this->hierarchical_config.data_already_rotated
        );
        // Create quantizer
        this->quantizer = this->CreateQuantizer();
        this->quantizer->Fit(data_to_cluster, this->n_samples, this->d);
        this->code_size = this->quantizer->CodeSize(this->d);
        this->state.code_size = this->code_size;
        this->state.n_encoded = this->n_samples;

        // Non-PDX quantizers must override vertical_d to full d
        if (this->quantizer->SupportsPruning() && !this->quantizer->NeedsPDXLayout()) {
            this->vertical_d = this->d;
            this->partial_horizontal_centroids = AllocateBuffer<centroid_value_t>(
                this->GetAllocator(), this->n_clusters * this->vertical_d
            );
        }
        this->partial_d = this->quantizer->InitialPartialD(this->vertical_d);
        auto initial_partial_d = this->partial_d;

        if (this->hierarchical_config.verbose) {
            std::cout << "Front dimensions (d') = " << this->partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << this->d - this->vertical_d << std::endl;
        }

        // Data encoding
        const vector_value_t* encoded_data_p;
        if constexpr (q == Quantization::f32) {
            encoded_data_p = data_to_cluster;
        } else {
            this->quantized_data = AllocateBuffer<vector_value_t>(
                this->GetAllocator(), this->n_samples * this->code_size
            );
            this->quantizer->Encode(
                data_to_cluster, this->quantized_data.get(), this->n_samples, this->d
            );
            encoded_data_p = this->quantized_data.get();
        }
        this->quantized_centroids = AllocateBuffer<vector_value_t>(
            this->GetAllocator(), this->n_clusters * this->code_size
        );

        // Encode initial centroids
        this->quantizer->Encode(
            this->prev_centroids.get(), this->quantized_centroids.get(), n_mesoclusters, this->d
        );

        // Setup quantized PDX layout for pruning (f32 PDX is set up in GenerateCentroids)
        if constexpr (q != Quantization::f32) {
            if (this->quantizer->SupportsPruning() && this->quantizer->NeedsPDXLayout()) {
                this->pdxified_quantized_centroids = AllocateBuffer<vector_value_t>(
                    this->GetAllocator(), this->n_clusters * this->PDXDim(this->d)
                );
                this->partial_hor_quantized_centroids = AllocateBuffer<vector_value_t>(
                    this->GetAllocator(), this->n_clusters * this->PDXDim(this->vertical_d)
                );
                PDXLayout<q>::template PDXify<false>(
                    this->quantized_centroids.get(),
                    this->pdxified_quantized_centroids.get(),
                    n_mesoclusters,
                    this->PDXDim(this->d)
                );
                this->QuantizedCentroidsToAuxiliaryHorizontal(n_mesoclusters);
                centroids_pdx_wrapper = this->MakeCentroidLayout(n_mesoclusters);
            }
        }

        this->quantizer->ComputeNorms(
            encoded_data_p, this->n_samples, this->d, this->data_norms.get()
        );
        this->quantizer->ComputeNorms(
            this->quantized_centroids.get(), n_mesoclusters, this->d, this->centroid_norms.get()
        );

        auto not_pruned_counts = AllocateBuffer<size_t>(this->GetAllocator(), this->n_samples);

        bool always_gemm_only = !this->PruningEnabled(n_mesoclusters);
        bool partial_norms_computed = false;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;

        for (size_t iter_idx = 0; iter_idx < this->hierarchical_config.iters_mesoclustering;
             ++iter_idx) {
            bool use_gemm_only = (iter_idx == 0) || always_gemm_only;
            if (!use_gemm_only && !partial_norms_computed) {
                this->quantizer->CacheDataPartialNorms(
                    encoded_data_p, this->n_samples, this->d, this->partial_d
                );
                partial_norms_computed = true;
            }
            if (use_gemm_only) {
                this->template RunIteration<true>(
                    data_to_cluster,
                    encoded_data_p,
                    centroids_pdx_wrapper,
                    not_pruned_counts.get(),
                    nullptr, // queries
                    0,       // n_queries
                    this->n_samples,
                    n_mesoclusters,
                    iter_idx,
                    iter_idx == 0,
                    this->hierarchical_iteration_stats.mesoclustering_iteration_stats
                );
            } else {
                this->template RunIteration<false>(
                    data_to_cluster,
                    encoded_data_p,
                    centroids_pdx_wrapper,
                    not_pruned_counts.get(),
                    nullptr, // queries
                    0,       // n_queries
                    this->n_samples,
                    n_mesoclusters,
                    iter_idx,
                    false,
                    this->hierarchical_iteration_stats.mesoclustering_iteration_stats
                );
            }
            if (this->hierarchical_config.early_termination &&
                this->ShouldStopEarly(false, best_recall, iters_without_improvement, iter_idx)) {
                break;
            }
        }
        timer_mesoclustering.Toc();

        {
            SKM_PROFILE_SCOPE("allocator");
            mesoclusters_assignments.assign(
                this->assignments.get(), this->assignments.get() + this->n_samples
            );
            if (this->n_split == 0) {
                mesoclusters_sizes.assign(
                    this->cluster_sizes.get(), this->cluster_sizes.get() + n_mesoclusters
                );
            } else {
                mesoclusters_sizes.assign(n_mesoclusters, 0);
                for (const uint32_t label : mesoclusters_assignments) {
                    mesoclusters_sizes[label]++;
                }
            }
        }

        auto mesocluster_indices_flat =
            AllocateBuffer<size_t>(this->GetAllocator(), this->n_samples);
        const auto mesocluster_offsets = GroupPointsByMesocluster(
            mesoclusters_assignments.data(),
            this->n_samples,
            mesoclusters_sizes,
            mesocluster_indices_flat.get()
        );

        //
        // FINE-CLUSTERING
        // Each mesocluster is re-clustered sequentially
        // Potential improvement: Doing 2 only-GEMM iterations and delegate the rest to GEMM+PRUNING
        // seems an interesting idea. However, we need to evaluate this with a larger dataset (+100M
        // vectors)
        //
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 2: FINE-CLUSTERING (subdividing " << n_mesoclusters
                      << " mesoclusters into total " << this->n_clusters
                      << " clusters) ===" << std::endl;
        }
        // Calculate proportional allocation of fine clusters per mesocluster
        auto fine_clusters_nums = ArrangeFineClusters(
            this->n_clusters, n_mesoclusters, this->n_samples, mesoclusters_sizes
        );

        TicToc timer_fineclustering;
        timer_fineclustering.Tic();

        std::vector<size_t> fineclusters_offsets(n_mesoclusters, 0);
        for (size_t k = 1; k < n_mesoclusters; ++k) {
            fineclusters_offsets[k] = fineclusters_offsets[k - 1] + fine_clusters_nums[k - 1];
        }
        std::vector<std::vector<SuperKMeansIterationStats>> mesocluster_fineclustering_stats(
            n_mesoclusters
        );
        std::fill_n(this->cluster_sizes.get(), this->n_clusters, 0u);
        FineclusteringContext<q> context{
            data_to_cluster,
            encoded_data_p,
            this->data_norms.get(),
            mesocluster_indices_flat.get(),
            mesocluster_offsets.data(),
            mesoclusters_sizes.data(),
            fine_clusters_nums.data(),
            fineclusters_offsets.data(),
            initial_partial_d,
            final_assignments.get(),
            this->distances.get(),
            this->cluster_sizes.get(),
            final_centroids.get(),
            mesocluster_fineclustering_stats.data()
        };
        const auto mesocluster_order =
            OrderMesoclustersBySize(mesoclusters_sizes, fine_clusters_nums);
        SerialExecutor worker_executor;
        const size_t max_fineclusters =
            *std::max_element(fine_clusters_nums.begin(), fine_clusters_nums.end());
        std::vector<std::unique_ptr<HierarchicalSuperKMeans>> workers(
            std::min(this->GetExecutor().NumWorkers(), mesocluster_order.size())
        );
        for (auto& worker : workers) {
            worker = MakeWorkerEngine(max_fineclusters, worker_executor);
        }
        std::atomic<size_t> next_mesocluster{0};
        this->GetExecutor().ParallelFor(workers.size(), [&](size_t begin, size_t end, size_t) {
            for (size_t w = begin; w < end; ++w) {
                for (size_t i = next_mesocluster++; i < mesocluster_order.size();
                     i = next_mesocluster++) {
                    workers[w]->ClusterMesocluster(context, mesocluster_order[i]);
                }
            }
        });
        for (const auto& stats : mesocluster_fineclustering_stats) {
            hierarchical_iteration_stats.fineclustering_iteration_stats.insert(
                hierarchical_iteration_stats.fineclustering_iteration_stats.end(),
                stats.begin(),
                stats.end()
            );
        }
        timer_fineclustering.Toc();

        // Now we move to the last refinement phase in which we perform clustering with all
        // n_clusters. Recall our initial buffers for centroids have enough space for n_clusters.
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 3: REFINEMENT (fine-tuning all " << this->n_clusters
                      << " clusters) ===" << std::endl;
        }
        // In the refinement phase, we use an even smaller partial d (around 8% of d) because the
        // clusters are already well-formed, and pruning rate is expected to be high.
        this->partial_d = this->quantizer->AlignPartialD(
            std::max<uint32_t>(MIN_PARTIAL_D, this->vertical_d / 3), this->vertical_d
        );

        // We just transfer the state of centroids to the proper class variables, no rotation.
        auto final_refinement_pdx_wrapper = SetupCentroids(final_centroids.get(), this->n_clusters);

        // (RunIteration with is_first_iter=false will swap horizontal_centroids and
        // prev_centroids) Copy final_centroids to prev_centroids so the swap in RunIteration
        // works correctly We could avoid this copies by managing an offset on assignments and
        // centroids in the core functions of SuperKMeans. But this would just complicate the code.
        {
            SKM_PROFILE_SCOPE("copy_final_centroids_and_assignments");
            memcpy(
                static_cast<void*>(this->prev_centroids.get()),
                static_cast<void*>(final_centroids.get()),
                sizeof(centroid_value_t) * this->n_clusters * this->d
            );
            memcpy(
                static_cast<void*>(this->assignments.get()),
                static_cast<void*>(final_assignments.get()),
                sizeof(uint32_t) * this->n_samples
            );
        }
        this->quantizer->Encode(
            this->prev_centroids.get(), this->quantized_centroids.get(), this->n_clusters, this->d
        );
        this->quantizer->ComputeNorms(
            this->quantized_centroids.get(), this->n_clusters, this->d, this->centroid_norms.get()
        );

        TicToc timer_refinement;
        timer_refinement.Tic();

        // Refinement iterations
        bool refinement_always_gemm_only = !this->PruningEnabled(this->n_clusters);
        bool refinement_partial_norms_computed = false;
        for (size_t refinement_iter_idx = 0;
             refinement_iter_idx < this->hierarchical_config.iters_refinement;
             ++refinement_iter_idx) {
            if (!refinement_always_gemm_only && !refinement_partial_norms_computed) {
                this->quantizer->CacheDataPartialNorms(
                    encoded_data_p, this->n_samples, this->d, this->partial_d
                );
                refinement_partial_norms_computed = true;
            }
            if (refinement_always_gemm_only) {
                this->template RunIteration<true>(
                    data_to_cluster,
                    encoded_data_p,
                    final_refinement_pdx_wrapper,
                    not_pruned_counts.get(),
                    nullptr, // queries
                    0,       // n_queries
                    this->n_samples,
                    this->n_clusters,
                    refinement_iter_idx,
                    false,
                    this->hierarchical_iteration_stats.refinement_iteration_stats
                );
            } else {
                this->template RunIteration<false>(
                    data_to_cluster,
                    encoded_data_p,
                    final_refinement_pdx_wrapper,
                    not_pruned_counts.get(),
                    nullptr, // queries
                    0,       // n_queries
                    this->n_samples,
                    this->n_clusters,
                    refinement_iter_idx,
                    false,
                    this->hierarchical_iteration_stats.refinement_iteration_stats
                );
            }
        }

        // When quantized centroid update was used but we want full-precision final centroids,
        // recompute by averaging the raw (rotated) float training data using final assignments.
        if constexpr (q != Quantization::f32) {
            if (this->hierarchical_config.quantized_centroid_update &&
                this->hierarchical_config.full_precision_final_centroids) {
                this->ResetCentroids(this->n_clusters);
                this->f32_quantizer.UpdateCentroids(
                    data_to_cluster,
                    this->assignments.get(),
                    this->horizontal_centroids.get(),
                    this->cluster_sizes.get(),
                    this->n_samples,
                    this->n_clusters,
                    this->d
                );
                this->f32_quantizer.FinalizeCentroids(
                    this->horizontal_centroids.get(),
                    this->cluster_sizes.get(),
                    this->n_clusters,
                    this->d
                );
                if (this->hierarchical_config.angular) {
                    this->PostprocessCentroids(this->n_clusters);
                }
            }
        }

        timer_refinement.Toc();

        if (this->hierarchical_config.verbose) {
            std::cout << "Mesoclustering time: " << timer_mesoclustering.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Fineclustering time: " << timer_fineclustering.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Refinement time: " << timer_refinement.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Total time: "
                      << timer_mesoclustering.GetMilliseconds() +
                             timer_fineclustering.GetMilliseconds() +
                             timer_refinement.GetMilliseconds()
                      << " ms" << std::endl;
        }
        this->state.trained = true;

        auto output_centroids =
            this->GetOutputCentroids(this->hierarchical_config.unrotate_centroids);
        if (this->hierarchical_config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /**
     * @brief Run hierarchical k-means clustering to determine centroids.
     *
     * Transformations (rotation and/or quantization) are done in place using the caller's buffer.
     * `data` is overwritten with its rotated form and is NOT restored, so on return the caller
     * holds rotated vectors. The returned centroids are rotated too: unrotate_centroids is forced
     * to false, so that data and centroids stay in the same domain and remain directly comparable
     * (Assign(data, centroids) is valid). AssignTrainingPoints()/QuantizedAssign() take the
     * rotated buffer as usual. Call Unrotate() to map back to the original domain.
     *
     * Requires sampling_fraction == 1.0; a smaller value is overridden (with a warning) since
     * sampling gathers scattered rows and cannot be done in place.
     *
     * @param data Pointer to the data matrix (row-major, n × d). Overwritten with rotated data.
     * @param n Number of points (rows) in the data matrix
     *
     * @return std::vector<skmeans_centroid_value_t> Trained centroids
     */
    std::vector<skmeans_centroid_value_t> TrainInPlace(
        float* SKM_RESTRICT data,
        const size_t n,
        const float* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        this->ConfigInPlaceTraining(data, n);
        static_cast<SuperKMeansConfig&>(hierarchical_config) = this->config;
        return Train(data, n, queries, n_queries);
    }

    /**
     * @brief Computes the number of vectors to sample based on sampling_fraction.
     *
     * @param n Total number of vectors
     * @return Number of vectors to sample
     */
    [[nodiscard]] size_t GetNVectorsToSample(const size_t n, size_t n_clusters) const override {
        if (this->hierarchical_config.sampling_fraction == 1.0) {
            return n;
        }
        auto samples_by_n = static_cast<size_t>(
            std::floor(static_cast<double>(n) * this->hierarchical_config.sampling_fraction)
        );
        return samples_by_n;
    }

    /**
     * @brief Override SplitClusters with more aggressive balancing similar to cuVS
     *
     * This version not only handles empty clusters but also actively rebalances
     * small clusters (those below a threshold) by moving their centers toward
     * points from larger clusters.
     *
     * @param n_samples Total number of samples
     * @param n_clusters Number of clusters
     */
    void SplitClusters(const size_t n_samples, const size_t n_clusters) override {
        constexpr float CENTER_ADJUSTMENT_WEIGHT =
            7.0f; // Weight for current center in weighted average
        constexpr float BALANCING_THRESHOLD =
            0.25f; // Clusters smaller than 25% of average are adjusted

        this->n_split = 0;
        std::mt19937 rng(this->config.seed);
        auto horizontal_centroids_p = this->horizontal_centroids.get();

        size_t average_size = n_samples / n_clusters;
        size_t threshold_size =
            static_cast<size_t>(static_cast<float>(average_size) * BALANCING_THRESHOLD);
        {
            SKM_PROFILE_SCOPE("consolidate/empty");
            for (size_t ci = 0; ci < n_clusters; ci++) {
                if (this->cluster_sizes[ci] == 0) {
                    size_t cj;
                    for (cj = 0; true; cj = (cj + 1) % n_clusters) {
                        float p = (this->cluster_sizes[cj] - 1.0f) /
                                  static_cast<float>(n_samples - n_clusters);
                        float r = std::uniform_real_distribution<float>(0, 1)(rng);
                        if (r < p) {
                            break;
                        }
                    }

                    memcpy(
                        (void*) (horizontal_centroids_p + ci * this->d),
                        (void*) (horizontal_centroids_p + cj * this->d),
                        sizeof(centroid_value_t) * this->d
                    );

                    // Small symmetric perturbation
                    for (size_t j = 0; j < this->d; j++) {
                        if (j % 2 == 0) {
                            horizontal_centroids_p[ci * this->d + j] *=
                                1.0f + CENTROID_PERTURBATION_EPS;
                            horizontal_centroids_p[cj * this->d + j] *=
                                1.0f - CENTROID_PERTURBATION_EPS;
                        } else {
                            horizontal_centroids_p[ci * this->d + j] *=
                                1.0f - CENTROID_PERTURBATION_EPS;
                            horizontal_centroids_p[cj * this->d + j] *=
                                1.0f + CENTROID_PERTURBATION_EPS;
                        }
                    }

                    // Assume even split of the cluster
                    this->cluster_sizes[ci] = this->cluster_sizes[cj] / 2;
                    this->cluster_sizes[cj] -= this->cluster_sizes[ci];
                    this->n_split++;
                }
            }
        }

        // Adjust small clusters (cuVS-style balancing)
        // Pick large clusters with probability proportional to their size
        {
            SKM_PROFILE_SCOPE("consolidate/balancing");
            for (size_t ci = 0; ci < n_clusters; ci++) {
                size_t csize = this->cluster_sizes[ci];
                if (csize == 0 || csize > threshold_size)
                    continue;

                // Find a large cluster with probability proportional to its size
                size_t large_cluster_idx;
                for (large_cluster_idx = 0; true;
                     large_cluster_idx = (large_cluster_idx + 1) % n_clusters) {
                    size_t large_size = this->cluster_sizes[large_cluster_idx];
                    if (large_size < average_size)
                        continue;
                    // Probability proportional to how much larger this cluster is than average
                    float p =
                        static_cast<float>(large_size - average_size + 1) /
                        static_cast<float>(n_samples - average_size * n_clusters + n_clusters);
                    float r = std::uniform_real_distribution<float>(0, 1)(rng);
                    if (r < p) {
                        break; // Found our cluster to be split
                    }
                }

                // Adjust the center of the selected smaller cluster to gravitate towards
                // a sample from the selected larger cluster.
                // Weight of the current center for the weighted average.
                // We dump it for anomalously small clusters, but keep constant otherwise.
                float wc = std::min(static_cast<float>(csize), CENTER_ADJUSTMENT_WEIGHT);
                float wd = 1.0f; // Weight for the datapoint used to shift the center.
                for (size_t j = 0; j < this->d; j++) {
                    float val = 0.0f;
                    val += wc * horizontal_centroids_p[ci * this->d + j];
                    val += wd * horizontal_centroids_p[large_cluster_idx * this->d + j];
                    val /= (wc + wd);
                    horizontal_centroids_p[ci * this->d + j] = val;
                }

                this->n_split++;
            }
        }
    }

    std::unique_ptr<HierarchicalSuperKMeans> MakeWorkerEngine(
        const size_t n_worker_clusters,
        ParallelExecutor& worker_executor
    ) const {
        HierarchicalSuperKMeansConfig worker_config = hierarchical_config;
        worker_config.executor = &worker_executor;
        worker_config.data_already_rotated = true;
        worker_config.verbose = false;
        worker_config.suppress_warnings = true;
        auto worker =
            std::make_unique<HierarchicalSuperKMeans>(n_worker_clusters, this->d, worker_config);
        worker->ConfigWorkerEngine(*this->quantizer);
        worker->ConfigFineclusteringChunks();
        return worker;
    }

    void ConfigFineclusteringChunks() {
        n_chunk_points = std::max(
            MINI_BATCH_SIZE, FINECLUSTERING_CHUNK_BYTES / (this->code_size * sizeof(vector_value_t))
        );
        tmp_chunk_codes =
            AllocateBuffer<vector_value_t>(this->GetAllocator(), n_chunk_points * this->code_size);
    }

    void ClusterMesocluster(const FineclusteringContext<q>& context, const size_t k) {
        const size_t n_fineclusters = context.fine_clusters_nums[k];
        const size_t mesocluster_size = context.mesoclusters_sizes[k];
        const size_t* mesocluster_indices =
            context.mesocluster_indices + context.mesocluster_offsets[k];
        auto& allocator = this->GetAllocator();
        EnsureBufferSize(this->data_norms, allocator, mesocluster_size);
        EnsureBufferSize(this->assignments, allocator, mesocluster_size);
        EnsureBufferSize(this->distances, allocator, mesocluster_size);
        EnsureBufferSize(tmp_not_pruned_counts, allocator, mesocluster_size);

        this->partial_d = context.initial_partial_d;
        this->quantizer->InvalidateCaches();
        for (size_t j = 0; j < mesocluster_size; ++j) {
            this->data_norms[j] = context.data_norms[mesocluster_indices[j]];
        }

        auto mesocluster_centroids_pdx_wrapper = this->GenerateCentroids(
            context.data_to_cluster, mesocluster_size, n_fineclusters, false, mesocluster_indices
        );
        // Copy centroids to prev_centroids for use in the first RunFineclusteringIteration
        // (is_first_iter=true skips the swap, so prev_centroids must be populated)
        memcpy(
            this->prev_centroids.get(),
            this->horizontal_centroids.get(),
            sizeof(centroid_value_t) * n_fineclusters * this->d
        );

        // Encode fine centroids + setup quantized PDX for pruning
        this->quantizer->Encode(
            this->prev_centroids.get(), this->quantized_centroids.get(), n_fineclusters, this->d
        );
        this->quantizer->ComputeNorms(
            this->quantized_centroids.get(), n_fineclusters, this->d, this->centroid_norms.get()
        );
        if constexpr (q != Quantization::f32) {
            if (this->quantizer->SupportsPruning() && this->quantizer->NeedsPDXLayout()) {
                PDXLayout<q>::template PDXify<false>(
                    this->quantized_centroids.get(),
                    this->pdxified_quantized_centroids.get(),
                    n_fineclusters,
                    this->PDXDim(this->d)
                );
                this->QuantizedCentroidsToAuxiliaryHorizontal(n_fineclusters);
                mesocluster_centroids_pdx_wrapper = this->MakeCentroidLayout(n_fineclusters);
            }
        }

        bool fine_always_gemm_only = !this->PruningEnabled(n_fineclusters);
        float fine_best_recall = 0.0f;
        size_t iters_without_improvement = 0;
        for (size_t fine_iter_idx = 0; fine_iter_idx < hierarchical_config.iters_fineclustering;
             ++fine_iter_idx) {
            bool use_gemm_only = (fine_iter_idx == 0) || fine_always_gemm_only;
            if (use_gemm_only) {
                RunFineclusteringIteration<false>(
                    context, k, mesocluster_centroids_pdx_wrapper, fine_iter_idx, fine_iter_idx == 0
                );
            } else {
                RunFineclusteringIteration<true>(
                    context, k, mesocluster_centroids_pdx_wrapper, fine_iter_idx, false
                );
            }
            if (hierarchical_config.early_termination &&
                this->ShouldStopEarly(
                    false, fine_best_recall, iters_without_improvement, fine_iter_idx
                )) {
                break;
            }
        }

        GetTrueAssignmentsFromIndirectionBuffer(
            mesocluster_indices,
            context.final_assignments,
            context.final_distances,
            context.final_cluster_sizes,
            this->assignments.get(),
            this->distances.get(),
            mesocluster_size,
            context.fineclusters_offsets[k]
        );

        // We move the resulting centroids from this fineclustering to the final buffer of
        // centroids
        {
            SKM_PROFILE_SCOPE("copy_fine_centroids");
            memcpy(
                static_cast<void*>(
                    context.final_centroids + context.fineclusters_offsets[k] * this->d
                ),
                static_cast<void*>(this->horizontal_centroids.get()),
                sizeof(centroid_value_t) * n_fineclusters * this->d
            );
        }
    }

    template <bool PRUNE>
    void RunFineclusteringIteration(
        const FineclusteringContext<q>& context,
        const size_t k,
        layout_t& centroids_layout,
        const size_t iter_idx,
        const bool is_first_iter
    ) {
        const size_t n_points = context.mesoclusters_sizes[k];
        const size_t n_clusters = context.fine_clusters_nums[k];
        const size_t* mesocluster_indices =
            context.mesocluster_indices + context.mesocluster_offsets[k];
        if (!is_first_iter) {
            std::swap(this->horizontal_centroids, this->prev_centroids);
        }
        this->quantizer->Encode(
            this->prev_centroids.get(), this->quantized_centroids.get(), n_clusters, this->d
        );
        if constexpr (PRUNE) {
            std::fill_n(tmp_not_pruned_counts.get(), n_points, size_t{0});
            this->quantizer->CacheCentroidPartialNorms(
                this->quantized_centroids.get(), n_clusters, this->d, this->partial_d
            );
        } else {
            this->quantizer->ComputeNorms(
                this->quantized_centroids.get(), n_clusters, this->d, this->centroid_norms.get()
            );
        }
        this->ResetCentroids(n_clusters);
        this->quantizer->ResetCentroidAccumulators(n_clusters, this->d);
        AssignAndAccumulateChunks(
            PRUNE, context, mesocluster_indices, n_points, n_clusters, centroids_layout
        );

        SuperKMeansIterationStats stats;
        stats.iteration = iter_idx + 1;
        stats.pruning = PRUNE;
        if constexpr (PRUNE) {
            bool partial_d_changed = false;
            stats.partial_d = this->partial_d;
            stats.not_pruned_pct = this->TunePartialD(
                tmp_not_pruned_counts.get(), n_points, n_clusters, partial_d_changed
            );
        }
        this->ConsolidateCentroids(n_points, n_clusters);
        this->ComputeCost(n_points);
        this->ComputeShift(n_clusters);
        stats.objective = this->cost;
        stats.shift = this->shift;
        stats.split = this->n_split;
        stats.recall = this->recall;
        context.fineclustering_stats[k].push_back(stats);
    }

    void AssignAndAccumulateChunks(
        const bool prune,
        const FineclusteringContext<q>& context,
        const size_t* SKM_RESTRICT mesocluster_indices,
        const size_t n_points,
        const size_t n_clusters,
        layout_t& centroids_layout
    ) {
        const bool quantized_update = this->config.quantized_centroid_update;
        const bool gather_points =
            q != Quantization::f32 && (!quantized_update || (q == Quantization::rabitq && prune));
        const float* chunk_points;
        if constexpr (q == Quantization::f32) {
            chunk_points = tmp_chunk_codes.get();
        } else {
            if (gather_points) {
                EnsureBufferSize(tmp_chunk_points, this->GetAllocator(), n_chunk_points * this->d);
            }
            chunk_points = tmp_chunk_points.get();
        }
        for (size_t i = 0; i < n_points; i += n_chunk_points) {
            const size_t n_chunk = std::min(n_chunk_points, n_points - i);
            GatherMesoclusterChunk(context, mesocluster_indices + i, n_chunk, gather_points);
            if (n_points > n_chunk_points) {
                this->quantizer->InvalidateCaches();
            }
            if (prune) {
                this->quantizer->CacheDataPartialNorms(
                    tmp_chunk_codes.get(), n_chunk, this->d, this->partial_d
                );
            }
            this->AssignPoints(
                prune,
                tmp_chunk_codes.get(),
                chunk_points,
                this->data_norms.get() + i,
                this->quantized_centroids.get(),
                this->prev_centroids.get(),
                n_chunk,
                n_clusters,
                centroids_layout,
                this->assignments.get() + i,
                this->distances.get() + i,
                prune ? tmp_not_pruned_counts.get() + i : nullptr
            );
            this->AccumulatePoints(
                quantized_update,
                tmp_chunk_codes.get(),
                chunk_points,
                this->assignments.get() + i,
                n_chunk,
                n_clusters,
                this->horizontal_centroids.get(),
                this->cluster_sizes.get()
            );
        }
    }

    void GatherMesoclusterChunk(
        const FineclusteringContext<q>& context,
        const size_t* SKM_RESTRICT chunk_indices,
        const size_t n_chunk,
        const bool gather_points
    ) {
        SKM_PROFILE_SCOPE("gather_mesocluster_chunk");
        const size_t code_size = this->code_size;
        const size_t d = this->d;
        this->GetExecutor().ParallelFor(n_chunk, [&](size_t begin, size_t end, size_t) {
            for (size_t j = begin; j < end; ++j) {
                const size_t i = chunk_indices[j];
                memcpy(
                    tmp_chunk_codes.get() + j * code_size,
                    context.encoded_data + i * code_size,
                    sizeof(vector_value_t) * code_size
                );
                if (gather_points) {
                    memcpy(
                        tmp_chunk_points.get() + j * d,
                        context.data_to_cluster + i * d,
                        sizeof(float) * d
                    );
                }
            }
        });
    }

    void GetTrueAssignmentsFromIndirectionBuffer(
        const size_t* SKM_RESTRICT mesocluster_indices,
        uint32_t* SKM_RESTRICT output_assignments,
        distance_t* SKM_RESTRICT output_distances,
        uint32_t* SKM_RESTRICT output_cluster_sizes,
        const uint32_t* SKM_RESTRICT input_assignments,
        const distance_t* SKM_RESTRICT input_distances,
        const size_t n_samples_in_mesocluster,
        const size_t cluster_id_offset
    ) {
        SKM_PROFILE_SCOPE("get_indirection_assignments");
        for (size_t i = 0; i < n_samples_in_mesocluster; ++i) {
            size_t original_idx = mesocluster_indices[i];
            uint32_t local_cluster_id = input_assignments[i];
            // Assignments in fineclustering are from 0 to n_fineclusters-1
            // We need to add the offset to put the global cluster id in the final assignments
            uint32_t global_cluster_id = local_cluster_id + cluster_id_offset;
            output_assignments[original_idx] = global_cluster_id;
            output_distances[original_idx] = input_distances[i];
            output_cluster_sizes[global_cluster_id]++;
        }
    }

    /**
     * @brief Setup centroids to be used for the refinement clustering phase
     *
     * @param centroids Centroids to setup
     * @param n_clusters Number of centroids to setupt
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q> SetupCentroids(
        const centroid_value_t* SKM_RESTRICT centroids,
        const size_t n_clusters
    ) {
        SKM_PROFILE_SCOPE("consolidate");
        memcpy(
            (void*) (this->horizontal_centroids.get()),
            (void*) (centroids),
            sizeof(centroid_value_t) * n_clusters * this->d
        );
        if constexpr (q == Quantization::f32) {
            {
                SKM_PROFILE_SCOPE("consolidate/pdxify");
                PDXLayout<q>::template PDXify<false>(
                    this->horizontal_centroids.get(), this->centroids.get(), n_clusters, this->d
                );
            }
            //! We wrap centroids and partial_horizontal_centroids in the PDXLayout wrapper
            //! Any updates to these objects is reflected in the PDXLayout
            auto pdx_centroids = this->MakeCentroidLayout(n_clusters);
            this->CentroidsToAuxiliaryHorizontal(n_clusters);
            return pdx_centroids;
        } else {
            // Encode float centroids to quantized form
            this->quantizer->Encode(
                this->horizontal_centroids.get(),
                this->quantized_centroids.get(),
                n_clusters,
                this->d
            );
            if (this->quantizer->SupportsPruning() && this->quantizer->NeedsPDXLayout()) {
                SKM_PROFILE_SCOPE("consolidate/pdxify");
                PDXLayout<q>::template PDXify<false>(
                    this->quantized_centroids.get(),
                    this->pdxified_quantized_centroids.get(),
                    n_clusters,
                    this->PDXDim(this->d)
                );
                this->QuantizedCentroidsToAuxiliaryHorizontal(n_clusters);
                return this->MakeCentroidLayout(n_clusters);
            }
            // Non-PDX quantizers (e.g., RaBitQ) — return empty layout
            return layout_t{};
        }
    }

    size_t n_mesoclusters = 0;

    std::vector<uint32_t> mesoclusters_assignments;
    std::vector<uint32_t> mesoclusters_sizes;
    AllocatedBuffer<uint32_t> final_assignments;
    AllocatedBuffer<centroid_value_t> final_centroids;
    size_t n_chunk_points = 0;
    AllocatedBuffer<vector_value_t> tmp_chunk_codes;
    AllocatedBuffer<float> tmp_chunk_points;
    AllocatedBuffer<size_t> tmp_not_pruned_counts;
    HierarchicalSuperKMeansConfig hierarchical_config;
    HierarchicalSuperKMeansIterationStats hierarchical_iteration_stats;
};

using HierarchicalSuperKMeansSQ8 = HierarchicalSuperKMeans<Quantization::sq8>;
using HierarchicalSuperKMeansLVQ4 = HierarchicalSuperKMeans<Quantization::lvq4>;
using HierarchicalSuperKMeansRabitQ = HierarchicalSuperKMeans<Quantization::rabitq>;

} // namespace skmeans
