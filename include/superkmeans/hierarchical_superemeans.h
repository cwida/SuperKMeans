#pragma once

#include <atomic>

#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superemeans.h"

namespace skmeans {

struct HierarchicalSuperEMeansConfig : SuperEMeansConfig {
    uint32_t epochs_mesoclustering = 2;
    uint32_t epochs_fineclustering = 2;
};

template <Quantization q = Quantization::f32>
class HierarchicalSuperEMeans : public SuperEMeans<q> {
    using typename SuperKMeans<q>::centroid_value_t;
    using typename SuperKMeans<q>::vector_value_t;
    using typename SuperKMeans<q>::distance_t;
    using typename SuperKMeans<q>::pruner_t;
    using typename SuperKMeans<q>::layout_t;
    using typename SuperKMeans<q>::TrainingData;

  public:
    HierarchicalSuperEMeans(
        size_t n_clusters,
        size_t dimensionality,
        const HierarchicalSuperEMeansConfig& config
    )
        : SuperEMeans<q>(n_clusters, dimensionality, config), hierarchical_config(config) {
        this->pruner = std::make_unique<pruner_t>(
            dimensionality,
            HIERARCHICAL_PRUNER_INITIAL_THRESHOLD,
            this->config.seed,
            this->config.data_already_rotated,
            &this->GetAllocator()
        );
        this->pruner->SetExecutor(&this->GetExecutor());
        SKMEANS_ENSURE_POSITIVE(config.epochs_mesoclustering);
        SKMEANS_ENSURE_POSITIVE(config.epochs_fineclustering);
        static_cast<SuperKMeansConfig&>(hierarchical_config) = this->config;
    }

    HierarchicalSuperEMeans(size_t n_clusters, size_t dimensionality)
        : HierarchicalSuperEMeans(n_clusters, dimensionality, HierarchicalSuperEMeansConfig{}) {}

    std::vector<skmeans_centroid_value_t> Train(const float* SKM_RESTRICT data, const size_t n) {
        this->CheckTrainable(n);
        hierarchical_iteration_stats.mesoclustering_iteration_stats.clear();
        hierarchical_iteration_stats.fineclustering_iteration_stats.clear();
        ParallelSection parallel_section(this->GetExecutor());
        auto training_data = this->SetupTraining(data, n);
        const uint32_t initial_partial_d = this->partial_d;
        const size_t n_mesoclusters = GetNMesoclusters(this->n_clusters);

        TicToc timer_mesoclustering;
        timer_mesoclustering.Tic();
        training_data.centroids_pdx = this->MakeCentroidLayout(n_mesoclusters);
        this->problem = {
            nullptr,
            this->n_samples,
            n_mesoclusters,
            hierarchical_config.epochs_mesoclustering,
            this->assignments.get(),
            this->distances.get(),
            this->data_norms.get(),
            &hierarchical_iteration_stats.mesoclustering_iteration_stats
        };
        this->RunEMeans(training_data);
        timer_mesoclustering.Toc();

        std::vector<uint32_t> mesoclusters_sizes(n_mesoclusters, 0);
        for (size_t i = 0; i < this->n_samples; ++i) {
            mesoclusters_sizes[this->assignments[i]]++;
        }
        auto mesocluster_indices_flat =
            AllocateBuffer<size_t>(this->GetAllocator(), this->n_samples);
        const auto mesocluster_offsets = GroupPointsByMesocluster(
            this->assignments.get(),
            this->n_samples,
            mesoclusters_sizes,
            mesocluster_indices_flat.get()
        );
        const auto fine_clusters_nums = ArrangeFineClusters(
            this->n_clusters, n_mesoclusters, this->n_samples, mesoclusters_sizes
        );
        std::vector<size_t> fineclusters_offsets(n_mesoclusters, 0);
        for (size_t k = 1; k < n_mesoclusters; ++k) {
            fineclusters_offsets[k] = fineclusters_offsets[k - 1] + fine_clusters_nums[k - 1];
        }

        TicToc timer_fineclustering;
        timer_fineclustering.Tic();
        std::vector<std::vector<SuperKMeansIterationStats>> mesocluster_fineclustering_stats(
            n_mesoclusters
        );
        std::fill_n(this->cluster_sizes.get(), this->n_clusters, 0u);
        FineclusteringContext<q> context{
            training_data.points,
            training_data.codes,
            this->data_norms.get(),
            mesocluster_indices_flat.get(),
            mesocluster_offsets.data(),
            mesoclusters_sizes.data(),
            fine_clusters_nums.data(),
            fineclusters_offsets.data(),
            initial_partial_d,
            this->assignments.get(),
            this->distances.get(),
            this->cluster_sizes.get(),
            this->horizontal_centroids.get(),
            mesocluster_fineclustering_stats.data()
        };
        const auto mesocluster_order =
            OrderMesoclustersBySize(mesoclusters_sizes, fine_clusters_nums);
        SerialExecutor worker_executor;
        const size_t max_fineclusters =
            *std::max_element(fine_clusters_nums.begin(), fine_clusters_nums.end());
        std::vector<std::unique_ptr<HierarchicalSuperEMeans>> workers(
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

        this->PublishCentroids(this->n_clusters);
        this->partial_d = this->quantizer->AlignPartialD(
            std::max<uint32_t>(MIN_PARTIAL_D, this->vertical_d / 3), this->vertical_d
        );
        this->quantizer->InvalidateCaches();
        this->state.trained = true;
        if (this->config.verbose) {
            std::cout << "Mesoclustering time: " << timer_mesoclustering.GetMilliseconds()
                      << " ms" << std::endl;
            std::cout << "Fineclustering time: " << timer_fineclustering.GetMilliseconds()
                      << " ms" << std::endl;
        }
        return this->GetOutputCentroids(this->config.unrotate_centroids);
    }

    std::vector<skmeans_centroid_value_t> TrainInPlace(float* SKM_RESTRICT data, const size_t n) {
        this->ConfigInPlaceTraining(data, n);
        return Train(data, n);
    }

    HierarchicalSuperEMeansConfig hierarchical_config;
    HierarchicalSuperKMeansIterationStats hierarchical_iteration_stats;

  protected:
    std::unique_ptr<HierarchicalSuperEMeans> MakeWorkerEngine(
        const size_t n_worker_clusters,
        ParallelExecutor& worker_executor
    ) const {
        HierarchicalSuperEMeansConfig worker_config = hierarchical_config;
        worker_config.executor = &worker_executor;
        worker_config.data_already_rotated = true;
        worker_config.verbose = false;
        worker_config.suppress_warnings = true;
        worker_config.assignment_chunk_size = std::max(
            MINI_BATCH_SIZE, FINECLUSTERING_CHUNK_BYTES / (this->code_size * sizeof(vector_value_t))
        );
        auto worker =
            std::make_unique<HierarchicalSuperEMeans>(n_worker_clusters, this->d, worker_config);
        worker->ConfigWorkerEngine(*this->quantizer);
        return worker;
    }

    void ClusterMesocluster(const FineclusteringContext<q>& context, const size_t k) {
        const size_t n_fineclusters = context.fine_clusters_nums[k];
        const size_t mesocluster_size = context.mesoclusters_sizes[k];
        const size_t* mesocluster_indices =
            context.mesocluster_indices + context.mesocluster_offsets[k];
        const size_t fineclusters_offset = context.fineclusters_offsets[k];
        this->partial_d = context.initial_partial_d;
        this->quantizer->InvalidateCaches();
        this->GenerateCentroids(
            context.data_to_cluster, mesocluster_size, n_fineclusters, false, mesocluster_indices
        );
        memcpy(
            this->prev_centroids.get(),
            this->horizontal_centroids.get(),
            sizeof(centroid_value_t) * n_fineclusters * this->d
        );
        TrainingData training_data;
        training_data.points = context.data_to_cluster;
        training_data.codes = context.encoded_data;
        training_data.centroids_pdx = this->MakeCentroidLayout(n_fineclusters);
        this->problem = {
            mesocluster_indices,
            mesocluster_size,
            n_fineclusters,
            hierarchical_config.epochs_fineclustering,
            context.final_assignments,
            context.final_distances,
            context.data_norms,
            &context.fineclustering_stats[k]
        };
        this->RunEMeans(training_data);

        for (size_t i = 0; i < mesocluster_size; ++i) {
            uint32_t& label = context.final_assignments[mesocluster_indices[i]];
            label += static_cast<uint32_t>(fineclusters_offset);
            context.final_cluster_sizes[label]++;
        }
        memcpy(
            context.final_centroids + fineclusters_offset * this->d,
            this->horizontal_centroids.get(),
            sizeof(centroid_value_t) * n_fineclusters * this->d
        );
    }
};

using HierarchicalSuperEMeansSQ8 = HierarchicalSuperEMeans<Quantization::sq8>;
using HierarchicalSuperEMeansLVQ4 = HierarchicalSuperEMeans<Quantization::lvq4>;
using HierarchicalSuperEMeansRabitQ = HierarchicalSuperEMeans<Quantization::rabitq>;

} // namespace skmeans
