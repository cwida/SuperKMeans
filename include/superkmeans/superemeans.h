#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace skmeans {

struct SuperEMeansConfig : SuperKMeansConfig {
    size_t batch_size = 0;
    float initial_step = 0.0f;
    size_t assignment_chunk_size = EMEANS_ASSIGNMENT_CHUNK_SIZE;

    SuperEMeansConfig() {
        iters = EMEANS_DEFAULT_EPOCHS;
        sampling_fraction = 1.0f;
    }
};

inline size_t EMeansBatchSize(const size_t n, const size_t n_clusters, const size_t n_epochs) {
    const double batch_size = EMEANS_BATCH_LAW_C *
                              std::pow(static_cast<double>(n_clusters), EMEANS_BATCH_LAW_K_EXP) *
                              std::pow(static_cast<double>(n), EMEANS_BATCH_LAW_N_EXP) *
                              std::pow(static_cast<double>(n_epochs), EMEANS_BATCH_LAW_T_EXP);
    return std::min(
        {RoundToPowerOfTwo(batch_size), RoundToPowerOfTwo(static_cast<double>(n) / 2.0), n}
    );
}

inline float EMeansInitialStep(
    const size_t n,
    const size_t n_clusters,
    const size_t n_epochs,
    const size_t batch_size
) {
    const double step = EMEANS_STEP_LAW_C *
                        std::pow(static_cast<double>(n_clusters), EMEANS_STEP_LAW_K_EXP) *
                        std::pow(static_cast<double>(batch_size), EMEANS_STEP_LAW_B_EXP) *
                        std::pow(static_cast<double>(n), EMEANS_STEP_LAW_N_EXP) *
                        std::pow(static_cast<double>(n_epochs), EMEANS_STEP_LAW_T_EXP);
    return static_cast<float>(std::clamp(step, EMEANS_MIN_STEP, 1.0));
}

inline float EMeansScheduledStep(
    const float initial_step,
    const size_t step,
    const size_t n_steps
) {
    const double progress = static_cast<double>(std::min(step, n_steps)) /
                            static_cast<double>(std::max<size_t>(n_steps, 1));
    const double cosine = 0.5 * (1.0 + std::cos(EMEANS_PI * progress));
    return static_cast<float>(
        initial_step * (EMEANS_STEP_FLOOR + (1.0 - EMEANS_STEP_FLOOR) * cosine)
    );
}

template <Quantization q = Quantization::f32>
class SuperEMeans : public SuperKMeans<q> {
    using typename SuperKMeans<q>::centroid_value_t;
    using typename SuperKMeans<q>::vector_value_t;
    using typename SuperKMeans<q>::distance_t;
    using typename SuperKMeans<q>::TrainingData;

  public:
    SuperEMeans(size_t n_clusters, size_t dimensionality, const SuperEMeansConfig& config)
        : SuperKMeans<q>(n_clusters, dimensionality, config), config_batch_size(config.batch_size),
          config_initial_step(config.initial_step), chunk_size(config.assignment_chunk_size) {
        SKMEANS_ENSURE_POSITIVE(config.assignment_chunk_size);
        if (config.initial_step < 0.0f || config.initial_step > 1.0f) {
            throw std::invalid_argument("initial_step must be in [0, 1]");
        }
    }

    SuperEMeans(size_t n_clusters, size_t dimensionality)
        : SuperEMeans(n_clusters, dimensionality, SuperEMeansConfig{}) {}

    [[nodiscard]] size_t GetBatchSize() const noexcept { return batch_size; }

    [[nodiscard]] float GetInitialStep() const noexcept { return initial_step; }

    std::vector<skmeans_centroid_value_t> Train(const float* SKM_RESTRICT data, const size_t n) {
        this->CheckTrainable(n);
        this->iteration_stats.clear();
        ParallelSection parallel_section(this->GetExecutor());
        auto training_data = this->SetupTraining(data, n);
        problem = {
            nullptr,
            this->n_samples,
            this->n_clusters,
            this->config.iters,
            this->assignments.get(),
            this->distances.get(),
            this->data_norms.get(),
            &this->iteration_stats
        };
        RunEMeans(training_data);
        this->state.trained = true;

        auto output_centroids = this->GetOutputCentroids(this->config.unrotate_centroids);
        if (this->config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    std::vector<skmeans_centroid_value_t> TrainInPlace(float* SKM_RESTRICT data, const size_t n) {
        this->ConfigInPlaceTraining(data, n);
        return Train(data, n);
    }

  protected:
    struct EMeansProblem {
        const size_t* point_ids = nullptr;
        size_t n_points = 0;
        size_t n_clusters = 0;
        uint32_t n_epochs = 0;
        uint32_t* labels = nullptr;
        distance_t* distances = nullptr;
        const float* norms = nullptr;
        std::vector<SuperKMeansIterationStats>* stats = nullptr;
    };

    void RunEMeans(TrainingData& training_data) {
        const size_t n_points = problem.n_points;
        const size_t n_clusters = problem.n_clusters;
        batch_size = config_batch_size > 0
                         ? config_batch_size
                         : EMeansBatchSize(n_points, n_clusters, problem.n_epochs);
        batch_size = std::min(batch_size, n_points);
        initial_step = config_initial_step > 0.0f
                           ? config_initial_step
                           : EMeansInitialStep(n_points, n_clusters, problem.n_epochs, batch_size);
        if (this->config.verbose) {
            std::cout << "Batch size = " << batch_size << ", initial step = " << initial_step
                      << std::endl;
        }
        AllocateEMeansBuffers();

        std::swap(this->horizontal_centroids, this->prev_centroids);
        if (this->config.angular) {
            this->PostprocessCentroids(n_clusters);
            memcpy(
                positions.get(),
                this->horizontal_centroids.get(),
                sizeof(centroid_value_t) * n_clusters * this->d
            );
        }
        std::fill_n(usage.get(), n_clusters, EMEANS_INITIAL_USAGE);
        revive_rng.seed(this->config.seed);

        for (uint32_t epoch = 0; epoch < problem.n_epochs; ++epoch) {
            RunEpoch(training_data, epoch);
        }
        FinalizeTraining(training_data);
    }

    void AllocateEMeansBuffers() {
        SKM_PROFILE_SCOPE("allocator");
        auto& allocator = this->GetAllocator();
        const size_t n_clusters = problem.n_clusters;
        const size_t n_chunk = std::min(batch_size, chunk_size);
        EnsureBufferSize(usage, allocator, n_clusters);
        if (this->config.angular) {
            EnsureBufferSize(positions, allocator, n_clusters * this->d);
        }
        EnsureBufferSize(permutation, allocator, problem.n_points);
        EnsureBufferSize(tmp_chunk_codes, allocator, n_chunk * this->code_size);
        if constexpr (q != Quantization::f32) {
            if (!this->config.quantized_centroid_update ||
                (q == Quantization::rabitq && this->PruningEnabled(n_clusters))) {
                EnsureBufferSize(tmp_chunk_points, allocator, n_chunk * this->d);
            }
        }
        EnsureBufferSize(tmp_chunk_norms, allocator, n_chunk);
        EnsureBufferSize(tmp_chunk_labels, allocator, n_chunk);
        EnsureBufferSize(tmp_chunk_distances, allocator, n_chunk);
        if (this->PruningEnabled(n_clusters)) {
            EnsureBufferSize(tmp_not_pruned_counts, allocator, batch_size);
        }
        EnsureBufferSize(tmp_donor_keys, allocator, n_clusters);
        EnsureBufferSize(tmp_donors, allocator, n_clusters);
        EnsureBufferSize(tmp_children, allocator, n_clusters);
    }

    void RunEpoch(TrainingData& training_data, const uint32_t epoch) {
        const size_t n_points = problem.n_points;
        const size_t n_clusters = problem.n_clusters;
        const size_t n_batches = (n_points + batch_size - 1) / batch_size;
        const size_t n_steps = n_batches * problem.n_epochs;
        if (problem.point_ids) {
            std::copy_n(problem.point_ids, n_points, permutation.get());
        } else {
            std::iota(permutation.get(), permutation.get() + n_points, size_t{0});
        }
        std::mt19937 shuffle_rng(static_cast<uint32_t>(
            static_cast<size_t>(this->config.seed) * EMEANS_EPOCH_SEED_MULTIPLIER + epoch
        ));
        std::shuffle(permutation.get(), permutation.get() + n_points, shuffle_rng);

        const bool prune = epoch > 0 && this->PruningEnabled(n_clusters);
        const uint32_t epoch_partial_d = this->partial_d;
        this->n_split = 0;
        double objective = 0.0;
        double not_pruned_pct = 0.0;
        for (size_t i = 0; i < n_batches; ++i) {
            const size_t offset = i * batch_size;
            const size_t n_batch = std::min(batch_size, n_points - offset);
            objective += RunBatch(training_data, permutation.get() + offset, n_batch, prune);
            if (prune) {
                bool partial_d_changed = false;
                not_pruned_pct +=
                    static_cast<double>(n_batch) *
                    this->TunePartialD(
                        tmp_not_pruned_counts.get(), n_batch, n_clusters, partial_d_changed
                    );
            }
            const size_t step = epoch * n_batches + i;
            ApplyEMA(EMeansScheduledStep(initial_step, step, n_steps));
            if ((step + 1) % EMEANS_REVIVE_EVERY == 0) {
                ReviveUnused();
            }
            if (this->config.angular) {
                this->NormalizePoints(
                    positions.get(), this->horizontal_centroids.get(), n_clusters
                );
            }
        }

        SuperKMeansIterationStats stats;
        stats.iteration = epoch + 1;
        stats.objective = static_cast<float>(objective);
        stats.split = this->n_split;
        stats.pruning = prune;
        if (prune) {
            stats.not_pruned_pct =
                static_cast<float>(not_pruned_pct / static_cast<double>(n_points));
            stats.partial_d = epoch_partial_d;
        }
        problem.stats->push_back(stats);
        if (this->config.verbose) {
            std::cout << "Epoch " << epoch + 1 << "/" << problem.n_epochs
                      << " | Objective: " << objective << " | Split: " << this->n_split;
            if (prune) {
                std::cout << " | Not Pruned %: " << stats.not_pruned_pct * 100.0f
                          << " | d': " << epoch_partial_d << " -> " << this->partial_d;
            } else {
                std::cout << " [no pruning]";
            }
            std::cout << std::endl;
        }
    }

    double RunBatch(
        TrainingData& training_data,
        const size_t* batch_ids,
        const size_t n_batch,
        const bool prune
    ) {
        auto& quantizer = *this->quantizer;
        const size_t n_clusters = problem.n_clusters;
        const size_t d = this->d;
        const vector_value_t* centroid_codes;
        const float* chunk_points;
        if constexpr (q == Quantization::f32) {
            centroid_codes = this->horizontal_centroids.get();
            chunk_points = tmp_chunk_codes.get();
        } else {
            centroid_codes = this->quantized_centroids.get();
            chunk_points = tmp_chunk_points.get();
        }
        if (prune) {
            this->PublishCentroids(n_clusters);
            quantizer.CacheCentroidPartialNorms(centroid_codes, n_clusters, d, this->partial_d);
        } else {
            if constexpr (q != Quantization::f32) {
                quantizer.Encode(
                    this->horizontal_centroids.get(), this->quantized_centroids.get(), n_clusters, d
                );
            }
            quantizer.ComputeNorms(centroid_codes, n_clusters, d, this->centroid_norms.get());
        }

        const bool quantized_update = this->config.quantized_centroid_update;
        const bool gather_points =
            q != Quantization::f32 && (!quantized_update || (q == Quantization::rabitq && prune));
        quantizer.ResetCentroidAccumulators(n_clusters, d);
        std::fill_n(this->prev_centroids.get(), n_clusters * d, 0.0f);
        std::fill_n(this->cluster_sizes.get(), n_clusters, 0u);
        if (prune) {
            std::fill_n(tmp_not_pruned_counts.get(), n_batch, size_t{0});
        }

        double objective = 0.0;
        for (size_t i = 0; i < n_batch; i += chunk_size) {
            const size_t n_chunk = std::min(chunk_size, n_batch - i);
            GatherPoints(training_data, batch_ids + i, n_chunk, prune, gather_points);
            quantizer.InvalidateCaches();
            if (prune) {
                quantizer.CacheDataPartialNorms(tmp_chunk_codes.get(), n_chunk, d, this->partial_d);
            }
            this->AssignPoints(
                prune,
                tmp_chunk_codes.get(),
                chunk_points,
                tmp_chunk_norms.get(),
                centroid_codes,
                this->horizontal_centroids.get(),
                n_chunk,
                n_clusters,
                training_data.centroids_pdx,
                tmp_chunk_labels.get(),
                tmp_chunk_distances.get(),
                prune ? tmp_not_pruned_counts.get() + i : nullptr
            );
            this->AccumulatePoints(
                quantized_update,
                tmp_chunk_codes.get(),
                chunk_points,
                tmp_chunk_labels.get(),
                n_chunk,
                n_clusters,
                this->prev_centroids.get(),
                this->cluster_sizes.get()
            );
            objective += ScatterPoints(batch_ids + i, n_chunk);
        }
        this->FinalizeMeans(
            quantized_update, this->prev_centroids.get(), this->cluster_sizes.get(), n_clusters
        );
        return objective;
    }

    void GatherPoints(
        const TrainingData& training_data,
        const size_t* ids,
        const size_t n_chunk,
        const bool prune,
        const bool gather_points
    ) {
        SKM_PROFILE_SCOPE("emeans_gather");
        const size_t code_size = this->code_size;
        const size_t d = this->d;
        this->GetExecutor().ParallelFor(n_chunk, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const size_t id = ids[i];
                memcpy(
                    tmp_chunk_codes.get() + i * code_size,
                    training_data.codes + id * code_size,
                    sizeof(vector_value_t) * code_size
                );
                if (gather_points) {
                    memcpy(
                        tmp_chunk_points.get() + i * d,
                        training_data.points + id * d,
                        sizeof(float) * d
                    );
                }
                if (prune) {
                    tmp_chunk_labels[i] = problem.labels[id];
                } else {
                    tmp_chunk_norms[i] = problem.norms[id];
                }
            }
        });
    }

    double ScatterPoints(const size_t* ids, const size_t n_chunk) {
        double objective = 0.0;
        for (size_t i = 0; i < n_chunk; ++i) {
            problem.labels[ids[i]] = tmp_chunk_labels[i];
            problem.distances[ids[i]] = tmp_chunk_distances[i];
            objective += static_cast<double>(tmp_chunk_distances[i]);
        }
        return objective;
    }

    float* EMAPositions() {
        return this->config.angular ? positions.get() : this->horizontal_centroids.get();
    }

    void ApplyEMA(const float alpha) {
        SKM_PROFILE_SCOPE("emeans_ema");
        const size_t d = this->d;
        const float decay = 1.0f - alpha;
        const bool replace = alpha == 1.0f;
        float* SKM_RESTRICT positions_p = EMAPositions();
        const float* SKM_RESTRICT means = this->prev_centroids.get();
        const uint32_t* SKM_RESTRICT counts = this->cluster_sizes.get();
        float* SKM_RESTRICT usage_p = usage.get();
        this->GetExecutor().ParallelFor(problem.n_clusters, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float retained = replace ? 0.0f : usage_p[i] * decay;
                if (counts[i] == 0) {
                    if (!replace) {
                        usage_p[i] = retained;
                    }
                    continue;
                }
                const float mass = retained + alpha * static_cast<float>(counts[i]);
                const float weight = alpha * static_cast<float>(counts[i]) / mass;
                float* SKM_RESTRICT position = positions_p + i * d;
                const float* SKM_RESTRICT mean = means + i * d;
                if (weight == 1.0f) {
                    memcpy(position, mean, sizeof(float) * d);
                } else {
                    SKM_VECTORIZE_LOOP
                    for (size_t j = 0; j < d; ++j) {
                        position[j] += weight * (mean[j] - position[j]);
                    }
                }
                usage_p[i] = mass;
            }
        });
    }

    void ReviveUnused() {
        SKM_PROFILE_SCOPE("emeans_revive");
        const size_t n_clusters = problem.n_clusters;
        float* usage_p = usage.get();
        const double total_usage = std::accumulate(usage_p, usage_p + n_clusters, 0.0);
        const float threshold = static_cast<float>(
            total_usage * EMEANS_REVIVE_THRESHOLD / static_cast<double>(n_clusters)
        );
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        size_t n_children = 0;
        size_t n_donors = 0;
        for (uint32_t i = 0; i < n_clusters; ++i) {
            const double u = std::max(uniform(revive_rng), std::numeric_limits<double>::min());
            if (usage_p[i] < threshold) {
                tmp_children[n_children++] = i;
            } else if (usage_p[i] > 0.0f) {
                tmp_donor_keys[i] = std::log(usage_p[i]) - std::log(-std::log(u));
                tmp_donors[n_donors++] = i;
            }
        }
        const size_t n_splits = std::min(n_children, n_donors);
        if (n_splits == 0) {
            return;
        }
        std::partial_sort(
            tmp_donors.get(),
            tmp_donors.get() + n_splits,
            tmp_donors.get() + n_donors,
            [&](uint32_t a, uint32_t b) { return tmp_donor_keys[a] > tmp_donor_keys[b]; }
        );
        float* positions_p = EMAPositions();
        for (size_t i = 0; i < n_splits; ++i) {
            const uint32_t donor = tmp_donors[i];
            const uint32_t child = tmp_children[i];
            this->SplitCentroid(
                positions_p + child * this->d, positions_p + donor * this->d, EMEANS_SPLIT_EPS
            );
            usage_p[donor] /= 2.0f;
            usage_p[child] = usage_p[donor];
        }
        this->n_split += n_splits;
    }

    void AccumulateProblemPoints(const TrainingData& training_data, const bool quantized) {
        if (!problem.point_ids) {
            this->AccumulatePoints(
                quantized,
                training_data.codes,
                training_data.points,
                problem.labels,
                problem.n_points,
                problem.n_clusters,
                this->prev_centroids.get(),
                this->cluster_sizes.get()
            );
            return;
        }
        const size_t n_chunk_points = std::min(batch_size, chunk_size);
        const bool gather_points = q != Quantization::f32 && !quantized;
        const float* chunk_points;
        if constexpr (q == Quantization::f32) {
            chunk_points = tmp_chunk_codes.get();
        } else {
            if (gather_points) {
                EnsureBufferSize(tmp_chunk_points, this->GetAllocator(), n_chunk_points * this->d);
            }
            chunk_points = tmp_chunk_points.get();
        }
        for (size_t i = 0; i < problem.n_points; i += n_chunk_points) {
            const size_t n_chunk = std::min(n_chunk_points, problem.n_points - i);
            GatherPoints(training_data, problem.point_ids + i, n_chunk, true, gather_points);
            this->AccumulatePoints(
                quantized,
                tmp_chunk_codes.get(),
                chunk_points,
                tmp_chunk_labels.get(),
                n_chunk,
                problem.n_clusters,
                this->prev_centroids.get(),
                this->cluster_sizes.get()
            );
        }
    }

    void FinalizeTraining(const TrainingData& training_data) {
        SKM_PROFILE_SCOPE("emeans_finalize");
        const size_t n_clusters = problem.n_clusters;
        const size_t d = this->d;
        const bool quantized =
            this->config.quantized_centroid_update && !this->config.full_precision_final_centroids;
        this->quantizer->ResetCentroidAccumulators(n_clusters, d);
        std::fill_n(this->prev_centroids.get(), n_clusters * d, 0.0f);
        std::fill_n(this->cluster_sizes.get(), n_clusters, 0u);
        AccumulateProblemPoints(training_data, quantized);
        this->FinalizeMeans(
            quantized, this->prev_centroids.get(), this->cluster_sizes.get(), n_clusters
        );
        const float* positions_p = EMAPositions();
        for (size_t i = 0; i < n_clusters; ++i) {
            if (this->cluster_sizes[i] == 0) {
                memcpy(
                    this->prev_centroids.get() + i * d,
                    positions_p + i * d,
                    sizeof(centroid_value_t) * d
                );
            }
        }
        std::swap(this->horizontal_centroids, this->prev_centroids);
        if (this->config.angular) {
            this->PostprocessCentroids(n_clusters);
        }
        this->PublishCentroids(n_clusters);
        this->quantizer->InvalidateCaches();
    }

    const size_t config_batch_size;
    const float config_initial_step;
    size_t batch_size = 0;
    float initial_step = 0.0f;
    const size_t chunk_size;
    EMeansProblem problem;
    std::mt19937 revive_rng;

    AllocatedBuffer<float> usage;
    AllocatedBuffer<float> positions;
    AllocatedBuffer<size_t> permutation;

    AllocatedBuffer<vector_value_t> tmp_chunk_codes;
    AllocatedBuffer<float> tmp_chunk_points;
    AllocatedBuffer<float> tmp_chunk_norms;
    AllocatedBuffer<uint32_t> tmp_chunk_labels;
    AllocatedBuffer<distance_t> tmp_chunk_distances;
    AllocatedBuffer<size_t> tmp_not_pruned_counts;
    AllocatedBuffer<double> tmp_donor_keys;
    AllocatedBuffer<uint32_t> tmp_donors;
    AllocatedBuffer<uint32_t> tmp_children;
};

using SuperEMeansSQ8 = SuperEMeans<Quantization::sq8>;
using SuperEMeansLVQ4 = SuperEMeans<Quantization::lvq4>;
using SuperEMeansRabitQ = SuperEMeans<Quantization::rabitq>;

} // namespace skmeans
