#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/executor.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/f32.h"
#include "superkmeans/quantizers/lvq4.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/rabitq.h"
#include "superkmeans/quantizers/sq8.h"

namespace skmeans {

/**
 * @brief Configuration parameters for SuperKMeans clustering.
 * Can be passed to the SuperKMeans constructor.
 *
 */
struct SuperKMeansConfig {
    // Training parameters
    uint32_t iters = 10; // Number of k-means iterations
    // We provide 2 ways to define the number of points to sample:
    float sampling_fraction = 0.3f; // Fraction of data to sample (0.0 to 1.0)
    uint32_t max_points_per_cluster =
        256;                    // Maximum number of points per cluster to sample (FAISS style)
    uint32_t n_threads = 0;     // Threads of the default executor (0 = max); unused with executor
    uint32_t seed = 42;         // Random seed for reproducibility
    bool use_blas_only = false; // Use BLAS-only computation for all iterations

    // Convergence parameters
    float tol = 1e-4f;                  // Tolerance for shift-based early termination
    float recall_tol = 0.005f;          // Tolerance for recall-based early termination
    bool early_termination = true;      // Whether to stop early on convergence
    bool sample_queries = false;        // Whether to sample queries from data
    size_t objective_k = 100;           // Number of nearest neighbors for recall computation
    float ann_explore_fraction = 0.01f; // Fraction of centroids to explore (0.0 to 1.0)

    // Sweet range for d' tuning
    float min_not_pruned_pct = 0.03f; // Minimum percentage of vectors not pruned (3% = 97% pruned)
    float max_not_pruned_pct = 0.05f; // Maximum percentage of vectors not pruned (5% = 95% pruned)
    // Adjustment factor for d' tuning when outside the sweet range
    float adjustment_factor_for_partial_d = 0.20f;

    // Output parameters
    bool unrotate_centroids = true; // Whether to unrotate centroids before returning
    bool verbose = false;           // Whether to print progress information
    bool angular = false;           // Whether to use spherical k-means
    bool suppress_warnings = false; // Whether to suppress warnings

    bool data_already_rotated = false;     // Whether input data is already rotated (skip rotation)
    bool quantized_centroid_update = true; // Accumulate centroids in quantized domain (u8 only)
    bool full_precision_final_centroids = false; // Recompute final centroids from raw float data

    // Runs the parallel loops (not owned). nullptr: SuperKMeans creates a default one of n_threads.
    ParallelExecutor* executor = nullptr;
};

/**
 * @brief Observable state of a SuperKMeans instance.
 */
struct SuperKMeansState {
    bool trained = false;
    bool trained_in_place = false;
    bool training_data_rotated = false;
    const ADSamplingPruner* rotator = nullptr;
    size_t code_size = 0;
    size_t n_encoded = 0;
};

/**
 * @brief Statistics for a single iteration of SuperKMeans clustering.
 */
struct SuperKMeansIterationStats {
    size_t iteration = 0;   // Iteration number (1-indexed)
    float objective = 0.0f; // Total clustering cost (WCSS)
    float shift = 0.0f;     // Average squared centroid shift from previous iteration
    size_t split = 0;       // Number of clusters that were split (empty cluster handling)
    float recall = 0.0f;    // Recall@k value (0.0 to 1.0, only when queries provided)
    // Percentage of vectors not pruned (0.0 to 1.0, -1.0 if not applicable)
    float not_pruned_pct = -1.0f;
    // Number of dimensions used for partial distance computation (d')
    uint32_t partial_d = 0;
    // Whether this iteration used BLAS-only computation (no PDX pruning)
    bool is_gemm_only = false;

    std::string ToJson() const {
        std::ostringstream oss;
        oss << "{\"iteration\":" << iteration << ",\"objective\":" << std::setprecision(6)
            << objective << ",\"shift\":" << std::setprecision(4) << shift << ",\"split\":" << split
            << ",\"recall\":" << std::setprecision(4) << recall
            << ",\"not_pruned_pct\":" << not_pruned_pct << ",\"partial_d\":" << partial_d
            << ",\"is_gemm_only\":" << (is_gemm_only ? "true" : "false") << "}";
        return oss.str();
    }

    static std::string VectorToJson(const std::vector<SuperKMeansIterationStats>& stats) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < stats.size(); ++i) {
            if (i > 0)
                oss << ",";
            oss << stats[i].ToJson();
        }
        oss << "]";
        return oss.str();
    }
};

/**
 * @brief Statistics about cluster size balance.
 */
struct ClusterBalanceStats {
    float mean = 0.0f;
    float geometric_mean = 0.0f;
    float stdev = 0.0f;
    float cv = 0.0f;
    size_t min = 0;
    size_t max = 0;
    size_t n_empty = 0;
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    float iqr = 0.0f;
    float whisker_low = 0.0f;
    float whisker_high = 0.0f;
    static constexpr std::array<float, 15> PERCENTILE_KEYS = {
        5.0f,
        10.0f,
        20.0f,
        25.0f,
        30.0f,
        40.0f,
        50.0f,
        60.0f,
        70.0f,
        75.0f,
        80.0f,
        90.0f,
        95.0f,
        99.0f,
        99.9f
    };
    std::array<float, 15> percentiles = {};

    std::string ToJson() const {
        std::ostringstream oss;
        oss << std::setprecision(4);
        oss << "{\"mean\":" << mean << ",\"geometric_mean\":" << geometric_mean
            << ",\"stdev\":" << stdev << ",\"cv\":" << cv << ",\"min\":" << min
            << ",\"max\":" << max << ",\"n_empty\":" << n_empty << ",\"skewness\":" << skewness
            << ",\"kurtosis\":" << kurtosis << ",\"iqr\":" << iqr
            << ",\"whisker_low\":" << whisker_low << ",\"whisker_high\":" << whisker_high
            << ",\"percentiles\":{";
        for (size_t i = 0; i < PERCENTILE_KEYS.size(); ++i) {
            if (i > 0)
                oss << ",";
            oss << "\"" << PERCENTILE_KEYS[i] << "\":" << percentiles[i];
        }
        oss << "}}";
        return oss.str();
    }

    void print() const {
        std::cout << "Cluster size stats: "
                  << "mean=" << mean << ", gmean=" << geometric_mean << ", std=" << stdev
                  << ", CV=" << cv << ", min=" << min << ", max=" << max << ", empty=" << n_empty
                  << ", skew=" << std::fixed << std::setprecision(3) << skewness
                  << ", kurt=" << kurtosis << std::defaultfloat << "\n  IQR=" << iqr
                  << ", whiskers=[" << whisker_low << ", " << whisker_high << "]"
                  << "\n  P5=" << percentiles[0] << " P25=" << percentiles[3]
                  << " P50=" << percentiles[6] << " P75=" << percentiles[9]
                  << " P95=" << percentiles[12] << " P99=" << percentiles[13] << std::endl;
    }
};

template <Quantization q = Quantization::f32>
class SuperKMeans : protected ExecutorHolder {
    static_assert(q != Quantization::sq4, "sq4 is not implemented as a quantizer");

  public:
    virtual ~SuperKMeans() = default;

  protected:
    using centroid_value_t = skmeans_centroid_value_t;
    using vector_value_t = skmeans_value_t<q>;
    // Pruner always operates on float data (rotation/unrotation are float-space operations)
    using pruner_t = ADSamplingPruner;
    using layout_t = PDXLayout<q>;
    using distance_t = skmeans_distance_t;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;
    using batch_computer = BatchComputer<DistanceFunction::l2, q>;

    /**
     * @brief Converts a real dimension count to the packed dimension count used for PDX.
     * Dimensions map 1:1 to storage elements.
     */
    static constexpr size_t PDXDim(size_t dim) { return dim; }

  public:
    /**
     * @brief Constructor with custom configuration
     *
     * @param n_clusters Number of clusters to create
     * @param dimensionality Number of dimensions in the data
     * @param config Configuration parameters (see SuperKMeansConfig)
     */
    SuperKMeans(size_t n_clusters, size_t dimensionality, const SuperKMeansConfig& config)
        : d(dimensionality), n_clusters(n_clusters), config(config) {
        SKMEANS_ENSURE_POSITIVE(n_clusters);
        SKMEANS_ENSURE_POSITIVE(dimensionality);
        SKMEANS_ENSURE_POSITIVE(config.iters);
        SKMEANS_ENSURE_POSITIVE(config.sampling_fraction);
        if (config.sampling_fraction > 1.0) {
            throw std::invalid_argument("sampling_fraction must be <= 1.0");
        }
        if (dimensionality > SKM_MAX_DIMS) {
            throw std::invalid_argument(
                "dimensionality exceeds SKM_MAX_DIMS (" + std::to_string(SKM_MAX_DIMS) + "), got " +
                std::to_string(dimensionality)
            );
        }
        if (config.executor == nullptr) {
            owned_executor = MakeDefaultExecutor(config.n_threads);
        }
        SetExecutor(config.executor != nullptr ? config.executor : owned_executor.get());
        f32_quantizer.SetExecutor(&GetExecutor());
        pruner = std::make_unique<pruner_t>(
            dimensionality, PRUNER_INITIAL_THRESHOLD, config.seed, config.data_already_rotated
        );
        pruner->SetExecutor(&GetExecutor());

        // If data is already rotated, we must not unrotate output centroids
        if (this->config.data_already_rotated) {
            this->config.unrotate_centroids = false;
        }
    }

    /**
     * @brief Default constructor
     *
     * @param n_clusters Number of clusters to create
     * @param dimensionality Number of dimensions in the data
     */
    SuperKMeans(size_t n_clusters, size_t dimensionality)
        : SuperKMeans(n_clusters, dimensionality, SuperKMeansConfig{}) {}

    /**
     * @brief Run k-means clustering to determine centroids
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors (ignored if queries is nullptr and sample_queries is
     * false)
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
        if (state.trained) {
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
        ParallelSection parallel_section(GetExecutor());
        const float* SKM_RESTRICT data_p = data;
        n_samples = GetNVectorsToSample(n, n_clusters);
        n_train = n;
        state.training_data_rotated = config.data_already_rotated;
        state.rotator = pruner.get();
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
            data_norms.reset(new float[n_samples]);
            centroid_norms.reset(new float[n_clusters]);
        }
        std::unique_ptr<size_t[]> not_pruned_counts(new size_t[n_samples]);
        EnsureTmpDistancesBuffer();
        vertical_d = PDXLayout<q>::GetDimensionSplit(PDXDim(d)).vertical_d;
        partial_horizontal_centroids.reset(new centroid_value_t[n_clusters * vertical_d]);

        auto centroids_pdx_wrapper =
            GenerateCentroids(data_p, n_samples, n_clusters, !config.data_already_rotated);
        if (config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::unique_ptr<float[]> data_samples_buffer;
        if (n_samples < n || !config.data_already_rotated) {
            data_samples_buffer.reset(new float[n_samples * d]);
        }
        auto data_to_cluster = SampleAndRotateVectors(
            data_p, data_samples_buffer.get(), n, n_samples, !config.data_already_rotated
        );

        RotateOrCopy(
            horizontal_centroids.get(),
            prev_centroids.get(),
            n_clusters,
            !config.data_already_rotated
        );

        quantizer = CreateQuantizer();
        quantizer->Fit(data_to_cluster, n_samples, d);
        code_size = quantizer->CodeSize(d);
        state.code_size = code_size;
        state.n_encoded = n_samples;

        // Compute partial_d: non-PDX quantizers must override vertical_d to full d
        if (quantizer->SupportsPruning() && !quantizer->NeedsPDXLayout()) {
            vertical_d = d;
        }
        partial_d = quantizer->InitialPartialD(vertical_d);
        if (config.verbose) {
            std::cout << "Front dimensions (d') = " << partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << d - vertical_d << std::endl;
        }

        // Data encoding
        const vector_value_t* encoded_data_p;
        if constexpr (q == Quantization::f32) {
            encoded_data_p = data_to_cluster;
        } else {
            quantized_data.reset(new vector_value_t[n_samples * code_size]);
            quantizer->Encode(data_to_cluster, quantized_data.get(), n_samples, d);
            encoded_data_p = quantized_data.get();
        }

        // Encode initial centroids (for f32 this is a memcpy, for u8 it quantizes)
        quantized_centroids.reset(new vector_value_t[n_clusters * code_size]);
        quantizer->Encode(prev_centroids.get(), quantized_centroids.get(), n_clusters, d);

        // Compute full norms via quantizer
        quantizer->ComputeNorms(encoded_data_p, n_samples, d, data_norms.get());
        quantizer->ComputeNorms(quantized_centroids.get(), n_clusters, d, centroid_norms.get());

        // Setup quantized PDX layout for pruning (f32 PDX is set up in GenerateCentroids)
        if constexpr (q != Quantization::f32) {
            if (quantizer->SupportsPruning() && quantizer->NeedsPDXLayout()) {
                pdxified_quantized_centroids.reset(new vector_value_t[n_clusters * PDXDim(d)]);
                partial_hor_quantized_centroids.reset(
                    new vector_value_t[n_clusters * PDXDim(vertical_d)]
                );
                PDXLayout<q>::template PDXify<false>(
                    quantized_centroids.get(),
                    pdxified_quantized_centroids.get(),
                    n_clusters,
                    PDXDim(d)
                );
                QuantizedCentroidsToAuxiliaryHorizontal(n_clusters);
                centroids_pdx_wrapper = layout_t(
                    pdxified_quantized_centroids.get(),
                    *pruner,
                    n_clusters,
                    PDXDim(d),
                    partial_hor_quantized_centroids.get()
                );
            }
        }

        std::unique_ptr<float[]> rotated_queries;
        // Recall tracking uses batch_computer which is only available for f32
        if constexpr (q == Quantization::f32) {
            if (n_queries) {
                centroids_to_explore = std::max<size_t>(
                    static_cast<size_t>(n_clusters * config.ann_explore_fraction), 1
                );
                if (config.verbose) {
                    std::cout << "Centroids to explore: " << centroids_to_explore << " ("
                              << config.ann_explore_fraction * 100.0f << "% of " << n_clusters
                              << ")" << std::endl;
                }
                {
                    SKM_PROFILE_SCOPE("allocator");
                    gt_assignments.reset(new uint32_t[n_queries * config.objective_k]);
                    gt_distances.reset(new distance_t[n_queries * config.objective_k]);
                    promising_centroids.reset(new uint32_t[n_queries * centroids_to_explore]);
                    recall_distances.reset(new distance_t[n_queries * centroids_to_explore]);
                    query_norms.reset(new distance_t[n_queries]);
                }
                rotated_queries.reset(new float[n_queries * d]);
                if (config.sample_queries) {
                    std::cout << "Sampling queries from data..." << std::endl;
                    SampleAndRotateVectors(
                        data_to_cluster, rotated_queries.get(), n_samples, n_queries, false
                    );
                } else {
                    RotateOrCopy(
                        queries, rotated_queries.get(), n_queries, !config.data_already_rotated
                    );
                }
                GetL2NormsRowMajor(rotated_queries.get(), n_queries, query_norms.get());
                GetGTAssignmentsAndDistances(data_to_cluster, rotated_queries.get(), n_queries);
            }
        }

        bool always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING || config.use_blas_only ||
                                n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING ||
                                !quantizer->SupportsPruning();
        bool partial_norms_computed = false;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;

        for (size_t iter_idx = 0; iter_idx < config.iters; ++iter_idx) {
            bool use_gemm_only = (iter_idx == 0) || always_gemm_only;
            if (!use_gemm_only && !partial_norms_computed) {
                quantizer->CacheDataPartialNorms(encoded_data_p, n_samples, d, partial_d);
                partial_norms_computed = true;
            }
            if (use_gemm_only) {
                RunIteration<true>(
                    data_to_cluster,
                    encoded_data_p,
                    centroids_pdx_wrapper,
                    not_pruned_counts.get(),
                    rotated_queries.get(),
                    n_queries,
                    n_samples,
                    n_clusters,
                    iter_idx,
                    iter_idx == 0,
                    iteration_stats
                );
            } else {
                RunIteration<false>(
                    data_to_cluster,
                    encoded_data_p,
                    centroids_pdx_wrapper,
                    not_pruned_counts.get(),
                    rotated_queries.get(),
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

        state.trained = true;

        // When quantized centroid update was used but we want full-precision final centroids,
        // recompute by averaging the raw (rotated) float training data using final assignments.
        if constexpr (q != Quantization::f32) {
            if (config.quantized_centroid_update && config.full_precision_final_centroids) {
                ResetCentroids(n_clusters);
                f32_quantizer.UpdateCentroids(
                    data_to_cluster,
                    assignments.get(),
                    horizontal_centroids.get(),
                    cluster_sizes.get(),
                    n_samples,
                    n_clusters,
                    d
                );
                f32_quantizer.FinalizeCentroids(
                    horizontal_centroids.get(), cluster_sizes.get(), n_clusters, d
                );
                if (config.angular) {
                    PostprocessCentroids(n_clusters);
                }
            }
        }

        auto output_centroids = GetOutputCentroids(config.unrotate_centroids);
        if (config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /**
     * @brief Run k-means clustering to determine centroids.
     *
     * Transformations (rotation and/or quantization) are done in place using the caller's buffer.
     * `unrotate_centroids is forced to false, so that data and centroids stay in the same domain
     *
     * Requires sampling_fraction == 1.0; a smaller value is overridden (with a warning) since
     * sampling gathers scattered rows and cannot be done in place.
     *
     * @param data Pointer to the data matrix (row-major, n × d). Overwritten with rotated data.
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors
     *
     * @return std::vector<skmeans_centroid_value_t> Trained centroids (in the rotated domain)
     */
    std::vector<skmeans_centroid_value_t> TrainInPlace(
        float* SKM_RESTRICT data,
        const size_t n,
        const float* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        ConfigInPlaceTraining(data, n);
        return Train(data, n, queries, n_queries);
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
        const float* SKM_RESTRICT vectors,
        const float* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) {
        SKM_PROFILE_SCOPE("assign");
        ParallelSection parallel_section(GetExecutor());
        using f32_batch_computer = BatchComputer<DistanceFunction::l2, Quantization::f32>;
        EnsureTmpDistancesBuffer();

        std::vector<uint32_t> result_assignments(n_vectors);
        std::vector<float> result_distances(n_vectors);

        std::vector<float> vector_norms(n_vectors);
        std::vector<float> centroid_norms_local(n_centroids);

        Eigen::Map<const MatrixR> vectors_mat(vectors, n_vectors, d);
        Eigen::Map<VectorR> v_norms(vector_norms.data(), n_vectors);
        v_norms.noalias() = vectors_mat.rowwise().squaredNorm();

        Eigen::Map<const MatrixR> centroids_mat(centroids, n_centroids, d);
        Eigen::Map<VectorR> c_norms(centroid_norms_local.data(), n_centroids);
        c_norms.noalias() = centroids_mat.rowwise().squaredNorm();

        f32_batch_computer::FindNearestNeighbor(
            GetExecutor(),
            vectors,
            centroids,
            n_vectors,
            n_centroids,
            d,
            vector_norms.data(),
            centroid_norms_local.data(),
            result_assignments.data(),
            result_distances.data(),
            tmp_distances_buffer.get()
        );

        return result_assignments;
    }

    /**
     * @brief Quantized assignment that fits a fresh quantizer on the input.
     *
     * Standalone like Assign(): fits a new quantizer on the input vectors,
     * encodes vectors and centroids, and performs nearest-neighbor assignment
     * entirely in quantized space. Does not reuse or mutate trained state.
     * Rotation is only applied for RaBitQ.
     *
     * @param vectors The data matrix (row-major, n_vectors x d, float)
     * @param centroids The centroids matrix (row-major, n_centroids x d, float)
     * @param n_vectors Number of vectors
     * @param n_centroids Number of centroids
     * @return std::vector<uint32_t> Assignment for each vector (index of nearest centroid)
     */
    [[nodiscard]] std::vector<uint32_t> QuantizedAssign(
        const float* SKM_RESTRICT vectors,
        const float* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) {
        SKM_PROFILE_SCOPE("quantized_assign");
        ParallelSection parallel_section(GetExecutor());
        EnsureTmpDistancesBuffer();

        auto local_quantizer = CreateQuantizer();

        const float* encode_vectors = vectors;
        const float* encode_centroids = centroids;
        std::unique_ptr<float[]> rotated_vectors_buf;
        std::unique_ptr<float[]> rotated_centroids_buf;
        const bool rotate = !config.data_already_rotated && q == Quantization::rabitq;
        if (rotate) {
            rotated_vectors_buf.reset(new float[n_vectors * d]);
            rotated_centroids_buf.reset(new float[n_centroids * d]);
            RotateOrCopy(vectors, rotated_vectors_buf.get(), n_vectors, true);
            RotateOrCopy(centroids, rotated_centroids_buf.get(), n_centroids, true);
            encode_vectors = rotated_vectors_buf.get();
            encode_centroids = rotated_centroids_buf.get();
        }

        local_quantizer->Fit(encode_vectors, n_vectors, d);

        const size_t cs = local_quantizer->CodeSize(d);
        std::unique_ptr<vector_value_t[]> q_vectors(new vector_value_t[n_vectors * cs]);
        std::unique_ptr<vector_value_t[]> q_centroids(new vector_value_t[n_centroids * cs]);
        local_quantizer->Encode(encode_vectors, q_vectors.get(), n_vectors, d);
        local_quantizer->Encode(encode_centroids, q_centroids.get(), n_centroids, d);

        std::vector<uint32_t> result_assignments(n_vectors);
        std::unique_ptr<distance_t[]> result_distances(new distance_t[n_vectors]);
        std::unique_ptr<float[]> v_norms(new float[n_vectors]);
        std::unique_ptr<float[]> c_norms(new float[n_centroids]);
        local_quantizer->ComputeNorms(q_vectors.get(), n_vectors, d, v_norms.get());
        local_quantizer->ComputeNorms(q_centroids.get(), n_centroids, d, c_norms.get());
        local_quantizer->FindNearestNeighbor(
            q_vectors.get(),
            q_centroids.get(),
            encode_vectors,
            encode_centroids,
            n_vectors,
            n_centroids,
            d,
            v_norms.get(),
            c_norms.get(),
            result_assignments.data(),
            result_distances.get(),
            tmp_distances_buffer.get()
        );

        return result_assignments;
    }

    /**
     * @brief Fast assignment using GEMM+PRUNING with trained state.
     *
     * Requires that the vectors passed here are the same as those used in .Train().
     * Leverages the assignments from training for a faster
     * assignment than brute force Assign().
     *
     * TODO(@lkuffo, high): Signature is misleading, as vectors
     * and centroids incoming in params are not used
     *
     * @param vectors The training data matrix (row-major, n_vectors x d)
     * @param centroids The centroids matrix (row-major, n_centroids x d)
     * @param n_vectors Number of vectors
     * @param n_centroids Number of centroids
     * @return std::vector<uint32_t> Assignment for each vector (index of nearest centroid)
     */
    [[nodiscard]] std::vector<uint32_t> AssignTrainingPoints(
        const float* SKM_RESTRICT vectors,
        const float* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) {
        SKM_PROFILE_SCOPE("assign_training_points");
        if (!state.trained) {
            throw std::runtime_error("AssignTrainingPoints requires SuperKMeans to be trained first"
            );
        }
        if (n_vectors != n_train) {
            throw std::runtime_error(
                "AssignTrainingPoints requires the same vectors used in Train() "
                "(n_vectors must equal the training set size)"
            );
        }
        ParallelSection parallel_section(GetExecutor());

        if constexpr (q != Quantization::f32) {
            if (config.sampling_fraction == 1.0f && quantizer->SupportsPruning() &&
                !config.use_blas_only && d >= DIMENSION_THRESHOLD_FOR_PRUNING &&
                n_clusters > N_CLUSTERS_THRESHOLD_FOR_PRUNING && config.iters > 1) {
                std::vector<uint32_t> result_assignments(
                    assignments.get(), assignments.get() + n_vectors
                );
                std::vector<distance_t> result_distances(
                    n_vectors, std::numeric_limits<distance_t>::max()
                );
                std::vector<size_t> not_pruned_counts(n_vectors, 0);

                quantizer->CacheCentroidPartialNorms(
                    quantized_centroids.get(), n_centroids, d, partial_d
                );

                layout_t pdx_wrapper;
                if (quantizer->NeedsPDXLayout()) {
                    pdx_wrapper = layout_t(
                        pdxified_quantized_centroids.get(),
                        *pruner,
                        n_centroids,
                        PDXDim(d),
                        partial_hor_quantized_centroids.get()
                    );
                }

                EnsureTmpDistancesBuffer();
                quantizer->FindNearestNeighborWithPruning(
                    quantized_data.get(),
                    quantized_centroids.get(),
                    vectors,
                    horizontal_centroids.get(),
                    n_vectors,
                    n_centroids,
                    d,
                    result_assignments.data(),
                    result_distances.data(),
                    pdx_wrapper,
                    partial_d,
                    not_pruned_counts.data(),
                    tmp_distances_buffer.get()
                );

                return result_assignments;
            }

            if (config.sampling_fraction == 1.0f) {
                EnsureTmpDistancesBuffer();
                quantizer->InvalidateCaches();
                std::vector<uint32_t> result_assignments(n_vectors);
                std::unique_ptr<distance_t[]> result_distances(new distance_t[n_vectors]);
                std::unique_ptr<float[]> v_norms(new float[n_vectors]);
                std::unique_ptr<float[]> c_norms(new float[n_centroids]);
                quantizer->ComputeNorms(quantized_data.get(), n_vectors, d, v_norms.get());
                quantizer->ComputeNorms(quantized_centroids.get(), n_centroids, d, c_norms.get());
                quantizer->FindNearestNeighbor(
                    quantized_data.get(),
                    quantized_centroids.get(),
                    vectors,
                    horizontal_centroids.get(),
                    n_vectors,
                    n_centroids,
                    d,
                    v_norms.get(),
                    c_norms.get(),
                    result_assignments.data(),
                    result_distances.get(),
                    tmp_distances_buffer.get()
                );
                return result_assignments;
            }

            if (!config.suppress_warnings) {
                std::cout << "WARNING: AssignTrainingPoints cannot reuse trained state, falling "
                             "back to QuantizedAssign"
                          << std::endl;
            }
            return QuantizedAssign(vectors, centroids, n_vectors, n_centroids);
        }

        if constexpr (q == Quantization::f32) {
            if (config.use_blas_only || d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                if (!config.suppress_warnings) {
                    std::cout
                        << "WARNING: AssignTrainingPoints cannot use pruning, falling back to "
                           "brute force Assign"
                        << std::endl;
                }
                return Assign(vectors, centroids, n_vectors, n_centroids);
            }
            if (config.verbose) {
                Profiler::Get().Reset();
            }

            std::vector<uint32_t> result_assignments(n_vectors);

            std::unique_ptr<size_t[]> not_pruned_counts(new size_t[n_vectors]);
            std::fill(not_pruned_counts.get(), not_pruned_counts.get() + n_vectors, 0);
            std::unique_ptr<float[]> data_buffer;
            const float* data_p;
            if (state.training_data_rotated) {
                data_p = vectors;
            } else {
                data_buffer.reset(new float[n_vectors * d]);
                RotateOrCopy(vectors, data_buffer.get(), n_vectors, true);
                data_p = data_buffer.get();
            }
            quantizer->CacheCentroidPartialNorms(
                horizontal_centroids.get(), n_centroids, d, partial_d
            );

            // Consolidate ran at the end of the last RunIteration, so centroids are final.
            auto pdx_centroids = PDXLayout<q>(
                this->centroids.get(), *pruner, n_clusters, d, partial_horizontal_centroids.get()
            );

            if (config.sampling_fraction == 1.0f) {
                quantizer->CacheDataPartialNorms(data_p, n_vectors, d, partial_d);
                EnsureTmpDistancesBuffer();
                quantizer->FindNearestNeighborWithPruning(
                    data_p,
                    horizontal_centroids.get(),
                    data_p,
                    horizontal_centroids.get(),
                    n_vectors,
                    n_clusters,
                    d,
                    assignments.get(),
                    distances.get(),
                    pdx_centroids,
                    partial_d,
                    not_pruned_counts.get(),
                    tmp_distances_buffer.get()
                );
                memcpy(result_assignments.data(), assignments.get(), n_vectors * sizeof(uint32_t));
                return result_assignments;
            } else if (config.sampling_fraction > 0.8f) {
                size_t cur_vector_idx = 0;
                for (; cur_vector_idx < n_samples; ++cur_vector_idx) {
                    result_assignments[sampled_indices[cur_vector_idx]] =
                        assignments[cur_vector_idx];
                }
                std::mt19937 rng(config.seed + 1);
                std::discrete_distribution<uint32_t> cluster_dist(
                    cluster_sizes.get(), cluster_sizes.get() + n_clusters
                );
                for (; cur_vector_idx < n_vectors; ++cur_vector_idx) {
                    result_assignments[sampled_indices[cur_vector_idx]] = cluster_dist(rng);
                }

                quantizer->CacheDataPartialNorms(data_p, n_vectors, d, partial_d);
                EnsureTmpDistancesBuffer();
                quantizer->FindNearestNeighborWithPruning(
                    data_p,
                    horizontal_centroids.get(),
                    data_p,
                    horizontal_centroids.get(),
                    n_vectors,
                    n_clusters,
                    d,
                    result_assignments.data(),
                    distances.get(),
                    pdx_centroids,
                    partial_d,
                    not_pruned_counts.get(),
                    tmp_distances_buffer.get()
                );
                return result_assignments;
            } else {
                // Try to approximate the assignment of each non-sampled vector
                // by first clustering the centroids into a smaller number of "meso-centroids"
                // then assigning each non-sampled vector to the nearest meso-centroid, and finally
                // mapping that meso-centroid back to a original centroid.
                SuperKMeansConfig tmp_config;
                tmp_config.iters = 10;
                tmp_config.sampling_fraction = 1.0f;
                tmp_config.use_blas_only = false;
                tmp_config.verbose = config.verbose;
                tmp_config.suppress_warnings = config.suppress_warnings;
                tmp_config.seed = config.seed;
                tmp_config.angular = config.angular;
                tmp_config.data_already_rotated = state.training_data_rotated;
                tmp_config.executor = &GetExecutor();
                auto new_n_centroids = static_cast<size_t>(std::sqrt(n_centroids));
                SuperKMeans tmp_kmeans(new_n_centroids, d, tmp_config);
                auto meso_centroids = tmp_kmeans.Train(centroids, n_centroids);
                auto meso_assignments =
                    tmp_kmeans.Assign(vectors, meso_centroids.data(), n_vectors, new_n_centroids);

                auto centroids_to_meso = tmp_kmeans.Assign(
                    centroids, meso_centroids.data(), n_centroids, new_n_centroids
                );
                std::vector<uint32_t> meso_to_original(new_n_centroids, 0);
                for (size_t c = 0; c < n_centroids; ++c) {
                    meso_to_original[centroids_to_meso[c]] = static_cast<uint32_t>(c);
                }

                size_t cur_vector_idx = 0;
                for (; cur_vector_idx < n_samples; ++cur_vector_idx) {
                    result_assignments[sampled_indices[cur_vector_idx]] =
                        assignments[cur_vector_idx];
                }
                for (; cur_vector_idx < n_vectors; ++cur_vector_idx) {
                    size_t orig_idx = sampled_indices[cur_vector_idx];
                    result_assignments[orig_idx] = meso_to_original[meso_assignments[orig_idx]];
                }

                quantizer->CacheDataPartialNorms(data_p, n_vectors, d, partial_d);
                EnsureTmpDistancesBuffer();
                quantizer->FindNearestNeighborWithPruning(
                    data_p,
                    horizontal_centroids.get(),
                    data_p,
                    horizontal_centroids.get(),
                    n_vectors,
                    n_clusters,
                    d,
                    result_assignments.data(),
                    distances.get(),
                    pdx_centroids,
                    partial_d,
                    not_pruned_counts.get(),
                    tmp_distances_buffer.get()
                );
                return result_assignments;
            }
        }
    }

    /** @brief Returns the number of clusters. */
    [[nodiscard]] size_t GetNClusters() const noexcept { return n_clusters; }

    /** @brief Returns the dimensionality of the vectors. */
    [[nodiscard]] size_t GetDimensionality() const noexcept { return d; }

    /** @brief Returns whether the model has been trained. */
    [[nodiscard]] bool IsTrained() const noexcept { return state.trained; }

    /** @brief Returns how training was carried out (see SuperKMeansState). */
    [[nodiscard]] const SuperKMeansState& GetState() const noexcept { return state; }

    /**
     * @brief Returns the fitted quantizer, or nullptr before training.
     */
    [[nodiscard]] const skmeans_quantizer_t<q>* GetQuantizer() const noexcept {
        return quantizer.get();
    }

    /**
     * @brief Returns the encoded training vectors, laid out as n_encoded rows of code_size
     * elements (bytes for the u8 quantizers).
     *
     * nullptr for f32, which clusters the training data directly instead of encoding a copy.
     */
    [[nodiscard]] const vector_value_t* GetQuantizedData() const noexcept {
        return quantized_data.get();
    }

    /** @brief Returns the sampling fraction used during training. */
    [[nodiscard]] float GetSamplingFraction() const noexcept { return config.sampling_fraction; }

    /** @brief Returns the per-iteration statistics collected during training. */
    [[nodiscard]] const std::vector<SuperKMeansIterationStats>& GetIterationStats() const noexcept {
        return iteration_stats;
    }

    /** @brief Returns a pointer to the distances array. */
    [[nodiscard]] distance_t* GetDistancesPointer() { return distances.get(); }

    /**
     * @brief Calculate cluster balance statistics from assignments
     *
     * @param assignments Array of cluster assignments [n_samples]
     * @param n_samples Number of samples
     * @param n_clusters Number of clusters
     * @return ClusterBalanceStats containing mean, stdev, CV, min, max
     */
    [[nodiscard]] static ClusterBalanceStats GetClustersBalanceStats(
        const uint32_t* assignments,
        size_t n_samples,
        size_t n_clusters
    ) {
        ClusterBalanceStats stats;
        std::vector<size_t> cluster_sizes(n_clusters, 0);
        for (size_t i = 0; i < n_samples; ++i) {
            cluster_sizes[assignments[i]]++;
        }

        const float n_f = static_cast<float>(n_clusters);

        auto sum = std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), size_t{0});
        stats.mean = static_cast<float>(sum) / n_f;

        // Geometric mean and empty count
        float log_sum = 0.0f;
        size_t non_zero_count = 0;
        for (size_t size : cluster_sizes) {
            if (size > 0) {
                log_sum += std::log(static_cast<float>(size));
                non_zero_count++;
            }
        }
        stats.geometric_mean =
            (non_zero_count > 0) ? std::exp(log_sum / static_cast<float>(non_zero_count)) : 0.0f;
        stats.n_empty = n_clusters - non_zero_count;

        // Variance, stdev, CV
        double var_acc = 0.0;
        for (size_t size : cluster_sizes) {
            double diff = static_cast<double>(size) - static_cast<double>(stats.mean);
            var_acc += diff * diff;
        }
        float variance = static_cast<float>(var_acc / static_cast<double>(n_clusters));
        stats.stdev = std::sqrt(variance);
        stats.cv = (stats.mean > 0.0f) ? stats.stdev / stats.mean : 0.0f;

        // Skewness and kurtosis (population, using stdev already computed)
        if (stats.stdev > 0.0f) {
            double m3 = 0.0, m4 = 0.0;
            for (size_t size : cluster_sizes) {
                double z = (static_cast<double>(size) - static_cast<double>(stats.mean)) /
                           static_cast<double>(stats.stdev);
                double z2 = z * z;
                m3 += z2 * z;
                m4 += z2 * z2;
            }
            stats.skewness = static_cast<float>(m3 / static_cast<double>(n_clusters));
            stats.kurtosis = static_cast<float>(m4 / static_cast<double>(n_clusters) - 3.0);
        }

        // Min, max
        auto minmax = std::minmax_element(cluster_sizes.begin(), cluster_sizes.end());
        stats.min = *minmax.first;
        stats.max = *minmax.second;

        // Sort for percentiles, IQR, whiskers
        std::vector<size_t> sorted(cluster_sizes);
        std::sort(sorted.begin(), sorted.end());

        // Percentiles (linear interpolation)
        auto percentile = [&](float p) -> float {
            double rank = (p / 100.0) * static_cast<double>(n_clusters - 1);
            size_t lo = static_cast<size_t>(rank);
            size_t hi = std::min(lo + 1, n_clusters - 1);
            double frac = rank - static_cast<double>(lo);
            return static_cast<float>(
                static_cast<double>(sorted[lo]) * (1.0 - frac) +
                static_cast<double>(sorted[hi]) * frac
            );
        };
        for (size_t i = 0; i < ClusterBalanceStats::PERCENTILE_KEYS.size(); ++i) {
            stats.percentiles[i] = percentile(ClusterBalanceStats::PERCENTILE_KEYS[i]);
        }

        // IQR, whiskers (Tukey fences)
        float q1 = percentile(25.0f);
        float q3 = percentile(75.0f);
        stats.iqr = q3 - q1;
        float fence_low = q1 - 1.5f * stats.iqr;
        float fence_high = q3 + 1.5f * stats.iqr;
        // Clamp whiskers to actual data range
        stats.whisker_low = static_cast<float>(sorted.front());
        stats.whisker_high = static_cast<float>(sorted.back());
        for (size_t v : sorted) {
            if (static_cast<float>(v) >= fence_low) {
                stats.whisker_low = static_cast<float>(v);
                break;
            }
        }
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            if (static_cast<float>(*it) <= fence_high) {
                stats.whisker_high = static_cast<float>(*it);
                break;
            }
        }
        return stats;
    }

    /**
     * @brief Compute WCSS in the original float32 domain.
     *
     * Sums L2 squared distances from each vector to its assigned centroid
     * using the SIMD-accelerated Horizontal kernel.
     *
     * @param data Float32 data matrix (row-major, n_vectors x d)
     * @param centroids Float32 centroid matrix (row-major, n_centroids x d)
     * @param assignments Cluster assignment for each vector [n_vectors]
     * @param n_vectors Number of data vectors
     * @param d Dimensionality
     * @param executor Runs the loop; nullptr uses a default one on all cores
     * @return WCSS (sum of squared L2 distances to assigned centroids)
     */
    [[nodiscard]] static double ComputeWCSS(
        const float* SKM_RESTRICT data,
        const float* SKM_RESTRICT centroids,
        const uint32_t* assignments,
        size_t n_vectors,
        size_t d,
        ParallelExecutor* executor = nullptr
    ) {
        using f32_l2 = DistanceComputer<DistanceFunction::l2, Quantization::f32>;
        ExecutorScope executor_scope(executor, 0);
        std::vector<double> worker_wcss(executor_scope.Get().NumWorkers(), 0.0);
        executor_scope.Get().ParallelFor(n_vectors, [&](size_t begin, size_t end, size_t worker) {
            double local_wcss = 0.0;
            for (size_t i = begin; i < end; ++i) {
                local_wcss += static_cast<double>(
                    f32_l2::Horizontal(data + i * d, centroids + assignments[i] * d, d)
                );
            }
            worker_wcss[worker] = local_wcss;
        });
        return std::accumulate(worker_wcss.begin(), worker_wcss.end(), 0.0);
    }

  protected:
    /**
     * @brief Updates centroids by accumulating assigned vectors.
     *
     * After this call, horizontal_centroids contains the sum of assigned vectors.
     * ConsolidateCentroids() must be called to normalize by cluster sizes.
     *
     * @param data Data matrix (row-major, n_samples × d)
     * @param n_samples
     * @param n_clusters
     */
    /**
     * @brief Zeros the centroid accumulators and cluster sizes for a new assignment pass.
     */
    void ResetCentroids(const size_t n_clusters) {
        SKM_PROFILE_SCOPE("fill");
        std::fill(horizontal_centroids.get(), horizontal_centroids.get() + (n_clusters * d), 0.0f);
        std::fill(cluster_sizes.get(), cluster_sizes.get() + n_clusters, 0);
    }

    /**
     * @brief Runs a single K-Means iteration with either GEMM-only or GEMM+PRUNING strategy.
     *
     *
     * @tparam GEMM_ONLY If true, uses full GEMM
     *                   If false, uses GEMM+PRUNING (with TunePartialD).
     *
     * @param data_to_cluster Training data (rotated, row-major)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param centroids_pdx_wrapper PDX-layout centroids (only used when !GEMM_ONLY)
     * @param centroids_partial_norms Partial norms buffer (only used when !GEMM_ONLY)
     * @param not_pruned_counts Pruning statistics buffer (only used when !GEMM_ONLY)
     * @param rotated_queries Query vectors for recall computation (nullptr if n_queries==0)
     * @param n_queries Number of query vectors
     * @param n_samples Number of training samples
     * @param n_clusters Number of clusters
     * @param iter_idx Current iteration index (0-based)
     * @param is_first_iter Whether this is the first iteration (skips centroid swap)
     */
    template <bool GEMM_ONLY>
    void RunIteration(
        const float* SKM_RESTRICT data_to_cluster,
        const vector_value_t* SKM_RESTRICT encoded_data_p,
        layout_t& centroids_pdx_wrapper,
        size_t* SKM_RESTRICT not_pruned_counts,
        const float* SKM_RESTRICT rotated_queries,
        const size_t n_queries,
        const size_t n_samples,
        const size_t n_clusters,
        size_t& iter_idx,
        const bool is_first_iter,
        std::vector<SuperKMeansIterationStats>& target_stats
    ) {
        if (!is_first_iter) {
            std::swap(horizontal_centroids, prev_centroids);
        }

        if constexpr (GEMM_ONLY) {
            // Encode centroids for this iteration (for f32 this is a memcpy)
            quantizer->Encode(prev_centroids.get(), quantized_centroids.get(), n_clusters, d);
            quantizer->ComputeNorms(quantized_centroids.get(), n_clusters, d, centroid_norms.get());
            quantizer->FindNearestNeighbor(
                encoded_data_p,
                quantized_centroids.get(),
                data_to_cluster,
                prev_centroids.get(),
                n_samples,
                n_clusters,
                d,
                data_norms.get(),
                centroid_norms.get(),
                assignments.get(),
                distances.get(),
                tmp_distances_buffer.get()
            );
            ResetCentroids(n_clusters);
            quantizer->ResetCentroidAccumulators(n_clusters, d);
        } else {
            std::fill(not_pruned_counts, not_pruned_counts + n_samples, 0);
            quantizer->Encode(prev_centroids.get(), quantized_centroids.get(), n_clusters, d);
            quantizer->CacheCentroidPartialNorms(
                quantized_centroids.get(), n_clusters, d, partial_d
            );
            EnsureTmpDistancesBuffer();
            quantizer->FindNearestNeighborWithPruning(
                encoded_data_p,
                quantized_centroids.get(),
                data_to_cluster,
                prev_centroids.get(),
                n_samples,
                n_clusters,
                d,
                assignments.get(),
                distances.get(),
                centroids_pdx_wrapper,
                partial_d,
                not_pruned_counts,
                tmp_distances_buffer.get()
            );
            ResetCentroids(n_clusters);
            quantizer->ResetCentroidAccumulators(n_clusters, d);
        }

        if (config.quantized_centroid_update) {
            quantizer->UpdateCentroids(
                encoded_data_p,
                assignments.get(),
                horizontal_centroids.get(),
                cluster_sizes.get(),
                n_samples,
                n_clusters,
                d
            );
        } else {
            f32_quantizer.UpdateCentroids(
                data_to_cluster,
                assignments.get(),
                horizontal_centroids.get(),
                cluster_sizes.get(),
                n_samples,
                n_clusters,
                d
            );
        }

        float avg_not_pruned_pct = -1.0f;
        uint32_t old_partial_d = partial_d;
        if constexpr (!GEMM_ONLY) {
            bool partial_d_changed = false;
            avg_not_pruned_pct =
                TunePartialD(not_pruned_counts, n_samples, n_clusters, partial_d_changed);
            if (partial_d_changed) {
                quantizer->CacheDataPartialNorms(encoded_data_p, n_samples, d, partial_d);
            }
        }

        ConsolidateCentroids(n_samples, n_clusters);

        ComputeCost(n_samples);
        ComputeShift(n_clusters);

        // Recall computation uses batch_computer which is only available for f32
        if constexpr (q == Quantization::f32) {
            if (n_queries) {
                GetL2NormsRowMajor(horizontal_centroids.get(), n_clusters, centroid_norms.get());
                recall = ComputeRecall(rotated_queries, n_queries);
            }
        }

        SuperKMeansIterationStats stats;
        stats.iteration = iter_idx + 1;
        stats.objective = cost;
        stats.shift = shift;
        stats.split = n_split;
        stats.recall = recall;
        stats.is_gemm_only = GEMM_ONLY;
        if constexpr (!GEMM_ONLY) {
            stats.not_pruned_pct = avg_not_pruned_pct;
            stats.partial_d = old_partial_d;
        }
        target_stats.push_back(stats);

        if (config.verbose) {
            std::cout << "Iteration " << iter_idx + 1 << "/" << config.iters
                      << " | Objective: " << cost << " | Objective improvement: "
                      << (iter_idx > 0 ? 1 - (cost / prev_cost) : 0.0f) << " | Shift: " << shift
                      << " | Split: " << n_split << " | Recall: " << recall;
            if constexpr (GEMM_ONLY) {
                std::cout << " [BLAS-only]";
            } else {
                std::cout << " | Not Pruned %: " << avg_not_pruned_pct * 100.0f
                          << " | d': " << old_partial_d << " -> " << partial_d;
            }
            std::cout << "\n";
            std::cout << std::endl;
        }
    }

    /**
     * @brief Handles empty clusters by splitting large clusters.
     * Taken from FAISS implementation:
     * https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp
     *
     * When a cluster becomes empty (no points assigned), this method splits
     * a large cluster to repopulate it. Selection is probabilistic based on
     * cluster sizes.
     */
    virtual void SplitClusters(const size_t n_samples, const size_t n_clusters) {
        n_split = 0;
        std::mt19937 rng(config.seed);
        auto horizontal_centroids_p = horizontal_centroids.get();
        const float split_denom = static_cast<float>(std::max(size_t{1}, n_samples - n_clusters));
        for (size_t ci = 0; ci < n_clusters; ci++) {
            if (cluster_sizes[ci] == 0) {
                size_t cj;
                for (cj = 0; true; cj = (cj + 1) % n_clusters) {
                    // Probability to pick this cluster for split
                    float p = (cluster_sizes[cj] - 1.0) / split_denom;
                    float r = std::uniform_real_distribution<float>(0, 1)(rng);
                    if (r < p) {
                        break; // Found our cluster to be split
                    }
                }
                memcpy(
                    (void*) (horizontal_centroids_p + ci * d),
                    (void*) (horizontal_centroids_p + cj * d),
                    sizeof(centroid_value_t) * d
                );

                // Small symmetric perturbation
                for (size_t j = 0; j < d; j++) {
                    if (j % 2 == 0) {
                        horizontal_centroids_p[ci * d + j] *= 1 + CENTROID_PERTURBATION_EPS;
                        horizontal_centroids_p[cj * d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                    } else {
                        horizontal_centroids_p[ci * d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                        horizontal_centroids_p[cj * d + j] *= 1 + CENTROID_PERTURBATION_EPS;
                    }
                }

                // Assume even split of the cluster
                cluster_sizes[ci] = cluster_sizes[cj] / 2;
                cluster_sizes[cj] -= cluster_sizes[ci];
                n_split++;
            }
        }
    }

    /**
     * @brief Finalizes centroid computation after assignment.
     *
     * Divides accumulated sums by cluster sizes to get mean centroids,
     * handles empty clusters via splitting, and converts to PDX layout.
     */
    void ConsolidateCentroids(const size_t n_samples, const size_t n_clusters) {
        SKM_PROFILE_SCOPE("consolidate");
        {
            SKM_PROFILE_SCOPE("consolidate/splitting");
            if (config.quantized_centroid_update) {
                quantizer->FinalizeCentroids(
                    horizontal_centroids.get(), cluster_sizes.get(), n_clusters, d
                );
            } else {
                f32_quantizer.FinalizeCentroids(
                    horizontal_centroids.get(), cluster_sizes.get(), n_clusters, d
                );
            }
            SplitClusters(n_samples, n_clusters);
        }
        {
            SKM_PROFILE_SCOPE("consolidate/normalize");
            if (config.angular)
                PostprocessCentroids(n_clusters);
        }
        if (quantizer->SupportsPruning()) {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            if constexpr (q == Quantization::f32) {
                PDXLayout<q>::template PDXify<false>(
                    horizontal_centroids.get(), centroids.get(), n_clusters, d
                );
                CentroidsToAuxiliaryHorizontal(n_clusters);
            } else {
                quantizer->Encode(
                    horizontal_centroids.get(), quantized_centroids.get(), n_clusters, d
                );
                if (quantizer->NeedsPDXLayout()) {
                    PDXLayout<q>::template PDXify<false>(
                        quantized_centroids.get(),
                        pdxified_quantized_centroids.get(),
                        n_clusters,
                        PDXDim(d)
                    );
                    QuantizedCentroidsToAuxiliaryHorizontal(n_clusters);
                }
            }
        }
    }

    /**
     * @brief Computes Within-Cluster Sum of Squares (WCSS).
     */
    void ComputeCost(const size_t n_samples) {
        SKM_PROFILE_SCOPE("compute_cost");
        prev_cost = cost;
        cost = 0.0f;
        // TODO(@lkuffo, low): accumulate in double; float loses precision over many samples
        SKM_VECTORIZE_LOOP
        for (size_t i = 0; i < n_samples; ++i) {
            cost += distances[i];
        }
    }

    /**
     * @brief Computes the squared centroid shift from previous iteration.
     *
     * Used for convergence detection - small shift indicates centroids have stabilized.
     */
    void ComputeShift(const size_t n_clusters) {
        SKM_PROFILE_SCOPE("shift");
        ParallelExecutor& executor = GetExecutor();
        Eigen::Map<const MatrixR> new_mat(horizontal_centroids.get(), n_clusters, d);
        Eigen::Map<const MatrixR> prev_mat(prev_centroids.get(), n_clusters, d);
        std::vector<float> worker_shift(executor.NumWorkers(), 0.0f);
        executor.ParallelFor(n_clusters, [&](size_t begin, size_t end, size_t worker) {
            float local_shift = 0.0f;
            for (size_t i = begin; i < end; ++i) {
                local_shift += (new_mat.row(i) - prev_mat.row(i)).squaredNorm();
            }
            worker_shift[worker] = local_shift;
        });
        shift = std::accumulate(worker_shift.begin(), worker_shift.end(), 0.0f);
    }

    /**
     * @brief Computes ground truth assignments for recall calculation.
     *
     * Finds the top-k nearest data points for each query using exact search.
     * These assignments are used as ground truth for evaluating centroid quality.
     *
     * @param data Data matrix (sampled data points)
     * @param queries Query vectors
     * @param n_queries Number of query vectors
     */
    void GetGTAssignmentsAndDistances(
        const float* SKM_RESTRICT data,
        const float* SKM_RESTRICT queries,
        const size_t n_queries
    ) {
        SKM_PROFILE_SCOPE("gt_assignments");
        std::vector<distance_t> gt_query_norms(n_queries);
        GetL2NormsRowMajor(queries, n_queries, gt_query_norms.data());
        batch_computer::FindKNearestNeighbors(
            GetExecutor(),
            queries,
            data,
            n_queries,
            n_samples,
            d,
            gt_query_norms.data(),
            data_norms.get(),
            config.objective_k,
            gt_assignments.get(),
            gt_distances.get(),
            tmp_distances_buffer.get()
        );
    }

    /**
     * @brief Computes recall@k for current centroids.
     *
     * For each query, checks how many of its ground truth nearest neighbors
     * would be found when searching only the top centroids. Higher recall
     * indicates better centroid quality for ANN indexing.
     *
     * @param queries Query vectors
     * @param n_queries Number of query vectors
     * @return Recall value (0.0 to 1.0)
     */
    float ComputeRecall(const float* SKM_RESTRICT queries, const size_t n_queries) {
        SKM_PROFILE_SCOPE("recall");
        batch_computer::FindKNearestNeighbors(
            GetExecutor(),
            queries,
            horizontal_centroids.get(),
            n_queries,
            n_clusters,
            d,
            query_norms.get(),
            centroid_norms.get(),
            centroids_to_explore,
            promising_centroids.get(),
            recall_distances.get(),
            tmp_distances_buffer.get()
        );
        // For each query, compute recall@objective_k: how many of the GT clusters are found in the
        // top-%centroids_to_explore assignments
        // Recall per query = (# matched GT assignments in top-%centroids_to_explore) / objective_k
        // Final recall = average over all queries
        float sum_recall = 0.0f;
        for (size_t i = 0; i < n_queries; ++i) {
            size_t found_in_query = 0;
            // For each GT assignment for query q
            for (size_t j = 0; j < config.objective_k; ++j) {
                uint32_t gt = gt_assignments[i * config.objective_k + j]; // gt is a vector index
                // Check if this GT assignment is present in the top-%centroids_to_explore
                // assignments for this query
                bool found = false;
                for (size_t t = 0; t < centroids_to_explore; ++t) {
                    // If a centroid is the same as the GT centroid assignment, then we have a match
                    if (promising_centroids[i * centroids_to_explore + t] == assignments[gt]) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    ++found_in_query;
                }
            }
            sum_recall +=
                static_cast<float>(found_in_query) / static_cast<float>(config.objective_k);
        }
        return sum_recall / static_cast<float>(n_queries);
    }

    /**
     * @brief Generates initial centroids from the data.
     *
     * Forgy sampling: Randomly samples n_clusters vectors as initial centroids,
     * rotates them, PDXifies them, and wraps them in a PDXLayout wrapper.
     *
     * @param data Data matrix
     * @param n_clusters Number of centroids to generate
     * @param rotate Wheter to rotate the sampled centroids
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q> GenerateCentroids(
        const float* SKM_RESTRICT data,
        const size_t n_points,
        const size_t n_clusters,
        const bool rotate = true
    ) {
        {
            SKM_PROFILE_SCOPE("generating_centroids");
            auto tmp_centroids_p = horizontal_centroids.get();

            std::mt19937 rng(config.seed);
            std::vector<size_t> indices(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            for (size_t i = 0; i < n_clusters; i += 1) {
                memcpy(
                    (void*) tmp_centroids_p,
                    (void*) (data + (indices[i] * d)),
                    sizeof(centroid_value_t) * d
                );
                tmp_centroids_p += d;
            }
        }
        // We populate the centroids buffer with the centroids in the PDX layout
        std::vector<centroid_value_t> rotated_centroids(n_clusters * d);
        RotateOrCopy(horizontal_centroids.get(), rotated_centroids.data(), n_clusters, rotate);
        if constexpr (q == Quantization::f32) {
            {
                SKM_PROFILE_SCOPE("consolidate/pdxify");
                PDXLayout<q>::template PDXify<false>(
                    rotated_centroids.data(), centroids.get(), n_clusters, d
                );
            }
            //! We wrap centroids and partial_horizontal_centroids in the PDXLayout wrapper
            //! Any updates to these objects is reflected in the PDXLayout
            //! partial_horizontal_centroids are not filled until ConsolidateCentroids is called()
            // after the first iteration
            return PDXLayout<q>(
                centroids.get(), *pruner, n_clusters, d, partial_horizontal_centroids.get()
            );
        }
        return layout_t{};
    }

    /**
     * @brief Computes partial L2 squared norms (first partial_d dimensions).
     */
    void GetPartialL2NormsRowMajor(
        const float* SKM_RESTRICT data,
        const size_t n,
        float* SKM_RESTRICT out_norm,
        const size_t partial_d
    ) {
        SKM_PROFILE_SCOPE("norms_calc");
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    /**
     * @brief Computes full L2 squared norms for each vector.
     */
    void GetL2NormsRowMajor(
        const float* SKM_RESTRICT data,
        const size_t n,
        float* SKM_RESTRICT out_norm
    ) {
        SKM_PROFILE_SCOPE("norms_calc");
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
    }

    void EnsureTmpDistancesBuffer() {
        size_t needed =
            BatchComputer<DistanceFunction::l2, Quantization::f32>::ScratchSize(GetExecutor());
        if constexpr (q != Quantization::f32) {
            // The quantizers' assignment tiles are X_BATCH_SIZE x Y_BATCH_SIZE
            needed = std::max(needed, X_BATCH_SIZE * Y_BATCH_SIZE);
        }
        if (needed > tmp_distances_buffer_size) {
            tmp_distances_buffer.reset(new distance_t[needed]);
            tmp_distances_buffer_size = needed;
        }
    }

    std::unique_ptr<skmeans_quantizer_t<q>> CreateQuantizer() const {
        auto created = std::make_unique<skmeans_quantizer_t<q>>();
        created->SetExecutor(&GetExecutor());
        return created;
    }

    /**
     * @brief Rotates or copies vectors based on rotate parameter.
     *
     * If rotate is false, performs a simple memcpy. Otherwise, applies rotation.
     *
     * @param in Input buffer (potentially unrotated)
     * @param out Output buffer (rotated or copied)
     * @param n_vectors Number of vectors to process
     * @param rotate Whether to rotate (true) or just copy (false)
     */
    void RotateOrCopy(
        const centroid_value_t* SKM_RESTRICT in,
        centroid_value_t* SKM_RESTRICT out,
        const size_t n_vectors,
        const bool rotate
    ) {
        SKM_PROFILE_SCOPE("rotator");
        if (rotate) { // NOLINT(bugprone-branch-clone)
            pruner->Rotate(in, out, n_vectors);
        } else {
            memcpy(
                static_cast<void*>(out),
                static_cast<const void*>(in),
                sizeof(centroid_value_t) * n_vectors * d
            );
        }
    }

    /**
     * @brief Copies the first vertical_d dimensions of centroids for efficient PRUNING
     * TODO(@lkuffo, high): We can avoid this by using the full horizontal
     * centroids in PRUNING
     */
    void CentroidsToAuxiliaryHorizontal(const size_t n_clusters) {
        Eigen::Map<MatrixR> hor_centroids(horizontal_centroids.get(), n_clusters, d);
        Eigen::Map<MatrixR> out_aux_centroids(
            partial_horizontal_centroids.get(), n_clusters, vertical_d
        );
        out_aux_centroids.noalias() = hor_centroids.leftCols(vertical_d);
    }

    /**
     * @brief Copies the first vertical_d dimensions of quantized centroids
     * for the auxiliary horizontal layout used by PDXearch.
     */
    void QuantizedCentroidsToAuxiliaryHorizontal(const size_t n_clusters) {
        const size_t row_stride = PDXDim(d);
        const size_t copy_size = PDXDim(vertical_d);
        for (size_t i = 0; i < n_clusters; ++i) {
            memcpy(
                partial_hor_quantized_centroids.get() + i * copy_size,
                quantized_centroids.get() + i * row_stride,
                copy_size * sizeof(vector_value_t)
            );
        }
    }

    /**
     * @brief Tune partial_d based on the average not-pruned percentage.
     *
     * A safe range for pruning is between 95% - 97% of vectors pruned (i.e., 3% - 5% not pruned).
     * - If avg_not_pruned_pct > 5% (i.e., less than 95% pruned), we reduce partial_d by
     * config.adjustment_factor_for_partial_d to be more aggressive in pruning
     * - If avg_not_pruned_pct < 3% (i.e., more than 97% pruned), we increase partial_d by
     * config.adjustment_factor_for_partial_d to be less aggressive
     * - partial_d is clamped between MIN_PARTIAL_D and vertical_d
     *
     * @param not_pruned_counts Buffer containing per-vector not-pruned counts
     * @param n_samples Number of X vectors
     * @param n_y Number of Y vectors (centroids)
     * @param partial_d_changed Output parameter: set to true if partial_d was changed
     * @return The computed average not-pruned percentage
     */
    float TunePartialD(
        const size_t* not_pruned_counts,
        size_t n_samples,
        size_t n_y,
        bool& partial_d_changed
    ) {
        float avg_not_pruned_pct = 0.0f;
        for (size_t i = 0; i < n_samples; ++i) {
            avg_not_pruned_pct += static_cast<float>(not_pruned_counts[i]);
        }
        avg_not_pruned_pct /= static_cast<float>(n_samples) * static_cast<float>(n_y);

        uint32_t old_partial_d = partial_d;
        if (avg_not_pruned_pct > config.max_not_pruned_pct) {
            // Too many vectors not pruned (< max_not_pruned_pct pruned), need more GEMM dimensions
            // Increase partial_d by adjustment_factor_for_partial_d * 2
            auto increase =
                static_cast<uint32_t>(partial_d * config.adjustment_factor_for_partial_d * 2);
            partial_d = std::min(partial_d + std::max(increase, 1u), vertical_d);
        } else if (avg_not_pruned_pct < config.min_not_pruned_pct) {
            // Too few vectors not pruned (> min_not_pruned_pct pruned), can reduce GEMM dimensions
            // Decrease partial_d by adjustment_factor_for_partial_d
            auto decrease =
                static_cast<uint32_t>(partial_d * config.adjustment_factor_for_partial_d);
            partial_d = std::max(partial_d - std::max(decrease, 1u), MIN_PARTIAL_D);
        }
        partial_d = quantizer->AlignPartialD(partial_d, vertical_d);
        partial_d_changed = (old_partial_d != partial_d);
        return avg_not_pruned_pct;
    }

    /**
     * @brief Computes the number of vectors to sample based on sampling_fraction.
     * To be conservative, we implement two heuristics:
     * - We sample at most max_points_per_cluster points per cluster (FAISS style)
     * - We sample at most sampling_fraction * n points (our style)
     * We return the minimum of the two.
     * @param n Total number of vectors
     * @return Number of vectors to sample
     */
    [[nodiscard]] virtual size_t GetNVectorsToSample(const size_t n, size_t n_clusters) const {
        if (config.sampling_fraction == 1.0) {
            return n;
        }
        auto samples_byn_clusters = n_clusters * config.max_points_per_cluster;
        auto samples_by_n =
            static_cast<size_t>(std::floor(static_cast<float>(n) * config.sampling_fraction));
        return std::min(samples_by_n, samples_byn_clusters);
    }

    /**
     * @brief Check if the core loop should stop early based on convergence criteria.
     *
     * Convergence is detected when either:
     * - Shift is below tolerance (shift < config.tol) or
     * - Recall hasn't improved by more than config.recall_tol in RECALL_CONVERGENCE_PATIENCE
     * consecutive iterations (when tracking recall)
     *
     * @param tracking_recall Whether recall is being tracked (n_queries > 0)
     * @param best_recall Reference to the best recall seen so far (updated if current is better)
     * @param iters_without_improvement Reference to counter of iterations without recall
     * improvement
     * @param iter_idx Current iteration index
     * @return true if training should stop, false otherwise
     */
    bool ShouldStopEarly(
        const bool tracking_recall,
        float& best_recall,
        size_t& iters_without_improvement,
        const size_t iter_idx
    ) {
        if (shift < config.tol) {
            if (config.verbose)
                std::cout << "Converged at iteration " << iter_idx + 1 << " (shift " << shift
                          << " < tol " << config.tol << ")" << std::endl;
            return true;
        }
        if (iter_idx > 0) {
            auto cost_delta = cost / prev_cost;
            if (cost_delta > 1 - config.tol) {
                if (config.verbose)
                    std::cout << "Converged at iteration " << iter_idx + 1 << " (cost "
                              << " improved by only " << 1 - cost_delta << ")" << std::endl;
                return true;
            }
        }
        if (tracking_recall) {
            float improvement = recall - best_recall;
            if (improvement > config.recall_tol) {
                best_recall = recall;
                iters_without_improvement = 0;
            } else {
                iters_without_improvement++;
                if (iters_without_improvement >= RECALL_CONVERGENCE_PATIENCE) {
                    if (config.verbose)
                        std::cout << "Converged at iteration " << iter_idx + 1 << " (recall "
                                  << recall << " hasn't improved by more than " << config.recall_tol
                                  << " in " << RECALL_CONVERGENCE_PATIENCE
                                  << " iterations, best: " << best_recall << ")" << std::endl;
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Unrotate centroids to original space.
     * @param should_unrotate If true, unrotates centroids to original space, if false, returns
     * rotated centroids.
     * @return Centroids
     */
    std::vector<centroid_value_t> GetOutputCentroids(bool should_unrotate) {
        if (should_unrotate) {
            SKM_PROFILE_SCOPE("unrotator");
            std::vector<centroid_value_t> unrotated(n_clusters * d);
            pruner->Unrotate(horizontal_centroids.get(), unrotated.data(), n_clusters);
            return unrotated;
        }
        return std::vector<centroid_value_t>(
            horizontal_centroids.get(), horizontal_centroids.get() + n_clusters * d
        );
    }

    /**
     * @brief Normalizes centroids to unit length for inner product distance.
     *
     */
    void PostprocessCentroids(const size_t n_clusters) {
        auto horizontal_centroids_p = horizontal_centroids.get();
        GetExecutor().ParallelFor(n_clusters, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                auto horizontal_centroids_p_i = horizontal_centroids_p + i * d;
                float sum = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    sum += horizontal_centroids_p_i[j] * horizontal_centroids_p_i[j];
                }
                float norm = 1.0f / std::sqrt(sum);
                for (size_t j = 0; j < d; ++j) {
                    horizontal_centroids_p_i[j] *= norm;
                }
            }
        });
    }

    /**
     * @brief Reconfigures SuperKMeans for in-place training.
     *
     * data_already_rotated is set so that every downstream path treats it as such
     * unrotate_centroids is cleared so the returned centroids stay in the same domain as the
     * caller's buffer.
     *
     * @param data Data matrix to rotate in place (row-major, n × d)
     * @param n Number of vectors
     */
    void ConfigInPlaceTraining(float* SKM_RESTRICT data, const size_t n) {
        if (config.sampling_fraction != 1.0f) {
            if (!config.suppress_warnings) {
                std::cout << "WARNING: in-place training requires sampling_fraction == 1.0, "
                             "overriding it (was "
                          << config.sampling_fraction << ")" << std::endl;
            }
            config.sampling_fraction = 1.0f;
        }
        if (config.data_already_rotated) {
            return;
        }
        {
            SKM_PROFILE_SCOPE("rotator");
            pruner->template Rotate<true>(data, data, static_cast<uint32_t>(n));
        }
        config.data_already_rotated = true;
        config.unrotate_centroids = false;
        state.trained_in_place = true;
    }

    /**
     * @brief Samples and optionally rotates vectors for training.
     *
     * Performs random sampling without replacement using shuffle technique,
     * and optionally rotates the sampled vectors using the pruner's rotation matrix.
     *
     * @tparam ROTATE Whether to apply rotation (default true)
     * @param data Input data matrix
     * @param out_buffer Output buffer for sampled (and optionally rotated) vectors
     * @param n Total number of input vectors
     * @param n_samples Number of vectors to sample
     */
    const float* SampleAndRotateVectors(
        const float* SKM_RESTRICT data,
        float* SKM_RESTRICT out,
        const size_t n,
        const size_t n_samples,
        const bool rotate = true
    ) {
        ParallelExecutor& executor = GetExecutor();
        // Intermediate buffer needed only when both sampling and rotating
        // (we have not yet implemented rotation in-place)
        std::unique_ptr<float[]> samples_tmp;
        const float* src_data = data;

        if (n_samples < n) {
            if (config.verbose)
                std::cout << "Sampling " << n_samples << " vectors" << std::endl;
            SKM_PROFILE_SCOPE("sampling");
            // Random sampling without replacement
            std::mt19937 rng(config.seed);
            sampled_indices.reset(new size_t[n]);
            for (size_t i = 0; i < n; ++i) {
                sampled_indices[i] = i;
            }
            std::shuffle(sampled_indices.get(), sampled_indices.get() + n, rng);
            if (rotate) {
                samples_tmp.reset(new float[n_samples * d]);
                // Need intermediate buffer: sample first, then rotate
                executor.ParallelFor(n_samples, [&](size_t begin, size_t end, size_t) {
                    for (size_t i = begin; i < end; ++i) {
                        memcpy(
                            static_cast<void*>(samples_tmp.get() + i * d),
                            static_cast<const void*>(data + sampled_indices[i] * d),
                            sizeof(float) * d
                        );
                    }
                });
                src_data = samples_tmp.get();
            } else {
                // No rotation: copy directly into output buffer
                executor.ParallelFor(n_samples, [&](size_t begin, size_t end, size_t) {
                    for (size_t i = begin; i < end; ++i) {
                        memcpy(
                            static_cast<void*>(out + i * d),
                            static_cast<const void*>(data + sampled_indices[i] * d),
                            sizeof(float) * d
                        );
                    }
                });
                return out;
            }
        }
        if (config.verbose)
            std::cout << "Using " << n_samples << " vectors" << std::endl;

        // No rotation and no sampling: use the original data directly (avoid redundant memcpy)
        if (!rotate) {
            return data;
        }

        RotateOrCopy(src_data, out, n_samples, rotate);
        return out;
    }

    const size_t d;
    const size_t n_clusters;
    SuperKMeansConfig config;

    size_t n_samples = 0;
    size_t n_train = 0;
    uint32_t partial_d = 0; // d'

    // Iteration state
    SuperKMeansState state;
    size_t n_split = 0;
    size_t centroids_to_explore = 0;
    uint32_t vertical_d = 0;
    float prev_cost = 0.0f;
    float cost = 0.0f;
    float shift = 0.0f;
    float recall = 0.0f;

    std::unique_ptr<pruner_t> pruner;
    std::unique_ptr<ParallelExecutor> owned_executor; // the default, when config.executor is null

    // Centroids data (unoptimized space)
    // TODO(@lkuffo, high): 3 copies of the centroids? Can we do better?
    //    We can trivially avoid partial_horizontal_centroids by using the full horizontal
    //    centroids in PDXearch. We can also avoid prev_centroids if we dont care about the shift
    //    convergence check.
    std::unique_ptr<centroid_value_t[]> centroids;            // PDX-layout centroids
    std::unique_ptr<centroid_value_t[]> horizontal_centroids; // Row-major centroids
    std::unique_ptr<centroid_value_t[]> prev_centroids;       // Previous iteration centroids
    std::unique_ptr<centroid_value_t[]> partial_horizontal_centroids; // First vertical_d dimensions

    // Buffers for assignment and distance computation
    std::unique_ptr<distance_t[]> distances;
    std::unique_ptr<uint32_t[]> cluster_sizes;
    std::unique_ptr<float[]> data_norms;
    std::unique_ptr<float[]> centroid_norms;

    // Quantization state
    std::unique_ptr<skmeans_quantizer_t<q>> quantizer;
    F32Quantizer f32_quantizer; // float centroid updates (quantized_centroid_update off)
    size_t code_size = 0; // bytes per encoded vector (= d for SQ8, variable for RaBitQ)
    std::unique_ptr<vector_value_t[]> quantized_data;
    std::unique_ptr<vector_value_t[]> quantized_centroids;
    std::unique_ptr<vector_value_t[]> pdxified_quantized_centroids;
    std::unique_ptr<vector_value_t[]> partial_hor_quantized_centroids;

    // Buffers for ground truth and recall computation
    std::unique_ptr<uint32_t[]> gt_assignments;
    std::unique_ptr<distance_t[]> gt_distances;
    std::unique_ptr<distance_t[]> query_norms;
    std::unique_ptr<distance_t[]> tmp_distances_buffer;
    size_t tmp_distances_buffer_size = 0;
    std::unique_ptr<uint32_t[]> promising_centroids;
    std::unique_ptr<distance_t[]> recall_distances;

  public:
    std::unique_ptr<uint32_t[]> assignments;
    std::unique_ptr<size_t[]> sampled_indices;
    std::vector<SuperKMeansIterationStats> iteration_stats;
};

using SuperKMeansSQ8 = SuperKMeans<Quantization::sq8>;
using SuperKMeansLVQ4 = SuperKMeans<Quantization::lvq4>;
using SuperKMeansRabitQ = SuperKMeans<Quantization::rabitq>;
} // namespace skmeans
