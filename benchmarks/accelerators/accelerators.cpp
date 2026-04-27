#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

#include <Eigen/Dense>
#include <faiss/VectorTransform.h>

using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SKM_f32 = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool ParseBool(const char* s) {
    return std::string(s) == "true";
}

/**
 * @brief Generate a random orthonormal projection matrix P of size (target_d x d).
 *
 * Draws a d x d random Gaussian matrix, QR-decomposes it, and takes the first
 * target_d rows. The resulting rows are orthonormal, so P * P^T = I_{target_d}
 * and the pseudo-inverse of P^T is P itself.
 */
static MatrixR GenerateJLProjectionMatrix(size_t d, size_t target_d, uint32_t seed) {
    MatrixR random_matrix(d, d);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            random_matrix(i, j) = dist(gen);
        }
    }
    const Eigen::HouseholderQR<MatrixR> qr(random_matrix);
    MatrixR Q = qr.householderQ() * MatrixR::Identity(d, d);
    return Q.topRows(target_d);
}

/**
 * @brief Recompute full-d centroids as the mean of original vectors per cluster.
 */
static void RecomputeFullDCentroids(
    const float* original_data,
    const std::vector<uint32_t>& assignments,
    float* out_centroids,
    size_t n,
    size_t d,
    size_t n_clusters
) {
    std::memset(out_centroids, 0, n_clusters * d * sizeof(float));
    std::vector<size_t> cluster_sizes(n_clusters, 0);

    for (size_t i = 0; i < n; ++i) {
        const uint32_t c = assignments[i];
        cluster_sizes[c]++;
        for (size_t j = 0; j < d; ++j) {
            out_centroids[c * d + j] += original_data[i * d + j];
        }
    }
    for (size_t c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 0) {
            const float inv = 1.0f / static_cast<float>(cluster_sizes[c]);
            for (size_t j = 0; j < d; ++j) {
                out_centroids[c * d + j] *= inv;
            }
        }
    }
}

/**
 * @brief Build config_dict for CSV output.
 */
static std::unordered_map<std::string, std::string> BuildConfigDict(
    const std::string& dim_reduction,
    const std::string& quantizer,
    const std::string& unproject_centroids,
    size_t target_d,
    const skmeans::SuperKMeansConfig& cfg
) {
    std::unordered_map<std::string, std::string> c;
    // Pipeline-specific columns
    c["dim_reduction"] = "\"" + dim_reduction + "\"";
    c["quantizer"] = "\"" + quantizer + "\"";
    c["unproject_centroids"] = "\"" + unproject_centroids + "\"";
    c["target_d"] = std::to_string(target_d);
    // Algorithm config
    c["iters"] = std::to_string(cfg.iters);
    c["sampling_fraction"] = std::to_string(cfg.sampling_fraction);
    c["n_threads"] = std::to_string(cfg.n_threads);
    c["seed"] = std::to_string(cfg.seed);
    c["use_blas_only"] = cfg.use_blas_only ? "true" : "false";
    c["tol"] = std::to_string(cfg.tol);
    c["early_termination"] = cfg.early_termination ? "true" : "false";
    c["quantized_centroid_update"] = cfg.quantized_centroid_update ? "true" : "false";
    c["full_precision_final_centroids"] = cfg.full_precision_final_centroids ? "true" : "false";
    c["angular"] = cfg.angular ? "true" : "false";
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core pipeline
// ─────────────────────────────────────────────────────────────────────────────

template <skmeans::Quantization Q>
void RunPipeline(
    const std::string& dataset,
    const std::string& dim_reduction,
    const std::string& quantizer_name,
    bool quantized_centroid_update,
    bool full_precision_final_centroids,
    bool use_blas_only
) {
    using SKM = skmeans::SuperKMeans<Q, skmeans::DistanceFunction::l2>;

    // ── Dataset params ──
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    const bool has_quantizer = (quantizer_name != "f32");
    const bool is_raw = (dim_reduction == "raw");
    const std::string experiment_name =
        "accelerators_" + dim_reduction + "_" + quantizer_name;
    const std::string algorithm = "superkmeans";

    std::cout << "=== Accelerators Benchmark ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;
    std::cout << "Dim reduction: " << dim_reduction
              << ", Quantizer: " << quantizer_name << std::endl;
    std::cout << "quantized_centroid_update=" << quantized_centroid_update
              << " full_precision_final_centroids=" << full_precision_final_centroids
              << " use_blas_only=" << use_blas_only << std::endl;
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " threads=" << THREADS << std::endl;

    // ── Load data and queries ──
    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

    {
        std::ifstream f(bench_utils::get_data_path(dataset), std::ios::binary);
        if (!f) { std::cerr << "Failed to open data file\n"; return; }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(bench_utils::get_query_path(dataset), std::ios::binary);
        if (!f) { std::cerr << "Failed to open query file\n"; return; }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    // ── Angular detection ──
    bool is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(),
        bench_utils::ANGULAR_DATASETS.end(), dataset
    ) != bench_utils::ANGULAR_DATASETS.end();

    // ── Ground truth ──
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    bool has_gt = std::ifstream(gt_filename).good();
    std::unordered_map<int, std::vector<int>> gt_map;
    if (has_gt) {
        gt_map = bench_utils::parse_ground_truth_json(gt_filename);
    }

    // ── Determine target_d values to iterate ──
    std::vector<size_t> target_d_list;
    if (is_raw) {
        target_d_list = {d};
    } else {
        for (size_t td : bench_utils::TARGET_D_VALUES) {
            if (td < d) target_d_list.push_back(td);
        }
        if (target_d_list.empty()) {
            std::cerr << "No valid TARGET_D values for d=" << d
                      << " (min is 64, d must be > 64)\n";
            return;
        }
    }

    // ── Lambda: compute recall, WCSS, balance, and write CSV row ──
    auto write_row = [&](
        size_t target_d,
        double construction_time_ms,
        int actual_iterations,
        const std::vector<skmeans::SuperKMeansIterationStats>& iter_stats,
        const std::vector<uint32_t>& assignments,
        const std::vector<uint32_t>& q_assignments,
        const float* full_d_centroids,
        const std::string& unproject_str,
        const skmeans::SuperKMeansConfig& cfg
    ) {
        // Compute WCSS in full-d space
        double wcss_assign = SKM_f32::ComputeWCSS(
            data.data(), full_d_centroids, assignments.data(), n, d
        );
        std::cout << "WCSS (Assign): " << std::fixed << std::setprecision(2)
                  << wcss_assign << std::endl;

        if (!q_assignments.empty()) {
            double wcss_q = SKM_f32::ComputeWCSS(
                data.data(), full_d_centroids, q_assignments.data(), n, d
            );
            std::cout << "WCSS (QuantizedAssign): " << std::fixed << std::setprecision(2)
                      << wcss_q << std::endl;
        }

        // Balance stats (from Assign)
        auto balance_stats = SKM_f32::GetClustersBalanceStats(
            assignments.data(), n, n_clusters
        );
        balance_stats.print();

        // Iteration stats JSON
        std::string iter_stats_json =
            skmeans::SuperKMeansIterationStats::vector_to_json(iter_stats);

        // Recall computation
        bench_utils::recall_results_t assign_r10, assign_r100;
        bench_utils::recall_results_t q_assign_r10, q_assign_r100;

        if (has_gt) {
            assign_r10 = bench_utils::compute_recall(
                gt_map, assignments, queries.data(), full_d_centroids,
                n_queries, n_clusters, d, 10
            );
            assign_r100 = bench_utils::compute_recall(
                gt_map, assignments, queries.data(), full_d_centroids,
                n_queries, n_clusters, d, 100
            );
            std::cout << "  [Assign()]" << std::endl;
            bench_utils::print_recall_results(assign_r10, 10);
            bench_utils::print_recall_results(assign_r100, 100);

            if (!q_assignments.empty()) {
                q_assign_r10 = bench_utils::compute_recall(
                    gt_map, q_assignments, queries.data(), full_d_centroids,
                    n_queries, n_clusters, d, 10
                );
                q_assign_r100 = bench_utils::compute_recall(
                    gt_map, q_assignments, queries.data(), full_d_centroids,
                    n_queries, n_clusters, d, 100
                );
                std::cout << "  [QuantizedAssign()]" << std::endl;
                bench_utils::print_recall_results(q_assign_r10, 10);
                bench_utils::print_recall_results(q_assign_r100, 100);
            }
        }

        auto config_dict = BuildConfigDict(
            is_raw ? "none" : dim_reduction,
            quantizer_name,
            unproject_str,
            target_d,
            cfg
        );

        bench_utils::write_results_to_csv_v2(
            experiment_name, algorithm, dataset,
            n_iters, actual_iterations,
            static_cast<int>(target_d),
            n, static_cast<int>(n_clusters),
            construction_time_ms,
            static_cast<int>(THREADS),
            wcss_assign,
            config_dict,
            assign_r10, assign_r100,
            q_assign_r10, q_assign_r100,
            balance_stats.to_json(),
            iter_stats_json
        );
    };

    // ── Main loop over target dimensions ──
    for (size_t target_d : target_d_list) {
        const size_t working_d = target_d;

        std::cout << "\n════════════════════════════════════════════════" << std::endl;
        std::cout << "target_d = " << target_d << " / " << d << std::endl;
        std::cout << "════════════════════════════════════════════════" << std::endl;

        // ── Phase 1: Preprocessing ──
        const float* working_data = nullptr;
        std::vector<float> projected_data;
        MatrixR jl_matrix;
        std::unique_ptr<faiss::PCAMatrix> pca_holder;

        bench_utils::TicToc preprocess_timer;
        preprocess_timer.Tic();

        if (is_raw) {
            working_data = data.data();
        } else if (dim_reduction == "pca") {
            std::cout << "Training PCA (" << d << " -> " << target_d << ")..." << std::endl;
            pca_holder = std::make_unique<faiss::PCAMatrix>(d, target_d, 0, false);
            pca_holder->train(n, data.data());

            projected_data.resize(n * target_d);
            pca_holder->apply_noalloc(n, data.data(), projected_data.data());
            working_data = projected_data.data();
        } else if (dim_reduction == "jlt") {
            std::cout << "Generating JLT matrix (" << target_d << " x " << d
                      << ")..." << std::endl;
            jl_matrix = GenerateJLProjectionMatrix(d, target_d, 42);

            projected_data.resize(n * target_d);
            Eigen::Map<const MatrixR> data_mat(data.data(), n, d);
            Eigen::Map<MatrixR> proj_mat(projected_data.data(), n, target_d);
            proj_mat.noalias() = data_mat * jl_matrix.transpose();
            working_data = projected_data.data();
        }

        preprocess_timer.Toc();
        if (!is_raw) {
            std::cout << "Preprocessing completed in "
                      << preprocess_timer.GetMilliseconds() << " ms" << std::endl;
        }

        // ── Phase 2: Configure and Train ──
        skmeans::SuperKMeansConfig config;
        config.iters = n_iters;
        config.verbose = true;
        config.verbose_detail = true;
        config.n_threads = THREADS;
        config.unrotate_centroids = true;
        config.early_termination = false;
        config.sampling_fraction = 1.0f;
        config.tol = 1e-3f;
        config.use_blas_only = use_blas_only;
        config.quantized_centroid_update = quantized_centroid_update;
        config.full_precision_final_centroids = full_precision_final_centroids;

        if (quantizer_name == "sq8")
            config.quantizer_type = skmeans::QuantizerType::sq8;
        else if (quantizer_name == "sq4")
            config.quantizer_type = skmeans::QuantizerType::sq4;
        else if (quantizer_name == "rabitq")
            config.quantizer_type = skmeans::QuantizerType::rabitq;

        if (is_angular) {
            std::cout << "Using spherical k-means" << std::endl;
            config.angular = true;
        }

        auto kmeans = SKM(n_clusters, working_d, config);

        bench_utils::TicToc train_timer;
        train_timer.Tic();
        std::vector<float> working_centroids = kmeans.Train(working_data, n);
        train_timer.Toc();
        double construction_time_ms = train_timer.GetMilliseconds();
        int actual_iterations = static_cast<int>(kmeans.iteration_stats.size());

        std::cout << "Training completed in " << construction_time_ms << " ms" << std::endl;
        std::cout << "Actual iterations: " << actual_iterations << std::endl;

        // ── Phase 3 & 4: Final Assignments + CSV Output ──

        // Full-d assign helper (reused across rows)
        skmeans::SuperKMeansConfig assign_config;
        assign_config.n_threads = THREADS;
        assign_config.use_blas_only = true;
        auto kmeans_fulld = SKM_f32(n_clusters, d, assign_config);

        if (is_raw) {
            // ── RAW: single row ──
            std::cout << "\n--- Final Assignments (raw) ---" << std::endl;
            auto assignments = kmeans.Assign(
                data.data(), working_centroids.data(), n, n_clusters
            );

            std::vector<uint32_t> q_assignments;
            if (has_quantizer) {
                q_assignments = kmeans.QuantizedAssign(
                    data.data(), working_centroids.data(), n, n_clusters
                );
            }

            write_row(
                target_d, construction_time_ms, actual_iterations,
                kmeans.iteration_stats,
                assignments, q_assignments,
                working_centroids.data(), "none", config
            );

        } else {
            // ── PREPROCESSING: two rows per target_d ──

            // --- Row 1: UNPROJECT=true ---
            std::cout << "\n--- UNPROJECT=true ---" << std::endl;
            std::vector<float> full_d_centroids(n_clusters * d);

            if (dim_reduction == "pca") {
                pca_holder->reverse_transform(
                    n_clusters, working_centroids.data(), full_d_centroids.data()
                );
            } else { // jlt: c_full = c_proj * P
                Eigen::Map<const MatrixR> proj_c(
                    working_centroids.data(), n_clusters, target_d
                );
                Eigen::Map<MatrixR> full_c(full_d_centroids.data(), n_clusters, d);
                full_c.noalias() = proj_c * jl_matrix;
            }

            auto assignments_unproj = kmeans_fulld.Assign(
                data.data(), full_d_centroids.data(), n, n_clusters
            );

            // Requantize in full-d: fit quantizer on full-d data (1 cheap iter),
            // then QuantizedAssign with unprojected centroids
            std::vector<uint32_t> q_assignments_unproj;
            if (has_quantizer) {
                skmeans::SuperKMeansConfig refit_cfg;
                refit_cfg.iters = 1;
                refit_cfg.n_threads = THREADS;
                refit_cfg.use_blas_only = true;
                refit_cfg.verbose = false;
                refit_cfg.quantizer_type = config.quantizer_type;

                auto kmeans_refit = SKM(n_clusters, d, refit_cfg);
                kmeans_refit.Train(data.data(), n);

                q_assignments_unproj = kmeans_refit.QuantizedAssign(
                    data.data(), full_d_centroids.data(), n, n_clusters
                );
            }

            write_row(
                target_d, construction_time_ms, actual_iterations,
                kmeans.iteration_stats,
                assignments_unproj, q_assignments_unproj,
                full_d_centroids.data(), "true", config
            );

            // --- Row 2: UNPROJECT=false ---
            std::cout << "\n--- UNPROJECT=false ---" << std::endl;

            // Assign in projected space to get cluster membership
            auto projected_assignments = kmeans.Assign(
                projected_data.data(), working_centroids.data(), n, n_clusters
            );

            // Recompute full-d centroids from original vectors
            std::vector<float> recomputed_centroids(n_clusters * d);
            RecomputeFullDCentroids(
                data.data(), projected_assignments,
                recomputed_centroids.data(), n, d, n_clusters
            );

            // Re-assign in full-d with recomputed centroids
            auto assignments_nounproj = kmeans_fulld.Assign(
                data.data(), recomputed_centroids.data(), n, n_clusters
            );

            // QuantizedAssign in projected space (if quantizer available)
            std::vector<uint32_t> q_assignments;
            if (has_quantizer) {
                q_assignments = kmeans.QuantizedAssign(
                    projected_data.data(), working_centroids.data(), n, n_clusters
                );
            }

            write_row(
                target_d, construction_time_ms, actual_iterations,
                kmeans.iteration_stats,
                assignments_nounproj, q_assignments,
                recomputed_centroids.data(), "false", config
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main: parse CLI args and dispatch to templated pipeline
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <pca|jlt|raw> <f32|sq8|sq4|rabitq>"
                  << " <quantized_centroid_update=true|false>"
                  << " <full_precision_final_centroids=true|false>"
                  << " <use_blas_only=true|false>"
                  << std::endl;
        return 1;
    }

    const std::string dataset = argv[1];
    const std::string dim_reduction = argv[2];
    const std::string quantizer = argv[3];
    const bool quantized_centroid_update = ParseBool(argv[4]);
    const bool full_precision_final_centroids = ParseBool(argv[5]);
    const bool use_blas_only = ParseBool(argv[6]);

    // Validate dim_reduction
    if (dim_reduction != "raw" && dim_reduction != "pca" && dim_reduction != "jlt") {
        std::cerr << "Invalid dim_reduction: " << dim_reduction
                  << " (expected: raw, pca, jlt)\n";
        return 1;
    }

    // Validate quantizer and dispatch to correct template instantiation
    if (quantizer == "f32") {
        RunPipeline<skmeans::Quantization::f32>(
            dataset, dim_reduction, quantizer,
            quantized_centroid_update, full_precision_final_centroids, use_blas_only
        );
    } else if (quantizer == "sq8") {
        RunPipeline<skmeans::Quantization::u8>(
            dataset, dim_reduction, quantizer,
            quantized_centroid_update, full_precision_final_centroids, use_blas_only
        );
    } else if (quantizer == "sq4") {
        RunPipeline<skmeans::Quantization::u4>(
            dataset, dim_reduction, quantizer,
            quantized_centroid_update, full_precision_final_centroids, use_blas_only
        );
    } else if (quantizer == "rabitq") {
        RunPipeline<skmeans::Quantization::u8>(
            dataset, dim_reduction, quantizer,
            quantized_centroid_update, full_precision_final_centroids, use_blas_only
        );
    } else {
        std::cerr << "Invalid quantizer: " << quantizer
                  << " (expected: f32, sq8, sq4, rabitq)\n";
        return 1;
    }

    return 0;
}
