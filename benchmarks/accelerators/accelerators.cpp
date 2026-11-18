#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

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
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

#include <Eigen/Dense>
#include <faiss/VectorTransform.h>

using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SKM_f32 = skmeans::SuperKMeans<>;

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
    const skmeans::SuperKMeansConfig& cfg,
    double preprocessing_time_ms
) {
    std::unordered_map<std::string, std::string> c;
    // Pipeline-specific columns
    c["dim_reduction"] = "\"" + dim_reduction + "\"";
    c["quantizer"] = "\"" + quantizer + "\"";
    c["unproject_centroids"] = "\"" + unproject_centroids + "\"";
    c["target_d"] = std::to_string(target_d);
    c["preprocessing_time_ms"] = std::to_string(preprocessing_time_ms);
    // Algorithm config
    c["iters"] = std::to_string(cfg.iters);
    c["sampling_fraction"] = std::to_string(cfg.sampling_fraction);
    c["n_threads"] = std::to_string(cfg.n_threads);
    c["seed"] = std::to_string(cfg.seed);
    c["pruning"] = cfg.pruning ? "true" : "false";
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
    bool pruning
) {
    using SKM = skmeans::SuperKMeans<Q>;

    skmeans::Profiler::Get().Reset();

    // ── Dataset params ──
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    const int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    const bool has_quantizer = (quantizer_name != "f32");
    const bool is_raw = (dim_reduction == "raw");
    const std::string experiment_name = "accelerators_" + dim_reduction + "_" + quantizer_name;
    const std::string algorithm = "superkmeans";

    std::cout << "=== Accelerators Benchmark ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;
    std::cout << "Dim reduction: " << dim_reduction << ", Quantizer: " << quantizer_name
              << std::endl;
    std::cout << "quantized_centroid_update=" << quantized_centroid_update
              << " full_precision_final_centroids=" << full_precision_final_centroids
              << " pruning=" << pruning << std::endl;
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters << " threads=" << THREADS
              << std::endl;

    // ── Load data and queries ──
    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

    {
        std::ifstream f(bench_utils::GetDataPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open data file\n";
            return;
        }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(bench_utils::GetQueryPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open query file\n";
            return;
        }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    // ── Angular detection ──
    bool is_angular =
        std::find(
            bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
        ) != bench_utils::ANGULAR_DATASETS.end();

    // ── Ground truth ──
    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    bool has_gt = std::ifstream(gt_filename).good();
    std::unordered_map<int, std::vector<int>> gt_map;
    if (has_gt) {
        gt_map = bench_utils::ParseGroundTruthJson(gt_filename);
    }

    // ── Determine target_d values to iterate ──
    std::vector<size_t> target_d_list;
    if (is_raw) {
        target_d_list = {d};
    } else {
        for (size_t td : bench_utils::TARGET_D_VALUES) {
            if (td < d)
                target_d_list.push_back(td);
        }
        if (target_d_list.empty()) {
            std::cerr << "No valid TARGET_D values for d=" << d << " (min is 32, d must be > 32)\n";
            return;
        }
    }

    // ── Lambda: compute recall, WCSS, balance, and write CSV row ──
    auto write_row = [&](size_t target_d,
                         double construction_time_ms,
                         double preprocessing_time_ms,
                         int actual_iterations,
                         const std::vector<skmeans::SuperKMeansIterationStats>& iter_stats,
                         const std::vector<uint32_t>& assignments,
                         const std::vector<uint32_t>& q_assignments,
                         const float* full_d_centroids,
                         const std::string& unproject_str,
                         const skmeans::SuperKMeansConfig& cfg,
                         const std::string& profiler_train_json,
                         const std::string& profiler_assign_json,
                         const std::string& profiler_q_assign_json,
                         const float* eval_data,
                         const float* eval_queries,
                         size_t eval_d) {
        // Compute WCSS in the eval space (full-d for rows 1/2, projected-d for row 3)
        double wcss_assign =
            SKM_f32::ComputeWCSS(eval_data, full_d_centroids, assignments.data(), n, eval_d);
        std::cout << "WCSS (Assign): " << std::fixed << std::setprecision(2) << wcss_assign
                  << std::endl;

        double wcss_q_assign = -1.0;
        if (!q_assignments.empty()) {
            wcss_q_assign =
                SKM_f32::ComputeWCSS(eval_data, full_d_centroids, q_assignments.data(), n, eval_d);
            std::cout << "WCSS (QuantizedAssign): " << std::fixed << std::setprecision(2)
                      << wcss_q_assign << std::endl;
        }

        // Balance stats (from Assign)
        auto balance_stats = SKM_f32::GetClustersBalanceStats(assignments.data(), n, n_clusters);
        balance_stats.print();

        // Balance stats (from QuantizedAssign), if available
        std::string q_balance_stats_json;
        if (!q_assignments.empty()) {
            auto q_balance_stats =
                SKM_f32::GetClustersBalanceStats(q_assignments.data(), n, n_clusters);
            q_balance_stats.print();
            q_balance_stats_json = q_balance_stats.ToJson();
        }

        // Iteration stats JSON
        std::string iter_stats_json = skmeans::SuperKMeansIterationStats::VectorToJson(iter_stats);

        // Recall computation
        bench_utils::recall_results_t assign_r10, assign_r100;
        bench_utils::recall_results_t q_assign_r10, q_assign_r100;

        if (has_gt) {
            assign_r10 = bench_utils::ComputeRecall(
                gt_map,
                assignments,
                eval_queries,
                full_d_centroids,
                n_queries,
                n_clusters,
                eval_d,
                10
            );
            assign_r100 = bench_utils::ComputeRecall(
                gt_map,
                assignments,
                eval_queries,
                full_d_centroids,
                n_queries,
                n_clusters,
                eval_d,
                100
            );
            std::cout << "  [Assign()]" << std::endl;
            bench_utils::PrintRecallResults(assign_r10, 10);
            bench_utils::PrintRecallResults(assign_r100, 100);

            if (!q_assignments.empty()) {
                q_assign_r10 = bench_utils::ComputeRecall(
                    gt_map,
                    q_assignments,
                    eval_queries,
                    full_d_centroids,
                    n_queries,
                    n_clusters,
                    eval_d,
                    10
                );
                q_assign_r100 = bench_utils::ComputeRecall(
                    gt_map,
                    q_assignments,
                    eval_queries,
                    full_d_centroids,
                    n_queries,
                    n_clusters,
                    eval_d,
                    100
                );
                std::cout << "  [QuantizedAssign()]" << std::endl;
                bench_utils::PrintRecallResults(q_assign_r10, 10);
                bench_utils::PrintRecallResults(q_assign_r100, 100);
            }
        }

        auto config_dict = BuildConfigDict(
            is_raw ? "none" : dim_reduction,
            quantizer_name,
            unproject_str,
            target_d,
            cfg,
            preprocessing_time_ms
        );
        config_dict["wcss_assign"] = std::to_string(wcss_assign);
        if (wcss_q_assign >= 0.0) {
            config_dict["wcss_quantized_assign"] = std::to_string(wcss_q_assign);
        }
        config_dict["profiler"] = profiler_train_json;
        config_dict["profiler_assign"] = profiler_assign_json;
        if (!profiler_q_assign_json.empty()) {
            config_dict["profiler_quantized_assign"] = profiler_q_assign_json;
        }

        // Build human-readable run label (matches accelerators.sh echo strings)
        std::string dim_label = is_raw ? "raw" : dim_reduction;
        std::string run_label;
        {
            std::string quant_label = quantizer_name;
            std::string update_label;
            if (quantizer_name == "f32") {
                update_label = "";
            } else if (cfg.quantized_centroid_update && cfg.full_precision_final_centroids) {
                update_label = " / quant-update + full-prec-final";
            } else if (cfg.quantized_centroid_update) {
                update_label = " / quant-update";
            } else {
                update_label = " / no-quant-update";
            }
            std::string path_label = cfg.pruning ? " / pruning" : " / no-pruning";
            run_label = dim_label + " / " + quant_label + update_label + path_label;
        }

        bench_utils::WriteResultsToCsvV2(
            experiment_name,
            algorithm,
            dataset,
            n_iters,
            actual_iterations,
            static_cast<int>(target_d),
            n,
            static_cast<int>(n_clusters),
            construction_time_ms,
            static_cast<int>(THREADS),
            wcss_assign,
            config_dict,
            assign_r10,
            assign_r100,
            q_assign_r10,
            q_assign_r100,
            balance_stats.ToJson(),
            q_balance_stats_json,
            iter_stats_json,
            run_label
        );
    };

    // ── Main loop over target dimensions ──
    for (size_t target_d : target_d_list) {
        const size_t working_d = target_d;

        skmeans::Profiler::Get().Reset();

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
            std::cout << "Generating JLT matrix (" << target_d << " x " << d << ")..." << std::endl;
            jl_matrix = GenerateJLProjectionMatrix(d, target_d, 42);

            projected_data.resize(n * target_d);
            Eigen::Map<const MatrixR> data_mat(data.data(), n, d);
            Eigen::Map<MatrixR> proj_mat(projected_data.data(), n, target_d);
            proj_mat.noalias() = data_mat * jl_matrix.transpose();
            working_data = projected_data.data();
        } else if (dim_reduction == "mat") {
            std::cout << "Truncating to first " << target_d << " dimensions (matryoshka)..."
                      << std::endl;
            projected_data.resize(n * target_d);
            for (size_t i = 0; i < n; ++i) {
                std::copy_n(data.data() + i * d, target_d, projected_data.data() + i * target_d);
            }
            working_data = projected_data.data();
        }

        preprocess_timer.Toc();
        if (!is_raw) {
            std::cout << "Preprocessing completed in " << preprocess_timer.GetMilliseconds()
                      << " ms" << std::endl;
        }

        // ── Phase 2: Configure and Train ──
        skmeans::SuperKMeansConfig config;
        config.iters = n_iters;
        config.verbose = true;
        config.n_threads = THREADS;
        config.unrotate_centroids = true;
        config.early_termination = false;
        config.sampling_fraction = 1.0f;
        config.tol = 1e-3f;
        config.pruning = pruning;
        config.quantized_centroid_update = quantized_centroid_update;
        config.full_precision_final_centroids = full_precision_final_centroids;

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

        // Capture Train()'s profiler bucket and reset so subsequent
        // Assign()/QuantizedAssign() timings don't spill into it.
        std::string profiler_train_json = skmeans::Profiler::Get().ToJson();
        skmeans::Profiler::Get().Reset();

        // ── Phase 3 & 4: Final Assignments + CSV Output ──

        // Full-d assign helper (reused across rows)
        skmeans::SuperKMeansConfig assign_config;
        assign_config.n_threads = THREADS;
        assign_config.pruning = false;
        auto kmeans_fulld = SKM_f32(n_clusters, d, assign_config);

        if (is_raw) {
            // ── RAW: single row ──
            std::cout << "\n--- Final Assignments (raw) ---" << std::endl;
            auto assignments = kmeans.Assign(data.data(), working_centroids.data(), n, n_clusters);
            std::string profiler_assign_json = skmeans::Profiler::Get().ToJson();
            skmeans::Profiler::Get().Reset();

            std::vector<uint32_t> q_assignments;
            std::string profiler_q_assign_json;
            if (has_quantizer) {
                q_assignments =
                    kmeans.QuantizedAssign(data.data(), working_centroids.data(), n, n_clusters);
                profiler_q_assign_json = skmeans::Profiler::Get().ToJson();
                skmeans::Profiler::Get().Reset();
            }

            write_row(
                target_d,
                construction_time_ms,
                0.0,
                actual_iterations,
                kmeans.iteration_stats,
                assignments,
                q_assignments,
                working_centroids.data(),
                "none",
                config,
                profiler_train_json,
                profiler_assign_json,
                profiler_q_assign_json,
                data.data(),
                queries.data(),
                d
            );

        } else {
            // ── PREPROCESSING: two rows per target_d (except matryoshka: only recompute) ──

            // --- Row 1: UNPROJECT=true (PCA/JLT only — matryoshka has no inverse) ---
            if (dim_reduction != "mat") {
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

                auto assignments_unproj =
                    kmeans_fulld.Assign(data.data(), full_d_centroids.data(), n, n_clusters);
                std::string profiler_assign_json_u = skmeans::Profiler::Get().ToJson();
                skmeans::Profiler::Get().Reset();

                // Requantize in full-d: fit quantizer on full-d data (1 cheap iter),
                // then QuantizedAssign with unprojected centroids
                std::vector<uint32_t> q_assignments_unproj;
                std::string profiler_q_assign_json_u;
                if (has_quantizer) {
                    skmeans::SuperKMeansConfig refit_cfg;
                    refit_cfg.iters = 1;
                    refit_cfg.n_threads = THREADS;
                    refit_cfg.pruning = false;
                    refit_cfg.verbose = false;
                    refit_cfg.data_already_rotated =
                        true; // skip rotation so quantizer fits unrotated data
                    refit_cfg.sampling_fraction = 1.0f; // use all data for quantizer fitting

                    auto kmeans_refit = SKM(n_clusters, d, refit_cfg);
                    kmeans_refit.Train(data.data(), n);

                    q_assignments_unproj = kmeans_refit.QuantizedAssign(
                        data.data(), full_d_centroids.data(), n, n_clusters
                    );
                    profiler_q_assign_json_u = skmeans::Profiler::Get().ToJson();
                    skmeans::Profiler::Get().Reset();
                }

                write_row(
                    target_d,
                    construction_time_ms,
                    preprocess_timer.GetMilliseconds(),
                    actual_iterations,
                    kmeans.iteration_stats,
                    assignments_unproj,
                    q_assignments_unproj,
                    full_d_centroids.data(),
                    "true",
                    config,
                    profiler_train_json,
                    profiler_assign_json_u,
                    profiler_q_assign_json_u,
                    data.data(),
                    queries.data(),
                    d
                );
            }

            // --- Row 2: UNPROJECT=false (recompute centroids from full-d vectors) ---
            std::cout << "\n--- UNPROJECT=false ---" << std::endl;

            // Assign in projected space to get cluster membership
            auto projected_assignments =
                kmeans.Assign(projected_data.data(), working_centroids.data(), n, n_clusters);

            // Recompute full-d centroids from original vectors
            // This is a hidden design decision we must mention: it depends
            // on whether you want to materialize the index in full-d or projected space
            // We assume here we materialize the centroids in full-d
            std::vector<float> recomputed_centroids(n_clusters * d);
            // That why we must average again in full-d using the projected assignments
            // We only do the averaging: O(N), no argmin search here since we keep the same
            // centroids as Train
            RecomputeFullDCentroids(
                data.data(), projected_assignments, recomputed_centroids.data(), n, d, n_clusters
            );

            // Re-assign in full-d with recomputed centroids
            auto assignments_nounproj =
                kmeans_fulld.Assign(data.data(), recomputed_centroids.data(), n, n_clusters);
            std::string profiler_assign_json_n = skmeans::Profiler::Get().ToJson();
            skmeans::Profiler::Get().Reset();

            // QuantizedAssign in projected space (if quantizer available)
            // The quantizedAssign never looks at the full-d data
            std::vector<uint32_t> q_assignments;
            std::string profiler_q_assign_json_n;
            if (has_quantizer) {
                q_assignments = kmeans.QuantizedAssign(
                    projected_data.data(), working_centroids.data(), n, n_clusters
                );
                profiler_q_assign_json_n = skmeans::Profiler::Get().ToJson();
                skmeans::Profiler::Get().Reset();
            }

            write_row(
                target_d,
                construction_time_ms,
                preprocess_timer.GetMilliseconds(),
                actual_iterations,
                kmeans.iteration_stats,
                assignments_nounproj,
                q_assignments,
                recomputed_centroids.data(),
                "false",
                config,
                profiler_train_json,
                profiler_assign_json_n,
                profiler_q_assign_json_n,
                data.data(),
                queries.data(),
                d
            );

            // --- Row 3: UNPROJECT=projected (keep centroids in reduced-d space) ---
            std::cout << "\n--- UNPROJECT=projected ---" << std::endl;

            // Project queries into the target_d space (same transform as data).
            std::vector<float> projected_queries(n_queries * target_d);
            if (dim_reduction == "pca") {
                pca_holder->apply_noalloc(n_queries, queries.data(), projected_queries.data());
            } else if (dim_reduction == "jlt") {
                Eigen::Map<const MatrixR> qm(queries.data(), n_queries, d);
                Eigen::Map<MatrixR> pqm(projected_queries.data(), n_queries, target_d);
                pqm.noalias() = qm * jl_matrix.transpose();
            } else { // mat
                for (size_t i = 0; i < n_queries; ++i) {
                    std::copy_n(
                        queries.data() + i * d, target_d, projected_queries.data() + i * target_d
                    );
                }
            }

            // Re-run Assign in projected space for a clean profiler bucket
            // (the projected_assignments from UNPROJECT=false shared its profile
            // with the subsequent full-d Assign).
            auto assignments_proj =
                kmeans.Assign(projected_data.data(), working_centroids.data(), n, n_clusters);
            std::string profiler_assign_json_p = skmeans::Profiler::Get().ToJson();
            skmeans::Profiler::Get().Reset();

            std::vector<uint32_t> q_assignments_proj;
            std::string profiler_q_assign_json_p;
            if (has_quantizer) {
                q_assignments_proj = kmeans.QuantizedAssign(
                    projected_data.data(), working_centroids.data(), n, n_clusters
                );
                profiler_q_assign_json_p = skmeans::Profiler::Get().ToJson();
                skmeans::Profiler::Get().Reset();
            }

            write_row(
                target_d,
                construction_time_ms,
                preprocess_timer.GetMilliseconds(),
                actual_iterations,
                kmeans.iteration_stats,
                assignments_proj,
                q_assignments_proj,
                working_centroids.data(),
                "projected",
                config,
                profiler_train_json,
                profiler_assign_json_p,
                profiler_q_assign_json_p,
                projected_data.data(),
                projected_queries.data(),
                target_d
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hierarchical pipeline (raw only, f32 or sq8 only)
//
// Kept separate from RunPipeline because:
//   - No dimensionality reduction is exercised (raw input).
//   - Only f32 and sq8 quantizers are supported.
//   - Uses HierarchicalSuperKMeans (meso + fine + refine) rather than the
//     flat SuperKMeans, so the config type differs.
// ─────────────────────────────────────────────────────────────────────────────

template <skmeans::Quantization Q>
void RunHierarchicalPipeline(
    const std::string& dataset,
    const std::string& quantizer_name,
    bool pruning
) {
    using HSKM = skmeans::HierarchicalSuperKMeans<Q>;

    skmeans::Profiler::Get().Reset();

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t n_clusters = bench_utils::GetDefaultNClusters(n);
    const int n_iters = bench_utils::MAX_ITERS; // base SKM iters (per refinement pass)
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    const bool has_quantizer = (quantizer_name != "f32");
    const std::string experiment_name = "accelerators_raw_" + quantizer_name + "_hsk";
    const std::string algorithm = "hierarchical_superkmeans";

    std::cout << "=== Hierarchical Accelerators Benchmark ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;
    std::cout << "Quantizer: " << quantizer_name << " pruning=" << pruning << std::endl;
    std::cout << "n_clusters=" << n_clusters << " threads=" << THREADS << std::endl;

    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);
    {
        std::ifstream f(bench_utils::GetDataPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open data file\n";
            return;
        }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(bench_utils::GetQueryPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open query file\n";
            return;
        }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    bool is_angular =
        std::find(
            bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
        ) != bench_utils::ANGULAR_DATASETS.end();

    std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    bool has_gt = std::ifstream(gt_filename).good();
    std::unordered_map<int, std::vector<int>> gt_map;
    if (has_gt)
        gt_map = bench_utils::ParseGroundTruthJson(gt_filename);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.tol = 1e-3f;
    config.pruning = pruning;
    if (has_quantizer) {
        config.quantized_centroid_update = true;
    }
    config.angular = is_angular;
    if (is_angular)
        std::cout << "Using spherical k-means" << std::endl;
    // Hierarchical-specific (defaults taken from ad_hoc_hierarchical_superkmeans*.cpp)
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 5;
    config.iters_refinement = 0;

    auto kmeans = HSKM(n_clusters, d, config);
    bench_utils::TicToc train_timer;
    train_timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    train_timer.Toc();
    double construction_time_ms = train_timer.GetMilliseconds();
    std::string profiler_train_json = skmeans::Profiler::Get().ToJson();
    skmeans::Profiler::Get().Reset();

    std::cout << "Training completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Iteration config: meso=" << config.iters_mesoclustering
              << ", fine=" << config.iters_fineclustering << ", refine=" << config.iters_refinement
              << "\n";

    // ── Final assignments ──
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    // std::vector<uint32_t> assignments(n);
    std::string profiler_assign_json = skmeans::Profiler::Get().ToJson();
    skmeans::Profiler::Get().Reset();

    std::vector<uint32_t> q_assignments;
    std::string profiler_q_assign_json;
    if (has_quantizer) {
        q_assignments = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);
        profiler_q_assign_json = skmeans::Profiler::Get().ToJson();
        skmeans::Profiler::Get().Reset();
    }

    double wcss_assign =
        SKM_f32::ComputeWCSS(data.data(), centroids.data(), assignments.data(), n, d);
    std::cout << "WCSS (Assign): " << std::fixed << std::setprecision(2) << wcss_assign
              << std::endl;
    double wcss_q_assign = -1.0;
    if (!q_assignments.empty()) {
        wcss_q_assign =
            SKM_f32::ComputeWCSS(data.data(), centroids.data(), q_assignments.data(), n, d);
        std::cout << "WCSS (QuantizedAssign): " << std::fixed << std::setprecision(2)
                  << wcss_q_assign << std::endl;
    }

    auto balance_stats = SKM_f32::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance_stats.print();
    std::string q_balance_stats_json;
    if (!q_assignments.empty()) {
        auto q_balance_stats =
            SKM_f32::GetClustersBalanceStats(q_assignments.data(), n, n_clusters);
        q_balance_stats.print();
        q_balance_stats_json = q_balance_stats.ToJson();
    }

    bench_utils::recall_results_t assign_r10, assign_r100, q_assign_r10, q_assign_r100;
    if (has_gt) {
        assign_r10 = bench_utils::ComputeRecall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        assign_r100 = bench_utils::ComputeRecall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        std::cout << "  [Assign()]" << std::endl;
        bench_utils::PrintRecallResults(assign_r10, 10);
        bench_utils::PrintRecallResults(assign_r100, 100);
        if (!q_assignments.empty()) {
            q_assign_r10 = bench_utils::ComputeRecall(
                gt_map,
                q_assignments,
                queries.data(),
                centroids.data(),
                n_queries,
                n_clusters,
                d,
                10
            );
            q_assign_r100 = bench_utils::ComputeRecall(
                gt_map,
                q_assignments,
                queries.data(),
                centroids.data(),
                n_queries,
                n_clusters,
                d,
                100
            );
            std::cout << "  [QuantizedAssign()]" << std::endl;
            bench_utils::PrintRecallResults(q_assign_r10, 10);
            bench_utils::PrintRecallResults(q_assign_r100, 100);
        }
    }

    auto config_dict = BuildConfigDict(
        /*dim_reduction=*/"none",
        quantizer_name,
        /*unproject_centroids=*/"none",
        /*target_d=*/d,
        config,
        /*preprocessing_time_ms=*/0.0
    );
    config_dict["wcss_assign"] = std::to_string(wcss_assign);
    if (wcss_q_assign >= 0.0) {
        config_dict["wcss_quantized_assign"] = std::to_string(wcss_q_assign);
    }
    config_dict["iters_mesoclustering"] = std::to_string(config.iters_mesoclustering);
    config_dict["iters_fineclustering"] = std::to_string(config.iters_fineclustering);
    config_dict["iters_refinement"] = std::to_string(config.iters_refinement);
    config_dict["profiler"] = profiler_train_json;
    config_dict["profiler_assign"] = profiler_assign_json;
    if (!profiler_q_assign_json.empty()) {
        config_dict["profiler_quantized_assign"] = profiler_q_assign_json;
    }

    std::string run_label =
        "raw / " + quantizer_name + " / hsk" + (pruning ? " / pruning" : " / no-pruning");

    bench_utils::WriteResultsToCsvV2(
        experiment_name,
        algorithm,
        dataset,
        n_iters,
        /*actual_iterations=*/0,
        static_cast<int>(d),
        n,
        static_cast<int>(n_clusters),
        construction_time_ms,
        static_cast<int>(THREADS),
        wcss_assign,
        config_dict,
        assign_r10,
        assign_r100,
        q_assign_r10,
        q_assign_r100,
        balance_stats.ToJson(),
        q_balance_stats_json,
        /*iteration_stats_json=*/"",
        run_label
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main: parse CLI args and dispatch to templated pipeline
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <pca|jlt|mat|raw|hsk> <f32|sq8|rabitq|lvq4>"
                  << " <quantized_centroid_update=true|false>"
                  << " <full_precision_final_centroids=true|false>"
                  << " <pruning=true|false>" << std::endl;
        return 1;
    }

    const std::string dataset = argv[1];
    const std::string dim_reduction = argv[2];
    const std::string quantizer = argv[3];
    const bool quantized_centroid_update = ParseBool(argv[4]);
    const bool full_precision_final_centroids = ParseBool(argv[5]);
    const bool pruning = bench_utils::ParsePruning(argc, argv, 6);

    // Validate dim_reduction (also accept "hsk" as a sentinel for the
    // hierarchical pipeline — raw input, no projection).
    if (dim_reduction != "raw" && dim_reduction != "pca" && dim_reduction != "jlt" &&
        dim_reduction != "mat" && dim_reduction != "hsk") {
        std::cerr << "Invalid dim_reduction: " << dim_reduction
                  << " (expected: raw, pca, jlt, mat, hsk)\n";
        return 1;
    }

    if (dim_reduction == "hsk") {
        if (quantizer == "f32") {
            RunHierarchicalPipeline<skmeans::Quantization::f32>(dataset, quantizer, pruning);
        } else if (quantizer == "sq8") {
            RunHierarchicalPipeline<skmeans::Quantization::sq8>(dataset, quantizer, pruning);
        } else if (quantizer == "rabitq") {
            RunHierarchicalPipeline<skmeans::Quantization::rabitq>(dataset, quantizer, pruning);
        } else if (quantizer == "lvq4") {
            RunHierarchicalPipeline<skmeans::Quantization::lvq4>(dataset, quantizer, pruning);
        } else {
            std::cerr << "Invalid quantizer: " << quantizer
                      << " (expected: f32, sq8, rabitq, lvq4)\n";
            return 1;
        }
        return 0;
    }

    // Validate quantizer and dispatch to correct template instantiation
    const auto run = [&](auto tag) {
        RunPipeline<decltype(tag)::value>(
            dataset,
            dim_reduction,
            quantizer,
            quantized_centroid_update,
            full_precision_final_centroids,
            pruning
        );
    };
    if (quantizer == "f32") {
        run(skmeans::QuantizationTag<skmeans::Quantization::f32>{});
    } else if (quantizer == "sq8") {
        run(skmeans::QuantizationTag<skmeans::Quantization::sq8>{});
    } else if (quantizer == "rabitq") {
        run(skmeans::QuantizationTag<skmeans::Quantization::rabitq>{});
    } else if (quantizer == "lvq4") {
        run(skmeans::QuantizationTag<skmeans::Quantization::lvq4>{});
    } else {
        std::cerr << "Invalid quantizer: " << quantizer << " (expected: f32, sq8, rabitq, lvq4)\n";
        return 1;
    }

    return 0;
}
