#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

using SKM_f32 = skmeans::SuperKMeans<>;

// Cluster counts (k) to sweep for the scalability study.
static const std::vector<size_t> SCALABILITY_K_VALUES = {10000, 50000, 100000, 200000};

// Quantizers to sweep.
static const std::vector<std::string> SCALABILITY_QUANTIZERS = {"f32", "sq8", "rabitq", "lvq4"};

static std::unordered_map<std::string, std::string> BuildConfigDict(
    const std::string& quantizer,
    const skmeans::HierarchicalSuperKMeansConfig& cfg
) {
    std::unordered_map<std::string, std::string> c;
    c["dim_reduction"] = "\"none\"";
    c["quantizer"] = "\"" + quantizer + "\"";
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
    c["iters_mesoclustering"] = std::to_string(cfg.iters_mesoclustering);
    c["iters_fineclustering"] = std::to_string(cfg.iters_fineclustering);
    c["iters_refinement"] = std::to_string(cfg.iters_refinement);
    return c;
}

// Run one hierarchical-clustering configuration against already-loaded data.
template <skmeans::Quantization Q>
static void RunHierarchical(
    const std::string& dataset,
    size_t n,
    size_t d,
    const float* data,
    const float* queries,
    size_t n_clusters,
    bool use_blas_only,
    bool is_angular,
    bool has_gt,
    const std::unordered_map<int, std::vector<int>>& gt_map
) {
    using HSKM = skmeans::HierarchicalSuperKMeans<Q>;
    const std::string quantizer_name = skmeans::QuantizationName(Q);

    skmeans::Profiler::Get().Reset();

    const size_t n_queries = bench_utils::N_QUERIES;
    const int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = skmeans::ResolveNumThreads(0);

    const bool has_quantizer = (Q != skmeans::Quantization::f32);
    const std::string experiment_name = "scalability";
    const std::string algorithm = "hierarchical_superkmeans";

    std::cout << "\n=== Scalability: " << quantizer_name << " / k=" << n_clusters << " / "
              << (use_blas_only ? "blas-only" : "pruning") << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ") threads=" << THREADS
              << std::endl;

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.tol = 1e-3f;
    config.use_blas_only = use_blas_only;
    if (has_quantizer) {
        config.quantized_centroid_update = true;
    }
    config.angular = is_angular;
    if (is_angular)
        std::cout << "Using spherical k-means" << std::endl;
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 5;
    config.iters_refinement = 0;

    auto kmeans = HSKM(n_clusters, d, config);
    bench_utils::TicToc train_timer;
    train_timer.Tic();
    std::vector<float> centroids = kmeans.Train(data, n);
    train_timer.Toc();
    const double construction_time_ms = train_timer.GetMilliseconds();
    std::string profiler_train_json = skmeans::Profiler::Get().ToJson();
    skmeans::Profiler::Get().Reset();

    std::cout << "Training completed in " << construction_time_ms << " ms" << std::endl;

    // auto assignments = kmeans.Assign(data, centroids.data(), n, n_clusters);
    std::vector<uint32_t> assignments(n);
    if (quantizer_name == "f32") {
        assignments = kmeans.Assign(data, centroids.data(), n, n_clusters);
    }

    std::string profiler_assign_json = skmeans::Profiler::Get().ToJson();
    skmeans::Profiler::Get().Reset();

    std::vector<uint32_t> q_assignments;
    std::string profiler_q_assign_json;
    if (has_quantizer) {
        q_assignments = kmeans.AssignTrainingPoints(data, centroids.data(), n, n_clusters);
        profiler_q_assign_json = skmeans::Profiler::Get().ToJson();
        skmeans::Profiler::Get().Reset();
    }

    const double wcss_assign =
        SKM_f32::ComputeWCSS(data, centroids.data(), assignments.data(), n, d);
    std::cout << "WCSS (Assign): " << std::fixed << std::setprecision(2) << wcss_assign
              << std::endl;
    double wcss_q_assign = -1.0;
    if (!q_assignments.empty()) {
        wcss_q_assign = SKM_f32::ComputeWCSS(data, centroids.data(), q_assignments.data(), n, d);
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
            gt_map, assignments, queries, centroids.data(), n_queries, n_clusters, d, 10
        );
        assign_r100 = bench_utils::ComputeRecall(
            gt_map, assignments, queries, centroids.data(), n_queries, n_clusters, d, 100
        );
        std::cout << "  [Assign()]" << std::endl;
        bench_utils::PrintRecallResults(assign_r10, 10);
        bench_utils::PrintRecallResults(assign_r100, 100);
        if (!q_assignments.empty()) {
            q_assign_r10 = bench_utils::ComputeRecall(
                gt_map, q_assignments, queries, centroids.data(), n_queries, n_clusters, d, 10
            );
            q_assign_r100 = bench_utils::ComputeRecall(
                gt_map, q_assignments, queries, centroids.data(), n_queries, n_clusters, d, 100
            );
            std::cout << "  [QuantizedAssign()]" << std::endl;
            bench_utils::PrintRecallResults(q_assign_r10, 10);
            bench_utils::PrintRecallResults(q_assign_r100, 100);
        }
    }

    auto config_dict = BuildConfigDict(quantizer_name, config);
    config_dict["wcss_assign"] = std::to_string(wcss_assign);
    if (wcss_q_assign >= 0.0) {
        config_dict["wcss_quantized_assign"] = std::to_string(wcss_q_assign);
    }
    config_dict["profiler"] = profiler_train_json;
    config_dict["profiler_assign"] = profiler_assign_json;
    if (!profiler_q_assign_json.empty()) {
        config_dict["profiler_quantized_assign"] = profiler_q_assign_json;
    }

    const std::string run_label =
        "hsk / " + quantizer_name + (use_blas_only ? " / blas-only" : " / pruning");

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return 1;
    }
    const std::string dataset = argv[1];

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_queries = bench_utils::N_QUERIES;

    // ── Load data + queries once (reserve, no zero-init) ──
    std::vector<float> data;
    std::vector<float> queries;
    data.reserve(n * d);
    queries.reserve(n_queries * d);
    {
        std::ifstream f(bench_utils::GetDataPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open data file\n";
            return 1;
        }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
        f.close();
    }
    {
        std::ifstream f(bench_utils::GetQueryPath(dataset), std::ios::binary);
        if (!f) {
            std::cerr << "Failed to open query file\n";
            return 1;
        }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
        f.close();
    }

    const bool is_angular =
        std::find(
            bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
        ) != bench_utils::ANGULAR_DATASETS.end();

    const std::string gt_filename = bench_utils::GetGroundTruthPath(dataset);
    const bool has_gt = std::ifstream(gt_filename).good();
    std::unordered_map<int, std::vector<int>> gt_map;
    if (has_gt)
        gt_map = bench_utils::ParseGroundTruthJson(gt_filename);

    for (bool use_blas_only : {true, false}) {
        for (const std::string& quantizer : SCALABILITY_QUANTIZERS) {
            for (size_t k : SCALABILITY_K_VALUES) {
                if (k > n) {
                    std::cout << "Skipping k=" << k << " (> n=" << n << ")" << std::endl;
                    continue;
                }
                const auto run = [&](auto tag) {
                    RunHierarchical<decltype(tag)::value>(
                        dataset,
                        n,
                        d,
                        data.data(),
                        queries.data(),
                        k,
                        use_blas_only,
                        is_angular,
                        has_gt,
                        gt_map
                    );
                };
                if (quantizer == "f32") {
                    run(skmeans::QuantizationTag<skmeans::Quantization::f32>{});
                } else if (quantizer == "sq8") {
                    run(skmeans::QuantizationTag<skmeans::Quantization::sq8>{});
                } else if (quantizer == "lvq4") {
                    run(skmeans::QuantizationTag<skmeans::Quantization::lvq4>{});
                } else {
                    run(skmeans::QuantizationTag<skmeans::Quantization::rabitq>{});
                }
            }
        }
    }

    return 0;
}
