#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <faiss/utils/utils.h>

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "bench_utils.h"
#include "superkmeans/nanobench.h"

#include "superkmeans/distance_computers/kernels.cuh"

int main(int argc, char* argv[]) {
#ifdef USE_CUDA
    // Trigger GPU Initialization
    // We need to do this before benchmarking, as the GPU will only initialize
    // when the first kernel is launched. Therefore we now launch a bogus kernel first.
    printf("Trigger GPU initialization.\n");
    skmeans::kernels::trigger_gpu_initialization();
    printf("Triggered GPU initialization.\n");
#endif

    // Experiment configuration
    const std::string algorithm = "faiss";

    // First argument is required and indicates whether to use cuvs (0 => no, 1 => yes)
		if (argc < 2) exit(1);
    bool use_cuvs = argv[1][0] == '1';

		if (use_cuvs) {
			std::cout << "cuVS Enabled" << std::endl;
		}
		else {
			std::cout << "cuVS Disabled" << std::endl;
		}

    // dataset name can be passed as second argument (default: "end_to_end")
    std::string dataset = (argc > 2) ? std::string(argv[2]) : std::string("arxiv");


    // Experiment name can be passed as fourth argument (default: "end_to_end")
    std::string experiment_name = (argc > 3) ? std::string(argv[3]) : std::string("end_to_end");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;
     const size_t n_clusters = 
          std::max<int>(1u, static_cast<int>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Compile options: " << faiss::get_compile_options() << std::endl;

    std::vector<float> data;
    try {
        data.resize(n * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    // Loop over different n_clusters values
    //for (int n_clusters : bench_utils::VARYING_K_VALUES) {
		{
        std::cout << "\n========================================" << std::endl;
        std::cout << "n_clusters=" << n_clusters << std::endl;
        std::cout << "========================================" << std::endl;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
		config.use_cuvs = use_cuvs;
    config.device = 0;
    config.useFloat16 = false;

    faiss::gpu::GpuIndexFlatL2 index(&res, d, config);

    // Set up clustering parameters
    faiss::ClusteringParameters cp;
    cp.niter = n_iters;
    cp.verbose = false;
    cp.max_points_per_centroid = 999999; // We don't want to take samples
    cp.nredo = 1;

    // Check if this dataset should use angular/spherical k-means
    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(),
        bench_utils::ANGULAR_DATASETS.end(),
        dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means for dataset: " << dataset << std::endl;
        cp.spherical = true;
    }

    // Create the clustering object
    faiss::Clustering clus(d, n_clusters, cp);

    // Time the training
    bench_utils::TicToc timer;
    timer.Tic();
    clus.train(n, data.data(), index);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();

    // Get actual iterations and final objective
    int actual_iterations = static_cast<int>(clus.iteration_stats.size());
    double final_objective = clus.iteration_stats.back().obj;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective: " << final_objective << std::endl;

    // Compute recall if ground truth file exists
    // std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    // std::string queries_filename = bench_utils::get_query_path(dataset);

    // std::ifstream gt_file(gt_filename);
    // std::ifstream queries_file(queries_filename, std::ios::binary);

    //if (gt_file.good() && queries_file.good()) {
 				// gt_file.close();
        // std::cout << "\n--- Computing Recall ---" << std::endl;
        // std::cout << "Ground truth file: " << gt_filename << std::endl;
        // std::cout << "Queries file: " << queries_filename << std::endl;

        // // Load ground truth
        // auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);

        // // Use only first N_QUERIES queries
        // int n_queries = bench_utils::N_QUERIES;
        // std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
        //           << " from ground truth)" << std::endl;

        // // Load query vectors (only first n_queries)
        // std::vector<float> queries(n_queries * d);
        // queries_file.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        // queries_file.close();

        // // Get cluster assignments from FAISS
        // // FAISS doesn't store assignments directly, so we need to assign data points to nearest
        // // centroids
        // std::vector<faiss::idx_t> assignments(n);
        // std::vector<float> distances_to_centroids(n);

        // // Get centroids from clustering result
        // const float* centroids = clus.centroids.data();

        // // Assign each data point to its nearest centroid
        // faiss::IndexFlatL2 centroid_index(d);
        // centroid_index.add(n_clusters, centroids);
        // centroid_index.search(n, data.data(), 1, distances_to_centroids.data(), assignments.data());

        // // Compute recall for both KNN values
        // auto results_knn_10 = bench_utils::compute_recall(
        //     gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 10
        // );
        // bench_utils::print_recall_results(results_knn_10, 10);

        // auto results_knn_100 = bench_utils::compute_recall(
        //     gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 100
        // );
        // bench_utils::print_recall_results(results_knn_100, 100);
        std::vector<std::tuple<int, float, float, float, float>> results_knn_10;   // Empty
        std::vector<std::tuple<int, float, float, float, float>> results_knn_100;  // Empty

        // Create config dictionary with FAISS parameters
        std::unordered_map<std::string, std::string> config_map;
        config_map["niter"] = std::to_string(cp.niter);
        config_map["nredo"] = std::to_string(cp.nredo);
        config_map["max_points_per_centroid"] = std::to_string(cp.max_points_per_centroid);
        config_map["min_points_per_centroid"] = std::to_string(cp.min_points_per_centroid);
        config_map["seed"] = std::to_string(cp.seed);
        config_map["spherical"] = cp.spherical ? "true" : "false";
        config_map["int_centroids"] = cp.int_centroids ? "true" : "false";
        config_map["update_index"] = cp.update_index ? "true" : "false";
        config_map["frozen_centroids"] = cp.frozen_centroids ? "true" : "false";
        config_map["verbose"] = cp.verbose ? "true" : "false";

        // Write results to CSV
        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            n_iters,
            actual_iterations,
            d,
            n,
            n_clusters,
            construction_time_ms,
            static_cast<int>(THREADS),
            final_objective,
            config_map,
            results_knn_10,
            results_knn_100
        );
    } 
		/*
		else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file.good()) {
            std::cout << "Queries file not found: " << queries_filename << std::endl;
        }
        std::cout << "Skipping CSV output (recall computation requires ground truth)" << std::endl;
    }
		*/
}
