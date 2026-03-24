#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <optional>

#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

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

    const std::string algorithm = "cuvs";

    if (argc < 2) exit(1);
    bool use_cuvs = argv[1][0] == '1'; // compatibility

    std::cout << (use_cuvs ? "cuVS Enabled\n" : "cuVS Disabled (ignored)\n");

    std::string dataset =
        (argc > 2) ? std::string(argv[2]) : std::string("arxiv");

    std::string experiment_name =
        (argc > 3) ? std::string(argv[3])
                   : std::string("end_to_end_gpu_ivf");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;

    const size_t n_clusters = std::max<int>(1u, static_cast<int>(std::sqrt(static_cast<double>(n)) * 4.0));

    int n_iters = bench_utils::MAX_ITERS;

    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    //std::cout << "n_clusters: " << n_clusters << "\n";

    // ------------------------------------------------------------
    // Load data
    // ------------------------------------------------------------
    std::vector<float> data;
    try { data.resize(n * d); }
    catch (...) { std::cerr << "Allocation failure for data\n"; return 1; }

    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "Failed to open dataset file\n"; return 1; }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    std::cout << "Data loaded successfully\n";

    // ------------------------------------------------------------
    // Detect angular/spherical dataset
    // ------------------------------------------------------------
    auto is_ang = std::find(
        bench_utils::ANGULAR_DATASETS.begin(),
        bench_utils::ANGULAR_DATASETS.end(),
        dataset);

    if (is_ang != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means for dataset: " << dataset << "\n";

        // L2-normalize host data
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float norm = 0.f;
            for (size_t j = 0; j < d; ++j)
                norm += data[i * d + j] * data[i * d + j];
            norm = std::sqrt(norm) + 1e-12f;
            for (size_t j = 0; j < d; ++j)
                data[i * d + j] /= norm;
        }
    }

    // ------------------------------------------------------------
    // Upload to GPU and setup cuVS KMeans
    // ------------------------------------------------------------
    // Loop over different n_clusters values
    //for (int n_clusters : bench_utils::VARYING_K_VALUES) {
		{
        std::cout << "\n========================================" << std::endl;
        std::cout << "n_clusters=" << n_clusters << std::endl;
        std::cout << "========================================" << std::endl;
    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    float* d_data = nullptr;
    RAFT_CUDA_TRY(cudaMalloc(&d_data, n * d * sizeof(float)));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_data, data.data(), n * d * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));

    float* d_centroids = nullptr;
    RAFT_CUDA_TRY(cudaMalloc(&d_centroids, n_clusters * d * sizeof(float)));

    cuvs::cluster::kmeans::params params{};
    params.n_clusters = n_clusters;
    params.max_iter   = n_iters;
    params.tol        = 1e-20;
    params.inertia_check = false;
		params.init       = cuvs::cluster::kmeans::params::Random;
		// params.seed       = 42; 

    // device matrix views
    auto X = raft::make_device_matrix_view<const float, int64_t>(d_data, n, d);
    auto centroids = raft::make_device_matrix_view<float, int64_t>(d_centroids, n_clusters, d);

    // host scalar views for inertia and iterations
    float inertia_host = 0.f;
    int iters_host = 0;
    auto inertia = raft::make_host_scalar_view<float>(&inertia_host);
    auto iters   = raft::make_host_scalar_view<int>(&iters_host);

    // ------------------------------------------------------------
    // Run KMeans
    // ------------------------------------------------------------
    bench_utils::TicToc timer;
    timer.Tic();

    cuvs::cluster::kmeans::fit(handle, params, X, std::nullopt, centroids, inertia, iters);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    timer.Toc();

    double construction_time_ms = timer.GetMilliseconds();
    std::cout << "\ncuVS KMeans completed in " << construction_time_ms << " ms\n";

    // ------------------------------------------------------------
    // Download centroids
    // ------------------------------------------------------------
    std::vector<float> host_centroids(n_clusters * d);
    RAFT_CUDA_TRY(cudaMemcpy(host_centroids.data(), d_centroids, n_clusters * d * sizeof(float),
                             cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_centroids);

    int actual_iterations = iters_host;
    double final_objective = inertia_host;

    // ------------------------------------------------------------
    // Config map
    // ------------------------------------------------------------
    std::unordered_map<std::string, std::string> config_map;
    config_map["backend"] = "cuvs";
    config_map["n_lists"] = std::to_string(n_clusters);
    config_map["niter"] = std::to_string(n_iters);
    config_map["spherical"] = (is_ang != bench_utils::ANGULAR_DATASETS.end()) ? "true" : "false";

    // ------------------------------------------------------------
    // Empty recall results (kept for compatibility)
    // ------------------------------------------------------------
    std::vector<std::tuple<int, float, float, float, float>> results_knn_10;
    std::vector<std::tuple<int, float, float, float, float>> results_knn_100;

    std::cout << "\nNote: Recall computation skipped\n";
    std::cout << "Writing results to CSV...\n";

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
        results_knn_100);
}

    std::cout << "Benchmark complete\n";
    return 0;
}

