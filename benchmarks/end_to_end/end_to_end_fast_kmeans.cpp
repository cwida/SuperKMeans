#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#include "dataset.h"
#include "general_functions.h"
#include "naive_kmeans.h"
#include "hamerly_kmeans.h"
#include "elkan_kmeans.h"
#include "annulus_kmeans.h"
#include "drake_kmeans.h"
#include "sort_kmeans.h"
#include "heap_kmeans.h"
#include "compare_kmeans.h"

#include "bench_utils.h"

const std::unordered_map<std::string, std::string> VALID_ALGORITHMS = {
    {"naive", "NaiveKmeans (Lloyd's)"},
    {"hamerly", "HamerlyKmeans"},
    {"elkan", "ElkanKmeans"},
    {"annulus", "AnnulusKmeans"},
    {"drake", "DrakeKmeans"},
    {"sort", "SortKmeans"},
    {"heap", "HeapKmeans"},
    {"compare", "CompareKmeans"},
};

Kmeans* create_algorithm(const std::string& variant, size_t n_clusters) {
    if (variant == "naive")   return new NaiveKmeans();
    if (variant == "hamerly") return new HamerlyKmeans();
    if (variant == "elkan")   return new ElkanKmeans();
    if (variant == "annulus") return new AnnulusKmeans();
    if (variant == "drake") {
        // Drake uses b lower bounds; sqrt(k) is a common heuristic
        int b = std::max(2, static_cast<int>(std::sqrt(static_cast<double>(n_clusters))));
        return new DrakeKmeans(b);
    }
    if (variant == "sort")    return new SortKmeans();
    if (variant == "heap")    return new HeapKmeans();
    if (variant == "compare") return new CompareKmeans();
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("glove200");
    std::string variant = (argc > 2) ? std::string(argv[2]) : std::string("naive");
    std::string experiment_name = (argc > 3) ? std::string(argv[3]) : std::string("end_to_end");

    auto alg_it = VALID_ALGORITHMS.find(variant);
    if (alg_it == VALID_ALGORITHMS.end()) {
        std::cerr << "Unknown algorithm variant '" << variant << "'\n";
        std::cerr << "Choose from: naive, hamerly, elkan, annulus, drake, sort, heap, compare\n";
        return 1;
    }
    const std::string algorithm = "fast_kmeans_" + variant;

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== Running algorithm: " << algorithm
              << " (" << alg_it->second << ") ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters << std::endl;
    std::cout << "NOTE: fast-kmeans uses double precision (float64) internally" << std::endl;

    // Load raw float32 data from binary file
    std::string filename = bench_utils::get_data_path(dataset);
    std::vector<float> raw_data(n * d);
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << filename << std::endl;
            return 1;
        }
        file.read(reinterpret_cast<char*>(raw_data.data()), n * d * sizeof(float));
    }

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "\nWARNING: Dataset '" << dataset << "' should use spherical k-means, "
                  << "but fast-kmeans does not support this. Results may be suboptimal." << std::endl;
    }

    // Create Dataset (row-major doubles, n x d) — library only supports double
    Dataset x(static_cast<int>(n), static_cast<int>(d));
    for (size_t i = 0; i < n * d; ++i) {
        x.data[i] = static_cast<double>(raw_data[i]);
    }
    raw_data.clear();
    raw_data.shrink_to_fit();

    // Seed RNG and initialize centers (random selection, matching scikit's init='random')
    std::srand(42);
    Dataset* initial_centers = init_centers(x, static_cast<unsigned short>(n_clusters));

    // Initial assignment
    auto* assignment = new unsigned short[n];
    assign(x, *initial_centers, assignment);

    // Create and run algorithm
    Kmeans* algo = create_algorithm(variant, n_clusters);
    algo->initialize(&x, static_cast<unsigned short>(n_clusters), assignment, static_cast<int>(THREADS));

    bench_utils::TicToc timer;
    timer.Tic();
    int actual_iterations = algo->run(n_iters);
    timer.Toc();

    double construction_time_ms = timer.GetMilliseconds();
    double final_objective = algo->getSSE();
    const Dataset* final_centers = algo->getCenters();

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations
              << " (requested: " << n_iters << ")" << std::endl;
    std::cout << "Final objective (SSE): " << final_objective << std::endl;

    // Convert centroids to row-major float for recall computation (n_clusters x d)
    std::vector<float> centroids_row(n_clusters * d);
    for (size_t i = 0; i < n_clusters; ++i) {
        for (size_t j = 0; j < d; ++j) {
            centroids_row[i * d + j] = static_cast<float>(final_centers->data[i * d + j]);
        }
    }

    // Convert assignments to vector<size_t> for bench_utils::compute_recall
    std::vector<size_t> assignments_vec(n);
    for (size_t i = 0; i < n; ++i) {
        assignments_vec[i] = static_cast<size_t>(assignment[i]);
    }

    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::string queries_filename = bench_utils::get_query_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);

    if (gt_file.good() && queries_file.good()) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << queries_filename << std::endl;

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        int n_queries = bench_utils::N_QUERIES;
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        std::vector<float> queries(n_queries * d);
        queries_file.read(reinterpret_cast<char*>(queries.data()),
                          n_queries * d * sizeof(float));
        queries_file.close();

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments_vec, queries.data(), centroids_row.data(),
            n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);

        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments_vec, queries.data(), centroids_row.data(),
            n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        std::unordered_map<std::string, std::string> config_map;
        config_map["algorithm"] = "\"" + variant + "\"";
        config_map["algorithm_class"] = "\"" + alg_it->second + "\"";
        config_map["max_iterations"] = std::to_string(n_iters);
        config_map["actual_iterations"] = std::to_string(actual_iterations);
        config_map["seed"] = "42";
        config_map["precision"] = "\"float64\"";

        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            n_iters,
            actual_iterations,
            static_cast<int>(d),
            n,
            static_cast<int>(n_clusters),
            construction_time_ms,
            static_cast<int>(THREADS),
            final_objective,
            config_map,
            results_knn_10,
            results_knn_100
        );
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file.good()) {
            std::cout << "Queries file not found: " << queries_filename << std::endl;
        }
        std::cout << "Skipping CSV output (recall computation requires ground truth)" << std::endl;
    }

    delete algo;
    delete initial_centers;
    delete[] assignment;
}
