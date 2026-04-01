#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bench_utils {

/**
 * @brief Compute the default number of clusters for a dataset.
 *
 * Uses the heuristic: n_clusters = max(1, sqrt(n) * 4)
 *
 * @param n Number of data points
 * @return Default number of clusters
 */
inline size_t get_default_n_clusters(size_t n) {
    return std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
}

// Path constants for benchmark data
inline const std::string BENCHMARKS_ROOT = std::string(CMAKE_SOURCE_DIR) + "/benchmarks";
inline const std::string DATA_DIR = BENCHMARKS_ROOT + "/data";
inline const std::string GROUND_TRUTH_DIR = BENCHMARKS_ROOT + "/ground_truth";

inline std::string get_data_path(const std::string& dataset) {
    return DATA_DIR + "/data_" + dataset + ".bin";
}

inline std::string get_query_path(const std::string& dataset) {
    return DATA_DIR + "/data_" + dataset + "_test.bin";
}

inline std::string get_ground_truth_path(const std::string& dataset) {
    return GROUND_TRUTH_DIR + "/" + dataset + ".json";
}

class TicToc {
  public:
    size_t accum_time = 0;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    void Reset() {
        accum_time = 0;
        start = std::chrono::high_resolution_clock::now();
    }

    void Tic() { start = std::chrono::high_resolution_clock::now(); }

    void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    double GetMilliseconds() const {
        return accum_time / 1e6; // Convert nanoseconds to milliseconds
    }
};

// Dataset configurations: name -> (num_vectors, num_dimensions)
const std::unordered_map<std::string, std::pair<size_t, size_t>> DATASET_PARAMS = {
    {"fmnist", {60000, 784}},
    {"yi", {187843, 128}},
    {"llama", {256921, 128}},
    {"sift", {1000000, 128}},
    {"yahoo", {677305, 384}},
    {"yandex", {1000000, 200}},
    {"glove200", {1183514, 200}},
    {"clip", {1281167, 512}},
    {"mxbai", {769382, 1024}},
    {"wiki", {260372, 3072}},
    {"contriever", {990000, 768}},
    {"gist", {1000000, 960}},
    {"openai", {999000, 1536}},
    {"arxiv", {2253000, 768}},
    {"cohere", {10000000, 1024}},
    {"cohere50m", {50000000, 1024}},
};

const std::vector<std::string> ANGULAR_DATASETS =
    {"yandex", "glove200", "glove100", "glove50", "llama"};

// Standard exploration fractions for recall computation
const std::vector<float> EXPLORE_FRACTIONS = {0.001f,  0.002f,  0.003f,  0.004f,  0.005f,  0.006f,
                                              0.007f,  0.008f,  0.009f,  0.0100f, 0.0125f, 0.0150f,
                                              0.0175f, 0.0200f, 0.0225f, 0.0250f, 0.0275f, 0.0300f,
                                              0.0325f, 0.0350f, 0.0375f, 0.0400f, 0.0425f, 0.0450f,
                                              0.0475f, 0.0500f, 0.1f};

// KNN values to test
const std::vector<int> KNN_VALUES = {10, 100};

// Benchmark configuration
const int MAX_ITERS = 10;
const int N_QUERIES = 1000;

// Early termination benchmark configuration
const std::vector<float> RECALL_TOL_VALUES =
    {0.03f, 0.02f, 0.01f, 0.0075f, 0.005f, 0.0025f, 0.001f};
const std::vector<int> FAISS_EARLY_TERM_ITERS = {10};
const int SCIKIT_EARLY_TERM_MAX_ITERS = 300;
const float SCIKIT_EARLY_TERM_TOL = 1e-8f;

// Sampling fraction values for sampling experiment
const std::vector<float> SAMPLING_FRACTION_VALUES = {
    1.0f,  0.9f,  0.8f, 0.7f,  0.6f,    0.5f,   0.4f,    0.3f,   0.2f,    0.1f,   0.05f,
    0.04f, 0.03f, 0.02, 0.01f, 0.0075f, 0.005f, 0.0025f, 0.001f, 0.0005f, 0.0001f
};

// Iteration values for pareto experiment (grid search)
const std::vector<int> PARETO_ITERS_VALUES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Hierarchical SuperKMeans pareto hyperparameter grids
const std::vector<int> HIERARCHICAL_PARETO_MESOCLUSTERING_ITERS = {1, 3, 5, 7, 9};
const std::vector<int> HIERARCHICAL_PARETO_FINECLUSTERING_ITERS = {1, 3, 5, 7, 9};
const std::vector<int> HIERARCHICAL_PARETO_REFINEMENT_ITERS = {0, 1, 2, 3, 5};

// Hierarchical SuperKMeans sampling iteration parameters
const int HIERARCHICAL_SAMPLING_MESOCLUSTERING_ITERS = 10;
const int HIERARCHICAL_SAMPLING_FINECLUSTERING_ITERS = 10;
const int HIERARCHICAL_SAMPLING_REFINEMENT_ITERS = 2;

// n_clusters values for varying_k experiment
const std::vector<int> VARYING_K_VALUES = {100, 1000, 10000, 100000};

/**
 * @brief Parse ground truth JSON file.
 *
 * Simple JSON parser for our specific use case: {"query_idx": [vector_ids...], ...}
 *
 * @param filename Path to JSON file
 * @return Map of query index to vector IDs
 */
inline std::unordered_map<int, std::vector<int>> parse_ground_truth_json(const std::string& filename
) {
    std::unordered_map<int, std::vector<int>> gt_map;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return gt_map;
    }

    std::string line;
    std::getline(file, line); // Read entire file as one line

    // Simple parser: look for "query_idx": [vector_ids...]
    size_t pos = 0;
    while ((pos = line.find("\"", pos)) != std::string::npos) {
        size_t key_start = pos + 1;
        size_t key_end = line.find("\"", key_start);
        if (key_end == std::string::npos)
            break;

        std::string key_str = line.substr(key_start, key_end - key_start);
        int query_idx = std::stoi(key_str);

        // Find the array of vector IDs
        size_t arr_start = line.find("[", key_end);
        size_t arr_end = line.find("]", arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos)
            break;

        std::string arr_str = line.substr(arr_start + 1, arr_end - arr_start - 1);
        std::vector<int> vector_ids;
        std::istringstream iss(arr_str);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Remove whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                vector_ids.push_back(std::stoi(token));
            }
        }

        gt_map[query_idx] = vector_ids;
        pos = arr_end + 1;
    }

    return gt_map;
}

/**
 * @brief Compute recall@K for different exploration fractions.
 *
 * @tparam AssignmentType Type of assignment values (int, uint32_t, faiss::idx_t, etc.)
 * @param gt_map Ground truth map (query_idx -> vector_ids)
 * @param assignments Cluster assignments for all data vectors
 * @param queries Query vectors (row-major, n_queries × d)
 * @param centroids Centroid vectors (row-major, n_clusters × d)
 * @param n_queries Number of queries
 * @param n_clusters Number of clusters
 * @param d Dimensionality
 * @param knn Number of ground truth neighbors to consider
 * @return Vector of tuples (centroids_to_explore, explore_fraction, recall_mean, recall_std,
 * avg_vectors_to_visit)
 */
template <typename AssignmentType>
std::vector<std::tuple<int, float, float, float, float>> compute_recall(
    const std::unordered_map<int, std::vector<int>>& gt_map,
    const std::vector<AssignmentType>& assignments,
    const float* queries,
    const float* centroids,
    size_t n_queries,
    size_t n_clusters,
    size_t d,
    int knn
) {
    // Count cluster sizes to compute vectors to visit
    std::vector<size_t> cluster_sizes(n_clusters, 0);
    for (const auto& assignment : assignments) {
        cluster_sizes[static_cast<size_t>(assignment)]++;
    }

    // Compute distances from queries to centroids
    // Using L2 distance: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
    std::vector<float> query_norms(n_queries);
    std::vector<float> centroid_norms(n_clusters);

    // Compute query norms
    for (size_t i = 0; i < n_queries; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float val = queries[i * d + j];
            norm += val * val;
        }
        query_norms[i] = norm;
    }

    // Compute centroid norms
    for (size_t i = 0; i < n_clusters; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float val = centroids[i * d + j];
            norm += val * val;
        }
        centroid_norms[i] = norm;
    }

    // Compute query-centroid distances
    std::vector<float> distances(n_queries * n_clusters);
    for (size_t i = 0; i < n_queries; ++i) {
        for (size_t j = 0; j < n_clusters; ++j) {
            // Dot product
            float dot = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                dot += queries[i * d + k] * centroids[j * d + k];
            }
            distances[i * n_clusters + j] = query_norms[i] + centroid_norms[j] - 2.0f * dot;
        }
    }

    std::vector<std::tuple<int, float, float, float, float>> results;
    for (float explore_frac : EXPLORE_FRACTIONS) {
        int centroids_to_explore = std::max(1, static_cast<int>(n_clusters * explore_frac));

        // For each query, find top-N nearest centroids
        std::vector<float> query_recalls;
        size_t total_vectors_to_visit = 0;

        for (int query_idx = 0; query_idx < static_cast<int>(n_queries); ++query_idx) {
            if (gt_map.find(query_idx) == gt_map.end()) {
                continue;
            }

            // Get distances for this query
            std::vector<std::pair<float, int>> query_distances;
            for (size_t j = 0; j < n_clusters; ++j) {
                query_distances.push_back(
                    {distances[query_idx * n_clusters + j], static_cast<int>(j)}
                );
            }

            // Sort by distance to get top-N centroids
            std::partial_sort(
                query_distances.begin(),
                query_distances.begin() + centroids_to_explore,
                query_distances.end()
            );

            // Create set of top centroid indices and count vectors to visit
            std::unordered_set<int> top_centroids;
            size_t vectors_to_visit = 0;
            for (int t = 0; t < centroids_to_explore; ++t) {
                int centroid_idx = query_distances[t].second;
                top_centroids.insert(centroid_idx);
                vectors_to_visit += cluster_sizes[centroid_idx];
            }
            total_vectors_to_visit += vectors_to_visit;

            // Check how many ground truth vectors have their assigned centroid in top-N
            const auto& gt_vector_ids = gt_map.at(query_idx);
            int found = 0;
            int gt_count = std::min(knn, static_cast<int>(gt_vector_ids.size()));

            for (int i = 0; i < gt_count; ++i) {
                int vector_id = gt_vector_ids[i];
                int assigned_centroid = static_cast<int>(assignments[vector_id]);
                if (top_centroids.find(assigned_centroid) != top_centroids.end()) {
                    ++found;
                }
            }

            float query_recall = static_cast<float>(found) / static_cast<float>(gt_count);
            query_recalls.push_back(query_recall);
        }

        // Compute mean and standard deviation
        float average_recall = 0.0f;
        for (float recall : query_recalls) {
            average_recall += recall;
        }
        average_recall /= static_cast<float>(n_queries);

        float std_recall = 0.0f;
        if (query_recalls.size() > 1) {
            float variance = 0.0f;
            for (float recall : query_recalls) {
                float diff = recall - average_recall;
                variance += diff * diff;
            }
            variance /= static_cast<float>(
                query_recalls.size() - 1
            ); // Sample standard deviation (Bessel's correction)
            std_recall = std::sqrt(variance);
        }

        float avg_vectors_to_visit =
            static_cast<float>(total_vectors_to_visit) / static_cast<float>(n_queries);
        results.push_back(std::make_tuple(
            centroids_to_explore, explore_frac, average_recall, std_recall, avg_vectors_to_visit
        ));
    }

    return results;
}

inline void print_recall_results(
    const std::vector<std::tuple<int, float, float, float, float>>& results,
    int knn
) {
    printf("\n--- Recall@%d ---\n", knn);
    for (const auto& [centroids_to_explore, explore_frac, recall, std_recall, avg_vectors] :
         results) {
        printf(
            "Recall@%4d (%5.2f%% centroids, %8.0f avg vectors): %.4f ± %.4f\n",
            centroids_to_explore,
            explore_frac * 100.0f,
            avg_vectors,
            recall,
            std_recall
        );
    }
}

/**
 * @brief Create directory recursively if it doesn't exist.
 */
inline bool create_directory_recursive(const std::string& path) {
    std::string current_path;
    std::istringstream path_stream(path);
    std::string segment;

    while (std::getline(path_stream, segment, '/')) {
        if (segment.empty())
            continue;
        current_path += "/" + segment;

        struct stat st;
        if (stat(current_path.c_str(), &st) != 0) {
            if (mkdir(current_path.c_str(), 0755) != 0) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Write results to CSV file.
 *
 * @param experiment_name Name of the experiment (e.g., "end_to_end")
 * @param algorithm Name of the algorithm (e.g., "superkmeans", "faiss")
 * @param dataset Dataset name
 * @param n_iters Number of iterations (max requested)
 * @param actual_iterations Actual iterations performed (may be less if early termination)
 * @param dimensionality Data dimensionality
 * @param data_size Number of data points
 * @param n_clusters Number of clusters
 * @param construction_time_ms Construction time in milliseconds
 * @param threads Number of threads used
 * @param final_objective Final k-means objective value
 * @param config_dict Dictionary with algorithm-specific configuration (will be serialized to JSON)
 * @param results_knn_10 Results for KNN=10
 * @param results_knn_100 Results for KNN=100
 */
inline void write_results_to_csv(
    const std::string& experiment_name,
    const std::string& algorithm,
    const std::string& dataset,
    int n_iters,
    int actual_iterations,
    int dimensionality,
    size_t data_size,
    int n_clusters,
    double construction_time_ms,
    int threads,
    double final_objective,
    const std::unordered_map<std::string, std::string>& config_dict,
    const std::vector<std::tuple<int, float, float, float, float>>& results_knn_10,
    const std::vector<std::tuple<int, float, float, float, float>>& results_knn_100,
    const std::string& balance_stats_json = ""
) {
    const char* arch_env = std::getenv("SKM_ARCH");
    std::string arch = arch_env ? std::string(arch_env) : "default";
    std::string results_dir = std::string(CMAKE_SOURCE_DIR) + "/benchmarks/results/" + arch;
    create_directory_recursive(results_dir);
    std::string csv_path = results_dir + "/" + experiment_name + ".csv";
    bool file_exists = (std::ifstream(csv_path).good());
    std::ofstream csv_file(csv_path, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return;
    }
    bool has_recall_data = !results_knn_10.empty() || !results_knn_100.empty();
    if (!file_exists) {
        csv_file << "timestamp,algorithm,dataset,n_iters,actual_iterations,dimensionality,data_"
                    "size,n_clusters,"
                 << "construction_time_ms,threads,final_objective";
        // Add columns for each KNN and explore fraction combination (only if we have recall data)
        if (has_recall_data) {
            for (int knn : KNN_VALUES) {
                for (float explore_frac : EXPLORE_FRACTIONS) {
                    csv_file << ",recall@" << knn << "@" << std::fixed << std::setprecision(2)
                             << (explore_frac * 100.0f);
                    csv_file << ",recall_std@" << knn << "@" << std::fixed << std::setprecision(2)
                             << (explore_frac * 100.0f);
                    csv_file << ",centroids_explored@" << knn << "@" << std::fixed
                             << std::setprecision(2) << (explore_frac * 100.0f);
                    csv_file << ",vectors_explored@" << knn << "@" << std::fixed
                             << std::setprecision(2) << (explore_frac * 100.0f);
                }
            }
        }
        csv_file << ",balance_stats,config\n";
    }
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    localtime_r(&now_time_t, &now_tm);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &now_tm);
    csv_file << timestamp << "," << algorithm << "," << dataset << "," << n_iters << ","
             << actual_iterations << "," << dimensionality << "," << data_size << "," << n_clusters
             << "," << std::fixed << std::setprecision(2) << construction_time_ms << "," << threads
             << "," << std::setprecision(6) << final_objective;
    if (has_recall_data) {
        // Write KNN=10 results
        for (const auto& [centroids_to_explore, explore_frac, recall, std_recall, avg_vectors] :
             results_knn_10) {
            csv_file << "," << std::setprecision(6) << recall;
            csv_file << "," << std::setprecision(6) << std_recall;
            csv_file << "," << centroids_to_explore;
            csv_file << "," << std::setprecision(2) << avg_vectors;
        }
        // Write KNN=100 results
        for (const auto& [centroids_to_explore, explore_frac, recall, std_recall, avg_vectors] :
             results_knn_100) {
            csv_file << "," << std::setprecision(6) << recall;
            csv_file << "," << std::setprecision(6) << std_recall;
            csv_file << "," << centroids_to_explore;
            csv_file << "," << std::setprecision(2) << avg_vectors;
        }
    }

    // Write balance_stats JSON
    if (!balance_stats_json.empty()) {
        std::string escaped_balance = balance_stats_json;
        size_t pos = 0;
        while ((pos = escaped_balance.find("\"", pos)) != std::string::npos) {
            escaped_balance.replace(pos, 1, "\"\"");
            pos += 2;
        }
        csv_file << ",\"" << escaped_balance << "\"";
    } else {
        csv_file << ",";
    }

    std::ostringstream config_json_ss;
    config_json_ss << "{";
    bool first = true;
    for (const auto& [key, value] : config_dict) {
        if (!first) {
            config_json_ss << ",";
        }
        config_json_ss << "\"" << key << "\":" << value;
        first = false;
    }
    config_json_ss << "}";
    std::string config_json = config_json_ss.str();

    // Write config JSON (escape quotes for CSV)
    std::string escaped_config = config_json;
    size_t pos = 0;
    while ((pos = escaped_config.find("\"", pos)) != std::string::npos) {
        escaped_config.replace(pos, 1, "\"\"");
        pos += 2;
    }
    csv_file << ",\"" << escaped_config << "\"\n";

    csv_file.close();
    std::cout << "Results written to: " << csv_path << std::endl;
}

/**
 * @brief Generate random int8 matrix (row-major, n × d).
 */
inline std::vector<int8_t> generate_random_i8(size_t n, size_t d, uint32_t seed = 42) {
    std::vector<int8_t> data(n * d);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& v : data) {
        v = static_cast<int8_t>(dist(rng));
    }
    return data;
}

/**
 * @brief Generate random uint8 matrix (row-major, n × d).
 */
inline std::vector<uint8_t> generate_random_u8(size_t n, size_t d, uint32_t seed = 42) {
    std::vector<uint8_t> data(n * d);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data) {
        v = static_cast<uint8_t>(dist(rng));
    }
    return data;
}

} // namespace bench_utils
