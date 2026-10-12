#undef HAS_FFTW

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "recall_utils.h"

namespace {

constexpr size_t N_SAMPLES = 10000;
constexpr size_t MAX_D = 1024;
constexpr unsigned int SEED = 42;
constexpr int N_ITERS = 10;
constexpr int ITERS_MESOCLUSTERING = 5;
constexpr int ITERS_FINECLUSTERING = 5;
constexpr int ITERS_REFINEMENT = 2;

const std::vector<size_t> K_VALUES = {10, 100, 250};
const std::vector<size_t> D_VALUES = {4, 16, 32, 64, 100, 128, 384, 512, 600, 768, 1024};

void PrintFlatWCSS(const std::string& data_file) {
    std::cout << "// Ground truth WCSS values for test_wcss.cpp (data: tests/test_data.bin)\n";
    std::cout << "// Copy-paste the following into GROUND_TRUTH map:\n\n";
    for (size_t k : K_VALUES) {
        std::cout << "    // k=" << k << "\n";
        for (size_t d : D_VALUES) {
            auto data = skm_test::LoadTestDataSubdim(data_file, N_SAMPLES, MAX_D, d);

            // These config values MUST match those in test_wcss.cpp
            skmeans::SuperKMeansConfig config;
            config.iters = N_ITERS;
            config.verbose = false;
            config.seed = SEED;
            config.early_termination = false;
            config.sampling_fraction = 1.0f;
            config.max_points_per_cluster = 99999;
            config.min_not_pruned_pct = 0.03f;
            config.max_not_pruned_pct = 0.05f;
            config.adjustment_factor_for_partial_d = 0.20f;
            config.angular = false;
            config.n_threads = 1;

            auto kmeans = skmeans::SuperKMeans(k, d, config);
            kmeans.Train(data.data(), N_SAMPLES);
            const float wcss = kmeans.iteration_stats.back().objective;
            std::cout << "    {{" << k << ", " << d << "}, " << wcss << "f},\n";
        }
    }
}

void PrintHierarchicalWCSS(const std::string& data_file) {
    std::cout << "// Ground truth WCSS values for test_hierarchical_superkmeans.cpp\n";
    std::cout << "// Copy-paste the following into GROUND_TRUTH map:\n\n";
    for (size_t k : K_VALUES) {
        std::cout << "    // k=" << k << "\n";
        for (size_t d : D_VALUES) {
            auto data = skm_test::LoadTestDataSubdim(data_file, N_SAMPLES, MAX_D, d);

            // These config values MUST match those in test_hierarchical_superkmeans.cpp
            skmeans::HierarchicalSuperKMeansConfig config;
            config.iters_mesoclustering = ITERS_MESOCLUSTERING;
            config.iters_fineclustering = ITERS_FINECLUSTERING;
            config.iters_refinement = ITERS_REFINEMENT;
            config.verbose = false;
            config.seed = SEED;
            config.early_termination = false;
            config.sampling_fraction = 1.0f;
            config.max_points_per_cluster = 99999;
            config.min_not_pruned_pct = 0.03f;
            config.max_not_pruned_pct = 0.05f;
            config.adjustment_factor_for_partial_d = 0.20f;
            config.angular = false;
            config.n_threads = 1;

            auto kmeans = skmeans::HierarchicalSuperKMeans(k, d, config);
            kmeans.Train(data.data(), N_SAMPLES);

            const auto& hstats = kmeans.hierarchical_iteration_stats;
            const float wcss = hstats.refinement_iteration_stats.empty()
                                   ? hstats.fineclustering_iteration_stats.back().objective
                                   : hstats.refinement_iteration_stats.back().objective;
            std::cout << "    {{" << k << ", " << d << "}, " << wcss << "f},\n";
        }
    }
}

} // namespace

int main() {
    const std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";

    std::cout << std::scientific << std::setprecision(5);
    PrintFlatWCSS(data_file);
    std::cout << "\n";
    PrintHierarchicalWCSS(data_file);
    return 0;
}
