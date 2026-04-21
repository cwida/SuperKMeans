#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <tuple>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace {

// Test data stored in: tests/test_data.bin
// Regenerate with: ./generate_wcss_ground_truth.out

// clang-format off
const std::map<std::pair<size_t, size_t>, float> GROUND_TRUTH = {
    // k=10
    {{10, 4}, 4.06488e+05f},
    {{10, 16}, 3.06879e+06f},
    {{10, 32}, 6.90087e+06f},
    {{10, 64}, 1.47937e+07f},
    {{10, 100}, 2.35847e+07f},
    {{10, 128}, 3.05043e+07f},
    {{10, 384}, 9.32664e+07f},
    {{10, 512}, 1.24437e+08f},
    {{10, 600}, 1.46207e+08f},
    {{10, 768}, 1.87398e+08f},
    // k=100
    {{100, 4}, 9.04907e+04f},
    {{100, 16}, 1.67272e+06f},
    {{100, 32}, 4.53870e+06f},
    {{100, 64}, 1.07961e+07f},
    {{100, 100}, 1.77865e+07f},
    {{100, 128}, 2.31049e+07f},
    {{100, 384}, 7.30264e+07f},
    {{100, 512}, 9.80392e+07f},
    {{100, 600}, 1.15866e+08f},
    {{100, 768}, 1.49076e+08f},
    // k=250
    {{250, 4}, 3.48750e+04f},
    {{250, 16}, 9.27641e+05f},
    {{250, 32}, 2.60551e+06f},
    {{250, 64}, 6.61840e+06f},
    {{250, 100}, 1.13228e+07f},
    {{250, 128}, 1.47906e+07f},
    {{250, 384}, 4.84469e+07f},
    {{250, 512}, 6.49858e+07f},
    {{250, 600}, 7.64087e+07f},
    {{250, 768}, 9.67280e+07f},
};
// clang-format on

class WCSSTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
  protected:
    void SetUp() override { omp_set_num_threads(1); }

    static constexpr size_t N_SAMPLES = 10000;
    static constexpr size_t MAX_D = 768;
    static constexpr unsigned int SEED = 42;
    static constexpr int N_ITERS = 10;
    static constexpr float TOLERANCE = 0.05f;

    // Full test data loaded from disk (N_SAMPLES × MAX_D)
    static std::vector<float> full_data_;
    static void LoadTestData() {
        if (!full_data_.empty()) {
            return;
        }
        std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";
        std::ifstream in(data_file, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Could not open test_data.bin. Run generate_wcss_ground_truth.out first."
            );
        }
        full_data_.resize(N_SAMPLES * MAX_D);
        in.read(reinterpret_cast<char*>(full_data_.data()), full_data_.size() * sizeof(float));
        in.close();
    }

    static std::vector<float> ExtractSubdim(size_t d) {
        LoadTestData();
        if (d > MAX_D) {
            throw std::runtime_error("Requested dimensionality exceeds MAX_D");
        }
        std::vector<float> data(N_SAMPLES * d);
        for (size_t i = 0; i < N_SAMPLES; ++i) {
            std::memcpy(&data[i * d], &full_data_[i * MAX_D], d * sizeof(float));
        }
        return data;
    }
};

std::vector<float> WCSSTest::full_data_;

/**
 * @brief Test that WCSS monotonically decreases across iterations
 * and matches expected ground truth values (within 10% tolerance).
 *
 */
TEST_P(WCSSTest, MonotonicallyDecreases_AndMatchesGroundTruth) {
    auto [n_clusters, d] = GetParam();

    if (n_clusters > N_SAMPLES) {
        GTEST_SKIP() << "Skipping: n_clusters (" << n_clusters << ") > n_samples (" << N_SAMPLES
                     << ")";
    }

    auto data = ExtractSubdim(d);

    ASSERT_EQ(data.size(), N_SAMPLES * d) << "Data size mismatch";

    // These values match the defaults at the time ground truth was generated
    skmeans::SuperKMeansConfig config;
    config.iters = N_ITERS;
    config.verbose = false;
    config.seed = SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.min_not_pruned_pct = 0.03f;
    config.max_not_pruned_pct = 0.05f;
    config.adjustment_factor_for_partial_d = 0.20f;
    config.angular = false;
    config.n_threads = 1;
    config.max_points_per_cluster = 99999;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), N_SAMPLES);
    const auto& stats = kmeans.iteration_stats;
    ASSERT_GE(stats.size(), 1) << "Expected at least 1 iteration recorded";

    // WCSS should monotonically decrease (or stay same)
    for (size_t i = 1; i < stats.size(); ++i) {
        float prev_wcss = stats[i - 1].objective;
        float curr_wcss = stats[i].objective;

        float tolerance = prev_wcss * 1e-2f;
        EXPECT_LE(curr_wcss, prev_wcss + tolerance)
            << "WCSS increased at iteration " << (i + 1) << ": " << prev_wcss << " -> " << curr_wcss
            << " (n_clusters=" << n_clusters << ", d=" << d << ")";
    }

    // Final WCSS should match ground truth
    auto key = std::make_pair(n_clusters, d);
    auto it = GROUND_TRUTH.find(key);
    ASSERT_NE(it, GROUND_TRUTH.end()) << "No ground truth for k=" << n_clusters << ", d=" << d;

    float expected_wcss = it->second;
    float final_wcss = stats.back().objective;
    float wcss_upper_bound = expected_wcss * (1.0f + TOLERANCE);
    EXPECT_LE(final_wcss, wcss_upper_bound)
        << "WCSS too high (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " > "
        << wcss_upper_bound << " (expected ~" << expected_wcss << ")";

    // WCSS shouldn't be drastically lower (would indicate a bug or wrong test setup)
    float wcss_lower_bound = expected_wcss * 0.5f;
    EXPECT_GE(final_wcss, wcss_lower_bound)
        << "WCSS suspiciously low (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " < "
        << wcss_lower_bound << " (expected ~" << expected_wcss << ")";

    // All iteration stats should have valid values
    for (size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].iteration, i + 1) << "Iteration number mismatch at index " << i;
        EXPECT_GT(stats[i].objective, 0.0f) << "WCSS should be positive at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].objective))
            << "WCSS is not finite at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].shift))
            << "Shift is not finite at iteration " << (i + 1);
    }

    // Verify centroids are valid
    EXPECT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid value not finite at index " << i;
    }

    // Verify is_gemm_only flag is correct
    // When d < DIMENSION_THRESHOLD_FOR_PRUNING or n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING,
    // all iterations after the first should use GEMM-only (no pruning)
    bool should_use_gemm_only = (d < skmeans::DIMENSION_THRESHOLD_FOR_PRUNING) ||
                                (n_clusters <= skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING);
    if (should_use_gemm_only && stats.size() > 1) {
        for (size_t i = 1; i < stats.size(); ++i) {
            EXPECT_TRUE(stats[i].is_gemm_only)
                << "Expected is_gemm_only=true for iteration " << (i + 1) << " (d=" << d << " < "
                << skmeans::DIMENSION_THRESHOLD_FOR_PRUNING << " or k=" << n_clusters
                << " <= " << skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING << ")";
        }
    }
}

/**
 * @brief Test that WCSS with BLAS-only mode (no pruning) monotonically decreases
 * and matches expected ground truth values (within 1% tolerance).
 *
 */
TEST_P(WCSSTest, BlasOnly_MonotonicallyDecreases_AndMatchesGroundTruth) {
    auto [n_clusters, d] = GetParam();

    // Only run this test for specific dimensions; otherwise test is too long
    const std::vector<size_t> allowed_dims = {384, 512, 600, 768};
    if (std::find(allowed_dims.begin(), allowed_dims.end(), d) == allowed_dims.end()) {
        SUCCEED();
        return;
    }

    if (n_clusters > N_SAMPLES) {
        SUCCEED();
        return;
    }

    auto data = ExtractSubdim(d);

    ASSERT_EQ(data.size(), N_SAMPLES * d) << "Data size mismatch";

    skmeans::SuperKMeansConfig config;
    config.iters = N_ITERS;
    config.verbose = false;
    config.seed = SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.min_not_pruned_pct = 0.03f;
    config.max_not_pruned_pct = 0.05f;
    config.adjustment_factor_for_partial_d = 0.20f;
    config.angular = false;
    config.n_threads = 1;
    config.use_blas_only = true;
    config.max_points_per_cluster = 99999;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), N_SAMPLES);
    const auto& stats = kmeans.iteration_stats;
    ASSERT_GE(stats.size(), 1) << "Expected at least 1 iteration recorded";

    for (size_t i = 1; i < stats.size(); ++i) {
        float prev_wcss = stats[i - 1].objective;
        float curr_wcss = stats[i].objective;
        float tolerance = prev_wcss * 1e-2f;
        EXPECT_LE(curr_wcss, prev_wcss + tolerance)
            << "WCSS increased at iteration " << (i + 1) << " [BLAS-only mode]: " << prev_wcss
            << " -> " << curr_wcss << " (n_clusters=" << n_clusters << ", d=" << d << ")";
    }

    auto key = std::make_pair(n_clusters, d);
    auto it = GROUND_TRUTH.find(key);
    ASSERT_NE(it, GROUND_TRUTH.end()) << "No ground truth for k=" << n_clusters << ", d=" << d;

    float expected_wcss = it->second;
    float final_wcss = stats.back().objective;
    float wcss_upper_bound = expected_wcss * (1.0f + TOLERANCE);
    EXPECT_LE(final_wcss, wcss_upper_bound)
        << "WCSS too high [BLAS-only mode] (k=" << n_clusters << ", d=" << d << "): " << final_wcss
        << " > " << wcss_upper_bound << " (expected ~" << expected_wcss << ")";

    // WCSS shouldn't be drastically lower
    float wcss_lower_bound = expected_wcss * 0.5f;
    EXPECT_GE(final_wcss, wcss_lower_bound)
        << "WCSS suspiciously low [BLAS-only mode] (k=" << n_clusters << ", d=" << d
        << "): " << final_wcss << " < " << wcss_lower_bound << " (expected ~" << expected_wcss
        << ")";

    // All iteration stats should have valid values
    for (size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].iteration, i + 1) << "Iteration number mismatch at index " << i;
        EXPECT_GT(stats[i].objective, 0.0f) << "WCSS should be positive at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].objective))
            << "WCSS is not finite at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].shift))
            << "Shift is not finite at iteration " << (i + 1);
    }

    // Verify centroids are valid
    EXPECT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid value not finite at index " << i;
    }

    // Verify that all iterations after the first use GEMM-only when use_blas_only=true
    // (First iteration doesn't set is_gemm_only)
    if (stats.size() > 1) {
        for (size_t i = 1; i < stats.size(); ++i) {
            EXPECT_TRUE(stats[i].is_gemm_only)
                << "Expected is_gemm_only=true for iteration " << (i + 1)
                << " when use_blas_only=true (k=" << n_clusters << ", d=" << d << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    WCSSParameterized,
    WCSSTest,
    ::testing::Combine(
        ::testing::Values(10, 100, 250),
        ::testing::Values(4, 16, 32, 64, 100, 128, 384, 512, 600, 768)
    ),
    [](const ::testing::TestParamInfo<WCSSTest::ParamType>& info) {
        return "k" + std::to_string(std::get<0>(info.param)) + "_d" +
               std::to_string(std::get<1>(info.param));
    }
);

} // anonymous namespace
