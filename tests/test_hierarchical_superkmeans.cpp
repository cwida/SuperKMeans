#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "recall_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/pdx/utils.h"

class HierarchicalSuperKMeansTest : public ::testing::Test {};

TEST_F(HierarchicalSuperKMeansTest, ConfigSynchronizationWithParent) {
    const size_t n_clusters = 256;
    const size_t d = 128;

    skmeans::HierarchicalSuperKMeansConfig config;
    config.data_already_rotated = true;
    config.unrotate_centroids = true;
    config.iters_mesoclustering = 7;
    config.iters_fineclustering = 9;
    config.iters_refinement = 3;
    config.seed = 123;
    config.sampling_fraction = 0.5f;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);

    EXPECT_FALSE(kmeans.hierarchical_config.unrotate_centroids)
        << "unrotate_centroids should be forced to false when data_already_rotated=true";

    EXPECT_EQ(kmeans.hierarchical_config.iters_mesoclustering, 7)
        << "iters_mesoclustering should not be modified";
    EXPECT_EQ(kmeans.hierarchical_config.iters_fineclustering, 9)
        << "iters_fineclustering should not be modified";
    EXPECT_EQ(kmeans.hierarchical_config.iters_refinement, 3)
        << "iters_refinement should not be modified";

    EXPECT_EQ(kmeans.hierarchical_config.seed, 123)
        << "Other base config fields should be preserved";
    EXPECT_EQ(kmeans.hierarchical_config.sampling_fraction, 0.5f)
        << "Other base config fields should be preserved";
}

TEST_F(HierarchicalSuperKMeansTest, BasicTraining_SmallDataset) {
    const size_t n = 50000;
    const size_t d = 32;
    const size_t n_clusters = 100;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.verbose = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);

    EXPECT_FALSE(kmeans.IsTrained());

    auto centroids = kmeans.Train(data.data(), n);

    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_F(HierarchicalSuperKMeansTest, AllClustersUsed) {
    const size_t n = 10000;
    const size_t d = 128;
    const size_t n_clusters = 256;

    std::vector<float> data =
        skm_test::LoadTestDataSubdim(CMAKE_SOURCE_DIR "/tests/test_data.bin", n, 1024, d);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 10;
    config.iters_fineclustering = 10;
    config.iters_refinement = 4;
    config.verbose = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());

    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters << " but only "
        << used_clusters.size() << " were assigned.";
}

TEST_F(HierarchicalSuperKMeansTest, PerformAssignments_PopulatesAssignments) {
    const size_t n = 50000;
    const size_t d = 64;
    const size_t n_clusters = 128;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.verbose = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    EXPECT_EQ(assignments.size(), n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_LT(assignments[i], n_clusters)
            << "Assignment " << i << " has invalid cluster index: " << assignments[i];
    }
}

TEST_F(HierarchicalSuperKMeansTest, InvalidInputs_ThrowExceptions) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 256;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    // More clusters than data points
    EXPECT_THROW(
        ([&]() {
            auto kmeans = skmeans::HierarchicalSuperKMeans(n + 10, d);
            kmeans.Train(data.data(), n);
        }()),
        std::runtime_error
    );

    // Zero n_clusters
    EXPECT_THROW(
        ([&]() { auto kmeans = skmeans::HierarchicalSuperKMeans(0, d); }()), std::invalid_argument
    );

    // Zero dimensionality
    EXPECT_THROW(
        ([&]() { auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, 0); }()),
        std::invalid_argument
    );

    // Zero mesoclustering iterations in config
    EXPECT_THROW(
        ([&]() {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.iters_mesoclustering = 0;
            auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
        }()),
        std::invalid_argument
    );

    // Zero sampling_fraction
    EXPECT_THROW(
        ([&]() {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.sampling_fraction = 0.0f;
            auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
        }()),
        std::invalid_argument
    );

    // Negative sampling_fraction
    EXPECT_THROW(
        ([&]() {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.sampling_fraction = -0.5f;
            auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
        }()),
        std::invalid_argument
    );

    // sampling_fraction > 1.0
    EXPECT_THROW(
        ([&]() {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.sampling_fraction = 1.5f;
            auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
        }()),
        std::invalid_argument
    );

    // Training twice
    EXPECT_THROW(
        ([&]() {
            auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d);
            kmeans.Train(data.data(), n);
            kmeans.Train(data.data(), n); // Should throw
        }()),
        std::runtime_error
    );
}

TEST_F(HierarchicalSuperKMeansTest, IterationStats_Populated) {
    const size_t n = 50000;
    const size_t d = 64;
    const size_t n_clusters = 256;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 7;
    config.iters_refinement = 3;
    config.verbose = false;
    config.early_termination = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    const auto& hierarchical_stats = kmeans.hierarchical_iteration_stats;

    // Mesoclustering stats
    EXPECT_EQ(hierarchical_stats.mesoclustering_iteration_stats.size(), 5)
        << "Expected 5 mesoclustering iterations";
    for (const auto& stat : hierarchical_stats.mesoclustering_iteration_stats) {
        EXPECT_GT(stat.objective, 0.0f) << "Objective should be positive";
        EXPECT_TRUE(std::isfinite(stat.objective)) << "Objective should be finite";
        EXPECT_GE(stat.shift, 0.0f) << "Shift should be non-negative";
    }

    // Fineclustering stats
    // With early_termination=false, we expect exactly n_mesoclusters * iters_fineclustering
    // iterations
    size_t n_mesoclusters = skmeans::HierarchicalSuperKMeans<>::GetNMesoclusters(n_clusters);
    size_t expected_fineclustering_iters = n_mesoclusters * config.iters_fineclustering;
    EXPECT_EQ(
        hierarchical_stats.fineclustering_iteration_stats.size(), expected_fineclustering_iters
    ) << "Expected "
      << expected_fineclustering_iters << " fineclustering iterations (" << n_mesoclusters
      << " mesoclusters * " << config.iters_fineclustering << " iters)";

    for (const auto& stat : hierarchical_stats.fineclustering_iteration_stats) {
        EXPECT_GT(stat.objective, 0.0f) << "Objective should be positive";
        EXPECT_TRUE(std::isfinite(stat.objective)) << "Objective should be finite";
        EXPECT_GE(stat.shift, 0.0f) << "Shift should be non-negative";
    }

    // Refinement stats
    EXPECT_EQ(hierarchical_stats.refinement_iteration_stats.size(), 3)
        << "Expected 3 refinement iterations";
    for (const auto& stat : hierarchical_stats.refinement_iteration_stats) {
        EXPECT_GT(stat.objective, 0.0f) << "Objective should be positive";
        EXPECT_TRUE(std::isfinite(stat.objective)) << "Objective should be finite";
        EXPECT_GE(stat.shift, 0.0f) << "Shift should be non-negative";
    }
}

TEST_F(HierarchicalSuperKMeansTest, Objective_MonotonicallyDecreases) {
    const size_t n = 10000;
    const size_t d = 128;
    const size_t n_clusters = 256;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 10;
    config.iters_fineclustering = 10;
    config.iters_refinement = 5;
    config.verbose = false;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    const auto& hierarchical_stats = kmeans.hierarchical_iteration_stats;

    const auto& meso_stats = hierarchical_stats.mesoclustering_iteration_stats;
    for (size_t i = 1; i < meso_stats.size(); ++i) {
        float prev_obj = meso_stats[i - 1].objective;
        float curr_obj = meso_stats[i].objective;
        float tolerance = prev_obj * 1e-2f;
        EXPECT_LE(curr_obj, prev_obj + tolerance)
            << "Mesoclustering objective increased at iteration " << (i + 1) << ": " << prev_obj
            << " -> " << curr_obj;
    }

    const auto& fine_stats = hierarchical_stats.fineclustering_iteration_stats;
    const size_t iters_fine = config.iters_fineclustering;
    for (size_t i = iters_fine; i < fine_stats.size(); i += iters_fine) {
        float prev_mesocluster_final_obj = fine_stats[i - iters_fine].objective;
        float curr_mesocluster_final_obj = fine_stats[i - 1].objective;
        float tolerance = prev_mesocluster_final_obj * 1e-2f;
        EXPECT_LE(curr_mesocluster_final_obj, prev_mesocluster_final_obj + tolerance)
            << "Fineclustering objective increased between mesoclusters at iteration " << (i + 1)
            << ": " << prev_mesocluster_final_obj << " -> " << curr_mesocluster_final_obj;
    }

    const auto& refine_stats = hierarchical_stats.refinement_iteration_stats;
    for (size_t i = 1; i < refine_stats.size(); ++i) {
        float prev_obj = refine_stats[i - 1].objective;
        float curr_obj = refine_stats[i].objective;
        float tolerance = prev_obj * 1e-2f;
        EXPECT_LE(curr_obj, prev_obj + tolerance)
            << "Refinement objective increased at iteration " << (i + 1) << ": " << prev_obj
            << " -> " << curr_obj;
    }

    // Verify centroids are valid
    EXPECT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid value not finite at index " << i;
    }
}

TEST_F(HierarchicalSuperKMeansTest, EarlyTermination_Mesoclustering) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 256;
    const size_t max_iters = 100;

    // Generate well-separated blobs
    std::mt19937 gen(42);
    std::vector<float> data(n * d);
    std::vector<std::vector<float>> centers(n_clusters, std::vector<float>(d));
    for (size_t c = 0; c < n_clusters; ++c) {
        for (size_t j = 0; j < d; ++j) {
            centers[c][j] = static_cast<float>(c) * 20.0f + (j % 2 == 0 ? 5.0f : -5.0f);
        }
    }
    std::normal_distribution<float> noise(0.0f, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        size_t cluster = i % n_clusters;
        for (size_t j = 0; j < d; ++j) {
            data[i * d + j] = centers[cluster][j] + noise(gen);
        }
    }

    // Test WITH early termination
    skmeans::HierarchicalSuperKMeansConfig config_early;
    config_early.iters_mesoclustering = max_iters;
    config_early.iters_fineclustering = 5;
    config_early.iters_refinement = 2;
    config_early.early_termination = true;
    config_early.tol = 1e-2f;
    config_early.verbose = false;
    config_early.seed = 42;
    config_early.sampling_fraction = 1.0f;

    auto kmeans_early = skmeans::HierarchicalSuperKMeans(n_clusters, d, config_early);
    kmeans_early.Train(data.data(), n);
    const auto& stats_early =
        kmeans_early.hierarchical_iteration_stats.mesoclustering_iteration_stats;
    size_t iters_with_early = stats_early.size();

    // Test WITHOUT early termination
    skmeans::HierarchicalSuperKMeansConfig config_no_early;
    config_no_early.iters_mesoclustering = max_iters;
    config_no_early.iters_fineclustering = 5;
    config_no_early.iters_refinement = 2;
    config_no_early.early_termination = false;
    config_no_early.verbose = false;
    config_no_early.seed = 42;
    config_no_early.sampling_fraction = 1.0f;

    auto kmeans_no_early = skmeans::HierarchicalSuperKMeans(n_clusters, d, config_no_early);
    kmeans_no_early.Train(data.data(), n);
    const auto& stats_no_early =
        kmeans_no_early.hierarchical_iteration_stats.mesoclustering_iteration_stats;
    size_t iters_without_early = stats_no_early.size();

    // With early termination, should stop before max_iters
    EXPECT_LT(iters_with_early, max_iters)
        << "Early termination should stop mesoclustering before max_iters=" << max_iters;

    // Without early termination, should run all iterations
    EXPECT_EQ(iters_without_early, max_iters) << "Without early termination, should run all "
                                              << max_iters << " mesoclustering iterations";

    // Early termination should use fewer iterations
    EXPECT_LT(iters_with_early, iters_without_early)
        << "Early termination (" << iters_with_early << " iters) should use fewer iterations "
        << "than no early termination (" << iters_without_early << " iters)";
}

TEST_F(HierarchicalSuperKMeansTest, SLOW_Sampling_AffectsSpeed) {
    const size_t n = 100000;
    const size_t d = 512;
    const size_t n_clusters = 512;
    const size_t n_runs = 5;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig base_config;
    base_config.iters_mesoclustering = 10;
    base_config.iters_fineclustering = 10;
    base_config.iters_refinement = 2;
    base_config.early_termination = false;
    base_config.verbose = false;

    // Full sampling
    skmeans::TicToc timer_full;
    for (size_t i = 0; i < n_runs; ++i) {
        skmeans::HierarchicalSuperKMeansConfig config = base_config;
        config.sampling_fraction = 1.0f;
        config.seed = static_cast<uint32_t>(42 + i);

        auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);

        timer_full.Tic();
        kmeans.Train(data.data(), n);
        timer_full.Toc();
    }

    // 30% sampling
    skmeans::TicToc timer_sampled;
    for (size_t i = 0; i < n_runs; ++i) {
        skmeans::HierarchicalSuperKMeansConfig config = base_config;
        config.sampling_fraction = 0.3f;
        config.seed = static_cast<uint32_t>(42 + i);

        auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);

        timer_sampled.Tic();
        kmeans.Train(data.data(), n);
        timer_sampled.Toc();
    }

    double full_time_ms = timer_full.accum_time / 1e6;
    double sampled_time_ms = timer_sampled.accum_time / 1e6;
    double speedup = full_time_ms / sampled_time_ms;

    // Sampling should provide some speedup (at least 1.5x)
    EXPECT_GE(speedup, 1.5) << "Sampling should provide at least 1.5x speedup. "
                            << "Full: " << full_time_ms << "ms, Sampled: " << sampled_time_ms
                            << "ms, "
                            << "Speedup: " << speedup << "x";
}

TEST_F(HierarchicalSuperKMeansTest, Reproducibility_SameSeed) {
    const size_t n = 10000;
    const size_t d = 128;
    const size_t n_clusters = 256;
    const uint32_t seed = 12345;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 10;
    config.iters_fineclustering = 10;
    config.iters_refinement = 2;
    config.verbose = false;
    config.seed = seed;
    config.sampling_fraction = 1.0f;

    // First run
    auto kmeans1 = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids1 = kmeans1.Train(data.data(), n);

    // Second run with same seed
    auto kmeans2 = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids2 = kmeans2.Train(data.data(), n);

    // Centroids should be identical
    ASSERT_EQ(centroids1.size(), centroids2.size());
    for (size_t i = 0; i < centroids1.size(); ++i) {
        EXPECT_FLOAT_EQ(centroids1[i], centroids2[i])
            << "Centroid mismatch at index " << i << " with same seed";
    }
}

TEST_F(HierarchicalSuperKMeansTest, SmallClusters_PrintsWarning) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 64; // < 128, should warn

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.verbose = false;

    // Should print warning for n_clusters < 128
    testing::internal::CaptureStdout();
    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("WARNING") != std::string::npos) << "Should warn for n_clusters < 128";
}

TEST_F(HierarchicalSuperKMeansTest, AssignMethod_ProducesValidAssignments) {
    const size_t n = 100000;
    const size_t d = 128;
    const size_t n_clusters = 256;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.sampling_fraction = 1.0f;
    config.verbose = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    // Use Assign method
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    EXPECT_EQ(assignments.size(), n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_LT(assignments[i], n_clusters)
            << "Assignment " << i << " has invalid cluster index: " << assignments[i];
    }

    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());
    EXPECT_GE(used_clusters.size(), static_cast<size_t>(0.95 * n_clusters))
        << "Less than 95% of clusters were used in assignments";
}

TEST_F(HierarchicalSuperKMeansTest, AngularMode_Normalizes) {
    const size_t n = 5000;
    const size_t d = 64;
    const size_t n_clusters = 256;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.angular = true;
    config.verbose = false;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    EXPECT_EQ(centroids.size(), n_clusters * d);

    for (size_t c = 0; c < n_clusters; ++c) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float val = centroids[c * d + j];
            norm += val * val;
        }
        norm = std::sqrt(norm);

        EXPECT_NEAR(norm, 1.0f, 1e-5f)
            << "Centroid " << c << " should be normalized in angular mode (norm=" << norm << ")";
    }
}

TEST_F(HierarchicalSuperKMeansTest, Quantized_AssignPathsValid) {
    const size_t n = 15000;
    const size_t d = 128;
    const size_t n_clusters = 300;
    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    auto run = [&](auto tag, int iters_refinement) {
        constexpr skmeans::Quantization qt = decltype(tag)::value;
        SCOPED_TRACE(
            std::string(skmeans::QuantizationName(qt)) +
            " iters_refinement=" + std::to_string(iters_refinement)
        );
        {
            skmeans::HierarchicalSuperKMeansConfig config;
            config.iters_mesoclustering = 5;
            config.iters_fineclustering = 5;
            config.iters_refinement = iters_refinement;
            config.seed = 42;
            config.sampling_fraction = 1.0f;

            auto kmeans = skmeans::HierarchicalSuperKMeans<qt>(n_clusters, d, config);
            auto centroids = kmeans.Train(data.data(), n);
            ASSERT_EQ(centroids.size(), n_clusters * d);

            auto exact = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
            auto quantized = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);
            auto reuse = kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);

            ASSERT_EQ(exact.size(), n);
            ASSERT_EQ(quantized.size(), n);
            ASSERT_EQ(reuse.size(), n);
            for (size_t i = 0; i < n; ++i) {
                ASSERT_LT(exact[i], n_clusters);
                ASSERT_LT(quantized[i], n_clusters);
                ASSERT_LT(reuse[i], n_clusters);
            }

            std::unordered_set<uint32_t> used(exact.begin(), exact.end());
            EXPECT_GE(used.size(), static_cast<size_t>(0.80 * n_clusters));

            size_t agree_quant_exact = 0, agree_reuse_quant = 0;
            for (size_t i = 0; i < n; ++i) {
                if (quantized[i] == exact[i])
                    agree_quant_exact++;
                if (reuse[i] == quantized[i])
                    agree_reuse_quant++;
            }
            EXPECT_GE(static_cast<double>(agree_quant_exact) / n, 0.85)
                << "quantized vs exact agreement too low";
            if (qt == skmeans::Quantization::rabitq) {
                EXPECT_GE(static_cast<double>(agree_reuse_quant) / n, 0.90)
                    << "reuse vs quantized agreement too low";
            }
        }
    };
    for (int iters_refinement : {0, 1}) {
        run(skmeans::QuantizationTag<skmeans::Quantization::sq8>{}, iters_refinement);
        run(skmeans::QuantizationTag<skmeans::Quantization::lvq4>{}, iters_refinement);
        run(skmeans::QuantizationTag<skmeans::Quantization::rabitq>{}, iters_refinement);
    }
}

template <typename WrapperT>
class HierarchicalQuantizedWrapperTest : public ::testing::Test {};

using HierarchicalWrapperTypes = ::testing::Types<
    skmeans::HierarchicalSuperKMeansSQ8,
    skmeans::HierarchicalSuperKMeansLVQ4,
    skmeans::HierarchicalSuperKMeansRabitQ>;
TYPED_TEST_SUITE(HierarchicalQuantizedWrapperTest, HierarchicalWrapperTypes);

TYPED_TEST(HierarchicalQuantizedWrapperTest, DefaultConfigForcesSchemeAndTrains) {
    const size_t n = 15000, d = 128, n_clusters = 300;
    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.seed = 42;

    TypeParam kmeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    ASSERT_EQ(centroids.size(), n_clusters * d);

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    ASSERT_EQ(assignments.size(), n);
    for (size_t i = 0; i < n; ++i)
        ASSERT_LT(assignments[i], n_clusters) << "at " << i;
}

TEST_F(HierarchicalSuperKMeansTest, Quantized_AssignTrainingPointsBlasOnlyReuse) {
    const size_t n = 15000;
    const size_t d = 128;
    const size_t n_clusters = 300;
    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    auto run = [&](auto tag) {
        constexpr skmeans::Quantization qt = decltype(tag)::value;
        SCOPED_TRACE(skmeans::QuantizationName(qt));
        skmeans::HierarchicalSuperKMeansConfig config;
        config.iters_mesoclustering = 5;
        config.iters_fineclustering = 5;
        config.iters_refinement = 0;
        config.seed = 42;
        config.sampling_fraction = 1.0f;
        config.use_blas_only = true;

        auto kmeans = skmeans::HierarchicalSuperKMeans<qt>(n_clusters, d, config);
        auto centroids = kmeans.Train(data.data(), n);
        ASSERT_EQ(centroids.size(), n_clusters * d);

        auto exact = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
        auto reuse = kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);

        ASSERT_EQ(reuse.size(), n);
        size_t agree_reuse_exact = 0;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_LT(reuse[i], n_clusters);
            if (reuse[i] == exact[i])
                agree_reuse_exact++;
        }
        EXPECT_GE(static_cast<double>(agree_reuse_exact) / n, 0.85)
            << "gemm-only reuse vs exact agreement too low";
    };
    run(skmeans::QuantizationTag<skmeans::Quantization::sq8>{});
    run(skmeans::QuantizationTag<skmeans::Quantization::lvq4>{});
    run(skmeans::QuantizationTag<skmeans::Quantization::rabitq>{});
}

namespace {
void ExpectHierarchicalTrainAndAssignValid(
    const std::vector<float>& data,
    size_t n,
    size_t d,
    size_t n_clusters
) {
    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 3;
    config.iters_fineclustering = 3;
    config.iters_refinement = 1;
    config.seed = 42;
    config.sampling_fraction = 1.0f;

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    ASSERT_EQ(centroids.size(), n_clusters * d);
    for (float v : centroids)
        ASSERT_TRUE(std::isfinite(v));

    auto a = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    ASSERT_EQ(a.size(), n);
    for (size_t i = 0; i < n; ++i)
        ASSERT_LT(a[i], n_clusters);
}
} // namespace

TEST_F(HierarchicalSuperKMeansTest, EdgeCase_SingleCluster) {
    const size_t n = 500, d = 16, n_clusters = 1;
    std::vector<float> data = skmeans::MakeBlobs(n, d, 5, false, 1.0f, 10.0f, 42);
    ExpectHierarchicalTrainAndAssignValid(data, n, d, n_clusters);
}

TEST_F(HierarchicalSuperKMeansTest, EdgeCase_ClustersEqualPoints) {
    const size_t n = 128, d = 16, n_clusters = 128;
    std::vector<float> data = skmeans::MakeBlobs(n, d, 16, false, 1.0f, 10.0f, 42);
    ExpectHierarchicalTrainAndAssignValid(data, n, d, n_clusters);
}

TEST_F(HierarchicalSuperKMeansTest, EdgeCase_SinglePoint) {
    const size_t n = 1, d = 16, n_clusters = 1;
    std::vector<float> data(d);
    for (size_t j = 0; j < d; ++j)
        data[j] = static_cast<float>(j) * 0.1f + 0.5f;
    ExpectHierarchicalTrainAndAssignValid(data, n, d, n_clusters);
}

// =============================================================================
// WCSS Ground Truth Tests
// =============================================================================

namespace {

// Regenerate with: ./generate_wcss_ground_truth.out
// Data: tests/test_data.bin; ITERS_MESOCLUSTERING=5, ITERS_FINECLUSTERING=5, ITERS_REFINEMENT=2

// clang-format off
const std::map<std::pair<size_t, size_t>, float> HIERARCHICAL_GROUND_TRUTH = {
    // k=10
    {{10, 4}, 3.95821e+03f},
    {{10, 16}, 2.80619e+04f},
    {{10, 32}, 5.99182e+04f},
    {{10, 64}, 1.23962e+05f},
    {{10, 100}, 1.94923e+05f},
    {{10, 128}, 2.52417e+05f},
    {{10, 384}, 7.59521e+05f},
    {{10, 512}, 1.01746e+06f},
    {{10, 600}, 1.18858e+06f},
    {{10, 768}, 1.51884e+06f},
    {{10, 1024}, 2.02314e+06f},
    // k=100
    {{100, 4}, 1.35213e+03f},
    {{100, 16}, 2.07180e+04f},
    {{100, 32}, 4.89141e+04f},
    {{100, 64}, 1.04607e+05f},
    {{100, 100}, 1.66228e+05f},
    {{100, 128}, 2.16121e+05f},
    {{100, 384}, 6.55114e+05f},
    {{100, 512}, 8.75961e+05f},
    {{100, 600}, 1.02503e+06f},
    {{100, 768}, 1.31593e+06f},
    {{100, 1024}, 1.74501e+06f},
    // k=250
    {{250, 4}, 8.43353e+02f},
    {{250, 16}, 1.78634e+04f},
    {{250, 32}, 4.45185e+04f},
    {{250, 64}, 9.64738e+04f},
    {{250, 100}, 1.54322e+05f},
    {{250, 128}, 2.00625e+05f},
    {{250, 384}, 6.10249e+05f},
    {{250, 512}, 8.14914e+05f},
    {{250, 600}, 9.57009e+05f},
    {{250, 768}, 1.22039e+06f},
    {{250, 1024}, 1.62301e+06f},
};
// clang-format on

class HierarchicalWCSSTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
  protected:
    static constexpr size_t N_SAMPLES = 10000;
    static constexpr size_t MAX_D = 1024;
    static constexpr unsigned int SEED = 42;
    static constexpr int ITERS_MESOCLUSTERING = 5;
    static constexpr int ITERS_FINECLUSTERING = 5;
    static constexpr int ITERS_REFINEMENT = 2;
    static constexpr float TOLERANCE = 0.10f;

    static std::vector<float> full_data_;
    static void LoadTestData() {
        if (!full_data_.empty())
            return;
        std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";
        std::ifstream in(data_file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open test_data.bin.");
        }
        full_data_.resize(N_SAMPLES * MAX_D);
        in.read(reinterpret_cast<char*>(full_data_.data()), full_data_.size() * sizeof(float));
        in.close();
    }

    static std::vector<float> ExtractSubdim(size_t d) {
        LoadTestData();
        std::vector<float> data(N_SAMPLES * d);
        for (size_t i = 0; i < N_SAMPLES; ++i) {
            std::memcpy(&data[i * d], &full_data_[i * MAX_D], d * sizeof(float));
        }
        return data;
    }
};

std::vector<float> HierarchicalWCSSTest::full_data_;

/**
 * @brief Test that final WCSS from HierarchicalSuperKMeans matches the ground truth
 * within tolerance, and that WCSS within each mesocluster's fineclustering phase
 * decreases monotonically.
 */
TEST_P(HierarchicalWCSSTest, MatchesGroundTruth_AndFineClusteringDecreases) {
    auto [n_clusters, d] = GetParam();

    auto data = ExtractSubdim(d);
    ASSERT_EQ(data.size(), N_SAMPLES * d) << "Data size mismatch";

    // These values MUST match those in generate_wcss_ground_truth.cpp
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

    auto kmeans = skmeans::HierarchicalSuperKMeans(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), N_SAMPLES);

    const auto& hierarchical_stats = kmeans.hierarchical_iteration_stats;

    // Final WCSS comes from last refinement iteration
    ASSERT_FALSE(hierarchical_stats.refinement_iteration_stats.empty())
        << "Expected refinement stats to be populated";
    float final_wcss = hierarchical_stats.refinement_iteration_stats.back().objective;

    // Compare against ground truth
    auto key = std::make_pair(n_clusters, d);
    auto it = HIERARCHICAL_GROUND_TRUTH.find(key);
    ASSERT_NE(it, HIERARCHICAL_GROUND_TRUTH.end())
        << "No ground truth for k=" << n_clusters << ", d=" << d;

    float expected_wcss = it->second;
    EXPECT_LE(final_wcss, expected_wcss * (1.0f + TOLERANCE))
        << "WCSS too high (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " > "
        << expected_wcss * (1.0f + TOLERANCE) << " (expected ~" << expected_wcss << ")";
    EXPECT_GE(final_wcss, expected_wcss * 0.5f)
        << "WCSS suspiciously low (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " < "
        << expected_wcss * 0.5f << " (expected ~" << expected_wcss << ")";

    // Refinement WCSS should decrease monotonically
    const auto& ref_stats = hierarchical_stats.refinement_iteration_stats;
    for (size_t i = 1; i < ref_stats.size(); ++i) {
        float prev = ref_stats[i - 1].objective;
        float curr = ref_stats[i].objective;
        EXPECT_LE(curr, prev * (1.0f + 1e-2f))
            << "Refinement WCSS increased at iteration " << (i + 1) << ": " << prev << " -> "
            << curr << " (k=" << n_clusters << ", d=" << d << ")";
    }

    // Within each mesocluster's fineclustering block, WCSS should decrease
    const auto& fine_stats = hierarchical_stats.fineclustering_iteration_stats;
    const size_t n_mesoclusters = skmeans::HierarchicalSuperKMeans<>::GetNMesoclusters(n_clusters);
    const size_t iters_fine = ITERS_FINECLUSTERING;
    ASSERT_EQ(fine_stats.size(), n_mesoclusters * iters_fine)
        << "Expected " << n_mesoclusters * iters_fine << " fineclustering iterations";
    for (size_t m = 0; m < n_mesoclusters; ++m) {
        for (size_t i = 1; i < iters_fine; ++i) {
            size_t idx = m * iters_fine + i;
            float prev = fine_stats[idx - 1].objective;
            float curr = fine_stats[idx].objective;
            EXPECT_LE(curr, prev * (1.0f + 1e-2f))
                << "Fineclustering WCSS increased at mesocluster=" << m << " iter=" << (i + 1)
                << ": " << prev << " -> " << curr << " (k=" << n_clusters << ", d=" << d << ")";
        }
    }

    // Centroids should be finite
    EXPECT_EQ(centroids.size(), n_clusters * d);
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid not finite at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    HierarchicalWCSSParameterized,
    HierarchicalWCSSTest,
    ::testing::Combine(
        ::testing::Values(10, 100, 250),
        ::testing::Values(4, 16, 32, 64, 100, 128, 384, 512, 600, 768, 1024)
    ),
    [](const ::testing::TestParamInfo<HierarchicalWCSSTest::ParamType>& info) {
        return "k" + std::to_string(std::get<0>(info.param)) + "_d" +
               std::to_string(std::get<1>(info.param));
    }
);

// IVF-recall ground truth (test_data.bin, mxbai 10k x 1024)

TEST(HierarchicalRecallGroundTruthTest, F32_MatchesGroundTruth) {
    float recall =
        skm_test::HierarchicalClusteringRecall<skmeans::Quantization::f32>(CMAKE_SOURCE_DIR
                                                                           "/tests/test_data.bin");
    EXPECT_GE(recall, skm_test::RECALL_GROUND_TRUTH.at("hierarchical_f32") - skm_test::RECALL_TOL);
}

using QuantizedTags = ::testing::Types<
    skmeans::QuantizationTag<skmeans::Quantization::sq8>,
    skmeans::QuantizationTag<skmeans::Quantization::lvq4>,
    skmeans::QuantizationTag<skmeans::Quantization::rabitq>>;

struct QuantizerNames {
    template <typename T>
    static std::string GetName(int) {
        return skmeans::QuantizationName(T::value);
    }
};

template <typename T>
class HierarchicalQuantizedRecallTest : public ::testing::Test {};

TYPED_TEST_SUITE(HierarchicalQuantizedRecallTest, QuantizedTags, QuantizerNames);

TYPED_TEST(HierarchicalQuantizedRecallTest, MatchesGroundTruth) {
    float recall = skm_test::HierarchicalClusteringRecall<TypeParam::value>(CMAKE_SOURCE_DIR
                                                                            "/tests/test_data.bin");
    const std::string gt_key =
        std::string("hierarchical_") + skmeans::QuantizationName(TypeParam::value);
    EXPECT_GE(recall, skm_test::RECALL_GROUND_TRUTH.at(gt_key) - skm_test::RECALL_TOL);
}

TEST(HierarchicalRecallGroundTruthTest, F32_TrainInPlace_RecallMatchesTrain) {
    const char* path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float recall = skm_test::HierarchicalClusteringRecall<skmeans::Quantization::f32>(path);
    float recall_in_place =
        skm_test::HierarchicalClusteringRecall<skmeans::Quantization::f32, true>(path);
    EXPECT_NEAR(recall_in_place, recall, skm_test::RECALL_TOL);
}

TYPED_TEST(HierarchicalQuantizedRecallTest, TrainInPlace_RecallMatchesTrain) {
    const char* path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float recall = skm_test::HierarchicalClusteringRecall<TypeParam::value>(path);
    float recall_in_place = skm_test::HierarchicalClusteringRecall<TypeParam::value, true>(path);
    EXPECT_NEAR(recall_in_place, recall, skm_test::RECALL_TOL);
}

template <typename T>
class HierarchicalQuantizedStateTest : public ::testing::Test {};

TYPED_TEST_SUITE(HierarchicalQuantizedStateTest, QuantizedTags, QuantizerNames);

TYPED_TEST(HierarchicalQuantizedStateTest, ExposesQuantizerAndQuantizedBuffer) {
    const size_t n = 8000, d = 128, k = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 2;
    config.iters_fineclustering = 2;
    config.seed = 42;
    config.n_threads = 1;
    config.suppress_warnings = true;
    auto kmeans = skmeans::HierarchicalSuperKMeans<TypeParam::value>(k, d, config);
    EXPECT_EQ(kmeans.GetQuantizer(), nullptr);
    kmeans.Train(data.data(), n);

    const auto* quantizer = kmeans.GetQuantizer();
    ASSERT_NE(quantizer, nullptr);
    EXPECT_TRUE(quantizer->IsFitted());
    EXPECT_NE(kmeans.GetQuantizedData(), nullptr);

    const auto& state = kmeans.GetState();
    EXPECT_EQ(state.code_size, quantizer->CodeSize(d));
    EXPECT_EQ(state.n_encoded, n);
    EXPECT_NE(state.rotator, nullptr);
}

} // anonymous namespace
