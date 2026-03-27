#include <cmath>
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

class AssignTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

/**
 * @brief Test that each point is assigned to its actual nearest centroid
 *
 */
TEST_F(AssignTest, EachPointAssignedToNearestCentroid) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 100;
    const int n_iters = 10;
    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;
    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";
    ASSERT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";

    // Verify each assignment using brute-force nearest neighbor
    size_t incorrect_assignments = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* point = data.data() + i * d;
        uint32_t assigned_cluster = assignments[i];
        uint32_t nearest_cluster =
            skmeans::FindNearestCentroidBruteForce(point, centroids.data(), n_clusters, d);
        if (assigned_cluster != nearest_cluster) {
            float assigned_dist = skmeans::ComputeL2DistanceSquared(
                point, centroids.data() + assigned_cluster * d, d
            );
            float nearest_dist =
                skmeans::ComputeL2DistanceSquared(point, centroids.data() + nearest_cluster * d, d);
            if (std::abs(assigned_dist - nearest_dist) > 1e-4f * nearest_dist) {
                ++incorrect_assignments;
            }
        }
    }
    EXPECT_EQ(incorrect_assignments, 0)
        << "Found " << incorrect_assignments << " points not assigned to their nearest centroid";
}

/**
 * @brief Test that each point is assigned to its actual nearest centroid (high-dimensional)
 *
 * Same as above but with higher dimensions to test the DCT rotation path.
 */
TEST_F(AssignTest, EachPointAssignedToNearestCentroid_HighDim) {
    const size_t n = 5000;
    const size_t d = 512;
    const size_t n_clusters = 50;
    const int n_iters = 10;
    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 123);
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 123;
    config.unrotate_centroids = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";
    size_t incorrect_assignments = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* point = data.data() + i * d;
        uint32_t assigned_cluster = assignments[i];
        uint32_t nearest_cluster =
            skmeans::FindNearestCentroidBruteForce(point, centroids.data(), n_clusters, d);

        if (assigned_cluster != nearest_cluster) {
            float assigned_dist = skmeans::ComputeL2DistanceSquared(
                point, centroids.data() + assigned_cluster * d, d
            );
            float nearest_dist =
                skmeans::ComputeL2DistanceSquared(point, centroids.data() + nearest_cluster * d, d);
            if (std::abs(assigned_dist - nearest_dist) > 1e-4f * nearest_dist) {
                ++incorrect_assignments;
            }
        }
    }
    EXPECT_EQ(incorrect_assignments, 0)
        << "Found " << incorrect_assignments
        << " points not assigned to their nearest centroid (high-dim)";
}

/**
 * @brief Test that Assign with use_train_state=true matches use_train_state=false
 *
 * The GEMM+PRUNING fast path should produce at least 98% identical assignments
 * compared to the brute-force path.
 */
TEST_F(AssignTest, UseTrainState_MatchesBruteForce) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;
    const int n_iters = 15;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments_fast =
        kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    auto assignments_brute = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments_fast.size(), n);
    ASSERT_EQ(assignments_brute.size(), n);

    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (assignments_fast[i] == assignments_brute[i]) {
            ++matches;
        }
    }
    double match_pct = 100.0 * static_cast<double>(matches) / static_cast<double>(n);
    EXPECT_GE(
        match_pct, 98.0
    ) << "use_train_state=true should match brute force at least 98% of the time, got "
      << match_pct << "%";
}

/**
 * @brief Test that Assign with use_train_state matches brute force when sampling
 *
 * With sampling_fraction=0.5 the sampled path is exercised (seeding + recompute).
 */
TEST_F(AssignTest, UseTrainState_MatchesBruteForce_Sampled) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;
    const int n_iters = 15;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 0.5f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments_fast =
        kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    auto assignments_brute = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments_fast.size(), n);
    ASSERT_EQ(assignments_brute.size(), n);

    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (assignments_fast[i] == assignments_brute[i]) {
            ++matches;
        }
    }
    double match_pct = 100.0 * static_cast<double>(matches) / static_cast<double>(n);
    EXPECT_GE(
        match_pct, 98.0
    ) << "use_train_state=true (sampled) should match brute force at least 98%, got "
      << match_pct << "%";
}

/**
 * @brief Test that HierarchicalSuperKMeans Assign with use_train_state matches brute force
 */
TEST_F(AssignTest, UseTrainState_MatchesBruteForce_Hierarchical) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 5;
    config.iters_fineclustering = 5;
    config.iters_refinement = 2;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;

    auto kmeans =
        skmeans::HierarchicalSuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config
        );
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments_fast =
        kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    auto assignments_brute = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments_fast.size(), n);
    ASSERT_EQ(assignments_brute.size(), n);

    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (assignments_fast[i] == assignments_brute[i]) {
            ++matches;
        }
    }
    double match_pct = 100.0 * static_cast<double>(matches) / static_cast<double>(n);
    EXPECT_GE(
        match_pct, 98.0
    ) << "HierarchicalSuperKMeans use_train_state=true should match brute force at least 98%, got "
      << match_pct << "%";
}

/**
 * @brief Test that all clusters are non-empty after training
 *
 */
TEST_F(AssignTest, AllClustersNonEmpty) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;
    const int n_iters = 15;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;
    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";

    std::vector<size_t> cluster_counts(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(assignments[i], n_clusters) << "Invalid cluster index " << assignments[i];
        cluster_counts[assignments[i]]++;
    }
    std::vector<size_t> empty_clusters;
    for (size_t c = 0; c < n_clusters; ++c) {
        if (cluster_counts[c] == 0) {
            empty_clusters.push_back(c);
        }
    }
    EXPECT_EQ(empty_clusters.size(), 0)
        << "Found " << empty_clusters.size() << " empty clusters out of " << n_clusters;
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());
    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters << " but only "
        << used_clusters.size() << " were assigned.";
}
