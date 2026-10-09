#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace {

/**
 * @brief Compute L2 norm of a vector
 */
float ComputeNorm(const float* v, size_t d) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        norm_sq += v[i] * v[i];
    }
    return std::sqrt(norm_sq);
}

/**
 * @brief Compute inner product of two vectors
 */
float ComputeInnerProduct(const float* a, const float* b, size_t d) {
    return skmeans::ScalarComputer<skmeans::DistanceFunction::dp, skmeans::Quantization::f32>::
        Horizontal(a, b, d);
}

/**
 * @brief Get rotation method name for a given dimension
 */
std::string GetRotationMethod(size_t d) {
    // Note: On x86 with AVX2, DCT requires power-of-2 dimensions
#ifdef __AVX2__
    bool uses_dct = (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION) &&
                    skmeans::IsPowerOf2(static_cast<uint32_t>(d));
#else
    bool uses_dct = (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION);
#endif
    return uses_dct ? "DCT" : "Orthonormal Matrix";
}

class RotationTest : public ::testing::Test {};

/**
 * @brief Test that Rotate followed by Unrotate returns original vectors (low dim)
 */
TEST_F(RotationTest, RotateUnrotateInverse_LowDim) {
    const size_t d = 128;
    const size_t n = 100;

    auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

    skmeans::ADSamplingPruner pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

/**
 * @brief Test that Rotate followed by Unrotate returns original vectors (high dim, DCT)
 */
TEST_F(RotationTest, RotateUnrotateInverse_HighDim_DCT) {
    const size_t d = 1024;
    const size_t n = 100;

    ASSERT_GE(d, skmeans::D_THRESHOLD_FOR_DCT_ROTATION)
        << "Test expects d >= D_THRESHOLD_FOR_DCT_ROTATION to use DCT rotation";

    auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

    skmeans::ADSamplingPruner pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

/**
 * @brief Test Rotate/Unrotate inverse across multiple dimensions
 */
TEST_F(RotationTest, RotateUnrotateInverse_MultipleDimensions) {
    std::vector<size_t> dimensions = {50, 128, 256, 512, 768, 1024, 2048};
    const size_t n = 100;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        std::vector<float> recovered(n * d);
        pruner.Unrotate(rotated.data(), recovered.data(), n);

        double max_error = 0.0;
        double sum_error = 0.0;
        for (size_t i = 0; i < n * d; ++i) {
            double error = std::abs(original[i] - recovered[i]);
            max_error = std::max(max_error, error);
            sum_error += error;
        }
        double avg_error = sum_error / (n * d);

        EXPECT_LT(max_error, 1e-4)
            << "Max error too large for d=" << d << " (" << GetRotationMethod(d) << ")";
        EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
    }
}

/**
 * @brief Test that both rotation methods preserve vector L2 norms
 *
 * An orthogonal transformation (rotation) preserves the L2 norm of vectors.
 * This is a fundamental property that ensures distances are preserved.
 */
TEST_F(RotationTest, BothRotationMethodsPreserveNorms) {
    // Test dimensions that use orthonormal matrix (d < 512)
    std::vector<size_t> orthonormal_dims = {64, 128, 256, 384};
    // Test dimensions that use DCT (d >= 512, power of 2 on x86)
    std::vector<size_t> dct_dims = {512, 1024, 2048};

    const size_t n = 50;

    // Test orthonormal matrix rotation
    for (size_t d : orthonormal_dims) {
        SCOPED_TRACE("Testing orthonormal rotation d=" + std::to_string(d));

        auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);
        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        for (size_t i = 0; i < n; ++i) {
            float original_norm = ComputeNorm(original.data() + i * d, d);
            float rotated_norm = ComputeNorm(rotated.data() + i * d, d);

            float rel_error = std::abs(original_norm - rotated_norm) / original_norm;
            EXPECT_LT(rel_error, 1e-5f)
                << "Orthonormal rotation should preserve norm for vector " << i << " at d=" << d
                << " (original=" << original_norm << ", rotated=" << rotated_norm << ")";
        }
    }

    // Test DCT rotation
    for (size_t d : dct_dims) {
        SCOPED_TRACE("Testing DCT rotation d=" + std::to_string(d));

        auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 123);
        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        for (size_t i = 0; i < n; ++i) {
            float original_norm = ComputeNorm(original.data() + i * d, d);
            float rotated_norm = ComputeNorm(rotated.data() + i * d, d);

            float rel_error = std::abs(original_norm - rotated_norm) / original_norm;
            EXPECT_LT(rel_error, 1e-5f)
                << "DCT rotation should preserve norm for vector " << i << " at d=" << d
                << " (original=" << original_norm << ", rotated=" << rotated_norm << ")";
        }
    }
}

/**
 * @brief Test that rotation preserves inner products between vectors
 *
 * For an orthogonal transformation Q: <Qx, Qy> = <x, y>
 * This is essential because L2 distance is derived from inner products:
 * ||x - y||² = ||x||² + ||y||² - 2<x, y>
 *
 * If inner products are preserved, distances are preserved.
 */
TEST_F(RotationTest, RotationPreservesInnerProducts) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};
    const size_t n = 30;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto vectors = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(vectors.data(), rotated.data(), n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const float* vi_orig = vectors.data() + i * d;
                const float* vj_orig = vectors.data() + j * d;
                const float* vi_rot = rotated.data() + i * d;
                const float* vj_rot = rotated.data() + j * d;

                float orig_dot = ComputeInnerProduct(vi_orig, vj_orig, d);
                float rot_dot = ComputeInnerProduct(vi_rot, vj_rot, d);

                float abs_error = std::abs(orig_dot - rot_dot);

                // For very small dot products, use absolute error threshold
                // For larger dot products, use relative error threshold
                // Both should be satisfied for orthogonal transformations within numerical
                // precision
                bool abs_ok = abs_error < 1e-4f;
                float rel_error = abs_error / std::max(std::abs(orig_dot), 1.0f);
                bool rel_ok = rel_error < 1e-3f;

                EXPECT_TRUE(abs_ok || rel_ok)
                    << "Inner product not preserved for vectors " << i << " and " << j
                    << " at d=" << d << " (" << GetRotationMethod(d) << ")"
                    << ": original=" << orig_dot << ", rotated=" << rot_dot
                    << " (abs_err=" << abs_error << ", rel_err=" << rel_error << ")";
            }
        }
    }
}

/**
 * @brief Test that rotation preserves L2 distances (derived from inner products)
 */
TEST_F(RotationTest, RotationPreservesDistances) {
    std::vector<size_t> dimensions = {128, 512, 1024};
    const size_t n = 20;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto vectors = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(vectors.data(), rotated.data(), n);

        // Check distances for all pairs
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const float* vi_orig = vectors.data() + i * d;
                const float* vj_orig = vectors.data() + j * d;
                const float* vi_rot = rotated.data() + i * d;
                const float* vj_rot = rotated.data() + j * d;

                // Compute L2 distance squared
                float orig_dist_sq = 0.0f;
                float rot_dist_sq = 0.0f;
                for (size_t k = 0; k < d; ++k) {
                    float orig_diff = vi_orig[k] - vj_orig[k];
                    float rot_diff = vi_rot[k] - vj_rot[k];
                    orig_dist_sq += orig_diff * orig_diff;
                    rot_dist_sq += rot_diff * rot_diff;
                }

                float rel_error = std::abs(orig_dist_sq - rot_dist_sq) / orig_dist_sq;

                EXPECT_LT(rel_error, 1e-4f)
                    << "Distance not preserved for vectors " << i << " and " << j << " at d=" << d
                    << " (" << GetRotationMethod(d) << ")"
                    << ": original=" << std::sqrt(orig_dist_sq)
                    << ", rotated=" << std::sqrt(rot_dist_sq);
            }
        }
    }
}

/**
 * @brief Test rotation with a single vector
 */
TEST_F(RotationTest, SingleVector) {
    std::vector<size_t> dimensions = {64, 512, 1024};

    for (size_t d : dimensions) {
        auto original = skmeans::GenerateRandomVectors(1, d, -1.0f, 1.0f, 42);

        skmeans::ADSamplingPruner pruner(d, 2.1f);

        std::vector<float> rotated(d);
        pruner.Rotate(original.data(), rotated.data(), 1);

        std::vector<float> recovered(d);
        pruner.Unrotate(rotated.data(), recovered.data(), 1);

        float max_error = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            max_error = std::max(max_error, std::abs(original[i] - recovered[i]));
        }

        EXPECT_LT(max_error, 1e-4f) << "Single vector test failed for d=" << d;
    }
}

/**
 * @brief Test that different seeds produce different rotations
 */
TEST_F(RotationTest, DifferentSeedsProduceDifferentRotations) {
    const size_t d = 256;
    const size_t n = 10;

    auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

    skmeans::ADSamplingPruner pruner1(d, 2.1f, 42);
    skmeans::ADSamplingPruner pruner2(d, 2.1f, 123);

    std::vector<float> rotated1(n * d);
    std::vector<float> rotated2(n * d);

    pruner1.Rotate(original.data(), rotated1.data(), n);
    pruner2.Rotate(original.data(), rotated2.data(), n);

    // Rotated vectors should be different
    bool found_difference = false;
    for (size_t i = 0; i < n * d; ++i) {
        if (std::abs(rotated1[i] - rotated2[i]) > 1e-6f) {
            found_difference = true;
            break;
        }
    }

    EXPECT_TRUE(found_difference) << "Different seeds should produce different rotation matrices";
}

/**
 * @brief Test that same seed produces identical rotations
 */
TEST_F(RotationTest, SameSeedProducesIdenticalRotations) {
    const size_t d = 256;
    const size_t n = 10;

    auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

    skmeans::ADSamplingPruner pruner1(d, 2.1f, 42);
    skmeans::ADSamplingPruner pruner2(d, 2.1f, 42);

    std::vector<float> rotated1(n * d);
    std::vector<float> rotated2(n * d);

    pruner1.Rotate(original.data(), rotated1.data(), n);
    pruner2.Rotate(original.data(), rotated2.data(), n);

    // Rotated vectors should be identical
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_FLOAT_EQ(rotated1[i], rotated2[i])
            << "Same seed should produce identical rotations at index " << i;
    }
}

/**
 * @brief Test that SuperKMeans produces identical results with data_already_rotated flag
 *
 * This test verifies that:
 * 1. Running SuperKMeans with default config (data_already_rotated = false, unrotate_centroids =
 * false) → SuperKMeans rotates data internally, returns rotated centroids
 * 2. Manually rotating data with the same pruner config, then running SuperKMeans with
 *    (data_already_rotated = true, unrotate_centroids = true -> will be forced to false by
 * constructor) → SuperKMeans skips rotation, returns rotated centroids Both should produce
 * identical rotated centroids since everything is deterministic with seeds.
 */
TEST_F(RotationTest, SuperKMeansWithPreRotatedDataProducesIdenticalResults) {
    const size_t n = 10000;
    const size_t d = 256;
    const size_t k = 100;
    const uint32_t seed = 42;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, seed);

    // Test 1: Run SuperKMeans with default config (data will be rotated internally)
    skmeans::SuperKMeansConfig config1;
    config1.iters = 10;
    config1.seed = seed;
    config1.verbose = false;
    config1.data_already_rotated = false; // Default: data needs rotation
    config1.unrotate_centroids = false;
    config1.sampling_fraction = 1.0f;

    auto kmeans1 = skmeans::SuperKMeans(k, d, config1);
    auto centroids1 = kmeans1.Train(data.data(), n);

    skmeans::ADSamplingPruner pruner(d, 1.5f, seed);
    std::vector<float> rotated_data(n * d);
    pruner.Rotate(data.data(), rotated_data.data(), n);

    skmeans::SuperKMeansConfig config2;
    config2.iters = 10;
    config2.seed = seed;
    config2.verbose = false;
    config2.data_already_rotated = true; // Data is pre-rotated
    config2.unrotate_centroids = true;   // Try to set true - will be forced to false by constructor
    config2.sampling_fraction = 1.0f;

    auto kmeans2 = skmeans::SuperKMeans(k, d, config2);

    auto centroids2 = kmeans2.Train(rotated_data.data(), n);

    ASSERT_EQ(centroids1.size(), centroids2.size()) << "Centroid vectors should have the same size";
    ASSERT_EQ(centroids1.size(), k * d) << "Centroid size mismatch";

    size_t mismatches = 0;
    float max_abs_error = 0.0f;
    float sum_abs_error = 0.0f;
    for (size_t i = 0; i < k * d; ++i) {
        float abs_error = std::abs(centroids1[i] - centroids2[i]);
        max_abs_error = std::max(max_abs_error, abs_error);
        sum_abs_error += abs_error;
        if (abs_error > 1e-4f) {
            mismatches++;
        }
    }
    float avg_abs_error = sum_abs_error / (k * d);

    EXPECT_EQ(mismatches, 0) << "Centroids should match exactly (within numerical precision). "
                             << "Mismatches: " << mismatches << "/" << (k * d)
                             << ", Max error: " << max_abs_error
                             << ", Avg error: " << avg_abs_error;

    EXPECT_LT(max_abs_error, 1e-3f)
        << "Maximum absolute error between centroids should be very small. Got: " << max_abs_error;

    EXPECT_LT(avg_abs_error, 1e-5f)
        << "Average absolute error between centroids should be very small. Got: " << avg_abs_error;
}

// ── In-place rotation ──

void ExpectInPlaceRotationMatchesOutOfPlace(const size_t n, const size_t d) {
    auto original = skmeans::GenerateRandomVectors(n, d, -1.0f, 1.0f, 42);

    skmeans::ADSamplingPruner pruner(d, 1.5f, 42);

    std::vector<float> out_of_place(n * d);
    pruner.Rotate(original.data(), out_of_place.data(), n);

    std::vector<float> in_place(original);
    pruner.Rotate<true>(in_place.data(), in_place.data(), n);

    double max_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        max_error = std::max(max_error, std::abs(double(in_place[i]) - double(out_of_place[i])));
    }
    EXPECT_LT(max_error, 1e-4) << "d=" << d << " (" << GetRotationMethod(d)
                               << "), max error: " << max_error;
}

TEST_F(RotationTest, RotateInPlaceMatchesOutOfPlace_MatrixPath) {
    ExpectInPlaceRotationMatchesOutOfPlace(100, 128);
}

TEST_F(RotationTest, RotateInPlaceMatchesOutOfPlace_MatrixPathMultipleBlocks) {
    ExpectInPlaceRotationMatchesOutOfPlace(skmeans::ROTATION_BLOCK_SIZE * 2 + 137, 128);
}

TEST_F(RotationTest, RotateInPlaceMatchesOutOfPlace_DCTPath) {
    ExpectInPlaceRotationMatchesOutOfPlace(5000, 1024);
}

TEST_F(RotationTest, TrainInPlaceMatchesTrainOnPreRotatedData) {
    const size_t n = 10000;
    const size_t d = 256;
    const size_t k = 100;
    const uint32_t seed = 42;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, seed);

    skmeans::SuperKMeansConfig config;
    config.iters = 10;
    config.seed = seed;
    config.verbose = false;
    config.unrotate_centroids = false;
    config.sampling_fraction = 1.0f;

    skmeans::ADSamplingPruner pruner(d, 1.5f, seed);
    std::vector<float> pre_rotated(n * d);
    pruner.Rotate(data.data(), pre_rotated.data(), n);

    auto reference_config = config;
    reference_config.data_already_rotated = true;
    auto reference = skmeans::SuperKMeans(k, d, reference_config);
    auto reference_centroids = reference.Train(pre_rotated.data(), n);

    std::vector<float> in_place(data);
    auto kmeans = skmeans::SuperKMeans(k, d, config);
    auto centroids = kmeans.TrainInPlace(in_place.data(), n);

    ASSERT_EQ(centroids.size(), reference_centroids.size());
    auto reference_assignments =
        reference.Assign(pre_rotated.data(), reference_centroids.data(), n, k);
    auto assignments = kmeans.Assign(pre_rotated.data(), centroids.data(), n, k);
    const double reference_wcss = skmeans::SuperKMeans<>::ComputeWCSS(
        pre_rotated.data(), reference_centroids.data(), reference_assignments.data(), n, d
    );
    const double wcss = skmeans::SuperKMeans<>::ComputeWCSS(
        pre_rotated.data(), centroids.data(), assignments.data(), n, d
    );
    EXPECT_NEAR(wcss / reference_wcss, 1.0, 1e-4);

    for (size_t i = 0; i < n * d; ++i) {
        ASSERT_NEAR(in_place[i], pre_rotated[i], 1e-4) << "at " << i;
    }
}

TEST_F(RotationTest, StateRecordsHowTrainingWasCarriedOut) {
    const size_t n = 5000;
    const size_t d = 128;
    const size_t k = 50;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = 5;
    config.seed = 42;
    config.verbose = false;
    config.suppress_warnings = true;
    config.sampling_fraction = 1.0f;

    using SKM = skmeans::SuperKMeans<>;
    {
        auto kmeans = SKM(k, d, config);
        EXPECT_FALSE(kmeans.GetState().trained);
        kmeans.Train(data.data(), n);
        EXPECT_TRUE(kmeans.GetState().trained);
        EXPECT_FALSE(kmeans.GetState().trained_in_place);
        EXPECT_FALSE(kmeans.GetState().training_data_rotated);
    }
    {
        auto kmeans = SKM(k, d, config);
        std::vector<float> in_place(data);
        kmeans.TrainInPlace(in_place.data(), n);
        EXPECT_TRUE(kmeans.GetState().trained);
        EXPECT_TRUE(kmeans.GetState().trained_in_place);
        EXPECT_TRUE(kmeans.GetState().training_data_rotated);
    }
    {
        auto pre_rotated_config = config;
        pre_rotated_config.data_already_rotated = true;
        auto kmeans = SKM(k, d, pre_rotated_config);
        skmeans::ADSamplingPruner pruner(d, skmeans::PRUNER_INITIAL_THRESHOLD, config.seed);
        std::vector<float> pre_rotated(n * d);
        pruner.Rotate(data.data(), pre_rotated.data(), n);
        kmeans.Train(pre_rotated.data(), n);
        EXPECT_TRUE(kmeans.GetState().trained);
        EXPECT_FALSE(kmeans.GetState().trained_in_place);
        EXPECT_TRUE(kmeans.GetState().training_data_rotated);
    }
}

TEST_F(RotationTest, HierarchicalStateRecordsHowTrainingWasCarriedOut) {
    const size_t n = 8000;
    const size_t d = 128;
    const size_t k = 200;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, 42);

    skmeans::HierarchicalSuperKMeansConfig config;
    config.iters_mesoclustering = 2;
    config.iters_fineclustering = 2;
    config.seed = 42;
    config.verbose = false;
    config.suppress_warnings = true;

    using HSKM = skmeans::HierarchicalSuperKMeans<>;
    {
        auto kmeans = HSKM(k, d, config);
        kmeans.Train(data.data(), n);
        EXPECT_TRUE(kmeans.GetState().trained);
        EXPECT_FALSE(kmeans.GetState().trained_in_place);
        EXPECT_FALSE(kmeans.GetState().training_data_rotated);
    }
    {
        auto kmeans = HSKM(k, d, config);
        std::vector<float> in_place(data);
        kmeans.TrainInPlace(in_place.data(), n);
        EXPECT_TRUE(kmeans.GetState().trained);
        EXPECT_TRUE(kmeans.GetState().trained_in_place);
        EXPECT_TRUE(kmeans.GetState().training_data_rotated);
    }
}

void ExpectStateRotatorRecoversOriginal(const size_t n, const size_t d) {
    const size_t k = 50;
    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = 3;
    config.seed = 42;
    config.verbose = false;
    config.suppress_warnings = true;
    config.sampling_fraction = 1.0f;

    auto kmeans = skmeans::SuperKMeans(k, d, config);
    EXPECT_EQ(kmeans.GetState().rotator, nullptr);

    std::vector<float> rotated(data);
    kmeans.TrainInPlace(rotated.data(), n);

    const auto* rotator = kmeans.GetState().rotator;
    ASSERT_NE(rotator, nullptr);

    std::vector<float> recovered(n * d);
    rotator->Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        max_error = std::max(max_error, std::abs(double(recovered[i]) - double(data[i])));
    }
    EXPECT_LT(max_error, 1e-3) << "d=" << d << " (" << GetRotationMethod(d)
                               << "), max error: " << max_error;
}

TEST_F(RotationTest, StateRotatorRecoversOriginal_MatrixPath) {
    ExpectStateRotatorRecoversOriginal(2000, 128);
}

TEST_F(RotationTest, StateRotatorRecoversOriginal_DCTPath) {
    ExpectStateRotatorRecoversOriginal(2000, 1024);
}

TEST_F(RotationTest, StateRotatorRotatesNewVectorsIntoTheTrainedDomain) {
    const size_t n = 2000;
    const size_t d = 1024;
    const size_t k = 50;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = 3;
    config.seed = 42;
    config.verbose = false;
    config.suppress_warnings = true;
    config.sampling_fraction = 1.0f;

    auto kmeans = skmeans::SuperKMeans(k, d, config);
    std::vector<float> rotated(data);
    kmeans.TrainInPlace(rotated.data(), n);

    // Rotating the original vectors through the exposed rotator must reproduce what TrainInPlace
    // left in the buffer
    std::vector<float> rotated_again(n * d);
    kmeans.GetState().rotator->Rotate(data.data(), rotated_again.data(), n);

    double max_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        max_error = std::max(max_error, std::abs(double(rotated_again[i]) - double(rotated[i])));
    }
    EXPECT_LT(max_error, 1e-4) << "max error: " << max_error;
}

TEST_F(RotationTest, TrainInPlaceOverridesSamplingFraction) {
    const size_t n = 5000;
    const size_t d = 128;
    const size_t k = 50;

    auto data = skmeans::MakeBlobs(n, d, k, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = 5;
    config.seed = 42;
    config.verbose = false;
    config.suppress_warnings = true;
    config.sampling_fraction = 0.3f;

    auto kmeans = skmeans::SuperKMeans(k, d, config);
    std::vector<float> in_place(data);
    auto centroids = kmeans.TrainInPlace(in_place.data(), n);

    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), k * d);

    auto assignments = kmeans.AssignTrainingPoints(in_place.data(), centroids.data(), n, k);
    EXPECT_EQ(assignments.size(), n);
}

} // anonymous namespace
