#include <cmath>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/pdx/utils.h"

namespace {

class DistanceComputerTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

/**
 * @brief Test that SIMD L2 distance computation matches scalar reference
 *
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_L2) {
    std::vector<size_t> dimensions = {1,   3,   7,   8,   15,  16,  31,  32,   63,   64,   100,
                                      127, 128, 255, 256, 384, 512, 768, 1000, 1024, 1536, 2048};
    const size_t n_pairs = 100;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        auto vectors1 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 42);
        auto vectors2 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 123);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* v1 = vectors1.data() + i * d;
            const float* v2 = vectors2.data() + i * d;

            float scalar_dist =
                skmeans::ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
                    Horizontal(v1, v2, d);

            float simd_dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::f32>::Horizontal(v1, v2, d);

            float abs_error = std::abs(scalar_dist - simd_dist);
            float rel_error = abs_error / std::max(scalar_dist, 1e-6f);

            EXPECT_LT(rel_error, 1e-5f)
                << "SIMD/Scalar mismatch at d=" << d << ", pair " << i << ": scalar=" << scalar_dist
                << ", simd=" << simd_dist << ", rel_error=" << rel_error;
        }
    }
}

/**
 * @brief Test that distance is zero for identical vectors
 */
TEST_F(DistanceComputerTest, SIMD_ZeroDistanceForIdenticalVectors) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};

    for (size_t d : dimensions) {
        auto vectors = skmeans::GenerateRandomVectors(10, d, -10.0f, 10.0f, 42);

        for (size_t i = 0; i < 10; ++i) {
            const float* v = vectors.data() + i * d;

            float dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::f32>::Horizontal(v, v, d);

            EXPECT_NEAR(dist, 0.0f, 1e-10f) << "Distance to self should be zero at d=" << d;
        }
    }
}

/**
 * @brief Test that BatchComputer::FindNearestNeighbor finds the true nearest neighbor
 *
 * Compares BLAS-based batch nearest neighbor search against brute force reference.
 */
TEST_F(DistanceComputerTest, BatchComputer_FindNearestNeighbor_Correctness) {
    struct TestCase {
        size_t n_x;
        size_t n_y;
        size_t d;
    };
    std::vector<TestCase> test_cases = {
        {100, 50, 64},
        {500, 100, 128},
        {1000, 200, 256},
        {2000, 500, 512},
        {5000, 1000, 128},
    };
    for (const auto& tc : test_cases) {
        SCOPED_TRACE(
            "Testing n_x=" + std::to_string(tc.n_x) + ", n_y=" + std::to_string(tc.n_y) +
            ", d=" + std::to_string(tc.d)
        );

        auto x = skmeans::GenerateRandomVectors(tc.n_x, tc.d, -10.0f, 10.0f, 42);
        auto y = skmeans::GenerateRandomVectors(tc.n_y, tc.d, -10.0f, 10.0f, 123);
        auto norms_x = skmeans::ComputeNorms(x.data(), tc.n_x, tc.d);
        auto norms_y = skmeans::ComputeNorms(y.data(), tc.n_y, tc.d);

        std::vector<uint32_t> bf_knn(tc.n_x);
        std::vector<float> bf_distances(tc.n_x);
        skmeans::FindNearestNeighborBruteForce(
            x.data(), y.data(), tc.n_x, tc.n_y, tc.d, bf_knn.data(), bf_distances.data()
        );

        std::vector<uint32_t> batch_knn(tc.n_x);
        std::vector<float> batch_distances(tc.n_x);
        std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);
        skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
            FindNearestNeighbor(
                x.data(),
                y.data(),
                tc.n_x,
                tc.n_y,
                tc.d,
                norms_x.data(),
                norms_y.data(),
                batch_knn.data(),
                batch_distances.data(),
                tmp_buf.data()
            );

        size_t mismatches = 0;
        for (size_t i = 0; i < tc.n_x; ++i) {
            if (batch_knn[i] != bf_knn[i]) {
                // Check if distances are the same (tie-breaking)
                float rel_diff = std::abs(batch_distances[i] - bf_distances[i]) /
                                 std::max(bf_distances[i], 1e-6f);
                if (rel_diff > 1e-4f) {
                    ++mismatches;
                }
            }
        }
        EXPECT_EQ(mismatches, 0) << "Found " << mismatches
                                 << " mismatches in nearest neighbor results";

        for (size_t i = 0; i < tc.n_x; ++i) {
            float rel_error =
                std::abs(batch_distances[i] - bf_distances[i]) / std::max(bf_distances[i], 1e-6f);
            EXPECT_LT(rel_error, 1e-4f)
                << "Distance mismatch at index " << i << ": batch=" << batch_distances[i]
                << ", bf=" << bf_distances[i];
        }
    }
}

/**
 * @brief Test BatchComputer with edge case: single query
 */
TEST_F(DistanceComputerTest, BatchComputer_SingleQuery) {
    const size_t n_y = 100;
    const size_t d = 128;

    auto x = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 42);
    auto y = skmeans::GenerateRandomVectors(n_y, d, -10.0f, 10.0f, 123);

    auto norms_x = skmeans::ComputeNorms(x.data(), 1, d);
    auto norms_y = skmeans::ComputeNorms(y.data(), n_y, d);

    std::vector<uint32_t> bf_knn(1);
    std::vector<float> bf_distances(1);
    skmeans::FindNearestNeighborBruteForce(
        x.data(), y.data(), 1, n_y, d, bf_knn.data(), bf_distances.data()
    );

    std::vector<uint32_t> batch_knn(1);
    std::vector<float> batch_distances(1);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);
    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindNearestNeighbor(
            x.data(),
            y.data(),
            1,
            n_y,
            d,
            norms_x.data(),
            norms_y.data(),
            batch_knn.data(),
            batch_distances.data(),
            tmp_buf.data()
        );

    EXPECT_EQ(batch_knn[0], bf_knn[0]);
    float rel_error =
        std::abs(batch_distances[0] - bf_distances[0]) / std::max(bf_distances[0], 1e-6f);
    EXPECT_LT(rel_error, 1e-4f);
}

/**
 * @brief Test that BatchComputer::FindKNearestNeighbors finds the true k nearest neighbors
 *
 * Compares BLAS-based batch k-NN search against brute force reference with k=10.
 */
TEST_F(DistanceComputerTest, BatchComputer_FindKNearestNeighbors_Correctness) {
    const size_t n_x = 1000;
    const size_t n_y = 500;
    const size_t d = 128;
    const size_t k = 10;

    auto x = skmeans::GenerateRandomVectors(n_x, d, -10.0f, 10.0f, 42);
    auto y = skmeans::GenerateRandomVectors(n_y, d, -10.0f, 10.0f, 123);
    auto norms_x = skmeans::ComputeNorms(x.data(), n_x, d);
    auto norms_y = skmeans::ComputeNorms(y.data(), n_y, d);

    std::vector<uint32_t> bf_knn(n_x * k);
    std::vector<float> bf_distances(n_x * k);
    skmeans::FindKNearestNeighborsBruteForce(
        x.data(), y.data(), n_x, n_y, d, k, bf_knn.data(), bf_distances.data()
    );

    std::vector<uint32_t> batch_knn(n_x * k);
    std::vector<float> batch_distances(n_x * k);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);
    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindKNearestNeighbors(
            x.data(),
            y.data(),
            n_x,
            n_y,
            d,
            norms_x.data(),
            norms_y.data(),
            k,
            batch_knn.data(),
            batch_distances.data(),
            tmp_buf.data()
        );

    size_t index_mismatches = 0;
    size_t distance_mismatches = 0;
    for (size_t i = 0; i < n_x; ++i) {
        for (size_t ki = 0; ki < k; ++ki) {
            size_t idx = i * k + ki;
            // Check if indices match
            if (batch_knn[idx] != bf_knn[idx]) {
                // Allow mismatch if distances are equal (tie-breaking)
                float rel_diff = std::abs(batch_distances[idx] - bf_distances[idx]) /
                                 std::max(bf_distances[idx], 1e-6f);
                if (rel_diff > 1e-4f) {
                    ++index_mismatches;
                }
            }
            // Check distance accuracy
            float rel_error = std::abs(batch_distances[idx] - bf_distances[idx]) /
                              std::max(bf_distances[idx], 1e-6f);
            if (rel_error > 1e-4f) {
                ++distance_mismatches;
            }
        }
    }

    EXPECT_EQ(index_mismatches, 0)
        << "Found " << index_mismatches << " index mismatches in k-NN results";
    EXPECT_EQ(distance_mismatches, 0)
        << "Found " << distance_mismatches << " distance mismatches in k-NN results";

    for (size_t i = 0; i < n_x; ++i) {
        for (size_t ki = 1; ki < k; ++ki) {
            EXPECT_LE(batch_distances[i * k + ki - 1], batch_distances[i * k + ki])
                << "Results not sorted for query " << i << " at position " << ki;
        }
    }
}

/**
 * @brief Test that SIMD FlipSign matches scalar reference
 *
 */
TEST_F(DistanceComputerTest, FlipSign_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 512, 1024, 2048};

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        std::vector<float> data(d);
        std::vector<uint32_t> masks(d);
        skmeans::GenerateRandomDataWithMasks(data.data(), masks.data(), d, 0.5f, 42);

        std::vector<float> scalar_output(d);
        std::vector<float> simd_output(d);

        skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::FlipSign(
            data.data(), scalar_output.data(), masks.data(), d
        );
        skmeans::UtilsComputer<skmeans::Quantization::f32>::FlipSign(
            data.data(), simd_output.data(), masks.data(), d
        );

        for (size_t i = 0; i < d; ++i) {
            uint32_t scalar_bits = *reinterpret_cast<const uint32_t*>(&scalar_output[i]);
            uint32_t simd_bits = *reinterpret_cast<const uint32_t*>(&simd_output[i]);
            EXPECT_EQ(scalar_bits, simd_bits)
                << "FlipSign mismatch at index " << i << " for d=" << d
                << ": scalar=" << scalar_output[i] << " (0x" << std::hex << scalar_bits << ")"
                << ", simd=" << simd_output[i] << " (0x" << simd_bits << ")" << std::dec;
        }
    }
}

/**
 * @brief Test that SIMD InitPositionsArray matches scalar reference
 *
 * Verifies that the SIMD implementation produces the same count and positions
 * as the scalar reference implementation.
 */
TEST_F(DistanceComputerTest, InitPositionsArray_SIMD_MatchesScalar) {
    std::vector<size_t> vector_counts = {32, 64, 128, 256, 512, 1024, 2048};
    std::vector<float> selectivities = {0.01f, 0.03f, 0.05f, 0.10f, 0.25f, 0.50f};
    const float threshold = 100.0f;

    for (size_t n : vector_counts) {
        for (float selectivity : selectivities) {
            SCOPED_TRACE(
                "Testing n=" + std::to_string(n) + ", selectivity=" + std::to_string(selectivity)
            );

            std::vector<float> pruning_distances(n);
            skmeans::GenerateRandomDistances(
                pruning_distances.data(), n, threshold, selectivity, 42
            );

            std::vector<uint32_t> scalar_positions(n);
            std::vector<uint32_t> simd_positions(n);
            size_t scalar_count = 0;
            size_t simd_count = 0;

            skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
                n, scalar_count, scalar_positions.data(), threshold, pruning_distances.data()
            );
            skmeans::UtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
                n, simd_count, simd_positions.data(), threshold, pruning_distances.data()
            );

            EXPECT_EQ(scalar_count, simd_count)
                << "Count mismatch for n=" << n << ", selectivity=" << selectivity;

            for (size_t i = 0; i < scalar_count; ++i) {
                EXPECT_EQ(scalar_positions[i], simd_positions[i])
                    << "Position mismatch at index " << i << " for n=" << n
                    << ", selectivity=" << selectivity;
            }
        }
    }
}

/**
 * @brief Test that SIMD u8 L2 distance computation matches scalar reference
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_L2_U8) {
    std::vector<size_t> dimensions = {1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 512, 1024};
    const size_t n_pairs = 100;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        std::vector<uint8_t> vectors1(n_pairs * d), vectors2(n_pairs * d);
        for (auto& v : vectors1) v = static_cast<uint8_t>(dist(rng));
        for (auto& v : vectors2) v = static_cast<uint8_t>(dist(rng));

        for (size_t i = 0; i < n_pairs; ++i) {
            const uint8_t* v1 = vectors1.data() + i * d;
            const uint8_t* v2 = vectors2.data() + i * d;

            uint32_t scalar_dist =
                skmeans::ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::u8>::
                    Horizontal(v1, v2, d);

            uint32_t simd_dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::u8>::Horizontal(v1, v2, d);

            EXPECT_EQ(scalar_dist, simd_dist)
                << "SIMD/Scalar mismatch at d=" << d << ", pair " << i;
        }
    }
}

/**
 * @brief Test that SIMD u4 L2 distance computation matches scalar reference
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_L2_U4) {
    std::vector<size_t> packed_byte_counts = {1, 2, 8, 15, 16, 31, 32, 63, 64, 128, 256, 512};
    const size_t n_pairs = 100;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t num_packed_bytes : packed_byte_counts) {
        SCOPED_TRACE("Testing num_packed_bytes=" + std::to_string(num_packed_bytes));

        std::vector<uint8_t> vectors1(n_pairs * num_packed_bytes);
        std::vector<uint8_t> vectors2(n_pairs * num_packed_bytes);
        for (auto& v : vectors1) v = static_cast<uint8_t>(dist(rng));
        for (auto& v : vectors2) v = static_cast<uint8_t>(dist(rng));

        for (size_t i = 0; i < n_pairs; ++i) {
            const uint8_t* v1 = vectors1.data() + i * num_packed_bytes;
            const uint8_t* v2 = vectors2.data() + i * num_packed_bytes;

            uint32_t scalar_dist =
                skmeans::ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::u4>::
                    Horizontal(v1, v2, num_packed_bytes);

            uint32_t simd_dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::u4>::Horizontal(v1, v2, num_packed_bytes);

            EXPECT_EQ(scalar_dist, simd_dist)
                << "SIMD/Scalar mismatch at num_packed_bytes=" << num_packed_bytes << ", pair " << i;
        }
    }
}

/**
 * @brief Test that SIMD u4 InitPositionsArray matches scalar reference
 */
TEST_F(DistanceComputerTest, InitPositionsArray_SIMD_MatchesScalar_U4) {
    std::vector<size_t> vector_counts = {32, 64, 128, 256, 512, 1024};
    std::vector<float> selectivities = {0.01f, 0.05f, 0.10f, 0.25f, 0.50f};
    const uint32_t threshold = 100000;

    std::mt19937 rng(42);

    for (size_t n : vector_counts) {
        for (float selectivity : selectivities) {
            SCOPED_TRACE(
                "Testing n=" + std::to_string(n) + ", selectivity=" + std::to_string(selectivity)
            );

            // Generate uint32_t distances with approximately `selectivity` fraction below threshold
            std::vector<uint32_t> pruning_distances(n);
            std::uniform_int_distribution<uint32_t> dist_above(threshold, threshold * 10);
            std::uniform_int_distribution<uint32_t> dist_below(0, threshold - 1);
            std::bernoulli_distribution below_thresh(selectivity);
            for (size_t i = 0; i < n; ++i) {
                pruning_distances[i] = below_thresh(rng) ? dist_below(rng) : dist_above(rng);
            }

            std::vector<uint32_t> scalar_positions(n), simd_positions(n);
            size_t scalar_count = 0, simd_count = 0;

            skmeans::ScalarUtilsComputer<skmeans::Quantization::u4>::InitPositionsArray(
                n, scalar_count, scalar_positions.data(), threshold, pruning_distances.data()
            );
            skmeans::UtilsComputer<skmeans::Quantization::u4>::InitPositionsArray(
                n, simd_count, simd_positions.data(), threshold, pruning_distances.data()
            );

            EXPECT_EQ(scalar_count, simd_count)
                << "Count mismatch for n=" << n << ", selectivity=" << selectivity;

            for (size_t i = 0; i < scalar_count; ++i) {
                EXPECT_EQ(scalar_positions[i], simd_positions[i])
                    << "Position mismatch at index " << i;
            }
        }
    }
}

/**
 * @brief Test that SIMD PackU8ToU4x2 matches scalar reference
 */
TEST_F(DistanceComputerTest, PackU8ToU4x2_SIMD_MatchesScalar) {
    const std::vector<size_t> sizes = {2, 4, 14, 16, 30, 32, 62, 64, 100, 128, 256, 1024};

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 15);

    for (size_t count : sizes) {
        SCOPED_TRACE("Testing count=" + std::to_string(count));

        std::vector<uint8_t> src(count);
        for (auto& v : src) v = static_cast<uint8_t>(dist(rng));

        // Scalar reference
        std::vector<uint8_t> expected(count / 2);
        skmeans::ScalarUtilsComputer<skmeans::Quantization::u4>::PackU8ToU4x2(
            src.data(), expected.data(), count
        );

        // SIMD kernel
        std::vector<uint8_t> actual(count / 2, 0xFF);
        skmeans::UtilsComputer<skmeans::Quantization::u4>::PackU8ToU4x2(
            src.data(), actual.data(), count
        );

        for (size_t i = 0; i < count / 2; ++i) {
            EXPECT_EQ(actual[i], expected[i])
                << "mismatch at byte " << i << " for count=" << count;
        }
    }
}

} // anonymous namespace
