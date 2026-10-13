#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "superkmeans/allocator.h"
#include "superkmeans/executor.h"
#include "superkmeans/two_means.h"

namespace {

constexpr size_t N_VECTORS = 1000;
constexpr size_t DIMENSIONS = 32;
constexpr uint32_t ITERS = 25;
constexpr float TOL = 1e-6f;
constexpr uint32_t SEED = 42;

// Two Gaussian blobs around orthogonal centers: vector i belongs to blob i % 2
std::vector<float> MakeTwoBlobs() {
    std::mt19937 rng(SEED);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::vector<float> data(N_VECTORS * DIMENSIONS);
    for (size_t i = 0; i < N_VECTORS; ++i) {
        for (size_t j = 0; j < DIMENSIONS; ++j) {
            data[i * DIMENSIONS + j] = noise(rng);
        }
        data[i * DIMENSIONS + i % 2] += 5.0f;
    }
    return data;
}

// Each blob becomes one cluster, whose centroid is the mean (angular: the unit mean) of its
// vectors, on any executor.
TEST(TwoMeansTest, SeparatesTwoBlobs) {
    const std::vector<float> data = MakeTwoBlobs();
    skmeans::VanillaAllocator allocator;
    std::vector<std::unique_ptr<skmeans::ParallelExecutor>> executors;
    executors.push_back(std::make_unique<skmeans::SerialExecutor>());
    executors.push_back(skmeans::MakeDefaultExecutor(4));
    for (auto& executor : executors) {
        for (const bool angular : {false, true}) {
            SCOPED_TRACE(
                "workers=" + std::to_string(executor->NumWorkers()) +
                " angular=" + std::to_string(angular)
            );
            std::vector<float> centroids(2 * DIMENSIONS);
            std::vector<uint32_t> assignments(N_VECTORS);
            skmeans::TwoMeans(
                data.data(),
                N_VECTORS,
                DIMENSIONS,
                ITERS,
                TOL,
                angular,
                SEED,
                centroids.data(),
                assignments.data(),
                *executor,
                allocator
            );
            ASSERT_NE(assignments[0], assignments[1]);
            for (size_t i = 0; i < N_VECTORS; ++i) {
                ASSERT_EQ(assignments[i], assignments[i % 2]) << "vector " << i;
            }
            for (uint32_t ci = 0; ci < 2; ++ci) {
                std::vector<double> mean(DIMENSIONS, 0.0);
                for (size_t i = 0; i < N_VECTORS; ++i) {
                    if (assignments[i] != ci) {
                        continue;
                    }
                    for (size_t j = 0; j < DIMENSIONS; ++j) {
                        mean[j] += data[i * DIMENSIONS + j];
                    }
                }
                double norm = 0.0;
                for (size_t j = 0; j < DIMENSIONS; ++j) {
                    norm += mean[j] * mean[j];
                }
                const double mult = angular ? 1.0 / std::sqrt(norm) : 2.0 / N_VECTORS;
                for (size_t j = 0; j < DIMENSIONS; ++j) {
                    EXPECT_NEAR(centroids[ci * DIMENSIONS + j], mean[j] * mult, 1e-4)
                        << "centroid " << ci << " dimension " << j;
                }
            }
        }
    }
}

TEST(TwoMeansTest, FewerThanTwoVectorsThrows) {
    const std::vector<float> data = MakeTwoBlobs();
    skmeans::VanillaAllocator allocator;
    skmeans::SerialExecutor executor;
    std::vector<float> centroids(2 * DIMENSIONS);
    EXPECT_THROW(
        skmeans::TwoMeans(
            data.data(),
            1,
            DIMENSIONS,
            ITERS,
            TOL,
            false,
            SEED,
            centroids.data(),
            nullptr,
            executor,
            allocator
        ),
        std::invalid_argument
    );
}

} // namespace
