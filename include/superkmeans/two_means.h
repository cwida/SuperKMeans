#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "superkmeans/allocator.h"
#include "superkmeans/common.h"
#include "superkmeans/executor.h"
#include "superkmeans/profiler.h"

namespace skmeans {

/**
 * @brief The closer of two centroids to a vector. |vec|^2 is common to both distances, so the
 * centroids compare |c|^2 - 2 vec.c, both dot products in one pass over the vector.
 */
inline uint32_t ClosestOfTwoCentroids(
    const float* SKM_RESTRICT vec,
    const float* SKM_RESTRICT centroids,
    const float* centroid_norms,
    const size_t d
) {
    const float* centroid_0 = centroids;
    const float* centroid_1 = centroids + d;
    float dot_0 = 0.0f;
    float dot_1 = 0.0f;
    SKM_VECTORIZE_LOOP
    for (size_t j = 0; j < d; ++j) {
        dot_0 += vec[j] * centroid_0[j];
        dot_1 += vec[j] * centroid_1[j];
    }
    return centroid_norms[0] - 2.0f * dot_0 <= centroid_norms[1] - 2.0f * dot_1 ? 0 : 1;
}

/**
 * @brief k-means with 2 clusters, e.g. to split one cluster. Each iteration is one pass over the
 * vectors: a vector picks the closer centroid and is added to its accumulator while still in
 * cache, without the GEMM tiles and PDX layouts that make SuperKMeans with 2 clusters slower.
 *
 * Picks the initial centroids like SuperKMeans (the first two vectors of a seeded shuffle). An
 * empty cluster keeps its previous centroid.
 *
 * @param data Vectors to cluster (n × d, row-major)
 * @param n Number of vectors (at least 2)
 * @param d Dimensionality
 * @param iters Maximum number of iterations
 * @param tol Stops once the total squared centroid shift of an iteration falls below it
 * @param angular Normalizes the centroids after every update (spherical k-means)
 * @param seed Seed of the shuffle that picks the initial centroids
 * @param out_centroids Output centroids (2 × d, row-major)
 * @param out_assignments Each vector's cluster (0 or 1) for the final centroids, or nullptr
 * @param executor Runs each pass over the vectors, one range of vectors per worker
 * @param allocator Allocates the buffers sized by n and by the number of workers
 */
inline void TwoMeans(
    const float* SKM_RESTRICT data,
    const size_t n,
    const size_t d,
    const uint32_t iters,
    const float tol,
    const bool angular,
    const uint32_t seed,
    float* SKM_RESTRICT out_centroids,
    uint32_t* SKM_RESTRICT out_assignments,
    ParallelExecutor& executor,
    MemoryAllocator& allocator
) {
    SKM_PROFILE_SCOPE("two_means");
    if (n < 2) {
        throw std::invalid_argument("TwoMeans needs at least two vectors");
    }
    {
        std::mt19937 rng(seed);
        auto indices = AllocateBuffer<size_t>(allocator, n);
        for (size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.get(), indices.get() + n, rng);
        std::memcpy(out_centroids, data + indices[0] * d, d * sizeof(float));
        std::memcpy(out_centroids + d, data + indices[1] * d, d * sizeof(float));
    }

    const size_t n_workers = executor.NumWorkers();
    auto worker_centroid_accumulators = AllocateBuffer<float>(allocator, n_workers * 2 * d);
    std::vector<uint32_t> worker_cluster_sizes(n_workers * 2);
    float centroid_norms[2];
    const auto compute_centroid_norms = [&]() {
        for (size_t ci = 0; ci < 2; ++ci) {
            const float* centroid = out_centroids + ci * d;
            float sum = 0.0f;
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                sum += centroid[j] * centroid[j];
            }
            centroid_norms[ci] = sum;
        }
    };

    for (uint32_t iter = 0; iter < iters; ++iter) {
        compute_centroid_norms();
        std::memset(worker_centroid_accumulators.get(), 0, n_workers * 2 * d * sizeof(float));
        std::fill(worker_cluster_sizes.begin(), worker_cluster_sizes.end(), 0);
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t worker) {
            float* centroid_accumulators = worker_centroid_accumulators.get() + worker * 2 * d;
            uint32_t* cluster_sizes = worker_cluster_sizes.data() + worker * 2;
            for (size_t i = begin; i < end; ++i) {
                const float* vec = data + i * d;
                const uint32_t ci = ClosestOfTwoCentroids(vec, out_centroids, centroid_norms, d);
                float* acc = centroid_accumulators + ci * d;
                cluster_sizes[ci] += 1;
                SKM_VECTORIZE_LOOP
                for (size_t j = 0; j < d; ++j) {
                    acc[j] += vec[j];
                }
            }
        });

        float shift = 0.0f;
        for (size_t ci = 0; ci < 2; ++ci) {
            // Worker 0's accumulator gathers the others' and becomes the new centroid
            float* new_centroid = worker_centroid_accumulators.get() + ci * d;
            uint32_t cluster_size = worker_cluster_sizes[ci];
            for (size_t worker = 1; worker < n_workers; ++worker) {
                const float* acc = worker_centroid_accumulators.get() + (worker * 2 + ci) * d;
                SKM_VECTORIZE_LOOP
                for (size_t j = 0; j < d; ++j) {
                    new_centroid[j] += acc[j];
                }
                cluster_size += worker_cluster_sizes[worker * 2 + ci];
            }
            if (cluster_size == 0) {
                continue;
            }
            float mult = 1.0f / static_cast<float>(cluster_size);
            if (angular) {
                float sum = 0.0f;
                SKM_VECTORIZE_LOOP
                for (size_t j = 0; j < d; ++j) {
                    sum += new_centroid[j] * new_centroid[j];
                }
                mult = 1.0f / std::sqrt(sum);
            }
            float* centroid = out_centroids + ci * d;
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                const float delta = new_centroid[j] * mult - centroid[j];
                shift += delta * delta;
                centroid[j] = new_centroid[j] * mult;
            }
        }
        if (shift < tol) {
            break;
        }
    }

    if (out_assignments != nullptr) {
        compute_centroid_norms();
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                out_assignments[i] =
                    ClosestOfTwoCentroids(data + i * d, out_centroids, centroid_norms, d);
            }
        });
    }
}

} // namespace skmeans
