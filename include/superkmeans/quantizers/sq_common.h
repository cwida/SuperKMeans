#pragma once

#include "superkmeans/common.h"
#include "superkmeans/executor.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace skmeans {

struct ScalarQuantizationParams {
    float quantization_base;
    float quantization_scale;
    float inv_quantization_scale;
};

inline ScalarQuantizationParams ComputeScalarQuantizationParams(
    ParallelExecutor& executor,
    const float* embeddings,
    size_t total_elements,
    float max_value
) {
    // Per-worker minima and maxima, combined after the loop (exact: order does not matter)
    std::vector<float> worker_min(executor.NumWorkers(), std::numeric_limits<float>::max());
    std::vector<float> worker_max(executor.NumWorkers(), std::numeric_limits<float>::lowest());
    executor.ParallelFor(total_elements, [&](size_t begin, size_t end, size_t worker) {
        float local_min = std::numeric_limits<float>::max();
        float local_max = std::numeric_limits<float>::lowest();
        for (size_t i = begin; i < end; ++i) {
            local_min = std::min(local_min, embeddings[i]);
            local_max = std::max(local_max, embeddings[i]);
        }
        worker_min[worker] = local_min;
        worker_max[worker] = local_max;
    });
    const float global_min = *std::min_element(worker_min.begin(), worker_min.end());
    const float global_max = *std::max_element(worker_max.begin(), worker_max.end());

    const float range = global_max - global_min;
    const float scale = (range > 0) ? max_value / range : 1.0f;
    return {global_min, scale, 1.0f / scale};
}

} // namespace skmeans
