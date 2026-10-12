#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <omp.h>

#include "superkmeans/executor.h"

namespace skmeans {

/**
 * @brief ParallelExecutor on OpenMP: one parallel region per ParallelFor.
 *
 * A worker's exception is rethrown in the caller (one escaping the region would terminate).
 */
class OpenMPExecutor final : public ParallelExecutor {
  public:
    explicit OpenMPExecutor(size_t n_workers) : n_workers(std::max<size_t>(n_workers, 1)) {}

    size_t NumWorkers() const override { return n_workers; }

    void ParallelFor(size_t n, const std::function<void(size_t, size_t, size_t)>& fn) override {
        if (n == 0) {
            return;
        }
        const size_t n_tasks = std::min(n_workers, n);
        if (n_tasks == 1) {
            fn(0, n, 0);
            return;
        }
        std::atomic<bool> failed{false};
        std::exception_ptr error;
#pragma omp parallel num_threads(static_cast<int>(n_tasks))
        {
            // A short team still runs every range.
            for (size_t w = static_cast<size_t>(omp_get_thread_num()); w < n_tasks;
                 w += static_cast<size_t>(omp_get_num_threads())) {
                try {
                    fn(n * w / n_tasks, n * (w + 1) / n_tasks, w);
                } catch (...) {
                    if (!failed.exchange(true)) {
                        error = std::current_exception();
                    }
                }
            }
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

  private:
    size_t n_workers;
};

} // namespace skmeans
