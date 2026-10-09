#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>

#include <forkunion/flat.hpp>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "superkmeans/executor.h"

namespace skmeans {

/**
 * @brief ParallelExecutor on a ForkUnion flat pool; the calling thread is worker 0.
 *
 * Adds what ForkUnion leaves to its user: a worker's exception is rethrown in the caller, and a
 * forked child spawns a new pool (the inherited one has no threads). Destroying it stops the pool.
 */
class ForkUnionExecutor final : public ParallelExecutor {
  public:
    explicit ForkUnionExecutor(size_t n_workers) : n_workers(std::max<size_t>(n_workers, 1)) {}

    ~ForkUnionExecutor() override { ReleaseThreads(); }

    void ReleaseThreads() override {
        if (pool != nullptr && pid != CurrentPid()) {
            // Inherited through fork(): its threads do not exist here, so it must never be joined.
            [[maybe_unused]] auto* leaked_pool = pool.release();
        }
        pool.reset();
    }

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
        EnsurePool();
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        pool->for_threads([&](auto thread) noexcept {
            for (size_t w = static_cast<size_t>(thread); w < n_tasks; w += n_workers) {
                try {
                    fn(n * w / n_tasks, n * (w + 1) / n_tasks, w);
                } catch (...) {
                    if (!failed.exchange(true)) {
                        error = std::current_exception();
                    }
                }
            }
        });
        if (error) {
            std::rethrow_exception(error);
        }
    }

  private:
    static long CurrentPid() {
#if defined(_WIN32)
        return static_cast<long>(_getpid());
#else
        return static_cast<long>(getpid());
#endif
    }

    void EnsurePool() {
        if (pool != nullptr && pid == CurrentPid()) {
            return;
        }
        ReleaseThreads();
        pool = std::make_unique<ashvardanian::forkunion::flat_pool_t>();
        if (!pool->try_spawn(n_workers)) {
            pool.reset();
            throw std::runtime_error("ForkUnion could not spawn its threads");
        }
        pid = CurrentPid();
    }

    size_t n_workers;
    long pid = 0;
    std::unique_ptr<ashvardanian::forkunion::flat_pool_t> pool;
};

} // namespace skmeans
