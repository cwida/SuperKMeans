#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

namespace skmeans {

/**
 * @brief Runs SuperKMeans' parallel loops; the caller may inject its own (e.g. a DuckDB scheduler).
 *
 * ParallelFor(n, fn) splits [0, n) into at most NumWorkers() contiguous ranges
 * [n * w / T, n * (w + 1) / T), calls fn(begin, end, worker) once per range with worker in
 * [0, T), and returns when all ranges are done. A body must not call ParallelFor again.
 */
class ParallelExecutor {
  public:
    virtual ~ParallelExecutor() = default;
    virtual size_t NumWorkers() const = 0;
    virtual void ParallelFor(size_t n, const std::function<void(size_t, size_t, size_t)>& fn) = 0;
    /// No parallel work is expected soon: a pool may stop its threads (the next ParallelFor
    /// restarts them). Does nothing by default.
    virtual void ReleaseThreads() {}
};

class SerialExecutor final : public ParallelExecutor {
  public:
    size_t NumWorkers() const override { return 1; }
    void ParallelFor(size_t n, const std::function<void(size_t, size_t, size_t)>& fn) override {
        if (n > 0) {
            fn(0, n, 0);
        }
    }
};

/// n_threads == 0 means one thread per hardware thread.
inline size_t ResolveNumThreads(size_t n_threads) {
    if (n_threads > 0) {
        return n_threads;
    }
    return std::max<size_t>(std::thread::hardware_concurrency(), 1);
}

inline std::unique_ptr<ParallelExecutor> MakeDefaultExecutor(size_t n_threads);

/**
 * @brief The executor of one top-level call: the injected one, or a default it owns.
 *
 * An owned default lives until the scope ends, which also stops its threads.
 */
class ExecutorScope {
  public:
    ExecutorScope(ParallelExecutor* executor, size_t n_threads) : borrowed(executor) {
        if (borrowed == nullptr) {
            owned = MakeDefaultExecutor(n_threads);
        }
    }
    ParallelExecutor& Get() const { return borrowed != nullptr ? *borrowed : *owned; }

  private:
    ParallelExecutor* borrowed;
    std::unique_ptr<ParallelExecutor> owned;
};

/**
 * @brief Base of the classes whose methods run on a bound executor (borrowed, not owned).
 * Unbound, they run serially.
 */
class ExecutorHolder {
  public:
    void SetExecutor(ParallelExecutor* executor) { this->executor = executor; }

  protected:
    ParallelExecutor& GetExecutor() const {
        static SerialExecutor serial;
        return executor != nullptr ? *executor : serial;
    }

  private:
    ParallelExecutor* executor = nullptr;
};

/**
 * @brief One public call: releases the executor's threads when it ends.
 */
class ParallelSection {
  public:
    explicit ParallelSection(ParallelExecutor& executor) : executor(executor) {}
    ~ParallelSection() { executor.ReleaseThreads(); }
    ParallelSection(const ParallelSection&) = delete;
    ParallelSection& operator=(const ParallelSection&) = delete;

  private:
    ParallelExecutor& executor;
};

} // namespace skmeans

// The default backend is chosen at build time (SKMEANS_EXECUTOR in CMake). Wasm builds without
// threads always run serially.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
#define SKMEANS_EXECUTOR_IS_SERIAL 1
#elif defined(SKMEANS_EXECUTOR_FORKUNION)
#include "superkmeans/executors/forkunion.h"
#elif defined(SKMEANS_EXECUTOR_OPENMP)
#include "superkmeans/executors/openmp.h"
#else
#define SKMEANS_EXECUTOR_IS_SERIAL 1
#endif

namespace skmeans {

inline std::unique_ptr<ParallelExecutor> MakeDefaultExecutor(size_t n_threads) {
#if defined(SKMEANS_EXECUTOR_IS_SERIAL)
    (void) n_threads;
    return std::make_unique<SerialExecutor>();
#elif defined(SKMEANS_EXECUTOR_FORKUNION)
    return std::make_unique<ForkUnionExecutor>(ResolveNumThreads(n_threads));
#else
    return std::make_unique<OpenMPExecutor>(ResolveNumThreads(n_threads));
#endif
}

} // namespace skmeans
