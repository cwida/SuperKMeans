#include <atomic>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "superkmeans/executor.h"

namespace {

using skmeans::ParallelExecutor;

constexpr size_t DEFAULT_EXECUTOR_THREADS[] = {1, 3, 0};
constexpr size_t LOOP_SIZES[] = {0, 1, 2, 7, 1000, 1001};

std::vector<std::unique_ptr<ParallelExecutor>> MakeExecutors() {
    std::vector<std::unique_ptr<ParallelExecutor>> executors;
    executors.push_back(std::make_unique<skmeans::SerialExecutor>());
    for (const size_t n_threads : DEFAULT_EXECUTOR_THREADS) {
        executors.push_back(skmeans::MakeDefaultExecutor(n_threads));
    }
    return executors;
}

// Every index is visited once, by a valid worker that gets at most one range.
TEST(ExecutorTest, ParallelForCoversEveryIndexOnce) {
    for (auto& executor : MakeExecutors()) {
        for (const size_t n : LOOP_SIZES) {
            SCOPED_TRACE(
                "workers=" + std::to_string(executor->NumWorkers()) + " n=" + std::to_string(n)
            );
            std::vector<std::atomic<int>> hits(n);
            std::vector<std::atomic<int>> ranges_per_worker(executor->NumWorkers());
            std::atomic<bool> invalid_worker{false};
            executor->ParallelFor(n, [&](size_t begin, size_t end, size_t worker) {
                if (worker >= ranges_per_worker.size()) {
                    invalid_worker = true;
                    return;
                }
                ranges_per_worker[worker]++;
                for (size_t i = begin; i < end; ++i) {
                    hits[i]++;
                }
            });
            EXPECT_FALSE(invalid_worker);
            for (size_t i = 0; i < n; ++i) {
                ASSERT_EQ(hits[i], 1) << "index " << i;
            }
            for (const auto& ranges : ranges_per_worker) {
                EXPECT_LE(ranges, 1);
            }
        }
    }
}

// A worker's exception is rethrown in the caller, and the executor keeps working.
TEST(ExecutorTest, ExceptionReachesCaller) {
    const size_t n = 1000;
    for (auto& executor : MakeExecutors()) {
        SCOPED_TRACE("workers=" + std::to_string(executor->NumWorkers()));
        EXPECT_THROW(
            executor->ParallelFor(
                n,
                [&](size_t, size_t end, size_t) {
                    if (end == n) {
                        throw std::runtime_error("last range");
                    }
                }
            ),
            std::runtime_error
        );
        std::atomic<size_t> visited{0};
        executor->ParallelFor(n, [&](size_t begin, size_t end, size_t) { visited += end - begin; });
        EXPECT_EQ(visited, n);
    }
}

#if defined(SKMEANS_EXECUTOR_FORKUNION) && !defined(_WIN32)
// A child forked while the parent's pool is alive runs a ParallelFor to completion.
TEST(ExecutorTest, ForkedChildRunsParallelFor) {
    const size_t n = 1000;
    auto executor = skmeans::MakeDefaultExecutor(4);
    std::atomic<size_t> visited{0};
    executor->ParallelFor(n, [&](size_t begin, size_t end, size_t) { visited += end - begin; });
    ASSERT_EQ(visited, n);

    const pid_t pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
        alarm(10); // a hang kills the child instead of blocking the test
        std::atomic<size_t> child_visited{0};
        executor->ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            child_visited += end - begin;
        });
        _exit(child_visited == n ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child status " << status;
}
#endif

// n_threads sizes the default executor; an injected executor is used as is.
TEST(ExecutorTest, ScopeSizesDefaultAndBorrowsInjected) {
    EXPECT_GE(skmeans::ResolveNumThreads(0), 1u);
    EXPECT_EQ(skmeans::ResolveNumThreads(3), 3u);

    skmeans::ExecutorScope owned(nullptr, 2);
#if defined(SKMEANS_EXECUTOR_IS_SERIAL)
    EXPECT_EQ(owned.Get().NumWorkers(), 1u);
#else
    EXPECT_EQ(owned.Get().NumWorkers(), 2u);
#endif

    skmeans::SerialExecutor injected;
    skmeans::ExecutorScope borrowed(&injected, 8);
    EXPECT_EQ(&borrowed.Get(), &injected);
}

} // namespace
