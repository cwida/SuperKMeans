#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include <fork_union.hpp>
#include <numkong/numkong.h>

#include "bench_utils.h"

namespace fu = ashvardanian::fork_union;

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 10;

/**
 * @brief Run a single NumKong f32 packed GEMM: C(m, n) = A(m, d) × B(n, d)^T
 *
 * B is pre-packed. Each thread processes a contiguous chunk of rows from A.
 * Note: NumKong f32 outputs f64 (double) for accumulation precision.
 */
double RunNumKongGemm(
    const float* a,
    const void* b_packed,
    size_t m,
    size_t n,
    size_t d,
    double* out,
    fu::basic_pool_t& pool
) {
    const size_t c_stride = n * sizeof(double);
    auto t0 = std::chrono::high_resolution_clock::now();
    pool.for_slices(m, [&](auto prong, auto count) noexcept {
        nk_configure_thread(nk_capabilities());
        size_t start = prong.task;
        nk_dots_packed_f32(
            a + start * d,    // contiguous row chunk of A
            b_packed,          // shared pre-packed B
            out + start * n,   // contiguous row chunk of C
            count,             // height: multiple rows per thread
            n,                 // width: number of B rows
            d,                 // depth: shared dimension
            d * sizeof(float), // a_stride in bytes
            c_stride           // c_stride in bytes
        );
    });
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <m> <n> <d> [threads]" << std::endl;
    std::cerr << "  m       - number of rows in A (left matrix)" << std::endl;
    std::cerr << "  n       - number of rows in B (right matrix)" << std::endl;
    std::cerr << "  d       - number of columns (dimensions)" << std::endl;
    std::cerr << "  threads - number of threads (default: all cores)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    const size_t m = std::stoull(argv[1]);
    const size_t n = std::stoull(argv[2]);
    const size_t d = std::stoull(argv[3]);
    const size_t threads = (argc >= 5) ? std::stoull(argv[4]) : std::thread::hardware_concurrency();

    fu::basic_pool_t pool;
    if (!pool.try_spawn(threads)) {
        std::cerr << "Failed to spawn thread pool" << std::endl;
        return 1;
    }

    std::cout << "=== NumKong f32→f64 Packed GEMM Microbenchmark ===" << std::endl;
    std::cout << "C(" << m << ", " << n << ") = A(" << m << ", " << d << ") * B(" << n << ", " << d
              << ")^T" << std::endl;
    std::cout << "Threads: " << threads << std::endl;
    std::cout << "Warmup runs: " << WARMUP_RUNS << ", Measured runs: " << MEASURED_RUNS << std::endl;
    std::cout << std::endl;

    nk_configure_thread(nk_capabilities());

    auto a = bench_utils::generate_random_f32(m, d, 42);
    auto b = bench_utils::generate_random_f32(n, d, 123);
    std::vector<double> out(m * n);

    // Pack B once (not timed — amortized across queries in real usage)
    size_t packed_size = nk_dots_packed_size_f32(n, d);
    std::vector<char> b_packed(packed_size);
    std::cout << "Packing B (" << packed_size / (1024 * 1024) << " MB)..." << std::flush;
    nk_dots_pack_f32(b.data(), n, d, d * sizeof(float), b_packed.data());
    std::cout << " done" << std::endl;

    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunNumKongGemm(a.data(), b_packed.data(), m, n, d, out.data(), pool);
    }
    std::cout << " done" << std::endl;

    // Measured runs
    std::cout << "Measuring..." << std::flush;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        double ms = RunNumKongGemm(a.data(), b_packed.data(), m, n, d, out.data(), pool);
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    std::cout << " done" << std::endl;
    std::cout << std::endl;

    double avg_ms = total_ms / MEASURED_RUNS;
    double gflops = (2.0 * m * n * d) / (avg_ms * 1e6);

    std::cout << "=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Avg: " << avg_ms << " ms" << std::endl;
    std::cout << "  Min: " << min_ms << " ms" << std::endl;
    std::cout << "  Max: " << max_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << gflops << " GFLOPS" << std::endl;

    return 0;
}
