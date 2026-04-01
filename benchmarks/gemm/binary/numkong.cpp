#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <fork_union.hpp>
#include <numkong/numkong.h>

#include "bench_utils.h"

namespace fu = ashvardanian::fork_union;

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 10;

/**
 * @brief Generate random binary data packed as u1x8 (8 bits per byte).
 * @param n Number of vectors (rows).
 * @param d Number of dimensions (bits). Each row is ceil(d/8) bytes.
 */
std::vector<uint8_t> generate_random_binary(size_t n, size_t d, uint32_t seed = 42) {
    size_t bytes_per_row = (d + 7) / 8;
    std::vector<uint8_t> data(n * bytes_per_row);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data) {
        v = static_cast<uint8_t>(dist(rng));
    }
    return data;
}

/**
 * @brief Run a single NumKong binary Hamming distance computation.
 *
 * C(m, n) = hamming(A, B) where A and B are binary matrices (1-bit per dimension).
 * B is pre-packed. Output is u32 Hamming distances.
 */
double RunNumKongHamming(
    const uint8_t* a,
    const void* b_packed,
    size_t m,
    size_t n,
    size_t d,
    uint32_t* out,
    size_t bytes_per_row,
    fu::basic_pool_t& pool
) {
    const size_t c_stride = n * sizeof(uint32_t);
    auto t0 = std::chrono::high_resolution_clock::now();
    pool.for_slices(m, [&](auto prong, auto count) noexcept {
        nk_configure_thread(nk_capabilities());
        size_t start = prong.task;
        nk_hammings_packed_u1(
            reinterpret_cast<const nk_u1x8_t*>(a + start * bytes_per_row),
            b_packed,
            out + start * n,
            count,              // rows
            n,                  // cols
            d,                  // depth in bits
            bytes_per_row,      // v_stride in bytes
            c_stride            // r_stride in bytes
        );
    });
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <m> <n> <d> [threads]" << std::endl;
    std::cerr << "  m       - number of rows in A (left matrix)" << std::endl;
    std::cerr << "  n       - number of rows in B (right matrix)" << std::endl;
    std::cerr << "  d       - number of dimensions (bits)" << std::endl;
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
    const size_t bytes_per_row = (d + 7) / 8;

    fu::basic_pool_t pool;
    if (!pool.try_spawn(threads)) {
        std::cerr << "Failed to spawn thread pool" << std::endl;
        return 1;
    }

    std::cout << "=== NumKong Binary Hamming Packed Microbenchmark ===" << std::endl;
    std::cout << "C(" << m << ", " << n << ") = hamming(A(" << m << ", " << d << "b), B(" << n << ", " << d
              << "b))" << std::endl;
    std::cout << "Bytes per row: " << bytes_per_row << std::endl;
    std::cout << "Threads: " << threads << std::endl;
    std::cout << "Warmup runs: " << WARMUP_RUNS << ", Measured runs: " << MEASURED_RUNS << std::endl;
    std::cout << std::endl;

    nk_configure_thread(nk_capabilities());

    auto a = generate_random_binary(m, d, 42);
    auto b = generate_random_binary(n, d, 123);
    std::vector<uint32_t> out(m * n);

    // Pack B once (not timed — amortized across queries in real usage)
    size_t packed_size = nk_dots_packed_size_u1(n, d);
    std::vector<char> b_packed(packed_size);
    std::cout << "Packing B (" << packed_size / (1024 * 1024) << " MB)..." << std::flush;
    nk_dots_pack_u1(reinterpret_cast<const nk_u1x8_t*>(b.data()), n, d, bytes_per_row, b_packed.data());
    std::cout << " done" << std::endl;

    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunNumKongHamming(a.data(), b_packed.data(), m, n, d, out.data(), bytes_per_row, pool);
    }
    std::cout << " done" << std::endl;

    // Measured runs
    std::cout << "Measuring..." << std::flush;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        double ms = RunNumKongHamming(a.data(), b_packed.data(), m, n, d, out.data(), bytes_per_row, pool);
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    std::cout << " done" << std::endl;
    std::cout << std::endl;

    double avg_ms = total_ms / MEASURED_RUNS;
    // For binary: each (row, col) pair does ceil(d/8) byte XORs + popcount
    // Report as billion bit-operations per second
    double gbops = (static_cast<double>(m) * n * d) / (avg_ms * 1e6);

    std::cout << "=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Avg: " << avg_ms << " ms" << std::endl;
    std::cout << "  Min: " << min_ms << " ms" << std::endl;
    std::cout << "  Max: " << max_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << gbops << " Gbitops" << std::endl;

    return 0;
}
