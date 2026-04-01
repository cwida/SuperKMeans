#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <omp.h>
#include <dnnl.hpp>

#include "bench_utils.h"

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 10;

/**
 * @brief Run a single oneDNN u8×s8→s32 matmul: C(m, n) = A(m, d) × B(n, d)^T
 *
 * Uses the matmul primitive with no scales or zero-points,
 * so the output is the raw int32 dot product.
 */
double RunOneDnnGemm(
    dnnl::stream& strm,
    dnnl::matmul& prim,
    dnnl::memory& a_mem,
    dnnl::memory& b_mem,
    dnnl::memory& c_mem
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    prim.execute(
        strm,
        {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem},
        }
    );
    strm.wait();
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

    const auto m = static_cast<dnnl_dim_t>(std::stoull(argv[1]));
    const auto n = static_cast<dnnl_dim_t>(std::stoull(argv[2]));
    const auto d = static_cast<dnnl_dim_t>(std::stoull(argv[3]));
    if (argc >= 5) {
        omp_set_num_threads(std::stoi(argv[4]));
    }

    std::cout << "=== oneDNN u8×s8→s32 MatMul Microbenchmark ===" << std::endl;
    std::cout << "C(" << m << ", " << n << ") = A(" << m << ", " << d << ") * B(" << n << ", " << d
              << ")^T" << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Warmup runs: " << WARMUP_RUNS << ", Measured runs: " << MEASURED_RUNS << std::endl;
    std::cout << std::endl;

    auto a = bench_utils::generate_random_u8(m, d, 42);
    auto b = bench_utils::generate_random_i8(n, d, 123);
    std::vector<int32_t> out(m * n);

    // Create engine and stream
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    // Memory descriptors:
    //   A is (m, d) row-major, u8
    //   B is stored as (n, d) row-major, but used as (d, n) via transposed strides, s8
    //   C is (m, n) row-major, s32
    dnnl::memory::desc a_md({m, d}, dnnl::memory::data_type::u8, {d, 1});
    dnnl::memory::desc b_md({d, n}, dnnl::memory::data_type::s8, {1, d});
    dnnl::memory::desc c_md({m, n}, dnnl::memory::data_type::s32, {n, 1});

    // Create primitive (no scales, no zero-points → raw integer dot product)
    auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto prim = dnnl::matmul(pd);

    // Create memory objects
    dnnl::memory a_mem(a_md, eng, a.data());
    dnnl::memory b_mem(b_md, eng, b.data());
    dnnl::memory c_mem(c_md, eng, out.data());

    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunOneDnnGemm(strm, prim, a_mem, b_mem, c_mem);
    }
    std::cout << " done" << std::endl;

    // Measured runs
    std::cout << "Measuring..." << std::flush;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        double ms = RunOneDnnGemm(strm, prim, a_mem, b_mem, c_mem);
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    std::cout << " done" << std::endl;
    std::cout << std::endl;

    double avg_ms = total_ms / MEASURED_RUNS;
    double gops = (2.0 * m * n * d) / (avg_ms * 1e6);

    std::cout << "=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Avg: " << avg_ms << " ms" << std::endl;
    std::cout << "  Min: " << min_ms << " ms" << std::endl;
    std::cout << "  Max: " << max_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << gops << " GOPS" << std::endl;

    return 0;
}
