#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <pthreadpool.h>
#include <xnnpack.h>

#include "bench_utils.h"

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 10;

/**
 * @brief Run a single XNNPack qd8_f32_qc8w GEMM: C(m, n) = A(m, d) * B(n, d)^T
 *
 * Uses identity quantization (scale=1.0, zero_point=0) so the output is
 * the integer dot product cast to float.
 */
double RunXnnpackGemm(
    const int8_t* a,
    const int8_t* b,
    size_t m,
    size_t n,
    size_t d,
    float* out,
    pthreadpool_t tp
) {
    std::vector<float> kernel_scale(n, 1.0f);
    std::vector<xnn_quantization_params> qparams(m, {0, 1.0f});

    xnn_operator_t op = nullptr;
    xnn_create_fully_connected_nc_qd8_f32_qc8w(
        d, n, d, n,
        kernel_scale.data(), b, nullptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0, nullptr, &op
    );

    size_t ws_size = 0;
    xnn_reshape_fully_connected_nc_qd8_f32_qc8w(op, m, &ws_size, tp);
    std::vector<char> ws(ws_size);
    xnn_setup_fully_connected_nc_qd8_f32_qc8w(op, a, out, ws.data(), qparams.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    xnn_run_operator(op, tp);
    auto t1 = std::chrono::high_resolution_clock::now();

    xnn_delete_operator(op);
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
    const size_t threads = (argc >= 5) ? std::stoull(argv[4]) : 0; // 0 = pthreadpool default

    xnn_initialize(nullptr);

    pthreadpool_t tp = pthreadpool_create(threads);
    std::cout << "=== XNNPack int8 GEMM Microbenchmark ===" << std::endl;
    std::cout << "C(" << m << ", " << n << ") = A(" << m << ", " << d << ") * B(" << n << ", " << d
              << ")^T" << std::endl;
    std::cout << "Threads: " << pthreadpool_get_threads_count(tp) << std::endl;
    std::cout << "Warmup runs: " << WARMUP_RUNS << ", Measured runs: " << MEASURED_RUNS << std::endl;
    std::cout << std::endl;

    auto a = bench_utils::generate_random_i8(m, d, 42);
    auto b = bench_utils::generate_random_i8(n, d, 123);
    std::vector<float> out(m * n);

    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunXnnpackGemm(a.data(), b.data(), m, n, d, out.data(), tp);
    }
    std::cout << " done" << std::endl;

    // Measured runs
    std::cout << "Measuring..." << std::flush;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        double ms = RunXnnpackGemm(a.data(), b.data(), m, n, d, out.data(), tp);
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

    pthreadpool_destroy(tp);
    return 0;
}
