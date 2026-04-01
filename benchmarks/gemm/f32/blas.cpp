#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "bench_utils.h"

extern "C" {
int sgemm_(
    const char* transa,
    const char* transb,
    int* m,
    int* n,
    int* k,
    const float* alpha,
    const float* a,
    int* lda,
    const float* b,
    int* ldb,
    float* beta,
    float* c,
    int* ldc
);
}

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 10;

/**
 * @brief Run a single BLAS sgemm: C(m, n) = A(m, d) × B(n, d)^T
 *
 * BLAS uses column-major, so we compute C^T = B × A^T in Fortran layout,
 * which is equivalent to C = A × B^T in row-major.
 */
double RunBlasGemm(
    const float* a,
    const float* b,
    size_t m,
    size_t n,
    size_t d,
    float* out
) {
    // Row-major C(m,n) = A(m,d) × B(n,d)^T  ↔  col-major C'(n,m) = B'^T(n,d) × A'(d,m)
    char transa = 'T'; // B stored col-major as (d,n), transpose to get (n,d)
    char transb = 'N'; // A stored col-major as (d,m), no transpose
    int blas_m = static_cast<int>(n);
    int blas_n = static_cast<int>(m);
    int blas_k = static_cast<int>(d);
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = static_cast<int>(d); // leading dim of B in col-major = d
    int ldb = static_cast<int>(d); // leading dim of A in col-major = d
    int ldc = static_cast<int>(n); // leading dim of C in col-major = n

    auto t0 = std::chrono::high_resolution_clock::now();
    sgemm_(&transa, &transb, &blas_m, &blas_n, &blas_k,
           &alpha, b, &lda, a, &ldb, &beta, out, &ldc);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <m> <n> <d>" << std::endl;
    std::cerr << "  m - number of rows in A (left matrix)" << std::endl;
    std::cerr << "  n - number of rows in B (right matrix)" << std::endl;
    std::cerr << "  d - number of columns (dimensions)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    const size_t m = std::stoull(argv[1]);
    const size_t n = std::stoull(argv[2]);
    const size_t d = std::stoull(argv[3]);

    std::cout << "=== BLAS sgemm f32 Microbenchmark ===" << std::endl;
    std::cout << "C(" << m << ", " << n << ") = A(" << m << ", " << d << ") * B(" << n << ", " << d
              << ")^T" << std::endl;
    std::cout << "Warmup runs: " << WARMUP_RUNS << ", Measured runs: " << MEASURED_RUNS << std::endl;
    std::cout << std::endl;

    auto a = bench_utils::generate_random_f32(m, d, 42);
    auto b = bench_utils::generate_random_f32(n, d, 123);
    std::vector<float> out(m * n);

    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunBlasGemm(a.data(), b.data(), m, n, d, out.data());
    }
    std::cout << " done" << std::endl;

    // Measured runs
    std::cout << "Measuring..." << std::flush;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        double ms = RunBlasGemm(a.data(), b.data(), m, n, d, out.data());
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
