#pragma once

#include <cinttypes>
#include <cpuinfo.h>
#include <cstdint>
#include <cstdio>

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

#if defined(__APPLE__)
// Accelerate ships two BLAS implementations: `sgemm_` binds to the legacy one (frozen at LAPACK
// 3.2.1, with known wrong-result bugs on some Apple Silicon / macOS combinations) while the
// maintained one sits behind the $NEWLAPACK symbols (macOS >= 13.3). Declared with an asm label so
// no Accelerate header (which clashes with Eigen's BLAS declarations) is needed; weak so it is
// null on older systems and we fall back to `sgemm_`.
extern "C" void skm_cblas_sgemm_newlapack(
    int order,
    int transa,
    int transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float beta,
    float* c,
    int ldc
) __asm("_cblas_sgemm$NEWLAPACK") __attribute__((weak_import));
#endif

#define SKMEANS_ENSURE_POSITIVE(x)                                                                 \
    if ((x) <= 0) {                                                                                \
        throw std::invalid_argument("Value must be positive: " #x);                                \
    }

#ifndef SKM_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define SKM_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SKM_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
#define SKM_RESTRICT __restrict__
#else
#define SKM_RESTRICT
#endif
#endif

#ifndef SKM_ALWAYS_INLINE
#if __has_cpp_attribute(gnu::always_inline)
#define SKM_ALWAYS_INLINE [[gnu::always_inline]]
#elif defined(__GNUC__) || defined(__clang__)
#define SKM_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SKM_ALWAYS_INLINE __forceinline
#else
#define SKM_ALWAYS_INLINE
#endif
#endif

#ifndef SKM_NO_INLINE
#define SKM_NO_INLINE __attribute__((noinline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SKM_LIKELY(x) __builtin_expect(!!(x), 1)
#define SKM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SKM_LIKELY(x) (x)
#define SKM_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SKM_PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define SKM_PREFETCH(addr, rw, locality)                                                           \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define SKM_PREFETCH(addr, rw, locality) ((void) 0)
#endif

// Cross-compiler vectorization hint for loops.
// Clang: #pragma clang loop vectorize(enable)
// GCC:   #pragma GCC ivdep (asserts no loop-carried dependencies, enabling vectorization)
#if defined(__clang__)
#define SKM_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define SKM_VECTORIZE_LOOP _Pragma("GCC ivdep")
#else
#define SKM_VECTORIZE_LOOP
#endif

namespace skmeans {

// SGEMM. Every GEMM in the code base goes through here.
inline void Sgemm(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    float beta,
    float* c,
    int ldc
) {
#if defined(__APPLE__)
    if (&skm_cblas_sgemm_newlapack != nullptr) {
        constexpr int COL_MAJOR = 102, NO_TRANS = 111, TRANS = 112;
        skm_cblas_sgemm_newlapack(
            COL_MAJOR,
            transa == 'N' ? NO_TRANS : TRANS,
            transb == 'N' ? NO_TRANS : TRANS,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc
        );
        return;
    }
#endif
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

static inline constexpr float PROPORTION_HORIZONTAL_DIM = 0.75;
static inline constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
static inline constexpr size_t H_DIM_SIZE = 64;
static inline constexpr size_t INPLACE_ROTATION_BLOCK_ROWS = 4096;

// Below 32, GEMM stops accelerating
static inline constexpr uint32_t MIN_PARTIAL_D = 32;

// Thresholds below which GEMM-only (no pruning) is used
static inline constexpr size_t DIMENSION_THRESHOLD_FOR_PRUNING = 128;
static inline constexpr size_t N_CLUSTERS_THRESHOLD_FOR_PRUNING = 256;

#if defined(__APPLE__)
// AMX (used with Apple Accelerate) benefits from larger batch sizes
static inline constexpr size_t X_BATCH_SIZE = 40960;
static inline constexpr size_t Y_BATCH_SIZE = 2048;
static inline constexpr size_t MINI_BATCH_SIZE = 256;
#else
static inline constexpr size_t X_BATCH_SIZE = 4096;
static inline constexpr size_t Y_BATCH_SIZE = 1024;
#endif

static inline constexpr size_t VECTOR_CHUNK_SIZE = Y_BATCH_SIZE;

static inline constexpr size_t SKM_MAX_DIMS = 16384;

// Why 4096? To avoid overflow in RabitQ
// RaBitQ FastScan accumulates SQ dot products of nibbles into uint16 lanes
static inline constexpr size_t RABITQ_MAX_DIMS = 4096;

static inline constexpr uint32_t RABITQ_MIDDLE_CHECKPOINT_D_THRESHOLD = 768;

static inline constexpr int RABITQ_SQ_BITS = 4;

static inline constexpr size_t RECALL_CONVERGENCE_PATIENCE = 2;
static inline constexpr float CENTROID_PERTURBATION_EPS = 1.0f / 1024.0f;
// Epsilon parameter of ADSampling (Reference: https://dl.acm.org/doi/abs/10.1145/3589282)
static inline constexpr float PRUNER_INITIAL_THRESHOLD = 1.5f;
static inline constexpr float HIERARCHICAL_PRUNER_INITIAL_THRESHOLD = 1.1f;

// Global thread count for OpenMP parallel regions
// This is set by SuperKMeans constructor. Not ideal but needed for
// external functions (adsampling, batch_computers) that can't access class members.
inline uint32_t g_n_threads = 1;

template <class T, T val = 8>
static constexpr uint32_t AlignValue(T n) {
    return ((n + (val - 1)) / val) * val;
}

// Below this threshold, NumKong loses to Ruy for 8-bit GEMM
static constexpr size_t THIN_MATRIX_THRESHOLD = 256;

#if defined(__ARM_NEON)
inline constexpr bool IS_ARM = true;
#else
inline constexpr bool IS_ARM = false;
#endif

inline bool DetectAMX() {
    cpuinfo_initialize();
    return cpuinfo_has_x86_amx_int8();
}

enum class DistanceFunction : uint8_t { l2, dp };

enum class Quantization : uint8_t { f32, sq8, lvq4, rabitq, sq4 };

inline const char* QuantizationName(Quantization q) {
    switch (q) {
    case Quantization::f32:
        return "f32";
    case Quantization::sq8:
        return "sq8";
    case Quantization::lvq4:
        return "lvq4";
    case Quantization::rabitq:
        return "rabitq";
    case Quantization::sq4:
        return "sq4";
    }
    return "unknown";
}

template <Quantization q>
struct QuantizationTag {
    static constexpr Quantization value = q;
};

class F32Quantizer;
class SQ8Quantizer;
class LVQ4Quantizer;
class RaBitQQuantizer;

template <Quantization q>
struct QuantizerClass;
template <>
struct QuantizerClass<Quantization::f32> {
    using type = F32Quantizer;
};
template <>
struct QuantizerClass<Quantization::sq8> {
    using type = SQ8Quantizer;
};
template <>
struct QuantizerClass<Quantization::lvq4> {
    using type = LVQ4Quantizer;
};
template <>
struct QuantizerClass<Quantization::rabitq> {
    using type = RaBitQQuantizer;
};
template <Quantization q>
using skmeans_quantizer_t = typename QuantizerClass<q>::type;

// Distances and centroids are always float, regardless of the quantization scheme
using skmeans_distance_t = float;
using skmeans_centroid_value_t = float;

// PDX-internal distance type: uint32_t for the quantized schemes (exact integer accumulation in
// quantized domain), float for f32. Final distances (out_distances, KNNCandidate) always use float.
template <Quantization q>
struct PDXDistanceType {
    using type = skmeans_distance_t;
};
template <>
struct PDXDistanceType<Quantization::sq8> {
    using type = uint32_t;
};
template <>
struct PDXDistanceType<Quantization::sq4> {
    using type = uint32_t;
};
template <>
struct PDXDistanceType<Quantization::rabitq> {
    using type = uint32_t;
};
template <Quantization q>
using pdx_distance_t = typename PDXDistanceType<q>::type;

template <Quantization q>
struct DataType {
    using type = uint8_t;
};
template <>
struct DataType<Quantization::f32> {
    using type = float;
};
template <Quantization q>
using skmeans_value_t = typename DataType<q>::type;

struct KNNCandidate {
    uint32_t index;
    float distance;
};

struct VectorComparator {
    bool operator()(const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

template <Quantization q>
struct Cluster {
    uint32_t num_embeddings{};
    uint32_t* indices = nullptr;
    skmeans_value_t<q>* data = nullptr;
    skmeans_value_t<q>* aux_vertical_dimensions_in_horizontal_layout =
        nullptr; // Contains the vertical dimensions minus partial_d in a horizontal layout, aka the
                 // ones not visited by GEMM
};

} // namespace skmeans
