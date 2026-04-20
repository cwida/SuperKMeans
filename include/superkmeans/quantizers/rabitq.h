#pragma once

#ifdef HAS_FAISS

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/impl/RaBitQuantizer.h>

namespace skmeans {

// Matches faiss::FactorsData layout (defined in RaBitQuantizer.cpp, not exported).
// Each RaBitQ code stores binary_bytes + this struct at the end.
struct RaBitQFactors {
    float or_minus_c_l2sqr; // ||original - centroid||²  (for L2 metric)
    float dp_multiplier;    // scaling factor for dot-product estimation
};
static_assert(sizeof(RaBitQFactors) == 8, "RaBitQFactors must match FAISS FactorsData");

/**
 * @brief RaBitQ (1-bit) quantizer wrapping faiss::RaBitQuantizer.
 *
 * RaBitQ sign-quantizes centered vectors to 1 bit per dimension plus 8 bytes
 * of metadata per vector. Distance is asymmetric: one side must be float
 * (the "query" in FAISS terms), the other side is binary codes.
 *
 * For k-means we flip the FAISS roles: centroids (small, K) are FAISS "queries"
 * kept as float, data points (large, N) are FAISS "codes" stored as compact
 * binary. This avoids decoding/re-quantizing data points on every iteration.
 */
template <Quantization q>
class RaBitQQuantizer : public IQuantizer<q> {
    static_assert(q == Quantization::u8, "RaBitQQuantizer only supports u8");

  public:
    using quantized_t = typename IQuantizer<q>::quantized_t;

    void Fit(const float* data, size_t n, size_t d) override {
        d_ = d;
        centroid_.resize(d, 0.0f);

        // Compute dataset mean as the centering centroid for RaBitQ
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0;
            for (size_t i = 0; i < n; ++i) {
                sum += data[i * d + j];
            }
            centroid_[j] = static_cast<float>(sum / static_cast<double>(n));
        }

        faiss_quantizer_ = std::make_unique<faiss::RaBitQuantizer>(d, faiss::METRIC_L2);
        faiss_quantizer_->centroid = centroid_.data();
        faiss_code_size_ = faiss_quantizer_->code_size;
        assert(faiss_code_size_ == CodeSize(d));
        fitted_ = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        std::cout << "RaBitQQuantizer::Encode called with n=" << n << ", d=" << d << std::endl;
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->compute_codes(in, reinterpret_cast<uint8_t*>(out), n);
        std::cout << "RaBitQQuantizer::Encode completed" << std::endl;
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->decode(reinterpret_cast<const uint8_t*>(in), out, n);
    }

    void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted_);
        const size_t binary_bytes = (d + 7) / 8;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code =
                reinterpret_cast<const uint8_t*>(data) + i * faiss_code_size_;
            const auto* factors =
                reinterpret_cast<const RaBitQFactors*>(code + binary_bytes);
            out_norms[i] = factors->or_minus_c_l2sqr;
        }
    }

    /**
     * @brief Find top-1 nearest neighbor using RaBitQ asymmetric distance.
     *
     * Centroids (y_float) are used directly as FAISS "queries".
     * Data points (x) stay as compact binary codes (FAISS "codes").
     */
    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        assert(fitted_);
        (void)y;
        (void)x_float;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        std::cout << "RaBitQQuantizer::FindNearestNeighbor called with n_x=" << n_x
                  << ", n_y=" << n_y << ", d=" << d << std::endl;
        CoreDistanceLoop(
            reinterpret_cast<const uint8_t*>(x),
            y_float,
            n_x, n_y, d,
            out_knn, out_distances
        );
        std::cout << "RaBitQQuantizer::FindNearestNeighbor completed" << std::endl;
    }

    size_t DefaultRerankK() const override { return 0; }

    /**
     * @brief Coarse RaBitQ search + exact f32 reranking.
     *
     * Phase 1: RaBitQ asymmetric distance (float centroid × binary data point)
     *          to find top rerank_k candidates per data point.
     * Phase 2: Exact f32 L2² reranking among candidates.
     */
    void FindNearestNeighborWithReranking(
        const quantized_t* x_quantized,
        const quantized_t* y_quantized,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        size_t rerank_k,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        assert(fitted_);
        (void)y_quantized;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        using f32_computer = DistanceComputer<DistanceFunction::l2, Quantization::f32>;

        // Phase 1: Coarse search — centroids (y_float) as FAISS queries
        std::vector<uint32_t> topk_indices(n_x * rerank_k, static_cast<uint32_t>(-1));
        std::vector<float> topk_distances(n_x * rerank_k, std::numeric_limits<float>::max());

        CoreDistanceLoopTopK(
            reinterpret_cast<const uint8_t*>(x_quantized),
            y_float,
            n_x, n_y, d,
            rerank_k,
            topk_indices.data(),
            topk_distances.data()
        );

        // Phase 2: F32 reranking among top-k candidates
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_idx = 0;

            for (size_t ki = 0; ki < rerank_k; ++ki) {
                const uint32_t cand_idx = topk_indices[i * rerank_k + ki];
                if (cand_idx == static_cast<uint32_t>(-1)) break;

                const float dist =
                    f32_computer::Horizontal(x_float + i * d, y_float + cand_idx * d, d);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = cand_idx;
                }
            }

            out_knn[i] = best_idx;
            out_distances[i] = best_dist;
        }
    }

    size_t CodeSize(size_t d) const override {
        return (d + 7) / 8 + sizeof(RaBitQFactors);
    }

    bool IsFitted() const override { return fitted_; }

  private:
    /// Core distance loop: centroids are FAISS "queries" (float), data points
    /// are FAISS "codes" (binary). Returns top-1 per data point.
    void CoreDistanceLoop(
        const uint8_t* x_codes,
        const float* y_float_centroids,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances
    ) const {
        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        const uint32_t num_threads = g_n_threads;

        // Per-thread FAISS distance computers (not thread-safe, each thread needs its own)
        std::vector<std::unique_ptr<faiss::FlatCodesDistanceComputer>> dcs(num_threads);
        for (uint32_t t = 0; t < num_threads; ++t) {
            dcs[t].reset(faiss_quantizer_->get_distance_computer(0, centroid_.data()));
        }

        for (size_t j = 0; j < n_y; ++j) {
            // Set centroid j as FAISS query on all thread-local distance computers
            for (uint32_t t = 0; t < num_threads; ++t) {
                dcs[t]->set_query(y_float_centroids + j * d);
            }

#pragma omp parallel for num_threads(g_n_threads)
            for (size_t i = 0; i < n_x; ++i) {
                const int tid = omp_get_thread_num();
                const float dist =
                    dcs[tid]->distance_to_code(x_codes + i * faiss_code_size_);
                if (dist < out_distances[i]) {
                    out_distances[i] = dist;
                    out_knn[i] = static_cast<uint32_t>(j);
                }
            }
        }
    }

    /// Core distance loop returning top-k candidates per data point.
    void CoreDistanceLoopTopK(
        const uint8_t* x_codes,
        const float* y_float_centroids,
        size_t n_x,
        size_t n_y,
        size_t d,
        size_t rerank_k,
        uint32_t* topk_indices,
        float* topk_distances
    ) const {
        std::fill_n(topk_distances, n_x * rerank_k, std::numeric_limits<float>::max());
        std::fill_n(topk_indices, n_x * rerank_k, static_cast<uint32_t>(-1));

        const uint32_t num_threads = g_n_threads;

        std::vector<std::unique_ptr<faiss::FlatCodesDistanceComputer>> dcs(num_threads);
        for (uint32_t t = 0; t < num_threads; ++t) {
            dcs[t].reset(faiss_quantizer_->get_distance_computer(0, centroid_.data()));
        }

        for (size_t j = 0; j < n_y; ++j) {
            for (uint32_t t = 0; t < num_threads; ++t) {
                dcs[t]->set_query(y_float_centroids + j * d);
            }

#pragma omp parallel for num_threads(g_n_threads)
            for (size_t i = 0; i < n_x; ++i) {
                const int tid = omp_get_thread_num();
                const float dist =
                    dcs[tid]->distance_to_code(x_codes + i * faiss_code_size_);

                float* tk_dist = topk_distances + i * rerank_k;
                uint32_t* tk_idx = topk_indices + i * rerank_k;

                // Find the worst (largest distance) in current top-k
                size_t worst_pos = 0;
                for (size_t ki = 1; ki < rerank_k; ++ki) {
                    if (tk_dist[ki] > tk_dist[worst_pos]) {
                        worst_pos = ki;
                    }
                }

                if (dist < tk_dist[worst_pos]) {
                    tk_dist[worst_pos] = dist;
                    tk_idx[worst_pos] = static_cast<uint32_t>(j);
                }
            }
        }
    }

    size_t d_ = 0;
    size_t faiss_code_size_ = 0;
    std::vector<float> centroid_;
    std::unique_ptr<faiss::RaBitQuantizer> faiss_quantizer_;
    bool fitted_ = false;
};

} // namespace skmeans

#endif // HAS_FAISS
