#pragma once

// #ifdef HAS_FAISS

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

#include <faiss/IndexRaBitQFastScan.h>
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
 * @brief RaBitQ (1-bit) quantizer using IndexRaBitQFastScan for distance.
 *
 * Data points are encoded to 1-bit RaBitQ codes (32x compression).
 * Distance computation uses FastScan with centroids as the database:
 * - K centroids indexed in FastScan (sign-quantized, rebuilt each iteration)
 * - N data points decoded from 1-bit codes to approximate float in streaming
 *   chunks, then searched as queries (SQ-quantized to qb bits by FastScan)
 * - FastScan processes 32 centroids at a time with SIMD
 *
 * This gives per-data-point results (correct direction for k-means) while
 * keeping the N data points stored as compact 1-bit codes.
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
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->compute_codes(in, reinterpret_cast<uint8_t*>(out), n);
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
     * @brief Find top-1 nearest centroid per data point.
     *
     * Data points (x) are 1-bit codes decoded in streaming chunks.
     * Centroids (y_float) are indexed in a FastScan database.
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

        CoreDistanceLoop(
            reinterpret_cast<const uint8_t*>(x),
            y_float, n_x, n_y, d,
            out_knn, out_distances
        );
    }

    size_t DefaultRerankK() const override { return 0; }

    /**
     * @brief Coarse RaBitQ search (decoded 1-bit) + exact f32 reranking.
     *
     * Phase 1: Decode 1-bit codes in chunks → FastScan top-k candidates.
     * Phase 2: Exact f32 L2² reranking using original float data.
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

        // Phase 1: Decoded chunked FastScan coarse search
        std::vector<float> topk_distances(n_x * rerank_k);
        std::vector<faiss::idx_t> topk_labels(n_x * rerank_k);
        CoreDistanceLoopTopK(
            reinterpret_cast<const uint8_t*>(x_quantized),
            y_float, n_x, n_y, d,
            rerank_k, topk_labels.data(), topk_distances.data()
        );

        // Phase 2: F32 reranking among top-k candidates
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_idx = 0;

            for (size_t ki = 0; ki < rerank_k; ++ki) {
                const faiss::idx_t cand_idx = topk_labels[i * rerank_k + ki];
                if (cand_idx < 0) break;

                const float dist =
                    f32_computer::Horizontal(x_float + i * d, y_float + cand_idx * d, d);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = static_cast<uint32_t>(cand_idx);
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
    static constexpr size_t DECODE_CHUNK = 8192;

    /// Build a FastScan index with centroids as the database.
    faiss::IndexRaBitQFastScan BuildFastScanIndex(
        const float* y_float_centroids,
        size_t n_y,
        size_t d
    ) const {
        faiss::IndexRaBitQFastScan fast_index(d, faiss::METRIC_L2, 32, 1);
        fast_index.qb = qb_;

        // Train on centroids (computes center)
        fast_index.train(n_y, y_float_centroids);

        // Override center with our full-dataset centroid for consistency
        fast_index.center = centroid_;
        fast_index.rabitq.centroid = fast_index.center.data();

        // Add centroids to the index (sign-quantized)
        fast_index.add(n_y, y_float_centroids);
        return fast_index;
    }

    /// Decode 1-bit codes in streaming chunks, search each chunk via FastScan.
    void CoreDistanceLoop(
        const uint8_t* x_codes,
        const float* y_float_centroids,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances
    ) const {
        auto fast_index = BuildFastScanIndex(y_float_centroids, n_y, d);

        std::vector<float> decoded(DECODE_CHUNK * d);
        std::vector<float> chunk_dists(DECODE_CHUNK);
        std::vector<faiss::idx_t> chunk_labels(DECODE_CHUNK);

        for (size_t i = 0; i < n_x; i += DECODE_CHUNK) {
            const size_t chunk_n = std::min(DECODE_CHUNK, n_x - i);

            // Decode 1-bit codes → approximate float
            faiss_quantizer_->decode(
                x_codes + i * faiss_code_size_, decoded.data(), chunk_n
            );

            // Search decoded chunk as queries against centroid database
            fast_index.search(
                chunk_n, decoded.data(), 1,
                chunk_dists.data(), chunk_labels.data()
            );

            for (size_t j = 0; j < chunk_n; ++j) {
                out_knn[i + j] = static_cast<uint32_t>(chunk_labels[j]);
                out_distances[i + j] = chunk_dists[j];
            }
        }
    }

    /// Decode 1-bit codes in streaming chunks, search each chunk for top-k.
    void CoreDistanceLoopTopK(
        const uint8_t* x_codes,
        const float* y_float_centroids,
        size_t n_x,
        size_t n_y,
        size_t d,
        size_t rerank_k,
        faiss::idx_t* topk_labels,
        float* topk_distances
    ) const {
        auto fast_index = BuildFastScanIndex(y_float_centroids, n_y, d);

        std::vector<float> decoded(DECODE_CHUNK * d);

        for (size_t i = 0; i < n_x; i += DECODE_CHUNK) {
            const size_t chunk_n = std::min(DECODE_CHUNK, n_x - i);

            faiss_quantizer_->decode(
                x_codes + i * faiss_code_size_, decoded.data(), chunk_n
            );

            fast_index.search(
                chunk_n, decoded.data(),
                static_cast<faiss::idx_t>(rerank_k),
                topk_distances + i * rerank_k,
                topk_labels + i * rerank_k
            );
        }
    }

    uint8_t qb_ = 1;
    size_t d_ = 0;
    size_t faiss_code_size_ = 0;
    std::vector<float> centroid_;
    std::unique_ptr<faiss::RaBitQuantizer> faiss_quantizer_;
    bool fitted_ = false;
};

} // namespace skmeans

// #endif // HAS_FAISS
