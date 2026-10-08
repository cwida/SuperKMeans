#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace skmeans {

struct RaBitQFactors {
    float or_minus_c_l2sqr; // ||original - centroid||²  (for L2 metric)
    float dp_multiplier;    // scaling factor for dot-product estimation
};
static_assert(sizeof(RaBitQFactors) == 8);

/**
 * @brief Trained RaBitQ parameters.
 */
struct RaBitQParams {
    const float* centroid;
    size_t binary_bytes;
};

/**
 * @brief RaBitQ quantizer with FastScan-accelerated distance kernel.
 *
 * Distance computation uses a custom FastScan kernel (VPSHUFB/TBL lookups)
 * that processes 32 data points simultaneously per centroid.
 *
 * For k-means: centroids (K, small) are SQ-quantized to qb bits once,
 * then LUTs are built from the quantized values. Data points (N, large)
 * are byte-transposed into blocks of 32 for SIMD lookup.
 *
 * Code layout per vector:
 *   [(d+7)/8 sign bytes] [float or_minus_c_l2sqr: 4B] [float dp_multiplier: 4B]
 *   CodeSize(d) = (d+7)/8 + 8
 * The centering centroid is global (see GetParams()); the two factors are per-vector.
 */
class RaBitQQuantizer : public IQuantizer<Quantization::rabitq> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    void InvalidateCaches() override {
        cached_x_ptr_ = nullptr;
        cached_n_x_ = 0;
        cached_n_blocks_ = 0;
        pruning_partial_norms_dirty_ = true;
    }

    void Fit(ParallelExecutor& executor, const float* data, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("RQ::Fit");
        if (d % 8 != 0) {
            throw std::invalid_argument(
                "RaBitQ requires dimensionality divisible by 8 (got " + std::to_string(d) + ")"
            );
        }
        if (d > RABITQ_MAX_DIMS) {
            throw std::invalid_argument(
                "RaBitQ requires dimensionality <= " + std::to_string(RABITQ_MAX_DIMS) + " (got " +
                std::to_string(d) + ")"
            );
        }
        d_ = d;
        binary_bytes_ = (d + 7) / 8;
        centroid_.resize(d, 0.0f);

        // Compute dataset mean as the centering centroid for RaBitQ
        const size_t n_workers = executor.NumWorkers();
        std::vector<float> worker_sums(n_workers * d, 0.0f);
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t worker) {
            std::vector<float> local(d, 0.0);
            for (size_t i = begin; i < end; ++i) {
                for (size_t j = 0; j < d; ++j) {
                    local[j] += data[i * d + j];
                }
            }
            std::copy(local.begin(), local.end(), worker_sums.begin() + worker * d);
        });
        std::vector<float> sums(d, 0.0);
        for (size_t w = 0; w < n_workers; ++w) {
            for (size_t j = 0; j < d; ++j)
                sums[j] += worker_sums[w * d + j];
        }
        for (size_t j = 0; j < d; ++j) {
            centroid_[j] = static_cast<float>(sums[j] / static_cast<float>(n));
        }

        code_size_ = binary_bytes_ + sizeof(RaBitQFactors);
        fitted_ = true;
    }

    void Encode(ParallelExecutor& executor, const float* in, quantized_t* out, size_t n, size_t d)
        const override {
        SKM_PROFILE_SCOPE("RQ::Encode");
        uint8_t* codes = reinterpret_cast<uint8_t*>(out);
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                RaBitQCodec::EncodeOne(
                    in + i * d, codes + i * code_size_, d, binary_bytes_, centroid_.data()
                );
            }
        });
    }

    void Decode(ParallelExecutor& executor, const quantized_t* in, float* out, size_t n, size_t d)
        const override {
        SKM_PROFILE_SCOPE("RQ::Decode");
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(in);
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                RaBitQCodec::DecodeOne(
                    codes + i * code_size_, out + i * d, d, binary_bytes_, centroid_.data()
                );
            }
        });
    }

    void ComputeNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        size_t /*d*/,
        float* out_norms
    ) const override {
        SKM_PROFILE_SCOPE("RQ::ComputeNorms");
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = reinterpret_cast<const uint8_t*>(data) + i * code_size_;
                float or_minus_c_l2sqr;
                std::memcpy(&or_minus_c_l2sqr, code + binary_bytes_, sizeof(float));
                out_norms[i] = or_minus_c_l2sqr;
            }
        });
    }

    void FindNearestNeighbor(
        ParallelExecutor& executor,
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
        SKM_PROFILE_SCOPE("RQ::Search::FastScanDistance");
        (void) y;
        (void) x_float;
        (void) norms_x;
        (void) norms_y;
        (void) tmp_buf;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);

        // Cache per-data-point factors (depend only on x, reused across iterations)
        EnsureCodeFactorsCache(executor, x_codes, n_x);
        EnsureTransposedBlocksCache(executor, x_codes, n_x);
        const uint32_t* sum_q = cached_sum_q_.data();
        const float* or_c_l2sqr = cached_or_c_l2sqr_.data();
        const float* dp_mult = cached_dp_mult_.data();

        // Quantize centroids and build LUTs (changes every iteration)
        const size_t n_sub = 2 * binary_bytes_; // 2 sub-quantizers per byte position
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        QuantizeCentroidsAndBuildLUTs(
            executor,
            y_float,
            n_y,
            d,
            all_luts.data(),
            c1.data(),
            c2.data(),
            c34.data(),
            qr_to_c_l2sqr.data()
        );

        const size_t lut_stride = n_sub * 16;
        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        const size_t n_blocks = cached_n_blocks_;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        {
            constexpr size_t kSuperBlock = 4;
            constexpr size_t kBS = FastScanComputer::kBlockSize;
            const size_t n_groups = (n_blocks + kSuperBlock - 1) / kSuperBlock;

            executor.ParallelFor(n_groups, [&](size_t group_begin, size_t group_end, size_t) {
                for (size_t group = group_begin; group < group_end; ++group) {
                    const size_t blk_base = group * kSuperBlock;
                    const size_t n_blks = std::min(kSuperBlock, n_blocks - blk_base);

                    const uint8_t* packed_ptrs[kSuperBlock];
                    size_t blk_starts[kSuperBlock], blk_counts[kSuperBlock];
                    for (size_t bi = 0; bi < n_blks; ++bi) {
                        const size_t blk = blk_base + bi;
                        packed_ptrs[bi] = cached_transposed_.get() + blk * block_bytes;
                        blk_starts[bi] = blk * kBS;
                        blk_counts[bi] = std::min(kBS, n_x - blk_starts[bi]);
                    }

                    float best_dist[kSuperBlock][kBS];
                    uint32_t best_idx[kSuperBlock][kBS];
                    float dist_buf[kSuperBlock][kBS];
                    float sum_q_f32[kSuperBlock][kBS];

                    for (size_t bi = 0; bi < n_blks; ++bi) {
                        std::fill_n(best_dist[bi], kBS, std::numeric_limits<float>::max());
                        std::fill_n(best_idx[bi], kBS, 0u);
                        for (size_t k = 0; k < blk_counts[bi]; ++k) {
                            sum_q_f32[bi][k] = static_cast<float>(sum_q[blk_starts[bi] + k]);
                        }
                    }

                    for (size_t j = 0; j < n_y; ++j) {
                        const uint8_t* lut_j = all_luts.data() + j * lut_stride;

                        // Multi-block ScanBlock: share LUT across all blocks
                        uint16_t dot_qo[kSuperBlock][kBS];
                        if (n_blks == kSuperBlock) {
                            uint16_t* out_ptrs[kSuperBlock];
                            for (size_t bi = 0; bi < kSuperBlock; ++bi) {
                                out_ptrs[bi] = dot_qo[bi];
                            }
                            FastScanComputer::ScanBlockMulti<4>(
                                packed_ptrs, lut_j, binary_bytes_, out_ptrs
                            );
                        } else {
                            for (size_t bi = 0; bi < n_blks; ++bi) {
                                FastScanComputer::ScanBlock(
                                    packed_ptrs[bi],
                                    lut_j,
                                    binary_bytes_,
                                    dot_qo[bi],
                                    blk_counts[bi]
                                );
                            }
                        }

                        // Per-block correction + best update
                        for (size_t bi = 0; bi < n_blks; ++bi) {
                            FastScanComputer::RabitQCorrection(
                                dot_qo[bi],
                                c1[j],
                                c2[j],
                                c34[j],
                                qr_to_c_l2sqr[j],
                                sum_q_f32[bi],
                                or_c_l2sqr + blk_starts[bi],
                                dp_mult + blk_starts[bi],
                                dist_buf[bi],
                                blk_counts[bi]
                            );

                            for (size_t k = 0; k < blk_counts[bi]; ++k) {
                                if (dist_buf[bi][k] < best_dist[bi][k]) {
                                    best_dist[bi][k] = dist_buf[bi][k];
                                    best_idx[bi][k] = static_cast<uint32_t>(j);
                                }
                            }
                        }
                    }

                    for (size_t bi = 0; bi < n_blks; ++bi) {
                        for (size_t k = 0; k < blk_counts[bi]; ++k) {
                            out_distances[blk_starts[bi] + k] = best_dist[bi][k];
                            out_knn[blk_starts[bi] + k] = best_idx[bi][k];
                        }
                    }
                }
            });
        }
    }

    size_t CodeSize(size_t d) const override { return (d + 7) / 8 + sizeof(RaBitQFactors); }

    bool IsFitted() const override { return fitted_; }
    bool SupportsPruning() const override { return true; }
    bool NeedsPDXLayout() const override { return false; }
    RaBitQParams GetParams() const { return {centroid_.data(), binary_bytes_}; }

    /// Front checkpoint fraction, 8-aligned for bitplane layout.
    /// Note: vertical_d == d for RaBitQ (NeedsPDXLayout() is false).
    /// d ≤ 768: 25% front (= mid_d), eliminating the gap checkpoint whose
    ///          HorizontalMultiPlane would be ≤ 12 bytes — not worth the overhead.
    /// d > 768: 12.5% front, gap checkpoint between front and mid_d is active.
    uint32_t InitialPartialD(uint32_t vertical_d) const override {
        if (vertical_d <= RABITQ_MIDDLE_CHECKPOINT_D_THRESHOLD) {
            return (vertical_d / 3 + 7) & ~7u;
        }
        return (vertical_d / 8 + 7) & ~7u;
    }

    /// Always returns the fixed value — tuning has no effect on RaBitQ.
    uint32_t AlignPartialD(uint32_t /*partial_d*/, uint32_t vertical_d) const override {
        return InitialPartialD(vertical_d);
    }
    void UpdateCentroids(
        ParallelExecutor& executor,
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
        uint32_t* cluster_sizes,
        size_t n,
        size_t n_clusters,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("RQ::UpdateCentroids");
        assert(fitted_ && d == d_);
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(encoded_data);
        // One rank per centroid range; the executor runs every rank exactly once.
        const size_t nt = executor.NumWorkers();
        executor.ParallelFor(nt, [&](size_t rank_begin, size_t rank_end, size_t) {
            std::unique_ptr<float[]> decode_buf(new float[d]);
            for (size_t rank = rank_begin; rank < rank_end; ++rank) {
                const size_t c0 = (n_clusters * rank) / nt;
                const size_t c1 = (n_clusters * (rank + 1)) / nt;
                for (size_t i = 0; i < n; ++i) {
                    uint32_t ci = assignments[i];
                    if (ci >= c0 && ci < c1) {
                        RaBitQCodec::DecodeOne(
                            codes + i * code_size_,
                            decode_buf.get(),
                            d,
                            binary_bytes_,
                            centroid_.data()
                        );
                        cluster_sizes[ci] += 1;
                        float* acc = centroid_accumulators + ci * d;
                        SKM_VECTORIZE_LOOP
                        for (size_t j = 0; j < d; ++j) {
                            acc[j] += decode_buf[j];
                        }
                    }
                }
            }
        });
    }

    void CacheDataPartialNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        size_t /*d*/,
        uint32_t partial_d
    ) override {
        ComputeDataPartialNorms(executor, data, n, partial_d);
    }

    void ComputeDataPartialNorms(
        ParallelExecutor& executor,
        const quantized_t* data,
        size_t n,
        uint32_t partial_d
    ) const {
        SKM_PROFILE_SCOPE("RQ::CacheDataPartialNorms");
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(data);
        const size_t front_bytes = partial_d / 8;
        const size_t mid_bytes = d_ / 32; // d/4, byte-aligned
        cached_sum_q_front_.resize(n);
        cached_sum_q_mid_.resize(n);
        cached_partial_d_ = partial_d;
        pruning_partial_norms_dirty_ = true;

        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = codes + i * code_size_;
                uint32_t pc = 0;
                size_t b = 0;
                for (; b < std::min(front_bytes, mid_bytes); ++b) {
                    pc += static_cast<uint32_t>(SKM_POPCOUNT(code[b]));
                }
                if (front_bytes <= mid_bytes) {
                    cached_sum_q_front_[i] = pc;
                    for (; b < mid_bytes; ++b) {
                        pc += static_cast<uint32_t>(SKM_POPCOUNT(code[b]));
                    }
                    cached_sum_q_mid_[i] = pc;
                } else {
                    cached_sum_q_mid_[i] = pc;
                    for (; b < front_bytes; ++b) {
                        pc += static_cast<uint32_t>(SKM_POPCOUNT(code[b]));
                    }
                    cached_sum_q_front_[i] = pc;
                }
            }
        });
    }

    void CacheCentroidPartialNorms(
        ParallelExecutor& /*executor*/,
        const quantized_t* /*centroids*/,
        size_t /*n*/,
        size_t /*d*/,
        uint32_t /*partial_d*/
    ) override {
        // No-op: centroid-side partial values are computed inside
        // FindNearestNeighborWithPruning during LUT building.
    }

    void FindNearestNeighborWithPruning(
        ParallelExecutor& executor,
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<Quantization::rabitq>& /*pdx_centroids*/,
        uint32_t partial_d,
        size_t* out_not_pruned_counts,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("RQ::FindNearestNeighborWithPruning");
        assert(fitted_);
        (void) y;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);

        // Ensure data-side caches
        EnsureCodeFactorsCache(executor, x_codes, n_x);
        EnsureTransposedBlocksCache(executor, x_codes, n_x);
        if (cached_sum_q_front_.size() != n_x || cached_partial_d_ != partial_d) {
            ComputeDataPartialNorms(executor, x, n_x, partial_d);
        }
        EnsurePartialNormsCache(executor, x_float, n_x, d, partial_d);

        const uint32_t* sum_q = cached_sum_q_.data();
        const float* or_c_l2sqr = cached_or_c_l2sqr_.data();
        const float* dp_mult = cached_dp_mult_.data();
        const uint32_t* sum_q_front = cached_sum_q_front_.data();
        const float* or_c_l2sqr_front = cached_or_c_l2sqr_front_.data();
        const uint32_t* sum_q_mid = cached_sum_q_mid_.data();
        const float* or_c_l2sqr_mid = cached_or_c_l2sqr_mid_.data();

        // Quantize centroids and build LUTs with partial bounds
        const size_t front_bytes = partial_d / 8;
        const size_t front_d = front_bytes * 8;
        const size_t mid_bytes = d / 32; // d/4, byte-aligned
        const size_t mid_d = mid_bytes * 8;
        const bool use_mid_checkpoint = (front_bytes < mid_bytes) && (mid_bytes < binary_bytes_);
        const size_t gap_bytes = use_mid_checkpoint ? (mid_bytes - front_bytes) : 0;
        const size_t phase3_start = use_mid_checkpoint ? mid_bytes : front_bytes;
        const size_t phase3_bytes = binary_bytes_ - phase3_start;
        const size_t n_sub = 2 * binary_bytes_;

        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<float> qr_to_c_l2sqr_front(n_y), c34_front(n_y);
        std::vector<float> qr_to_c_l2sqr_mid(n_y), c34_mid(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        const size_t rest_bytes = phase3_bytes;
        // Chunk-interleaved layout: for each 16B data chunk, all qb bitplanes
        // are stored contiguously (qb*16 bytes). Regions padded to 16B multiple.
        const size_t gap_chunks = (gap_bytes + 15) / 16;
        const size_t rest_chunks = (rest_bytes + 15) / 16;
        const size_t centroid_stride = (gap_chunks + rest_chunks) * qb_ * 16;
        std::vector<uint8_t> centroid_planes(n_y * centroid_stride, 0);
        QuantizeCentroidsAndBuildLUTsWithBounds(
            executor,
            y_float,
            n_y,
            d,
            front_d,
            mid_d,
            all_luts.data(),
            c1.data(),
            c2.data(),
            c34.data(),
            qr_to_c_l2sqr.data(),
            qr_to_c_l2sqr_front.data(),
            c34_front.data(),
            qr_to_c_l2sqr_mid.data(),
            c34_mid.data(),
            centroid_planes.data()
        );

        // ADSampling ratios
        const float adsampling_ratio_front =
            ComputeADSamplingRatio(front_d, d, PRUNER_INITIAL_THRESHOLD);
        const float adsampling_ratio_mid =
            use_mid_checkpoint ? ComputeADSamplingRatio(mid_d, d, PRUNER_INITIAL_THRESHOLD) : 1.0f;

        const size_t lut_stride = n_sub * 16;
        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        const size_t n_blocks = cached_n_blocks_;

        using b8_computer = DistanceComputer<DistanceFunction::l2, Quantization::rabitq>;

        {
            constexpr size_t kSuperBlock = 4;
            constexpr size_t kBS = FastScanComputer::kBlockSize;
            const size_t n_groups = (n_blocks + kSuperBlock - 1) / kSuperBlock;

            executor.ParallelFor(n_groups, [&](size_t group_begin, size_t group_end, size_t) {
            for (size_t group = group_begin; group < group_end; ++group) {
                const size_t blk_base = group * kSuperBlock;
                const size_t n_blks = std::min(kSuperBlock, n_blocks - blk_base);

                const uint8_t* packed_ptrs[kSuperBlock];
                for (size_t bi = 0; bi < n_blks; ++bi) {
                    packed_ptrs[bi] = cached_transposed_.get() + (blk_base + bi) * block_bytes;
                }

                // All partial dots for all blocks in super-block
                std::unique_ptr<uint16_t[]> all_partial_dots(new uint16_t[n_blks * n_y * kBS]);

                // Pass 1a: Multi-block FastScan all centroids
                {
                    if (n_blks == kSuperBlock) {
                        for (size_t j = 0; j < n_y; ++j) {
                            uint16_t* out_ptrs[kSuperBlock];
                            for (size_t bi = 0; bi < kSuperBlock; ++bi) {
                                out_ptrs[bi] = all_partial_dots.get() + (bi * n_y + j) * kBS;
                            }
                            FastScanComputer::ScanBlockMulti<4>(
                                packed_ptrs, all_luts.data() + j * lut_stride, front_bytes, out_ptrs
                            );
                        }
                    } else {
                        for (size_t j = 0; j < n_y; ++j) {
                            for (size_t bi = 0; bi < n_blks; ++bi) {
                                const size_t blk = blk_base + bi;
                                const size_t blk_count = std::min(kBS, n_x - blk * kBS);
                                FastScanComputer::ScanBlock(
                                    packed_ptrs[bi],
                                    all_luts.data() + j * lut_stride,
                                    front_bytes,
                                    all_partial_dots.get() + (bi * n_y + j) * kBS,
                                    blk_count
                                );
                            }
                        }
                    }
                }

                //  Per-block processing: fused correction + checkpoint pipeline
                for (size_t bi = 0; bi < n_blks; ++bi) {
                    const size_t blk = blk_base + bi;
                    const size_t blk_start = blk * kBS;
                    const size_t blk_count = std::min(kBS, n_x - blk_start);

                    float best_dist[kBS];
                    uint32_t best_idx[kBS];

                    for (size_t k = 0; k < blk_count; ++k) {
                        const size_t i = blk_start + k;
                        const uint32_t prev_j = out_knn[i];
                        best_idx[k] = prev_j;
                        best_dist[k] = ComputeFullDistanceViaLUT(
                            x_codes + i * code_size_,
                            all_luts.data() + prev_j * lut_stride,
                            c1[prev_j],
                            c2[prev_j],
                            c34[prev_j],
                            qr_to_c_l2sqr[prev_j],
                            sum_q[i],
                            or_c_l2sqr[i],
                            dp_mult[i]
                        );
                        out_not_pruned_counts[i] = 0;
                    }

                    // Precompute per-block buffers (reused across all centroids)
                    float threshold_buf[kBS];
                    float sum_q_front_f32[kBS];
                    float sum_q_mid_f32[kBS];
                    float sum_q_f32[kBS];
                    float neg2_dp_buf[kBS];
                    float dp_sum_q_front_buf[kBS];
                    uint32_t accumulated_dots[kBS];
                    uint32_t local_survivors[kBS];

                    for (size_t k = 0; k < blk_count; ++k) {
                        sum_q_front_f32[k] = static_cast<float>(sum_q_front[blk_start + k]);
                        sum_q_mid_f32[k] = static_cast<float>(sum_q_mid[blk_start + k]);
                        sum_q_f32[k] = static_cast<float>(sum_q[blk_start + k]);
                        threshold_buf[k] = best_dist[k] * adsampling_ratio_front;
                        neg2_dp_buf[k] = -2.0f * dp_mult[blk_start + k];
                        dp_sum_q_front_buf[k] = dp_mult[blk_start + k] * sum_q_front_f32[k];
                    }

                    //  Fused loop: for each centroid, checkpoint1 → checkpoint2 → phase3
                    for (size_t j = 0; j < n_y; ++j) {
                        const uint16_t* partial_dot_qo =
                            all_partial_dots.get() + (bi * n_y + j) * kBS;

                        // Checkpoint 1: fused front correction + survivor compaction
                        size_t n_survivors = 0;
                        FastScanComputer::RabitQCorrectionAndCompact(
                            partial_dot_qo,
                            c1[j],
                            c34_front[j],
                            qr_to_c_l2sqr_front[j],
                            -2.0f * c2[j],
                            or_c_l2sqr_front + blk_start,
                            neg2_dp_buf,
                            dp_sum_q_front_buf,
                            threshold_buf,
                            local_survivors,
                            n_survivors,
                            blk_count
                        );

                        if (n_survivors == 0)
                            continue;

                        for (size_t si = 0; si < n_survivors; ++si) {
                            out_not_pruned_counts[blk_start + local_survivors[si]]++;
                        }

                        // Initialize accumulated dots only for survivors
                        for (size_t si = 0; si < n_survivors; ++si) {
                            const uint32_t k = local_survivors[si];
                            accumulated_dots[k] = static_cast<uint32_t>(partial_dot_qo[k]);
                        }

                        size_t n_phase3 = n_survivors;

                        // Checkpoint 2: extend to mid_d dims (sparse)
                        if (use_mid_checkpoint) {
                            for (size_t si = 0; si < n_survivors; ++si) {
                                const size_t k = local_survivors[si];
                                const size_t i = blk_start + k;
                                accumulated_dots[k] += b8_computer::HorizontalMultiPlane(
                                    x_codes + i * code_size_ + front_bytes,
                                    centroid_planes.data() + j * centroid_stride,
                                    gap_bytes,
                                    qb_
                                );
                            }

                            // Fused mid correction + filter (scalar, ~1 survivor)
                            size_t write = 0;
                            for (size_t si = 0; si < n_survivors; ++si) {
                                const uint32_t k = local_survivors[si];
                                const float dot_f = static_cast<float>(accumulated_dots[k]);
                                const float fdt =
                                    c1[j] * dot_f + c2[j] * sum_q_mid_f32[k] - c34_mid[j];
                                const float dist = or_c_l2sqr_mid[blk_start + k] +
                                                   qr_to_c_l2sqr_mid[j] -
                                                   2.0f * dp_mult[blk_start + k] * fdt;
                                if (dist <= best_dist[k] * adsampling_ratio_mid) {
                                    local_survivors[write++] = k;
                                }
                            }
                            n_phase3 = write;
                        }

                        // Phase 3: remaining popcount for survivors
                        for (size_t si = 0; si < n_phase3; ++si) {
                            const size_t k = local_survivors[si];
                            const size_t i = blk_start + k;
                            accumulated_dots[k] += b8_computer::HorizontalMultiPlane(
                                x_codes + i * code_size_ + phase3_start,
                                centroid_planes.data() + j * centroid_stride +
                                    gap_chunks * qb_ * 16,
                                rest_bytes,
                                qb_
                            );
                        }

                        // Fused final correction + best update (scalar, ~1 survivor)
                        for (size_t si = 0; si < n_phase3; ++si) {
                            const uint32_t k = local_survivors[si];
                            const float dot_f = static_cast<float>(accumulated_dots[k]);
                            const float fdt = c1[j] * dot_f + c2[j] * sum_q_f32[k] - c34[j];
                            const float dist = or_c_l2sqr[blk_start + k] + qr_to_c_l2sqr[j] -
                                               2.0f * dp_mult[blk_start + k] * fdt;
                            if (dist < best_dist[k]) {
                                best_dist[k] = dist;
                                best_idx[k] = static_cast<uint32_t>(j);
                                threshold_buf[k] = best_dist[k] * adsampling_ratio_front;
                            }
                        }
                    } // centroid loop

                    for (size_t k = 0; k < blk_count; ++k) {
                        out_distances[blk_start + k] = best_dist[k];
                        out_knn[blk_start + k] = best_idx[k];
                    }
                } // per-block loop
            } // group loop
            });
        }
    }

  private:
    /// Extract per-code metadata: popcount, or_c_l2sqr, dp_multiplier.
    void PrecomputeCodeFactors(
        ParallelExecutor& executor,
        const uint8_t* codes,
        size_t n,
        uint32_t* sum_q,
        float* or_c_l2sqr,
        float* dp_mult
    ) const {
        SKM_PROFILE_SCOPE("RQ::PrecomputeCodeFactors");
        executor.ParallelFor(n, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* code = codes + i * code_size_;

                // Popcount of the binary part
                uint32_t pc = 0;
                size_t b = 0;
                for (; b + 8 <= binary_bytes_; b += 8) {
                    uint64_t word;
                    std::memcpy(&word, code + b, 8);
                    pc += static_cast<uint32_t>(SKM_POPCOUNT64(word));
                }
                for (; b < binary_bytes_; ++b) {
                    pc += static_cast<uint32_t>(SKM_POPCOUNT(code[b]));
                }
                sum_q[i] = pc;

                RaBitQFactors fac;
                std::memcpy(&fac, code + binary_bytes_, sizeof(fac));
                or_c_l2sqr[i] = fac.or_minus_c_l2sqr;
                dp_mult[i] = fac.dp_multiplier;
            }
        });
    }

    /// SQ-quantize centroids to qb bits and build 16-entry LUTs for FastScan.
    ///
    /// Each centroid produces d SQ-quantized uint8 values.
    /// Sub-quantizer m corresponds to 4 consecutive dimensions (4m..4m+3).
    /// LUT[m][c] = sum of quantized values at dimensions where bit k is set in c.
    ///
    /// Layout: lut[j * n_sub * 16 + m * 16 + c] for centroid j, sub-quantizer m, code c.
    /// Sub-quantizers are ordered: byte 0 low nibble, byte 0 high nibble,
    ///                             byte 1 low nibble, byte 1 high nibble, ...
    void QuantizeCentroidsAndBuildLUTs(
        ParallelExecutor& executor,
        const float* y_float,
        size_t n_y,
        size_t d,
        uint8_t* all_luts,
        float* c1,
        float* c2,
        float* c34,
        float* qr_to_c_l2sqr
    ) const {
        SKM_PROFILE_SCOPE("RQ::QuantizeCentroidsAndBuildLUTs");
        QuantizeCentroidsAndBuildLUTsWithBounds(
            executor,
            y_float,
            n_y,
            d,
            d,
            d,
            all_luts,
            c1,
            c2,
            c34,
            qr_to_c_l2sqr,
            qr_to_c_l2sqr,
            c34,
            qr_to_c_l2sqr,
            c34,
            nullptr
        );
    }

    /// Nibble-split transpose with kPerm0 interleaving for fast SIMD scanning.
    ///
    /// For each byte position b, output layout (32 bytes):
    ///   out[0..15]:  lo nibbles, kPerm0-interleaved, 2 vectors per byte
    ///   out[16..31]: hi nibbles, kPerm0-interleaved, 2 vectors per byte
    ///
    /// This allows a single 256-bit LUT load (lo_LUT|hi_LUT) and
    /// vpshufb per-lane to match codes to correct LUTs without broadcasts.
    static constexpr int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    void TransposeBlock(const uint8_t* codes, size_t blk_start, size_t blk_count, uint8_t* packed)
        const {
        std::memset(packed, 0, binary_bytes_ * FastScanComputer::kBlockSize);

        for (size_t b = 0; b < binary_bytes_; ++b) {
            uint8_t* out = packed + b * FastScanComputer::kBlockSize;

            // Gather byte b from each of the 32 vectors
            uint8_t col[32] = {0};
            for (size_t k = 0; k < blk_count; ++k) {
                col[k] = codes[(blk_start + k) * code_size_ + b];
            }

            for (int j = 0; j < 16; ++j) {
                const int vA = kPerm0[j];
                const int vB = kPerm0[j] + 16;
                out[j] = (col[vA] & 0x0F) | ((col[vB] & 0x0F) << 4);
                out[j + 16] = (col[vA] >> 4) | ((col[vB] >> 4) << 4);
            }
        }
    }

    /// Cache PrecomputeCodeFactors results (depend only on x, not centroids).
    void EnsureCodeFactorsCache(ParallelExecutor& executor, const uint8_t* x_codes, size_t n_x)
        const {
        SKM_PROFILE_SCOPE("RQ::EnsureCodeFactorsCache");
        if (cached_x_ptr_ == x_codes && cached_n_x_ == n_x)
            return;

        // Data changed → transposed blocks cache is also stale.
        // Reset so EnsureTransposedBlocksCache doesn't skip recomputation
        // after we update cached_x_ptr_ below.
        cached_n_blocks_ = 0;

        cached_sum_q_.resize(n_x);
        cached_or_c_l2sqr_.resize(n_x);
        cached_dp_mult_.resize(n_x);
        PrecomputeCodeFactors(
            executor,
            x_codes,
            n_x,
            cached_sum_q_.data(),
            cached_or_c_l2sqr_.data(),
            cached_dp_mult_.data()
        );
        cached_x_ptr_ = x_codes;
        cached_n_x_ = n_x;
    }

    int qb_ = RABITQ_SQ_BITS;
    size_t d_ = 0;
    size_t binary_bytes_ = 0;
    size_t code_size_ = 0;
    std::vector<float> centroid_;
    bool fitted_ = false;

    // Cached per-data-point factors (reused across iterations)
    mutable const uint8_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<uint32_t> cached_sum_q_;
    mutable std::vector<float> cached_or_c_l2sqr_;
    mutable std::vector<float> cached_dp_mult_;

    // Cached transposed blocks (reused across iterations)
    mutable std::unique_ptr<uint8_t[]> cached_transposed_;
    mutable size_t cached_n_blocks_ = 0;

    /// Transpose all data blocks once and cache for reuse across iterations.
    void EnsureTransposedBlocksCache(ParallelExecutor& executor, const uint8_t* x_codes, size_t n_x)
        const {
        SKM_PROFILE_SCOPE("RQ::EnsureTransposedBlocksCache");
        // Piggyback on the same pointer check as code factors
        if (cached_n_blocks_ > 0 && cached_x_ptr_ == x_codes && cached_n_x_ == n_x)
            return;

        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        cached_n_blocks_ = (n_x + FastScanComputer::kBlockSize - 1) / FastScanComputer::kBlockSize;
        cached_transposed_.reset(new uint8_t[cached_n_blocks_ * block_bytes]);

        executor.ParallelFor(cached_n_blocks_, [&](size_t blk_begin, size_t blk_end, size_t) {
            for (size_t blk = blk_begin; blk < blk_end; ++blk) {
                const size_t blk_start = blk * FastScanComputer::kBlockSize;
                const size_t blk_count = std::min(FastScanComputer::kBlockSize, n_x - blk_start);
                TransposeBlock(
                    x_codes, blk_start, blk_count, cached_transposed_.get() + blk * block_bytes
                );
            }
        });
    }

    /// Lazily compute or_c_l2sqr_front[i] and or_c_l2sqr_mid[i] from float data.
    void EnsurePartialNormsCache(
        ParallelExecutor& executor,
        const float* x_float,
        size_t n_x,
        size_t d,
        uint32_t partial_d
    ) const {
        if (!pruning_partial_norms_dirty_ && cached_pruning_partial_d_ == partial_d)
            return;
        SKM_PROFILE_SCOPE("RQ::EnsurePartialNormsCache");

        const size_t front_d = (static_cast<size_t>(partial_d) / 8) * 8;
        const size_t mid_d = (d / 32) * 8; // d/4, byte-aligned
        cached_or_c_l2sqr_front_.resize(n_x);
        cached_or_c_l2sqr_mid_.resize(n_x);

        executor.ParallelFor(n_x, [&](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float* xi = x_float + i * d;
                float sum_front = 0, sum_mid = 0;
                const size_t min_fm = std::min(front_d, mid_d);
                const size_t max_fm = std::max(front_d, mid_d);
                size_t dim = 0;
                for (; dim < min_fm; ++dim) {
                    const float diff = xi[dim] - centroid_[dim];
                    sum_front += diff * diff;
                }
                sum_mid = sum_front;
                if (front_d <= mid_d) {
                    // front_d < mid_d: continue accumulating into mid
                    for (; dim < max_fm; ++dim) {
                        const float diff = xi[dim] - centroid_[dim];
                        sum_mid += diff * diff;
                    }
                } else {
                    // front_d > mid_d: continue accumulating into front
                    for (; dim < max_fm; ++dim) {
                        const float diff = xi[dim] - centroid_[dim];
                        sum_front += diff * diff;
                    }
                }
                cached_or_c_l2sqr_front_[i] = sum_front;
                cached_or_c_l2sqr_mid_[i] = sum_mid;
            }
        });

        cached_pruning_partial_d_ = partial_d;
        pruning_partial_norms_dirty_ = false;
    }

    /// Extended LUT builder that also outputs partial centroid norms at front_d and mid_d.
    void QuantizeCentroidsAndBuildLUTsWithBounds(
        ParallelExecutor& executor,
        const float* y_float,
        size_t n_y,
        size_t d,
        size_t front_d,
        size_t mid_d,
        uint8_t* all_luts,
        float* c1,
        float* c2,
        float* c34,
        float* qr_to_c_l2sqr,
        float* qr_to_c_l2sqr_front,
        float* c34_front,
        float* qr_to_c_l2sqr_mid,
        float* c34_mid,
        uint8_t* centroid_planes
    ) const {
        SKM_PROFILE_SCOPE("RQ::QuantizeCentroidsAndBuildLUTsWithBounds");
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float max_val = static_cast<float>((1 << qb_) - 1);
        const size_t n_sub = 2 * binary_bytes_;
        const size_t front_d_clamped = std::min(front_d, d);
        const size_t mid_d_clamped = std::min(mid_d, d);

        executor.ParallelFor(n_y, [&](size_t j_begin, size_t j_end, size_t) {
        for (size_t j = j_begin; j < j_end; ++j) {
            std::vector<float> rotated(d);
            float v_min = std::numeric_limits<float>::max();
            float v_max = std::numeric_limits<float>::lowest();
            float norm_sq = 0;
            float norm_sq_front = 0;
            float norm_sq_mid = 0;

            for (size_t dim = 0; dim < d; ++dim) {
                rotated[dim] = y_float[j * d + dim] - centroid_[dim];
                v_min = std::min(v_min, rotated[dim]);
                v_max = std::max(v_max, rotated[dim]);
                const float r2 = rotated[dim] * rotated[dim];
                norm_sq += r2;
                if (dim < front_d_clamped)
                    norm_sq_front += r2;
                if (dim < mid_d_clamped)
                    norm_sq_mid += r2;
            }
            qr_to_c_l2sqr[j] = norm_sq;
            qr_to_c_l2sqr_front[j] = norm_sq_front;
            qr_to_c_l2sqr_mid[j] = norm_sq_mid;

            float delta = (v_max - v_min) / max_val;
            if (delta < std::numeric_limits<float>::epsilon())
                delta = 1.0f;
            const float inv_delta = 1.0f / delta;
            float sum_qq = 0;
            float sum_qq_front = 0;
            float sum_qq_mid = 0;

            std::vector<uint8_t> quantized(d);
            for (size_t dim = 0; dim < d; ++dim) {
                int v = static_cast<int>(std::lround((rotated[dim] - v_min) * inv_delta));
                v = std::max(0, std::min(v, static_cast<int>(max_val)));
                quantized[dim] = static_cast<uint8_t>(v);
                const float fv = static_cast<float>(v);
                sum_qq += fv;
                if (dim < front_d_clamped)
                    sum_qq_front += fv;
                if (dim < mid_d_clamped)
                    sum_qq_mid += fv;
            }

            c1[j] = 2.0f * delta * inv_sqrt_d;
            c2[j] = 2.0f * v_min * inv_sqrt_d;
            c34[j] = inv_sqrt_d * (delta * sum_qq + static_cast<float>(d) * v_min);
            c34_front[j] =
                inv_sqrt_d * (delta * sum_qq_front + static_cast<float>(front_d_clamped) * v_min);
            c34_mid[j] =
                inv_sqrt_d * (delta * sum_qq_mid + static_cast<float>(mid_d_clamped) * v_min);

            // Build LUTs
            uint8_t* lut_j = all_luts + j * n_sub * 16;
            for (size_t b = 0; b < binary_bytes_; ++b) {
                uint8_t* lut_lo = lut_j + (2 * b) * 16;
                uint8_t* lut_hi = lut_j + (2 * b + 1) * 16;
                uint8_t sq[8] = {0};
                for (int k = 0; k < 8 && (8 * b + k) < d; ++k) {
                    sq[k] = quantized[8 * b + k];
                }
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1)
                        val += sq[0];
                    if (c & 2)
                        val += sq[1];
                    if (c & 4)
                        val += sq[2];
                    if (c & 8)
                        val += sq[3];
                    lut_lo[c] = val;
                }
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1)
                        val += sq[4];
                    if (c & 2)
                        val += sq[5];
                    if (c & 4)
                        val += sq[6];
                    if (c & 8)
                        val += sq[7];
                    lut_hi[c] = val;
                }
            }

            // Bit-transpose SQ values into chunk-interleaved bitplanes.
            // Layout: for each 16-byte data chunk, all qb bitplanes are contiguous.
            // [chunk0_bp0_16B][chunk0_bp1_16B]...[chunk0_bpN_16B][chunk1_bp0_16B]...
            const size_t front_bytes_l = front_d_clamped / 8;
            const size_t mid_bytes_l = mid_d_clamped / 8;
            const bool has_gap = (front_bytes_l < mid_bytes_l) && (mid_bytes_l < binary_bytes_);
            const size_t gap_bytes_l = has_gap ? (mid_bytes_l - front_bytes_l) : 0;
            const size_t phase3_start_l = has_gap ? mid_bytes_l : front_bytes_l;
            const size_t rest_bytes_l = binary_bytes_ - phase3_start_l;
            const size_t gap_chunks_l = (gap_bytes_l + 15) / 16;
            const size_t rest_chunks_l = (rest_bytes_l + 15) / 16;
            const size_t cstride = (gap_chunks_l + rest_chunks_l) * qb_ * 16;
            uint8_t* cent_base = centroid_planes + j * cstride;

            // Gap region: dims [front_d, mid_d)
            for (size_t dim = front_d_clamped; dim < mid_d_clamped; ++dim) {
                size_t local_byte = (dim - front_d_clamped) / 8;
                uint8_t bit = static_cast<uint8_t>(1 << (dim % 8));
                size_t chunk = local_byte / 16;
                size_t byte_in_chunk = local_byte % 16;
                for (int b = 0; b < qb_; ++b) {
                    if ((quantized[dim] >> b) & 1)
                        cent_base[chunk * qb_ * 16 + static_cast<size_t>(b) * 16 + byte_in_chunk] |=
                            bit;
                }
            }

            // Rest region: dims [phase3_start_d, d)
            const size_t phase3_start_d = phase3_start_l * 8;
            uint8_t* rest_base = cent_base + gap_chunks_l * qb_ * 16;
            for (size_t dim = phase3_start_d; dim < d; ++dim) {
                size_t local_byte = (dim - phase3_start_d) / 8;
                uint8_t bit = static_cast<uint8_t>(1 << (dim % 8));
                size_t chunk = local_byte / 16;
                size_t byte_in_chunk = local_byte % 16;
                for (int b = 0; b < qb_; ++b) {
                    if ((quantized[dim] >> b) & 1)
                        rest_base[chunk * qb_ * 16 + static_cast<size_t>(b) * 16 + byte_in_chunk] |=
                            bit;
                }
            }
        }
        });
    }

    /// Compute full RaBitQ distance for a single (data, centroid) pair via LUT.
    static float ComputeFullDistanceViaLUT(
        const uint8_t* data_code,
        const uint8_t* lut_j,
        float c1j,
        float c2j,
        float c34j,
        float qr_j,
        uint32_t sum_q_i,
        float or_c_l2sqr_i,
        float dp_mult_i,
        size_t binary_bytes
    ) {
        uint16_t dot_qo = 0;
        for (size_t b = 0; b < binary_bytes; ++b) {
            const uint8_t byte = data_code[b];
            dot_qo += static_cast<uint16_t>(lut_j[(2 * b) * 16 + (byte & 0x0F)]) +
                      static_cast<uint16_t>(lut_j[(2 * b + 1) * 16 + (byte >> 4)]);
        }
        const float final_dot =
            c1j * static_cast<float>(dot_qo) + c2j * static_cast<float>(sum_q_i) - c34j;
        return or_c_l2sqr_i + qr_j - 2.0f * dp_mult_i * final_dot;
    }

    /// Overload that uses the instance's binary_bytes_.
    float ComputeFullDistanceViaLUT(
        const uint8_t* data_code,
        const uint8_t* lut_j,
        float c1j,
        float c2j,
        float c34j,
        float qr_j,
        uint32_t sum_q_i,
        float or_c_l2sqr_i,
        float dp_mult_i
    ) const {
        return ComputeFullDistanceViaLUT(
            data_code, lut_j, c1j, c2j, c34j, qr_j, sum_q_i, or_c_l2sqr_i, dp_mult_i, binary_bytes_
        );
    }

    // Pruning caches — checkpoint 1 (front, at partial_d)
    mutable std::vector<uint32_t> cached_sum_q_front_;   // [n_x] popcount of front bytes
    mutable std::vector<float> cached_or_c_l2sqr_front_; // [n_x] partial norm over front dims
    mutable uint32_t cached_partial_d_ = 0;
    mutable uint32_t cached_pruning_partial_d_ = 0;
    mutable bool pruning_partial_norms_dirty_ = true;

    // Pruning caches — checkpoint 2 (mid, at d/4)
    mutable std::vector<uint32_t> cached_sum_q_mid_;   // [n_x] popcount of mid bytes
    mutable std::vector<float> cached_or_c_l2sqr_mid_; // [n_x] partial norm over mid dims
};

} // namespace skmeans
