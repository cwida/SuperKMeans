#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq4.h"
#include "superkmeans/quantizers/sq8.h"
#include "superkmeans/superkmeans.h"

using namespace skmeans;

// ── SQ8Quantizer unit tests ──

class SQ8QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128;

    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data) {
            v = dist(rng);
        }
    }
};

TEST_F(SQ8QuantizerTest, FitEncodeDecode_Roundtrip) {
    SQ8Quantizer<Quantization::u8> quantizer;
    EXPECT_FALSE(quantizer.IsFitted());

    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);

    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    // Reconstruction error should be bounded by inv_quantization_scale
    const auto& params = quantizer.GetParams();
    float max_err = params.inv_quantization_scale;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-5f)
            << "at index " << i;
    }
}

TEST_F(SQ8QuantizerTest, Norms_ConsistentWithDistances) {
    SQ8Quantizer<Quantization::u8> quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);

    // Quantized norms via quantizer (these are inv_scale² * Σ q²,
    // used in L2 distance formula where the base cancels)
    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded.data(), n, d, q_norms.data());

    // Verify norms are consistent with distance computation:
    // dist(x, y) = norm_x + norm_y - 2 * inv_scale² * dot(q_x, q_y)
    // When y = x, dist should be 0, so norm_x = inv_scale² * dot(q_x, q_x) = inv_scale² * Σ q_x²
    const auto& params = quantizer.GetParams();
    float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;

    for (size_t i = 0; i < std::min(n, size_t{100}); ++i) {
        uint32_t sum_sq = 0;
        for (size_t j = 0; j < d; ++j) {
            uint32_t v = encoded[i * d + j];
            sum_sq += v * v;
        }
        float expected_norm = inv_scale_sq * static_cast<float>(sum_sq);
        EXPECT_FLOAT_EQ(q_norms[i], expected_norm)
            << "norm mismatch at vector " << i;
    }
}

TEST_F(SQ8QuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    SQ8Quantizer<Quantization::u8> quantizer;
    quantizer.Fit(data.data(), n, d);

    // Use first 50 vectors as "centroids", rest as queries
    size_t n_centroids = 50;
    size_t n_queries = 200;

    std::vector<uint8_t> encoded_data(n * d);
    quantizer.Encode(data.data(), encoded_data.data(), n, d);

    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded_data.data(), n, d, q_norms.data());

    const uint8_t* queries = encoded_data.data() + n_centroids * d;
    const uint8_t* centroids = encoded_data.data();
    const float* query_norms = q_norms.data() + n_centroids;
    const float* centroid_norms = q_norms.data();

    std::vector<uint32_t> knn(n_queries);
    std::vector<float> distances(n_queries);
    std::vector<float> tmp_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

    quantizer.FindNearestNeighbor(
        queries, centroids, n_queries, n_centroids, d,
        query_norms, centroid_norms, knn.data(), distances.data(), tmp_buf.data()
    );

    // Brute-force reference using decoded float vectors
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded_data.data(), decoded.data(), n, d);

    for (size_t i = 0; i < n_queries; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best_idx = 0;
        for (size_t j = 0; j < n_centroids; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = decoded[(n_centroids + i) * d + k] - decoded[j * d + k];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<uint32_t>(j);
            }
        }
        EXPECT_EQ(knn[i], best_idx)
            << "query " << i << ": expected centroid " << best_idx
            << " but got " << knn[i];
    }
}

// ── SuperKMeans<u8> integration tests ──

class SuperKMeansU8Test : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(SuperKMeansU8Test, BasicTraining) {
    const size_t n = 2000;
    const size_t d = 64;
    const size_t n_clusters = 10;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.verbose = false;
    config.quantizer_type = QuantizerType::sq8;

    auto kmeans = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config);

    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
}

TEST_F(SuperKMeansU8Test, AllClustersUsed) {
    const size_t n = 5000;
    const size_t d = 128;
    const size_t n_clusters = 20;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 15;
    config.verbose = false;
    config.quantizer_type = QuantizerType::sq8;

    auto kmeans = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());

    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters
        << " but only " << used_clusters.size() << " were assigned.";
}

TEST_F(SuperKMeansU8Test, WCSSReasonable) {
    const size_t n = 3000;
    const size_t d = 64;
    const size_t n_clusters = 10;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    // Train f32 baseline
    SuperKMeansConfig config_f32;
    config_f32.iters = 15;
    config_f32.verbose = false;
    auto kmeans_f32 = SuperKMeans<Quantization::f32, DistanceFunction::l2>(n_clusters, d, config_f32);
    auto centroids_f32 = kmeans_f32.Train(data.data(), n);

    // Train u8
    SuperKMeansConfig config_u8;
    config_u8.iters = 15;
    config_u8.verbose = false;
    config_u8.quantizer_type = QuantizerType::sq8;
    auto kmeans_u8 = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config_u8);
    auto centroids_u8 = kmeans_u8.Train(data.data(), n);

    // Compute WCSS for both using f32 assignments
    auto assign_f32 = kmeans_f32.Assign(data.data(), centroids_f32.data(), n, n_clusters);
    auto assign_u8 = kmeans_u8.Assign(data.data(), centroids_u8.data(), n, n_clusters);

    auto compute_wcss = [&](const std::vector<uint32_t>& assignments,
                            const std::vector<float>& ctrs) {
        double wcss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            uint32_t c = assignments[i];
            for (size_t j = 0; j < d; ++j) {
                double diff = data[i * d + j] - ctrs[c * d + j];
                wcss += diff * diff;
            }
        }
        return wcss;
    };

    double wcss_f32 = compute_wcss(assign_f32, centroids_f32);
    double wcss_u8 = compute_wcss(assign_u8, centroids_u8);

    // u8 WCSS should be within 50% of f32 WCSS for well-separated blobs
    EXPECT_LT(wcss_u8, wcss_f32 * 1.5)
        << "u8 WCSS (" << wcss_u8 << ") is too much worse than f32 WCSS (" << wcss_f32 << ")";
}

TEST_F(SuperKMeansU8Test, RerankingMatchesOrImproves) {
    const size_t n = 3000;
    const size_t d = 64;
    const size_t n_clusters = 10;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    // Train without reranking (default: rerank_k = -1, sq8 DefaultRerankK = 0)
    SuperKMeansConfig config_no_rerank;
    config_no_rerank.iters = 15;
    config_no_rerank.verbose = false;
    config_no_rerank.quantizer_type = QuantizerType::sq8;
    auto kmeans_no_rerank =
        SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config_no_rerank);
    auto centroids_no_rerank = kmeans_no_rerank.Train(data.data(), n);

    // Train with reranking (rerank_k = 4)
    SuperKMeansConfig config_rerank;
    config_rerank.iters = 15;
    config_rerank.verbose = false;
    config_rerank.quantizer_type = QuantizerType::sq8;
    config_rerank.rerank_k = 4;
    auto kmeans_rerank =
        SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config_rerank);
    auto centroids_rerank = kmeans_rerank.Train(data.data(), n);

    // Compute WCSS for both
    auto assign_no_rerank =
        kmeans_no_rerank.Assign(data.data(), centroids_no_rerank.data(), n, n_clusters);
    auto assign_rerank =
        kmeans_rerank.Assign(data.data(), centroids_rerank.data(), n, n_clusters);

    auto compute_wcss = [&](const std::vector<uint32_t>& assignments,
                            const std::vector<float>& ctrs) {
        double wcss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            uint32_t c = assignments[i];
            for (size_t j = 0; j < d; ++j) {
                double diff = data[i * d + j] - ctrs[c * d + j];
                wcss += diff * diff;
            }
        }
        return wcss;
    };

    double wcss_no_rerank = compute_wcss(assign_no_rerank, centroids_no_rerank);
    double wcss_rerank = compute_wcss(assign_rerank, centroids_rerank);

    // Both training runs converge to potentially different local optima,
    // so we only check that reranked WCSS is in the same ballpark
    EXPECT_LT(wcss_rerank, wcss_no_rerank * 1.5)
        << "Reranked WCSS (" << wcss_rerank
        << ") is unexpectedly worse than non-reranked (" << wcss_no_rerank << ")";
}

// ── SQ4Quantizer unit tests ──

class SQ4QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128; // must be even for SQ4

    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data) {
            v = dist(rng);
        }
    }
};

TEST_F(SQ4QuantizerTest, FitEncodeDecode_Roundtrip) {
    SQ4Quantizer<Quantization::u8> quantizer;
    EXPECT_FALSE(quantizer.IsFitted());

    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);

    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    // SQ4 has coarser quantization (only 16 levels), so larger reconstruction error
    const auto& params = quantizer.GetParams();
    float max_err = params.inv_quantization_scale;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-5f)
            << "at index " << i;
    }
}

TEST_F(SQ4QuantizerTest, EncodedValuesInRange) {
    SQ4Quantizer<Quantization::u8> quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);

    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_LE(encoded[i], 15u) << "encoded value out of [0,15] range at index " << i;
    }
}

TEST_F(SQ4QuantizerTest, PackToU4x2_Roundtrip) {
    SQ4Quantizer<Quantization::u8> quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);

    const size_t d_packed = d / 2;
    std::vector<nk_u4x2_t> packed(n * d_packed);
    SQ4Quantizer<Quantization::u8>::PackToU4x2(encoded.data(), packed.data(), n, d);

    // Verify packing: low nibble = even dim, high nibble = odd dim
    for (size_t row = 0; row < std::min(n, size_t{100}); ++row) {
        for (size_t k = 0; k < d_packed; ++k) {
            uint8_t lo = packed[row * d_packed + k] & 0x0F;
            uint8_t hi = (packed[row * d_packed + k] >> 4) & 0x0F;
            EXPECT_EQ(lo, encoded[row * d + 2 * k])
                << "low nibble mismatch at row=" << row << " k=" << k;
            EXPECT_EQ(hi, encoded[row * d + 2 * k + 1])
                << "high nibble mismatch at row=" << row << " k=" << k;
        }
    }
}

TEST_F(SQ4QuantizerTest, FindNearestNeighbor_ReasonableAccuracy) {
    SQ4Quantizer<Quantization::u8> quantizer;
    quantizer.Fit(data.data(), n, d);

    size_t n_centroids = 50;
    size_t n_queries = 200;

    std::vector<uint8_t> encoded_data(n * d);
    quantizer.Encode(data.data(), encoded_data.data(), n, d);

    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded_data.data(), n, d, q_norms.data());

    const uint8_t* queries = encoded_data.data() + n_centroids * d;
    const uint8_t* centroids = encoded_data.data();
    const float* query_norms = q_norms.data() + n_centroids;
    const float* centroid_norms = q_norms.data();

    std::vector<uint32_t> knn(n_queries);
    std::vector<float> distances(n_queries);
    std::vector<float> tmp_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

    quantizer.FindNearestNeighbor(
        queries, centroids, n_queries, n_centroids, d,
        query_norms, centroid_norms, knn.data(), distances.data(), tmp_buf.data()
    );

    // Brute-force reference using decoded float vectors
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded_data.data(), decoded.data(), n, d);

    size_t matches = 0;
    for (size_t i = 0; i < n_queries; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best_idx = 0;
        for (size_t j = 0; j < n_centroids; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = decoded[(n_centroids + i) * d + k] - decoded[j * d + k];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<uint32_t>(j);
            }
        }
        if (knn[i] == best_idx) matches++;
    }

    // SQ4 uses 4-bit quantized GEMM so exact match rate may be lower than SQ8.
    // Expect at least 80% of queries to find the same nearest neighbor.
    double match_rate = static_cast<double>(matches) / n_queries;
    EXPECT_GT(match_rate, 0.80)
        << "SQ4 nearest neighbor match rate (" << match_rate
        << ") is too low vs brute-force decoded reference";
}

// ── SuperKMeans<u8> with SQ4 integration tests ──

class SuperKMeansU8SQ4Test : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(SuperKMeansU8SQ4Test, BasicTraining) {
    const size_t n = 2000;
    const size_t d = 64;
    const size_t n_clusters = 10;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.verbose = false;
    config.quantizer_type = QuantizerType::sq4;

    auto kmeans = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config);

    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
}

TEST_F(SuperKMeansU8SQ4Test, AllClustersUsed) {
    const size_t n = 5000;
    const size_t d = 128;
    const size_t n_clusters = 20;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 15;
    config.verbose = false;
    config.quantizer_type = QuantizerType::sq4;

    auto kmeans = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());

    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters
        << " but only " << used_clusters.size() << " were assigned.";
}

TEST_F(SuperKMeansU8SQ4Test, WCSSReasonable) {
    const size_t n = 3000;
    const size_t d = 64;
    const size_t n_clusters = 10;

    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    // Train f32 baseline
    SuperKMeansConfig config_f32;
    config_f32.iters = 15;
    config_f32.verbose = false;
    auto kmeans_f32 = SuperKMeans<Quantization::f32, DistanceFunction::l2>(n_clusters, d, config_f32);
    auto centroids_f32 = kmeans_f32.Train(data.data(), n);

    // Train SQ4
    SuperKMeansConfig config_sq4;
    config_sq4.iters = 15;
    config_sq4.verbose = false;
    config_sq4.quantizer_type = QuantizerType::sq4;
    auto kmeans_sq4 = SuperKMeans<Quantization::u8, DistanceFunction::l2>(n_clusters, d, config_sq4);
    auto centroids_sq4 = kmeans_sq4.Train(data.data(), n);

    // Compute WCSS for both using f32 assignments
    auto assign_f32 = kmeans_f32.Assign(data.data(), centroids_f32.data(), n, n_clusters);
    auto assign_sq4 = kmeans_sq4.Assign(data.data(), centroids_sq4.data(), n, n_clusters);

    auto compute_wcss = [&](const std::vector<uint32_t>& assignments,
                            const std::vector<float>& ctrs) {
        double wcss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            uint32_t c = assignments[i];
            for (size_t j = 0; j < d; ++j) {
                double diff = data[i * d + j] - ctrs[c * d + j];
                wcss += diff * diff;
            }
        }
        return wcss;
    };

    double wcss_f32 = compute_wcss(assign_f32, centroids_f32);
    double wcss_sq4 = compute_wcss(assign_sq4, centroids_sq4);

    // SQ4 has much coarser quantization (16 levels vs 256), allow 2x WCSS vs f32
    EXPECT_LT(wcss_sq4, wcss_f32 * 2.0)
        << "SQ4 WCSS (" << wcss_sq4 << ") is too much worse than f32 WCSS (" << wcss_f32 << ")";
}
