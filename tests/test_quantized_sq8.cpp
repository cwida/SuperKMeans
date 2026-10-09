#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq8.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {} // namespace

//  SQ8Quantizer unit tests

class SQ8QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128;
    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data)
            v = dist(rng);
    }
};

TEST_F(SQ8QuantizerTest, FitEncodeDecode_Roundtrip) {
    SQ8Quantizer quantizer;
    EXPECT_FALSE(quantizer.IsFitted());
    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());
    EXPECT_EQ(quantizer.CodeSize(d), d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    const auto& params = quantizer.GetParams();
    float max_err = params.inv_quantization_scale;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-5f) << "at index " << i;
    }
}

TEST(SQ8StateParams, ExposedParamsDecodeTheExposedCodes) {
    const size_t n = 3000, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 3;
    config.seed = 42;
    config.n_threads = 1;
    config.sampling_fraction = 1.0f;
    config.max_points_per_cluster = 99999;
    config.suppress_warnings = true;

    auto kmeans = SuperKMeansSQ8(n_clusters, d, config);
    std::vector<float> rotated(data);
    kmeans.TrainInPlace(rotated.data(), n);

    const auto* sq8 = dynamic_cast<const SQ8Quantizer*>(kmeans.GetQuantizer());
    ASSERT_NE(sq8, nullptr);
    const uint8_t* codes = kmeans.GetQuantizedData();
    ASSERT_NE(codes, nullptr);
    ASSERT_EQ(kmeans.GetState().code_size, d);

    const auto& params = sq8->GetParams();
    const float tolerance = params.inv_quantization_scale + 1e-5f;
    for (size_t i = 0; i < n; i += 293) {
        for (size_t j = 0; j < d; ++j) {
            const float decoded =
                static_cast<float>(codes[i * d + j]) * params.inv_quantization_scale +
                params.quantization_base;
            ASSERT_NEAR(rotated[i * d + j], decoded, tolerance) << "at " << i << "," << j;
        }
    }
}

TEST_F(SQ8QuantizerTest, ComputeNorms_ConsistentWithDistances) {
    SQ8Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded.data(), n, d, q_norms.data());

    const auto& params = quantizer.GetParams();
    float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;
    for (size_t i = 0; i < std::min(n, size_t{100}); ++i) {
        uint32_t sum_sq = 0;
        for (size_t j = 0; j < d; ++j) {
            uint32_t v = encoded[i * d + j];
            sum_sq += v * v;
        }
        float expected_norm = inv_scale_sq * static_cast<float>(sum_sq);
        EXPECT_FLOAT_EQ(q_norms[i], expected_norm) << "norm mismatch at vector " << i;
    }
}

TEST_F(SQ8QuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    SQ8Quantizer quantizer;
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
        queries,
        centroids,
        data.data() + n_centroids * d,
        data.data(),
        n_queries,
        n_centroids,
        d,
        query_norms,
        centroid_norms,
        knn.data(),
        distances.data(),
        tmp_buf.data()
    );

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
        EXPECT_EQ(knn[i], best_idx) << "query " << i;
    }
}

TEST_F(SQ8QuantizerTest, Encode_ClampsOutOfRange) {
    SQ8Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<float> extreme(d);
    for (size_t j = 0; j < d; ++j)
        extreme[j] = (j % 2 == 0) ? -1e6f : 1e6f;
    std::vector<uint8_t> codes(d);
    quantizer.Encode(extreme.data(), codes.data(), 1, d);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_EQ(codes[j], (j % 2 == 0) ? 0u : 255u) << "no clamp at dim " << j;
    }
}

//  FinalizeCentroids unit test

TEST(SQ8FinalizeCentroidsTest, RecoversClusterMeans) {
    SQ8Quantizer quantizer;
    const size_t d = 8;
    const size_t n_clusters = 3;
    const size_t n_vectors = 6;

    std::vector<float> fit_data(100 * d);
    for (size_t i = 0; i < fit_data.size(); ++i)
        fit_data[i] = static_cast<float>(i % 256) / 255.0f;
    quantizer.Fit(fit_data.data(), 100, d);

    std::vector<float> vectors(n_vectors * d);
    for (size_t j = 0; j < 2 * d; ++j)
        vectors[j] = 0.5f;
    for (size_t j = 2 * d; j < 5 * d; ++j)
        vectors[j] = 0.25f;
    for (size_t j = 5 * d; j < 6 * d; ++j)
        vectors[j] = 0.75f;

    std::vector<uint8_t> encoded(n_vectors * d);
    quantizer.Encode(vectors.data(), encoded.data(), n_vectors, d);

    std::vector<uint32_t> assignments = {0, 0, 1, 1, 1, 2};
    std::vector<float> centroid_buf(n_clusters * d, 0.0f);
    std::vector<uint32_t> cluster_sizes(n_clusters, 0);

    quantizer.ResetCentroidAccumulators(n_clusters, d);
    quantizer.UpdateCentroids(
        encoded.data(),
        assignments.data(),
        centroid_buf.data(),
        cluster_sizes.data(),
        n_vectors,
        n_clusters,
        d
    );
    quantizer.FinalizeCentroids(centroid_buf.data(), cluster_sizes.data(), n_clusters, d);

    EXPECT_EQ(cluster_sizes[0], 2u);
    EXPECT_EQ(cluster_sizes[1], 3u);
    EXPECT_EQ(cluster_sizes[2], 1u);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(centroid_buf[0 * d + j], 0.5f, 0.02f) << "cluster 0, dim " << j;
        EXPECT_NEAR(centroid_buf[1 * d + j], 0.25f, 0.02f) << "cluster 1, dim " << j;
        EXPECT_NEAR(centroid_buf[2 * d + j], 0.75f, 0.02f) << "cluster 2, dim " << j;
    }
}
