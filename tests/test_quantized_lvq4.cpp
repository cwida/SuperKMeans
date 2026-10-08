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
#include "superkmeans/quantizers/lvq4.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {} // namespace

// LVQ4Quantizer unit tests

class LVQ4QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128;
    std::vector<float> data;
    float data_min = 0.0f, data_max = 0.0f;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        data_min = std::numeric_limits<float>::max();
        data_max = std::numeric_limits<float>::lowest();
        for (auto& v : data) {
            v = dist(rng);
            data_min = std::min(data_min, v);
            data_max = std::max(data_max, v);
        }
    }
};

TEST_F(LVQ4QuantizerTest, FitEncodeDecode_Roundtrip) {
    LVQ4Quantizer quantizer;
    EXPECT_FALSE(quantizer.IsFitted());
    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());

    const size_t cs = quantizer.CodeSize(d);
    EXPECT_EQ(cs, d / 2 + 8);

    std::vector<uint8_t> encoded(n * cs);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    // 4-bit per-vector quantization: error bounded by the per-vector step,
    // which is at most global_range / 15.
    const float max_err = (data_max - data_min) / 15.0f;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-4f) << "at index " << i;
    }
}

TEST_F(LVQ4QuantizerTest, OddDimensionality_Throws) {
    std::vector<float> tiny(63, 0.1f);
    EXPECT_THROW(LVQ4Quantizer().Fit(tiny.data(), 1, 63), std::invalid_argument);
    // Even dimensionality must not throw.
    EXPECT_NO_THROW(LVQ4Quantizer().Fit(tiny.data(), 1, 62));
}

TEST_F(LVQ4QuantizerTest, EncodedNibbles_InRange) {
    LVQ4Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);
    const size_t cs = quantizer.CodeSize(d);
    const size_t nibble_bytes = d / 2;

    std::vector<uint8_t> encoded(n * cs);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* code = encoded.data() + i * cs;
        for (size_t b = 0; b < nibble_bytes; ++b) {
            EXPECT_LE(code[b] & 0x0F, 15u);
            EXPECT_LE(code[b] >> 4, 15u);
        }
    }
}

TEST_F(LVQ4QuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    LVQ4Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);
    const size_t cs = quantizer.CodeSize(d);

    size_t n_centroids = 50;
    size_t n_queries = 200;

    std::vector<uint8_t> codes(n * cs);
    quantizer.Encode(data.data(), codes.data(), n, d);
    std::vector<float> norms(n);
    quantizer.ComputeNorms(codes.data(), n, d, norms.data());

    const uint8_t* queries = codes.data() + n_centroids * cs;
    const uint8_t* centroids = codes.data();

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
        norms.data() + n_centroids,
        norms.data(),
        knn.data(),
        distances.data(),
        tmp_buf.data()
    );

    // LVQ4's distance is exact L2 on the decoded vectors, so the returned
    // distance must match the brute-force minimum (up to float rounding).
    std::vector<float> decoded(n * d);
    quantizer.Decode(codes.data(), decoded.data(), n, d);
    for (size_t i = 0; i < n_queries; ++i) {
        float bf_min = std::numeric_limits<float>::max();
        for (size_t j = 0; j < n_centroids; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = decoded[(n_centroids + i) * d + k] - decoded[j * d + k];
                dist += diff * diff;
            }
            bf_min = std::min(bf_min, dist);
        }
        ASSERT_LT(knn[i], n_centroids);
        EXPECT_LE(distances[i], bf_min + std::max(1e-2f, bf_min * 1e-3f)) << "query " << i;
    }
}

TEST_F(LVQ4QuantizerTest, ComputeNorms_MatchDecodedNorm) {
    LVQ4Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    const size_t cs = quantizer.CodeSize(d);
    std::vector<uint8_t> codes(n * cs);
    quantizer.Encode(data.data(), codes.data(), n, d);

    std::vector<float> norms(n);
    quantizer.ComputeNorms(codes.data(), n, d, norms.data());

    std::vector<float> decoded(n * d);
    quantizer.Decode(codes.data(), decoded.data(), n, d);
    for (size_t i = 0; i < std::min(n, size_t{200}); ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double val = decoded[i * d + j];
            expected += val * val;
        }
        EXPECT_NEAR(
            norms[i], static_cast<float>(expected), static_cast<float>(expected) * 1e-3f + 1e-2f
        ) << "norm mismatch at vector "
          << i;
    }
}

// SuperKMeans<u8> integration tests (LVQ4)

class SuperKMeansLVQ4Test : public ::testing::Test {};

TEST_F(SuperKMeansLVQ4Test, SmallestEvenDimension) {
    const size_t n = 1000, d = 2, n_clusters = 8;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;

    auto kmeans = SuperKMeansLVQ4(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);
}

// LVQ4 has no global parameters, so the documented code layout is the contract:
//   [d/2 packed u4x2 bytes] [float scale: 4B] [float bias: 4B]
TEST(LVQ4CodeLayout, ScaleAndBiasSitAtTheDocumentedOffsets) {
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

    auto kmeans = SuperKMeansLVQ4(n_clusters, d, config);
    kmeans.Train(data.data(), n);

    const auto* quantizer = kmeans.GetQuantizer();
    ASSERT_NE(quantizer, nullptr);
    const uint8_t* codes = kmeans.GetQuantizedData();
    ASSERT_NE(codes, nullptr);

    const size_t nibble_bytes = d / 2;
    const size_t code_size = kmeans.GetState().code_size;
    ASSERT_EQ(code_size, nibble_bytes + 2 * sizeof(float));

    std::vector<float> decoded(n * d);
    quantizer->Decode(codes, decoded.data(), n, d);

    for (size_t i = 0; i < n; i += 293) {
        const uint8_t* code = codes + i * code_size;
        float scale = 0.0f;
        float bias = 0.0f;
        std::memcpy(&scale, code + nibble_bytes, sizeof(float));
        std::memcpy(&bias, code + nibble_bytes + sizeof(float), sizeof(float));
        for (size_t j = 0; j < d; ++j) {
            const uint8_t byte = code[j / 2];
            const uint8_t nibble = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            ASSERT_NEAR(decoded[i * d + j], scale * static_cast<float>(nibble) + bias, 1e-4f)
                << "at " << i << "," << j;
        }
    }
}
