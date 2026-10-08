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
#include "superkmeans/quantizers/rabitq.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

std::vector<float> ColumnMean(const std::vector<float>& data, size_t n, size_t d) {
    std::vector<double> acc(d, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            acc[j] += data[i * d + j];
    std::vector<float> mean(d);
    for (size_t j = 0; j < d; ++j)
        mean[j] = static_cast<float>(acc[j] / static_cast<double>(n));
    return mean;
}

// Well-separated clusters: queries are tight around distinct, far-apart centroids,
// so the true nearest centroid is unambiguous even for a 1-bit code.
struct SeparatedSet {
    std::vector<float> centroids; // n_c * d
    std::vector<float> queries;   // n_q * d
    std::vector<uint32_t> true_nn;
    size_t n_c, n_q, d;
};

SeparatedSet MakeSeparated(size_t d, size_t n_c, size_t per, uint32_t seed) {
    SeparatedSet s;
    s.d = d;
    s.n_c = n_c;
    s.n_q = n_c * per;
    std::mt19937 rng(seed);
    std::normal_distribution<float> cdist(0.0f, 20.0f);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    s.centroids.resize(n_c * d);
    for (auto& v : s.centroids)
        v = cdist(rng);
    s.queries.resize(s.n_q * d);
    s.true_nn.resize(s.n_q);
    for (size_t i = 0; i < s.n_q; ++i) {
        uint32_t c = static_cast<uint32_t>(i % n_c);
        s.true_nn[i] = c;
        for (size_t j = 0; j < d; ++j)
            s.queries[i * d + j] = s.centroids[c * d + j] + noise(rng);
    }
    return s;
}

double NearestNeighborRecall(size_t d, size_t n_c, size_t per, uint32_t seed) {
    SeparatedSet s = MakeSeparated(d, n_c, per, seed);
    RaBitQQuantizer q;
    q.Fit(s.queries.data(), s.n_q, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(s.n_q * cs);
    q.Encode(s.queries.data(), codes.data(), s.n_q, d);

    std::vector<uint32_t> knn(s.n_q);
    std::vector<float> dist(s.n_q);
    q.FindNearestNeighbor(
        codes.data(),
        nullptr,
        nullptr,
        s.centroids.data(),
        s.n_q,
        s.n_c,
        d,
        nullptr,
        nullptr,
        knn.data(),
        dist.data(),
        nullptr
    );
    size_t hits = 0;
    for (size_t i = 0; i < s.n_q; ++i)
        if (knn[i] == s.true_nn[i])
            hits++;
    return static_cast<double>(hits) / static_cast<double>(s.n_q);
}

} // namespace

// RaBitQQuantizer unit tests

class RaBitQQuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 2000;
    static constexpr size_t d = 64;
    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data)
            v = dist(rng);
    }
};

TEST_F(RaBitQQuantizerTest, InvalidDimensionality_Throws) {
    std::vector<float> buf(8192, 0.1f);
    // Not a multiple of 8.
    EXPECT_THROW(RaBitQQuantizer().Fit(buf.data(), 4, 60), std::invalid_argument);
    // Multiple of 8 but above RABITQ_MAX_DIMS (4096).
    EXPECT_THROW(RaBitQQuantizer().Fit(buf.data(), 1, RABITQ_MAX_DIMS + 8), std::invalid_argument);
    // Valid.
    EXPECT_NO_THROW(RaBitQQuantizer().Fit(buf.data(), 4, 64));
}

TEST_F(RaBitQQuantizerTest, CodeSize_Matches) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    EXPECT_EQ(q.CodeSize(d), (d + 7) / 8 + 8);
}

TEST_F(RaBitQQuantizerTest, ComputeNorms_MatchDistanceToCentroid) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(n * cs);
    q.Encode(data.data(), codes.data(), n, d);

    std::vector<float> norms(n);
    q.ComputeNorms(codes.data(), n, d, norms.data());

    std::vector<float> mean = ColumnMean(data, n, d);
    for (size_t i = 0; i < std::min(n, size_t{200}); ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double diff = static_cast<double>(data[i * d + j]) - mean[j];
            expected += diff * diff;
        }
        EXPECT_NEAR(
            norms[i], static_cast<float>(expected), static_cast<float>(expected) * 0.05f + 1e-1f
        ) << "vector "
          << i;
    }
}

TEST_F(RaBitQQuantizerTest, Decode_FinitePreservesDirection) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(n * cs);
    q.Encode(data.data(), codes.data(), n, d);
    std::vector<float> decoded(n * d);
    q.Decode(codes.data(), decoded.data(), n, d);

    std::vector<float> mean = ColumnMean(data, n, d);
    size_t positive = 0;
    for (size_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < d; ++j) {
            float v = decoded[i * d + j];
            ASSERT_TRUE(std::isfinite(v)) << "vector " << i << " dim " << j;
            dot += (static_cast<double>(data[i * d + j]) - mean[j]) *
                   (static_cast<double>(v) - mean[j]);
        }
        if (dot > 0.0)
            positive++;
    }
    EXPECT_GT(static_cast<double>(positive) / n, 0.9);
}

TEST_F(RaBitQQuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    EXPECT_GT(NearestNeighborRecall(64, 20, 30, 7), 0.9);
}

TEST_F(RaBitQQuantizerTest, FindNearestNeighbor_HighDimensionNoOverflow) {
    // d within the RABITQ_MAX_DIMS cap must still produce correct results
    // (guards the uint16 FastScan accumulator boundary).
    EXPECT_GT(NearestNeighborRecall(2048, 16, 20, 11), 0.9);
    EXPECT_GT(NearestNeighborRecall(RABITQ_MAX_DIMS, 16, 12, 13), 0.9);
}

TEST(RaBitQStateParams, ExposedCentroidIsTheRotatedDataMean) {
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

    auto kmeans = SuperKMeansRabitQ(n_clusters, d, config);
    std::vector<float> rotated(data);
    kmeans.TrainInPlace(rotated.data(), n);

    const auto* rabitq = dynamic_cast<const RaBitQQuantizer*>(kmeans.GetQuantizer());
    ASSERT_NE(rabitq, nullptr);
    const auto params = rabitq->GetParams();
    ASSERT_NE(params.centroid, nullptr);
    EXPECT_EQ(params.binary_bytes, (d + 7) / 8);
    EXPECT_EQ(kmeans.GetState().code_size, params.binary_bytes + sizeof(RaBitQFactors));

    // Fit runs on the rotated training data, so the exposed centroid is its column mean
    auto expected = ColumnMean(rotated, n, d);
    for (size_t j = 0; j < d; ++j) {
        ASSERT_NEAR(params.centroid[j], expected[j], 1e-4f) << "at dim " << j;
    }
}
