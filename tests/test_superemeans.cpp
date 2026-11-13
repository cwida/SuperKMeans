#undef HAS_FFTW

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superemeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

using EMeansTags = ::testing::Types<
    QuantizationTag<Quantization::f32>,
    QuantizationTag<Quantization::sq8>,
    QuantizationTag<Quantization::lvq4>,
    QuantizationTag<Quantization::rabitq>>;

struct QuantizerNames {
    template <typename T>
    static std::string GetName(int) {
        return QuantizationName(T::value);
    }
};

const char* const TEST_DATA_PATH = CMAKE_SOURCE_DIR "/tests/test_data.bin";

SuperEMeansConfig TestConfig(const uint32_t epochs, const bool pruning) {
    SuperEMeansConfig config;
    config.iters = epochs;
    config.seed = skm_test::RECALL_SEED;
    config.n_threads = 1;
    config.use_blas_only = !pruning;
    return config;
}

struct EMeansQuality {
    float recall = 0.0f;
    float reuse_recall = 0.0f;
    double wcss = 0.0;
};

template <Quantization q>
EMeansQuality TrainAndMeasure(
    std::vector<float> data,
    const size_t d,
    const size_t n_clusters,
    const SuperEMeansConfig& config,
    const bool in_place = false
) {
    using namespace skm_test;
    auto gt = BuildGroundTruthNN(data.data(), RECALL_N, d, RECALL_N_QUERIES, RECALL_KNN);
    SuperEMeans<q> emeans(n_clusters, d, config);
    auto centroids =
        in_place ? emeans.TrainInPlace(data.data(), RECALL_N) : emeans.Train(data.data(), RECALL_N);
    auto exact = emeans.Assign(data.data(), centroids.data(), RECALL_N, n_clusters);
    auto reuse = emeans.AssignTrainingPoints(data.data(), centroids.data(), RECALL_N, n_clusters);
    EMeansQuality quality;
    quality.recall = RecallAtFraction(
        gt,
        exact,
        data.data(),
        centroids.data(),
        RECALL_N_QUERIES,
        n_clusters,
        d,
        RECALL_KNN,
        RECALL_FRAC
    );
    quality.reuse_recall = RecallAtFraction(
        gt,
        reuse,
        data.data(),
        centroids.data(),
        RECALL_N_QUERIES,
        n_clusters,
        d,
        RECALL_KNN,
        RECALL_FRAC
    );
    quality.wcss =
        SuperKMeans<>::ComputeWCSS(data.data(), centroids.data(), exact.data(), RECALL_N, d);
    return quality;
}

} // namespace

TEST(SuperEMeansLawsTest, RoundToPowerOfTwo) {
    EXPECT_EQ(RoundToPowerOfTwo(0.5), 1u);
    EXPECT_EQ(RoundToPowerOfTwo(1.0), 1u);
    EXPECT_EQ(RoundToPowerOfTwo(3.0), 4u);
    EXPECT_EQ(RoundToPowerOfTwo(5.0), 4u);
    EXPECT_EQ(RoundToPowerOfTwo(6.0), 8u);
    EXPECT_EQ(RoundToPowerOfTwo(1000.0), 1024u);
    EXPECT_EQ(RoundToPowerOfTwo(1500.0), 2048u);
}

TEST(SuperEMeansLawsTest, BatchSizeLaw) {
    EXPECT_EQ(EMeansBatchSize(769382, 3508, 2), 32768u);
    EXPECT_EQ(EMeansBatchSize(769382, 3508, 10), 131072u);
    EXPECT_EQ(EMeansBatchSize(5000000, 8944, 2), 131072u);
    EXPECT_EQ(EMeansBatchSize(1000, 10, 2), 32u);
    EXPECT_EQ(EMeansBatchSize(1000, 10, 1000), 512u);
}

TEST(SuperEMeansLawsTest, StepLaw) {
    EXPECT_NEAR(EMeansInitialStep(769382, 3508, 2, 32768), 0.4626f, 1e-4f);
    EXPECT_FLOAT_EQ(EMeansInitialStep(769382, 3508, 10, 131072), 1.0f);
    EXPECT_FLOAT_EQ(EMeansInitialStep(1000000000, 1000000, 100, 1), 1e-4f);
}

TEST(SuperEMeansLawsTest, CosineSchedule) {
    EXPECT_FLOAT_EQ(EMeansScheduledStep(0.5f, 0, 100), 0.5f);
    EXPECT_FLOAT_EQ(EMeansScheduledStep(0.5f, 50, 100), 0.275f);
    EXPECT_FLOAT_EQ(EMeansScheduledStep(0.5f, 100, 100), 0.05f);
    EXPECT_FLOAT_EQ(EMeansScheduledStep(0.5f, 200, 100), 0.05f);
}

template <typename T>
class SuperEMeansTest : public ::testing::Test {};

TYPED_TEST_SUITE(SuperEMeansTest, EMeansTags, QuantizerNames);

TYPED_TEST(SuperEMeansTest, BasicTraining) {
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, n, skm_test::RECALL_D, d);

    SuperEMeansConfig config;
    config.n_threads = 1;
    SuperEMeans<TypeParam::value> emeans(n_clusters, d, config);
    EXPECT_FALSE(emeans.IsTrained());
    auto centroids = emeans.Train(data.data(), n);
    EXPECT_TRUE(emeans.IsTrained());

    ASSERT_EQ(centroids.size(), n_clusters * d);
    for (float v : centroids) {
        ASSERT_TRUE(std::isfinite(v));
    }
    EXPECT_GT(emeans.GetBatchSize(), 0u);
    EXPECT_LE(emeans.GetBatchSize(), n);
    EXPECT_GT(emeans.GetInitialStep(), 0.0f);
    EXPECT_LE(emeans.GetInitialStep(), 1.0f);

    const auto& stats = emeans.GetIterationStats();
    ASSERT_EQ(stats.size(), EMEANS_DEFAULT_EPOCHS);
    for (size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].iteration, i + 1);
    }
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(emeans.assignments[i], n_clusters);
    }
}

TYPED_TEST(SuperEMeansTest, InvalidInputs_Throw) {
    constexpr Quantization q = TypeParam::value;
    const size_t n = 1000, d = 16, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    SuperEMeansConfig config;
    config.n_threads = 1;

    SuperEMeans<q> too_many_clusters(n + 1, d, config);
    EXPECT_THROW(too_many_clusters.Train(data.data(), n), std::runtime_error);
    SuperEMeans<q> no_points(n_clusters, d, config);
    EXPECT_THROW(no_points.Train(data.data(), 0), std::invalid_argument);

    SuperEMeans<q> trained(n_clusters, d, config);
    trained.Train(data.data(), n);
    EXPECT_THROW(trained.Train(data.data(), n), std::runtime_error);

    SuperEMeansConfig bad_step = config;
    bad_step.initial_step = 1.5f;
    EXPECT_THROW(SuperEMeans<q>(n_clusters, d, bad_step), std::invalid_argument);
    SuperEMeansConfig bad_chunk = config;
    bad_chunk.assignment_chunk_size = 0;
    EXPECT_THROW(SuperEMeans<q>(n_clusters, d, bad_chunk), std::invalid_argument);
}

TYPED_TEST(SuperEMeansTest, Determinism) {
    constexpr Quantization q = TypeParam::value;
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, n, skm_test::RECALL_D, d);

    SuperEMeansConfig config;
    config.seed = 7;
    auto first = SuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);
    auto second = SuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);
    config.seed = 8;
    auto other_seed = SuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, other_seed);
}

TYPED_TEST(SuperEMeansTest, AngularMode) {
    const size_t n = 5000, d = 64, n_clusters = 20;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, true);

    SuperEMeansConfig config;
    config.n_threads = 1;
    config.angular = true;
    SuperEMeans<TypeParam::value> emeans(n_clusters, d, config);
    auto centroids = emeans.Train(data.data(), n);

    for (size_t i = 0; i < n_clusters; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            norm += centroids[i * d + j] * centroids[i * d + j];
        }
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-3f) << "centroid " << i;
    }
}

TYPED_TEST(SuperEMeansTest, Revival) {
    const size_t n_distinct = 1000, n_copies = 10, d = 32, n_clusters = 500;
    const size_t n = n_distinct * n_copies;
    std::vector<float> distinct = MakeBlobs(n_distinct, d, 50);
    std::vector<float> data;
    for (size_t i = 0; i < n_copies; ++i) {
        data.insert(data.end(), distinct.begin(), distinct.end());
    }

    SuperEMeansConfig config;
    config.n_threads = 1;
    SuperEMeans<TypeParam::value> emeans(n_clusters, d, config);
    auto centroids = emeans.Train(data.data(), n);

    size_t n_splits = 0;
    for (const auto& stats : emeans.GetIterationStats()) {
        n_splits += stats.split;
    }
    EXPECT_GT(n_splits, 0u);
    for (float v : centroids) {
        ASSERT_TRUE(std::isfinite(v));
    }
}

TYPED_TEST(SuperEMeansTest, TrainInPlace_RecallMatchesTrain) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);
    const auto config = TestConfig(EMEANS_DEFAULT_EPOCHS, false);

    auto train = TrainAndMeasure<q>(data, d, skm_test::RECALL_N_CLUSTERS, config);
    auto in_place = TrainAndMeasure<q>(data, d, skm_test::RECALL_N_CLUSTERS, config, true);
    EXPECT_NEAR(in_place.recall, train.recall, skm_test::RECALL_PRUNE_TOL);
}

TYPED_TEST(SuperEMeansTest, FloatCentroidUpdate) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);
    const auto config = TestConfig(EMEANS_DEFAULT_EPOCHS, false);
    auto float_config = config;
    float_config.quantized_centroid_update = false;
    float_config.full_precision_final_centroids = true;

    auto quantized_update = TrainAndMeasure<q>(data, d, skm_test::RECALL_N_CLUSTERS, config);
    auto float_update = TrainAndMeasure<q>(data, d, skm_test::RECALL_N_CLUSTERS, float_config);
    EXPECT_GE(float_update.recall, quantized_update.recall - skm_test::RECALL_TOL);
}

TYPED_TEST(SuperEMeansTest, SLOW_PruningEpochs) {
    constexpr Quantization q = TypeParam::value;
    const size_t n = skm_test::RECALL_N, d = skm_test::RECALL_D, n_clusters = 300;
    const uint32_t n_epochs = 3;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, n, d, d);

    SuperEMeans<q> pruned(n_clusters, d, TestConfig(n_epochs, true));
    pruned.Train(data.data(), n);
    const auto& stats = pruned.GetIterationStats();
    ASSERT_EQ(stats.size(), n_epochs);
    EXPECT_TRUE(stats[0].is_gemm_only);
    for (size_t i = 1; i < n_epochs; ++i) {
        EXPECT_FALSE(stats[i].is_gemm_only) << "epoch " << i + 1;
        EXPECT_GT(stats[i].not_pruned_pct, 0.0f) << "epoch " << i + 1;
        EXPECT_LT(stats[i].not_pruned_pct, 1.0f) << "epoch " << i + 1;
        EXPECT_GT(stats[i].partial_d, 0u) << "epoch " << i + 1;
    }

    SuperEMeans<q> gemm_only(n_clusters, d, TestConfig(n_epochs, false));
    gemm_only.Train(data.data(), n);
    for (const auto& epoch_stats : gemm_only.GetIterationStats()) {
        EXPECT_TRUE(epoch_stats.is_gemm_only);
    }
}

TYPED_TEST(SuperEMeansTest, SLOW_PrunedMatchesUnpruned) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);

    auto pruned = TrainAndMeasure<q>(data, d, n_clusters, TestConfig(EMEANS_DEFAULT_EPOCHS, true));
    auto gemm_only =
        TrainAndMeasure<q>(data, d, n_clusters, TestConfig(EMEANS_DEFAULT_EPOCHS, false));
    EXPECT_LT(pruned.wcss / gemm_only.wcss, 1.005);
    EXPECT_GE(pruned.recall, gemm_only.recall - skm_test::RECALL_TOL);
    EXPECT_GE(pruned.reuse_recall, pruned.recall - skm_test::RECALL_TOL);
}

TYPED_TEST(SuperEMeansTest, SLOW_ChunkedMatchesUnchunked) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);
    auto config = TestConfig(EMEANS_DEFAULT_EPOCHS, true);

    auto unchunked = TrainAndMeasure<q>(data, d, n_clusters, config);
    config.assignment_chunk_size = 100;
    auto chunked = TrainAndMeasure<q>(data, d, n_clusters, config);
    EXPECT_NEAR(chunked.wcss / unchunked.wcss, 1.0, 1e-3);
    EXPECT_NEAR(chunked.recall, unchunked.recall, skm_test::RECALL_PRUNE_TOL);
}

TEST(SuperEMeansAliasesTest, WrapperAliases) {
    EXPECT_TRUE((std::is_same_v<SuperEMeansSQ8, SuperEMeans<Quantization::sq8>>) );
    EXPECT_TRUE((std::is_same_v<SuperEMeansLVQ4, SuperEMeans<Quantization::lvq4>>) );
    EXPECT_TRUE((std::is_same_v<SuperEMeansRabitQ, SuperEMeans<Quantization::rabitq>>) );
}
