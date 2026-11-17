#undef HAS_FFTW

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superemeans.h"

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

HierarchicalSuperEMeansConfig TestConfig(const uint32_t n_threads) {
    HierarchicalSuperEMeansConfig config;
    config.seed = skm_test::RECALL_SEED;
    config.n_threads = n_threads;
    config.suppress_warnings = true;
    return config;
}

struct EMeansQuality {
    float recall = 0.0f;
    float reuse_recall = 0.0f;
    double wcss = 0.0;
};

template <typename Engine>
EMeansQuality TrainAndMeasure(
    Engine& engine,
    std::vector<float> data,
    const size_t d,
    const size_t n_clusters,
    const bool in_place = false
) {
    using namespace skm_test;
    auto gt = BuildGroundTruthNN(data.data(), RECALL_N, d, RECALL_N_QUERIES, RECALL_KNN);
    auto centroids =
        in_place ? engine.TrainInPlace(data.data(), RECALL_N) : engine.Train(data.data(), RECALL_N);
    auto exact = engine.Assign(data.data(), centroids.data(), RECALL_N, n_clusters);
    auto reuse = engine.AssignTrainingPoints(data.data(), centroids.data(), RECALL_N, n_clusters);
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

template <typename T>
class HierarchicalSuperEMeansTest : public ::testing::Test {};

TYPED_TEST_SUITE(HierarchicalSuperEMeansTest, EMeansTags, QuantizerNames);

TYPED_TEST(HierarchicalSuperEMeansTest, BasicTraining) {
    const size_t n = skm_test::RECALL_N, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, n, skm_test::RECALL_D, d);
    const auto config = TestConfig(4);
    HierarchicalSuperEMeans<TypeParam::value> emeans(n_clusters, d, config);
    EXPECT_FALSE(emeans.IsTrained());
    auto centroids = emeans.Train(data.data(), n);
    EXPECT_TRUE(emeans.IsTrained());

    ASSERT_EQ(centroids.size(), n_clusters * d);
    for (float v : centroids) {
        ASSERT_TRUE(std::isfinite(v));
    }
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(emeans.assignments[i], n_clusters);
    }
    const auto& stats = emeans.hierarchical_iteration_stats;
    EXPECT_EQ(stats.mesoclustering_iteration_stats.size(), config.epochs_mesoclustering);
    const size_t n_fine_stats = stats.fineclustering_iteration_stats.size();
    EXPECT_GT(n_fine_stats, 0u);
    EXPECT_EQ(n_fine_stats % config.epochs_fineclustering, 0u);
    EXPECT_LE(n_fine_stats, GetNMesoclusters(n_clusters) * config.epochs_fineclustering);
}

TYPED_TEST(HierarchicalSuperEMeansTest, InvalidInputs_Throw) {
    constexpr Quantization q = TypeParam::value;
    const size_t n = 1000, d = 16, n_clusters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    const auto config = TestConfig(1);

    HierarchicalSuperEMeans<q> too_many_clusters(n + 1, d, config);
    EXPECT_THROW(too_many_clusters.Train(data.data(), n), std::runtime_error);
    HierarchicalSuperEMeans<q> no_points(n_clusters, d, config);
    EXPECT_THROW(no_points.Train(data.data(), 0), std::invalid_argument);

    HierarchicalSuperEMeans<q> trained(n_clusters, d, config);
    trained.Train(data.data(), n);
    EXPECT_THROW(trained.Train(data.data(), n), std::runtime_error);

    auto no_meso_epochs = config;
    no_meso_epochs.epochs_mesoclustering = 0;
    EXPECT_THROW(HierarchicalSuperEMeans<q>(n_clusters, d, no_meso_epochs), std::invalid_argument);
    auto no_fine_epochs = config;
    no_fine_epochs.epochs_fineclustering = 0;
    EXPECT_THROW(HierarchicalSuperEMeans<q>(n_clusters, d, no_fine_epochs), std::invalid_argument);
}

TYPED_TEST(HierarchicalSuperEMeansTest, Determinism) {
    constexpr Quantization q = TypeParam::value;
    const size_t n = skm_test::RECALL_N, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, n, skm_test::RECALL_D, d);

    auto config = TestConfig(4);
    auto first = HierarchicalSuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);
    auto second = HierarchicalSuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);
    config.seed = skm_test::RECALL_SEED + 1;
    auto other_seed = HierarchicalSuperEMeans<q>(n_clusters, d, config).Train(data.data(), n);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, other_seed);
}

TYPED_TEST(HierarchicalSuperEMeansTest, RecallAndWCSSMatchAcrossThreadCounts) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = skm_test::RECALL_N_CLUSTERS;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);

    HierarchicalSuperEMeans<q> serial(n_clusters, d, TestConfig(1));
    HierarchicalSuperEMeans<q> parallel(n_clusters, d, TestConfig(4));
    const auto serial_quality = TrainAndMeasure(serial, data, d, n_clusters);
    const auto parallel_quality = TrainAndMeasure(parallel, data, d, n_clusters);
    EXPECT_NEAR(parallel_quality.recall, serial_quality.recall, skm_test::RECALL_TOL);
    EXPECT_NEAR(parallel_quality.wcss / serial_quality.wcss, 1.0, 0.01);
}

TYPED_TEST(HierarchicalSuperEMeansTest, AngularMode) {
    const size_t n = 5000, d = 64, n_clusters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, true);

    auto config = TestConfig(4);
    config.angular = true;
    HierarchicalSuperEMeans<TypeParam::value> emeans(n_clusters, d, config);
    auto centroids = emeans.Train(data.data(), n);

    for (size_t i = 0; i < n_clusters; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            norm += centroids[i * d + j] * centroids[i * d + j];
        }
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-3f) << "centroid " << i;
    }
}

TYPED_TEST(HierarchicalSuperEMeansTest, RecallComparableToFlatSuperEMeans) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = skm_test::RECALL_N_CLUSTERS;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);
    const auto config = TestConfig(4);

    HierarchicalSuperEMeans<q> hierarchical(n_clusters, d, config);
    SuperEMeans<q> flat(n_clusters, d, config);
    const auto hierarchical_quality = TrainAndMeasure(hierarchical, data, d, n_clusters);
    const auto flat_quality = TrainAndMeasure(flat, data, d, n_clusters);
    EXPECT_GE(hierarchical_quality.recall, flat_quality.recall - skm_test::RECALL_TOL);
}

TYPED_TEST(HierarchicalSuperEMeansTest, TrainInPlace_RecallMatchesTrain) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = skm_test::RECALL_N_CLUSTERS;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);

    HierarchicalSuperEMeans<q> train(n_clusters, d, TestConfig(4));
    HierarchicalSuperEMeans<q> in_place(n_clusters, d, TestConfig(4));
    const auto train_quality = TrainAndMeasure(train, data, d, n_clusters);
    const auto in_place_quality = TrainAndMeasure(in_place, data, d, n_clusters, true);
    EXPECT_NEAR(in_place_quality.recall, train_quality.recall, skm_test::RECALL_PRUNE_TOL);
}

TYPED_TEST(HierarchicalSuperEMeansTest, SLOW_AssignTrainingPointsAfterTrain) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);

    HierarchicalSuperEMeans<q> emeans(n_clusters, d, TestConfig(4));
    const auto quality = TrainAndMeasure(emeans, data, d, n_clusters);
    EXPECT_GE(quality.reuse_recall, quality.recall - skm_test::RECALL_TOL);
}

TYPED_TEST(HierarchicalSuperEMeansTest, FloatCentroidUpdate) {
    constexpr Quantization q = TypeParam::value;
    const size_t d = skm_test::RECALL_D, n_clusters = skm_test::RECALL_N_CLUSTERS;
    auto data = skm_test::LoadTestDataSubdim(TEST_DATA_PATH, skm_test::RECALL_N, d, d);
    auto config = TestConfig(4);
    config.quantized_centroid_update = false;
    config.full_precision_final_centroids = true;

    HierarchicalSuperEMeans<q> emeans(n_clusters, d, config);
    const auto quality = TrainAndMeasure(emeans, data, d, n_clusters);
    EXPECT_GT(quality.recall, 0.5f);
}

TEST(HierarchicalSuperEMeansAliasesTest, WrapperAliases) {
    EXPECT_TRUE(
        (std::is_same_v<HierarchicalSuperEMeansSQ8, HierarchicalSuperEMeans<Quantization::sq8>>)
    );
    EXPECT_TRUE(
        (std::is_same_v<HierarchicalSuperEMeansLVQ4, HierarchicalSuperEMeans<Quantization::lvq4>>)
    );
    EXPECT_TRUE((std::is_same_v<
                 HierarchicalSuperEMeansRabitQ,
                 HierarchicalSuperEMeans<Quantization::rabitq>>) );
}
