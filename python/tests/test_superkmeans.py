import numpy as np
import pytest
from superkmeans import SuperKMeans, SuperKMeansIterationStats, HierarchicalSuperKMeansIterationStats


class TestSuperKMeans:
    """Test suite for SuperKMeans class."""

    def test_init_default(self):
        kmeans = SuperKMeans(n_clusters=10, dimensionality=50)
        assert kmeans.n_clusters_ == 10
        assert not kmeans.is_trained_

    def test_init_with_params(self):
        kmeans = SuperKMeans(
            n_clusters=20,
            dimensionality=64,
            iters=15,
            sampling_fraction=0.5,
            verbose=True,
            seed=123
        )
        assert kmeans.n_clusters_ == 20
        assert not kmeans.is_trained_

    def test_init_invalid_params(self):
        with pytest.raises(ValueError):
            SuperKMeans(n_clusters=0, dimensionality=50)
        with pytest.raises(ValueError):
            SuperKMeans(n_clusters=10, dimensionality=0)
        with pytest.raises(ValueError):
            SuperKMeans(n_clusters=10, dimensionality=50, sampling_fraction=1.5)

    def test_train_simple(self):
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5, verbose=False)
        centroids = kmeans.train(data)

        assert centroids.shape == (k, d)
        assert centroids.dtype == np.float32
        assert kmeans.is_trained_

    def test_train_with_queries(self):
        """Test training with query vectors for recall monitoring."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100
        q = 50
        data = np.random.randn(n, d).astype(np.float32)
        queries = np.random.randn(q, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5, objective_k=10, verbose=False)
        centroids = kmeans.train(data, queries=queries)

        assert centroids.shape == (k, d)
        assert kmeans.is_trained_
        assert len(kmeans.iteration_stats) > 0

    def test_assign(self):
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100
        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5)
        centroids = kmeans.train(data)

        assignments = kmeans.assign(data, centroids)

        assert assignments.shape == (n,)
        assert assignments.dtype == np.uint32
        assert np.all(assignments >= 0)
        assert np.all(assignments < k)

    def test_assign_different_data(self):
        np.random.seed(42)
        n = 100000
        m = 500
        d = 512
        k = 100
        train_data = np.random.randn(n, d).astype(np.float32)
        test_data = np.random.randn(m, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5)
        centroids = kmeans.train(train_data)
        assignments = kmeans.assign(test_data, centroids)

        assert assignments.shape == (m,)
        assert np.all(assignments >= 0)
        assert np.all(assignments < k)

    def test_iteration_stats(self):
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100
        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5, verbose=True)
        kmeans.train(data)

        stats = kmeans.iteration_stats
        assert len(stats) == 5

        for i, stat in enumerate(stats):
            assert isinstance(stat, SuperKMeansIterationStats)
            assert stat.iteration == i + 1
            assert stat.objective > 0
            assert stat.shift >= 0

    def test_train_dtype_validation(self):
        data_wrong_dtype = np.random.randn(100, 32).astype(np.float64)
        kmeans = SuperKMeans(n_clusters=10, dimensionality=32)
        with pytest.raises(ValueError, match="dtype float32"):
            kmeans.train(data_wrong_dtype)

    def test_train_shape_validation(self):
        data_1d = np.random.randn(100).astype(np.float32)
        data_3d = np.random.randn(100, 32, 2).astype(np.float32)
        kmeans = SuperKMeans(n_clusters=10, dimensionality=32)
        with pytest.raises(ValueError, match="2-dimensional"):
            kmeans.train(data_1d)
        with pytest.raises(ValueError, match="2-dimensional"):
            kmeans.train(data_3d)

    def test_train_dimensionality_mismatch(self):
        data = np.random.randn(100, 64).astype(np.float32) 
        kmeans = SuperKMeans(n_clusters=10, dimensionality=32)
        with pytest.raises(ValueError, match="dimensionality"):
            kmeans.train(data)

    def test_assign_validation(self):
        np.random.seed(42)
        n = 100
        d = 512
        k = 10
        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d)
        centroids = kmeans.train(data)

        data_wrong = data.astype(np.float64)
        with pytest.raises(ValueError, match="dtype float32"):
            kmeans.assign(data_wrong, centroids)

        data_wrong_dim = np.random.randn(100, 16).astype(np.float32)
        with pytest.raises(ValueError, match="same dimensionality"):
            kmeans.assign(data_wrong_dim, centroids)

    def test_non_contiguous_arrays(self):
        np.random.seed(42)
        n = 100
        d = 256
        k = 10
        data = np.random.randn(n, d * 2).astype(np.float32)
        data_nc = data[:, ::2]  # This creates non-contiguous array with d=256

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d)
        centroids = kmeans.train(data_nc)

        assert centroids.shape == (k, d)

    def test_repr(self):
        k = 10
        d = 512
        kmeans = SuperKMeans(n_clusters=k, dimensionality=d)
        repr_str = repr(kmeans)

        assert "SuperKMeans" in repr_str
        assert f"n_clusters={k}" in repr_str
        assert f"dimensionality={d}" in repr_str
        assert "trained=False" in repr_str

    def test_reproducibility(self):
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100
        seed = 123
        iters = 5

        data = np.random.randn(n, d).astype(np.float32)
        kmeans1 = SuperKMeans(n_clusters=k, dimensionality=d, iters=iters, seed=seed)
        centroids1 = kmeans1.train(data)

        kmeans2 = SuperKMeans(n_clusters=k, dimensionality=d, iters=iters, seed=seed)
        centroids2 = kmeans2.train(data)

        np.testing.assert_array_equal(centroids1, centroids2)

    def test_hierarchical_explicit_true(self):
        """Test explicit hierarchical=True mode."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=True,
            iters_mesoclustering=5,
            iters_fineclustering=5,
            iters_refinement=2,
            verbose=False
        )
        centroids = kmeans.train(data)

        assert centroids.shape == (k, d)
        assert centroids.dtype == np.float32
        assert kmeans.is_trained_
        assert kmeans.hierarchical_ is True
        assert kmeans.hierarchical_iteration_stats is not None

    def test_hierarchical_explicit_false(self):
        """Test explicit hierarchical=False mode."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=False,
            iters=5,
            verbose=False
        )
        centroids = kmeans.train(data)

        assert centroids.shape == (k, d)
        assert kmeans.hierarchical_ is False
        assert kmeans.hierarchical_iteration_stats is None

    def test_hierarchical_auto_small_dataset(self):
        """Test automatic hierarchical mode selection for small dataset (n <= 100,000)."""
        np.random.seed(42)
        n = 50000  # Small dataset
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5, verbose=False)
        centroids = kmeans.train(data)

        assert centroids.shape == (k, d)
        assert kmeans.hierarchical_ is False  # Should automatically use non-hierarchical

    def test_hierarchical_auto_large_dataset(self):
        """Test automatic hierarchical mode selection for large dataset (n > 100,000)."""
        np.random.seed(42)
        n = 150000
        d = 128
        k = 256

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            iters_mesoclustering=3,
            iters_fineclustering=3,
            iters_refinement=1,
            verbose=False
        )
        centroids = kmeans.train(data)

        assert centroids.shape == (k, d)
        assert kmeans.hierarchical_ is True  # Should automatically use hierarchical
        assert kmeans.hierarchical_iteration_stats is not None

    def test_hierarchical_assign(self):
        """Test assign method in hierarchical mode."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=True,
            iters_mesoclustering=3,
            iters_fineclustering=3,
            iters_refinement=1
        )
        centroids = kmeans.train(data)
        assignments = kmeans.assign(data, centroids)

        assert assignments.shape == (n,)
        assert assignments.dtype == np.uint32
        assert np.all(assignments >= 0)
        assert np.all(assignments < k)

    def test_hierarchical_iteration_stats(self):
        """Test that hierarchical iteration stats are populated in hierarchical mode."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=True,
            iters_mesoclustering=3,
            iters_fineclustering=3,
            iters_refinement=2,
            verbose=True
        )
        kmeans.train(data)

        hierarchical_stats = kmeans.hierarchical_iteration_stats
        assert hierarchical_stats is not None
        assert isinstance(hierarchical_stats, HierarchicalSuperKMeansIterationStats)
        assert len(hierarchical_stats.mesoclustering_iteration_stats) == 3
        assert len(hierarchical_stats.fineclustering_iteration_stats) > 0
        assert len(hierarchical_stats.fineclustering_iteration_stats) <= 10 * 3  # At most 10 mesoclusters * 3 iters
        assert len(hierarchical_stats.refinement_iteration_stats) == 2

    def test_hierarchical_reproducibility(self):
        """Test reproducibility in hierarchical mode."""
        np.random.seed(42)
        n = 10000
        d = 512
        k = 100
        seed = 123

        data = np.random.randn(n, d).astype(np.float32)

        kmeans1 = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=True,
            iters_mesoclustering=3,
            iters_fineclustering=3,
            iters_refinement=1,
            seed=seed
        )
        centroids1 = kmeans1.train(data)

        kmeans2 = SuperKMeans(
            n_clusters=k,
            dimensionality=d,
            hierarchical=True,
            iters_mesoclustering=3,
            iters_fineclustering=3,
            iters_refinement=1,
            seed=seed
        )
        centroids2 = kmeans2.train(data)

        np.testing.assert_array_equal(centroids1, centroids2)

    def test_hierarchical_repr(self):
        """Test string representation with hierarchical mode."""
        k = 100
        d = 512

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, hierarchical=True)
        repr_str = repr(kmeans)

        assert "SuperKMeans" in repr_str
        assert f"n_clusters={k}" in repr_str
        assert f"dimensionality={d}" in repr_str
        assert "trained=False" in repr_str

        np.random.seed(42)
        data = np.random.randn(1000, d).astype(np.float32)
        kmeans.train(data)
        repr_str_trained = repr(kmeans)
        assert "hierarchical=True" in repr_str_trained


    def test_assign_training_points(self):
        """Test that assign_training_points produces similar results to assign on training data."""
        np.random.seed(42)
        n = 50000
        d = 512
        k = 500

        data = np.random.randn(n, d).astype(np.float32)

        kmeans = SuperKMeans(n_clusters=k, dimensionality=d, iters=5, verbose=False)
        centroids = kmeans.train(data)

        assignments = kmeans.assign(data, centroids)
        fast_assignments = kmeans.assign_training_points(data, centroids)

        assert fast_assignments.shape == (n,)
        assert fast_assignments.dtype == np.uint32
        assert np.all(fast_assignments < k)

        agreement = np.mean(assignments == fast_assignments)
        assert agreement >= 0.95, f"Expected >= 95% agreement, got {agreement:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
