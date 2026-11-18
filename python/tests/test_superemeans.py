from pathlib import Path

import numpy as np
import pytest

from superkmeans import SuperEMeans, SuperKMeans


QUANTIZERS = ["f32", "sq8", "lvq4", "rabitq"]

_TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "test_data.bin"
_TEST_DATA_D = 1024


def load_test_data(n, d):
    """First n rows and first d dims of the fixed test_data.bin."""
    full = np.fromfile(_TEST_DATA, dtype=np.float32, count=n * _TEST_DATA_D)
    return np.ascontiguousarray(full.reshape(n, _TEST_DATA_D)[:, :d])


class TestSuperEMeans:
    """Test suite for e-means (SuperEMeans) across all quantizers."""

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_train_shape_and_epoch_stats(self, quantizer):
        n, d, k, epochs = 5000, 128, 300, 3
        data = load_test_data(n, d)

        em = SuperEMeans(n_clusters=k, dimensionality=d, quantizer=quantizer, epochs=epochs)
        centroids = em.train(data)

        assert centroids.shape == (k, d)
        assert centroids.dtype == np.float32
        assert np.all(np.isfinite(centroids))
        assert em.is_trained_
        assert em.quantizer_ == quantizer
        assert not em.hierarchical_
        stats = em.iteration_stats
        assert len(stats) == epochs
        assert stats[0].is_gemm_only
        assert not any(s.is_gemm_only for s in stats[1:])

    def test_batch_size_and_step_follow_laws_or_overrides(self):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        by_law = SuperEMeans(n_clusters=k, dimensionality=d)
        assert by_law.batch_size_ is None
        assert by_law.initial_step_ is None
        by_law.train(data)
        batch_size = by_law.batch_size_
        assert 0 < batch_size <= n and batch_size & (batch_size - 1) == 0
        assert 0.0 < by_law.initial_step_ <= 1.0

        overridden = SuperEMeans(n_clusters=k, dimensionality=d, batch_size=1000, initial_step=0.5)
        overridden.train(data)
        assert overridden.batch_size_ == 1000
        assert overridden.initial_step_ == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"batch_size": 0},
            {"initial_step": 0.0},
            {"initial_step": 1.5},
            {"quantizer": "bogus"},
        ],
    )
    def test_invalid_params(self, kwargs):
        with pytest.raises(ValueError):
            SuperEMeans(n_clusters=10, dimensionality=128, **kwargs)

    def test_train_twice_raises(self):
        data = load_test_data(1000, 128)
        em = SuperEMeans(n_clusters=10, dimensionality=128)
        em.train(data)
        with pytest.raises(RuntimeError):
            em.train(data)

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_assign_family_valid(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        em = SuperEMeans(n_clusters=k, dimensionality=d, quantizer=quantizer)
        centroids = em.train(data)

        exact = em.assign(data, centroids)
        reuse = em.assign_training_points(data, centroids)
        for labels in (exact, em.quantized_assign(data, centroids), reuse):
            assert labels.shape == (n,)
            assert labels.dtype == np.uint32
            assert np.all(labels < k)
        reuse_vs_exact = np.mean(reuse == exact)
        assert reuse_vs_exact > 0.5, f"{quantizer}: reuse vs exact {reuse_vs_exact:.3f}"

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_wcss_close_to_superkmeans(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        def wcss(model, centroids):
            labels = model.assign(data, centroids)
            return float(np.sum((data - centroids[labels]) ** 2))

        em = SuperEMeans(n_clusters=k, dimensionality=d, quantizer=quantizer)
        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer, iters=10, sampling_fraction=1.0
        )
        ratio = wcss(em, em.train(data)) / wcss(km, km.train(data))
        assert ratio < 1.05, f"{quantizer}: e-means / SuperKMeans WCSS {ratio:.4f}"

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_overwrite_input_same_clustering_as_train(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)
        overwritten = data.copy()

        em = SuperEMeans(n_clusters=k, dimensionality=d, quantizer=quantizer)
        centroids = em.train(data)
        em_in_place = SuperEMeans(n_clusters=k, dimensionality=d, quantizer=quantizer)
        centroids_in_place = em_in_place.train(overwritten, overwrite_input=True)

        assert not np.allclose(overwritten, data)
        np.testing.assert_allclose(
            np.linalg.norm(overwritten, axis=1), np.linalg.norm(data, axis=1), rtol=1e-4
        )
        np.testing.assert_allclose(
            np.sort(np.linalg.norm(centroids_in_place, axis=1)),
            np.sort(np.linalg.norm(centroids, axis=1)),
            rtol=1e-3,
        )
        agreement = np.mean(
            em_in_place.assign(overwritten, centroids_in_place) == em.assign(data, centroids)
        )
        assert agreement > 0.95, f"{quantizer}: assignment agreement {agreement:.4f}"

    def test_repr(self):
        assert repr(SuperEMeans(n_clusters=10, dimensionality=128)) == (
            "SuperEMeans(n_clusters=10, dimensionality=128, epochs=2, trained=False)"
        )
        assert "quantizer='sq8'" in repr(
            SuperEMeans(n_clusters=10, dimensionality=128, quantizer="sq8")
        )
