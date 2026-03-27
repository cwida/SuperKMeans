from typing import Optional, List
import numpy as np
from numpy.typing import NDArray

try:
    from importlib.metadata import version as _version
    __version__ = _version("superkmeans")
except Exception:
    __version__ = "0.0.0+dev"

from ._superkmeans import (
    SuperKMeans as _SuperKMeansCpp,
    SuperKMeansConfig as _SuperKMeansConfigCpp,
    SuperKMeansIterationStats,
    HierarchicalSuperKMeans as _HierarchicalSuperKMeansCpp,
    HierarchicalSuperKMeansConfig as _HierarchicalSuperKMeansConfigCpp,
    HierarchicalSuperKMeansIterationStats,
)


class SuperKMeans:
    """
    A Super fast K-Means for High-Dimensional vectors

    Parameters
    ----------
    n_clusters : int
        Number of clusters to create.
    dimensionality : int
        Number of dimensions in the data.
    hierarchical : bool, optional (default=None)
        Whether to use hierarchical clustering. If None, automatically
        uses hierarchical=True for datasets with n > 100,000, otherwise False.
    iters : int, optional (default=10)
        Number of k-means iterations (only for non-hierarchical mode).
    iters_mesoclustering : int, optional (default=3)
        Number of mesoclustering iterations (only for hierarchical mode).
    iters_fineclustering : int, optional (default=5)
        Number of fineclustering iterations (only for hierarchical mode).
    iters_refinement : int, optional (default=0)
        Number of refinement iterations (only for hierarchical mode).
    sampling_fraction : float, optional (default=None)
        Fraction of data to sample, must be in (0.0, 1.0].
        If None, uses 0.3 for flat mode and 1.0 for hierarchical mode.
    max_points_per_cluster : int, optional (default=256)
        Maximum number of points per cluster to sample (FAISS style).
    n_threads : int, optional (default=0)
        Number of CPU threads to use. 0 means use all available.
    seed : int, optional (default=42)
        Random seed for reproducibility.
    use_blas_only : bool, optional (default=False)
        Use GEMM-only computation without pruning.
    tol : float, optional (default=1e-4)
        Tolerance for shift-based early termination.
    recall_tol : float, optional (default=0.0050)
        Tolerance for recall-based early termination.
    early_termination : bool, optional (default=True)
        Whether to stop early when convergence criteria are met.
    sample_queries : bool, optional (default=False)
        Whether to sample queries from data (if no queries provided).
    objective_k : int, optional (default=100)
        Number of nearest neighbors for recall computation.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    angular : bool, optional (default=False)
        Whether to use spherical k-means (normalize centroids).

    Attributes
    ----------
    n_clusters_ : int
        Number of clusters (read-only).
    is_trained_ : bool
        Whether the model has been trained (read-only).
    iteration_stats : List[SuperKMeansIterationStats] or HierarchicalSuperKMeansIterationStats
        Statistics for each iteration (available after training).
    hierarchical_ : bool
        Whether hierarchical mode is being used (read-only).
    """

    def __init__(
        self,
        n_clusters: int,
        dimensionality: int,
        hierarchical: Optional[bool] = None,
        # Training parameters
        iters: int = 10,
        iters_mesoclustering: int = 3,
        iters_fineclustering: int = 5,
        iters_refinement: int = 0,
        sampling_fraction: Optional[float] = None,
        max_points_per_cluster: int = 256,
        n_threads: int = 0,
        seed: int = 42,
        use_blas_only: bool = False,
        # Convergence parameters
        tol: float = 1e-4,
        recall_tol: float = 0.005,
        early_termination: bool = True,
        sample_queries: bool = False,
        objective_k: int = 100,
        # Other parameters
        verbose: bool = False,
        angular: bool = False,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if dimensionality <= 0:
            raise ValueError("dimensionality must be positive")
        if sampling_fraction is not None and not 0.0 < sampling_fraction <= 1.0:
            raise ValueError("sampling_fraction must be in (0.0, 1.0]")

        self._n_clusters = n_clusters
        self._dimensionality = dimensionality

        # We defer the class resolution to the train method
        self._hierarchical_param = hierarchical
        self._hierarchical = None
        self._cpp_skmeans_obj = None
        self._config_params = {
            'iters': iters,
            'iters_mesoclustering': iters_mesoclustering,
            'iters_fineclustering': iters_fineclustering,
            'iters_refinement': iters_refinement,
            'sampling_fraction': sampling_fraction,
            'max_points_per_cluster': max_points_per_cluster,
            'n_threads': n_threads,
            'seed': seed,
            'use_blas_only': use_blas_only,
            'tol': tol,
            'recall_tol': recall_tol,
            'early_termination': early_termination,
            'sample_queries': sample_queries,
            'objective_k': objective_k,
            'verbose': verbose,
            'angular': angular,
        }

    @staticmethod
    def validate_numpy_array(
        data: np.ndarray,
        name: str,
        expected_dimensionality: Optional[int] = None,
    ) -> np.ndarray:
        """Validate a 2D float32 array and ensure it is C-contiguous."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        if data.dtype != np.float32:
            raise ValueError(f"{name} must have dtype float32, got {data.dtype}")
        if data.ndim != 2:
            raise ValueError(f"{name} must be 2-dimensional, got {data.ndim}")
        if expected_dimensionality is not None and data.shape[1] != expected_dimensionality:
            raise ValueError(
                f"{name} must have dimensionality {expected_dimensionality}, "
                f"got {data.shape[1]}"
            )
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        return data

    def train(
        self,
        data: NDArray[np.float32],
        queries: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.float32]:
        """
        Run k-means clustering to determine centroids.

        Parameters
        ----------
        data : ndarray of shape (n_samples, dimensionality), dtype=float32
            Training data. Must be C-contiguous (row-major).
        queries : ndarray of shape (n_queries, dimensionality), dtype=float32, optional
            Query vectors for recall-based quality monitoring.
            If provided, enables early termination by recall

        Returns
        -------
        centroids : ndarray of shape (n_clusters, dimensionality), dtype=float32
            The computed cluster centroids.

        Raises
        ------
        ValueError
            If data or queries have wrong shape, dtype, or memory layout.
        RuntimeError
            If the model has already been trained.
        """
        data = self.validate_numpy_array(data, "data", self._dimensionality)
        n_samples = data.shape[0]

        # Determine hierarchical mode if not explicitly set
        if self._hierarchical_param is None:
            self._hierarchical = n_samples > 100_000
        else:
            self._hierarchical = self._hierarchical_param

        if self._cpp_skmeans_obj is None:
            if self._hierarchical:
                config = _HierarchicalSuperKMeansConfigCpp()
                config.iters_mesoclustering = self._config_params['iters_mesoclustering']
                config.iters_fineclustering = self._config_params['iters_fineclustering']
                config.iters_refinement = self._config_params['iters_refinement']
            else:
                config = _SuperKMeansConfigCpp()
                config.iters = self._config_params['iters']

            if self._config_params['sampling_fraction'] is not None:
                config.sampling_fraction = self._config_params['sampling_fraction']
            config.max_points_per_cluster = self._config_params['max_points_per_cluster']
            config.n_threads = self._config_params['n_threads']
            config.seed = self._config_params['seed']
            config.use_blas_only = self._config_params['use_blas_only']
            config.tol = self._config_params['tol']
            config.recall_tol = self._config_params['recall_tol']
            config.early_termination = self._config_params['early_termination']
            config.sample_queries = self._config_params['sample_queries']
            config.objective_k = self._config_params['objective_k']
            config.verbose = self._config_params['verbose']
            config.angular = self._config_params['angular']

            if self._hierarchical:
                self._cpp_skmeans_obj = _HierarchicalSuperKMeansCpp(
                    self._n_clusters, self._dimensionality, config
                )
            else:
                self._cpp_skmeans_obj = _SuperKMeansCpp(
                    self._n_clusters, self._dimensionality, config
                )

        n_queries = 0
        if queries is not None:
            queries = self.validate_numpy_array(queries, "queries", self._dimensionality)
            n_queries = queries.shape[0]

        return self._cpp_skmeans_obj.train(data, queries, n_queries)

    def assign(
        self,
        vectors: NDArray[np.float32],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.uint32]:
        """
        Assign vectors to their nearest centroid using brute force.

        Works with any vectors, not just the training data.

        Parameters
        ----------
        vectors : ndarray of shape (n_vectors, dimensionality), dtype=float32
            Vectors to assign. Must be C-contiguous.
        centroids : ndarray of shape (n_clusters, dimensionality), dtype=float32
            Cluster centroids. Must be C-contiguous.

        Returns
        -------
        assignments : ndarray of shape (n_vectors,), dtype=uint32
            Cluster index (0 to n_clusters-1) for each vector.

        Raises
        ------
        ValueError
            If inputs have wrong shape, dtype, or memory layout.
        """
        vectors = self.validate_numpy_array(vectors, "vectors")
        centroids = self.validate_numpy_array(centroids, "centroids")

        if vectors.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"vectors and centroids must have same dimensionality, "
                f"got {vectors.shape[1]} and {centroids.shape[1]}"
            )

        return self._cpp_skmeans_obj.assign(vectors, centroids)

    # Alias for assign() to match FAISS API
    add = assign

    def assign_training_points(
        self,
        vectors: NDArray[np.float32],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.uint32]:
        """
        Fast assignment using trained state.

        Requires that the vectors are the same as those used in train().
        Leverages training assignments for faster assignment than brute force assign().

        Parameters
        ----------
        vectors : ndarray of shape (n_vectors, dimensionality), dtype=float32
            The training data. Must be the same data passed to train(). Must be C-contiguous.
        centroids : ndarray of shape (n_clusters, dimensionality), dtype=float32
            Cluster centroids. Must be C-contiguous.

        Returns
        -------
        assignments : ndarray of shape (n_vectors,), dtype=uint32
            Cluster index (0 to n_clusters-1) for each vector.

        Raises
        ------
        ValueError
            If inputs have wrong shape, dtype, or memory layout.
        RuntimeError
            If the model has not been trained yet.
        """
        vectors = self.validate_numpy_array(vectors, "vectors")
        centroids = self.validate_numpy_array(centroids, "centroids")

        if vectors.shape[1] != centroids.shape[1]:
            raise ValueError(
                f"vectors and centroids must have same dimensionality, "
                f"got {vectors.shape[1]} and {centroids.shape[1]}"
            )

        return self._cpp_skmeans_obj.assign_training_points(vectors, centroids)

    @property
    def n_clusters_(self) -> int:
        """Number of clusters (read-only)."""
        if self._cpp_skmeans_obj is None:
            return self._n_clusters
        return self._cpp_skmeans_obj.get_n_clusters()

    @property
    def is_trained_(self) -> bool:
        """Whether the model has been trained (read-only)."""
        if self._cpp_skmeans_obj is None:
            return False
        return self._cpp_skmeans_obj.is_trained()

    @property
    def hierarchical_(self) -> Optional[bool]:
        """Whether hierarchical mode is being used (read-only)."""
        return self._hierarchical

    @property
    def iteration_stats(self):
        """Statistics for each iteration (available after training if verbose=True)."""
        if self._cpp_skmeans_obj is None:
            return []
        return self._cpp_skmeans_obj.iteration_stats

    @property
    def hierarchical_iteration_stats(self):
        """Hierarchical iteration statistics (only for hierarchical mode)."""
        if self._cpp_skmeans_obj is None or not self._hierarchical:
            return None
        if hasattr(self._cpp_skmeans_obj, 'hierarchical_iteration_stats'):
            return self._cpp_skmeans_obj.hierarchical_iteration_stats
        return None

    def __repr__(self) -> str:
        """String representation of the SuperKMeans object."""
        hierarchical_str = f", hierarchical={self._hierarchical}" if self._hierarchical is not None else ""
        return (
            f"SuperKMeans(n_clusters={self._n_clusters}, "
            f"dimensionality={self._dimensionality}, "
            f"trained={self.is_trained_}{hierarchical_str})"
        )


__all__ = [
    "__version__",
    "SuperKMeans",
    "SuperKMeansIterationStats",
    "HierarchicalSuperKMeansIterationStats",
]
