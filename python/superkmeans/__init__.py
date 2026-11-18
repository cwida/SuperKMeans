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
    SuperKMeansSQ8 as _SuperKMeansSQ8Cpp,
    SuperKMeansLVQ4 as _SuperKMeansLVQ4Cpp,
    SuperKMeansRabitQ as _SuperKMeansRabitQCpp,
    SuperKMeansConfig as _SuperKMeansConfigCpp,
    SuperKMeansIterationStats,
    HierarchicalSuperKMeans as _HierarchicalSuperKMeansCpp,
    HierarchicalSuperKMeansSQ8 as _HierarchicalSuperKMeansSQ8Cpp,
    HierarchicalSuperKMeansLVQ4 as _HierarchicalSuperKMeansLVQ4Cpp,
    HierarchicalSuperKMeansRabitQ as _HierarchicalSuperKMeansRabitQCpp,
    HierarchicalSuperKMeansConfig as _HierarchicalSuperKMeansConfigCpp,
    HierarchicalSuperKMeansIterationStats,
    SuperEMeans as _SuperEMeansCpp,
    SuperEMeansSQ8 as _SuperEMeansSQ8Cpp,
    SuperEMeansLVQ4 as _SuperEMeansLVQ4Cpp,
    SuperEMeansRabitQ as _SuperEMeansRabitQCpp,
    SuperEMeansConfig as _SuperEMeansConfigCpp,
)

_QUANTIZERS = ("f32", "sq8", "lvq4", "rabitq")
_QUANTIZER_MAP = {
    "f32": (_SuperKMeansCpp, _HierarchicalSuperKMeansCpp, _SuperEMeansCpp),
    "sq8": (_SuperKMeansSQ8Cpp, _HierarchicalSuperKMeansSQ8Cpp, _SuperEMeansSQ8Cpp),
    "lvq4": (_SuperKMeansLVQ4Cpp, _HierarchicalSuperKMeansLVQ4Cpp, _SuperEMeansLVQ4Cpp),
    "rabitq": (_SuperKMeansRabitQCpp, _HierarchicalSuperKMeansRabitQCpp, _SuperEMeansRabitQCpp),
}


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
    quantizer : str, optional (default="f32")
        Quantization scheme for clustering. One of "f32" (full precision),
        "sq8" (8-bit scalar), "lvq4" (4-bit LVQ), or "rabitq" (RaBitQ). The
        quantized schemes cluster in a compressed domain for speed/memory.
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
    unrotate_centroids : bool, optional (default=True)
        Whether to map the centroids back to the input domain before returning them.
        Forced to False by train(overwrite_input=True), which leaves the data rotated and so
        must return centroids in that same domain.

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
        quantizer: str = "f32",
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
        # Quantization parameters
        full_precision_final_centroids: bool = False,
        # Convergence parameters
        tol: float = 1e-4,
        recall_tol: float = 0.005,
        early_termination: bool = True,
        sample_queries: bool = False,
        objective_k: int = 100,
        # Other parameters
        verbose: bool = False,
        angular: bool = False,
        unrotate_centroids: bool = True,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if dimensionality <= 0:
            raise ValueError("dimensionality must be positive")
        if sampling_fraction is not None and not 0.0 < sampling_fraction <= 1.0:
            raise ValueError("sampling_fraction must be in (0.0, 1.0]")
        if quantizer not in _QUANTIZERS:
            raise ValueError(f"quantizer must be one of {_QUANTIZERS}, got {quantizer!r}")

        self._n_clusters = n_clusters
        self._dimensionality = dimensionality
        self._quantizer = quantizer

        # We defer the class resolution to the train method
        self._hierarchical_param = hierarchical
        self._hierarchical = None
        self._cpp_skmeans_obj = None
        self._assign_only_obj = None
        self._quantized_assign_only_obj = None
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
            'full_precision_final_centroids': full_precision_final_centroids,
            'tol': tol,
            'recall_tol': recall_tol,
            'early_termination': early_termination,
            'sample_queries': sample_queries,
            'objective_k': objective_k,
            'verbose': verbose,
            'angular': angular,
            'unrotate_centroids': unrotate_centroids,
        }

    @staticmethod
    def validate_numpy_array(
        data: np.ndarray,
        name: str,
        expected_dimensionality: Optional[int] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        """Validate a 2D float32 array and ensure it is C-contiguous.

        With overwrite=True the array is returned as-is or rejected: silently substituting a
        contiguous copy would defeat the point of overwriting the caller's buffer.
        """
        if not isinstance(data, np.ndarray):
            if overwrite:
                raise ValueError(f"{name} must be a NumPy array to be overwritten in place")
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
            if overwrite:
                raise ValueError(
                    f"{name} must be C-contiguous to be overwritten in place; pass "
                    f"np.ascontiguousarray({name}) if an extra copy is acceptable"
                )
            data = np.ascontiguousarray(data)
        if overwrite and not data.flags["WRITEABLE"]:
            raise ValueError(f"{name} must be writeable to be overwritten in place")
        return data

    def train(
        self,
        data: NDArray[np.float32],
        queries: Optional[NDArray[np.float32]] = None,
        overwrite_input: bool = False,
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
        overwrite_input : bool, optional (default=False)
            Rotate `data` in place instead of allocating a rotated copy, halving peak memory.
            `data` is overwritten with its rotated form.
            Requires sampling_fraction == 1.0, which is otherwise applied with a warning.

            `unrotate_centroids` is forced to False so that data and centroids stay 
            in the same domain and remain directly comparable, so
            both `assign()` and `assign_training_points()` work as usual.

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
        data = self.validate_numpy_array(
            data, "data", self._dimensionality, overwrite=overwrite_input
        )
        n_samples = data.shape[0]

        # Determine hierarchical mode if not explicitly set
        if self._hierarchical_param is None:
            self._hierarchical = n_samples > 100_000
        else:
            self._hierarchical = self._hierarchical_param

        if self._cpp_skmeans_obj is None:
            self._cpp_skmeans_obj = self._create_engine()

        n_queries = 0
        if queries is not None:
            queries = self.validate_numpy_array(queries, "queries", self._dimensionality)
            n_queries = queries.shape[0]

        if overwrite_input:
            return self._cpp_skmeans_obj.train_in_place(data, queries, n_queries)
        return self._cpp_skmeans_obj.train(data, queries, n_queries)

    def _configure(self, config):
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
        config.unrotate_centroids = self._config_params['unrotate_centroids']

        if self._quantizer != "f32":
            config.full_precision_final_centroids = (
                self._config_params['full_precision_final_centroids']
            )
        return config

    def _create_engine(self):
        if self._hierarchical:
            config = _HierarchicalSuperKMeansConfigCpp()
            config.iters_mesoclustering = self._config_params['iters_mesoclustering']
            config.iters_fineclustering = self._config_params['iters_fineclustering']
            config.iters_refinement = self._config_params['iters_refinement']
        else:
            config = _SuperKMeansConfigCpp()
            config.iters = self._config_params['iters']
        cpp_cls = _QUANTIZER_MAP[self._quantizer][1 if self._hierarchical else 0]
        return cpp_cls(self._n_clusters, self._dimensionality, self._configure(config))

    def _assign_engine(self):
        """C++ object used to run exact float32 assign().

        Exact assign() needs no trained state and is identical across the
        flat/hierarchical/quantized variants, so it can run before train():
        reuse the trained object if present, otherwise lazily build a flat
        float32 engine.
        """
        if self._cpp_skmeans_obj is not None:
            return self._cpp_skmeans_obj
        if self._assign_only_obj is None:
            self._assign_only_obj = _SuperKMeansCpp(self._n_clusters, self._dimensionality)
        return self._assign_only_obj

    def _quantized_assign_engine(self):
        """C++ object used to run quantized_assign().

        quantized_assign is standalone (it fits a fresh quantizer on the input and
        reuses no trained state), so it can run before train(): reuse the trained
        object if present, otherwise lazily build a flat engine configured with the
        chosen quantizer.
        """
        if self._cpp_skmeans_obj is not None:
            return self._cpp_skmeans_obj
        if self._quantizer == "f32":
            return self._assign_engine()
        if self._quantized_assign_only_obj is None:
            config = _SuperKMeansConfigCpp()
            config.seed = self._config_params['seed']
            self._quantized_assign_only_obj = _QUANTIZER_MAP[self._quantizer][0](
                self._n_clusters, self._dimensionality, config
            )
        return self._quantized_assign_only_obj

    def assign(
        self,
        vectors: NDArray[np.float32],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.uint32]:
        """
        Assign vectors to their nearest centroid using exact float32 brute force.

        Works with any vectors, not just the training data, and can be called
        before train() (exact assignment needs no trained state).

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

        return self._assign_engine().assign(vectors, centroids)

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

        if self._cpp_skmeans_obj is None:
            raise RuntimeError("assign_training_points requires train() to be called first")

        return self._cpp_skmeans_obj.assign_training_points(vectors, centroids)

    def quantized_assign(
        self,
        vectors: NDArray[np.float32],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.uint32]:
        """
        Assign vectors to their nearest centroid in the quantizer's compressed domain.

        Standalone: fits a fresh quantizer on the input vectors, encodes both the vectors
        and the centroids, then searches in the quantized domain (it does not reuse trained
        state). Works with any vectors, not just the training data, and can be called before
        train(). For quantizer="f32" this is equivalent to assign().

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

        return self._quantized_assign_engine().quantized_assign(vectors, centroids)

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
    def quantizer_(self) -> str:
        """The quantization scheme in use (read-only)."""
        return self._quantizer

    @property
    def iteration_stats(self):
        """Statistics for each iteration (available after training if verbose=True)."""
        if self._cpp_skmeans_obj is None:
            return []
        return self._cpp_skmeans_obj.iteration_stats

    @property
    def state(self):
        """How training was carried out, or None before training (read-only)."""
        if self._cpp_skmeans_obj is None:
            return None
        return self._cpp_skmeans_obj.state

    @property
    def quantization_params(self) -> Optional[dict]:
        """Global parameters of the fitted quantizer, or None before training.

        Keys depend on the scheme: "sq8" gives {"base", "scale", "inv_scale"}. Schemes without
        global parameters (e.g. "lvq4", which is per-vector) give an empty dict.
        """
        if self._cpp_skmeans_obj is None:
            return None
        return self._cpp_skmeans_obj.quantization_params

    @property
    def quantized_data(self) -> Optional[NDArray[np.uint8]]:
        """Read-only view of the encoded training vectors, shape (n_encoded, code_size).

        None before training, and None for quantizer="f32".
        """
        if self._cpp_skmeans_obj is None:
            return None
        codes = self._cpp_skmeans_obj.quantized_data
        if codes is None:
            return None
        codes.setflags(write=False)
        return codes

    @property
    def sampled_indices(self) -> Optional[NDArray[np.uint64]]:
        """Read-only view mapping encoded row i to original row sampled_indices[i].

        None when no sampling was applied (sampling_fraction == 1.0), where encoded row i is
        simply original row i.
        """
        if self._cpp_skmeans_obj is None:
            return None
        indices = self._cpp_skmeans_obj.sampled_indices
        if indices is None:
            return None
        indices.setflags(write=False)
        return indices

    def rotate(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the trained rotation, bringing vectors into the model's domain.

        Useful for query vectors: after train(overwrite_input=True) the data and centroids live in
        the rotated domain, so new vectors must be rotated before being compared against them.
        """
        if self._cpp_skmeans_obj is None:
            raise RuntimeError("rotate() requires a trained model")
        vectors = self.validate_numpy_array(vectors, "vectors", self._dimensionality)
        return self._cpp_skmeans_obj.rotate(vectors)

    def unrotate(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Undo the trained rotation, bringing vectors back to the input domain."""
        if self._cpp_skmeans_obj is None:
            raise RuntimeError("unrotate() requires a trained model")
        vectors = self.validate_numpy_array(vectors, "vectors", self._dimensionality)
        return self._cpp_skmeans_obj.unrotate(vectors)

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
        quantizer_str = f", quantizer={self._quantizer!r}" if self._quantizer != "f32" else ""
        return (
            f"SuperKMeans(n_clusters={self._n_clusters}, "
            f"dimensionality={self._dimensionality}, "
            f"trained={self.is_trained_}{quantizer_str}{hierarchical_str})"
        )


class SuperEMeans(SuperKMeans):
    """
    e-means: mini-batch k-means with EMA centroid updates (https://arxiv.org/abs/2611.09501),
    accelerated with SuperKMeans' pruning from the second epoch on.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to create.
    dimensionality : int
        Number of dimensions in the data.
    quantizer : str, optional (default="f32")
        Quantization scheme: "f32", "sq8", "lvq4", or "rabitq".
    epochs : int, optional (default=2)
        Number of passes over the data.
    batch_size : int, optional (default=None)
        Mini-batch size. If None, set by the e-means batch-size law from n, n_clusters and epochs.
    initial_step : float, optional (default=None)
        Initial EMA step size in (0.0, 1.0]. If None, set by the e-means step-size law.
    n_threads : int, optional (default=0)
        Number of CPU threads to use. 0 means use all available.
    seed : int, optional (default=42)
        Random seed for reproducibility.
    use_blas_only : bool, optional (default=False)
        Use GEMM-only computation without pruning.
    full_precision_final_centroids : bool, optional (default=False)
        Compute the final centroids from the raw float data (quantized schemes only).
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    angular : bool, optional (default=False)
        Whether to use spherical e-means (normalize centroids).
    unrotate_centroids : bool, optional (default=True)
        Whether to map the centroids back to the input domain before returning them.
        Forced to False by train(overwrite_input=True).

    Attributes
    ----------
    batch_size_ : int or None
        Mini-batch size used in training (None before training).
    initial_step_ : float or None
        Initial EMA step size used in training (None before training).
    iteration_stats : List[SuperKMeansIterationStats]
        Statistics for each epoch (available after training).
    """

    def __init__(
        self,
        n_clusters: int,
        dimensionality: int,
        quantizer: str = "f32",
        epochs: int = 2,
        batch_size: Optional[int] = None,
        initial_step: Optional[float] = None,
        n_threads: int = 0,
        seed: int = 42,
        use_blas_only: bool = False,
        full_precision_final_centroids: bool = False,
        verbose: bool = False,
        angular: bool = False,
        unrotate_centroids: bool = True,
    ):
        super().__init__(
            n_clusters,
            dimensionality,
            hierarchical=False,
            quantizer=quantizer,
            iters=epochs,
            n_threads=n_threads,
            seed=seed,
            use_blas_only=use_blas_only,
            full_precision_final_centroids=full_precision_final_centroids,
            verbose=verbose,
            angular=angular,
            unrotate_centroids=unrotate_centroids,
        )
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if initial_step is not None and not 0.0 < initial_step <= 1.0:
            raise ValueError("initial_step must be in (0.0, 1.0]")
        self._config_params['batch_size'] = batch_size
        self._config_params['initial_step'] = initial_step
        self._hierarchical = False

    def _create_engine(self):
        config = _SuperEMeansConfigCpp()
        config.iters = self._config_params['iters']
        config.batch_size = self._config_params['batch_size'] or 0
        config.initial_step = self._config_params['initial_step'] or 0.0
        cpp_cls = _QUANTIZER_MAP[self._quantizer][2]
        return cpp_cls(self._n_clusters, self._dimensionality, self._configure(config))

    def train(
        self,
        data: NDArray[np.float32],
        overwrite_input: bool = False,
    ) -> NDArray[np.float32]:
        """
        Run e-means clustering to determine centroids.

        Parameters
        ----------
        data : ndarray of shape (n_samples, dimensionality), dtype=float32
            Training data. Must be C-contiguous (row-major).
        overwrite_input : bool, optional (default=False)
            Rotate `data` in place instead of allocating a rotated copy, halving peak memory.
            `data` is overwritten with its rotated form, and the centroids are returned in that
            same domain.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, dimensionality), dtype=float32
            The computed cluster centroids.

        Raises
        ------
        ValueError
            If data has wrong shape, dtype, or memory layout.
        RuntimeError
            If the model has already been trained.
        """
        data = self.validate_numpy_array(
            data, "data", self._dimensionality, overwrite=overwrite_input
        )
        if self._cpp_skmeans_obj is None:
            self._cpp_skmeans_obj = self._create_engine()
        if overwrite_input:
            return self._cpp_skmeans_obj.train_in_place(data)
        return self._cpp_skmeans_obj.train(data)

    @property
    def batch_size_(self) -> Optional[int]:
        """Mini-batch size used in training, or None before training (read-only)."""
        if self._cpp_skmeans_obj is None or not self.is_trained_:
            return None
        return self._cpp_skmeans_obj.batch_size

    @property
    def initial_step_(self) -> Optional[float]:
        """Initial EMA step size used in training, or None before training (read-only)."""
        if self._cpp_skmeans_obj is None or not self.is_trained_:
            return None
        return self._cpp_skmeans_obj.initial_step

    def __repr__(self) -> str:
        """String representation of the SuperEMeans object."""
        quantizer_str = f", quantizer={self._quantizer!r}" if self._quantizer != "f32" else ""
        return (
            f"SuperEMeans(n_clusters={self._n_clusters}, "
            f"dimensionality={self._dimensionality}, "
            f"epochs={self._config_params['iters']}, "
            f"trained={self.is_trained_}{quantizer_str})"
        )


__all__ = [
    "__version__",
    "SuperKMeans",
    "SuperEMeans",
    "SuperKMeansIterationStats",
    "HierarchicalSuperKMeansIterationStats",
]
