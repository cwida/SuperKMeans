# SuperKMeans Python Bindings

Python bindings for SuperKMeans

### Install from source

```bash
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
pip install .
```

## Quick Start

```python
import numpy as np
from superkmeans import SuperKMeans

n = 100000
d = 512
k = 1000

data = np.random.randn(n, d).astype(np.float32)

kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d
)

# Compute centroids
centroids = kmeans.train(data)

# Assign points to clusters
assignments = kmeans.assign(data, centroids)
```

## API Reference

### SuperKMeans Class

#### Constructor Parameters

- `n_clusters` (int): Number of clusters to create
- `dimensionality` (int): Number of dimensions in the data
- `iters` (int, default=10): Number of k-means iterations
- `sampling_fraction` (float, default=0.3): Fraction of data to sample (0.0, 1.0]
- `max_points_per_cluster` (int, default=256): Maximum points per cluster to sample
- `n_threads` (int, default=0): Number of threads (0 = use all)
- `seed` (int, default=42): Random seed for reproducibility
- `use_blas_only` (bool, default=False): Disable pruning, use BLAS only
- `tol` (float, default=1e-4): Tolerance for shift-based early stopping
- `recall_tol` (float, default=0.005): Tolerance for recall-based early stopping
- `early_termination` (bool, default=True): Enable early stopping
- `sample_queries` (bool, default=False): Sample queries from data
- `objective_k` (int, default=100): Number of neighbors for recall
- `verbose` (bool, default=False): Print progress information
- `angular` (bool, default=False): Use spherical k-means

#### Methods

**`train(data, queries=None)`**

Run k-means clustering to compute centroids.

- **Parameters:**
  - `data` (ndarray): Shape (n_samples, dimensionality), dtype float32
  - `queries` (ndarray, optional): Shape (n_queries, dimensionality), dtype float32
- **Returns:** `centroids` (ndarray): Shape (n_clusters, dimensionality), dtype float32

**`assign(vectors, centroids)`**

Assign vectors to nearest centroids.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

#### Properties

- `n_clusters_` (int): Number of clusters (read-only)
- `is_trained_` (bool): Whether model has been trained (read-only)
- `iteration_stats` (list): List of `SuperKMeansIterationStats` objects

### SuperKMeansIterationStats Class

Statistics for a single iteration.

**Attributes:**
- `iteration` (int): Iteration number (1-indexed)
- `objective` (float): Within-cluster sum of squares (WCSS)
- `shift` (float): Average squared centroid shift
- `split` (int): Number of clusters split (empty cluster handling)
- `recall` (float): Recall@k value (0.0 to 1.0)
- `not_pruned_pct` (float): Percentage of vectors not pruned
- `partial_d` (int): Dimensions used for partial distance (d')
- `is_gemm_only` (bool): Whether iteration used BLAS-only

## Examples

See the `examples/` directory for complete examples:
- `simple_clustering.py [n] [d] [k]`

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest python/tests/ -v

```
