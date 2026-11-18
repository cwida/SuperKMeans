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

### Quantized clustering

Cluster in a compressed domain (`sq8`, `lvq4`, or `rabitq`) for faster clustering with a lower memory footprint:

```python
kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d,
    quantizer="sq8"
)

centroids = kmeans.train(data)

# Assign in the compressed domain
assignments = kmeans.quantized_assign(data, centroids)
```

### e-means clustering

[e-means](https://arxiv.org/abs/2611.09501) (mini-batch k-means with EMA centroid updates) reaches
k-means quality in 2 epochs over the data, accelerated with SuperKMeans' pruning:

```python
from superkmeans import SuperEMeans

emeans = SuperEMeans(n_clusters=k, dimensionality=d)  # also takes quantizer="sq8", etc.
centroids = emeans.train(data)

# Reuse the training state for a fast (pruned) assignment of the training points
assignments = emeans.assign_training_points(data, centroids)
```

## API Reference

### SuperKMeans Class

#### Constructor Parameters

- `n_clusters` (int): Number of clusters to create
- `dimensionality` (int): Number of dimensions in the data
- `quantizer` (str, default="f32"): Quantization scheme: "f32", "sq8", "lvq4", or "rabitq"
- `hierarchical` (bool, default=None): Use hierarchical clustering. If None, auto-enabled for n > 100,000
- `iters` (int, default=10): Number of k-means iterations (flat mode)
- `iters_mesoclustering` (int, default=3): Number of mesoclustering iterations (hierarchical mode)
- `iters_fineclustering` (int, default=5): Number of fineclustering iterations (hierarchical mode)
- `iters_refinement` (int, default=0): Number of refinement iterations (hierarchical mode)
- `sampling_fraction` (float, default=None): Fraction of data to sample (0.0, 1.0]. If None, uses 0.3 (flat) or 1.0 (hierarchical)
- `max_points_per_cluster` (int, default=256): Maximum points per cluster to sample
- `n_threads` (int, default=0): Number of threads (0 = use all)
- `seed` (int, default=42): Random seed for reproducibility
- `use_blas_only` (bool, default=False): Disable pruning, use BLAS only
- `full_precision_final_centroids` (bool, default=False): Recompute final centroids in full precision (quantized only)
- `tol` (float, default=1e-4): Tolerance for shift-based early stopping
- `recall_tol` (float, default=0.005): Tolerance for recall-based early stopping
- `early_termination` (bool, default=True): Enable early stopping
- `sample_queries` (bool, default=False): Sample queries from data
- `objective_k` (int, default=100): Number of neighbors for recall
- `verbose` (bool, default=False): Print progress information
- `angular` (bool, default=False): Use spherical k-means
- `unrotate_centroids` (bool, default=True): Map centroids back to the input domain before returning them. Forced to `False` by `train(overwrite_input=True)`

#### Methods

**`train(data, queries=None, overwrite_input=False)`**

Run k-means clustering to compute centroids.

- **Parameters:**
  - `data` (ndarray): Shape (n_samples, dimensionality), dtype float32
  - `queries` (ndarray, optional): Shape (n_queries, dimensionality), dtype float32
  - `overwrite_input` (bool, default=False): Rotate `data` in place instead of allocating a
    rotated copy, halving peak memory. See below.
- **Returns:** `centroids` (ndarray): Shape (n_clusters, dimensionality), dtype float32

##### Reducing peak memory with `overwrite_input`

Training rotates the data before clustering. By default that rotation goes into a fresh
`n × d` float32 buffer, so peak memory is roughly twice the input. `overwrite_input=True`
rotates the caller's array in place instead:

```python
centroids = kmeans.train(data, overwrite_input=True)   # data is now rotated
labels = kmeans.assign_training_points(data, centroids)
```

`data` is overwritten and is **not** restored, so it must be a writeable, C-contiguous float32
array — no conversion is performed, and a mismatch raises rather than silently copying. 
It also requires `sampling_fraction == 1.0`, which is otherwise
applied with a warning, since sampling gathers scattered rows and cannot be done in place.

The returned centroids are rotated too — `unrotate_centroids` is forced to `False` (data and
centroids stay in the same domain). 

**`assign(vectors, centroids)`**

Assign vectors to nearest centroids in full precision. Can be called before `train()`. Aliased as `add()`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

**`quantized_assign(vectors, centroids)`**

Assign vectors in the quantizer's compressed domain. Standalone — fits a fresh quantizer on the input vectors (reusing no trained state), so it can be called before `train()`. Equivalent to `assign()` when `quantizer="f32"`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

**`assign_training_points(vectors, centroids)`**

Fast assignment of the training data, reusing state from `train()`. `vectors` must be the same data passed to `train()`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

**`rotate(vectors)` / `unrotate(vectors)`**

Training rotates the data before clustering. `rotate()` brings vectors *into* that domain,
`unrotate()` brings them back out. Both require a trained model and return a new array.

Use `rotate()` after `train(overwrite_input=True)`, since the data and
centroids are left in the rotated domain:

```python
centroids = kmeans.train(data, overwrite_input=True)
labels = kmeans.assign_training_points(data, centroids)
hits = kmeans.assign(kmeans.rotate(queries), centroids)
```

- **Parameters:** `vectors` (ndarray): Shape (n, dimensionality), dtype float32
- **Returns:** ndarray of the same shape

#### Properties

- `n_clusters_` (int): Number of clusters (read-only)
- `is_trained_` (bool): Whether model has been trained (read-only)
- `hierarchical_` (bool): Whether hierarchical mode is used (read-only)
- `quantizer_` (str): Quantization scheme in use (read-only)
- `iteration_stats` (list): List of `SuperKMeansIterationStats` objects
- `hierarchical_iteration_stats` (`HierarchicalSuperKMeansIterationStats`): Per-phase statistics (hierarchical mode only)
- `state` (`SuperKMeansState`): How training was carried out, or `None` before training
- `quantization_params` (dict): Global parameters of the fitted quantizer, or `None` before
  training — see the table below
- `quantized_data` (ndarray): Read-only view of the encoded training vectors, shape
  `(n_encoded, code_size)`, dtype uint8. `None` for `quantizer="f32"`
- `sampled_indices` (ndarray): Read-only view mapping encoded row `i` to original row
  `sampled_indices[i]`. `None` when `sampling_fraction == 1.0`, where encoded row `i` *is* row `i`


##### Code layout per scheme

`quantization_params` is enough to interpret `quantized_data`. 
Note the codes encode the **rotated** vectors, so use `unrotate()` to get back to the
input domain.

| scheme | `code_size` | layout per vector | `quantization_params` |
| --- | --- | --- | --- |
| `f32` | — | no encoded buffer; the data is clustered directly | `{}` |
| `sq8` | `d` | `[d u8 bytes]` | `{"base", "scale", "inv_scale"}` — global |
| `lvq4` | `d/2 + 8` | `[d/2 packed u4x2][float scale][float bias]` | `{}` — scale/bias are **per-vector**, inside each code |
| `rabitq` | `(d+7)//8 + 8` | `[(d+7)//8 sign bytes][float or_minus_c_l2sqr][float dp_multiplier]` | `{"centroid"` (d floats), `"binary_bytes"}` |

For `lvq4` the nibbles are packed two per byte, even dimension in the low nibble and odd in the
high one. Reconstructing from sq8, where the parameters are global:

```python
p = kmeans.quantization_params
approx = kmeans.quantized_data.astype(np.float32) * p["inv_scale"] + p["base"]
original = kmeans.unrotate(approx)
```

### SuperEMeans Class

e-means on the SuperKMeans machinery: mini-batches with EMA centroid updates, a cosine step schedule
and dead-cluster revival. The first epoch is GEMM-only, and later epochs prune, seeded by the previous
epoch's labels. Flat only (no hierarchical mode).

#### Constructor Parameters

- `n_clusters` (int): Number of clusters
- `dimensionality` (int): Number of dimensions
- `quantizer` (str, default="f32"): `"f32"`, `"sq8"`, `"lvq4"`, or `"rabitq"`
- `epochs` (int, default=2): Number of passes over the data
- `batch_size` (int, default=None): Mini-batch size. None uses the e-means batch-size law
- `initial_step` (float, default=None): Initial EMA step in (0, 1]. None uses the e-means step-size law
- `n_threads`, `seed`, `use_blas_only`, `full_precision_final_centroids`, `verbose`, `angular`,
  `unrotate_centroids`: as in `SuperKMeans`

#### Methods

- `train(data, overwrite_input=False)`: Returns `(n_clusters, d)` centroids. No `queries`
  (e-means runs a fixed number of epochs, with no early termination)
- `assign`, `quantized_assign`, `assign_training_points`, `rotate`, `unrotate`: as in `SuperKMeans`

#### Properties

- `batch_size_` (int): Mini-batch size used in training (None before training)
- `initial_step_` (float): Initial EMA step used in training (None before training)
- `iteration_stats`: One `SuperKMeansIterationStats` per epoch
- `is_trained_`, `quantizer_`, `state`, `quantization_params`, `quantized_data`,
  `sampled_indices`: as in `SuperKMeans`

### SuperKMeansState Class

Read-only record of how training was carried out.

- `trained` (bool): Whether training completed
- `trained_in_place` (bool): Whether training rotated the caller's buffer instead of a copy
- `training_data_rotated` (bool): Whether the training data is in the rotated domain
- `code_size` (int): Elements per encoded vector (bytes for the quantized schemes)
- `n_encoded` (int): Number of encoded vectors

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

### HierarchicalSuperKMeansIterationStats Class

Per-phase statistics for hierarchical clustering.

**Attributes:**
- `mesoclustering_iteration_stats` (list): `SuperKMeansIterationStats` for the mesoclustering phase
- `fineclustering_iteration_stats` (list): `SuperKMeansIterationStats` for the fineclustering phase
- `refinement_iteration_stats` (list): `SuperKMeansIterationStats` for the refinement phase

## Examples

See the `examples/` directory for complete examples:
- `simple_clustering.py [n] [d] [k]`
- `quantized_clustering.py [quantizer] [n] [d] [k]`
- `emeans_clustering.py [n] [d] [k]`

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest python/tests/ -v

```
