# Super K-Means examples

These examples demonstrate how to use our clustering library.

## C++: Simple example

**File:** [`simple_clustering.cpp`](./simple_clustering.cpp)   
**Compile:** `make simple_clustering.out`   
**Run (assuming `pwd` is `./examples`):** `./simple_clustering.out <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create

## C++: Hierarchical Clustering example

Extremely fast while preserving quality. Recommended when n > 100K.

**File:** [`hierarchical_clustering.cpp`](./hierarchical_clustering.cpp)   
**Compile:** `make hierarchical_clustering.out`   
**Run (assuming `pwd` is `./examples`):** `./hierarchical_clustering.out <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create

## Python: Simple example

**File:** [`simple_clustering.py`](./simple_clustering.py)    
**Needs:** `pip install scikit-learn numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./simple_clustering.py <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create


## Python: Quantized Clustering example

Faster clustering and low memory footprint while preserving quality. Choose `sq8`, `lvq4`, or `rabitq`.

**File:** [`quantized_clustering.py`](./quantized_clustering.py)    
**Needs:** `pip install scikit-learn numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./quantized_clustering.py <quantizer> [n] [d] [k]`   
**Parameters**:   
- `quantizer`: Quantization scheme: `f32`, `sq8`, `lvq4`, or `rabitq`.   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create


## Python: e-means example

[e-means](https://arxiv.org/abs/2611.09501): mini-batch k-means with EMA centroid updates, 2 epochs over the data, accelerated with our pruning.

**File:** [`emeans_clustering.py`](./emeans_clustering.py)    
**Needs:** `pip install scikit-learn numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./emeans_clustering.py <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create


## Python: Reading .hdf5 example

**File:** [`hdf5_clustering.py`](./hdf5_clustering.py)    
**Needs:** `pip install h5py numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./hdf5_clustering.py <data_path> [<k>] [<quantizer>]`   
**Parameters**:   
- `data_path`: Path to your `.hdf5` file. We assume the `.hdf5` file has a `train` dataset with the vector embeddings.   
- [Optional] `k`: Number of clusters to create. Default: $4 * \sqrt{n}$.   
- [Optional] `quantizer`: Quantization scheme: `f32`, `sq8`, `lvq4`, or `rabitq`. Default: `f32`.   


