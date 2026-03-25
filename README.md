<h1 align="center">
  Super K-Means
<div align="center">
    <a href="https://arxiv.org/pdf/2603.20009"><img src="https://img.shields.io/badge/Paper-arXiv-blue" alt="Paper" /></a>
    <img src="https://github.com/cwida/SuperKMeans/actions/workflows/ci.yml/badge.svg?cacheSeconds=3600" alt="License" />
    <a href="https://github.com/cwida/SuperKMeans/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cwida/SuperKMeans?cacheSeconds=3600" alt="License" /></a>
    <a href="https://github.com/cwida/SuperKMeans/stargazers"><img src="https://img.shields.io/github/stars/cwida/SuperKMeans" alt="GitHub stars" /></a>
</div>
</h1>
<h3 align="center">
  A super-fast clustering library for high-dimensional vector embeddings
</h3>

<p align="center">
        <img src="./benchmarks/results/plots/github_1.png" height=260 alt="SuperKMeans vs FAISS and Scikit Learn" style="{max-height: 250px}">
</p>

## Why Super K-Means?
- **100x faster clustering** than FAISS of vector embeddings (Cohere, OpenAI, MXBAI, CLIP, MiniLM).
- Index 10M embeddings of 1024 dimensions [**in less than a minute**](https://www.lkuffo.com/superkmeans/) on a single CPU.
- Faster **without compromising clustering quality**.
- Efficient in **CPUs** (ARM and x86) and **GPUs**.

## Our secret sauce
- Carefully interleaving GEMM routines and pruning kernels that **prune dimensions** efficiently
- In the benchmarks you see in the cover image, **all algorithms are clustering the same data**: No dimensionality reduction, no sampling, no early-termination.

## Usage
```py
from superkmeans import SuperKMeans

data = ... # Numpy 2D matrix
k = 1000
d = 768

kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d
)

# Run the clustering
centroids = kmeans.train(data) # 2D array with centroids (k x d) 

# Get assignments
assignments = kmeans.assign(data)
```

Then, you can use the `centroids` to create an IVF index for Vector Search, for example, in FAISS.

<details>

<summary>Usage in C++</summary>

```c++
#include <vector>
#include <cstddef>
#include "superkmeans/superkmeans.h"
#include "superkmeans/hierarchical_superkmeans.h"

int main(int argc, char* argv[]) {
    std::vector<float> data; // Fill
    size_t n = 1000000;
    size_t k = 10000;
    size_t d = 768;

    auto kmeans = skmeans::SuperKMeans(k, d);

    // Or Hierarchical Super K-Means for extreme performance:
    // auto kmeans = skmeans::HierarchicalSuperKMeans(k, d);
    
    // Run the clustering
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    
    // Assign points
    std::vector<uint32_t> assignments = kmeans.Assign(data.data(), centroids.data(), n, k);
}

```

</details>

Check our [examples](./examples/) for fully working examples in Python and C++.

## Documentation
Check [our wiki](https://github.com/cwida/SuperKMeans/wiki/Documentation) for advanced usage.

## Installation
```sh
pip install superkmeans
```

> [!TIP]
> For maximum performance, we recommend compiling from source.

<details>

<summary>Compiling Python Bindings from source</summary>

### Prerequisites
- Clang 17, CMake 3.26
- OpenMP
- A BLAS implementation
- Python 3 (only for Python bindings)

```bash
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
git submodule update --init
pip install .

# Run plug-and-play example
python ./examples/simple_clustering.py

# Set a value for n, d and k
python ./examples/simple_clustering.py 200000 1536 1000
```

</details> 

<details>

<summary>Compiling C++ library from source</summary>

### Prerequisites
- Clang 17, CMake 3.26
- OpenMP
- A BLAS implementation
- Python 3 (only for Python bindings)

```bash
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
git submodule update --init

# Set proper path to clang if needed
export CXX="/usr/bin/clang++-18" 

# Compile
cmake .
make examples

# Run plug-and-play example
cd examples
./simple_clustering.out

# Set a value for n, d and k
./simple_clustering.out 100000 1536 1000
```
</details> 


For a more comprehensive installation and compilation guide, check [INSTALL.md](./INSTALL.md).

## Getting the Best Performance
Check [INSTALL.md](./INSTALL.md).

## Roadmap
We are actively developing Super K-Means and accepting contributions! Check [CONTRIBUTING.md](./CONTRIBUTING.md)

## Benchmarking
To run our benchmark suite in C++, refer to [BENCHMARKING.md](./BENCHMARKING.md).