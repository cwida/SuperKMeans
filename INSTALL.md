# Installation

## Python 
With `pip`:
```sh
pip install superkmeans
```

## C++
With CMake `FetchContent`. There are no dependencies to install: CMake fetches and builds everything it needs.

```cmake
FetchContent_Declare(
    superkmeans
    GIT_REPOSITORY https://github.com/cwida/superkmeans
)
FetchContent_MakeAvailable(superkmeans)

target_link_libraries(myapp PRIVATE superkmeans)
```

CMake options:
- `-DSKMEANS_MARCH`: `-march` value to use during SuperKMeans compilation (default=`native`).
- `-DSKMEANS_EXECUTOR`: the thread pool behind the parallel loops: `forkunion` (default), `openmp` or `serial`. You can also pass your own pool through `SuperKMeansConfig::executor`.
- `-DSKMEANS_GEMM`: the matrix multiplication backend: `auto` (default: Apple Accelerate on macOS, Eigen elsewhere), `eigen`, `accelerate` or `blas` (see [Using an external BLAS](#using-an-external-blas-optional)).
- `-DSKMEANS_SKIP_FFTW`: don't look for FFTW (optional; used for the rotation when found).

## From source

#### Super K-Means on CPU needs:
- Clang 17 or GCC 13 (MSVC 2022 on Windows), CMake 3.26
- Python 3 (only for Python bindings)

Once you have these requirements, you can install Python Bindings or compile our [C++ example](./examples/) code.

<details>
<summary> <b> Installing Python Bindings from source </b></summary>

```sh
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
git submodule update --init

# Create a venv if needed
python -m venv ./venv
source venv/bin/activate

pip install .
```
</details>

<details>
<summary> <b> Compiling C++ Library from source </b></summary>

```sh
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
git submodule update --init

# Set proper path to clang if needed
export CXX="/usr/bin/clang++-18" 

# Compile
cmake .
make
```
</details>

## Step by Step
* [Installing Clang](#installing-clang)
* [Installing CMake](#installing-cmake)
* [Using an external BLAS (optional)](#using-an-external-blas-optional)
* [Using OpenMP (optional)](#using-openmp-optional)
* [Troubleshooting](#troubleshooting)

## Installing Clang
We recommend LLVM
### Linux
```sh 
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 18
```

### MacOS
```sh 
brew install llvm
```

## Installing CMake
### Linux
```sh 
sudo apt update
sudo apt install make
sudo apt install cmake
```

### MacOS
```sh 
brew install cmake
```

## Using an external BLAS (optional)
By default, SuperKMeans multiplies matrices with Eigen (bundled), or with Apple Accelerate on macOS.

### MacOS
**Silicon Chips (M1 to M5)**: You don't need to do anything special. We automatically use [Apple Accelerate](https://developer.apple.com/documentation/accelerate), which uses the [AMX](https://github.com/corsix/amx) unit. 

**Intel Chips (older Macs)**: We use Apple Accelerate as well.

### Linux
In our benchmarks the default Eigen performs roughly on par with OpenBLAS (on par), MKL (5-10% slower e2e), and BLIS (on par). To use these instead, configure with `-DSKMEANS_GEMM=blas`. **BLAS must be single-threaded and thread-safe**:
- **Intel MKL**: detected automatically and linked in its sequential version.
- **OpenBLAS**: if no BLAS is found, CMake builds a suitable OpenBLAS from source. To build one yourself:
  ```sh
  git clone https://github.com/OpenMathLib/OpenBLAS.git
  cd OpenBLAS
  make -j$(nproc) DYNAMIC_ARCH=1 USE_THREAD=0 USE_LOCKING=1
  make PREFIX=/usr/local install
  ldconfig
  ```
  With a multi-threaded OpenBLAS (e.g. from `apt`), set `OPENBLAS_NUM_THREADS=1`.
- **AMD AOCL BLIS**: link the single-threaded library (`libblis.so`, not `libblis-mt.so`).

To force a specific library, pass its path:
```sh
# C++
cmake . -DSKMEANS_GEMM=blas -DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis.so

# Python Installation
pip install --force-reinstall . -C cmake.args="-DSKMEANS_GEMM=blas;-DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis.so"
```

## Using OpenMP (optional)
The default thread pool is [ForkUnion](https://github.com/ashvardanian/ForkUnion), fetched by CMake. To use OpenMP instead, install it and configure with `-DSKMEANS_EXECUTOR=openmp`.

### Linux
Most distributions come with OpenMP, or you can install it with:
```sh
sudo apt-get install libomp-dev
```

### MacOS
```sh 
brew install libomp
```


## Troubleshooting

### Python bindings installation fails

Error:
```
Could NOT find Python (missing: Development.Module) 
    Reason given by package:
        Development: Cannot find the directory "/usr/include/python3.12"
```

Solution: Install `python-dev` package:

```sh
sudo apt install python3-dev
```


### Does Super K-Means use SIMD?
Yes. We have optimizations for AVX512, AVX2, and NEON. You don't need to do anything special to activate these. If your machine doesn't have any of these, we rely on scalar code. 

### Quantized clustering is slower than full-precision clustering!
The speed of quantized clustering depends on your hardware capabilities. Here are some known situations in which `f32` ends up being faster than `sq8`:
- Apple M1-M5: The AMX unit from the silicon chips has optimized GEMM for `float32`. Using `sq8` or `lvq4` will regress clustering time.  

## GPU 
Looking for installation on GPU? We have an implementation (see `gpu` branch)! But the installation instructions are still WIP.