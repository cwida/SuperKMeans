# Installation

### PyPI
We are available in PyPI. However, we recommend installing from source for maximum performance.
```sh
pip install superkmeans
```

### Super K-Means on CPU needs:
- Clang 17, CMake 3.26
- OpenMP
- A BLAS implementation
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
* [Installing OpenMP](#installing-openmp)
* [Installing BLAS](#installing-blas)
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

## Installing OpenMP

### Linux
Most distributions come with OpenMP, or you can install it with:
```sh
sudo apt-get install libomp-dev
```

### MacOS
```sh 
brew install libomp
```


## Installing BLAS

BLAS is extremely important to achieve high performance. We recommend [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). 

### Linux
Most distributions come with [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), or you may have already installed OpenBLAS via `apt`. **THIS IS SLOW**. We recommend installing OpenBLAS from source with the commands below.

```sh
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) DYNAMIC_ARCH=1 USE_OPENMP=1 NUM_THREADS=128
make -j$(nproc) PREFIX=/usr/local install
ldconfig
```

### MacOS
**Silicon Chips (M1 to M5)**: You don't need to do anything special. We automatically detect [Apple Accelerate](https://developer.apple.com/documentation/accelerate) that uses the [AMX](https://github.com/corsix/amx) unit. 

**Intel Chips (older Macs)**: Install OpenBLAS as detailed above.


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

### I get a bunch of `warnings` when compiling SuperKMeans

If you see a lot of warnings like this one:
```warning: ignoring ‘#pragma clang loop’```

You are using GCC instead of Clang. If you installed Clang, you can set the correct compiler by doing the following:
```sh
export CXX="/usr/bin/clang++-18" # Linux

export CXX="/opt/homebrew/opt/llvm/bin/clang++" # MacOS
```


### Super K-Means is slow on my Apple Silicon
If you previously installed OpenBLAS in your machine, the installation may be linking to OpenBLAS instead of Apple Accelerate. You can try forcing the linking of Apple Accelerate: 

```sh
# C++
cmake . -DBLAS_LIBRARIES=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework
make 

# Python Installation
pip install --force-reinstall . -C cmake.args="-DBLAS_LIBRARIES=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework"
```

### Super K-Means is slow even after installing OpenBLAS from source in Linux
Try forcing the linking to the OpenBlas you just installed. For example:
```sh
# C++
cmake . -DBLAS_LIBRARIES=/usr/local/lib/libopenblas_neoversev2p-r0.3.31.dev.so
make 

# Python Installation
pip install --force-reinstall . -C cmake.args="-DBLAS_LIBRARIES=/usr/local/lib/libopenblas_neoversev2p-r0.3.31.dev.so"
```

Note that the name of the `.so` library changes if you target a specific CPU or use `DYNAMIC_ARCH=1`. You might want to verify its name by looking into the `/usr/local/lib` directory.

### Can I use IntelMKL or AMD AOCL BLIS instead of OpenBLAS?
We detect IntelMKL automatically. For AMD AOCL BLIS, you can force the linking by doing:

```sh
# C++
cmake . -DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis-mt.so

# Python Installation
pip install --force-reinstall . -C cmake.args="-DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis-mt.so"
```

### OpenBLAS binaries are too large. What can I do?
You can install OpenBLAS for a specific CPU instead of using `DYNAMIC_ARCH=1`. This will reduce the binary sizes. You can check the available CPU targets [here](https://github.com/OpenMathLib/OpenBLAS/blob/develop/TargetList.txt).

These are some commands for common CPUs:
```sh
make clean

# Intel Sapphire Rapids, Emerald Rapids, Granite Rapids
# Code-named in AWS as r8i, r7i
make TARGET=SAPPHIRERAPIDS DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# Intel Skylake, Icelake
# Code-named in AWS as r6i
make TARGET=SKYLAKEX DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AMD Zen 5, Zen 4, Zen 3
# Code-named in AWS as r8a, r7a, r6a
make TARGET=ZEN DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AWS Graviton 4
# Code-named in AWS as r8g
make TARGET=NEOVERSEV2 DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AWS Graviton 3
# Code-named in AWS as r7g
make TARGET=NEOVERSEV1 DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128
```

### Does Super K-Means use SIMD?
Yes. We have optimizations for AVX512, AVX2, and NEON. You don't need to do anything special to activate these. If your machine doesn't have any of these, we rely on scalar code. 

## GPU 
Looking for installation on GPU? We have an implementation (see `legacy_gpu` branch)! But the installation instructions are still WiP.