#!/bin/bash
set -euo pipefail

# Build dependencies for SuperKMeans macOS wheels
# Apple Accelerate is used for BLAS (no need to build OpenBLAS)
# libomp is built from source (not Homebrew) for macOS version compatibility

LLVM_VERSION="18.1.8"

echo "=== Building libomp from LLVM ${LLVM_VERSION} source ==="
curl -L "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/openmp-${LLVM_VERSION}.src.tar.xz" \
    -o /tmp/openmp.tar.xz
curl -L "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/cmake-${LLVM_VERSION}.src.tar.xz" \
    -o /tmp/cmake-modules.tar.xz

tar xf /tmp/openmp.tar.xz -C /tmp
tar xf /tmp/cmake-modules.tar.xz -C /tmp

cd /tmp/openmp-${LLVM_VERSION}.src
cmake -B build \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_MODULE_PATH=/tmp/cmake-${LLVM_VERSION}.src/Modules
cmake --build build -j"$(sysctl -n hw.ncpu)"
sudo cmake --install build

echo "=== Dependencies installed ==="
ls -la /usr/local/lib/libomp*
