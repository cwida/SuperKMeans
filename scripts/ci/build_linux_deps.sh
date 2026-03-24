#!/bin/bash
set -euo pipefail

# Build dependencies for SuperKMeans Linux wheels (manylinux_2_28 / AlmaLinux 8)
# Uses the system GCC (available by default) + OpenBLAS from source + libomp from LLVM source
#
# Why libomp? GCC's libgomp has significantly higher overhead for frequent OpenMP
# parallel region entry/exit (common in batched BLAS + distance computation loops).
# LLVM's libomp maintains a persistent spin-waiting thread pool that performs much better.
# libomp exports GOMP compatibility symbols, so GCC-compiled code works seamlessly.
#
# Strategy: build libomp, then replace the system libgomp with a symlink to libomp.
# When GCC's -fopenmp links -lgomp, it finds libomp instead. The resulting binary's
# DT_NEEDED entry points to libomp.so (via SONAME), so only libomp is loaded at runtime.
# This is equivalent to LD_PRELOAD=libomp but done at link time.

OPENBLAS_VERSION="0.3.31"
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
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_MODULE_PATH=/tmp/cmake-${LLVM_VERSION}.src/Modules \
    -DLIBOMP_ENABLE_SHARED=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF
cmake --build build -j"$(nproc)"
cmake --install build

# Replace libgomp with libomp so that GCC's -fopenmp resolves to libomp.
# The linker reads libomp's SONAME, so the binary depends on libomp.so.5 at runtime.
ln -sf /usr/local/lib/libomp.so /usr/local/lib/libgomp.so
ldconfig

echo "=== Building OpenBLAS ${OPENBLAS_VERSION} with DYNAMIC_ARCH ==="
curl -L "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
    -o /tmp/openblas.tar.gz
tar xzf /tmp/openblas.tar.gz -C /tmp
cd /tmp/OpenBLAS-${OPENBLAS_VERSION}

# USE_OPENMP=1: OpenBLAS uses OpenMP, which resolves to libomp via the symlink above.
# Single OpenMP runtime (libomp) for both OpenBLAS and the application.
make libs netlib shared FC= \
    DYNAMIC_ARCH=1 \
    USE_OPENMP=1 \
    NO_LAPACK=1 \
    NO_LAPACKE=1 \
    NO_FORTRAN=1 \
    NUM_THREADS=384 \
    -j"$(nproc)"
make install PREFIX=/usr/local
ldconfig

echo "=== Dependencies installed ==="
gcc --version
echo "=== Verifying libomp symlink ==="
ls -la /usr/local/lib/libgomp.so
ldd /usr/local/lib/libopenblas.so | grep -E "omp|gomp" || true
ls -la /usr/local/lib/libomp*
ls -la /usr/local/lib/libopenblas*
