#!/bin/bash
set -euo pipefail

# Build dependencies for SuperKMeans Linux wheels (manylinux_2_28 / AlmaLinux 8)
# Uses the system GCC (available by default) + OpenBLAS from source

OPENBLAS_VERSION="0.3.31"

echo "=== Building OpenBLAS ${OPENBLAS_VERSION} with DYNAMIC_ARCH ==="
curl -L "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
    -o /tmp/openblas.tar.gz
tar xzf /tmp/openblas.tar.gz -C /tmp
cd /tmp/OpenBLAS-${OPENBLAS_VERSION}

make FC= \
    DYNAMIC_ARCH=1 \
    USE_OPENMP=1 \
    NO_LAPACK=1 \
    NO_FORTRAN=1 \
    NUM_THREADS=384 \
    -j"$(nproc)" > /dev/null 2>&1
make install PREFIX=/usr/local > /dev/null 2>&1
ldconfig

echo "=== Dependencies installed ==="
gcc --version
ls -la /usr/local/lib/libopenblas*
