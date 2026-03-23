#!/bin/bash
set -euo pipefail

# Build dependencies for SuperKMeans Linux wheels (manylinux_2_28 / AlmaLinux 8)
# Clang 18 from pre-built LLVM tarball + OpenBLAS from source

LLVM_VERSION="18.1.8"
OPENBLAS_VERSION="0.3.31"

echo "=== Installing system dependencies ==="
dnf install -y ncurses-compat-libs

echo "=== Installing Clang ${LLVM_VERSION} ==="
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    LLVM_TARBALL="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
elif [ "$ARCH" = "aarch64" ]; then
    LLVM_TARBALL="clang+llvm-${LLVM_VERSION}-aarch64-linux-gnu.tar.xz"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

curl -L "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_TARBALL}" \
    -o /tmp/llvm.tar.xz
tar xf /tmp/llvm.tar.xz -C /usr/local --strip-components=1
rm /tmp/llvm.tar.xz

echo "=== Building OpenBLAS ${OPENBLAS_VERSION} with DYNAMIC_ARCH ==="
curl -L "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
    -o /tmp/openblas.tar.gz
tar xzf /tmp/openblas.tar.gz -C /tmp
cd /tmp/OpenBLAS-${OPENBLAS_VERSION}

make CC=clang FC= \
    DYNAMIC_ARCH=1 \
    USE_OPENMP=1 \
    NO_LAPACK=1 \
    NO_FORTRAN=1 \
    NUM_THREADS=384 \
    -j"$(nproc)"
make install PREFIX=/usr/local
ldconfig

echo "=== Dependencies installed ==="
clang --version
ls -la /usr/local/lib/libopenblas*
