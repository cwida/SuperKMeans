#!/bin/bash

# Accelerators benchmark runner — exhaustive parameter sweep
# Usage: ./accelerators.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./accelerators.sh                          # Run all datasets with default build dir
#   ./accelerators.sh mxbai openai             # Run only mxbai and openai
#   ./accelerators.sh -b ../build mxbai        # Run mxbai with custom build dir

set -e

BUILD_DIR="../cmake-build-release"

while getopts "b:" opt; do
    case $opt in
        b) BUILD_DIR="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mxbai openai)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$BUILD_DIR" = /* ]]; then
    BUILD_DIR_ABS="$BUILD_DIR"
else
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

BIN="$BUILD_DIR_ABS/benchmarks/accelerators.out"

echo "=========================================="
echo "Accelerators Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""

echo "Building accelerators.out..."
cd "$BUILD_DIR_ABS"
cmake --build . --target accelerators.out -j
echo "Build complete!"
echo ""

cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Naming convention for the arguments:
#   $BIN <dataset> <dim_reduction> <quantizer> <quantized_centroid_update>
#        <full_precision_final_centroids> <use_blas_only>
#
# Sensible combinations:
#   f32:    quantized_centroid_update and full_precision_final_centroids are N/A
#           → always false/false, only vary use_blas_only
#   sq8/sq4/rabitq:
#           full_precision_final_centroids only matters when quantized_centroid_update=true
#           → skip (quantized_centroid_update=false, full_precision_final_centroids=true)
# ─────────────────────────────────────────────────────────────────────────────

STEP=0

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# DATASET: $DATASET"
    echo "######################################################################"

    # ==================================================================
    #  RAW (no dimensionality reduction)
    # ==================================================================

    # ── raw + f32 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / f32 / pruning ──"
    "$BIN" "$DATASET" raw f32 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / f32 / blas-only ──"
    "$BIN" "$DATASET" raw f32 false false true

    # ── raw + sq8 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" raw sq8 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / quant-update / pruning ──"
    "$BIN" "$DATASET" raw sq8 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" raw sq8 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" raw sq8 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / quant-update / blas-only ──"
    "$BIN" "$DATASET" raw sq8 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq8 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" raw sq8 true true true

    # ── raw + sq4 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" raw sq4 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / quant-update / pruning ──"
    "$BIN" "$DATASET" raw sq4 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" raw sq4 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" raw sq4 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / quant-update / blas-only ──"
    "$BIN" "$DATASET" raw sq4 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / sq4 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" raw sq4 true true true

    # ── raw + rabitq ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / no-quant-update / pruning ──"
    "$BIN" "$DATASET" raw rabitq false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / quant-update / pruning ──"
    "$BIN" "$DATASET" raw rabitq true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" raw rabitq true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" raw rabitq false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / quant-update / blas-only ──"
    "$BIN" "$DATASET" raw rabitq true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] raw / rabitq / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" raw rabitq true true true

    # ==================================================================
    #  PCA (dimensionality reduction, iterates over TARGET_D internally)
    # ==================================================================

    # ── pca + f32 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / f32 / pruning ──"
    "$BIN" "$DATASET" pca f32 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / f32 / blas-only ──"
    "$BIN" "$DATASET" pca f32 false false true

    # ── pca + sq8 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" pca sq8 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / quant-update / pruning ──"
    "$BIN" "$DATASET" pca sq8 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" pca sq8 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" pca sq8 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / quant-update / blas-only ──"
    "$BIN" "$DATASET" pca sq8 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq8 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" pca sq8 true true true

    # ── pca + sq4 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" pca sq4 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / quant-update / pruning ──"
    "$BIN" "$DATASET" pca sq4 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" pca sq4 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" pca sq4 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / quant-update / blas-only ──"
    "$BIN" "$DATASET" pca sq4 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / sq4 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" pca sq4 true true true

    # ── pca + rabitq ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / no-quant-update / pruning ──"
    "$BIN" "$DATASET" pca rabitq false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / quant-update / pruning ──"
    "$BIN" "$DATASET" pca rabitq true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" pca rabitq true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" pca rabitq false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / quant-update / blas-only ──"
    "$BIN" "$DATASET" pca rabitq true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] pca / rabitq / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" pca rabitq true true true

    # ==================================================================
    #  JLT (dimensionality reduction, iterates over TARGET_D internally)
    # ==================================================================

    # ── jlt + f32 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / f32 / pruning ──"
    "$BIN" "$DATASET" jlt f32 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / f32 / blas-only ──"
    "$BIN" "$DATASET" jlt f32 false false true

    # ── jlt + sq8 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" jlt sq8 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / quant-update / pruning ──"
    "$BIN" "$DATASET" jlt sq8 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" jlt sq8 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt sq8 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt sq8 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq8 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" jlt sq8 true true true

    # ── jlt + sq4 ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / no-quant-update / pruning ──"
    "$BIN" "$DATASET" jlt sq4 false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / quant-update / pruning ──"
    "$BIN" "$DATASET" jlt sq4 true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" jlt sq4 true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt sq4 false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt sq4 true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / sq4 / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" jlt sq4 true true true

    # ── jlt + rabitq ──
    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / no-quant-update / pruning ──"
    "$BIN" "$DATASET" jlt rabitq false false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / quant-update / pruning ──"
    "$BIN" "$DATASET" jlt rabitq true false false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / quant-update + full-prec-final / pruning ──"
    "$BIN" "$DATASET" jlt rabitq true true false

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / no-quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt rabitq false false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / quant-update / blas-only ──"
    "$BIN" "$DATASET" jlt rabitq true false true

    STEP=$((STEP+1)); echo ""; echo "── [$STEP] jlt / rabitq / quant-update + full-prec-final / blas-only ──"
    "$BIN" "$DATASET" jlt rabitq true true true

done

echo ""
echo "=========================================="
echo "All accelerators benchmarks complete! ($STEP runs)"
echo "=========================================="
echo ""
echo "CSV files written to: $SCRIPT_DIR/results/${SKM_ARCH:-default}/accelerators_*.csv"
