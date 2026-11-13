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
#        <full_precision_final_centroids> <pruning>
#
# Sensible combinations:
#   f32:    quantized_centroid_update and full_precision_final_centroids are N/A
#           → always false/false, only vary pruning
#   sq8/lvq4/rabitq:
#           full_precision_final_centroids only matters when quantized_centroid_update=true
#           → skip (quantized_centroid_update=false, full_precision_final_centroids=true)
# ─────────────────────────────────────────────────────────────────────────────

STEP=0

run() {
    STEP=$((STEP+1))
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "── [$STEP] $1 ──"
    shift
    "$BIN" "$@"
}

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# DATASET: $DATASET"
    echo "######################################################################"

    # ==================================================================
    #  RAW (no dimensionality reduction)
    # ==================================================================

    # ── hierarchical superkmeans (raw only; f32, sq8, lvq4, rabitq) ──
    # No pruning: pruning=false
    run "raw / f32 / hsk / no-pruning"    "$DATASET" hsk f32    false false false
    run "raw / sq8 / hsk / no-pruning"    "$DATASET" hsk sq8    true  false false
    run "raw / lvq4 / hsk / no-pruning"   "$DATASET" hsk lvq4   true  false false
    run "raw / rabitq / hsk / no-pruning" "$DATASET" hsk rabitq true  false false
    # Pruning: pruning=true
    run "raw / f32 / hsk / pruning"       "$DATASET" hsk f32    false false true
    run "raw / sq8 / hsk / pruning"       "$DATASET" hsk sq8    true  false true
    run "raw / lvq4 / hsk / pruning"      "$DATASET" hsk lvq4   true  false true
    run "raw / rabitq / hsk / pruning"    "$DATASET" hsk rabitq true  false true

    continue

    # ── raw + f32 ──
    run "raw / f32 / pruning"                                        "$DATASET" raw f32 false false true
    run "raw / f32 / no-pruning"                                     "$DATASET" raw f32 false false false

    # ── raw + sq8 ──
    # run "raw / sq8 / no-quant-update / pruning"                      "$DATASET" raw sq8 false false true
    run "raw / sq8 / quant-update / pruning"                         "$DATASET" raw sq8 true false true
    # run "raw / sq8 / quant-update + full-prec-final / pruning"       "$DATASET" raw sq8 true true true
    # run "raw / sq8 / no-quant-update / no-pruning"                   "$DATASET" raw sq8 false false false
    run "raw / sq8 / quant-update / no-pruning"                      "$DATASET" raw sq8 true false false
    # run "raw / sq8 / quant-update + full-prec-final / no-pruning"    "$DATASET" raw sq8 true true false

    # ── raw + lvq4 ──
    # run "raw / lvq4 / no-quant-update / pruning"                      "$DATASET" raw lvq4 false false true
    run "raw / lvq4 / quant-update / pruning"                         "$DATASET" raw lvq4 true false true
    run "raw / lvq4 / quant-update + full-prec-final / pruning"       "$DATASET" raw lvq4 true true true
    # run "raw / lvq4 / no-quant-update / no-pruning"                   "$DATASET" raw lvq4 false false false
    run "raw / lvq4 / quant-update / no-pruning"                      "$DATASET" raw lvq4 true false false
    run "raw / lvq4 / quant-update + full-prec-final / no-pruning"    "$DATASET" raw lvq4 true true false

    # # ── raw + rabitq ──
    run "raw / rabitq / quant-update / pruning"                    "$DATASET" raw rabitq true false true
    run "raw / rabitq / quant-update + full-prec-final / pruning"  "$DATASET" raw rabitq true true true
    # run "raw / rabitq / no-quant-update / no-pruning"                "$DATASET" raw rabitq false false false
    run "raw / rabitq / quant-update / no-pruning"                   "$DATASET" raw rabitq true false false
    run "raw / rabitq / quant-update + full-prec-final / no-pruning" "$DATASET" raw rabitq true true false

    # ==================================================================
    #  PCA (dimensionality reduction, iterates over TARGET_D internally)
    # ==================================================================

    # ── pca + f32 ──
    run "pca / f32 / pruning"                                        "$DATASET" pca f32 false false true
    run "pca / f32 / no-pruning"                                     "$DATASET" pca f32 false false false

    # ── pca + sq8 ──
    # run "pca / sq8 / no-quant-update / pruning"                      "$DATASET" pca sq8 false false true
    run "pca / sq8 / quant-update / pruning"                         "$DATASET" pca sq8 true false true
    # run "pca / sq8 / quant-update + full-prec-final / pruning"       "$DATASET" pca sq8 true true true
    # run "pca / sq8 / no-quant-update / no-pruning"                   "$DATASET" pca sq8 false false false
    run "pca / sq8 / quant-update / no-pruning"                      "$DATASET" pca sq8 true false false
    # run "pca / sq8 / quant-update + full-prec-final / no-pruning"    "$DATASET" pca sq8 true true false

    # ── pca + lvq4 ──
    # run "pca / lvq4 / no-quant-update / pruning"                      "$DATASET" pca lvq4 false false true
    run "pca / lvq4 / quant-update / pruning"                         "$DATASET" pca lvq4 true false true
    # run "pca / lvq4 / quant-update + full-prec-final / pruning"       "$DATASET" pca lvq4 true true true
    # run "pca / lvq4 / no-quant-update / no-pruning"                   "$DATASET" pca lvq4 false false false
    run "pca / lvq4 / quant-update / no-pruning"                      "$DATASET" pca lvq4 true false false
    # run "pca / lvq4 / quant-update + full-prec-final / no-pruning"    "$DATASET" pca lvq4 true true false

    # ── pca + rabitq ──
    run "pca / rabitq / quant-update / pruning"                    "$DATASET" pca rabitq true false true
    run "pca / rabitq / quant-update + full-prec-final / pruning"  "$DATASET" pca rabitq true true true
    # run "pca / rabitq / no-quant-update / no-pruning"                "$DATASET" pca rabitq false false false
    run "pca / rabitq / quant-update / no-pruning"                   "$DATASET" pca rabitq true false false
    run "pca / rabitq / quant-update + full-prec-final / no-pruning" "$DATASET" pca rabitq true true false

    # ==================================================================
    #  JLT (dimensionality reduction, iterates over TARGET_D internally)
    #  Deactivated with Quantization as PCA always perform better
    # ==================================================================

    # ── jlt + f32 ──
    run "jlt / f32 / pruning"                                        "$DATASET" jlt f32 false false true
    run "jlt / f32 / no-pruning"                                     "$DATASET" jlt f32 false false false

    # # ── jlt + sq8 ──
    # run "jlt / sq8 / no-quant-update / pruning"                      "$DATASET" jlt sq8 false false true
    # run "jlt / sq8 / quant-update / pruning"                         "$DATASET" jlt sq8 true false true
    # run "jlt / sq8 / quant-update + full-prec-final / pruning"       "$DATASET" jlt sq8 true true true
    # run "jlt / sq8 / no-quant-update / no-pruning"                   "$DATASET" jlt sq8 false false false
    # run "jlt / sq8 / quant-update / no-pruning"                      "$DATASET" jlt sq8 true false false
    # run "jlt / sq8 / quant-update + full-prec-final / no-pruning"    "$DATASET" jlt sq8 true true false

    # # ── jlt + lvq4 ──
    # run "jlt / lvq4 / no-quant-update / pruning"                      "$DATASET" jlt lvq4 false false true
    # run "jlt / lvq4 / quant-update / pruning"                         "$DATASET" jlt lvq4 true false true
    # run "jlt / lvq4 / quant-update + full-prec-final / pruning"       "$DATASET" jlt lvq4 true true true
    # run "jlt / lvq4 / no-quant-update / no-pruning"                   "$DATASET" jlt lvq4 false false false
    # run "jlt / lvq4 / quant-update / no-pruning"                      "$DATASET" jlt lvq4 true false false
    # run "jlt / lvq4 / quant-update + full-prec-final / no-pruning"    "$DATASET" jlt lvq4 true true false

    # # ── jlt + rabitq (no pruning support) ──
    # run "jlt / rabitq / no-quant-update / no-pruning"                "$DATASET" jlt rabitq false false false
    # run "jlt / rabitq / quant-update / no-pruning"                   "$DATASET" jlt rabitq true false false
    # run "jlt / rabitq / quant-update + full-prec-final / no-pruning" "$DATASET" jlt rabitq true true false

    # ==================================================================
    #  Matryoshka (truncation, no quantizer combinations)
    #  Deactivated with Quantization as PCA always perform better
    # ==================================================================

    # ── mat + f32 ──
    run "mat / f32 / pruning"                                        "$DATASET" mat f32 false false true
    run "mat / f32 / no-pruning"                                     "$DATASET" mat f32 false false false

done

echo ""
echo "=========================================="
echo "All accelerators benchmarks complete! ($STEP runs)"
echo "=========================================="
echo ""
echo "CSV files written to: $SCRIPT_DIR/results/${SKM_ARCH:-default}/accelerators_*.csv"
