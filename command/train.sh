#!/bin/bash
# Train MGS on a benchmark scene.
#
# Usage:
#   bash command/train.sh <GPU> <BENCHMARK> <SCENE>
#
# Examples:
#   bash command/train.sh 0 mipnerf360 bicycle
#   bash command/train.sh 1 tanksandtemples truck
#   bash command/train.sh 2 deepblending DrJohnson
#   bash command/train.sh 3 bungeenerf rome
#
# Benchmarks: mipnerf360, tanksandtemples, deepblending, bungeenerf

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: bash command/train.sh <GPU> <BENCHMARK> <SCENE>"
    echo ""
    echo "  GPU          CUDA device index (e.g. 0)"
    echo "  BENCHMARK    mipnerf360 | tanksandtemples | deepblending | bungeenerf"
    echo "  SCENE        scene name (e.g. bicycle, truck, DrJohnson, rome)"
    exit 1
fi

GPU=$1
BENCHMARK=$2
SCENE=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$REPO_DIR")"

MIPNERF360_INDOOR="bonsai counter kitchen room"
MIPNERF360_OUTDOOR="bicycle flowers garden stump treehill"

case "$BENCHMARK" in
    mipnerf360)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/MipNeRF360/360_v2/${SCENE}"
        if echo "$MIPNERF360_INDOOR" | grep -qw "$SCENE"; then
            DATA_FACTOR=2
        else
            DATA_FACTOR=4
        fi
        REFINE_STOP=50000
        ;;
    tanksandtemples)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/TandT/tandt/${SCENE}"
        DATA_FACTOR=1
        REFINE_STOP=50000
        ;;
    deepblending)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/DeepBlending/${SCENE}"
        DATA_FACTOR=1
        REFINE_STOP=25000
        ;;
    bungeenerf)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/BungeeNeRF/${SCENE}"
        DATA_FACTOR=1
        REFINE_STOP=50000
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Choose from: mipnerf360, tanksandtemples, deepblending, bungeenerf"
        exit 1
        ;;
esac

RESULT_DIR="${WORKSPACE_DIR}/checkpoint/${BENCHMARK}/${SCENE}"

echo "=== MGS Training ==="
echo "GPU:        ${GPU}"
echo "Benchmark:  ${BENCHMARK}"
echo "Scene:      ${SCENE}"
echo "Data dir:   ${DATA_DIR}"
echo "Data factor:${DATA_FACTOR}"
echo "Result dir: ${RESULT_DIR}"
echo "===================="

CUDA_VISIBLE_DEVICES=${GPU} python "${REPO_DIR}/train.py" mcmc \
    --data_dir "${DATA_DIR}" \
    --data_factor ${DATA_FACTOR} \
    --result_dir "${RESULT_DIR}" \
    --max_steps 50000 \
    --strategy.refine_stop_iter ${REFINE_STOP} \
    --strategy.cap-max 5000000 \
    --sort_strategy by_opacity_descending \
    --diffusion_objective single_prefix_full \
    --diffusion_schedule uniform \
    --diffusion_min_keep_ratio 0.0 \
    --diffusion_max_keep_ratio 1.0 \
    --diffusion_include_full_subset \
    --eval-steps 49999 \
    --save-steps 49999 \
    --disable-viewer
