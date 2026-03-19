#!/bin/bash
# Evaluate a trained MGS checkpoint on a benchmark scene.
#
# Usage:
#   bash command/eval.sh <GPU> <BENCHMARK> <SCENE>
#
# Examples:
#   bash command/eval.sh 0 mipnerf360 bicycle
#   bash command/eval.sh 1 tanksandtemples truck
#   bash command/eval.sh 2 deepblending DrJohnson
#   bash command/eval.sh 3 bungeenerf rome
#
# Benchmarks: mipnerf360, tanksandtemples, deepblending, bungeenerf

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: bash command/eval.sh <GPU> <BENCHMARK> <SCENE>"
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
        ;;
    tanksandtemples)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/TandT/tandt/${SCENE}"
        DATA_FACTOR=1
        ;;
    deepblending)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/DeepBlending/${SCENE}"
        DATA_FACTOR=1
        ;;
    bungeenerf)
        DATA_DIR="${WORKSPACE_DIR}/benchmark/BungeeNeRF/${SCENE}"
        DATA_FACTOR=1
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Choose from: mipnerf360, tanksandtemples, deepblending, bungeenerf"
        exit 1
        ;;
esac

CKPT="${WORKSPACE_DIR}/checkpoint/${BENCHMARK}/${SCENE}/ckpts/ckpt_49999_rank0.pt"
OUTPUT_DIR="${WORKSPACE_DIR}/prediction-image/${BENCHMARK}/${SCENE}"

echo "=== MGS Evaluation ==="
echo "GPU:        ${GPU}"
echo "Benchmark:  ${BENCHMARK}"
echo "Scene:      ${SCENE}"
echo "Data dir:   ${DATA_DIR}"
echo "Data factor:${DATA_FACTOR}"
echo "Checkpoint: ${CKPT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "======================"

CUDA_VISIBLE_DEVICES=${GPU} python "${REPO_DIR}/eval.py" \
    --ckpt "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --data_factor ${DATA_FACTOR} \
    --output_dir "${OUTPUT_DIR}" \
    --save_images
