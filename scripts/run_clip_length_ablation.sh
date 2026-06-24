#!/bin/bash
# Launch T=16, T=32, T=128 on 3 GPUs in parallel.
# T=64 is already done; pass --existing_t64 path to merge its results.
#
# Usage:
#   bash run_clip_length_ablation.sh [GPU_0] [GPU_1] [GPU_2]
#   bash run_clip_length_ablation.sh 0 1 2    # default

GPU_0=${1:-0}
GPU_1=${2:-1}
GPU_2=${3:-2}

SPLIT_ROOT="NVFAIR_split"
FRAMES_ROOT="nvfair_frames"
OUTPUT_DIR="outputs/ablation_clip_length"
RESULTS_DIR="results/ablation_clip_length"
# The FEAT_DIFF trainer (src/feat_diff.py) is the single entry point; the
# clip-length ablation (Table VII) just sweeps its --clip_length argument.
# T=64 is the main paper model. Run this script from the repository root.
SCRIPT="src/feat_diff.py"

mkdir -p logs/ablation_clip_length

COMMON="--nvfair_split_root ${SPLIT_ROOT} --frames_root ${FRAMES_ROOT} \
  --output_dir ${OUTPUT_DIR} --results_dir ${RESULTS_DIR} \
  --num_epochs 30 --steps_per_epoch 200 \
  --num_ids 16 --vids_per_id 8 \
  --lr 1e-3 --wd 1e-4 --temperature 0.07 --num_workers 4"

echo "========================================================"
echo " Clip-Length Ablation: T=16, 32, 128"
echo " GPUs: $GPU_0 / $GPU_1 / $GPU_2"
echo "========================================================"

CUDA_VISIBLE_DEVICES=${GPU_0} python ${SCRIPT} --clip_length 16  --gpu 0 \
  ${COMMON} > logs/ablation_clip_length/T16.log 2>&1 &
PID0=$!; echo "▶ GPU${GPU_0} T=16  PID=${PID0}"

CUDA_VISIBLE_DEVICES=${GPU_1} python ${SCRIPT} --clip_length 32  --gpu 0 \
  ${COMMON} > logs/ablation_clip_length/T32.log 2>&1 &
PID1=$!; echo "▶ GPU${GPU_1} T=32  PID=${PID1}"

CUDA_VISIBLE_DEVICES=${GPU_2} python ${SCRIPT} --clip_length 128 --gpu 0 \
  ${COMMON} > logs/ablation_clip_length/T128.log 2>&1 &
PID2=$!; echo "▶ GPU${GPU_2} T=128 PID=${PID2}"

echo "Waiting for all 3 jobs..."
wait $PID0 $PID1 $PID2

echo ""
echo "All done! Plotting..."
python ${SCRIPT} --plot_only --results_dir "${RESULTS_DIR}"
echo "Results → ${RESULTS_DIR}/"