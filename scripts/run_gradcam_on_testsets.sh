#!/usr/bin/env bash
set -euo pipefail

# Portable Grad-CAM runner for boxed T1-weighted MRI folders.
#
# Example:
#   CKPT=/path/to/model_checkpoint.pt \
#   GOOD_DIR=/path/to/boxed_good \
#   BAD_DIR=/path/to/boxed_bad \
#   RANDOM_DIR=/path/to/boxed_random \
#   OUT_ROOT=/path/to/gradcam_outputs \
#   bash scripts/run_gradcam_on_testsets.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CKPT="${CKPT:-}"
PREDICT_SCRIPT="${PREDICT_SCRIPT:-${REPO_ROOT}/src/t1qc_predict_boxed_gradcam.py}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/gradcam}"

GOOD_DIR="${GOOD_DIR:-}"
BAD_DIR="${BAD_DIR:-}"
RANDOM_DIR="${RANDOM_DIR:-}"

DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

PASS_THRESHOLD="${PASS_THRESHOLD:-0.45}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-0.55}"
THRESHOLD="${THRESHOLD:-0.55}"
GRADCAM_LAYER="${GRADCAM_LAYER:-layer3.1.conv2}"

if [ -z "$CKPT" ]; then
  echo "ERROR: Set CKPT=/path/to/model_checkpoint.pt"
  exit 1
fi

mkdir -p "$OUT_ROOT"

run_one_set () {
  name="$1"
  input_dir="$2"

  if [ -z "$input_dir" ]; then
    echo "Skipping ${name}: directory variable not set"
    return 0
  fi

  if [ ! -d "$input_dir" ]; then
    echo "Skipping ${name}: directory does not exist: $input_dir"
    return 0
  fi

  echo "Running Grad-CAM for: ${name}"
  echo "Input: ${input_dir}"

  python "$PREDICT_SCRIPT" \
    --checkpoint "$CKPT" \
    --input-dir "$input_dir" \
    --out-csv "${OUT_ROOT}/${name}_predictions.csv" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --save-gradcam \
    --gradcam-dir "${OUT_ROOT}/${name}_gradcam" \
    --gradcam-target-layer "$GRADCAM_LAYER" \
    --pass-threshold "$PASS_THRESHOLD" \
    --fail-threshold "$FAIL_THRESHOLD"
}

run_one_set "good" "$GOOD_DIR"
run_one_set "bad" "$BAD_DIR"
run_one_set "random" "$RANDOM_DIR"

echo "Done. Outputs saved under: $OUT_ROOT"
