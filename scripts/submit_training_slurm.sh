#!/usr/bin/env bash
#SBATCH --job-name=t1qc_train
#SBATCH --output=logs/t1qc_train_%j.out
#SBATCH --error=logs/t1qc_train_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Portable SLURM training template.
#
# Example:
#   MANIFEST=/path/to/train_manifest.csv \
#   OUT_DIR=/path/to/training_output \
#   sbatch scripts/submit_training_slurm.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST="${MANIFEST:-}"
OUT_DIR="${OUT_DIR:-}"

if [ -z "$MANIFEST" ]; then
  echo "ERROR: Set MANIFEST=/path/to/train_manifest.csv"
  exit 1
fi

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: Set OUT_DIR=/path/to/training_output"
  exit 1
fi

mkdir -p logs
mkdir -p "$OUT_DIR"

python "${REPO_ROOT}/src/t1qc_train_model.py" \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --epochs 50 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --base-channels 8 \
  --lr 5e-5 \
  --num-workers 4
