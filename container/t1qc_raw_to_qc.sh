#!/bin/sh
set -eu

export PATH="/usr/local/bin:/opt/conda/envs/tenet/bin:/opt/fsl/bin:/opt/ants/bin:$PATH"
export PYTHONNOUSERSITE=1

export T1QC_HOME="/opt/t1qc"
export T1QC_MODEL="/opt/t1qc/models/t1qc_best_epoch021.pt"

export FSLDIR="/opt/fsl"
export ANTSPATH="/opt/ants/bin/"
export LD_LIBRARY_PATH="/opt/ants/lib:${LD_LIBRARY_PATH:-}"
export FSLOUTPUTTYPE="NIFTI_GZ"
export FSLMULTIFILEQUIT="TRUE"
export FSLLOCKDIR=""
export FSLMACHINELIST=""
export FSLREMOTECALL=""

RAW_INPUT_DIR=""
BOXED_OUTPUT_DIR=""
OUT_CSV=""
THRESHOLD="0.558"
BATCH_SIZE="2"
NUM_WORKERS="0"
DEVICE="cpu"
MNI_TEMPLATE=""
MNI_MASK=""
SKIP_BOX_EXISTING="1"
RECURSIVE="1"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --raw-input-dir)
            RAW_INPUT_DIR="$2"
            shift 2
            ;;
        --boxed-output-dir)
            BOXED_OUTPUT_DIR="$2"
            shift 2
            ;;
        --out-csv)
            OUT_CSV="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --mni-template)
            MNI_TEMPLATE="$2"
            shift 2
            ;;
        --mni-mask)
            MNI_MASK="$2"
            shift 2
            ;;
        --no-skip-box-existing)
            SKIP_BOX_EXISTING="0"
            shift 1
            ;;
        --no-recursive)
            RECURSIVE="0"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$RAW_INPUT_DIR" ]; then
    echo "ERROR: missing --raw-input-dir"
    exit 1
fi

if [ -z "$BOXED_OUTPUT_DIR" ]; then
    echo "ERROR: missing --boxed-output-dir"
    exit 1
fi

if [ -z "$OUT_CSV" ]; then
    echo "ERROR: missing --out-csv"
    exit 1
fi

if [ -z "$MNI_TEMPLATE" ]; then
    echo "ERROR: missing --mni-template"
    exit 1
fi

if [ -z "$MNI_MASK" ]; then
    echo "ERROR: missing --mni-mask"
    exit 1
fi

mkdir -p "$BOXED_OUTPUT_DIR"
mkdir -p "$(dirname "$OUT_CSV")"

echo "========== Environment check =========="
echo "PATH=$PATH"
echo "FSLDIR=$FSLDIR"
echo "ANTSPATH=$ANTSPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "antsRegistrationSyNQuick.sh=$(which antsRegistrationSyNQuick.sh || true)"
echo "antsRegistration=$(which antsRegistration || true)"
echo "antsApplyTransforms=$(which antsApplyTransforms || true)"
echo ""

echo "========== Step 1/3: Boxing raw T1 images =========="
echo "Raw input folder:    $RAW_INPUT_DIR"
echo "Boxed output folder: $BOXED_OUTPUT_DIR"
echo "MNI template:        $MNI_TEMPLATE"
echo "MNI mask:            $MNI_MASK"

BOX_ARGS="--input-dir \"$RAW_INPUT_DIR\" --out-root \"$BOXED_OUTPUT_DIR\" --mni-template \"$MNI_TEMPLATE\" --mni-mask \"$MNI_MASK\" --continue-on-error"

if [ "$RECURSIVE" = "1" ]; then
    BOX_ARGS="$BOX_ARGS --recursive"
fi

if [ "$SKIP_BOX_EXISTING" = "1" ]; then
    BOX_ARGS="$BOX_ARGS --skip-existing"
fi

eval python /opt/t1qc/app/box_t1.py "$BOX_ARGS"

echo ""
echo "========== Step 2/3: Running T-Net QC prediction =========="

python /opt/t1qc/app/predict_boxed276_folder.py \
    --checkpoint /opt/t1qc/models/t1qc_best_epoch021.pt \
    --input-dir "$BOXED_OUTPUT_DIR" \
    --out-csv "$OUT_CSV" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE"

echo ""
echo "========== Step 3/3: Adding PASS/REVIEW/FAIL QC category to same CSV =========="

python - "$OUT_CSV" <<'PY'
import sys
from pathlib import Path
import pandas as pd

csv_path = Path(sys.argv[1])

PASS_TH = 0.458
FAIL_TH = 0.558

df = pd.read_csv(csv_path)

if "prob_bad" not in df.columns:
    raise SystemExit("ERROR: CSV must contain column: prob_bad")

def qc_category_from_prob(p):
    p = float(p)
    if p < PASS_TH:
        return "PASS_likely_good"
    elif p < FAIL_TH:
        return "REVIEW_caution"
    else:
        return "FAIL_likely_bad"

df["qc_category"] = df["prob_bad"].apply(qc_category_from_prob)
df["pass_threshold"] = PASS_TH
df["fail_threshold"] = FAIL_TH
df["qc_rule"] = "PASS<0.458; REVIEW=0.458-<0.558; FAIL>=0.558"

df.to_csv(csv_path, index=False)

print("Saved final CSV:", csv_path)
print()
print("QC counts:")
print(df["qc_category"].value_counts(dropna=False).to_string())
PY

echo ""
echo "========== Done =========="
echo "Final CSV: $OUT_CSV"
