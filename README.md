# Neuroimaging Toolkit for Human-in-the-Loop Quality Control

This repository contains code for a prototype neuroimaging quality-control workflow. The current version focuses on **T1-weighted MRI quality control**, with the longer-term goal of supporting quality review across additional neuroimaging modalities.

The Toolkit is designed to take a T1-weighted MRI scan, prepare it in a consistent way, run an image-quality model, and produce review-ready outputs. The output is meant to support a human reviewer, not replace one.

## Current MVP

The current prototype supports:

- T1-weighted MRI quality-control prediction
- raw-to-boxed image preprocessing
- standardized 276 × 276 × 276 model input
- brain mask generation
- two-channel model input:
  - prepared T1-weighted image
  - corresponding brain mask
- 3D convolutional neural network prediction
- probability score for poor image quality
- PASS / REVIEW / FAIL triage output
- CSV output for downstream review
- optional Grad-CAM overlays for visual review support
- containerized execution using Apptainer/Singularity

## Repository structure

```text
.
├── src/
│   ├── t1qc_train_model.py
│   ├── t1qc_predict_boxed.py
│   ├── t1qc_predict_boxed_gradcam.py
│   └── t1qc_box_preprocess.py
├── scripts/
│   ├── create_synthetic_artifacts.py
│   ├── run_gradcam_on_testsets.sh
│   └── submit_training_slurm.sh
├── container/
│   ├── Singularity.def
│   └── t1qc_raw_to_qc.sh
├── requirements.txt
└── README.md
```

## What the Toolkit does

The Toolkit takes a T1-weighted MRI scan and produces a quality-control result.

The workflow can start from either:

1. a raw T1-weighted MRI scan, or
2. an already prepared boxed scan with a matching brain mask.

For raw scans, the workflow prepares the image by converting it into a standardized 276 × 276 × 276 volume and generating a corresponding brain mask.

The prepared image and brain mask are then used as the two model inputs. The model produces a probability score for poor image quality and a good/bad prediction.

The prediction score is then organized into a human-in-the-loop quality-control category:

```text
PASS   : likely acceptable scan
REVIEW : borderline or uncertain scan
FAIL   : likely poor-quality scan
```

The goal is to help prioritize review. Clearly acceptable scans can move forward, clearly poor scans can be flagged, and uncertain scans can remain in a manual review pathway.

## Container download

A prebuilt Apptainer/Singularity container is available through Box:

```text
https://upenn.box.com/s/l0q3v3cyipn4hjvawj0lg73xezepwwpu
```

Download the container from Box and save it as:

```bash
Neuroimaging_Toolkit_T1QC.sif
```

After downloading, check that the file is a valid container:

```bash
ls -lh Neuroimaging_Toolkit_T1QC.sif
file Neuroimaging_Toolkit_T1QC.sif
apptainer inspect Neuroimaging_Toolkit_T1QC.sif
```

If the downloaded file is very small or appears to be an HTML file, the Box link was not downloaded as the actual container. In that case, download the `.sif` file through the browser from Box and transfer it to the compute system.

## Running prediction on already boxed scans

Use this mode when the input folder already contains boxed T1-weighted images and corresponding brain masks.

Expected file pattern:

```text
*_T1_brain_box_276.nii.gz
*_brain_mask_box_276.nii.gz
```

Example CPU command:

```bash
apptainer run --cleanenv \
  Neuroimaging_Toolkit_T1QC.sif \
  --input-dir /path/to/boxed_t1_folder \
  --out-csv /path/to/predictions.csv \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

Example GPU command:

```bash
apptainer run --nv --cleanenv \
  Neuroimaging_Toolkit_T1QC.sif \
  --input-dir /path/to/boxed_t1_folder \
  --out-csv /path/to/predictions.csv \
  --batch-size 2 \
  --num-workers 2 \
  --device cuda
```

## Running prediction from raw T1-weighted MRI scans

Use this mode when the scans need preprocessing before prediction.

```bash
apptainer run --cleanenv \
  Neuroimaging_Toolkit_T1QC.sif \
  --raw-input-dir /path/to/raw_t1_folder \
  --boxed-output-dir /path/to/boxed_output_folder \
  --out-csv /path/to/predictions.csv \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

This runs the image preparation step first, then runs the prediction step on the prepared images.

The preprocessing step creates:

```text
boxed T1-weighted image
brain mask
```

These become the two inputs to the model.

## Output CSV

The main output is a CSV file. Important columns may include:

```text
image_path
mask_path
prob_bad
pred_qc_label
qc_category
pass_threshold
fail_threshold
qc_rule
```

Key fields:

- `prob_bad`: model probability score for poor image quality
- `pred_qc_label`: binary model prediction
- `qc_category`: PASS, REVIEW, or FAIL
- `pass_threshold`: threshold used for PASS
- `fail_threshold`: threshold used for FAIL
- `qc_rule`: text description of the decision rule

Current prototype triage rule:

```text
PASS   : prob_bad < 0.45
REVIEW : 0.45 <= prob_bad < 0.55
FAIL   : prob_bad >= 0.55
```

These thresholds are prototype settings and may need to be adjusted for different datasets or use cases.

## Grad-CAM visualizations

The Toolkit can optionally generate Grad-CAM overlays. These overlays provide a rough visual indication of where the model was focusing when making a prediction.

Grad-CAM outputs are intended as a review aid, especially for scans marked REVIEW or FAIL. They should not be treated as a complete explanation of the model decision.

Example command using the source script:

```bash
python src/t1qc_predict_boxed_gradcam.py \
  --checkpoint /path/to/model_checkpoint.pt \
  --input-dir /path/to/boxed_t1_folder \
  --out-csv /path/to/predictions_with_gradcam.csv \
  --threshold 0.55 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --save-gradcam \
  --gradcam-dir /path/to/gradcam_outputs \
  --gradcam-target-layer layer3.1.conv2
```

## Training

The training script is provided here:

```bash
python src/t1qc_train_model.py --help
```

Training requires a CSV manifest and access to prepared T1-weighted MRI images and corresponding masks. MRI data are not included in this repository.

For the current prototype, the training manifest was created locally using a helper script named similar to:

```text
make_manifest_strongMOD_plus_GTbad.py
```

That helper script and the resulting manifest are not included because they were specific to the local dataset layout. For new datasets, create a manifest CSV that matches the expected input columns used by the training script.

## Synthetic artifact generation

Synthetic artifact examples were used during development to expose the model to common image-quality problems.

The synthetic artifact generation script is:

```bash
python scripts/create_synthetic_artifacts.py --help
```

Artifact types used during development included:

- motion
- ghosting
- Gibbs/ringing

## SLURM training template

A portable SLURM template is included:

```bash
MANIFEST=/path/to/train_manifest.csv \
OUT_DIR=/path/to/training_output \
sbatch scripts/submit_training_slurm.sh
```

This script is meant as a starting point and may need to be modified for a specific cluster environment.

## Grad-CAM batch script

A portable Grad-CAM batch script is included:

```bash
CKPT=/path/to/model_checkpoint.pt \
GOOD_DIR=/path/to/boxed_good \
BAD_DIR=/path/to/boxed_bad \
RANDOM_DIR=/path/to/boxed_random \
OUT_ROOT=/path/to/gradcam_outputs \
bash scripts/run_gradcam_on_testsets.sh
```

Any of the dataset variables can be left unset, and that group will be skipped.

## Important notes

This is a research prototype. It is not a validated clinical tool.

The current version should be used for development, testing, and workflow evaluation only. Broader validation across scanners, sites, acquisition protocols, artifact types, and independent datasets is still needed.

The PASS / REVIEW / FAIL thresholds are prototype settings. They should be tuned and validated for the intended use case before being used in a formal workflow.

The repository does not include MRI data, trained model checkpoints, or large intermediate outputs.

## Data and model availability

This GitHub repository contains source code and workflow scripts.

Large files such as MRI data, model checkpoints, and the prebuilt container are not stored directly in the repository. The prebuilt container is provided separately through Box.

## Citation / acknowledgment

If this workflow or code is used or adapted, please acknowledge the Neuroimaging Toolkit / T-Net project as appropriate.
