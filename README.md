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

## Full Video Demonstration

Watch the complete NeuroQC-Assist demonstration (7 minutes).

The video covers T1 MRI preprocessing, AI-based quality
assessment, PASS/REVIEW/FAIL classifications, and
Grad-CAM visualizations.

[▶ Watch or Download the Full Demonstration](https://github.com/SaiKrishnaChaitanya24/NeuroQC-AssistAI-Assisted-Neuroimaging-Quality-Control/raw/refs/heads/main/NeuroQC-Assist_Demo.mp4)

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

## Workflow overview

The Toolkit takes a T1-weighted MRI scan and produces a quality-control result.

The workflow can start from either:

1. a raw T1-weighted MRI scan, or
2. an already prepared boxed scan with a matching brain mask.

For raw scans, the workflow prepares the image by converting it into a standardized 276 × 276 × 276 volume and generating a corresponding brain mask.

The prepared image and brain mask are then used as the two model inputs. The model produces a probability score for poor image quality and a good/bad prediction.

The prediction score is organized into a human-in-the-loop quality-control category:

```text
PASS   : likely acceptable scan
REVIEW : borderline or uncertain scan
FAIL   : likely poor-quality scan
```

The goal is to help prioritize review. Clearly acceptable scans can move forward, clearly poor scans can be flagged, and uncertain scans can remain in a manual review pathway.

## Model architecture

The current model is a custom 3D convolutional neural network named `T1QCHybridCNN`. It is designed for volumetric T1-weighted MRI quality-control prediction.

The model uses a two-channel 3D input:

```text
Channel 0: standardized / masked T1-weighted MRI volume
Channel 1: corresponding brain mask
```

Each input is expected to be a 276 × 276 × 276 volume.

The architecture uses a 3D CNN backbone with residual-style convolutional blocks. In practice, the network includes:

- an initial stem convolution block
- multiple 3D convolutional feature extraction stages
- residual-style blocks with downsampling
- global feature pooling
- prediction heads for quality-control output

The model was developed as a practical prototype rather than as a direct copy of a standard public architecture. It is best described as a custom 3D residual CNN for T1-weighted MRI quality control.

## Model outputs

For each scan, the model produces:

- `prob_bad`: probability score for poor image quality
- `pred_qc_label`: binary good/bad prediction
- `qc_category`: PASS, REVIEW, or FAIL category

During development, artifact-related labels were also used, including:

```text
none
motion
ghosting
gibbs
other
```

The main goal of the current MVP is the binary quality-control and human-in-the-loop triage workflow.

## Training overview

The model was trained using boxed T1-weighted MRI volumes and corresponding brain masks.

The training data included:

- visually acceptable T1-weighted MRI scans
- real poor-quality T1-weighted MRI scans
- synthetic artifact examples generated from acceptable scans

Synthetic artifacts were created to expose the model to common image-quality problems. Artifact types used during development included:

- motion
- ghosting
- Gibbs/ringing

The training setup used source-aware data organization so that scans from different datasets or sources were represented more carefully during training and evaluation. This was done to reduce the chance that the model would only learn source- or scanner-specific patterns.

The training script is included here:

```bash
python src/t1qc_train_model.py --help
```

Training requires a CSV manifest and access to prepared T1-weighted MRI images and corresponding masks.

## Synthetic artifact generation

Synthetic artifact examples were used during development to create additional poor-quality training examples from acceptable scans.

The synthetic artifact generation script is:

```bash
python scripts/create_synthetic_artifacts.py --help
```

The current script is included as a development utility. It may need path and parameter changes for a new dataset.

## Human-in-the-loop triage rule

The current prototype uses a simple triage rule:

```text
PASS   : prob_bad < 0.45
REVIEW : 0.45 <= prob_bad < 0.55
FAIL   : prob_bad >= 0.55
```

These thresholds are prototype settings. They should be tuned and validated for the intended dataset and use case.

The reason for using PASS / REVIEW / FAIL instead of only good / bad is that many MRI scans are not clearly acceptable or clearly unusable. The REVIEW category keeps borderline cases in a human review pathway.

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

## Running the Toolkit on a new dataset

To run the Toolkit on a different dataset, place all T1-weighted MRI scans in one input folder.

Example raw dataset layout:

```text
new_dataset/
├── sub-001_T1w.nii.gz
├── sub-002_T1w.nii.gz
├── sub-003_T1w.nii.gz
└── ...
```

Then choose an output folder for the prepared boxed images and an output CSV path for predictions.

Example:

```bash
apptainer run --cleanenv \
  Neuroimaging_Toolkit_T1QC.sif \
  --raw-input-dir /path/to/new_dataset \
  --boxed-output-dir /path/to/new_dataset_boxed \
  --out-csv /path/to/new_dataset_predictions.csv \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

This command performs two steps:

1. prepares each raw T1-weighted scan into a standardized 276 × 276 × 276 boxed image and brain mask
2. runs the quality-control prediction model on the prepared image and mask

The boxed output folder will contain the prepared model inputs. The CSV file will contain the quality-control predictions.

## Running prediction on already boxed scans

Use this mode only if the input folder already contains boxed T1-weighted images and corresponding brain masks.

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

## Optional Grad-CAM visualizations

The Toolkit can optionally generate Grad-CAM overlays. These overlays provide a rough visual indication of where the model was focusing when making a prediction.

Grad-CAM outputs are intended as a review aid, especially for scans marked REVIEW or FAIL. They should not be treated as a complete explanation of the model decision.

To generate Grad-CAM outputs for a new boxed dataset using the source script:

```bash
python src/t1qc_predict_boxed_gradcam.py \
  --checkpoint /path/to/model_checkpoint.pt \
  --input-dir /path/to/new_dataset_boxed \
  --out-csv /path/to/new_dataset_predictions_with_gradcam.csv \
  --threshold 0.55 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --save-gradcam \
  --gradcam-dir /path/to/new_dataset_gradcam_outputs \
  --gradcam-target-layer layer3.1.conv2
```

If using the prebuilt container, use the standard container prediction command first. Grad-CAM support depends on the container version and included entrypoint.

## Example: complete run on a new dataset

```bash
# 1. Download or copy the container
ls -lh Neuroimaging_Toolkit_T1QC.sif

# 2. Prepare and predict a new raw T1-weighted MRI dataset
apptainer run --cleanenv \
  Neuroimaging_Toolkit_T1QC.sif \
  --raw-input-dir /path/to/new_dataset \
  --boxed-output-dir /path/to/new_dataset_boxed \
  --out-csv /path/to/new_dataset_predictions.csv \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu

# 3. Inspect the prediction CSV
head /path/to/new_dataset_predictions.csv
```

## SLURM training template

A portable SLURM template is included:

```bash
MANIFEST=/path/to/train_manifest.csv \
OUT_DIR=/path/to/training_output \
sbatch scripts/submit_training_slurm.sh
```

This script is meant as a starting point and may need to be modified for a specific cluster environment.

## Important notes

This is a research prototype. It is not a validated clinical tool.

The current version should be used for development, testing, and workflow evaluation only. Broader validation across scanners, sites, acquisition protocols, artifact types, and independent datasets is still needed.

The PASS / REVIEW / FAIL thresholds are prototype settings. They should be tuned and validated for the intended use case before being used in a formal workflow.

The repository does not include MRI data, trained model checkpoints, or large intermediate outputs.

## Data and model availability

This GitHub repository contains source code and workflow scripts.

Large files such as MRI data, model checkpoints, and the prebuilt container are not stored directly in the repository. The prebuilt container is provided separately through Box.

## Citation / acknowledgment

If this workflow or code is used or adapted, please acknowledge the Neuroimaging Toolkit project as appropriate.
