#!/usr/bin/env python3
"""
train_t1qc_boxed276_artifact_detector.py

Train a 3D T1 QC artifact detector on boxed 276x276x276 images.

Designed for:
    /path/to/Boxed-Good-276-Artifacts/
        manifest_t1qc_boxed276_artifacts.csv
        images/good/
        images/bad/motion_mild/
        images/bad/motion_moderate/
        images/bad/ghosting_mild/
        images/bad/ghosting_moderate/
        images/bad/gibbs_mild/
        images/bad/gibbs_moderate/

Main model inputs:
    channel 0 = normalized boxed T1 multiplied by brain/QC mask
    channel 1 = brain/QC mask

Outputs:
    binary_head:
        GOOD / no artifact = 0
        BAD / artifact     = 1

    artifact_head:
        none, motion, ghosting, gibbs, coverage_cutoff, etc.

This version is adapted for 276x276x276 boxed native-grid T1s.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import random
import re
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

import nibabel as nib

try:
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except Exception:
    ndi = None
    HAS_SCIPY = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except Exception:
    autocast = None
    GradScaler = None
    HAS_AMP = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# -----------------------------
# Utilities
# -----------------------------


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print("[%s] %s" % (now(), msg), flush=True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def strip_nii_suffix(path_or_name):
    name = Path(str(path_or_name)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def parse_shape(text):
    text = str(text).strip()
    if "x" in text:
        parts = text.split("x")
    elif "," in text:
        parts = text.split(",")
    else:
        v = int(text)
        return (v, v, v)
    parts = [int(p.strip()) for p in parts if p.strip()]
    if len(parts) != 3:
        raise ValueError("Shape must be like 276 or 276x276x276")
    return tuple(parts)


def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_artifact_type_from_text(text):
    s = str(text).lower()

    if "coverage" in s or "cutoff" in s or "crop" in s or "fov" in s:
        return "coverage_cutoff"
    if "ghost" in s:
        return "ghosting"
    if "gibbs" in s or "ring" in s:
        return "gibbs"
    if "motion" in s:
        return "motion"
    if "bias" in s:
        return "bias"
    if "noise" in s:
        return "noise"
    if "blur" in s:
        return "blur"
    if "none" in s or "good" in s or "clean" in s or "normal" in s:
        return "none"

    return "unknown"


def infer_binary_label_from_row(row, binary_col=None):
    """
    Return:
        0 = good
        1 = bad/artifact
    """
    if binary_col is not None and binary_col in row.index:
        v = row[binary_col]

        if isinstance(v, str):
            s = v.strip().lower()
            if s in ["0", "good", "pass", "passed", "ok", "normal", "none", "clean"]:
                return 0
            if s in ["1", "bad", "fail", "failed", "artifact", "artifactual", "corrupt"]:
                return 1

        try:
            return int(float(v))
        except Exception:
            pass

    for c in ["qc_label", "label", "class", "status"]:
        if c in row.index:
            s = normalize_string(row[c]).lower()
            if s in ["good", "pass", "ok", "normal", "clean"]:
                return 0
            if s in ["bad", "fail", "artifact", "artifactual"]:
                return 1

    art = infer_artifact_type_from_row(row)
    if art == "none":
        return 0

    return 1


def infer_artifact_type_from_row(row, artifact_col=None, image_col=None):
    if artifact_col is not None and artifact_col in row.index:
        art = infer_artifact_type_from_text(row[artifact_col])
        if art != "unknown":
            return art

    for c in [
        "artifact_type",
        "artifact",
        "artifact_name",
        "bad_type",
        "corruption",
        "artifact_label",
        "artifact_variant",
    ]:
        if c in row.index:
            art = infer_artifact_type_from_text(row[c])
            if art != "unknown":
                return art

    for c in [image_col, "image_path", "path", "filename", "file"]:
        if c is not None and c in row.index:
            art = infer_artifact_type_from_text(row[c])
            if art != "unknown":
                return art

    for c in ["binary_label", "is_bad", "label", "qc_label"]:
        if c in row.index:
            try:
                b = infer_binary_label_from_row(row, c)
                if b == 0:
                    return "none"
            except Exception:
                pass

    return "unknown"


def infer_source_id_from_path_or_id(x):
    """
    Derive source ID by removing synthetic artifact suffixes.
    """
    stem = strip_nii_suffix(Path(str(x)).name)
    s = stem

    patterns = [
        r"__motion.*$",
        r"__ghosting.*$",
        r"__ghost.*$",
        r"__gibbs.*$",
        r"__ringing.*$",
        r"__coverage[_-]?cutoff.*$",
        r"__cutoff.*$",
        r"__bias.*$",
        r"__noise.*$",
        r"__blur.*$",
        r"__bad.*$",
        r"__good.*$",
        r"_T1_brain_box_276$",
        r"_brain_box_276$",
        r"_brain_mask_box_276$",
    ]

    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    return s


def resolve_path_relative_to_manifest(value, manifest_dir):
    s = normalize_string(value)
    if s == "":
        return ""

    p = Path(s)
    if p.is_absolute():
        return str(p)

    if p.exists():
        return str(p)

    return str(manifest_dir / p)


# -----------------------------
# Manifest
# -----------------------------


def prepare_manifest(manifest_path, out_dir, seed=42, train_frac=0.70, val_frac=0.15):
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent

    image_col = first_existing_col(df, [
        "image_path",
        "path",
        "filename",
        "file",
        "masked_image_path",
        "issue_masked_image_path",
    ])

    if image_col is None:
        raise RuntimeError("Could not find image path column. Expected image_path/path/filename/file.")

    # This is important for your generated artifact manifest.
    qc_mask_col = first_existing_col(df, [
        "qc_mask_path",
        "source_mask_path",
        "brain_mask_path",
        "mask_path",
        "mask",
        "qc_mask",
    ])

    binary_col = first_existing_col(df, [
        "binary_label",
        "is_bad",
        "bad_label",
        "qc_binary",
        "label_binary",
        "qc_label",
        "label",
    ])

    artifact_col = first_existing_col(df, [
        "artifact_type",
        "artifact",
        "artifact_name",
        "bad_type",
        "corruption",
        "artifact_label",
        "artifact_variant",
    ])

    source_col = first_existing_col(df, [
        "source_id",
        "source",
        "source_image",
        "base_id",
        "original_id",
        "subject_source",
    ])

    split_col = first_existing_col(df, [
        "split",
        "fold",
        "partition",
    ])

    df = df.copy()

    df["_image_path"] = df[image_col].apply(lambda v: resolve_path_relative_to_manifest(v, base_dir))

    if qc_mask_col is not None:
        df["_qc_mask_path"] = df[qc_mask_col].apply(lambda v: resolve_path_relative_to_manifest(v, base_dir))
    else:
        df["_qc_mask_path"] = ""

    binary_labels = []
    artifact_types = []
    source_ids = []

    for _, row in df.iterrows():
        b = infer_binary_label_from_row(row, binary_col=binary_col)
        art = infer_artifact_type_from_row(row, artifact_col=artifact_col, image_col=image_col)

        if b == 0:
            art = "none"

        if art == "unknown" and b == 1:
            art = infer_artifact_type_from_text(row["_image_path"])
            if art == "unknown":
                art = "other"

        if source_col is not None:
            src = normalize_string(row[source_col])
            if src == "":
                src = infer_source_id_from_path_or_id(row["_image_path"])
        else:
            src = infer_source_id_from_path_or_id(row["_image_path"])

        binary_labels.append(int(b))
        artifact_types.append(str(art))
        source_ids.append(str(src))

    df["_binary_label"] = binary_labels
    df["_artifact_type"] = artifact_types
    df["_source_id"] = source_ids

    preferred = [
        "none",
        "motion",
        "ghosting",
        "gibbs",
        "coverage_cutoff",
        "bias",
        "noise",
        "blur",
        "other",
    ]

    present = sorted(set(df["_artifact_type"].astype(str).tolist()))
    classes = []

    for c in preferred:
        if c in present:
            classes.append(c)

    for c in present:
        if c not in classes:
            classes.append(c)

    if "none" not in classes:
        classes = ["none"] + classes

    artifact_to_idx = {c: i for i, c in enumerate(classes)}
    df["_artifact_label"] = df["_artifact_type"].map(artifact_to_idx).astype(int)

    # MODERATE-ONLY TRAINING FILTER: keep all GOOD rows, keep only MODERATE synthetic BAD rows.
    # This drops mild generated artifacts from train/val/test splits.
    if "artifact_variant" in df.columns:
        _variant_text = df["artifact_variant"].astype(str).str.lower()
    elif "severity" in df.columns:
        _variant_text = df["severity"].astype(str).str.lower()
    else:
        _variant_text = df["_image_path"].astype(str).str.lower()

    _keep_moderate = (df["_binary_label"] == 0) | ((df["_binary_label"] == 1) & _variant_text.str.contains("moderate", na=False))
    _dropped_mild = int((~_keep_moderate).sum())
    df = df[_keep_moderate].reset_index(drop=True)
    log("MODERATE-ONLY FILTER: dropped %d non-moderate BAD rows; kept %d rows" % (_dropped_mild, len(df)))


    # Split by source ID unless manifest already has train/val/test.
    use_existing_split = False

    if split_col is not None:
        s = df[split_col].astype(str).str.lower()
        mapped = np.where(
            s.str.contains("train"),
            "train",
            np.where(
                s.str.contains("val"),
                "val",
                np.where(s.str.contains("test"), "test", s),
            ),
        )
        df["_split"] = mapped
        if set(df["_split"].unique()).intersection(set(["train", "val", "test"])):
            use_existing_split = True

    if not use_existing_split:
        srcs = sorted(df["_source_id"].unique().tolist())
        rng = np.random.RandomState(int(seed))
        rng.shuffle(srcs)

        n = len(srcs)
        n_train = int(round(float(train_frac) * n))
        n_val = int(round(float(val_frac) * n))

        train_src = set(srcs[:n_train])
        val_src = set(srcs[n_train:n_train + n_val])

        def assign_split(src):
            if src in train_src:
                return "train"
            if src in val_src:
                return "val"
            return "test"

        df["_split"] = df["_source_id"].apply(assign_split)

    std_path = Path(out_dir) / "standardized_manifest.csv"
    save_cols = [
        "_image_path",
        "_qc_mask_path",
        "_binary_label",
        "_artifact_type",
        "_artifact_label",
        "_source_id",
        "_split",
    ]
    df[save_cols].to_csv(std_path, index=False)

    log("Manifest rows: %d" % len(df))
    log("Image column: %s" % image_col)
    log("Mask column: %s" % str(qc_mask_col))
    log("Binary column: %s" % str(binary_col))
    log("Artifact column: %s" % str(artifact_col))
    log("Source column: %s" % str(source_col))
    log("Split column: %s" % str(split_col))
    log("Artifact classes: %s" % json.dumps(artifact_to_idx, indent=2))
    log("Binary counts: %s" % dict(Counter(df["_binary_label"].tolist())))
    log("Artifact counts: %s" % dict(Counter(df["_artifact_type"].tolist())))
    log("Split counts: %s" % dict(Counter(df["_split"].tolist())))
    log("Source counts by split: %s" % df.groupby("_split")["_source_id"].nunique().to_dict())
    log("Saved standardized manifest: %s" % std_path)

    return df, artifact_to_idx


# -----------------------------
# Image preprocessing
# -----------------------------


def load_nifti_float(path):
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise RuntimeError("Expected 3D image, got shape %s for %s" % (arr.shape, path))

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def assert_required_shape(arr, name, required_shape):
    shp = tuple(np.asarray(arr).shape)
    req = tuple(required_shape)

    if shp != req:
        raise RuntimeError(
            "%s has shape %s, expected %s. "
            "This training script does not resize. Use boxed 276x276x276 images and masks."
            % (name, shp, req)
        )


def make_auto_qc_mask(x, threshold_percentile=15.0, closing=2, dilate=1):
    """
    Fallback mask if qc/source mask is missing.
    Prefer real boxed brain mask whenever possible.
    """
    xx = np.asarray(x, dtype=np.float32)
    finite = xx[np.isfinite(xx)]

    if finite.size == 0:
        return np.zeros_like(xx, dtype=np.float32)

    nz = xx[np.abs(xx) > 1e-8]
    vals = nz if nz.size > 100 else finite

    thr = np.percentile(vals, threshold_percentile)
    if not np.isfinite(thr):
        thr = 0.0

    mask = xx > thr

    if HAS_SCIPY:
        try:
            if closing and closing > 0:
                mask = ndi.binary_closing(mask, iterations=int(closing))
            mask = ndi.binary_fill_holes(mask)
            if dilate and dilate > 0:
                mask = ndi.binary_dilation(mask, iterations=int(dilate))
        except Exception:
            pass

    return mask.astype(np.float32)


def brain_percentile_normalize(x, mask=None, p_low=0.5, p_high=99.5, eps=1e-6):
    """
    Normalize T1 intensity using only voxels inside the brain/QC mask.
    """
    x = np.asarray(x, dtype=np.float32)

    if mask is not None:
        m = np.asarray(mask) > 0.5
        vals = x[m]
    else:
        vals = x[np.abs(x) > eps]

    vals = vals[np.isfinite(vals)]

    if vals.size < 100:
        vals = x[np.isfinite(x)]

    if vals.size < 100:
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(vals, float(p_low)))
    hi = float(np.percentile(vals, float(p_high)))

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo + eps:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    if hi <= lo + eps:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = np.clip(x, lo, hi)
        y = (y - lo) / (hi - lo + eps)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

    return y


def mild_domain_augment(x, mask=None, rng=None):
    """
    Mild scanner/style augmentation. This should not change GOOD/BAD label.
    """
    if rng is None:
        rng = np.random

    y = x.astype(np.float32).copy()

    scale = rng.uniform(0.85, 1.15)
    shift = rng.uniform(-0.04, 0.04)
    y = y * scale + shift

    y = np.clip(y, 0.0, 1.0)

    gamma = rng.uniform(0.80, 1.25)
    y = np.power(y, gamma).astype(np.float32)

    if rng.rand() < 0.70:
        d, h, w = y.shape
        zd = np.linspace(-1.0, 1.0, d, dtype=np.float32)[:, None, None]
        yh = np.linspace(-1.0, 1.0, h, dtype=np.float32)[None, :, None]
        xw = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, None, :]

        a = rng.uniform(-0.12, 0.12)
        b = rng.uniform(-0.12, 0.12)
        c = rng.uniform(-0.12, 0.12)

        bias = 1.0 + a * zd + b * yh + c * xw
        bias = np.clip(bias, 0.75, 1.25).astype(np.float32)
        y = y * bias

    if rng.rand() < 0.75:
        sigma = rng.uniform(0.0, 0.025)
        noise = rng.normal(0.0, sigma, size=y.shape).astype(np.float32)

        if mask is not None:
            y = y + noise * (mask > 0.5).astype(np.float32)
        else:
            y = y + noise

    if HAS_SCIPY and rng.rand() < 0.20:
        sigma = rng.uniform(0.15, 0.45)
        y = ndi.gaussian_filter(y, sigma=sigma).astype(np.float32)

    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y


def make_two_channel_input(x, mask, required_shape):
    """
    channel 0 = skull-stripped normalized image
    channel 1 = brain/QC mask
    """
    x = np.asarray(x, dtype=np.float32)
    m = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)

    assert_required_shape(x, "image", required_shape)
    assert_required_shape(m, "mask", required_shape)

    skull = (x * m).astype(np.float32)
    inp = np.stack([skull, m], axis=0).astype(np.float32)

    return inp, m[None, ...].astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------


class T1QCDataset(Dataset):
    def __init__(
        self,
        df,
        required_shape,
        mask_mode="qc",
        brain_norm=True,
        domain_aug=False,
        training=False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.required_shape = tuple(required_shape)
        self.mask_mode = str(mask_mode)
        self.brain_norm = bool(brain_norm)
        self.domain_aug = bool(domain_aug)
        self.training = bool(training)

    def __len__(self):
        return len(self.df)

    def _load_row(self, row):
        image_path = row["_image_path"]
        x = load_nifti_float(image_path)

        assert_required_shape(x, "image %s" % image_path, self.required_shape)

        mask = None

        if self.mask_mode == "none":
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mp = normalize_string(row.get("_qc_mask_path", ""))
            if mp and Path(mp).exists():
                mask = load_nifti_float(mp)
                mask = (mask > 0.5).astype(np.float32)
            else:
                mask = make_auto_qc_mask(x)

        assert_required_shape(mask, "mask for %s" % image_path, self.required_shape)

        if self.brain_norm:
            x = brain_percentile_normalize(x, mask=mask)
        else:
            vals = x[np.isfinite(x)]
            if vals.size > 100:
                lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
                x = np.clip(x, lo, hi)
                x = (x - lo) / (hi - lo + 1e-6)
                x = np.clip(x, 0.0, 1.0).astype(np.float32)
            else:
                x = np.zeros_like(x, dtype=np.float32)

        if self.training and self.domain_aug:
            x = mild_domain_augment(x, mask=mask)

        x2, mask2 = make_two_channel_input(x, mask, self.required_shape)

        b = int(row["_binary_label"])
        a = int(row["_artifact_label"])

        return {
            "image": torch.from_numpy(x2),
            "mask": torch.from_numpy(mask2),
            "binary": torch.tensor(b, dtype=torch.float32),
            "artifact": torch.tensor(a, dtype=torch.long),
            "image_path": str(image_path),
            "source_id": str(row["_source_id"]),
            "artifact_type": str(row["_artifact_type"]),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        return self._load_row(row)


class SourceBalancedT1QCDataset(T1QCDataset):
    """
    Each epoch samples approximately:
        good_per_source GOOD rows
        bad_per_source BAD rows
    per source ID.

    This prevents 1 good + 6 generated artifacts from overwhelming the model.
    """
    def __init__(self, df, good_per_source=1, bad_per_source=1, seed=42, **kwargs):
        super(SourceBalancedT1QCDataset, self).__init__(df, training=True, **kwargs)

        self.good_per_source = int(good_per_source)
        self.bad_per_source = int(bad_per_source)
        self.seed = int(seed)
        self.epoch = 0

        self.groups = defaultdict(lambda: {"good": [], "bad": []})

        for i, row in self.df.iterrows():
            src = str(row["_source_id"])
            if int(row["_binary_label"]) == 0:
                self.groups[src]["good"].append(int(i))
            else:
                self.groups[src]["bad"].append(int(i))

        self.sources = sorted(self.groups.keys())
        self.selected = []
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        rng = np.random.RandomState(self.seed + self.epoch * 7919)

        selected = []
        complete = 0

        for src in self.sources:
            g = self.groups[src]["good"]
            b = self.groups[src]["bad"]

            if len(g) > 0 and len(b) > 0:
                complete += 1

            if len(g) > 0:
                replace = len(g) < self.good_per_source
                selected.extend(
                    rng.choice(g, size=self.good_per_source, replace=replace).tolist()
                )

            if len(b) > 0:
                replace = len(b) < self.bad_per_source
                selected.extend(
                    rng.choice(b, size=self.bad_per_source, replace=replace).tolist()
                )

        rng.shuffle(selected)
        self.selected = [int(i) for i in selected]

        counts = Counter([int(self.df.iloc[i]["_binary_label"]) for i in self.selected])
        log(
            "SourceBalanced epoch %03d: selected rows=%d, complete_sources=%d, binary_counts=%s"
            % (self.epoch, len(self.selected), complete, dict(counts))
        )

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        real_idx = self.selected[int(idx)]
        row = self.df.iloc[real_idx]
        return self._load_row(row)


def collate_keep_strings(batch):
    out = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [b[k] for b in batch]

        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals

    return out


def move_batch(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


# -----------------------------
# Model
# -----------------------------


class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)

        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class T1QCHybridCNN(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_artifacts=4,
        base_channels=8,
        dropout=0.25,
        pool_mode="hybrid_global_masked_avgmax",
    ):
        super(T1QCHybridCNN, self).__init__()

        self.in_channels = int(in_channels)
        self.num_artifacts = int(num_artifacts)
        self.base_channels = int(base_channels)
        self.pool_mode = str(pool_mode)

        c = self.base_channels

        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(c, c, blocks=2, stride=1)
        self.layer2 = self._make_layer(c, c * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(c * 2, c * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(c * 4, c * 8, blocks=2, stride=2)

        feat_ch = c * 8

        if self.pool_mode == "global_avg":
            pooled_dim = feat_ch
        elif self.pool_mode == "hybrid_global_masked_avgmax":
            pooled_dim = feat_ch * 3
        else:
            raise ValueError("Unknown pool_mode: %s" % self.pool_mode)

        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.binary_head = nn.Linear(256, 1)
        self.artifact_head = nn.Linear(256, self.num_artifacts)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock3D(in_ch, out_ch, stride=stride)]

        for _ in range(1, int(blocks)):
            layers.append(BasicBlock3D(out_ch, out_ch, stride=1))

        return nn.Sequential(*layers)

    def extract_feature_map(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def pool_features(self, feat, mask=None):
        b, c, d, h, w = feat.shape

        gavg = F.adaptive_avg_pool3d(feat, 1).view(b, c)

        if self.pool_mode == "global_avg":
            return gavg

        if mask is None:
            m = torch.ones((b, 1, d, h, w), dtype=feat.dtype, device=feat.device)
        else:
            m = F.interpolate(mask.float(), size=(d, h, w), mode="nearest")
            m = (m > 0.5).float()

        denom = m.sum(dim=(2, 3, 4)).clamp_min(1.0)
        mavg = (feat * m).sum(dim=(2, 3, 4)) / denom

        neg_fill = -1.0e4 if feat.dtype in (torch.float16, torch.bfloat16) else -1.0e6
        masked = feat.masked_fill(m.expand_as(feat) <= 0.5, neg_fill)
        mmax = masked.view(b, c, -1).max(dim=2)[0]

        gmax = F.adaptive_max_pool3d(feat, 1).view(b, c)
        valid = (m.sum(dim=(2, 3, 4)) > 0).float()
        mmax = mmax * valid + gmax * (1.0 - valid)

        return torch.cat([gavg, mavg, mmax], dim=1)

    def forward(self, x, mask=None):
        feat = self.extract_feature_map(x)
        pooled = self.pool_features(feat, mask=mask)
        z = self.shared(pooled)

        binary_logit = self.binary_head(z).view(-1)
        artifact_logits = self.artifact_head(z)

        return {
            "binary_logit": binary_logit,
            "artifact_logits": artifact_logits,
            "features": z,
        }


# -----------------------------
# Losses / metrics
# -----------------------------


def weighted_binary_bce_with_logits(logits, targets, good_weight=1.5, bad_weight=1.0):
    targets = targets.float().view_as(logits)

    per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    gw = torch.as_tensor(float(good_weight), dtype=per.dtype, device=per.device)
    bw = torch.as_tensor(float(bad_weight), dtype=per.dtype, device=per.device)

    weights = torch.where(targets >= 0.5, bw, gw)
    return (per * weights).mean()


def batch_prior_regularization(logits, targets):
    targets = targets.float().view_as(logits)
    prob = torch.sigmoid(logits)
    return (prob.mean() - targets.mean()).pow(2)


def good_margin_regularization(logits, targets, margin=0.35):
    targets = targets.float().view_as(logits)
    prob = torch.sigmoid(logits)

    good = targets < 0.5

    if not bool(good.any()):
        return torch.tensor(0.0, dtype=prob.dtype, device=prob.device)

    return F.relu(prob[good] - float(margin)).pow(2).mean()


def confusion_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return tp, tn, fp, fn


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def f1_score_binary(y_true, y_pred):
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    return safe_div(2 * precision * recall, precision + recall)


def macro_f1_multiclass(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    f1s = []

    for k in range(int(num_classes)):
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())

        if (tp + fp + fn) == 0:
            continue

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)

        f1s.append(safe_div(2 * p * r, p + r))

    if len(f1s) == 0:
        return 0.0

    return float(np.mean(f1s))


def compute_metrics(y_true, prob_bad, artifact_true, artifact_pred, threshold, num_artifacts):
    y_true = np.asarray(y_true).astype(int)
    prob_bad = np.asarray(prob_bad).astype(float)
    y_pred = (prob_bad >= float(threshold)).astype(int)

    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    bad_recall = safe_div(tp, tp + fn)
    good_recall = safe_div(tn, tn + fp)
    balacc = 0.5 * (bad_recall + good_recall)

    bad_precision = safe_div(tp, tp + fp)
    f1 = f1_score_binary(y_true, y_pred)

    auc = np.nan
    if HAS_SKLEARN and len(set(y_true.tolist())) == 2:
        try:
            auc = float(roc_auc_score(y_true, prob_bad))
        except Exception:
            auc = np.nan

    artifact_true = np.asarray(artifact_true).astype(int)
    artifact_pred = np.asarray(artifact_pred).astype(int)

    art_acc = float((artifact_true == artifact_pred).mean()) if len(artifact_true) else 0.0
    art_macro_f1 = macro_f1_multiclass(artifact_true, artifact_pred, num_artifacts)

    good_mask = y_true == 0
    bad_mask = y_true == 1

    mean_prob_bad_good = float(np.mean(prob_bad[good_mask])) if good_mask.any() else np.nan
    mean_prob_bad_bad = float(np.mean(prob_bad[bad_mask])) if bad_mask.any() else np.nan

    pred_bad_rate = float(np.mean(y_pred == 1)) if len(y_pred) else 0.0
    false_bad_rate = safe_div(fp, fp + tn)
    false_good_rate = safe_div(fn, fn + tp)

    return {
        "binary_accuracy": acc,
        "binary_balanced_accuracy": balacc,
        "binary_f1": f1,
        "binary_auc": auc,
        "good_recall_specificity": good_recall,
        "bad_recall_sensitivity": bad_recall,
        "bad_precision": bad_precision,
        "artifact_accuracy": art_acc,
        "artifact_macro_f1": art_macro_f1,
        "mean_prob_bad_good_true": mean_prob_bad_good,
        "mean_prob_bad_bad_true": mean_prob_bad_bad,
        "pred_bad_rate": pred_bad_rate,
        "false_bad_rate": false_bad_rate,
        "false_good_rate": false_good_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def validation_score(metrics, min_good=0.70, min_bad=0.50, imbalance_penalty=0.50, art_weight=0.05):
    good = float(metrics.get("good_recall_specificity", 0.0))
    bad = float(metrics.get("bad_recall_sensitivity", 0.0))
    balacc = float(metrics.get("binary_balanced_accuracy", 0.0))
    artf1 = float(metrics.get("artifact_macro_f1", 0.0))

    min_recall = min(good, bad)
    imbalance = abs(good - bad)

    score = balacc + min_recall - float(imbalance_penalty) * imbalance + float(art_weight) * artf1

    if good < min_good:
        score -= 2.0 * (min_good - good)

    if bad < min_bad:
        score -= 1.0 * (min_bad - bad)

    return float(score)


def compute_artifact_class_weights(df_train, num_artifacts):
    counts = Counter(df_train["_artifact_label"].astype(int).tolist())
    total = float(sum(counts.values()))

    weights = []

    for k in range(int(num_artifacts)):
        c = float(counts.get(k, 0))

        if c <= 0:
            w = 0.0
        else:
            w = total / (float(num_artifacts) * c)

        weights.append(w)

    weights = np.asarray(weights, dtype=np.float32)

    nonzero = weights[weights > 0]

    if nonzero.size > 0:
        weights[weights == 0] = float(nonzero.max())
        weights = np.clip(weights, 0.25, 5.0)
        weights = weights / np.mean(weights)
    else:
        weights = np.ones((num_artifacts,), dtype=np.float32)

    return weights


@torch.no_grad()
def evaluate(model, loader, device, threshold, num_artifacts, out_csv=None, idx_to_artifact=None):
    model.eval()

    all_y = []
    all_prob = []
    all_art_true = []
    all_art_pred = []
    rows = []

    for batch in loader:
        batch = move_batch(batch, device)

        x = batch["image"]
        mask = batch["mask"]
        y = batch["binary"]
        art = batch["artifact"]

        out = model(x, mask=mask)

        logits = out["binary_logit"]
        probs = torch.sigmoid(logits)

        art_logits = out["artifact_logits"]
        art_pred = torch.argmax(art_logits, dim=1)

        y_np = y.detach().cpu().numpy().astype(int)
        p_np = probs.detach().cpu().numpy().astype(float)
        art_np = art.detach().cpu().numpy().astype(int)
        artp_np = art_pred.detach().cpu().numpy().astype(int)

        all_y.extend(y_np.tolist())
        all_prob.extend(p_np.tolist())
        all_art_true.extend(art_np.tolist())
        all_art_pred.extend(artp_np.tolist())

        if out_csv is not None:
            for i in range(len(y_np)):
                pred_bin = int(p_np[i] >= float(threshold))

                if idx_to_artifact is not None:
                    pred_art_name = idx_to_artifact.get(int(artp_np[i]), str(int(artp_np[i])))
                    true_art_name = idx_to_artifact.get(int(art_np[i]), str(int(art_np[i])))
                else:
                    pred_art_name = str(int(artp_np[i]))
                    true_art_name = str(int(art_np[i]))

                rows.append({
                    "image_path": batch["image_path"][i],
                    "source_id": batch["source_id"][i],
                    "true_binary_label": int(y_np[i]),
                    "prob_bad": float(p_np[i]),
                    "pred_binary_label": pred_bin,
                    "true_artifact_label": int(art_np[i]),
                    "pred_artifact_label": int(artp_np[i]),
                    "true_artifact_type": true_art_name,
                    "pred_artifact_type": pred_art_name,
                })

    metrics = compute_metrics(
        all_y,
        all_prob,
        all_art_true,
        all_art_pred,
        threshold,
        num_artifacts,
    )

    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return metrics


def save_checkpoint(path, model, optimizer, scaler, epoch, score, metrics, args, artifact_to_idx):
    ckpt = {
        "epoch": int(epoch),
        "score": float(score),
        "val_metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "artifact_to_idx": artifact_to_idx,
        "idx_to_artifact": {int(v): k for k, v in artifact_to_idx.items()},
        "model_name": "T1QCHybridCNN_boxed276_artifact_detector_mask2ch",
    }

    torch.save(ckpt, str(path))


# -----------------------------
# Train
# -----------------------------


def train(args):
    set_seed(args.seed)

    required_shape = parse_shape(args.required_shape)

    args.input_mode = "boxed276_skullstrip_plus_mask_2ch"
    args.in_channels = 2
    args.required_shape_tuple = list(required_shape)

    out_dir = ensure_dir(args.out_dir)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    df, artifact_to_idx = prepare_manifest(
        args.manifest,
        out_dir=out_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    idx_to_artifact = {int(v): k for k, v in artifact_to_idx.items()}
    num_artifacts = len(artifact_to_idx)

    df_train = df[df["_split"] == "train"].reset_index(drop=True)
    df_val = df[df["_split"] == "val"].reset_index(drop=True)
    df_test = df[df["_split"] == "test"].reset_index(drop=True)

    df_train.to_csv(out_dir / "split_train.csv", index=False)
    df_val.to_csv(out_dir / "split_val.csv", index=False)
    df_test.to_csv(out_dir / "split_test.csv", index=False)

    if len(df_train) == 0 or len(df_val) == 0:
        raise RuntimeError("Train/val split is empty. Check manifest or split columns.")

    log("Rows train/val/test: %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    log("Sources train/val/test: %d %d %d" % (
        df_train["_source_id"].nunique(),
        df_val["_source_id"].nunique(),
        df_test["_source_id"].nunique(),
    ))

    log("Required shape: %s" % (required_shape,))
    log("Mask mode: %s" % args.mask_mode)
    log("Brain norm: %s" % str(args.brain_norm))
    log("Domain aug: %s" % str(args.domain_aug))
    log("Source balanced: %s" % str(args.source_balanced))
    log("Good per source: %d" % args.good_per_source)
    log("Bad per source: %d" % args.bad_per_source)
    log("Fixed threshold: %.4f" % args.fixed_threshold)

    if args.source_balanced:
        train_ds = SourceBalancedT1QCDataset(
            df_train,
            good_per_source=args.good_per_source,
            bad_per_source=args.bad_per_source,
            seed=args.seed,
            required_shape=required_shape,
            mask_mode=args.mask_mode,
            brain_norm=args.brain_norm,
            domain_aug=args.domain_aug,
        )
        shuffle_train = False
    else:
        train_ds = T1QCDataset(
            df_train,
            required_shape=required_shape,
            mask_mode=args.mask_mode,
            brain_norm=args.brain_norm,
            domain_aug=args.domain_aug,
            training=True,
        )
        shuffle_train = True

    val_ds = T1QCDataset(
        df_val,
        required_shape=required_shape,
        mask_mode=args.mask_mode,
        brain_norm=args.brain_norm,
        domain_aug=False,
        training=False,
    )

    test_ds = None
    if len(df_test) > 0:
        test_ds = T1QCDataset(
            df_test,
            required_shape=required_shape,
            mask_mode=args.mask_mode,
            brain_norm=args.brain_norm,
            domain_aug=False,
            training=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_strings,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_strings,
        drop_last=False,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_keep_strings,
            drop_last=False,
        )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Train loader has zero batches. Try smaller source-balanced settings or set --batch-size 1."
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log("Device: %s" % str(device))

    model = T1QCHybridCNN(
        in_channels=2,
        num_artifacts=num_artifacts,
        base_channels=args.base_channels,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
    ).to(device)

    log("Model: T1QCHybridCNN")
    log("Model input channels: 2")
    log("Channel 0: normalized T1 multiplied by mask")
    log("Channel 1: brain/QC mask")
    log("Number of artifact classes: %d" % num_artifacts)

    art_weights_np = compute_artifact_class_weights(df_train, num_artifacts)
    log("Artifact CE class weights: %s" % art_weights_np.tolist())

    art_weights = torch.tensor(art_weights_np, dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=art_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )

    use_amp = bool(args.amp and HAS_AMP and device.type == "cuda")
    scaler = GradScaler() if use_amp else None
    log("AMP: %s" % str(use_amp))

    # Sanity batch
    try:
        b0 = next(iter(train_loader))
        log("Sanity batch image shape: %s" % (tuple(b0["image"].shape),))
        log("Sanity batch mask shape: %s" % (tuple(b0["mask"].shape),))
        log("Sanity image min/mean/max: %.5f %.5f %.5f" % (
            float(b0["image"].min()),
            float(b0["image"].mean()),
            float(b0["image"].max()),
        ))
        log("Sanity mask nonzero fraction: %.5f" % float((b0["mask"] > 0.5).float().mean()))
        log("Sanity binary labels: %s" % b0["binary"].numpy().astype(int).tolist())
        log("Sanity artifact labels: %s" % b0["artifact"].numpy().astype(int).tolist())
    except Exception:
        log("WARNING: sanity batch failed")
        traceback.print_exc()

    metrics_path = out_dir / "epoch_metrics.csv"

    best_score = -1.0e9
    best_epoch = -1
    rows_metrics = []

    for epoch in range(1, args.epochs + 1):
        if args.source_balanced and hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        model.train()

        epoch_losses = []
        epoch_bin_losses = []
        epoch_art_losses = []
        epoch_batch_prior_losses = []
        epoch_good_margin_losses = []

        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)

            x = batch["image"]
            mask = batch["mask"]
            y = batch["binary"]
            art = batch["artifact"]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    out = model(x, mask=mask)

                    loss_bin = weighted_binary_bce_with_logits(
                        out["binary_logit"],
                        y,
                        good_weight=args.good_loss_weight,
                        bad_weight=args.bad_loss_weight,
                    )

                    loss_art = ce_loss(out["artifact_logits"], art)

                    art_w_epoch = 0.0 if epoch <= args.binary_only_epochs else float(args.artifact_loss_weight)

                    loss_batch_prior = batch_prior_regularization(out["binary_logit"], y)
                    loss_good_margin = good_margin_regularization(
                        out["binary_logit"],
                        y,
                        margin=args.good_margin,
                    )

                    loss = (
                        args.binary_loss_weight * loss_bin
                        + art_w_epoch * loss_art
                        + args.batch_prior_weight * loss_batch_prior
                        + args.good_margin_weight * loss_good_margin
                    )

                scaler.scale(loss).backward()

                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

            else:
                out = model(x, mask=mask)

                loss_bin = weighted_binary_bce_with_logits(
                    out["binary_logit"],
                    y,
                    good_weight=args.good_loss_weight,
                    bad_weight=args.bad_loss_weight,
                )

                loss_art = ce_loss(out["artifact_logits"], art)

                art_w_epoch = 0.0 if epoch <= args.binary_only_epochs else float(args.artifact_loss_weight)

                loss_batch_prior = batch_prior_regularization(out["binary_logit"], y)
                loss_good_margin = good_margin_regularization(
                    out["binary_logit"],
                    y,
                    margin=args.good_margin,
                )

                loss = (
                    args.binary_loss_weight * loss_bin
                    + art_w_epoch * loss_art
                    + args.batch_prior_weight * loss_batch_prior
                    + args.good_margin_weight * loss_good_margin
                )

                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_bin_losses.append(float(loss_bin.detach().cpu()))
            epoch_art_losses.append(float(loss_art.detach().cpu()))
            epoch_batch_prior_losses.append(float(loss_batch_prior.detach().cpu()))
            epoch_good_margin_losses.append(float(loss_good_margin.detach().cpu()))

        scheduler.step()

        val_pred_csv = out_dir / ("val_predictions_epoch_%03d.csv" % epoch)

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            threshold=args.fixed_threshold,
            num_artifacts=num_artifacts,
            out_csv=val_pred_csv,
            idx_to_artifact=idx_to_artifact,
        )

        score = validation_score(
            val_metrics,
            min_good=args.min_good_recall_for_best,
            min_bad=args.min_bad_recall_for_best,
            imbalance_penalty=args.best_imbalance_penalty,
            art_weight=args.best_artifact_f1_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan,
            "train_binary_loss": float(np.mean(epoch_bin_losses)) if epoch_bin_losses else np.nan,
            "train_artifact_loss": float(np.mean(epoch_art_losses)) if epoch_art_losses else np.nan,
            "train_batch_prior_loss": float(np.mean(epoch_batch_prior_losses)) if epoch_batch_prior_losses else np.nan,
            "train_good_margin_loss": float(np.mean(epoch_good_margin_losses)) if epoch_good_margin_losses else np.nan,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_score": float(score),
        }

        row.update({"val_" + k: v for k, v in val_metrics.items()})
        rows_metrics.append(row)
        pd.DataFrame(rows_metrics).to_csv(metrics_path, index=False)

        log(
            (
                "Epoch %03d | loss %.4f bin %.4f art %.4f | "
                "score %.4f balacc %.4f bad_recall %.4f good_recall %.4f "
                "pred_bad %.4f false_bad %.4f meanP_good %.4f meanP_bad %.4f artF1 %.4f auc %.4f"
            )
            % (
                epoch,
                row["train_loss"],
                row["train_binary_loss"],
                row["train_artifact_loss"],
                score,
                float(val_metrics.get("binary_balanced_accuracy", 0.0)),
                float(val_metrics.get("bad_recall_sensitivity", 0.0)),
                float(val_metrics.get("good_recall_specificity", 0.0)),
                float(val_metrics.get("pred_bad_rate", 0.0)),
                float(val_metrics.get("false_bad_rate", 0.0)),
                float(val_metrics.get("mean_prob_bad_good_true", np.nan))
                if not np.isnan(val_metrics.get("mean_prob_bad_good_true", np.nan))
                else -1.0,
                float(val_metrics.get("mean_prob_bad_bad_true", np.nan))
                if not np.isnan(val_metrics.get("mean_prob_bad_bad_true", np.nan))
                else -1.0,
                float(val_metrics.get("artifact_macro_f1", 0.0)),
                float(val_metrics.get("binary_auc", np.nan))
                if not np.isnan(val_metrics.get("binary_auc", np.nan))
                else -1.0,
            )
        )

        if args.save_every_epoch:
            ckpt_path = out_dir / ("checkpoint_epoch_%03d.pt" % epoch)
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scaler,
                epoch,
                score,
                val_metrics,
                args,
                artifact_to_idx,
            )

        # Save EVERY epoch checkpoint, not only the best epoch.
        epoch_ckpt_dir = ensure_dir(out_dir / "epoch_checkpoints" / ("epoch_%03d" % epoch))
        epoch_ckpt_path = epoch_ckpt_dir / ("model_epoch_%03d_score_%.6f.pt" % (epoch, score))
        save_checkpoint(epoch_ckpt_path, model, optimizer, scaler, epoch, score, val_metrics, args, artifact_to_idx)
        with open(epoch_ckpt_dir / "metrics.json", "w") as f:
            json.dump({"epoch": int(epoch), "score": float(score), "metrics": val_metrics}, f, indent=2, sort_keys=True)
        try:
            tmp_epoch_pred = pd.read_csv(val_pred_csv)
            tmp_epoch_pred.to_csv(epoch_ckpt_dir / ("val_predictions_epoch_%03d.csv" % epoch), index=False)
        except Exception:
            pass
        log("SAVED EPOCH checkpoint epoch=%03d score=%.6f path=%s" % (epoch, score, epoch_ckpt_path))

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)

            best_path = out_dir / "best_model_fixed0p5.pt"

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                score,
                val_metrics,
                args,
                artifact_to_idx,
            )

            best_dir = ensure_dir(out_dir / "best_checkpoints" / ("epoch_%03d" % epoch))
            archive_path = best_dir / ("best_model_epoch_%03d_score_%.6f.pt" % (epoch, score))

            save_checkpoint(
                archive_path,
                model,
                optimizer,
                scaler,
                epoch,
                score,
                val_metrics,
                args,
                artifact_to_idx,
            )

            try:
                pd.read_csv(val_pred_csv).to_csv(out_dir / "best_val_predictions.csv", index=False)
                pd.read_csv(val_pred_csv).to_csv(best_dir / ("val_predictions_epoch_%03d.csv" % epoch), index=False)
            except Exception:
                pass

            with open(out_dir / "best_selection.json", "w") as f:
                json.dump(
                    {
                        "epoch": int(epoch),
                        "score": float(score),
                        "metrics": val_metrics,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )

            log("NEW BEST epoch=%03d score=%.6f saved=%s" % (epoch, score, best_path))

    log("Training finished. Best epoch=%d best_score=%.6f" % (best_epoch, best_score))

    if test_loader is not None:
        best_path = out_dir / "best_model_fixed0p5.pt"
        ckpt = torch.load(str(best_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_pred_csv = out_dir / "test_predictions_best_model.csv"

        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            threshold=args.fixed_threshold,
            num_artifacts=num_artifacts,
            out_csv=test_pred_csv,
            idx_to_artifact=idx_to_artifact,
        )

        with open(out_dir / "test_metrics_best_model.json", "w") as f:
            json.dump(test_metrics, f, indent=2, sort_keys=True)

        log("Test metrics: %s" % json.dumps(test_metrics, indent=2, sort_keys=True))
        log("Test predictions: %s" % test_pred_csv)


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train boxed 276x276x276 T1 artifact detector with image+mask two-channel input."
    )

    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--required-shape", default="276x276x276")

    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--mask-mode", choices=["qc", "none"], default="qc")
    p.add_argument("--brain-norm", action="store_true")
    p.add_argument("--no-brain-norm", dest="brain_norm", action="store_false")
    p.set_defaults(brain_norm=True)

    p.add_argument("--domain-aug", action="store_true")
    p.add_argument("--no-domain-aug", dest="domain_aug", action="store_false")
    p.set_defaults(domain_aug=True)

    p.add_argument("--source-balanced", action="store_true")
    p.add_argument("--no-source-balanced", dest="source_balanced", action="store_false")
    p.set_defaults(source_balanced=True)

    p.add_argument("--good-per-source", type=int, default=1)
    p.add_argument("--bad-per-source", type=int, default=1)

    p.add_argument("--fixed-threshold", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument(
        "--pool-mode",
        choices=["global_avg", "hybrid_global_masked_avgmax"],
        default="hybrid_global_masked_avgmax",
    )

    p.add_argument("--binary-loss-weight", type=float, default=1.0)
    p.add_argument("--artifact-loss-weight", type=float, default=0.25)
    p.add_argument("--binary-only-epochs", type=int, default=10)

    p.add_argument("--good-loss-weight", type=float, default=1.5)
    p.add_argument("--bad-loss-weight", type=float, default=1.0)

    p.add_argument("--batch-prior-weight", type=float, default=0.02)
    p.add_argument("--good-margin-weight", type=float, default=0.05)
    p.add_argument("--good-margin", type=float, default=0.35)

    p.add_argument("--min-good-recall-for-best", type=float, default=0.70)
    p.add_argument("--min-bad-recall-for-best", type=float, default=0.50)
    p.add_argument("--best-imbalance-penalty", type=float, default=0.50)
    p.add_argument("--best-artifact-f1-weight", type=float, default=0.05)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every-epoch", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
