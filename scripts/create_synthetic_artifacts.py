#!/usr/bin/env python3
"""
Generate synthetic artifact-corrupted T1 MRI images from boxed GOOD T1 NIfTI files.

Designed for your current boxed output structure:

Input root:
    /path/to/Boxed-Input-Good-276/

Expected input files:
    subject_folder/*_T1_brain_box_276.nii.gz

Optional matching masks:
    subject_folder/*_brain_mask_box_276.nii.gz

Expected shape:
    276 x 276 x 276

Output structure:
    out_root/
        images/good/
        images/bad/motion_mild/
        images/bad/motion_moderate/
        images/bad/ghosting_mild/
        images/bad/ghosting_moderate/
        images/bad/gibbs_mild/
        images/bad/gibbs_moderate/
        qc_png/good/
        qc_png/bad/...
        manifest_t1qc_boxed276_artifacts.csv
        artifact_generation_failures.csv, if failures occur

Important:
    This script robust-normalizes each boxed T1 to 0-1 before artifact generation.
    This avoids clipping native T1 intensities incorrectly.
"""

from pathlib import Path
import argparse
import csv
import json
import random
import traceback

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torchio as tio
    HAS_TORCHIO = True
except Exception:
    torch = None
    tio = None
    HAS_TORCHIO = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except Exception:
    ndimage = None
    HAS_SCIPY = False


EXPECTED_SHAPE = (276, 276, 276)

DEFAULT_VARIANTS = [
    "motion_mild",
    "motion_moderate",
    "ghosting_mild",
    "ghosting_moderate",
    "gibbs_mild",
    "gibbs_moderate",
]

ALL_VARIANTS = DEFAULT_VARIANTS + [
    "coverage_cutoff_moderate",
]


def make_path(p):
    """
    Avoid resolve() by default because GPFS/CUBIC paths may involve symlinks.
    """
    return Path(p).expanduser()


def strip_nii_suffix(path):
    name = Path(path).name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(name).stem


def clean_source_id(path):
    """
    Convert:
        55147941-..._T1_brain_box_276.nii.gz
    to:
        55147941-...
    """
    base = strip_nii_suffix(path)

    suffixes = [
        "_T1_brain_box_276",
        "_brain_box_276",
        "_T1_brain_box",
        "_brain_box",
    ]

    for s in suffixes:
        if base.endswith(s):
            base = base[: -len(s)]

    return base


def collect_niftis(input_dir, input_glob, recursive, max_files=None):
    input_dir = make_path(input_dir)

    if recursive:
        files = sorted(input_dir.rglob(input_glob))
    else:
        files = sorted(input_dir.glob(input_glob))

    files2 = []
    broken = []

    for p in files:
        if p.is_file():
            files2.append(p)
        elif p.is_symlink() and not p.exists():
            broken.append(p)

    if broken:
        print("WARNING: broken symlinks found:", len(broken))
        print("First broken symlink:", broken[0])

    if max_files is not None:
        files2 = files2[:max_files]

    return files2


def matching_boxed_mask_path(t1_path):
    """
    For:
        subject_T1_brain_box_276.nii.gz
    return:
        subject_brain_mask_box_276.nii.gz
    """
    t1_path = Path(t1_path)
    name = t1_path.name

    if name.endswith("_T1_brain_box_276.nii.gz"):
        mask_name = name.replace(
            "_T1_brain_box_276.nii.gz",
            "_brain_mask_box_276.nii.gz",
        )
        mask_path = t1_path.parent / mask_name

        if mask_path.exists():
            return mask_path

    return None


def safe_form_code(header, key, default=1):
    try:
        return int(np.asarray(header[key]).item())
    except Exception:
        return int(default)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_nifti_3d(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError("Expected 3D image, got shape %s: %s" % (data.shape, path))

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return img, data


def load_mask_3d(mask_path, expected_shape):
    if mask_path is None:
        return None

    try:
        img = nib.load(str(mask_path))
        data = img.get_fdata()
        data = np.squeeze(data)

        if data.ndim != 3:
            return None

        if tuple(data.shape) != tuple(expected_shape):
            print("WARNING: mask shape mismatch:", mask_path, data.shape, expected_shape)
            return None

        return data > 0

    except Exception as e:
        print("WARNING: could not load mask:", mask_path, repr(e))
        return None


def clip01(x):
    x = np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def robust_normalize_t1(data, mask=None, low_pct=0.5, high_pct=99.5):
    """
    Normalize one boxed T1 to 0-1 using foreground/brain voxels.

    This is safer than direct clipping because boxed T1 intensities are not
    guaranteed to already be in 0-1 range.
    """
    x = np.nan_to_num(
        np.asarray(data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if mask is not None and np.any(mask):
        fg = mask
    else:
        fg = x > 0

    if np.count_nonzero(fg) < 100:
        fg = np.isfinite(x)

    vals = x[fg]

    if vals.size == 0:
        return clip01(x), {
            "normalization": "fallback_clip01_no_foreground",
            "low_pct": low_pct,
            "high_pct": high_pct,
            "lo": None,
            "hi": None,
        }

    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))

    if hi <= lo:
        return clip01(x), {
            "normalization": "fallback_clip01_bad_percentiles",
            "low_pct": low_pct,
            "high_pct": high_pct,
            "lo": lo,
            "hi": hi,
        }

    y = (x - lo) / (hi - lo + 1e-8)
    y = clip01(y)

    return y, {
        "normalization": "robust_foreground_percentile",
        "low_pct": low_pct,
        "high_pct": high_pct,
        "lo": lo,
        "hi": hi,
        "used_mask": bool(mask is not None and np.any(mask)),
    }


def robust_rescale01_like_input(x, ref):
    """
    Rescale transformed image back to 0-1 using foreground of reference.
    """
    x = np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.nanmax(x) <= 1.05 and np.nanmin(x) >= -0.05:
        return clip01(x)

    fg = ref > 0.01

    if np.any(fg):
        lo = np.percentile(x[fg], 0.5)
        hi = np.percentile(x[fg], 99.5)
    else:
        lo = np.percentile(x, 0.5)
        hi = np.percentile(x, 99.5)

    if hi <= lo:
        return clip01(x)

    x = (x - lo) / (hi - lo + 1e-8)
    return clip01(x)


def save_nifti_like(data, ref_img, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_data = data.astype(np.float32)

    out_header = ref_img.header.copy()
    out = nib.Nifti1Image(out_data, ref_img.affine, out_header)
    out.set_data_dtype(np.float32)

    qform_code = safe_form_code(ref_img.header, "qform_code", default=1)
    sform_code = safe_form_code(ref_img.header, "sform_code", default=1)

    out.set_qform(ref_img.affine, code=qform_code)
    out.set_sform(ref_img.affine, code=sform_code)

    out.header["cal_min"] = 0
    out.header["cal_max"] = 1
    out.header["scl_slope"] = np.nan
    out.header["scl_inter"] = np.nan

    nib.save(out, str(out_path))


def save_qc_png(volume, out_png, title):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    volume = clip01(volume)

    x = volume.shape[0] // 2
    y = volume.shape[1] // 2
    z = volume.shape[2] // 2

    fig = plt.figure(figsize=(18, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(
        np.rot90(volume[x, :, :]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax1.set_title("Sagittal x=%d" % x)
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(
        np.rot90(volume[:, y, :]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax2.set_title("Coronal y=%d" % y)
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(
        np.rot90(volume[:, :, z]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax3.set_title("Axial z=%d" % z)
    ax3.axis("off")

    z_indices = np.linspace(
        int(volume.shape[2] * 0.20),
        int(volume.shape[2] * 0.80),
        6,
    ).astype(int)

    for i, zi in enumerate(z_indices):
        ax = plt.subplot(2, 6, 7 + i)
        ax.imshow(
            np.rot90(volume[:, :, zi]),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        ax.set_title("z=%d" % zi)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(out_png), dpi=180, bbox_inches="tight")
    plt.close(fig)


def apply_torchio_transform(data, affine, transform):
    tensor = data[np.newaxis, ...].astype(np.float32)
    subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor, affine=affine))
    transformed = transform(subject)
    out = transformed.t1.data.numpy()[0]
    return out.astype(np.float32)


def get_base_artifact_and_severity(variant):
    if variant.endswith("_mild"):
        return variant.replace("_mild", ""), "mild"

    if variant.endswith("_moderate"):
        return variant.replace("_moderate", ""), "moderate"

    if variant.endswith("_severe"):
        return variant.replace("_severe", ""), "severe"

    return variant, "unknown"


def motion_params_for_severity(severity, rng):
    if severity == "mild":
        degrees = float(rng.uniform(4.0, 8.0))
        translation = float(rng.uniform(3.0, 6.0))
        num_transforms = int(rng.integers(2, 5))

    elif severity == "moderate":
        degrees = float(rng.uniform(13.0, 15.0))
        translation = float(rng.uniform(11.0, 13.0))
        num_transforms = int(rng.integers(6, 9))

    else:
        degrees = float(rng.uniform(12.0, 18.0))
        translation = float(rng.uniform(10.0, 16.0))
        num_transforms = int(rng.integers(6, 10))

    return degrees, translation, num_transforms


def ghosting_params_for_severity(severity, rng):
    if severity == "mild":
        num_ghosts = int(rng.integers(3, 6))
        intensity = float(rng.uniform(0.25, 0.45))

    elif severity == "moderate":
        num_ghosts = int(rng.integers(9, 12))
        intensity = float(rng.uniform(0.78, 0.88))

    else:
        num_ghosts = int(rng.integers(7, 13))
        intensity = float(rng.uniform(0.75, 1.00))

    axis = int(rng.choice([0, 1, 2]))

    return axis, num_ghosts, intensity


def gibbs_params_for_severity(severity, rng):
    if severity == "mild":
        keep_fraction = float(rng.uniform(0.75, 0.85))

    elif severity == "moderate":
        keep_fraction = float(rng.uniform(0.56, 0.62))

    else:
        keep_fraction = float(rng.uniform(0.55, 0.65))

    n_axes = int(rng.integers(2, 4))
    axes = tuple(sorted(rng.choice([0, 1, 2], size=n_axes, replace=False).tolist()))

    return keep_fraction, axes


def generate_motion(data, ref_img, seed, severity, force_fallback):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    degrees, translation, num_transforms = motion_params_for_severity(severity, rng)

    params = {
        "method": "torchio.RandomMotion" if HAS_TORCHIO and not force_fallback else "fallback_shift_rotate_blend",
        "severity": severity,
        "degrees": degrees,
        "translation_mm": translation,
        "num_transforms": num_transforms,
        "image_interpolation": "linear",
    }

    if HAS_TORCHIO and not force_fallback:
        transform = tio.RandomMotion(
            degrees=degrees,
            translation=translation,
            num_transforms=num_transforms,
            image_interpolation="linear",
            p=1.0,
        )

        out = apply_torchio_transform(data, ref_img.affine, transform)
        return robust_rescale01_like_input(out, data), params

    out = data.astype(np.float32) * 0.60
    n = max(2, num_transforms)

    for _ in range(n):
        shift_vox = rng.uniform(-translation / 2.0, translation / 2.0, size=3)

        if HAS_SCIPY:
            tmp = ndimage.shift(
                data,
                shift=shift_vox,
                order=1,
                mode="constant",
                cval=0.0,
            )

            angle = rng.uniform(-degrees / 3.0, degrees / 3.0)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])

            tmp = ndimage.rotate(
                tmp,
                angle=angle,
                axes=axes,
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )

        else:
            tmp = np.roll(data, shift=np.round(shift_vox).astype(int), axis=(0, 1, 2))

        out += tmp.astype(np.float32) * (0.40 / n)

    return clip01(out), params


def generate_ghosting(data, ref_img, seed, severity, force_fallback):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    axis, num_ghosts, intensity = ghosting_params_for_severity(severity, rng)

    params = {
        "method": "torchio.RandomGhosting" if HAS_TORCHIO and not force_fallback else "fallback_roll_repeats",
        "severity": severity,
        "axis": axis,
        "num_ghosts": num_ghosts,
        "intensity": intensity,
    }

    if HAS_TORCHIO and not force_fallback:
        transform = tio.RandomGhosting(
            num_ghosts=num_ghosts,
            axes=axis,
            intensity=intensity,
            p=1.0,
        )

        out = apply_torchio_transform(data, ref_img.affine, transform)
        return robust_rescale01_like_input(out, data), params

    out = data.astype(np.float32).copy()
    size = data.shape[axis]
    spacing = max(4, size // (num_ghosts + 1))
    ghost_weight = intensity / max(2, num_ghosts)

    for k in range(1, num_ghosts + 1):
        shift = k * spacing
        ghost = np.roll(data, shift=shift, axis=axis)
        out += ghost.astype(np.float32) * ghost_weight

    out = out / (1.0 + intensity)

    return clip01(out), params


def kspace_truncation_gibbs(data, keep_fraction, axes=(0, 1, 2)):
    x = data.astype(np.float32)

    k = np.fft.fftn(x)
    k = np.fft.fftshift(k)

    mask = np.ones(x.shape, dtype=np.float32)

    for ax in axes:
        n = x.shape[ax]
        keep = int(round(n * keep_fraction))
        keep = max(8, min(n, keep))

        start = (n - keep) // 2
        end = start + keep

        axis_mask = np.zeros(n, dtype=np.float32)
        axis_mask[start:end] = 1.0

        shape = [1, 1, 1]
        shape[ax] = n

        mask *= axis_mask.reshape(shape)

    k_trunc = k * mask
    out = np.fft.ifftn(np.fft.ifftshift(k_trunc))
    out = np.abs(out).astype(np.float32)

    return out


def generate_gibbs(data, ref_img, seed, severity):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    keep_fraction, axes = gibbs_params_for_severity(severity, rng)

    params = {
        "method": "kspace_high_frequency_truncation",
        "severity": severity,
        "keep_fraction": keep_fraction,
        "axes": axes,
    }

    out = kspace_truncation_gibbs(
        data,
        keep_fraction=keep_fraction,
        axes=axes,
    )

    return robust_rescale01_like_input(out, data), params


def foreground_bbox(data, threshold=0.02):
    mask = data > threshold

    if not np.any(mask):
        return [(0, s - 1) for s in data.shape]

    coords = np.where(mask)
    bbox = []

    for ax in range(3):
        bbox.append((int(coords[ax].min()), int(coords[ax].max())))

    return bbox


def generate_coverage_cutoff(data, ref_img, seed, severity):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    out = data.astype(np.float32).copy()
    bbox = foreground_bbox(out, threshold=0.02)

    side = str(rng.choice(["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]))

    axis_map = {
        "x_low": (0, "low"),
        "x_high": (0, "high"),
        "y_low": (1, "low"),
        "y_high": (1, "high"),
        "z_low": (2, "low"),
        "z_high": (2, "high"),
    }

    axis, direction = axis_map[side]

    lo, hi = bbox[axis]
    fg_extent = max(1, hi - lo + 1)

    if severity == "moderate":
        fraction = float(rng.uniform(0.18, 0.28))
        min_voxels = 25

    else:
        fraction = float(rng.uniform(0.25, 0.35))
        min_voxels = 35

    cutoff_voxels = int(round(fg_extent * fraction))
    cutoff_voxels = max(min_voxels, cutoff_voxels)

    if direction == "low":
        end = min(out.shape[axis], lo + cutoff_voxels)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = slice(0, end)
        out[tuple(slicer)] = 0.0
        cutoff_range = [0, end]

    else:
        start = max(0, hi - cutoff_voxels + 1)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = slice(start, out.shape[axis])
        out[tuple(slicer)] = 0.0
        cutoff_range = [start, out.shape[axis]]

    params = {
        "method": "foreground_bbox_spatial_cutoff",
        "severity": severity,
        "side": side,
        "axis": axis,
        "direction": direction,
        "foreground_bbox": bbox,
        "cutoff_fraction_of_foreground_extent": fraction,
        "cutoff_voxels": cutoff_voxels,
        "cutoff_range_voxels": cutoff_range,
    }

    return clip01(out), params


def write_row(
    rows,
    source_path,
    source_id,
    source_mask_path,
    output_path,
    qc_label,
    artifact_type,
    severity,
    artifact_variant,
    is_synthetic,
    seed,
    params,
    img,
):
    rows.append({
        "source_path": str(source_path),
        "source_id": source_id,
        "source_mask_path": "" if source_mask_path is None else str(source_mask_path),
        "image_path": str(output_path),
        "qc_label": qc_label,
        "binary_label": 0 if qc_label == "good" else 1,
        "artifact_type": artifact_type,
        "severity": severity,
        "artifact_variant": artifact_variant,
        "is_synthetic": int(is_synthetic),
        "seed": seed,
        "generation_params_json": json.dumps(params, sort_keys=True),
        "shape": "x".join(map(str, img.shape[:3])),
        "zooms": "x".join(["%.5g" % z for z in img.header.get_zooms()[:3]]),
    })


def process_one(input_path, idx, args, rows):
    ref_img, raw_data = load_nifti_3d(input_path)

    source_id = clean_source_id(input_path)
    source_mask_path = matching_boxed_mask_path(input_path)

    mask = load_mask_3d(
        source_mask_path,
        expected_shape=raw_data.shape,
    )

    if args.strict_shape and tuple(raw_data.shape) != tuple(EXPECTED_SHAPE):
        raise ValueError(
            "Expected shape %s, got %s: %s" %
            (EXPECTED_SHAPE, raw_data.shape, input_path)
        )

    if tuple(raw_data.shape) != tuple(EXPECTED_SHAPE):
        print(
            "WARNING: %s has shape %s, expected %s" %
            (input_path.name, raw_data.shape, EXPECTED_SHAPE)
        )

    data, norm_params = robust_normalize_t1(
        raw_data,
        mask=mask,
        low_pct=args.norm_low_pct,
        high_pct=args.norm_high_pct,
    )

    base_seed = args.seed + idx * 1000

    if args.copy_good:
        good_path = args.out_root / "images" / "good" / ("%s__good.nii.gz" % source_id)
        good_qc = args.out_root / "qc_png" / "good" / ("%s__good_qc.png" % source_id)

        if not (args.skip_existing and good_path.exists()):
            save_nifti_like(data, ref_img, good_path)

            if args.write_qc_png:
                save_qc_png(
                    data,
                    good_qc,
                    "%s\nGOOD boxed reference, robust normalized" % source_id,
                )

        good_params = {
            "method": "copy_boxed_good_276_robust_normalized",
            "normalization_params": norm_params,
        }

        write_row(
            rows=rows,
            source_path=input_path,
            source_id=source_id,
            source_mask_path=source_mask_path,
            output_path=good_path,
            qc_label="good",
            artifact_type="none",
            severity="none",
            artifact_variant="none",
            is_synthetic=False,
            seed=base_seed,
            params=good_params,
            img=ref_img,
        )

    for art_i, variant in enumerate(args.variants, start=1):
        artifact_type, severity = get_base_artifact_and_severity(variant)

        seed = base_seed + art_i

        out_path = (
            args.out_root /
            "images" /
            "bad" /
            variant /
            ("%s__%s.nii.gz" % (source_id, variant))
        )

        qc_path = (
            args.out_root /
            "qc_png" /
            "bad" /
            variant /
            ("%s__%s_qc.png" % (source_id, variant))
        )

        if args.skip_existing and out_path.exists():
            params = {
                "method": "skipped_existing",
                "severity": severity,
                "normalization_params": norm_params,
            }

        else:
            if artifact_type == "motion":
                out, params = generate_motion(
                    data,
                    ref_img,
                    seed,
                    severity,
                    args.force_fallback,
                )

            elif artifact_type == "ghosting":
                out, params = generate_ghosting(
                    data,
                    ref_img,
                    seed,
                    severity,
                    args.force_fallback,
                )

            elif artifact_type == "gibbs":
                out, params = generate_gibbs(
                    data,
                    ref_img,
                    seed,
                    severity,
                )

            elif artifact_type == "coverage_cutoff":
                out, params = generate_coverage_cutoff(
                    data,
                    ref_img,
                    seed,
                    severity,
                )

            else:
                raise RuntimeError(
                    "Unknown artifact type: %s from variant %s" %
                    (artifact_type, variant)
                )

            params["normalization_params"] = norm_params

            out = clip01(out)

            save_nifti_like(out, ref_img, out_path)

            if args.write_qc_png:
                save_qc_png(
                    out,
                    qc_path,
                    "%s\n%s" % (source_id, variant),
                )

        write_row(
            rows=rows,
            source_path=input_path,
            source_id=source_id,
            source_mask_path=source_mask_path,
            output_path=out_path,
            qc_label="bad",
            artifact_type=artifact_type,
            severity=severity,
            artifact_variant=variant,
            is_synthetic=True,
            seed=seed,
            params=params,
            img=ref_img,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic T1 QC artifacts from boxed GOOD 276x276x276 T1 images."
        )
    )

    p.add_argument(
        "--input-dir",
        required=True,
        help="Input root, e.g. /path/to/Boxed-Input-Good-276",
    )

    p.add_argument(
        "--out-root",
        required=True,
        help="Output root for generated good/bad artifact dataset",
    )

    p.add_argument(
        "--input-glob",
        default="*_T1_brain_box_276.nii.gz",
        help="Default: *_T1_brain_box_276.nii.gz",
    )

    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside subject folders",
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional testing limit",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing outputs that already exist",
    )

    p.add_argument(
        "--strict-shape",
        action="store_true",
        help="Fail if input shape is not exactly 276x276x276",
    )

    p.add_argument(
        "--force-fallback",
        action="store_true",
        help="Do not use TorchIO for motion/ghosting even if available",
    )

    p.add_argument(
        "--no-copy-good",
        dest="copy_good",
        action="store_false",
        help="Only generate bad artifacts; do not copy normalized good images",
    )

    p.add_argument(
        "--copy-good",
        dest="copy_good",
        action="store_true",
        help="Copy normalized good images into images/good. This is default.",
    )

    p.set_defaults(copy_good=True)

    p.add_argument(
        "--no-qc-png",
        dest="write_qc_png",
        action="store_false",
        help="Disable QC PNG writing to save time/space",
    )

    p.add_argument(
        "--write-qc-png",
        dest="write_qc_png",
        action="store_true",
        help="Write QC PNGs. This is default.",
    )

    p.set_defaults(write_qc_png=True)

    p.add_argument(
        "--norm-low-pct",
        type=float,
        default=0.5,
        help="Lower percentile for robust normalization. Default: 0.5",
    )

    p.add_argument(
        "--norm-high-pct",
        type=float,
        default=99.5,
        help="Upper percentile for robust normalization. Default: 99.5",
    )

    p.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=ALL_VARIANTS,
        help="Artifact variants to generate. Default excludes coverage_cutoff_moderate.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    args.input_dir = make_path(args.input_dir)
    args.out_root = make_path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("T1 boxed-276 artifact generation")
    print("Input dir    :", args.input_dir)
    print("Output root  :", args.out_root)
    print("Input glob   :", args.input_glob)
    print("Recursive    :", args.recursive)
    print("Expected     :", EXPECTED_SHAPE)
    print("Variants     :", ", ".join(args.variants))
    print("Copy GOOD    :", args.copy_good)
    print("Write QC PNG :", args.write_qc_png)
    print("Seed         :", args.seed)
    print("TorchIO      :", "available" if HAS_TORCHIO else "NOT available; using fallback for motion/ghosting")
    print("SciPy        :", "available" if HAS_SCIPY else "NOT available; limited fallback")
    print("=" * 100)

    if not args.input_dir.exists():
        raise FileNotFoundError("Input directory not found: %s" % args.input_dir)

    files = collect_niftis(
        args.input_dir,
        args.input_glob,
        args.recursive,
        args.max_files,
    )

    if not files:
        raise RuntimeError(
            "No valid files found using glob %s in %s" %
            (args.input_glob, args.input_dir)
        )

    print("Found %d boxed GOOD T1 file(s)." % len(files))

    rows = []
    failures = []

    for idx, path in enumerate(files, start=1):
        print("\n[%d/%d] %s" % (idx, len(files), path))

        try:
            process_one(path, idx, args, rows)

        except Exception as e:
            print("FAILED:", path)
            print(traceback.format_exc())

            failures.append({
                "input_path": str(path),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    manifest_path = args.out_root / "manifest_t1qc_boxed276_artifacts.csv"

    fieldnames = [
        "source_path",
        "source_id",
        "source_mask_path",
        "image_path",
        "qc_label",
        "binary_label",
        "artifact_type",
        "severity",
        "artifact_variant",
        "is_synthetic",
        "seed",
        "generation_params_json",
        "shape",
        "zooms",
    ]

    with open(str(manifest_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failure_path = args.out_root / "artifact_generation_failures.csv"

    if failures:
        with open(str(failure_path), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["input_path", "error", "traceback"],
            )
            writer.writeheader()
            writer.writerows(failures)

    print("\n" + "=" * 100)
    print("DONE")
    print("Inputs processed :", len(files))
    print("Manifest         :", manifest_path)
    print("Rows written     :", len(rows))
    print("Failures         :", len(failures))

    if failures:
        print("Failure CSV      :", failure_path)

    print("Output images    :", args.out_root / "images")
    print("QC PNGs          :", args.out_root / "qc_png")
    print("=" * 100)


if __name__ == "__main__":
    main()
