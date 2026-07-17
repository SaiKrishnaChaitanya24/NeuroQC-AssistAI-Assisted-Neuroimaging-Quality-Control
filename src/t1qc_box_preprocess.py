#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess
import shutil
import csv
import time
import math

import nibabel as nib
import numpy as np


TARGET_SHAPE_DEFAULT = (276, 276, 276)


def make_path(p):
    """
    Do not use resolve() here because some CUBIC/GPFS paths may be symlinks.
    resolve() can change the displayed path to the real symlink target.
    """
    return Path(p).expanduser()


def require_executable(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            "\nCould not find required command: {}\n"
            "Load ANTs first, for example:\n"
            "  module load ants\n".format(name)
        )
    return path


def run_cmd(cmd, log_path=None):
    """
    Python 3.6-compatible subprocess wrapper.
    Do not use text=True on old Python.
    """

    cmd = [str(x) for x in cmd]

    print("\n[RUN]", " ".join(cmd), flush=True)

    if log_path is not None:
        with open(str(log_path), "a") as f:
            f.write("\n[RUN] " + " ".join(cmd) + "\n")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    print(result.stdout)

    if log_path is not None:
        with open(str(log_path), "a") as f:
            f.write(result.stdout + "\n")

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {}:\n{}".format(
                result.returncode,
                " ".join(cmd),
            )
        )


def strip_nii_suffix(path):
    path = Path(path)
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def collect_niftis(input_dir, recursive=False):
    input_dir = make_path(input_dir)

    if recursive:
        files = list(input_dir.rglob("*.nii.gz")) + list(input_dir.rglob("*.nii"))
    else:
        files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))

    return sorted(set(files))


def parse_shape(shape_text):
    """
    Accepts:
      276
      276,276,276
      276x276x276
    """

    if shape_text is None:
        return TARGET_SHAPE_DEFAULT

    shape_text = str(shape_text).strip()

    if "," not in shape_text and "x" not in shape_text:
        v = int(shape_text)
        return (v, v, v)

    parts = shape_text.replace("x", ",").split(",")
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) != 3:
        raise ValueError(
            "--target-shape must be like 276 or 276,276,276 or 276x276x276"
        )

    return tuple(int(p) for p in parts)


def safe_form_code(header, key, default=1):
    """
    Safely get qform_code or sform_code from nibabel header.

    This avoids errors like:
      unhashable type: 'numpy.ndarray'
    """

    try:
        return int(np.asarray(header[key]).item())
    except Exception:
        return int(default)


def binarize_mask(mask_path, out_path, threshold=0.5):
    """
    Force mask to uint8 0/1 and preserve affine/header safely.
    """

    img = nib.load(str(mask_path))
    data = img.get_fdata()

    mask = (data >= threshold).astype(np.uint8)

    out_header = img.header.copy()
    out_img = nib.Nifti1Image(mask, img.affine, out_header)
    out_img.set_data_dtype(np.uint8)

    qform_code = safe_form_code(img.header, "qform_code", default=1)
    sform_code = safe_form_code(img.header, "sform_code", default=1)

    out_img.set_qform(img.affine, code=qform_code)
    out_img.set_sform(img.affine, code=sform_code)

    out_img.header["scl_slope"] = np.nan
    out_img.header["scl_inter"] = np.nan

    nib.save(out_img, str(out_path))


def get_bbox(mask_data, padding=0):
    """
    Compute bounding box around nonzero mask voxels.

    Returns:
      i0, i1, j0, j1, k0, k1

    i1, j1, k1 are exclusive Python slicing indices.
    """

    coords = np.argwhere(mask_data > 0)

    if coords.shape[0] == 0:
        raise RuntimeError("Mask is empty. Cannot compute bounding box.")

    min_i, min_j, min_k = coords.min(axis=0)
    max_i, max_j, max_k = coords.max(axis=0)

    shape = mask_data.shape

    i0 = max(int(min_i) - padding, 0)
    j0 = max(int(min_j) - padding, 0)
    k0 = max(int(min_k) - padding, 0)

    i1 = min(int(max_i) + padding + 1, shape[0])
    j1 = min(int(max_j) + padding + 1, shape[1])
    k1 = min(int(max_k) + padding + 1, shape[2])

    return i0, i1, j0, j1, k0, k1


def bbox_shape(bbox):
    i0, i1, j0, j1, k0, k1 = bbox
    return i1 - i0, j1 - j0, k1 - k0


def choose_fixed_crop_start_from_bbox(bbox, target_shape):
    """
    Choose crop start so the expanded brain bbox is centered inside
    the fixed 276x276x276 cube.

    The start can be negative. Negative start means the output will
    include zero padding before the original image starts.
    """

    i0, i1, j0, j1, k0, k1 = bbox
    target_i, target_j, target_k = target_shape

    box_i, box_j, box_k = bbox_shape(bbox)

    if box_i > target_i or box_j > target_j or box_k > target_k:
        raise RuntimeError(
            "The expanded brain bbox {} is larger than target shape {}. "
            "Using this target size would cut the brain. Increase target shape, "
            "reduce --bbox-padding, reduce --dilate-mask, or inspect this subject.".format(
                (box_i, box_j, box_k),
                target_shape,
            )
        )

    start_i = int(math.floor((i0 + i1 - target_i) / 2.0))
    start_j = int(math.floor((j0 + j1 - target_j) / 2.0))
    start_k = int(math.floor((k0 + k1 - target_k) / 2.0))

    return start_i, start_j, start_k


def get_output_dtype(data, dtype_mode):
    if dtype_mode == "float32":
        return np.float32

    if dtype_mode == "original":
        return data.dtype

    raise ValueError("Unknown dtype_mode: {}".format(dtype_mode))


def crop_or_pad_to_fixed_shape(
    input_img_path,
    bbox,
    target_shape,
    out_path,
    dtype_mode="float32",
    fill_value=0,
):
    """
    Crop or zero-pad a NIfTI image to fixed shape around bbox.

    This does NOT resample.
    This does NOT interpolate.
    This only copies existing native-grid voxels into a fixed-size array.

    If the fixed cube extends beyond the original image FOV,
    the missing region is filled with zeros.
    """

    img = nib.load(str(input_img_path))
    data = np.asanyarray(img.dataobj)

    if data.ndim != 3:
        raise RuntimeError(
            "Expected 3D image, but got shape {} for {}".format(
                data.shape,
                input_img_path,
            )
        )

    target_shape = tuple(int(x) for x in target_shape)

    start_i, start_j, start_k = choose_fixed_crop_start_from_bbox(
        bbox=bbox,
        target_shape=target_shape,
    )

    target_i, target_j, target_k = target_shape

    end_i = start_i + target_i
    end_j = start_j + target_j
    end_k = start_k + target_k

    src_i0 = max(start_i, 0)
    src_j0 = max(start_j, 0)
    src_k0 = max(start_k, 0)

    src_i1 = min(end_i, data.shape[0])
    src_j1 = min(end_j, data.shape[1])
    src_k1 = min(end_k, data.shape[2])

    dst_i0 = src_i0 - start_i
    dst_j0 = src_j0 - start_j
    dst_k0 = src_k0 - start_k

    dst_i1 = dst_i0 + (src_i1 - src_i0)
    dst_j1 = dst_j0 + (src_j1 - src_j0)
    dst_k1 = dst_k0 + (src_k1 - src_k0)

    out_dtype = get_output_dtype(data, dtype_mode)

    out_data = np.zeros(target_shape, dtype=out_dtype)

    if fill_value != 0:
        out_data[:] = fill_value

    out_data[
        dst_i0:dst_i1,
        dst_j0:dst_j1,
        dst_k0:dst_k1,
    ] = data[
        src_i0:src_i1,
        src_j0:src_j1,
        src_k0:src_k1,
    ].astype(out_dtype)

    old_affine = img.affine.copy()

    # Output voxel (0,0,0) corresponds to input voxel
    # (start_i, start_j, start_k).
    # This keeps the cropped/padded image spatially meaningful.
    new_affine = old_affine.copy()
    new_affine[:3, 3] = old_affine.dot([start_i, start_j, start_k, 1])[:3]

    out_header = img.header.copy()
    out_img = nib.Nifti1Image(out_data, new_affine, out_header)
    out_img.set_data_dtype(out_dtype)

    qform_code = safe_form_code(img.header, "qform_code", default=1)
    sform_code = safe_form_code(img.header, "sform_code", default=1)

    out_img.set_qform(new_affine, code=qform_code)
    out_img.set_sform(new_affine, code=sform_code)

    out_img.header["scl_slope"] = np.nan
    out_img.header["scl_inter"] = np.nan

    nib.save(out_img, str(out_path))

    crop_meta = {
        "input_shape_i": data.shape[0],
        "input_shape_j": data.shape[1],
        "input_shape_k": data.shape[2],
        "target_shape_i": target_shape[0],
        "target_shape_j": target_shape[1],
        "target_shape_k": target_shape[2],
        "crop_start_i": start_i,
        "crop_start_j": start_j,
        "crop_start_k": start_k,
        "crop_end_i": end_i,
        "crop_end_j": end_j,
        "crop_end_k": end_k,
        "src_i0": src_i0,
        "src_i1": src_i1,
        "src_j0": src_j0,
        "src_j1": src_j1,
        "src_k0": src_k0,
        "src_k1": src_k1,
        "dst_i0": dst_i0,
        "dst_i1": dst_i1,
        "dst_j0": dst_j0,
        "dst_j1": dst_j1,
        "dst_k0": dst_k0,
        "dst_k1": dst_k1,
    }

    return crop_meta


def save_crop_info_txt(out_path, info):
    with open(str(out_path), "w") as f:
        for key in sorted(info.keys()):
            f.write("{}: {}\n".format(key, info[key]))


def process_one(
    t1_path,
    mni_template,
    mni_mask,
    out_root,
    target_shape,
    transform_type,
    dilate_mask,
    bbox_padding,
    output_dtype,
    skip_existing,
    skip_existing_mask,
):
    t1_path = make_path(t1_path)
    mni_template = make_path(mni_template)
    mni_mask = make_path(mni_mask)
    out_root = make_path(out_root)

    stem = strip_nii_suffix(t1_path)

    subject_outdir = out_root / stem
    subject_outdir.mkdir(parents=True, exist_ok=True)

    log_path = subject_outdir / "{}_processing.log".format(stem)

    reg_prefix = subject_outdir / "{}_to_MNI_".format(stem)
    generic_affine = subject_outdir / "{}_to_MNI_0GenericAffine.mat".format(stem)

    mask_raw = subject_outdir / "{}_MNI_brain_mask_in_subject_raw.nii.gz".format(stem)
    mask_bin = subject_outdir / "{}_MNI_brain_mask_in_subject_bin.nii.gz".format(stem)
    mask_final = subject_outdir / "{}_MNI_brain_mask_in_subject_final.nii.gz".format(stem)

    boxed_t1 = subject_outdir / "{}_T1_brain_box_276.nii.gz".format(stem)
    boxed_mask = subject_outdir / "{}_brain_mask_box_276.nii.gz".format(stem)
    crop_info_txt = subject_outdir / "{}_brain_box_276_crop_info.txt".format(stem)

    if skip_existing and boxed_t1.exists() and boxed_mask.exists():
        print("[SKIP] Existing boxed output: {}".format(boxed_t1))
        return {
            "input": str(t1_path),
            "stem": stem,
            "status": "skipped_existing",
            "outdir": str(subject_outdir),
            "bbox_i": "",
            "bbox_j": "",
            "bbox_k": "",
            "target_dim1": target_shape[0],
            "target_dim2": target_shape[1],
            "target_dim3": target_shape[2],
            "mask_fullsize": str(mask_final),
            "boxed_t1": str(boxed_t1),
            "boxed_mask": str(boxed_mask),
            "crop_info": str(crop_info_txt),
            "error": "",
        }

    if transform_type == "rigid":
        ants_t = "r"
    elif transform_type == "affine":
        ants_t = "a"
    else:
        raise ValueError("Unknown transform type: {}".format(transform_type))

    with open(str(log_path), "w") as f:
        f.write("Input T1: {}\n".format(t1_path))
        f.write("MNI template: {}\n".format(mni_template))
        f.write("MNI mask: {}\n".format(mni_mask))
        f.write("Transform type: {}\n".format(transform_type))
        f.write("Target shape: {}\n".format(target_shape))
        f.write("Mask dilation: {}\n".format(dilate_mask))
        f.write("BBox padding: {}\n".format(bbox_padding))
        f.write("Output dtype: {}\n".format(output_dtype))

    # ------------------------------------------------------------
    # STEP 1:
    # Register subject T1 to MNI.
    #
    # This estimates subject -> MNI transform.
    # We do NOT use the warped T1 for ML.
    # ------------------------------------------------------------
    if not (skip_existing_mask and mask_final.exists()):
        run_cmd(
            [
                "antsRegistrationSyNQuick.sh",
                "-d", "3",
                "-f", mni_template,
                "-m", t1_path,
                "-o", reg_prefix,
                "-t", ants_t,
            ],
            log_path=log_path,
        )

        if not generic_affine.exists():
            raise RuntimeError(
                "Expected transform was not created: {}".format(generic_affine)
            )

        # ------------------------------------------------------------
        # STEP 2:
        # Bring MNI brain mask back to subject T1 space.
        #
        # Registration transform is subject -> MNI.
        # We need MNI -> subject.
        # Therefore we invert it using:
        # [transform.mat,1]
        #
        # The T1 image itself is NOT resampled here.
        # Only the mask is transformed.
        # ------------------------------------------------------------
        run_cmd(
            [
                "antsApplyTransforms",
                "-d", "3",
                "-i", mni_mask,
                "-r", t1_path,
                "-o", mask_raw,
                "-n", "NearestNeighbor",
                "-t", "[{},1]".format(generic_affine),
            ],
            log_path=log_path,
        )

        # ------------------------------------------------------------
        # STEP 3:
        # Binarize mask.
        # ------------------------------------------------------------
        binarize_mask(mask_raw, mask_bin)

        # ------------------------------------------------------------
        # STEP 4:
        # Optional dilation.
        #
        # This expands the mask before computing the final box.
        # It helps avoid tight brain boxes that cut cortex.
        # ------------------------------------------------------------
        if dilate_mask > 0:
            run_cmd(
                [
                    "ImageMath",
                    "3",
                    mask_final,
                    "MD",
                    mask_bin,
                    str(dilate_mask),
                ],
                log_path=log_path,
            )

            binarize_mask(mask_final, mask_final)

        else:
            shutil.copyfile(str(mask_bin), str(mask_final))

    else:
        print("[REUSE MASK] {}".format(mask_final))

    # ------------------------------------------------------------
    # STEP 5:
    # Compute bounding box around propagated subject-space mask.
    # ------------------------------------------------------------
    mask_img = nib.load(str(mask_final))
    mask_data = mask_img.get_fdata() > 0

    bbox = get_bbox(mask_data, padding=bbox_padding)
    bshape = bbox_shape(bbox)

    # ------------------------------------------------------------
    # STEP 6:
    # Crop/pad ORIGINAL T1 to fixed 276 cube.
    #
    # This is the ML input.
    # No T1 resampling.
    # No T1 interpolation.
    # ------------------------------------------------------------
    crop_meta_t1 = crop_or_pad_to_fixed_shape(
        input_img_path=t1_path,
        bbox=bbox,
        target_shape=target_shape,
        out_path=boxed_t1,
        dtype_mode=output_dtype,
        fill_value=0,
    )

    # Crop/pad mask to the same 276 cube for QC.
    crop_or_pad_to_fixed_shape(
        input_img_path=mask_final,
        bbox=bbox,
        target_shape=target_shape,
        out_path=boxed_mask,
        dtype_mode="original",
        fill_value=0,
    )

    crop_info = {}
    crop_info["input_t1"] = str(t1_path)
    crop_info["mask_fullsize"] = str(mask_final)
    crop_info["boxed_t1"] = str(boxed_t1)
    crop_info["boxed_mask"] = str(boxed_mask)
    crop_info["target_shape"] = target_shape
    crop_info["bbox"] = bbox
    crop_info["bbox_shape"] = bshape
    crop_info["bbox_padding"] = bbox_padding
    crop_info["dilate_mask"] = dilate_mask
    crop_info["transform_type"] = transform_type
    crop_info["output_dtype"] = output_dtype

    for k, v in crop_meta_t1.items():
        crop_info["t1_" + k] = v

    save_crop_info_txt(crop_info_txt, crop_info)

    return {
        "input": str(t1_path),
        "stem": stem,
        "status": "done",
        "outdir": str(subject_outdir),
        "bbox_i": bshape[0],
        "bbox_j": bshape[1],
        "bbox_k": bshape[2],
        "target_dim1": target_shape[0],
        "target_dim2": target_shape[1],
        "target_dim3": target_shape[2],
        "mask_fullsize": str(mask_final),
        "boxed_t1": str(boxed_t1),
        "boxed_mask": str(boxed_mask),
        "crop_info": str(crop_info_txt),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed 276x276x276 T1 brain boxes for ML without T1 resampling. "
            "Uses MNI mask propagation only to locate the brain."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing .nii.gz or .nii files",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root folder",
    )

    parser.add_argument(
        "--mni-template",
        required=True,
        help="MNI template image, e.g. MNI152_T1_1mm.nii.gz",
    )

    parser.add_argument(
        "--mni-mask",
        required=True,
        help="MNI brain mask image, e.g. MNI152_T1_1mm_brain_mask.nii.gz",
    )

    parser.add_argument(
        "--target-shape",
        default="276",
        help="Target fixed shape. Default: 276, meaning 276x276x276",
    )

    parser.add_argument(
        "--transform",
        choices=["rigid", "affine"],
        default="rigid",
        help="Registration type. Default: rigid",
    )

    parser.add_argument(
        "--dilate-mask",
        type=int,
        default=2,
        help="Dilate propagated mask by N voxels before boxing. Default: 2",
    )

    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=5,
        help="Additional voxel padding around mask bbox before fitting into 276 cube. Default: 5",
    )

    parser.add_argument(
        "--output-dtype",
        choices=["float32", "original"],
        default="float32",
        help="Output dtype for boxed T1. Default: float32 for ML",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subject if boxed T1 and boxed mask already exist",
    )

    parser.add_argument(
        "--skip-existing-mask",
        action="store_true",
        help="Reuse existing propagated full-size mask if present",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if one subject fails",
    )

    args = parser.parse_args()

    require_executable("antsRegistrationSyNQuick.sh")
    require_executable("antsApplyTransforms")

    if args.dilate_mask > 0:
        require_executable("ImageMath")

    input_dir = make_path(args.input_dir)
    out_root = make_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    target_shape = parse_shape(args.target_shape)

    files = collect_niftis(input_dir, recursive=args.recursive)

    if len(files) == 0:
        raise RuntimeError("No .nii.gz or .nii files found in {}".format(input_dir))

    print("\nFound {} NIfTI files.".format(len(files)))
    print("Input dir:       {}".format(input_dir))
    print("Output root:     {}".format(out_root))
    print("Target shape:    {}".format(target_shape))
    print("Transform:       {}".format(args.transform))
    print("Dilate mask:     {}".format(args.dilate_mask))
    print("BBox padding:    {}".format(args.bbox_padding))
    print("Output dtype:    {}".format(args.output_dtype))

    summary_csv = out_root / "batch_fixed_276_box_summary.csv"

    results = []

    for idx, t1_path in enumerate(files, start=1):
        print("\n" + "=" * 80)
        print("[{}/{}] Processing {}".format(idx, len(files), t1_path))
        print("=" * 80)

        start_time = time.time()

        try:
            row = process_one(
                t1_path=t1_path,
                mni_template=args.mni_template,
                mni_mask=args.mni_mask,
                out_root=out_root,
                target_shape=target_shape,
                transform_type=args.transform,
                dilate_mask=args.dilate_mask,
                bbox_padding=args.bbox_padding,
                output_dtype=args.output_dtype,
                skip_existing=args.skip_existing,
                skip_existing_mask=args.skip_existing_mask,
            )

        except Exception as e:
            print("\n[FAILED] {}".format(t1_path))
            print(str(e))

            row = {
                "input": str(t1_path),
                "stem": strip_nii_suffix(t1_path),
                "status": "failed",
                "outdir": "",
                "bbox_i": "",
                "bbox_j": "",
                "bbox_k": "",
                "target_dim1": target_shape[0],
                "target_dim2": target_shape[1],
                "target_dim3": target_shape[2],
                "mask_fullsize": "",
                "boxed_t1": "",
                "boxed_mask": "",
                "crop_info": "",
                "error": str(e),
            }

            if not args.continue_on_error:
                row["elapsed_seconds"] = round(time.time() - start_time, 2)
                results.append(row)
                break

        row["elapsed_seconds"] = round(time.time() - start_time, 2)
        results.append(row)

        with open(str(summary_csv), "w", newline="") as f:
            fieldnames = [
                "input",
                "stem",
                "status",
                "elapsed_seconds",
                "outdir",
                "bbox_i",
                "bbox_j",
                "bbox_k",
                "target_dim1",
                "target_dim2",
                "target_dim3",
                "mask_fullsize",
                "boxed_t1",
                "boxed_mask",
                "crop_info",
                "error",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 80)
    print("[DONE]")
    print("Summary CSV: {}".format(summary_csv))
    print("=" * 80)


if __name__ == "__main__":
    main()
