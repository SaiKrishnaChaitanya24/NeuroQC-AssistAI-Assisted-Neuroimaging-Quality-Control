from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_t1qc_boxed276_artifact_detector import (
    T1QCDataset,
    T1QCHybridCNN,
    collate_keep_strings,
    move_batch,
    parse_shape,
)


# -----------------------------
# Optional imports for Grad-CAM
# -----------------------------
try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_shape(args_dict):
    s = args_dict.get("required_shape", None)
    if s is None:
        s = args_dict.get("required_shape_tuple", None)
    if s is None:
        return (276, 276, 276)
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    return parse_shape(str(s))


def find_mask(img):
    img = Path(img)
    name = img.name
    if name.endswith("_T1_brain_box_276.nii.gz"):
        m = img.parent / name.replace(
            "_T1_brain_box_276.nii.gz",
            "_brain_mask_box_276.nii.gz",
        )
        if m.exists():
            return str(m)
    return ""


def make_df(input_dir, true_label):
    files = sorted(Path(input_dir).rglob("*_T1_brain_box_276.nii.gz"))
    if not files:
        raise RuntimeError("No boxed T1 files found")

    rows = []
    for f in files:
        rows.append({
            "_image_path": str(f),
            "_qc_mask_path": find_mask(f),
            "_binary_label": int(true_label),
            "_artifact_type": "none" if int(true_label) == 0 else "other",
            "_artifact_label": 0,
            "_source_id": f.parent.name,
            "_split": "predict",
        })

    return pd.DataFrame(rows)


def clean_nii_stem(path):
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path).stem


def add_qc_category_columns(pred_df, pass_threshold=0.478, fail_threshold=0.500):
    """
    Adds PASS / REVIEW / FAIL columns to prediction dataframe.

    PASS:   prob_bad < pass_threshold
    REVIEW: pass_threshold <= prob_bad < fail_threshold
    FAIL:   prob_bad >= fail_threshold
    """

    pass_threshold = float(pass_threshold)
    fail_threshold = float(fail_threshold)

    if pass_threshold >= fail_threshold:
        raise ValueError("pass_threshold must be smaller than fail_threshold")

    prob = pred_df["prob_bad"].astype(float)

    pred_df["qc_category"] = np.where(
        prob < pass_threshold,
        "PASS",
        np.where(prob < fail_threshold, "REVIEW", "FAIL"),
    )

    pred_df["pass_threshold"] = pass_threshold
    pred_df["fail_threshold"] = fail_threshold
    pred_df["qc_rule"] = (
        f"PASS prob_bad < {pass_threshold}; "
        f"REVIEW {pass_threshold} <= prob_bad < {fail_threshold}; "
        f"FAIL prob_bad >= {fail_threshold}"
    )

    return pred_df


# -----------------------------
# Grad-CAM implementation
# -----------------------------
def get_module_by_name(model, module_name):
    """
    Return a model submodule by exact named_modules() name.
    """

    for name, module in model.named_modules():
        if name == module_name:
            return module

    available = [name for name, _ in model.named_modules()]
    raise RuntimeError(
        f"Could not find layer named '{module_name}'. "
        f"Available layer names include:\n" + "\n".join(available[:200])
    )


def find_last_conv3d_layer(model):
    """
    Finds the final nn.Conv3d layer in the model.
    This is usually a reasonable default Grad-CAM target layer.
    """

    last_name = None
    last_layer = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            last_name = name
            last_layer = module

    if last_layer is None:
        raise RuntimeError("No Conv3d layer found in model. Cannot run Grad-CAM.")

    print(f"[GradCAM] Using target Conv3d layer: {last_name}")
    return last_layer, last_name


class GradCAM3D:
    """
    3D Grad-CAM for a PyTorch Conv3D model.

    This hooks a target Conv3D layer, backpropagates the BAD score,
    and produces a 3D heatmap resized to the input volume shape.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image, mask):
        """
        image: [B, C, D, H, W]
        mask:  [B, C, D, H, W] or whatever your model expects

        Returns:
            cam:    [B, D, H, W]
            output: model output dict
        """

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        output = self.model(image, mask=mask)

        if "binary_logit" not in output:
            raise RuntimeError(
                f"Expected model output to contain 'binary_logit'. "
                f"Found keys: {list(output.keys())}"
            )

        # binary_logit is the BAD score before sigmoid.
        bad_score = output["binary_logit"].view(-1)

        # Backpropagate from BAD score.
        bad_score.sum().backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations: [B, C, d, h, w]
        # gradients:   [B, C, d, h, w]
        activations = self.activations.detach()
        gradients = self.gradients.detach()

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=1)  # [B, d, h, w]
        cam = F.relu(cam)

        # Resize CAM to input volume size.
        cam = cam.unsqueeze(1)  # [B, 1, d, h, w]
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze(1)  # [B, D, H, W]

        # Normalize each CAM to [0, 1].
        norm_cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min = c.min()
            c_max = c.max()
            if (c_max - c_min) > 1e-8:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = torch.zeros_like(c)
            norm_cams.append(c)

        cam = torch.stack(norm_cams, dim=0)

        return cam, output


def save_gradcam_nifti(cam_3d, reference_nifti_path, out_path):
    """
    Save Grad-CAM heatmap as a NIfTI file using reference image affine/header.
    """

    if nib is None:
        raise RuntimeError(
            "nibabel is required to save Grad-CAM NIfTI files. "
            "Install nibabel in the container/environment."
        )

    ref = nib.load(str(reference_nifti_path))
    cam_np = np.asarray(cam_3d, dtype=np.float32)

    out_img = nib.Nifti1Image(cam_np, affine=ref.affine, header=ref.header)
    out_img.set_data_dtype(np.float32)

    nib.save(out_img, str(out_path))


def normalize_for_display(x):
    x = np.asarray(x, dtype=np.float32)

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return x.astype(np.float32)



def save_gradcam_overlay_png(t1_3d, cam_3d, out_png, title=None):
    """
    Save a presentation-friendly 2 x 3 PNG.

    Top row:
        T1 only

    Bottom row:
        T1 + Grad-CAM overlay

    Columns:
        Axial, Coronal, Sagittal

    Slice selection:
        Uses the slice with strongest Grad-CAM activation in each plane.
        If Grad-CAM is empty, falls back to center slices.
    """

    if plt is None:
        raise RuntimeError(
            "matplotlib is required to save Grad-CAM PNG overlays. "
            "Install matplotlib in the container/environment."
        )

    t1 = normalize_for_display(t1_3d)
    cam = np.asarray(cam_3d, dtype=np.float32)

    if cam.ndim != 3:
        raise RuntimeError(f"Expected 3D CAM, got shape {cam.shape}")

    d, h, w = t1.shape

    # Pick slices where Grad-CAM is strongest, not just center slices.
    if np.isfinite(cam).any() and float(np.nanmax(cam)) > 0:
        cam_pos = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

        z = int(np.argmax(cam_pos.sum(axis=(1, 2))))  # axial
        y = int(np.argmax(cam_pos.sum(axis=(0, 2))))  # coronal
        x = int(np.argmax(cam_pos.sum(axis=(0, 1))))  # sagittal

        # Rescale positive values for display.
        positive = cam_pos[cam_pos > 0]
        if positive.size > 0:
            vmax = float(np.percentile(positive, 99))
            if vmax <= 0:
                vmax = float(positive.max())
            if vmax > 0:
                cam = np.clip(cam_pos / vmax, 0, 1)
            else:
                cam = cam_pos
    else:
        z = d // 2
        y = h // 2
        x = w // 2

    views = [
        (f"Axial z={z}", t1[z, :, :], cam[z, :, :]),
        (f"Coronal y={y}", t1[:, y, :], cam[:, y, :]),
        (f"Sagittal x={x}", t1[:, :, x], cam[:, :, x]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Top row: T1 only
    for j, (view_name, img_slice, cam_slice) in enumerate(views):
        ax = axes[0, j]
        ax.imshow(np.rot90(img_slice), cmap="gray")
        ax.set_title(view_name, fontsize=13)
        ax.axis("off")

    # Bottom row: T1 + Grad-CAM
    for j, (view_name, img_slice, cam_slice) in enumerate(views):
        ax = axes[1, j]
        ax.imshow(np.rot90(img_slice), cmap="gray")
        ax.imshow(np.rot90(cam_slice), cmap="hot", alpha=0.70, vmin=0, vmax=1)
        ax.axis("off")

    # Row labels
    fig.text(0.015, 0.68, "T1 only", rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.015, 0.28, "T1 + Grad-CAM", rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    plt.tight_layout(rect=[0.03, 0.02, 1.0, 0.94])
    plt.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_gradcam_outputs(
    cam_batch,
    image_batch,
    mask_batch,
    image_paths,
    gradcam_dir,
    prob_bad_values,
    pred_labels,
):
    """
    Save Grad-CAM NIfTI and PNG overlay for every sample in the batch.
    """

    gradcam_dir = Path(gradcam_dir)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i in range(cam_batch.shape[0]):
        image_path = image_paths[i]
        stem = clean_nii_stem(image_path)

        cam_np = cam_batch[i].detach().cpu().numpy().astype(np.float32)

        # Brain-mask constrain Grad-CAM so background/crop corners are not visualized.
        mask_np = mask_batch[i]
        if mask_np.ndim == 4:
            mask_np = mask_np[0]
        mask_np = mask_np.detach().cpu().numpy().astype(np.float32)

        mask_bin = mask_np > 0.5
        cam_np = cam_np * mask_bin

        # Re-normalize only inside the brain mask.
        if mask_bin.any():
            inside = cam_np[mask_bin]
            cmin = inside.min()
            cmax = inside.max()
            if cmax > cmin:
                cam_np[mask_bin] = (cam_np[mask_bin] - cmin) / (cmax - cmin)
                cam_np[~mask_bin] = 0.0
            else:
                cam_np[:] = 0.0
        else:
            cam_np[:] = 0.0

        # Use channel 0 of the model input for display.
        # Expected image_batch shape: [B, C, D, H, W].
        t1_np = image_batch[i, 0].detach().cpu().numpy().astype(np.float32)

        out_nii = gradcam_dir / f"{stem}_gradcam_bad.nii.gz"
        out_png = gradcam_dir / f"{stem}_gradcam_bad_overlay.png"

        save_gradcam_nifti(
            cam_3d=cam_np,
            reference_nifti_path=image_path,
            out_path=out_nii,
        )

        title = (
            f"{stem} | prob_bad={float(prob_bad_values[i]):.4f} | "
            f"pred={pred_labels[i]}"
        )

        save_gradcam_overlay_png(
            t1_3d=t1_np,
            cam_3d=cam_np,
            out_png=out_png,
            title=title,
        )

        saved_paths.append({
            "gradcam_nifti": str(out_nii),
            "gradcam_png": str(out_png),
        })

    return saved_paths


def run_pred(
    model,
    loader,
    device,
    threshold,
    idx_to_art,
    save_gradcam=False,
    gradcam_dir=None,
    gradcam_target_layer=None,
    gradcam_max_cases=0,
):
    model.eval()
    rows = []

    gradcam = None
    gradcam_target_layer_name = ""

    if save_gradcam:
        if gradcam_dir is None:
            raise ValueError("--gradcam-dir is required when using --save-gradcam")

        if nib is None:
            raise RuntimeError(
                "nibabel is required for --save-gradcam but is not available."
            )
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for --save-gradcam but is not available."
            )

        Path(gradcam_dir).mkdir(parents=True, exist_ok=True)

        if gradcam_target_layer:
            target_layer = get_module_by_name(model, gradcam_target_layer)
            gradcam_target_layer_name = gradcam_target_layer
            print(f"[GradCAM] Using user-selected layer: {gradcam_target_layer_name}")
        else:
            target_layer, gradcam_target_layer_name = find_last_conv3d_layer(model)

        gradcam = GradCAM3D(model, target_layer)

    saved_gradcam_count = 0

    for batch in loader:
        batch = move_batch(batch, device)

        image = batch["image"]
        mask = batch["mask"]

        if save_gradcam:
            # Grad-CAM needs gradients, so do NOT use torch.no_grad().
            image = image.requires_grad_(True)
            cam_batch, out = gradcam(image, mask)

        else:
            with torch.no_grad():
                out = model(image, mask=mask)
            cam_batch = None

        prob_tensor = torch.sigmoid(out["binary_logit"]).view(-1)
        pred_tensor = (prob_tensor >= float(threshold)).long().view(-1)

        art_prob = torch.softmax(out["artifact_logits"], dim=1)
        art_idx_tensor = torch.argmax(art_prob, dim=1)
        art_conf_tensor = torch.max(art_prob, dim=1).values

        prob = prob_tensor.detach().cpu().numpy()
        pred = pred_tensor.detach().cpu().numpy()
        art_idx = art_idx_tensor.detach().cpu().numpy()
        art_conf = art_conf_tensor.detach().cpu().numpy()

        pred_labels = ["bad" if int(x) == 1 else "good" for x in pred]

        batch_gradcam_paths = None

        if save_gradcam:
            # Optional limit, useful for quick tests.
            if int(gradcam_max_cases) > 0:
                remaining = int(gradcam_max_cases) - int(saved_gradcam_count)
                if remaining <= 0:
                    batch_gradcam_paths = [
                        {"gradcam_nifti": "", "gradcam_png": ""}
                        for _ in range(len(prob))
                    ]
                else:
                    keep_n = min(remaining, len(prob))

                    batch_gradcam_paths = [
                        {"gradcam_nifti": "", "gradcam_png": ""}
                        for _ in range(len(prob))
                    ]

                    saved = save_gradcam_outputs(
                        cam_batch=cam_batch[:keep_n],
                        image_batch=image[:keep_n],
                        mask_batch=mask[:keep_n],
                        image_paths=batch["image_path"][:keep_n],
                        gradcam_dir=gradcam_dir,
                        prob_bad_values=prob[:keep_n],
                        pred_labels=pred_labels[:keep_n],
                    )

                    for j, item in enumerate(saved):
                        batch_gradcam_paths[j] = item

                    saved_gradcam_count += keep_n
            else:
                batch_gradcam_paths = save_gradcam_outputs(
                    cam_batch=cam_batch,
                    image_batch=image,
                    mask_batch=mask,
                    image_paths=batch["image_path"],
                    gradcam_dir=gradcam_dir,
                    prob_bad_values=prob,
                    pred_labels=pred_labels,
                )
                saved_gradcam_count += len(prob)

        for i in range(len(prob)):
            ai = int(art_idx[i])

            row = {
                "image_path": batch["image_path"][i],
                "source_id": batch["source_id"][i],
                "prob_bad": float(prob[i]),
                "pred_binary_label": int(pred[i]),
                "pred_qc_label": "bad" if int(pred[i]) == 1 else "good",
                "pred_artifact_label": ai,
                "pred_artifact_type": idx_to_art.get(ai, str(ai)),
                "pred_artifact_conf": float(art_conf[i]),
            }

            if save_gradcam:
                row["gradcam_target_layer"] = gradcam_target_layer_name
                row["gradcam_nifti"] = batch_gradcam_paths[i]["gradcam_nifti"]
                row["gradcam_png"] = batch_gradcam_paths[i]["gradcam_png"]

            rows.append(row)

    if gradcam is not None:
        gradcam.remove_hooks()

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out-csv", required=True)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--true-label", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")

    # PASS / REVIEW / FAIL thresholds
    p.add_argument("--pass-threshold", type=float, default=0.478)
    p.add_argument("--fail-threshold", type=float, default=0.500)

    # Grad-CAM options
    p.add_argument(
        "--save-gradcam",
        action="store_true",
        help="Save 3D Grad-CAM NIfTI and PNG overlay for each scan.",
    )
    p.add_argument(
        "--gradcam-dir",
        default=None,
        help="Directory where Grad-CAM outputs will be saved.",
    )
    p.add_argument(
        "--gradcam-target-layer",
        default=None,
        help=(
            "Optional exact model layer name for Grad-CAM. "
            "If omitted, the last Conv3d layer is used."
        ),
    )
    p.add_argument(
        "--gradcam-max-cases",
        type=int,
        default=0,
        help=(
            "Maximum number of cases to save Grad-CAM for. "
            "Use 0 to save all cases."
        ),
    )

    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    if args.save_gradcam and int(args.batch_size) != 1:
        print(
            "[WARNING] Grad-CAM is memory-heavy. "
            "Recommended batch size is 1."
        )

    ckpt = load_ckpt(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})

    idx_to_art = ckpt.get("idx_to_artifact", {})
    idx_to_art = {int(k): str(v) for k, v in idx_to_art.items()}

    artifact_to_idx = ckpt.get("artifact_to_idx", {})
    num_artifacts = len(artifact_to_idx)
    if num_artifacts == 0:
        num_artifacts = len(idx_to_art)
    if num_artifacts == 0:
        raise RuntimeError("No artifact mapping in checkpoint")

    shape = get_shape(ckpt_args)
    base_ch = int(ckpt_args.get("base_channels", 8))
    dropout = float(ckpt_args.get("dropout", 0.25))
    pool_mode = ckpt_args.get("pool_mode", "hybrid_global_masked_avgmax")

    model = T1QCHybridCNN(
        in_channels=2,
        num_artifacts=num_artifacts,
        base_channels=base_ch,
        dropout=dropout,
        pool_mode=pool_mode,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = make_df(args.input_dir, args.true_label)

    print("device", device)
    print("epoch", ckpt.get("epoch", "NA"))
    print("images", len(df))
    print("masks", int((df["_qc_mask_path"].astype(str) != "").sum()))
    print("shape", shape)
    print("save_gradcam", bool(args.save_gradcam))

    ds = T1QCDataset(
        df,
        required_shape=shape,
        mask_mode="qc",
        brain_norm=True,
        domain_aug=False,
        training=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_strings,
        drop_last=False,
    )

    pred_df = run_pred(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
        idx_to_art=idx_to_art,
        save_gradcam=args.save_gradcam,
        gradcam_dir=args.gradcam_dir,
        gradcam_target_layer=args.gradcam_target_layer,
        gradcam_max_cases=args.gradcam_max_cases,
    )

    pred_df["checkpoint"] = str(args.checkpoint)
    pred_df["checkpoint_epoch"] = ckpt.get("epoch", "")
    pred_df["threshold"] = float(args.threshold)
    pred_df["true_binary_label_assumed"] = int(args.true_label)

    pred_df = add_qc_category_columns(
        pred_df,
        pass_threshold=args.pass_threshold,
        fail_threshold=args.fail_threshold,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    n = len(pred_df)
    pred_bad = int((pred_df["pred_binary_label"] == 1).sum())
    pred_good = int((pred_df["pred_binary_label"] == 0).sum())

    summary = {
        "total": int(n),
        "pred_good": pred_good,
        "pred_bad": pred_bad,
        "mean_prob_bad": float(pred_df["prob_bad"].mean()),
        "median_prob_bad": float(pred_df["prob_bad"].median()),
        "min_prob_bad": float(pred_df["prob_bad"].min()),
        "max_prob_bad": float(pred_df["prob_bad"].max()),
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "threshold": float(args.threshold),
        "pass_threshold": float(args.pass_threshold),
        "fail_threshold": float(args.fail_threshold),
        "qc_category_counts": {
            str(k): int(v)
            for k, v in pred_df["qc_category"].value_counts(dropna=False).items()
        },
    }

    if int(args.true_label) == 0:
        summary["false_bad_rate_on_good_folder"] = float(pred_bad / n)
    if int(args.true_label) == 1:
        summary["false_good_rate_on_bad_folder"] = float(pred_good / n)

    if args.save_gradcam:
        summary["gradcam_dir"] = str(args.gradcam_dir)
        summary["gradcam_saved_count"] = int(
            (pred_df.get("gradcam_nifti", "").astype(str) != "").sum()
        )

    with open(out_csv.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("saved", out_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))

    print("\nPredicted QC labels:")
    print(pred_df["pred_qc_label"].value_counts(dropna=False))

    print("\nPASS / REVIEW / FAIL:")
    print(pred_df["qc_category"].value_counts(dropna=False))

    print("\nTop 20 highest prob_bad:")
    print(pred_df.sort_values("prob_bad", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
