from pathlib import Path
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_t1qc_boxed276_artifact_detector import (
    T1QCDataset,
    T1QCHybridCNN,
    collate_keep_strings,
    move_batch,
    parse_shape,
)


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


@torch.no_grad()
def run_pred(model, loader, device, threshold, idx_to_art):
    model.eval()
    rows = []

    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["image"], mask=batch["mask"])

        prob = torch.sigmoid(out["binary_logit"])
        pred = (prob >= float(threshold)).long()

        art_prob = torch.softmax(out["artifact_logits"], dim=1)
        art_idx = torch.argmax(art_prob, dim=1)
        art_conf = torch.max(art_prob, dim=1).values

        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()
        art_idx = art_idx.cpu().numpy()
        art_conf = art_conf.cpu().numpy()

        for i in range(len(prob)):
            ai = int(art_idx[i])
            rows.append({
                "image_path": batch["image_path"][i],
                "source_id": batch["source_id"][i],
                "prob_bad": float(prob[i]),
                "pred_binary_label": int(pred[i]),
                "pred_qc_label": "bad" if int(pred[i]) == 1 else "good",
                "pred_artifact_label": ai,
                "pred_artifact_type": idx_to_art.get(ai, str(ai)),
                "pred_artifact_conf": float(art_conf[i]),
            })

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
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
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

    pred_df = run_pred(model, loader, device, args.threshold, idx_to_art)

    pred_df["checkpoint"] = str(args.checkpoint)
    pred_df["checkpoint_epoch"] = ckpt.get("epoch", "")
    pred_df["threshold"] = float(args.threshold)
    pred_df["true_binary_label_assumed"] = int(args.true_label)

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
    }

    if int(args.true_label) == 0:
        summary["false_bad_rate_on_good_folder"] = float(pred_bad / n)
    if int(args.true_label) == 1:
        summary["false_good_rate_on_bad_folder"] = float(pred_good / n)

    with open(out_csv.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("saved", out_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(pred_df["pred_qc_label"].value_counts(dropna=False))
    print(pred_df.sort_values("prob_bad", ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
