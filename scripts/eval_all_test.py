# -*- coding: utf-8 -*-
"""
TongueExpert end-to-end evaluation (P11 + P12 + P13)

Pipeline:
  ROI (from outputs/roi/{split}) ->
    P11 Color Regression (ckpt may be backbone/head style) ->
    P13 Texture Regression (ckpt may be net.* style) ->
  Mask (from outputs/pred_masks_original/{split}) ->
    P12 Shape deterministic features ->
  Merge -> outputs/e2e_test/e2e_test_all.csv + outputs/e2e_test/e2e_test_summary.csv

✅ This script is robust to:
- Running from PyCharm / any working directory (auto chdir to scripts/)
- P11 ckpt with keys: backbone.* + head.*
- P13 ckpt with keys: net.* (standard torchvision resnet with net.fc)
- Checkpoints wrapped in {"state_dict": ...} / {"model": ...} / {"net": ...} or direct state_dict
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Make paths stable: always relative to this script
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_ids(split_txt: str) -> List[str]:
    ids = []
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def safe_load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ckpt_state(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "net" in ckpt and isinstance(ckpt["net"], dict):
            return ckpt["net"]
        # fallback: outer dict is already a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise ValueError(f"Unsupported ckpt dict keys: {list(ckpt.keys())[:30]}")
    raise ValueError("Unsupported ckpt format (expected dict/state_dict).")


def strip_prefix_module(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned


def infer_out_dim_from_state(state: Dict[str, torch.Tensor]) -> int:
    """
    Try to infer output dim from common last-layer weights.
    Works for:
      - head.weight (P11 style)
      - head.2.weight / head.0.weight
      - net.fc.weight (P13 style)
    """
    cand_keys = [
        "head.weight",
        "head.2.weight",
        "head.0.weight",
        "net.fc.weight",
        "fc.weight",
        "regressor.weight",
        "predictor.weight",
        "out.weight",
    ]
    for k in cand_keys:
        if k in state and isinstance(state[k], torch.Tensor) and state[k].ndim == 2:
            return int(state[k].shape[0])

    # fallback: any 2D weight that looks like a final linear layer
    candidates = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight") and v.ndim == 2:
            out_dim, in_dim = v.shape
            # common in_features of final layers around resnet features
            if in_dim in (512, 256, 1024, 2048):
                candidates.append((k, out_dim, in_dim))
    if candidates:
        # prefer in_dim==512 then smallest out_dim > 1
        candidates.sort(key=lambda x: (0 if x[2] == 512 else 1, x[1]))
        return int(candidates[0][1])

    raise ValueError("Cannot infer output dim from ckpt state_dict. Please pass --p11_dim/--p13_dim.")


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def apply_denorm(pred: np.ndarray, norm: Optional[Dict]) -> np.ndarray:
    """
    norm.json expected:
      {"mean": [...], "std": [...]}
    or
      {"mu": [...], "sigma": [...]}
    """
    if norm is None:
        return pred
    if "mean" in norm and "std" in norm:
        mean = np.array(norm["mean"], dtype=np.float32)
        std = np.array(norm["std"], dtype=np.float32)
    elif "mu" in norm and "sigma" in norm:
        mean = np.array(norm["mu"], dtype=np.float32)
        std = np.array(norm["sigma"], dtype=np.float32)
    else:
        raise ValueError(f"Unknown norm keys: {list(norm.keys())}")
    return pred * std + mean


# -----------------------------
# Models
# -----------------------------
class ResnetSeqBackboneRegressor(nn.Module):
    """
    Matches checkpoint keys like:
      backbone.0.weight, backbone.1.*, backbone.4.*, backbone.5.*, backbone.6.*, backbone.7.*
      head.weight, head.bias
    """
    def __init__(self, out_dim: int):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


class ResnetNetFCRegressor(nn.Module):
    """
    Matches checkpoint keys like:
      net.conv1.*, net.layer1.*, ..., net.fc.weight, net.fc.bias
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


def build_regressor_from_ckpt(ckpt_path: str, device: torch.device, out_dim: Optional[int] = None) -> nn.Module:
    state = strip_prefix_module(load_ckpt_state(ckpt_path, device=device))
    if out_dim is None:
        out_dim = infer_out_dim_from_state(state)

    has_backbone = any(k.startswith("backbone.") for k in state.keys())
    has_net = any(k.startswith("net.") for k in state.keys())

    # 1) direct detect
    if has_backbone:
        model = ResnetSeqBackboneRegressor(out_dim=out_dim).to(device).eval()
        model.load_state_dict(state, strict=True)
        return model

    if has_net:
        model = ResnetNetFCRegressor(out_dim=out_dim).to(device).eval()
        model.load_state_dict(state, strict=True)
        return model

    # 2) fallback: try both
    last_err = None
    for model in [ResnetSeqBackboneRegressor(out_dim=out_dim), ResnetNetFCRegressor(out_dim=out_dim)]:
        model = model.to(device).eval()
        try:
            model.load_state_dict(state, strict=True)
            return model
        except RuntimeError as e:
            last_err = e

    raise RuntimeError(f"Checkpoint structure mismatch for {ckpt_path}.\nLast error:\n{last_err}")


# -----------------------------
# Data IO
# -----------------------------
def default_img_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_roi_image(roi_path: str) -> torch.Tensor:
    img = Image.open(roi_path).convert("RGB")
    x = default_img_transform()(img)
    return x


def load_mask_binary(mask_path: str) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    arr = np.array(m, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


# -----------------------------
# P12 deterministic features from mask
# -----------------------------
def p12_from_mask(mask01: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return {
            "p12_num_tg": 0.0,
            "p12_tg_width": 0.0,
            "p12_tg_height": 0.0,
            "p12_tg_w_div_h": 0.0,
        }
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = float(x_max - x_min + 1)
    height = float(y_max - y_min + 1)
    area = float(mask01.sum())
    ratio10 = 0.0 if height < 1e-6 else 10.0 * (width / height)
    return {
        "p12_num_tg": area,
        "p12_tg_width": width,
        "p12_tg_height": height,
        "p12_tg_w_div_h": ratio10,
    }


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def infer_regression(
    ckpt_path: str,
    norm_path: str,
    roi_dir: str,
    ids: List[str],
    device: torch.device,
    out_dim_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print(f"Using CKPT: {ckpt_path}")
    print(f"Using NORM: {norm_path}")

    state = strip_prefix_module(load_ckpt_state(ckpt_path, device=device))
    out_dim = out_dim_override if out_dim_override is not None else infer_out_dim_from_state(state)
    model = build_regressor_from_ckpt(ckpt_path, device=device, out_dim=out_dim)

    norm = safe_load_json(norm_path) if (norm_path and os.path.exists(norm_path)) else None

    xs = []
    used = []
    for sid in ids:
        roi_path = os.path.join(roi_dir, f"{sid}.jpg")
        if not os.path.exists(roi_path):
            roi_path = os.path.join(roi_dir, f"{sid}.png")
        if not os.path.exists(roi_path):
            continue
        xs.append(load_roi_image(roi_path))
        used.append(sid)

    if len(xs) == 0:
        raise FileNotFoundError(f"No ROI images found in {roi_dir}")

    batch = torch.stack(xs, dim=0).to(device)
    pred = model(batch).detach().cpu().numpy().astype(np.float32)

    pred_denorm = apply_denorm(pred, norm) if norm is not None else pred
    feats = pred_denorm.copy()
    return feats, pred_denorm, used


def infer_p12(masks_dir: str, ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    used = []
    for sid in ids:
        mp = os.path.join(masks_dir, f"{sid}.png")
        if not os.path.exists(mp):
            mp = os.path.join(masks_dir, f"{sid}.jpg")
        if not os.path.exists(mp):
            continue
        mask01 = load_mask_binary(mp)
        feat = p12_from_mask(mask01)
        feat["id"] = sid
        rows.append(feat)
        used.append(sid)
    return pd.DataFrame(rows), used


# -----------------------------
# Metrics summary
# -----------------------------
def make_reg_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
    dim_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    assert y_true.shape == y_pred.shape
    n, d = y_true.shape
    rows = []
    for j in range(d):
        name = dim_names[j] if (dim_names and j < len(dim_names)) else f"{prefix}_{j:02d}"
        t = y_true[:, j]
        p = y_pred[:, j]
        rows.append({
            "feature": name,
            "pearsonr": pearsonr_np(t, p),
            "mae": mae_np(t, p),
            "rmse": rmse_np(t, p),
            "r2": r2_np(t, p),
            "n": n,
        })
    return pd.DataFrame(rows)


def load_gt_csv(path: str, id_col: str = "id") -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # 统一列名去掉空格
    df.columns = [c.strip() for c in df.columns]

    if id_col not in df.columns:
        # 常见 ID 列候选（你现在就是 SID）
        candidates = ["SID", "sid", "Id", "ID", "image_id", "img_id", "name", "filename"]
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break

        if found is None:
            raise ValueError(f"GT csv {path} has no id column. Columns: {df.columns.tolist()}")

        df = df.rename(columns={found: id_col})

    # 确保 id 是 string，避免 00123 被读成 123
    df[id_col] = df[id_col].astype(str).str.strip()


    return df



# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # run controls
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cpu")

    # ckpt & norm
    parser.add_argument("--p11_ckpt", type=str, default="../checkpoints/p11/p11_color_best.pth")
    parser.add_argument("--p11_norm", type=str, default="../checkpoints/p11/p11_color_norm.json")
    parser.add_argument("--p13_ckpt", type=str, default="../checkpoints/p13/p13_texture_best.pth")
    parser.add_argument("--p13_norm", type=str, default="../checkpoints/p13/p13_texture_norm.json")

    # data dirs
    parser.add_argument("--roi_dir", type=str, default="../outputs/roi")
    parser.add_argument("--mask_dir", type=str, default="../outputs/pred_masks_original")
    parser.add_argument("--splits_dir", type=str, default="../data/splits")
    parser.add_argument("--labels_dir", type=str, default="../data/labels")

    # dims (P13 fixed to 16 by default; P11 can be inferred from ckpt)
    parser.add_argument("--p11_dim", type=int, default=None)
    parser.add_argument("--p13_dim", type=int, default=16)

    args = parser.parse_args()
    print(">>> ARGS:", args)

    device = torch.device(args.device)

    split_txt = os.path.join(args.splits_dir, f"{args.split}.txt")
    ids = read_ids(split_txt)
    print(f"Using device: {args.device}")
    print(f"Num {args.split} ids: {len(ids)}")

    roi_split_dir = os.path.join(args.roi_dir, args.split)
    mask_split_dir = os.path.join(args.mask_dir, args.split)

    # ---- P11 ----
    p11_feats, p11_pred, used_ids_p11 = infer_regression(
        ckpt_path=args.p11_ckpt,
        norm_path=args.p11_norm,
        roi_dir=roi_split_dir,
        ids=ids,
        device=device,
        out_dim_override=args.p11_dim,
    )
    print(f"P11 done. Found ROI: {len(used_ids_p11)}")

    # ---- P13 ----
    p13_feats, p13_pred, used_ids_p13 = infer_regression(
        ckpt_path=args.p13_ckpt,
        norm_path=args.p13_norm,
        roi_dir=roi_split_dir,
        ids=ids,
        device=device,
        out_dim_override=args.p13_dim,
    )
    print(f"P13 done. Found ROI: {len(used_ids_p13)}")

    # ---- P12 ----
    p12_df, used_ids_p12 = infer_p12(mask_split_dir, ids)
    print(f"P12 done. Found masks: {len(used_ids_p12)}")

    # Align IDs intersection
    used_set = set(used_ids_p11) & set(used_ids_p13) & set(used_ids_p12)
    used = [sid for sid in ids if sid in used_set]
    print(f"Intersection (P11&P12&P13): {len(used)}")
    if len(used) == 0:
        raise RuntimeError("No samples have all ROI+mask available.")

    # Build output table
    out = pd.DataFrame({"id": used})

    # P11 preds
    p11_idx = {sid: i for i, sid in enumerate(used_ids_p11)}
    p11_mat = np.stack([p11_pred[p11_idx[sid]] for sid in used], axis=0)
    for j in range(p11_mat.shape[1]):
        out[f"p11_pred_{j:02d}"] = p11_mat[:, j]

    # P13 preds
    p13_idx = {sid: i for i, sid in enumerate(used_ids_p13)}
    p13_mat = np.stack([p13_pred[p13_idx[sid]] for sid in used], axis=0)
    for j in range(p13_mat.shape[1]):
        out[f"p13_pred_{j:02d}"] = p13_mat[:, j]

    # P12 feats
    p12_df2 = p12_df.set_index("id").loc[used].reset_index()
    out = out.merge(p12_df2, on="id", how="left")

    # ---- Load GT (optional) and compute summary ----
    summary_rows = []

    p11_gt_csv = os.path.join(args.labels_dir, "p11_tg_color.csv")
    p12_gt_csv = os.path.join(args.labels_dir, "p12_tg_shape.csv")
    p13_gt_csv = os.path.join(args.labels_dir, "p13_tg_texture.csv")

    p11_gt = load_gt_csv(p11_gt_csv, id_col="id")
    p12_gt = load_gt_csv(p12_gt_csv, id_col="id")
    p13_gt = load_gt_csv(p13_gt_csv, id_col="id")

    # P11 summary
    if p11_gt is not None:
        gt = p11_gt[p11_gt["id"].isin(used)].set_index("id").loc[used].reset_index()
        y_true = gt.drop(columns=["id"]).to_numpy(dtype=np.float32)
        y_pred = p11_mat.astype(np.float32)
        for j, col in enumerate(gt.drop(columns=["id"]).columns.tolist()):
            out[f"p11_gt_{col}"] = y_true[:, j]
        p11_cols = gt.drop(columns=["id"]).columns.tolist()
        df_sum = make_reg_summary(y_true, y_pred, prefix="p11", dim_names=p11_cols)
        df_sum["block"] = "p11"
        summary_rows.append(df_sum)

    # P13 summary
    if p13_gt is not None:
        gt = p13_gt[p13_gt["id"].isin(used)].set_index("id").loc[used].reset_index()
        y_true = gt.drop(columns=["id"]).to_numpy(dtype=np.float32)
        y_pred = p13_mat.astype(np.float32)
        for j, col in enumerate(gt.drop(columns=["id"]).columns.tolist()):
            out[f"p13_gt_{col}"] = y_true[:, j]
        p13_cols = gt.drop(columns=["id"]).columns.tolist()
        df_sum = make_reg_summary(y_true, y_pred, prefix="p13", dim_names=p13_cols)
        df_sum["block"] = "p13"
        summary_rows.append(df_sum)

    # P12 summary (try best-effort column mapping)
    if p12_gt is not None and len(p12_df2) > 0:
        gt = p12_gt[p12_gt["id"].isin(used)].set_index("id").loc[used].reset_index()

        colmap = {}
        for c in gt.columns:
            if c == "id":
                continue
            lc = c.lower()
            if ("num" in lc and "tg" in lc) or ("area" in lc):
                colmap[c] = "p12_num_tg"
            elif "width" in lc:
                colmap[c] = "p12_tg_width"
            elif "height" in lc:
                colmap[c] = "p12_tg_height"
            elif ("w_div_h" in lc) or ("ratio" in lc) or ("div" in lc):
                colmap[c] = "p12_tg_w_div_h"

        y_true_list, y_pred_list, feat_names = [], [], []
        for gc, pc in colmap.items():
            if pc in out.columns:
                y_true_list.append(gt[gc].to_numpy(dtype=np.float32))
                y_pred_list.append(out[pc].to_numpy(dtype=np.float32))
                feat_names.append(pc)

                out[f"p12_gt_{gc}"] = gt[gc].to_numpy(dtype=np.float32)

        if len(y_true_list) > 0:
            y_true = np.stack(y_true_list, axis=1)
            y_pred = np.stack(y_pred_list, axis=1)
            df_sum = make_reg_summary(y_true, y_pred, prefix="p12", dim_names=feat_names)
            df_sum["block"] = "p12"
            summary_rows.append(df_sum)

    # ---- Save ----
    out_dir = "../outputs/e2e_test"
    ensure_dir(out_dir)
    all_path = os.path.join(out_dir, "e2e_test_all.csv")
    out.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {all_path}")

    if len(summary_rows) > 0:
        summary = pd.concat(summary_rows, axis=0, ignore_index=True)
    else:
        summary = pd.DataFrame(columns=["block", "feature", "pearsonr", "mae", "rmse", "r2", "n"])

    summary_path = os.path.join(out_dir, "e2e_test_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    if len(summary) > 0:
        print("\nTop PearsonR per block:")
        for b in summary["block"].unique():
            sb = summary[summary["block"] == b].sort_values("pearsonr", ascending=False)
            if len(sb) > 0:
                r = sb.iloc[0]
                print(f"  {b}: {r['feature']} pearsonr={r['pearsonr']:.4f} mae={r['mae']:.4f} r2={r['r2']:.4f}")


if __name__ == "__main__":
    main()
