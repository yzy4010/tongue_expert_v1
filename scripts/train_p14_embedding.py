# -*- coding: utf-8 -*-
"""
scripts/train_p14_embedding.py

P14-A: CNN Embedding with multi-task proxy supervision (P11 color + P13 texture)

✅ Run directly (PyCharm click Run) — no CLI arguments needed.

Inputs:
  - data/splits/{train,val}.txt
  - outputs/roi/{train,val}/*.jpg|*.png
  - data/labels/p11_tg_color.csv   (ID column may be SID/ID/...)
  - data/labels/p13_tg_texture.csv (ID column may be SID/ID/...)

Outputs:
  - checkpoints/p14/p14_multitask_best.pth
  - checkpoints/p14/p14_norm.json   (mean/std from TRAIN split)

Model:
  ROI -> ResNet18 backbone -> 3 heads:
    - P14 embedding head (128-d)
    - P11 regression head (76-d, standardized target)
    - P13 regression head (16-d, standardized target)

Loss:
  L = w_p11 * MSE(p11_pred, p11_z) + w_p13 * MSE(p13_pred, p13_z)
"""

import os
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# -------------------------
# Make paths stable: always relative to this script
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# -------------------------
# Hard-coded run config (edit here)
# -------------------------
class Args:
    # device: "cpu" or "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # change to "cuda" if you have GPU

    # project-relative dirs
    splits_dir = "../data/splits"
    roi_dir = "../outputs/roi"
    labels_dir = "../data/labels"

    out_dir = "../checkpoints/p14"
    name = "p14_multitask_best.pth"

    # training hyperparams
    emb_dim = 128
    img_size = 224
    batch_size = 64
    epochs = 30
    lr = 2e-4
    weight_decay = 1e-4
    dropout = 0.0

    # loss weights
    w_p11 = 1.0
    w_p13 = 1.0

    # misc
    seed = 42
    num_workers = 4  # set 0 if Windows worker issues
    pin_memory = True  # set False if cpu-only


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_ids(txt_path: str) -> List[str]:
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def load_csv_with_id(path: str, id_col: str = "id") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if id_col not in df.columns:
        candidates = ["SID", "sid", "Id", "ID", "image_id", "img_id", "name", "filename"]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise ValueError(f"{path} has no id column. Columns={df.columns.tolist()}")
        df = df.rename(columns={found: id_col})

    df[id_col] = df[id_col].astype(str).str.strip()
    return df


def compute_mean_std(mat: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.maximum(std, eps)
    return mean, std


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Transforms
# -------------------------
def build_train_tf(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # safe light aug for ROI
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_tf(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# -------------------------
# Dataset
# -------------------------
class RoiMultiTaskDataset(Dataset):
    def __init__(
        self,
        ids: List[str],
        roi_dir: str,
        p11_map: Dict[str, np.ndarray],
        p13_map: Dict[str, np.ndarray],
        p11_mean: np.ndarray,
        p11_std: np.ndarray,
        p13_mean: np.ndarray,
        p13_std: np.ndarray,
        tfm,
    ):
        self.ids = []
        self.roi_paths = []
        self.p11 = []
        self.p13 = []
        self.tfm = tfm

        for sid in ids:
            jpg = os.path.join(roi_dir, f"{sid}.jpg")
            png = os.path.join(roi_dir, f"{sid}.png")
            if os.path.exists(jpg):
                roi_path = jpg
            elif os.path.exists(png):
                roi_path = png
            else:
                continue

            if sid not in p11_map or sid not in p13_map:
                continue

            self.ids.append(sid)
            self.roi_paths.append(roi_path)

            y11 = (p11_map[sid] - p11_mean) / p11_std
            y13 = (p13_map[sid] - p13_mean) / p13_std
            self.p11.append(y11.astype(np.float32))
            self.p13.append(y13.astype(np.float32))

        if len(self.ids) == 0:
            self.p11 = np.zeros((0, 1), dtype=np.float32)
            self.p13 = np.zeros((0, 1), dtype=np.float32)
        else:
            self.p11 = np.stack(self.p11, axis=0).astype(np.float32)
            self.p13 = np.stack(self.p13, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        path = self.roi_paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y11 = torch.from_numpy(self.p11[idx])
        y13 = torch.from_numpy(self.p13[idx])
        return sid, x, y11, y13


# -------------------------
# Model
# -------------------------
class P14MultiTaskNet(nn.Module):
    def __init__(self, emb_dim: int, p11_dim: int, p13_dim: int, dropout: float = 0.0):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # backbone ends at global pooled features
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.feat_dim = 512

        def head(out_dim: int):
            layers = [
                nn.Flatten(1),
                nn.Linear(self.feat_dim, 256),
                nn.ReLU(inplace=True),
            ]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(256, out_dim))
            return nn.Sequential(*layers)

        self.head_emb = head(emb_dim)
        self.head_p11 = head(p11_dim)
        self.head_p13 = head(p13_dim)

    def forward(self, x):
        f = self.backbone(x)
        emb = self.head_emb(f)
        p11 = self.head_p11(f)
        p13 = self.head_p13(f)
        return emb, p11, p13


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, device, w_p11: float, w_p13: float):
    model.train()
    mse = nn.MSELoss(reduction="mean")

    total = 0.0
    total_p11 = 0.0
    total_p13 = 0.0
    n = 0

    for _, x, y11, y13 in loader:
        x = x.to(device, non_blocking=True)
        y11 = y11.to(device, non_blocking=True)
        y13 = y13.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, p11, p13 = model(x)

        loss11 = mse(p11, y11)
        loss13 = mse(p13, y13)
        loss = w_p11 * loss11 + w_p13 * loss13

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += float(loss.item()) * bs
        total_p11 += float(loss11.item()) * bs
        total_p13 += float(loss13.item()) * bs
        n += bs

    return {
        "train_loss": total / max(n, 1),
        "train_mse_p11": total_p11 / max(n, 1),
        "train_mse_p13": total_p13 / max(n, 1),
        "n": n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, w_p11: float, w_p13: float):
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    total = 0.0
    total_p11 = 0.0
    total_p13 = 0.0
    n = 0

    for _, x, y11, y13 in loader:
        x = x.to(device, non_blocking=True)
        y11 = y11.to(device, non_blocking=True)
        y13 = y13.to(device, non_blocking=True)

        _, p11, p13 = model(x)
        loss11 = mse(p11, y11)
        loss13 = mse(p13, y13)
        loss = w_p11 * loss11 + w_p13 * loss13

        bs = x.size(0)
        total += float(loss.item()) * bs
        total_p11 += float(loss11.item()) * bs
        total_p13 += float(loss13.item()) * bs
        n += bs

    return {
        "val_loss": total / max(n, 1),
        "val_mse_p11": total_p11 / max(n, 1),
        "val_mse_p13": total_p13 / max(n, 1),
        "n": n,
    }


@dataclass
class P14Config:
    emb_dim: int
    p11_dim: int
    p13_dim: int
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    w_p11: float
    w_p13: float
    seed: int


def main():
    args = Args()

    # device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # seed
    set_seed(args.seed)

    # splits
    train_ids = read_ids(os.path.join(args.splits_dir, "train.txt"))
    val_ids = read_ids(os.path.join(args.splits_dir, "val.txt"))
    print(f"Train ids: {len(train_ids)} | Val ids: {len(val_ids)}")

    # labels
    p11_csv = os.path.join(args.labels_dir, "p11_tg_color.csv")
    p13_csv = os.path.join(args.labels_dir, "p13_tg_texture.csv")

    p11_df = load_csv_with_id(p11_csv, id_col="id")
    p13_df = load_csv_with_id(p13_csv, id_col="id")

    p11_cols = [c for c in p11_df.columns if c != "id"]
    p13_cols = [c for c in p13_df.columns if c != "id"]

    p11_dim = len(p11_cols)
    p13_dim = len(p13_cols)

    if p11_dim != 76:
        print(f"[WARN] P11 dim={p11_dim} (expected 76). Continue.")
    if p13_dim != 16:
        print(f"[WARN] P13 dim={p13_dim} (expected 16). Continue.")

    # maps: id -> vector
    p11_map = {row["id"]: row[p11_cols].to_numpy(dtype=np.float32) for _, row in p11_df.iterrows()}
    p13_map = {row["id"]: row[p13_cols].to_numpy(dtype=np.float32) for _, row in p13_df.iterrows()}

    # compute mean/std from TRAIN only
    train_p11 = []
    train_p13 = []
    for sid in train_ids:
        if sid in p11_map and sid in p13_map:
            train_p11.append(p11_map[sid])
            train_p13.append(p13_map[sid])

    if len(train_p11) == 0:
        raise RuntimeError("No train labels matched. Check ID formats between split and CSV (SID/id).")

    train_p11 = np.stack(train_p11, axis=0).astype(np.float32)
    train_p13 = np.stack(train_p13, axis=0).astype(np.float32)
    p11_mean, p11_std = compute_mean_std(train_p11)
    p13_mean, p13_std = compute_mean_std(train_p13)

    # datasets
    train_roi_dir = os.path.join(args.roi_dir, "train")
    val_roi_dir = os.path.join(args.roi_dir, "val")

    ds_train = RoiMultiTaskDataset(
        ids=train_ids,
        roi_dir=train_roi_dir,
        p11_map=p11_map,
        p13_map=p13_map,
        p11_mean=p11_mean, p11_std=p11_std,
        p13_mean=p13_mean, p13_std=p13_std,
        tfm=build_train_tf(args.img_size),
    )
    ds_val = RoiMultiTaskDataset(
        ids=val_ids,
        roi_dir=val_roi_dir,
        p11_map=p11_map,
        p13_map=p13_map,
        p11_mean=p11_mean, p11_std=p11_std,
        p13_mean=p13_mean, p13_std=p13_std,
        tfm=build_eval_tf(args.img_size),
    )

    print(f"Train samples (ROI+P11+P13): {len(ds_train)} | Val samples: {len(ds_val)}")
    if len(ds_train) == 0:
        raise RuntimeError("No training samples found. Check ROI paths and label IDs.")

    # dataloaders
    # Windows sometimes has DataLoader worker issues: set num_workers=0 if you hit problems.
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        drop_last=False,
    )

    # model
    model = P14MultiTaskNet(
        emb_dim=args.emb_dim,
        p11_dim=p11_dim,
        p13_dim=p13_dim,
        dropout=args.dropout,
    ).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = P14Config(
        emb_dim=args.emb_dim,
        p11_dim=p11_dim,
        p13_dim=p13_dim,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        w_p11=args.w_p11,
        w_p13=args.w_p13,
        seed=args.seed,
    )
    print(">>> P14 Config:", cfg)

    # output paths
    ensure_dir(args.out_dir)
    best_path = os.path.join(args.out_dir, args.name)
    norm_path = os.path.join(args.out_dir, "p14_norm.json")

    best_val = math.inf

    # train
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, dl_train, optimizer, device, args.w_p11, args.w_p13)

        if len(ds_val) > 0:
            va = eval_epoch(model, dl_val, device, args.w_p11, args.w_p13)
            val_loss = va["val_loss"]
        else:
            va = {"val_loss": float("nan"), "val_mse_p11": float("nan"), "val_mse_p13": float("nan"), "n": 0}
            val_loss = tr["train_loss"]

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={tr['train_loss']:.4f} (p11={tr['train_mse_p11']:.4f}, p13={tr['train_mse_p13']:.4f}, n={tr['n']}) | "
            f"val_loss={va['val_loss']:.4f} (p11={va['val_mse_p11']:.4f}, p13={va['val_mse_p13']:.4f}, n={va['n']})"
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "p11_cols": p11_cols,
                "p13_cols": p13_cols,
                "norm": {
                    "p11_mean": p11_mean.tolist(),
                    "p11_std": p11_std.tolist(),
                    "p13_mean": p13_mean.tolist(),
                    "p13_std": p13_std.tolist(),
                },
            }
            torch.save(ckpt, best_path)
            with open(norm_path, "w", encoding="utf-8") as f:
                json.dump(ckpt["norm"], f, ensure_ascii=False, indent=2)

            print(f"  ✅ Saved best: {best_path} (val_loss={best_val:.4f})")

    print("\nDone.")
    print("Best ckpt:", best_path)
    print("Norm json:", norm_path)


if __name__ == "__main__":
    main()
