import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.unet_roi import UNetROI  # 你刚创建的 6-class UNet


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "roi_seg_v1"
IMG_DIR = DATA_DIR / "images"
MSK_DIR = DATA_DIR / "masks_6class"
CKPT_DIR = ROOT / "checkpoints" / "roi_seg"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RoiSegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, input_size=(512, 512), augment=False):
        self.img_dir = Path(img_dir)
        self.msk_dir = Path(msk_dir)
        self.input_size = input_size
        self.augment = augment

        self.ids = sorted([p.stem for p in self.img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if len(self.ids) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.ids)

    def _aug(self, img, msk):
        # simple aug: horizontal flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            msk = np.ascontiguousarray(msk[:, ::-1])
        return img, msk

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = self.img_dir / f"{id_}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id {id_}")

        msk_path = self.msk_dir / f"{id_}.png"
        if not msk_path.exists():
            raise FileNotFoundError(f"Mask not found: {msk_path}")

        img_bgr = cv2.imread(str(img_path))
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if msk is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")

        # resize
        H, W = self.input_size
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img_bgr, msk = self._aug(img_bgr, msk)

        # IMPORTANT: keep consistent with inference cfg.bgr_to_rgb
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        x = torch.from_numpy(img).float()
        y = torch.from_numpy(msk.astype(np.int64))

        return x, y


def compute_class_weights(msk_dir: Path, num_classes=NUM_CLASSES):
    # very rough weights based on pixel frequency; good enough for v1
    counts = np.zeros(num_classes, dtype=np.float64)
    msk_paths = list(msk_dir.glob("*.png"))
    sample = msk_paths[: min(50, len(msk_paths))]  # sample for speed
    for p in sample:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    input_size = (512, 512)

    ds = RoiSegDataset(IMG_DIR, MSK_DIR, input_size=input_size, augment=True)
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(0.9 * n)
    train_ids, val_ids = idxs[:split], idxs[split:]

    train_ds = torch.utils.data.Subset(ds, train_ids)
    val_ds = torch.utils.data.Subset(RoiSegDataset(IMG_DIR, MSK_DIR, input_size=input_size, augment=False), val_ids)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = UNetROI(num_classes=NUM_CLASSES).to(device)

    class_w = compute_class_weights(MSK_DIR).to(device)
    ce = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best_val = 1e9
    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = ce(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)

        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = ce(logits, y)
                va_loss += loss.item() * x.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        ckpt_best = CKPT_DIR / "roi_seg_6class_v1_best.pth"
        ckpt_last = CKPT_DIR / "roi_seg_6class_v1_last.pth"

        # 每个 epoch 都保存 last
        torch.save(model.state_dict(), ckpt_last)

        # best 仍然按 val 最小保存
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), ckpt_best)
            print("  Saved BEST:", ckpt_best)

        print("  Saved LAST:", ckpt_last)


if __name__ == "__main__":
    train()
