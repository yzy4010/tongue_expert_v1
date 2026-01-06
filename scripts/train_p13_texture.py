import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from src.datasets.p13_texture_dataset import P13TextureDataset


# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 0  # Windows 建议 0，避免 spawn 报错

ROI_ROOT = "../outputs/roi"
LABEL_CSV = "../data/labels/p13_tg_texture.csv"
SPLITS = "../data/splits"

CKPT_DIR = "../checkpoints/p13"
CKPT_PATH = os.path.join(CKPT_DIR, "p13_texture_best.pth")
NORM_PATH = os.path.join(CKPT_DIR, "p13_texture_norm.json")


class Regressor(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        self.net = m

    def forward(self, x):
        return self.net(x)


def compute_norm(loader, device):
    ys = []
    for _, y, _ in tqdm(loader, desc="Compute target norm"):
        ys.append(y.numpy())
    Y = np.concatenate(ys, axis=0)  # [N, D]
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean, std


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = P13TextureDataset(
        roi_dir=os.path.join(ROI_ROOT, "train"),
        label_csv=LABEL_CSV,
        split_txt=os.path.join(SPLITS, "train.txt"),
        img_size=IMG_SIZE
    )
    val_ds = P13TextureDataset(
        roi_dir=os.path.join(ROI_ROOT, "val"),
        label_csv=LABEL_CSV,
        split_txt=os.path.join(SPLITS, "val.txt"),
        img_size=IMG_SIZE
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # target normalization (train stats)
    mean, std = compute_norm(train_loader, device)
    with open(NORM_PATH, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": train_ds.features}, f, indent=2)
    print("Saved norm:", NORM_PATH)

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    model = Regressor(out_dim=train_ds.target_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            y_n = (y - mean_t) / std_t

            pred = model(x)
            loss = criterion(pred, y_n)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(tr_losses)))

        # ---- val ----
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x = x.to(device)
                y = y.to(device)
                y_n = (y - mean_t) / std_t
                pred = model(x)
                loss = criterion(pred, y_n)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"[Epoch {epoch}] Train MSE: {tr:.6f} | Val MSE: {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict()}, CKPT_PATH)
            print(f"✅ Saved best: {CKPT_PATH} (val_mse={best_val:.6f})")


if __name__ == "__main__":
    main()
