import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from src.datasets.p13_texture_dataset import P13TextureDataset


IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 0

ROI_DIR = "../outputs/roi/test"
LABEL_CSV = "../data/labels/p13_tg_texture.csv"
SPLIT_TXT = "../data/splits/test.txt"

CKPT_PATH = "../checkpoints/p13/p13_texture_best.pth"
NORM_PATH = "../checkpoints/p13/p13_texture_norm.json"

OUT_PRED = "../outputs/p13_texture_pred_test.csv"
OUT_METRICS = "../outputs/p13_texture_metrics.csv"


class Regressor(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, out_dim)
        self.net = m

    def forward(self, x):
        return self.net(x)


def pearsonr(a, b, eps=1e-12):
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + eps
    return float((a*b).sum() / denom)


def r2_score_np(y, p, eps=1e-12):
    y = y.astype(np.float64); p = p.astype(np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < eps: return float("nan")
    return float(1.0 - ss_res / ss_tot)


def main():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = P13TextureDataset(ROI_DIR, LABEL_CSV, SPLIT_TXT, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    with open(NORM_PATH, "r", encoding="utf-8") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    features = norm["features"]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    model = Regressor(out_dim=len(features)).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded:", CKPT_PATH)

    all_sid, all_gt, all_pred = [], [], []

    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc="Infer test"):
            x = x.to(device)
            y = y.to(device)
            y_n = (y - mean_t) / std_t
            pred_n = model(x)

            pred = (pred_n * std_t + mean_t).cpu().numpy()
            gt = y.cpu().numpy()

            all_sid.extend(list(sid))
            all_gt.append(gt)
            all_pred.append(pred)

    Y = np.concatenate(all_gt, axis=0)
    P = np.concatenate(all_pred, axis=0)

    # overall
    mae = float(np.mean(np.abs(Y - P)))
    rmse = float(np.sqrt(np.mean((Y - P) ** 2)))
    print("\n=== Overall (original scale) ===")
    print("MAE  =", mae)
    print("RMSE =", rmse)

    # per-feature metrics
    rows = []
    for j, f in enumerate(features):
        yj = Y[:, j]
        pj = P[:, j]
        rows.append({
            "feature": f,
            "MAE": float(np.mean(np.abs(yj - pj))),
            "RMSE": float(np.sqrt(np.mean((yj - pj) ** 2))),
            "R2": r2_score_np(yj, pj),
            "PearsonR": pearsonr(yj, pj)
        })

    dfm = pd.DataFrame(rows).sort_values("PearsonR", ascending=False)
    dfm.to_csv(OUT_METRICS, index=False)
    print("\nSaved per-feature metrics to:", OUT_METRICS)

    # save pred csv
    out = pd.DataFrame({"SID": all_sid})
    for j, f in enumerate(features):
        out[f"gt_{f}"] = Y[:, j]
        out[f"pred_{f}"] = P[:, j]
    out.to_csv(OUT_PRED, index=False)
    print("Saved predictions to:", OUT_PRED)

    print("\nTop-10 features by PearsonR:")
    print(dfm.head(10))


if __name__ == "__main__":
    main()
