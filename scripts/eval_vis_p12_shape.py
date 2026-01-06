import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------
# 与 data/splits/<split>.txt 对齐（默认 test）
# 输出整体 MAE/RMSE/R²/PearsonR
# -------------------------
SPLIT = "test"  # train / val / test
SPLIT_FILE = f"../data/splits/{SPLIT}.txt"

GT_CSV = "../data/labels/p12_tg_shape.csv"
MASK_DIR = f"../outputs/pred_masks/{SPLIT}"

OUT_DIR = "../outputs/vis_p12_shape"
OUT_CSV = f"../outputs/p12_eval_{SPLIT}.csv"

MIN_AREA_RATIO = 0.001  # mask 太小就跳过（防止噪声）


# -------------------------
# Metrics utils
# -------------------------
def pearsonr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + eps
    return float((a * b).sum() / denom)


def r2_score_np(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < eps:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mae_np(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse_np(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# -------------------------
# Data utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def largest_connected_component(mask01: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_id = 1 + int(np.argmax(areas))
    return (labels == max_id).astype(np.uint8)


def bbox_from_mask01(mask01: np.ndarray):
    ys, xs = np.where(mask01 == 1)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def compute_p12_from_mask(mask_path: str):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None

    mask01 = (m > 0).astype(np.uint8)

    h, w = mask01.shape
    if mask01.sum() < (h * w * MIN_AREA_RATIO):
        return None

    # num_tg: 连通域数量（去掉背景）
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    num_tg = max(0, num_labels - 1)

    # bbox 只用最大连通域更稳
    mask01_lcc = largest_connected_component(mask01)
    bbox = bbox_from_mask01(mask01_lcc)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    # 注意：这里 width/height 定义为 bbox 的像素跨度
    tg_width = float((x2 - x1) + 1)
    tg_height = float((y2 - y1) + 1)
    tg_w_div_h = float(tg_width / (tg_height + 1e-6))

    return num_tg, tg_width, tg_height, tg_w_div_h


# -------------------------
# Plot
# -------------------------
def scatter_plot(gt, pred, title, out_path):
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(gt, pred, s=8, alpha=0.6)

    minv = min(gt.min(), pred.min())
    maxv = max(gt.max(), pred.max())
    plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)

    pr = pearsonr(gt, pred)
    mae = mae_np(gt, pred)
    r2 = r2_score_np(gt, pred)

    plt.xlabel("GT")
    plt.ylabel("Pred (from mask)")
    plt.title(f"{title}\nR={pr:.3f}, R2={r2:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ensure_dir("outputs")
    ensure_dir(OUT_DIR)

    ids = load_ids(SPLIT_FILE)
    gt = pd.read_csv(GT_CSV)
    gt["SID"] = gt["SID"].astype(str)

    gt_map = {row["SID"]: row for _, row in gt.iterrows()}

    rows = []
    skipped = 0

    for sid in tqdm(ids, desc=f"P12 eval ({SPLIT})"):
        if sid not in gt_map:
            skipped += 1
            continue

        mask_path = os.path.join(MASK_DIR, sid + ".png")
        pred = compute_p12_from_mask(mask_path)
        if pred is None:
            skipped += 1
            continue

        num_tg_p, w_p, h_p, r_p = pred
        row = gt_map[sid]

        rows.append({
            "SID": sid,
            "gt_num_tg": float(row["num_tg"]),
            "gt_tg_width": float(row["tg_width"]),
            "gt_tg_height": float(row["tg_height"]),
            "gt_tg_w_div_h": float(row["tg_w_div_h"]),
            "pred_num_tg": float(num_tg_p),
            "pred_tg_width": float(w_p),
            "pred_tg_height": float(h_p),
            "pred_tg_w_div_h": float(r_p),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved aligned file: {OUT_CSV}")
    print(f"Aligned samples: {len(df)} | Skipped: {skipped}")

    # ---- metrics summary ----
    def summary(name, gt_col, pred_col):
        y = df[gt_col].to_numpy()
        p = df[pred_col].to_numpy()
        print(f"\n{name}")
        print(f"  MAE  = {mae_np(y, p):.4f}")
        print(f"  RMSE = {rmse_np(y, p):.4f}")
        print(f"  R2   = {r2_score_np(y, p):.4f}")
        print(f"  R    = {pearsonr(y, p):.4f}")

    summary("tg_width", "gt_tg_width", "pred_tg_width")
    summary("tg_height", "gt_tg_height", "pred_tg_height")
    summary("tg_w_div_h", "gt_tg_w_div_h", "pred_tg_w_div_h")
    summary("num_tg", "gt_num_tg", "pred_num_tg")

    # ---- plots ----
    scatter_plot(df["gt_tg_width"].to_numpy(), df["pred_tg_width"].to_numpy(),
                 "P12 tg_width", os.path.join(OUT_DIR, f"scatter_tg_width_{SPLIT}.png"))
    scatter_plot(df["gt_tg_height"].to_numpy(), df["pred_tg_height"].to_numpy(),
                 "P12 tg_height", os.path.join(OUT_DIR, f"scatter_tg_height_{SPLIT}.png"))
    scatter_plot(df["gt_tg_w_div_h"].to_numpy(), df["pred_tg_w_div_h"].to_numpy(),
                 "P12 tg_w_div_h", os.path.join(OUT_DIR, f"scatter_tg_w_div_h_{SPLIT}.png"))

    print("\nSaved plots to:", OUT_DIR)


if __name__ == "__main__":
    main()
