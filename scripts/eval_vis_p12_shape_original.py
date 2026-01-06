import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------
# P12 对齐 + 指标 + scatter（修复 num_tg）
# -------------------------
SPLIT = "val"  # train/val/test
SPLIT_FILE = f"../data/splits/{SPLIT}.txt"

GT_CSV = "../data/labels/p12_tg_shape.csv"
MASK_DIR = f"../outputs/pred_masks_original/{SPLIT}"

OUT_DIR = "../outputs/vis_p12_shape_original"
OUT_CSV = f"../outputs/p12_eval_original_{SPLIT}.csv"


def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


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

def mae_np(y, p): return float(np.mean(np.abs(y - p)))
def rmse_np(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def bbox_from_mask01(mask01):
    ys, xs = np.where(mask01 == 1)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def compute_p12_from_original_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    mask01 = (m > 0).astype(np.uint8)

    # num_tg：按 tongue pixel count/area（更符合你 GT 的量级）
    num_tg_area = float(mask01.sum())  # 0/1 的像素个数

    bbox = bbox_from_mask01(mask01)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    tg_width = float((x2 - x1) + 1)
    tg_height = float((y2 - y1) + 1)
    tg_w_div_h = float(tg_width / (tg_height + 1e-6))

    return num_tg_area, tg_width, tg_height, tg_w_div_h


def scatter_plot(gt, pred, title, out_path):
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(gt, pred, s=8, alpha=0.6)
    minv = min(gt.min(), pred.min())
    maxv = max(gt.max(), pred.max())
    plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)

    pr = pearsonr(gt, pred)
    r2 = r2_score_np(gt, pred)
    mae = mae_np(gt, pred)

    plt.xlabel("GT")
    plt.ylabel("Pred (orig mask)")
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

    for sid in tqdm(ids, desc=f"P12 eval orig ({SPLIT})"):
        if sid not in gt_map:
            skipped += 1
            continue

        mask_path = os.path.join(MASK_DIR, sid + ".png")
        pred = compute_p12_from_original_mask(mask_path)
        if pred is None:
            skipped += 1
            continue

        num_p, w_p, h_p, r_p = pred
        row = gt_map[sid]

        rows.append({
            "SID": sid,
            "gt_num_tg": float(row["num_tg"]),
            "gt_tg_width": float(row["tg_width"]),
            "gt_tg_height": float(row["tg_height"]),
            "gt_tg_w_div_h": float(row["tg_w_div_h"]),
            "pred_num_tg": float(num_p),
            "pred_tg_width": float(w_p),
            "pred_tg_height": float(h_p),
            "pred_tg_w_div_h": float(r_p),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("\n❌ Aligned 为 0：说明 val 的原图尺度 mask 没有生成，或 MASK_DIR 路径不对。")
        print("请检查 MASK_DIR:", MASK_DIR)
        print("并确认该目录下存在 SID.png 文件。")
        return
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Aligned: {len(df)} | Skipped: {skipped}")

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
    summary("num_tg(area)", "gt_num_tg", "pred_num_tg")

    scatter_plot(df["gt_tg_width"].to_numpy(), df["pred_tg_width"].to_numpy(),
                 "P12 tg_width (orig)", os.path.join(OUT_DIR, f"scatter_tg_width_{SPLIT}.png"))
    scatter_plot(df["gt_tg_height"].to_numpy(), df["pred_tg_height"].to_numpy(),
                 "P12 tg_height (orig)", os.path.join(OUT_DIR, f"scatter_tg_height_{SPLIT}.png"))
    scatter_plot(df["gt_tg_w_div_h"].to_numpy(), df["pred_tg_w_div_h"].to_numpy(),
                 "P12 tg_w_div_h (orig)", os.path.join(OUT_DIR, f"scatter_tg_w_div_h_{SPLIT}.png"))

    print("\nSaved plots to:", OUT_DIR)


if __name__ == "__main__":
    main()
