import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# -------------------------
# Config (按你当前日志路径默认)
# -------------------------
PRED_CSV = "../outputs/p13_texture_pred_test.csv"
METRICS_CSV = "../outputs/p13_texture_metrics.csv"

ROI_DIR = "../outputs/roi/test"   # ROI 图目录：SID.jpg
OUT_DIR = "../outputs/vis_p13_texture"

TOP_K = 6          # 画 PearsonR Top-K 的散点图
WORST_N = 16       # 每个特征挑最差 N 个样本做可视化
GRID_COLS = 4      # 拼图列数
IMG_SIZE = 224     # ROI 读取后显示尺寸（不影响训练）


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_imread_rgb(path, size=224):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def scatter_one_feature(df_pred, feat, out_path):
    gt_col = f"gt_{feat}"
    pr_col = f"pred_{feat}"
    y = df_pred[gt_col].to_numpy(dtype=np.float64)
    p = df_pred[pr_col].to_numpy(dtype=np.float64)

    # metrics
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r = float(np.corrcoef(y, p)[0, 1])

    # plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y, p, s=10, alpha=0.6)

    mn = float(min(y.min(), p.min()))
    mx = float(max(y.max(), p.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)

    plt.xlabel("GT")
    plt.ylabel("Pred")
    plt.title(f"{feat}\nR={r:.3f}  MAE={mae:.4g}  RMSE={rmse:.4g}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def grid_worst_cases(df_pred, feat, roi_dir, out_path, worst_n=16, cols=4, img_size=224):
    gt_col = f"gt_{feat}"
    pr_col = f"pred_{feat}"

    df = df_pred[["SID", gt_col, pr_col]].copy()
    df["abs_err"] = (df[gt_col] - df[pr_col]).abs()
    df = df.sort_values("abs_err", ascending=False).head(worst_n)

    sids = df["SID"].tolist()
    errs = df["abs_err"].to_numpy()

    rows = int(math.ceil(worst_n / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))

    for i, sid in enumerate(sids):
        img_path = os.path.join(roi_dir, sid + ".jpg")
        img = safe_imread_rgb(img_path, size=img_size)

        ax = plt.subplot(rows, cols, i + 1)
        if img is None:
            ax.text(0.5, 0.5, f"Missing\n{sid}", ha="center", va="center")
            ax.axis("off")
            continue

        ax.imshow(img)
        gt = float(df.iloc[i][gt_col])
        pr = float(df.iloc[i][pr_col])
        ae = float(errs[i])
        ax.set_title(f"{sid}\nGT={gt:.4g}  Pred={pr:.4g}\n|e|={ae:.4g}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return df  # worst-case table


def main():
    ensure_dir(OUT_DIR)

    df_pred = pd.read_csv(PRED_CSV)
    df_met = pd.read_csv(METRICS_CSV).sort_values("PearsonR", ascending=False)

    # features list (from metrics)
    feats = df_met["feature"].tolist()
    top_feats = feats[:TOP_K]

    print("Top-K features:", top_feats)

    # 1) scatter for top-K
    scatter_dir = os.path.join(OUT_DIR, "scatter_topk")
    ensure_dir(scatter_dir)
    for feat in top_feats:
        out_path = os.path.join(scatter_dir, f"scatter_{feat}.png")
        scatter_one_feature(df_pred, feat, out_path)

    # 2) worst-case grids + save tables
    worst_dir = os.path.join(OUT_DIR, "worstcase_grids")
    ensure_dir(worst_dir)

    all_worst = []
    for feat in top_feats:
        out_path = os.path.join(worst_dir, f"worst_{feat}.png")
        df_w = grid_worst_cases(df_pred, feat, ROI_DIR, out_path,
                                worst_n=WORST_N, cols=GRID_COLS, img_size=IMG_SIZE)
        df_w.insert(1, "feature", feat)
        all_worst.append(df_w)

    df_all = pd.concat(all_worst, axis=0, ignore_index=True)
    worst_csv = os.path.join(OUT_DIR, "worst_cases_topk.csv")
    df_all.to_csv(worst_csv, index=False)

    print("Saved scatter to:", scatter_dir)
    print("Saved worst-case grids to:", worst_dir)
    print("Saved worst-case table:", worst_csv)
    print("Done.")


if __name__ == "__main__":
    main()
