import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# P11 颜色特征学习的可视化测试脚本
# 一体化脚本：生成 metrics + 可视化
# -------------------------
PRED_CSV = "../outputs/p11_color_pred_test.csv"
OUT_DIR = "../outputs/vis_p11_color"
METRIC_OUT = "../outputs/p11_color_metrics.csv"

TOP_K = 6          # 画 PearsonR 最好的 K 个特征
NUM_SAMPLE = 2     # 随机画几个样本的向量对比


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pearsonr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + eps
    return float((a * b).sum() / denom)


def r2_score_np(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    # 纯 numpy 版 R2，避免额外依赖
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < eps:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def scatter_plot(gt, pred, feature, pearson, mae, out_path):
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(gt, pred, s=8, alpha=0.6)
    minv = min(gt.min(), pred.min())
    maxv = max(gt.max(), pred.max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)

    plt.xlabel("GT")
    plt.ylabel("Pred")
    plt.title(f"{feature}\nR={pearson:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def bar_plot(gt, pred, feature_names, sid, out_path):
    x = np.arange(len(feature_names))
    width = 0.38

    plt.figure(figsize=(max(8, len(feature_names) * 0.45), 4.4))
    plt.bar(x - width / 2, gt, width, label="GT")
    plt.bar(x + width / 2, pred, width, label="Pred")
    plt.xticks(x, feature_names, rotation=90)
    plt.ylabel("Value")
    plt.title(f"SID: {sid}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ensure_dir("outputs")
    ensure_dir(OUT_DIR)

    df = pd.read_csv(PRED_CSV)

    # 从列名自动解析 feature 列表：gt_xxx / pred_xxx
    gt_cols = [c for c in df.columns if c.startswith("gt_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if len(gt_cols) == 0 or len(pred_cols) == 0:
        raise RuntimeError("pred csv 中没找到 gt_*/pred_* 列，请确认文件是否正确。")

    features = [c[len("gt_"):] for c in gt_cols]
    # 确保 pred_* 也齐全
    features = [f for f in features if f"pred_{f}" in df.columns]
    print("Detected feature dim:", len(features))

    # 计算每个特征的指标
    rows = []
    for f in features:
        gt = df[f"gt_{f}"].to_numpy()
        pred = df[f"pred_{f}"].to_numpy()

        mae = float(np.mean(np.abs(gt - pred)))
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        r2 = r2_score_np(gt, pred)
        pr = pearsonr(gt, pred)

        rows.append({"feature": f, "MAE": mae, "RMSE": rmse, "R2": r2, "PearsonR": pr})

    dfm = pd.DataFrame(rows).sort_values("PearsonR", ascending=False)
    dfm.to_csv(METRIC_OUT, index=False)
    print("Saved metrics:", METRIC_OUT)

    top_feats = dfm.head(TOP_K)
    feat_list = top_feats["feature"].tolist()
    print("Top features:", feat_list)

    # 1) Scatter：Top-K
    for _, r in top_feats.iterrows():
        f = r["feature"]
        gt = df[f"gt_{f}"].to_numpy()
        pred = df[f"pred_{f}"].to_numpy()
        out_path = os.path.join(OUT_DIR, f"scatter_{f}.png")
        scatter_plot(gt, pred, f, r["PearsonR"], r["MAE"], out_path)

    # 2) Bar：随机样本（Top-K 特征向量对比）
    sids = df["SID"].tolist() if "SID" in df.columns else list(range(len(df)))
    random.seed(42)
    sel = random.sample(sids, k=min(NUM_SAMPLE, len(sids)))

    for sid in sel:
        row = df[df["SID"] == sid].iloc[0] if "SID" in df.columns else df.iloc[int(sid)]
        gt = [row[f"gt_{f}"] for f in feat_list]
        pred = [row[f"pred_{f}"] for f in feat_list]
        out_path = os.path.join(OUT_DIR, f"bar_{sid}.png")
        bar_plot(gt, pred, feat_list, str(sid), out_path)

    print("All visualizations saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
