import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.datasets.p11_color_dataset import P11ColorDataset
from src.models.p11_color_regressor import P11ColorRegressor

# 标准化后的多维回归模型 可解释评估 + 导出预测特征

LABEL_CSV = "../data/labels/p11_tg_color.csv"
CKPT_PATH = "../checkpoints/p11/p11_color_best.pth"

TEST_ROI_DIR = "../outputs/roi/test"
TEST_SPLIT = "../data/splits/test.txt"

OUT_CSV = "../outputs/p11_color_pred_test.csv"


def pearsonr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + eps
    return float((a * b).sum() / denom)


def main():
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset (注意：这里 normalize_y=True，返回的是标准化 y)
    ds = P11ColorDataset(TEST_ROI_DIR, TEST_SPLIT, LABEL_CSV, normalize_y=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # load ckpt
    ckpt = torch.load(CKPT_PATH, map_location=device)
    label_cols = ckpt["label_cols"]
    y_mean = ckpt["y_mean"].astype(np.float32)
    y_std = ckpt["y_std"].astype(np.float32)

    model = P11ColorRegressor(out_dim=len(label_cols)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_sid = []
    all_yz = []
    all_pz = []

    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc="Infer test"):
            x = x.to(device)
            pred = model(x).cpu().numpy()   # standardized pred
            y = y.numpy()                   # standardized gt
            all_pz.append(pred)
            all_yz.append(y)
            all_sid.extend(list(sid))

    yz = np.concatenate(all_yz, axis=0)
    pz = np.concatenate(all_pz, axis=0)

    # 反归一化回原始尺度
    y = yz * y_std + y_mean
    p = pz * y_std + y_mean

    # overall metrics（在原始尺度下）
    overall_mae = mean_absolute_error(y, p)
    overall_rmse = np.sqrt(mean_squared_error(y, p))
    print("\n=== Overall (original scale) ===")
    print(f"MAE  = {overall_mae:.6f}")
    print(f"RMSE = {overall_rmse:.6f}")

    # per-dimension metrics
    maes = []
    rmses = []
    r2s = []
    prs = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        pj = p[:, j]
        maes.append(mean_absolute_error(yj, pj))
        rmses.append(np.sqrt(mean_squared_error(yj, pj)))
        # R2 在几乎常数的维度上会很不稳定，这里做个保护
        if np.std(yj) < 1e-8:
            r2s.append(np.nan)
        else:
            r2s.append(r2_score(yj, pj))
        prs.append(pearsonr(yj, pj))

    df_metrics = pd.DataFrame({
        "feature": label_cols,
        "MAE": maes,
        "RMSE": rmses,
        "R2": r2s,
        "PearsonR": prs
    }).sort_values("PearsonR", ascending=False)

    print("\nTop-10 features by PearsonR:")
    print(df_metrics.head(10).to_string(index=False))

    print("\nBottom-10 features by PearsonR:")
    print(df_metrics.tail(10).to_string(index=False))

    # export predictions
    # base SID
    df_sid = pd.DataFrame({"SID": all_sid})

    # gt / pred 分别建 DataFrame
    df_gt = pd.DataFrame(
        y,
        columns=[f"gt_{c}" for c in label_cols]
    )

    df_pred = pd.DataFrame(
        p,
        columns=[f"pred_{c}" for c in label_cols]
    )

    # 一次性拼接
    df_out = pd.concat([df_sid, df_gt, df_pred], axis=1)

    df_out.to_csv(OUT_CSV, index=False)

    df_out.to_csv(OUT_CSV, index=False)
    print("\nSaved predictions to:", OUT_CSV)

    # also export metrics
    metrics_path = "outputs/p11_color_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print("Saved per-feature metrics to:", metrics_path)


if __name__ == "__main__":
    main()
