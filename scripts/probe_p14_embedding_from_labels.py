# -*- coding: utf-8 -*-
"""
Probe P14 embedding using GT directly from labels CSV
(no dependency on eval_all_test / e2e files)

Usage:
  Just click Run in PyCharm.

What it does:
  - Train linear Ridge regressor on TRAIN embeddings
  - Predict P11 or P13 GT on TEST embeddings
  - Report PearsonR / R2 / MAE
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


class Args:
    # embeddings
    emb_train = "../outputs/p14_embedding/p14_emb_train.csv"
    emb_test  = "../outputs/p14_embedding/p14_emb_test.csv"

    # labels
    labels_dir = "../data/labels"

    # choose block: "p11" or "p13"
    block = "p11"

    ridge_alpha = 1.0


def load_labels(block: str):
    if block == "p11":
        path = "../data/labels/p11_tg_color.csv"
        prefix = "p11_gt_"
    else:
        path = "../data/labels/p13_tg_texture.csv"
        prefix = "p13_gt_"

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # handle ID column
    if "id" not in df.columns:
        for cand in ["SID", "sid", "ID", "Id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "id"})
                break

    df["id"] = df["id"].astype(str).str.strip()
    y_cols = [c for c in df.columns if c != "id"]
    return df, y_cols, prefix


def main():
    a = Args()

    # load embeddings
    trE = pd.read_csv(a.emb_train)
    teE = pd.read_csv(a.emb_test)
    emb_cols = [c for c in trE.columns if c.startswith("p14_emb_")]

    trE["id"] = trE["id"].astype(str)
    teE["id"] = teE["id"].astype(str)

    # load GT
    gt_df, y_cols, prefix = load_labels(a.block)

    # merge
    tr = trE.merge(gt_df, on="id", how="inner")
    te = teE.merge(gt_df, on="id", how="inner")

    print(f"Train samples: {len(tr)} | Test samples: {len(te)}")
    print(f"Target block: {a.block} | dims: {len(y_cols)}")

    Xtr = tr[emb_cols].to_numpy(np.float32)
    Ytr = tr[y_cols].to_numpy(np.float32)
    Xte = te[emb_cols].to_numpy(np.float32)
    Yte = te[y_cols].to_numpy(np.float32)

    # fit ridge
    reg = Ridge(alpha=a.ridge_alpha)
    reg.fit(Xtr, Ytr)
    Yhat = reg.predict(Xte)

    # metrics
    prs, r2s, maes = [], [], []
    for j in range(Yte.shape[1]):
        y, p = Yte[:, j], Yhat[:, j]
        prs.append(pearsonr(y, p)[0])
        r2s.append(r2_score(y, p))
        maes.append(mean_absolute_error(y, p))

    print("\n=== Linear Probe Result ===")
    print(f"Avg PearsonR: {np.nanmean(prs):.4f}")
    print(f"Avg R2      : {np.nanmean(r2s):.4f}")
    print(f"Avg MAE     : {np.nanmean(maes):.4f}")

    order = np.argsort(prs)[::-1]
    print("\nTop-5 dimensions:")
    for k in range(5):
        j = order[k]
        print(f"  {y_cols[j]:30s}  R={prs[j]:.4f}  R2={r2s[j]:.4f}")

    print("\nBottom-5 dimensions:")
    for k in range(5):
        j = order[-(k+1)]
        print(f"  {y_cols[j]:30s}  R={prs[j]:.4f}  R2={r2s[j]:.4f}")


if __name__ == "__main__":
    main()
