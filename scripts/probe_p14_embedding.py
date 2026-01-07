# -*- coding: utf-8 -*-
"""
scripts/probe_p14_embedding.py

Linear probe on P14 embeddings:
  - Train a linear regressor (Ridge) on train split embeddings to predict P11/P13 GT
  - Evaluate on test split

Run directly in PyCharm.

Requires:
  pip install scikit-learn pandas numpy

  证明 embedding 可预测 P11/P13
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
    emb_test = "../outputs/p14_embedding/p14_emb_test.csv"

    # GT source: best is your e2e_test_all.csv + e2e_train_all.csv
    # If you don't have train e2e yet, you can directly load from labels CSV, but easiest is e2e.
    gt_train = "../outputs/e2e_test/e2e_train_all.csv"   # if not exists, see note below
    gt_test = "../outputs/e2e_test/e2e_test_all.csv"

    # which target block to probe
    block = "p11"  # "p11" or "p13"

    # ridge strength
    alpha = 1.0


def find_target_cols(df: pd.DataFrame, block: str):
    if block == "p11":
        prefix = "p11_gt_"
    else:
        prefix = "p13_gt_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No target columns with prefix {prefix} found.")
    return cols


def main():
    args = Args()

    # load embeddings
    trE = pd.read_csv(args.emb_train)
    teE = pd.read_csv(args.emb_test)
    emb_cols = [c for c in trE.columns if c.startswith("p14_emb_")]

    # load GT tables
    if not os.path.exists(args.gt_train):
        raise RuntimeError(
            f"Missing gt_train: {args.gt_train}\n"
            f"Quick fix: run eval_all_test.py for split=train to produce e2e_train_all.csv\n"
            f"Or tell me and I’ll give a probe script that reads labels CSV directly."
        )
    trG = pd.read_csv(args.gt_train)
    teG = pd.read_csv(args.gt_test)

    # align by id
    for df in [trE, teE, trG, teG]:
        df["id"] = df["id"].astype(str)

    tr = trE.merge(trG, on="id", how="inner")
    te = teE.merge(teG, on="id", how="inner")

    ycols = find_target_cols(tr, args.block)

    Xtr = tr[emb_cols].to_numpy(dtype=np.float32)
    Xte = te[emb_cols].to_numpy(dtype=np.float32)
    Ytr = tr[ycols].to_numpy(dtype=np.float32)
    Yte = te[ycols].to_numpy(dtype=np.float32)

    # fit one ridge per dimension (multioutput supported)
    reg = Ridge(alpha=args.alpha)
    reg.fit(Xtr, Ytr)
    Yhat = reg.predict(Xte)

    # metrics per-dim + average
    r2s = []
    maes = []
    prs = []
    for j in range(Yte.shape[1]):
        y = Yte[:, j]
        p = Yhat[:, j]
        r2s.append(r2_score(y, p))
        maes.append(mean_absolute_error(y, p))
        prs.append(pearsonr(y, p)[0])

    print(f"Linear probe block={args.block} | test n={len(te)} dims={Yte.shape[1]}")
    print(f"Avg PearsonR: {float(np.nanmean(prs)):.4f}")
    print(f"Avg R2      : {float(np.nanmean(r2s)):.4f}")
    print(f"Avg MAE     : {float(np.nanmean(maes)):.4f}")

    # show best/worst dims
    order = np.argsort(prs)[::-1]
    print("\nTop-5 dims by PearsonR:")
    for k in range(5):
        j = int(order[k])
        print(f"  {ycols[j]}  pearsonr={prs[j]:.4f}  r2={r2s[j]:.4f}  mae={maes[j]:.4f}")

    order_w = np.argsort(prs)
    print("\nWorst-5 dims by PearsonR:")
    for k in range(5):
        j = int(order_w[k])
        print(f"  {ycols[j]}  pearsonr={prs[j]:.4f}  r2={r2s[j]:.4f}  mae={maes[j]:.4f}")


if __name__ == "__main__":
    main()
