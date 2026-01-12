# -*- coding: utf-8 -*-
"""
Build PCA for P14 embeddings and save to checkpoint.

Input:
  outputs/p14_embedding/p14_emb_train.csv

Output:
  checkpoints/p14/p14_pca.pkl
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.decomposition import PCA

# ---------- paths ----------
EMB_CSV = "../outputs/p14_embedding/p14_emb_train.csv"   # 你已有
OUT_PCA = "../checkpoints/p14/p14_pca.pkl"
N_COMP = 10

def main():
    assert os.path.exists(EMB_CSV), f"Missing {EMB_CSV}"

    df = pd.read_csv(EMB_CSV)

    emb_cols = [c for c in df.columns if c.startswith("p14_emb_")]
    assert len(emb_cols) > 0, "No p14_emb_* columns found"

    X = df[emb_cols].to_numpy(dtype=np.float32)

    print("Embedding shape:", X.shape)

    pca = PCA(n_components=N_COMP, random_state=0)
    pca.fit(X)

    Path(os.path.dirname(OUT_PCA)).mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, OUT_PCA)

    print("Saved PCA to:", OUT_PCA)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative:", pca.explained_variance_ratio_.sum())

if __name__ == "__main__":
    main()
