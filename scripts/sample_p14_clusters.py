# scripts/sample_p14_clusters.py
import os, shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

class Args:
    emb_csv = "../outputs/p14_embedding/p14_emb_test.csv"
    roi_dir = "../outputs/roi/test"
    out_dir = "../outputs/p14_cluster_samples"
    k = 2
    per_cluster = 40
    seed = 42

def main():
    a = Args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.emb_csv)
    emb_cols = [c for c in df.columns if c.startswith("p14_emb_")]
    X = df[emb_cols].to_numpy(dtype=np.float32)
    ids = df["id"].astype(str).tolist()

    km = KMeans(n_clusters=a.k, random_state=a.seed, n_init="auto")
    lab = km.fit_predict(X)

    for c in range(a.k):
        cdir = os.path.join(a.out_dir, f"cluster_{c}")
        os.makedirs(cdir, exist_ok=True)
        idx = np.where(lab == c)[0]
        np.random.seed(a.seed)
        pick = np.random.choice(idx, size=min(a.per_cluster, len(idx)), replace=False)

        for i in pick:
            sid = ids[i]
            jpg = os.path.join(a.roi_dir, f"{sid}.jpg")
            png = os.path.join(a.roi_dir, f"{sid}.png")
            src = jpg if os.path.exists(jpg) else png if os.path.exists(png) else None
            if src:
                shutil.copy(src, os.path.join(cdir, os.path.basename(src)))

        print(f"cluster {c}: {len(idx)} samples, copied {len(pick)}")

    print("Saved to:", a.out_dir)

if __name__ == "__main__":
    main()
