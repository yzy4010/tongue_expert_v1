# -*- coding: utf-8 -*-
"""
scripts/vis_p14_embedding.py

Visualize P14 embeddings (UMAP / t-SNE) and color by a chosen GT feature.
Run directly in PyCharm.

Requires:
  pip install umap-learn scikit-learn matplotlib pandas

  可视化：scripts/vis_p14_embedding.py（UMAP + t-SNE + 按 GT 上色）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

try:
    import umap
except Exception:
    umap = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


class Args:
    split = "test"
    emb_csv = "../outputs/p14_embedding/p14_emb_test.csv"

    # optional: color by GT (from your e2e_test_all.csv)
    # If you want: set this to your saved e2e file
    e2e_csv = "../outputs/e2e_test/e2e_test_all.csv"

    # choose one GT column to color points
    # examples: "p11_gt_tg_Red_avg", "p11_gt_tg_L_avg", "p13_gt_tg_homogeneity"
    color_col = "p11_gt_tg_Red_avg"

    out_dir = "../outputs/p14_embedding_vis"
    method = "umap"  # "umap" or "tsne"
    random_state = 42


def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.emb_csv)
    emb_cols = [c for c in df.columns if c.startswith("p14_emb_")]
    X = df[emb_cols].to_numpy(dtype=np.float32)
    ids = df["id"].astype(str).tolist()

    # color values (optional)
    color = None
    if args.e2e_csv and os.path.exists(args.e2e_csv):
        e2e = pd.read_csv(args.e2e_csv)
        e2e["id"] = e2e["id"].astype(str)
        if args.color_col in e2e.columns:
            m = e2e.set_index("id").loc[ids]
            color = m[args.color_col].to_numpy()
        else:
            print(f"[WARN] color_col not found in e2e_csv: {args.color_col}")
    else:
        print("[WARN] e2e_csv not found, will plot without color")

    # reduce to 2D
    if args.method.lower() == "umap":
        if umap is None:
            raise RuntimeError("umap-learn not installed. Run: pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, metric="cosine", random_state=args.random_state)
        Z = reducer.fit_transform(X)
        tag = "umap"
    else:
        reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.random_state, learning_rate="auto")
        Z = reducer.fit_transform(X)
        tag = "tsne"

    # plot
    plt.figure(figsize=(9, 7))
    if color is None:
        plt.scatter(Z[:, 0], Z[:, 1], s=8)
        plt.title(f"P14 embedding {tag} ({args.split})")
    else:
        sc = plt.scatter(Z[:, 0], Z[:, 1], s=8, c=color)
        plt.title(f"P14 embedding {tag} ({args.split}) colored by {args.color_col}")
        plt.colorbar(sc)

    out_png = os.path.join(args.out_dir, f"p14_{tag}_{args.split}_color_{args.color_col}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
