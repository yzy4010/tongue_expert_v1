# -*- coding: utf-8 -*-
"""
Generate a markdown report for P14 embedding reproduction.

Run directly in PyCharm.
Output:
  outputs/p14_report.md
"""

import os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


class Args:
    out_path = "../outputs/p14_report.md"

    # paths to existing artifacts (adjust only if you moved files)
    emb_test = "../outputs/p14_embedding/p14_emb_test.csv"
    umap_png = "../outputs/p14_embedding_vis/p14_umap_test_color_p11_gt_tg_Red_avg.png"
    cluster_dir = "../outputs/p14_cluster_samples"

    ckpt = "../checkpoints/p14/p14_multitask_best.pth"
    norm = "../checkpoints/p14/p14_norm.json"

    # probe results (REAL numbers from your run)
    probe_block = "P11 (Color)"
    avg_pearsonr = 0.9037
    avg_r2 = 0.8184
    avg_mae = 20.1864


def main():
    a = Args()
    os.makedirs(os.path.dirname(a.out_path), exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# TongueExpert — P14 Embedding Reproduction Report

**Generated at:** {now}

---

## 1. Background

This report documents the reproduction and analysis of **P14-A**, a CNN-based embedding module
within the TongueExpert pipeline.  
The goal of P14 is to learn a **general-purpose tongue image representation** from ROI images,
using **proxy supervision** from previously reproduced phenotype modules (P11 color, P13 texture).

---

## 2. P14-A Method Overview

**Input:** Tongue ROI images (from segmentation + ROI crop)

**Backbone:** ResNet18 (ImageNet pretrained)

**Heads:**
- Embedding head: 128-d vector
- P11 regression head (color features)
- P13 regression head (texture features)

**Training objective:**

\\[
\\mathcal{{L}} = \\lambda_{{11}} \\cdot \\text{{MSE}}(P11_{{pred}}, P11_{{gt}})
              + \\lambda_{{13}} \\cdot \\text{{MSE}}(P13_{{pred}}, P13_{{gt}})
\\]

where P11/P13 targets are standardized using mean/std computed **only on the training split**.

---

## 3. Outputs

- **Checkpoint:** `{a.ckpt}`
- **Normalization stats:** `{a.norm}`
- **Test embeddings:** `{a.emb_test}` (128-d per sample)

---

## 4. Embedding Visualization (UMAP)

The learned P14 embeddings were projected to 2D using **UMAP** and colored by
`p11_gt_tg_Red_avg`.

![UMAP visualization]({a.umap_png})

### Observation

- The embedding space forms **two clearly separated major clusters**.
- Within each cluster, the color feature exhibits a **continuous gradient**,
  indicating that the embedding captures structured color variation rather than noise.

---

## 5. Cluster-level Qualitative Analysis

Representative ROI samples were extracted from two clusters using KMeans (k=2):

- `{a.cluster_dir}/cluster_0`
- `{a.cluster_dir}/cluster_1`

**Visual inspection shows:**

- **Cluster 0:** More exposed tongue body, thinner coating, stronger specular reflection,
  higher color saturation.
- **Cluster 1:** Heavier white coating, reduced reflection, paler and more matte appearance.

This suggests that **tongue coating coverage** emerges as a dominant latent factor in the
embedding space, which is consistent with clinical tongue diagnosis practice.

---

## 6. Linear Probe Evaluation (Representation Quality)

To evaluate whether the learned embedding itself encodes phenotype information,
a **linear probe (Ridge regression)** was trained on **frozen embeddings**.

### Probe Target
- **{a.probe_block}**

### Results (Test Split)

| Metric | Value |
|------|------|
| Avg PearsonR | **{a.avg_pearsonr:.4f}** |
| Avg R² | **{a.avg_r2:.4f}** |
| Avg MAE | **{a.avg_mae:.4f}** |

### Interpretation

- High PearsonR (~0.90) indicates that phenotype information is **linearly decodable**
  from the embedding.
- This confirms that P14 learns a **high-quality, reusable representation** rather than
  relying on task-specific nonlinear heads.

---

## 7. Summary

- P14-A successfully learns a structured tongue image embedding using proxy supervision.
- The embedding naturally organizes samples by clinically meaningful factors
  (e.g., tongue coating).
- Linear probe results demonstrate strong information retention and representation quality.

The P14 module therefore completes the **feature-level + representation-level reproduction**
of the TongueExpert pipeline and is ready for downstream analysis
(e.g., clustering, retrieval, or syndrome classification).

---

## 8. Key Artifacts

- Checkpoint: `{a.ckpt}`
- Embeddings: `{a.emb_test}`
- UMAP visualization: `{a.umap_png}`
- Cluster samples: `{a.cluster_dir}/cluster_*`

---
"""

    with open(a.out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("✅ Report generated:")
    print(os.path.abspath(a.out_path))


if __name__ == "__main__":
    main()
