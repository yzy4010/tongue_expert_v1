# -*- coding: utf-8 -*-
"""
Generate bilingual (Chinese + English) markdown reports for P14 embedding reproduction.

Run directly in PyCharm.

Outputs:
  outputs/p14_report_zh.md
  outputs/p14_report_en.md
"""

import os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


class Args:
    out_zh = "../outputs/p14_report_zh.md"
    out_en = "../outputs/p14_report_en.md"

    # artifact paths (adjust only if moved)
    emb_test = "../outputs/p14_embedding/p14_emb_test.csv"
    umap_png = "../outputs/p14_embedding_vis/p14_umap_test_color_p11_gt_tg_Red_avg.png"
    cluster_dir = "../outputs/p14_cluster_samples"

    ckpt = "../checkpoints/p14/p14_multitask_best.pth"
    norm = "../checkpoints/p14/p14_norm.json"

    # probe results (REAL numbers from your run)
    avg_pearsonr = 0.9037
    avg_r2 = 0.8184
    avg_mae = 20.1864


def main():
    a = Args()
    os.makedirs(os.path.dirname(a.out_zh), exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # =======================
    # Chinese version
    # =======================
    md_zh = f"""# TongueExpert — P14 表征模块复现报告

**生成时间：** {now}

---

## 一、背景与目标

本报告记录了 TongueExpert 复现项目中 **P14-A（CNN Embedding）** 模块的实现与分析结果。

P14 的目标是在舌象 ROI 图像上学习一个 **通用的低维表征（embedding）**，
并通过 **P11（颜色特征）与 P13（纹理特征）** 作为代理监督，
使该 embedding 在脱离原始图像的情况下，仍能保留关键舌象表型信息。

---

## 二、P14-A 方法说明

**输入：** 舌体 ROI 图像（由分割 + 裁剪得到）

**骨干网络：** ResNet18（ImageNet 预训练）

**输出头（Heads）：**
- Embedding head：128 维向量
- P11 回归 head（颜色特征）
- P13 回归 head（纹理特征）

**训练目标函数：**

\\[
\\mathcal{{L}} = \\lambda_{{11}} \\cdot \\text{{MSE}}(P11_{{pred}}, P11_{{gt}})
              + \\lambda_{{13}} \\cdot \\text{{MSE}}(P13_{{pred}}, P13_{{gt}})
\\]

其中，P11 与 P13 特征均基于 **训练集** 统计的 mean / std 进行标准化。

---

## 三、输出结果

- **模型权重：** `{a.ckpt}`
- **标准化参数：** `{a.norm}`
- **测试集 embedding：** `{a.emb_test}`（每个样本 128 维）

---

## 四、Embedding 可视化分析（UMAP）

对 P14 embedding 进行 UMAP 降维，并使用 `p11_gt_tg_Red_avg` 进行着色：

![UMAP 可视化]({a.umap_png})

### 观察结论

- embedding 空间呈现出 **两个明显分离的主簇**；
- 每个簇内部存在连续的颜色梯度变化；
- 说明 P14 embedding 捕获了结构化且可解释的颜色主轴，而非随机噪声。

---

## 五、聚类样本的定性分析

使用 KMeans（k=2）对 embedding 进行聚类，并抽样对应 ROI：

- `{a.cluster_dir}/cluster_0`
- `{a.cluster_dir}/cluster_1`

**肉眼观察发现：**

- **cluster_0：** 舌质暴露更多、苔薄或少苔、反光较强、颜色更偏红；
- **cluster_1：** 白苔覆盖明显、整体偏哑光、颜色更偏粉白。

该结果表明：  
**“舌苔覆盖程度” 作为一个临床重要的宏观因素，自然地成为 embedding 空间中的主导潜在变量。**

---

## 六、线性探针（Linear Probe）评估

为验证 embedding 本身是否编码了颜色表型信息，
在冻结 P14 embedding 的情况下，使用 **线性 Ridge 回归** 预测 P11 特征。

### 探针任务
- P11（颜色特征）

### 测试集结果

| 指标 | 数值 |
|-----|-----|
| 平均 PearsonR | **{a.avg_pearsonr:.4f}** |
| 平均 R² | **{a.avg_r2:.4f}** |
| 平均 MAE | **{a.avg_mae:.4f}** |

### 结论

- 高 PearsonR（≈0.90）表明 P11 信息可被 **线性解码**；
- 说明 P14 embedding 是一个 **高质量、可复用的舌象表征**，
  而非仅服务于训练阶段的中间特征。

---

## 七、小结

- P14-A 成功学习到了结构化的舌象 embedding；
- embedding 自然反映了具有临床意义的舌苔与颜色差异；
- 线性探针验证了其表征能力与下游可用性。

至此，TongueExpert 的 **特征级 + 表征级复现闭环** 已完整建立。

---

## 八、关键产物汇总

- 模型权重：`{a.ckpt}`
- Embedding 文件：`{a.emb_test}`
- UMAP 图像：`{a.umap_png}`
- 聚类样本：`{a.cluster_dir}/cluster_*`

---
"""

    # =======================
    # English version
    # =======================
    md_en = f"""# TongueExpert — P14 Embedding Reproduction Report

**Generated at:** {now}

---

## 1. Background

This report documents the reproduction and analysis of **P14-A**, a CNN-based embedding module
within the TongueExpert pipeline.

The goal of P14 is to learn a **general-purpose tongue image representation** from ROI images,
using **proxy supervision** from previously reproduced phenotype modules
(P11 color and P13 texture).

---

## 2. P14-A Method Overview

**Input:** Tongue ROI images (from segmentation and ROI cropping)

**Backbone:** ResNet18 (ImageNet pretrained)

**Heads:**
- Embedding head: 128-dimensional vector
- P11 regression head (color features)
- P13 regression head (texture features)

**Training objective:**

\\[
\\mathcal{{L}} = \\lambda_{{11}} \\cdot \\text{{MSE}}(P11_{{pred}}, P11_{{gt}})
              + \\lambda_{{13}} \\cdot \\text{{MSE}}(P13_{{pred}}, P13_{{gt}})
\\]

P11 and P13 targets are standardized using mean/std computed **only on the training split**.

---

## 3. Outputs

- **Checkpoint:** `{a.ckpt}`
- **Normalization stats:** `{a.norm}`
- **Test embeddings:** `{a.emb_test}` (128-d per sample)

---

## 4. Embedding Visualization (UMAP)

The learned embeddings were projected to 2D using **UMAP** and colored by
`p11_gt_tg_Red_avg`.

![UMAP visualization]({a.umap_png})

### Observation

- The embedding space forms **two clearly separated major clusters**;
- A continuous color gradient is observed within each cluster;
- This indicates that the embedding captures structured color variation rather than noise.

---

## 5. Qualitative Cluster Analysis

Representative ROI samples were extracted from two clusters using KMeans (k=2):

- `{a.cluster_dir}/cluster_0`
- `{a.cluster_dir}/cluster_1`

**Visual inspection shows:**

- **Cluster 0:** More exposed tongue body, thinner coating, stronger specular reflection,
  and higher color saturation.
- **Cluster 1:** Heavier white coating, reduced reflection, and paler, more matte appearance.

This suggests that **tongue coating coverage** emerges as a dominant latent factor
in the embedding space, which is clinically meaningful.

---

## 6. Linear Probe Evaluation

To evaluate representation quality, a **linear probe (Ridge regression)** was trained
on frozen embeddings to predict P11 color features.

### Probe target
- P11 (Color)

### Results (Test Split)

| Metric | Value |
|------|------|
| Avg PearsonR | **{a.avg_pearsonr:.4f}** |
| Avg R² | **{a.avg_r2:.4f}** |
| Avg MAE | **{a.avg_mae:.4f}** |

### Interpretation

- High PearsonR (~0.90) indicates strong linear decodability of color phenotypes;
- The embedding therefore constitutes a **high-quality and reusable representation**
  suitable for downstream analysis.

---

## 7. Summary

- P14-A successfully learns a structured tongue image embedding using proxy supervision;
- The embedding naturally organizes samples by clinically relevant factors
  (e.g., tongue coating);
- Linear probe results confirm strong representation quality and information retention.

This completes the **feature-level and representation-level reproduction**
of the TongueExpert pipeline.

---

## 8. Key Artifacts

- Checkpoint: `{a.ckpt}`
- Embeddings: `{a.emb_test}`
- UMAP visualization: `{a.umap_png}`
- Cluster samples: `{a.cluster_dir}/cluster_*`

---
"""

    with open(a.out_zh, "w", encoding="utf-8") as f:
        f.write(md_zh)
    with open(a.out_en, "w", encoding="utf-8") as f:
        f.write(md_en)

    print("✅ Reports generated:")
    print(os.path.abspath(a.out_zh))
    print(os.path.abspath(a.out_en))


if __name__ == "__main__":
    main()
