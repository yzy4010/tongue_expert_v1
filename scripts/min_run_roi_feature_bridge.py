from pathlib import Path
import json
import cv2
import torch
import numpy as np

# -------------------------------------------------
# 1. infer_regression：直接从 eval_all_test.py 引入
# -------------------------------------------------
# 注意：必须从 scripts.eval_all_test import
# 如果你是从项目根目录运行 python scripts/xxx.py，这是 OK 的
from scripts.eval_all_test import infer_regression
from scripts.test_roi_seg_6class import roi_infer

# -------------------------------------------------
# 2. roi_feature_bridge（你刚生成的完整版本）
# -------------------------------------------------
from src.pipeline.roi_feature_bridge import extract_roi_features_all


# -------------------------------------------------
# 3. 一个“内嵌的最小 ROI 图像导出函数”
#    （替代 export_roi_images）
# -------------------------------------------------
def export_roi_images_min(img_bgr, roi_masks, sample_id, out_root: Path, min_area=200):
    out_root.mkdir(parents=True, exist_ok=True)

    for roi_name, mask in roi_masks.items():
        if roi_name == "tongue":
            continue
        if mask is None or (mask > 0).sum() < min_area:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        crop = img_bgr[y1:y2+1, x1:x2+1]
        roi_dir = out_root / roi_name
        roi_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(roi_dir / f"{sample_id}.jpg"), crop)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------------------------------
    # A. 输入：一张图 + 你已经得到的 roi_masks
    # -------------------------------------------------
    sample_id = "TE0000001"
    img_path = Path("data/raw/images") / f"{sample_id}.jpg"
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Failed to read image: {img_path}"

    # -------------------------------------------------
    # B. 这里假设你已经有 roi_masks
    #    👉 请直接从你 test_roi_seg_6class.py 中复制那一行
    # -------------------------------------------------
    # 示例（⚠️你要替换成你真实的那一行）：
    roi_masks = roi_infer.infer_roi_masks(img_bgr)
    raise RuntimeError(
        "请把你在 test_roi_seg_6class.py 中生成 roi_masks 的那一行代码粘到这里"
    )

    # -------------------------------------------------
    # C. 导出 ROI 图像（给 P11 / P13 用）
    # -------------------------------------------------
    roi_root = Path("outputs/roi_split_v1")
    export_roi_images_min(
        img_bgr=img_bgr,
        roi_masks=roi_masks,
        sample_id=sample_id,
        out_root=roi_root,
    )

    # -------------------------------------------------
    # D. 加载 P14 + PCA
    #    👉 请从 probe_p14_embedding.py 中复制“加载模型 + PCA”的代码
    # -------------------------------------------------
    # 示例（⚠️你要替换）：
    # p14_model = ...
    # pca = ...
    raise RuntimeError(
        "请从 probe_p14_embedding.py 中复制加载 p14_model 和 pca 的代码到这里"
    )

    # -------------------------------------------------
    # E. 调用 roi_feature_bridge
    # -------------------------------------------------
    tables = extract_roi_features_all(
        img_bgr=img_bgr,
        roi_masks=roi_masks,
        sample_id=sample_id,
        roi_root=roi_root,
        device=device,

        infer_regression_fn=infer_regression,

        p11_ckpt="checkpoints/p11/p11_color_best.pth",
        p11_norm="checkpoints/p11/p11_norm.npz",
        p11_dim=None,

        p13_ckpt="checkpoints/p13/p13_texture_best.pth",
        p13_norm="checkpoints/p13/p13_norm.npz",
        p13_dim=None,

        p14_model=p14_model,
        pca=pca,
    )

    # -------------------------------------------------
    # F. 打印 + 保存结果
    # -------------------------------------------------
    print("\n=== ROI tables generated ===")
    for k, v in tables.items():
        print(f"{k}: {len(v)} features")

    out_path = Path("outputs/roi_features_min.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
