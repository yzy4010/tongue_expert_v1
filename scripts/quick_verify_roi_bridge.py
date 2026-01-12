# -*- coding: utf-8 -*-
"""
Quick verify ROI feature bridge (REALISTIC VERSION)

- 不依赖 TongueSegInfer
- 舌体分割使用你项目中“已经存在的推理代码”
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import json
from src.pipeline.stages.p20_roi_seg_6class import (
    RoiSeg6ClassInfer,
    RoiSegConfig,
)


# -------------------------------------------------
# 1. ROI feature bridge（你已经完成）
# -------------------------------------------------
from src.pipeline.roi_feature_bridge import extract_roi_features_all

# -------------------------------------------------
# 2. infer_regression（在 eval_all_test.py 中定义）
# -------------------------------------------------
from scripts.eval_all_test import infer_regression

# -------------------------------------------------
# 3. ROI 6-class 分割（你项目真实存在）
# -------------------------------------------------
from src.pipeline.stages.p20_roi_seg_6class import RoiSeg6ClassInfer

# -------------------------------------------------
# 4. P14 模型
# -------------------------------------------------
from scripts.infer_p14_embedding import P14MultiTaskNet

# -------------------------------------------------
# 5. PCA
# -------------------------------------------------
import joblib

from src.pipeline.rules.roi_rules_tai_zhi import infer_tai_zhi_from_tongue_mask


def infer_tongue_mask(img_bgr: np.ndarray, device: str) -> np.ndarray:
    """
    舌体分割推理（复用已有 UNet 舌体模型）
    输入:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        device: "cpu" or "cuda"
    输出:
        tongue_mask: HxW, uint8, {0,1}
    """
    from src.models.unet import UNet
    import cv2
    import numpy as np
    import torch

    # 1. 加载模型（与你 test 脚本完全一致）
    tongue_model = UNet().to(device)
    tongue_model.load_state_dict(
        torch.load(
            "../checkpoints/seg/unet_tongue_best.pth",  # ⚠️ 确认路径
            map_location=device,
        )
    )
    tongue_model.eval()

    # 2. 预处理（resize + 归一化）
    with torch.no_grad():
        x = cv2.resize(img_bgr, (512, 512))
        x = x.astype("float32") / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

        logits = tongue_model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # 3. 后处理（阈值 + resize 回原尺寸）
    tongue_mask = (prob > 0.5).astype(np.uint8)
    tongue_mask = cv2.resize(
        tongue_mask,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return tongue_mask




def load_p14_model_and_pca(device: str):

    p14_ckpt = "../checkpoints/p14/p14_multitask_best.pth"  # 确认路径
    pca_path = "../checkpoints/p14/p14_pca.pkl"

    emb_dim = 128
    p11_dim = 76
    p13_dim = 16
    dropout = 0.0

    model = P14MultiTaskNet(
        emb_dim=emb_dim,
        p11_dim=p11_dim,
        p13_dim=p13_dim,
        dropout=dropout,
    ).to(device)

    ckpt = torch.load(p14_ckpt, map_location=device)

    assert "state_dict" in ckpt, "Invalid P14 checkpoint format"
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    pca = joblib.load(pca_path)

    return model, pca



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------------------------------
    # A) 输入图像
    # -------------------------------------------------
    sample_id = "TE0000004"
    img_path = Path("../data/images/tongue") / f"{sample_id}.jpg"
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    print("Image:", img_path, img_bgr.shape)

    # -------------------------------------------------
    # B) 舌体分割（你已有）
    # -------------------------------------------------
    tongue_mask = infer_tongue_mask(img_bgr, device)
    print("tongue_mask area:", int((tongue_mask > 0).sum()))

    # -------------------------------------------------
    # C) ROI 分割（6 类）
    # -------------------------------------------------
    roi_cfg = RoiSegConfig(
        ckpt_path="../checkpoints/roi_seg/roi_seg_6class_v1_best.pth",  # 改成你真实路径
        device=device,
        input_size=(512, 512),
        bgr_to_rgb=True,  # ⚠️ 如果 ROI 训练用的是 BGR，改成 False
        fill_zhi_from_tongue=True,
    )

    roi_infer = RoiSeg6ClassInfer(cfg=roi_cfg)
    roi_masks = roi_infer.infer_roi_masks(img_bgr, tongue_mask)

    print("roi_masks keys:", list(roi_masks.keys()))
    for k, v in roi_masks.items():
        if v is not None:
            print(f"  {k:8s} area={int((v > 0).sum())}")

    # -------------------------------------------------
    # C.1) 规则法补齐 Tai / Zhi（关键兜底）
    # -------------------------------------------------
    tai_rule, zhi_rule = infer_tai_zhi_from_tongue_mask(
        tongue_mask,
        tai_ratio=0.35,  # 0.25~0.45 都可以，先用 0.35
    )

    # 如果 ROI 模型没预测出 tai / zhi，则用规则结果兜底
    if int((roi_masks["tai"] > 0).sum()) == 0:
        roi_masks["tai"] = tai_rule
        print("[Rule] fallback tai:", int(tai_rule.sum()))

    if int((roi_masks["zhi"] > 0).sum()) == 0:
        roi_masks["zhi"] = zhi_rule
        print("[Rule] fallback zhi:", int(zhi_rule.sum()))

    # -------------------------------------------------
    # D) P14 + PCA
    # -------------------------------------------------
    p14_model, pca = load_p14_model_and_pca(device)

    # -------------------------------------------------
    # E) ROI feature bridge
    # -------------------------------------------------
    roi_root = Path("../outputs/roi_split_v1")
    roi_root.mkdir(parents=True, exist_ok=True)

    tables = extract_roi_features_all(
        img_bgr=img_bgr,
        roi_masks=roi_masks,
        sample_id=sample_id,
        roi_root=roi_root,
        device=device,

        infer_regression_fn=infer_regression,

        p11_ckpt="../checkpoints/p11/p11_color_best.pth",
        p11_norm="../checkpoints/p11/p11_norm.npz",
        p11_dim=None,

        p13_ckpt="../checkpoints/p13/p13_texture_best.pth",
        p13_norm="../checkpoints/p13/p13_norm.npz",
        p13_dim=None,

        p14_model=p14_model,
        pca=pca,
    )

    # -------------------------------------------------
    # F) 最小验证点
    # -------------------------------------------------
    print("\n===== VERIFY =====")
    print("P21_Tai_Color keys:", list(tables["P21_Tai_Color"].keys())[:10])
    print("P23_Tai_Texture keys:", list(tables["P23_Tai_Texture"].keys())[:10])

    pcs = [tables["P24_Tai_CNN"].get(f"tai_cnnPC{i}", np.nan) for i in range(1, 11)]
    print("P24_Tai_CNN:", pcs, "nan_count=", int(np.isnan(np.array(pcs)).sum()))

    out_path = Path("outputs/roi_features_quick_verify.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)
    print("OK if P21/P23 not empty and nan_count=0")


if __name__ == "__main__":
    main()
