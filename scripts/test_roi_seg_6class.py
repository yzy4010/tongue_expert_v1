import cv2
import numpy as np
import torch

from src.pipeline.stages.p20_roi_seg_6class import (
    RoiSeg6ClassInfer,
    RoiSegConfig,
)

# ===== 1. 加载测试图片 =====
IMG_PATH = "../data/images/tongue/TE0000004.jpg"   # 换成你自己的图片
img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None, "Failed to load image"

# ===== 2. 先跑舌体分割（已有模型）=====
from src.models.unet import UNet

tongue_model = UNet()
tongue_model.load_state_dict(
    torch.load("../checkpoints/seg/unet_tongue_best.pth", map_location="cpu")
)
tongue_model.eval()

with torch.no_grad():
    x = cv2.resize(img_bgr, (512, 512))
    x = x.astype("float32") / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    logits = tongue_model(x)
    prob = torch.sigmoid(logits)[0, 0].numpy()

tongue_mask = (prob > 0.5).astype(np.uint8)
tongue_mask = cv2.resize(
    tongue_mask,
    (img_bgr.shape[1], img_bgr.shape[0]),
    interpolation=cv2.INTER_NEAREST,
)

# ===== 3. ROI 6 类分割 =====
roi_cfg = RoiSegConfig(
    ckpt_path="../checkpoints/roi_seg/roi_seg_6class_v1_best.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_size=(512, 512),
    bgr_to_rgb=True,          # ⚠️ 如果训练用的是 BGR，改成 False
    fill_zhi_from_tongue=True,
)

roi_infer = RoiSeg6ClassInfer(cfg=roi_cfg)
roi_masks = roi_infer.infer_roi_masks(img_bgr, tongue_mask)

# ===== 4. 保存可视化结果 =====
out = img_bgr.copy()

colors = {
    "tai": (255, 0, 0),
    "zhi": (0, 255, 0),
    "fissure": (0, 0, 255),
    "tooth_mk": (255, 255, 0),
}

for name, color in colors.items():
    mask = roi_masks[name]
    out[mask == 1] = (
        0.5 * out[mask == 1] + 0.5 * np.array(color)
    ).astype(np.uint8)

cv2.imwrite("debug_roi_overlay.png", out)
print("Saved debug_roi_overlay.png")
