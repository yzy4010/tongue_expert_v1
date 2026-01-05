import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.unet import UNet

# -------------------------
# 分割推理专用脚本
# 1、用冻结模型对 val/test 的 ID 推理
# 2、输出预测 mask 到 outputs/pred_masks/（PNG），方便后续 ROI 裁剪/表型提取
# -------------------------
IMG_DIR = "../data/images/tongue"                 # .jpg
SPLIT_FILE = "../data/splits/test.txt"            # 你也可以改 val.txt
CKPT_PATH = "../checkpoints/seg/unet_tongue_best.pth"

OUT_DIR = "../outputs/pred_masks/test"                 # 输出预测 mask
IMG_SIZE = 256                                      # 必须和训练一致
THRESH = 0.5


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ensure_dir(OUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model (frozen infer)
    model = UNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print("Loaded ckpt:", CKPT_PATH)

    ids = load_ids(SPLIT_FILE)
    print("Num ids:", len(ids))

    with torch.no_grad():
        for sid in tqdm(ids, desc="Infer"):
            img_path = os.path.join(IMG_DIR, sid + ".jpg")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            x = (img_rgb.astype(np.float32) / 255.0)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(x)                 # [1,1,H,W]
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob >= THRESH).astype(np.uint8) * 255

            out_path = os.path.join(OUT_DIR, sid + ".png")
            cv2.imwrite(out_path, pred)

    print("Done. Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
