import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from src.models.unet import UNet  # 你项目里实际的 UNet 路径/类名按你的改
# 如果你模型定义在别处，把这一行改成你自己的 import

# -------------------------
# Config导出“原图尺度”分割 mask
# -------------------------
SPLIT = "test"  # train / val / test
SPLIT_FILE = f"../data/splits/{SPLIT}.txt"
IMG_DIR = "../data/images/tongue"

CKPT_PATH = "../checkpoints/seg/unet_tongue_best.pth"
OUT_DIR = f"../outputs/pred_masks_original/{SPLIT}"

INPUT_SIZE = 256      # 训练时的输入尺寸
THR = 0.5             # 二值化阈值


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def preprocess(img_bgr):
    # BGR -> RGB, resize -> [0,1] -> CHW tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = img_rs.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x


def main():
    ensure_dir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load model ----
    model = UNet().to(device)  # 按你实际 UNet 构造参数改
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # 兼容你保存的是 state_dict 或包含 key 的 dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.eval()
    print("Loaded ckpt:", CKPT_PATH)

    ids = load_ids(SPLIT_FILE)
    print("Num ids:", len(ids))

    for sid in tqdm(ids, desc=f"Infer original-size ({SPLIT})"):
        img_path = os.path.join(IMG_DIR, sid + ".jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        x = preprocess(img_bgr).to(device)

        with torch.no_grad():
            logits = model(x)              # [1,1,256,256]  (假设)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [256,256]

        # resize prob -> original size
        prob_ori = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_ori = (prob_ori >= THR).astype(np.uint8) * 255

        out_path = os.path.join(OUT_DIR, sid + ".png")
        cv2.imwrite(out_path, mask_ori)

    print("Done. Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
