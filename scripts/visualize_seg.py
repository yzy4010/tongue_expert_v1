import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.unet import UNet

# -------------------------
# Config可视化分割结果
# -------------------------
IMG_DIR = "../data/images/tongue"          # .jpg
MASK_DIR = "../data/masks/tongue"          # .png
SPLIT_FILE = "../data/splits/val.txt"      # 可改为 test.txt
MODEL_PATH = "unet_tongue_best.pth"     # 你的 best 模型
OUT_DIR = "outputs/vis_seg"

IMG_SIZE = 256     # 必须与你训练时 Dataset resize 一致
NUM_SAMPLES = 8    # 随机可视化多少张
THRESH = 0.5       # 二值化阈值


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_ids(split_file: str):
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def preprocess_image(img_rgb: np.ndarray, img_size: int):
    # img_rgb: HWC, uint8, RGB
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return img, t


def preprocess_mask(mask_gray: np.ndarray, img_size: int):
    # mask_gray: HW, uint8
    mask = cv2.resize(mask_gray, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return mask


def make_overlay(img_rgb: np.ndarray, pred_bin: np.ndarray):
    """
    img_rgb: HWC uint8 (RGB)
    pred_bin: HW uint8 {0,1}
    """
    overlay = img_rgb.copy()
    # 红色覆盖预测区域
    overlay[pred_bin == 1] = (0.6 * overlay[pred_bin == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    return overlay


def main():
    ensure_dir(OUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = UNet().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded model:", MODEL_PATH)

    ids = load_ids(SPLIT_FILE)
    if len(ids) == 0:
        raise RuntimeError(f"Split file empty: {SPLIT_FILE}")

    random.seed(42)
    sample_ids = random.sample(ids, k=min(NUM_SAMPLES, len(ids)))
    print(f"Visualizing {len(sample_ids)} samples from: {SPLIT_FILE}")

    for sid in sample_ids:
        img_path = os.path.join(IMG_DIR, sid + ".jpg")
        mask_path = os.path.join(MASK_DIR, sid + ".png")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[Skip] Cannot read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"[Skip] Cannot read mask: {mask_path}")
            continue

        img_resized_rgb, img_t = preprocess_image(img_rgb, IMG_SIZE)
        gt_mask = preprocess_mask(mask_gray, IMG_SIZE)

        img_t = img_t.to(device)

        with torch.no_grad():
            logits = model(img_t)                         # [1,1,H,W]
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_bin = (prob >= THRESH).astype(np.uint8)
        overlay = make_overlay((img_resized_rgb * 255).astype(np.uint8), pred_bin)

        # --- plot ---
        fig = plt.figure(figsize=(18, 4))
        fig.suptitle(f"{sid} | thr={THRESH}", fontsize=12)

        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(img_resized_rgb)
        ax1.set_title("Image")
        ax1.axis("off")

        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(gt_mask, cmap="gray")
        ax2.set_title("GT Mask")
        ax2.axis("off")

        ax3 = plt.subplot(1, 5, 3)
        ax3.imshow(prob, cmap="gray")
        ax3.set_title("Pred Prob")
        ax3.axis("off")

        ax4 = plt.subplot(1, 5, 4)
        ax4.imshow(pred_bin, cmap="gray")
        ax4.set_title("Pred Bin")
        ax4.axis("off")

        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(overlay)
        ax5.set_title("Overlay (Pred)")
        ax5.axis("off")

        plt.tight_layout()

        # save
        out_path = os.path.join(OUT_DIR, f"{sid}.png")
        plt.savefig(out_path, dpi=160)
        plt.show()
        plt.close(fig)

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
