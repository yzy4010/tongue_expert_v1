import os
import cv2
import numpy as np
from tqdm import tqdm

# -------------------------
# Config (val)新建 ROI 批量导出脚本
# -------------------------
IMG_DIR = "../data/images/tongue"                 # 原图 .jpg
MASK_DIR = "../outputs/pred_masks/val"           # 预测 mask .png
SPLIT_FILE = "../data/splits/val.txt"            # test ids
OUT_DIR = "../outputs/roi/val"                   # 输出 ROI

ROI_SIZE = 224                                 # 统一尺寸（后续分类通用）
EXPAND_RATIO = 0.08                            # bbox 外扩比例（8%）
MIN_AREA_RATIO = 0.001                         # 太小的 mask 直接跳过（防噪声）


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def largest_connected_component(mask01: np.ndarray):
    """
    保留最大连通域，避免偶尔的噪声点影响 bbox
    mask01: uint8 0/1
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return mask01
    # stats: [label, x, y, w, h, area] for each component, label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_id = 1 + int(np.argmax(areas))
    return (labels == max_id).astype(np.uint8)


def bbox_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 == 1)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, h, w, ratio):
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * ratio)
    dy = int(bh * ratio)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)
    return x1, y1, x2, y2


def main():
    ensure_dir(OUT_DIR)

    ids = load_ids(SPLIT_FILE)
    print("Num ids:", len(ids))
    print("IMG_DIR:", IMG_DIR)
    print("MASK_DIR:", MASK_DIR)
    print("OUT_DIR:", OUT_DIR)

    saved, skipped = 0, 0

    for sid in tqdm(ids, desc="Export ROI"):
        img_path = os.path.join(IMG_DIR, sid + ".jpg")
        mask_path = os.path.join(MASK_DIR, sid + ".png")

        img_bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # mask resize 回原图尺寸
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask01 = (mask > 0).astype(np.uint8)

        # 过滤异常：mask 太小
        if mask01.sum() < (h * w * MIN_AREA_RATIO):
            skipped += 1
            continue

        # 只保留最大连通域（更稳）
        mask01 = largest_connected_component(mask01)

        bbox = bbox_from_mask(mask01)
        if bbox is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = expand_bbox(*bbox, h, w, EXPAND_RATIO)

        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            skipped += 1
            continue

        roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)

        out_path = os.path.join(OUT_DIR, sid + ".jpg")
        # cv2.imwrite 需要 BGR
        cv2.imwrite(out_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"Done. Saved: {saved}, Skipped: {skipped}")
    print("ROI saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
