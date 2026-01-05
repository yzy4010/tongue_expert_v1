import os
import cv2
import numpy as np


# -------------------------
# test 集 Dice / IoU 汇总
# -------------------------
GT_MASK_DIR = "../data/masks/tongue"              # GT mask: .png
PRED_MASK_DIR = "../outputs/pred_masks/test"     # Pred mask: .png
SPLIT_FILE = "../data/splits/test.txt"

IMG_SIZE = 256   # 必须与你 pred 输出一致（你 export_seg_infer 用的 256）


def load_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_mask(path: str, img_size: int):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8)  # 0/1
    return m


def dice_iou(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-7):
    # pred01, gt01: uint8 0/1
    inter = np.logical_and(pred01 == 1, gt01 == 1).sum()
    pred_sum = (pred01 == 1).sum()
    gt_sum = (gt01 == 1).sum()
    union = pred_sum + gt_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def summarize(values: np.ndarray, name: str):
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    print(f"\n{name} summary:")
    print(f"  mean = {values.mean():.4f}, std = {values.std():.4f}")
    print(f"  p10  = {p10:.4f}, p25 = {p25:.4f}, p50 = {p50:.4f}, p75 = {p75:.4f}, p90 = {p90:.4f}")
    print(f"  min  = {values.min():.4f}, max = {values.max():.4f}")


def main():
    ids = load_ids(SPLIT_FILE)
    print("Num test ids:", len(ids))

    dice_list = []
    iou_list = []
    missing = 0

    for sid in ids:
        gt_path = os.path.join(GT_MASK_DIR, sid + ".png")
        pred_path = os.path.join(PRED_MASK_DIR, sid + ".png")

        gt = read_mask(gt_path, IMG_SIZE)
        pred = read_mask(pred_path, IMG_SIZE)

        if gt is None or pred is None:
            missing += 1
            continue

        d, j = dice_iou(pred, gt)
        dice_list.append(d)
        iou_list.append(j)

    dice_arr = np.array(dice_list, dtype=np.float32)
    iou_arr = np.array(iou_list, dtype=np.float32)

    print(f"Evaluated: {len(dice_arr)} samples, missing: {missing}")

    summarize(dice_arr, "Dice")
    summarize(iou_arr, "IoU")

    # Top worst cases（方便你之后看难例）
    if len(dice_arr) > 0:
        worst_k = 10
        # 重新算一遍带 sid（避免存全量 dict 也行，这里简单写）
        sid_scores = []
        for sid in ids:
            gt_path = os.path.join(GT_MASK_DIR, sid + ".png")
            pred_path = os.path.join(PRED_MASK_DIR, sid + ".png")
            gt = read_mask(gt_path, IMG_SIZE)
            pred = read_mask(pred_path, IMG_SIZE)
            if gt is None or pred is None:
                continue
            d, j = dice_iou(pred, gt)
            sid_scores.append((sid, d, j))

        sid_scores.sort(key=lambda x: x[1])  # 按 dice 升序
        print("\nWorst cases (lowest Dice):")
        for sid, d, j in sid_scores[:worst_k]:
            print(f"  {sid}  Dice={d:.4f}  IoU={j:.4f}")


if __name__ == "__main__":
    main()

# 运行结果
# Num test ids: 899
# Evaluated: 899 samples, missing: 0
#
# Dice summary:
#   mean = 0.9825, std = 0.0143
#   p10  = 0.9742, p25 = 0.9807, p50 = 0.9858, p75 = 0.9887, p90 = 0.9900
#   min  = 0.7828, max = 0.9921
#
# IoU summary:
#   mean = 0.9659, std = 0.0251
#   p10  = 0.9498, p25 = 0.9622, p50 = 0.9720, p75 = 0.9777, p90 = 0.9802
#   min  = 0.6432, max = 0.9844
#
# Worst cases (lowest Dice):
#   TE0003510  Dice=0.7828  IoU=0.6432
#   TE0003243  Dice=0.8014  IoU=0.6686
#   TE0002167  Dice=0.8588  IoU=0.7525
#   TE0002411  Dice=0.8651  IoU=0.7623
#   TE0000500  Dice=0.8919  IoU=0.8049
#   TE0002158  Dice=0.9000  IoU=0.8182
#   TE0004393  Dice=0.9116  IoU=0.8375
#   TE0004669  Dice=0.9364  IoU=0.8804
#   TE0002172  Dice=0.9384  IoU=0.8840
#   TE0005364  Dice=0.9395  IoU=0.8859
