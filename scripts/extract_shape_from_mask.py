import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


MASK_DIR = "../outputs/pred_masks/test"   # 先用 test
OUT_CSV = "../outputs/p12_shape_from_mask_test.csv"

# 从 mask 计算舌体形状特征

def compute_shape_features(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None

    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / (h + 1e-6)

    rect_area = w * h
    extent = area / (rect_area + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        major_axis = max(MA, ma)
        minor_axis = min(MA, ma)
    else:
        major_axis = minor_axis = 0.0

    return [
        area, perimeter, aspect_ratio,
        extent, solidity, circularity,
        major_axis, minor_axis
    ]


def main():
    rows = []
    ids = sorted([f[:-4] for f in os.listdir(MASK_DIR) if f.endswith(".png")])

    for sid in tqdm(ids, desc="Extract shape"):
        path = os.path.join(MASK_DIR, sid + ".png")
        mask = cv2.imread(path, 0)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8) * 255

        feats = compute_shape_features(mask)
        if feats is None:
            continue

        rows.append([sid] + feats)

    cols = [
        "SID",
        "area", "perimeter", "aspect_ratio",
        "extent", "solidity", "circularity",
        "major_axis", "minor_axis"
    ]

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()

# | 特征           | 含义    |
# | ------------ | ----- |
# | area         | 舌体面积  |
# | perimeter    | 周长    |
# | aspect_ratio | 长宽比   |
# | extent       | 填充率   |
# | solidity     | 凸包紧致度 |
# | circularity  | 圆度    |
# | major_axis   | 主轴    |
# | minor_axis   | 次轴    |
