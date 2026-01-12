# -*- coding: utf-8 -*-
import cv2
from src.pipeline.run_infer import build_bundle, infer_one_image

def main():
    bundle = build_bundle(device=None)  # 自动 cpu/cuda
    img = cv2.imread("../data/images/tongue/TE0000161.jpg")
    out = infer_one_image(bundle, img, sample_id="TE0000161")
    print(out["meta"])
    # 看看关键字段是否存在
    print("outputs keys:", list(out["outputs"].keys())[:10])
    roi_tables = out["outputs"]["roi_tables"]
    demo = out["outputs"]["demo"]

    print("demo keys:", list(demo.keys()))
    print("P21_Tai_Color len =", len(roi_tables.get("P21_Tai_Color", {})))
    print("P23_Tai_Texture len =", len(roi_tables.get("P23_Tai_Texture", {})))
    print("P24_Tai_CNN len =", len(roi_tables.get("P24_Tai_CNN", {})))
    print("P54_Toothmark_CNN len =", len(roi_tables.get("P54_Toothmark_CNN", {})))


if __name__ == "__main__":
    main()
