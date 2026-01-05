import os
import random

#
# 一键对齐检查脚本
#

SPLITS = ["train", "val", "test"]

SPLIT_DIR = "../data/splits"
PRED_DIR = "../outputs/pred_masks"
ROI_DIR = "../outputs/roi"

def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def exists_png(dir_path, sid):
    return os.path.exists(os.path.join(dir_path, sid + ".png"))

def exists_jpg(dir_path, sid):
    return os.path.exists(os.path.join(dir_path, sid + ".jpg"))

def main():
    random.seed(42)

    for split in SPLITS:
        split_file = os.path.join(SPLIT_DIR, f"{split}.txt")
        ids = load_ids(split_file)

        pred_dir = os.path.join(PRED_DIR, split)
        roi_dir = os.path.join(ROI_DIR, split)

        print(f"\n===== {split.upper()} =====")
        print("IDs in split:", len(ids))
        print("Pred dir:", pred_dir)
        print("ROI dir:", roi_dir)

        missing_pred = []
        missing_roi = []

        for sid in ids:
            if not exists_png(pred_dir, sid):
                missing_pred.append(sid)
            if not exists_jpg(roi_dir, sid):
                missing_roi.append(sid)

        print("Missing pred_masks:", len(missing_pred))
        print("Missing roi:", len(missing_roi))

        if len(missing_pred) > 0:
            print("  Example missing pred:", missing_pred[:10])

        if len(missing_roi) > 0:
            print("  Example missing roi:", missing_roi[:10])

        # quick sample check
        sample = random.sample(ids, k=min(5, len(ids)))
        print("Sample IDs:", sample)

        ok = True
        for sid in sample:
            if not exists_png(pred_dir, sid) or not exists_jpg(roi_dir, sid):
                ok = False
                break
        print("Sample check:", "OK ✅" if ok else "NOT OK ❌")

    print("\nAll checks done.")

if __name__ == "__main__":
    main()
