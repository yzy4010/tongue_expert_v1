from pathlib import Path
import cv2
import numpy as np

SRC = Path("../data/roi_seg_v1/masks")
DST = Path("../data/roi_seg_v1/masks_6class")
DST.mkdir(parents=True, exist_ok=True)

valid = set(range(6))  # 0..5

bad_files = []
for p in sorted(SRC.glob("*.png")):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {p}")

    vals = set(np.unique(m).tolist())
    is_bad = any(v not in valid for v in vals)

    out = m.copy().astype(np.uint8)
    # 把所有非法值 -> 0 (bg)
    mask_invalid = ~np.isin(out, list(valid))
    if mask_invalid.any():
        out[mask_invalid] = 0

    cv2.imwrite(str(DST / p.name), out)

    if is_bad:
        bad_files.append((p.name, sorted(list(vals))[:80]))

print("Done. Clean masks written to:", DST)
print("Bad files cleaned:", len(bad_files))
for name, vals in bad_files[:20]:
    print("CLEANED:", name, "orig values sample:", vals)
