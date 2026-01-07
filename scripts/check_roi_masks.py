from pathlib import Path
import cv2
import numpy as np

MSK_DIR = Path("../data/roi_seg_v1/masks")
assert MSK_DIR.exists(), f"Mask dir not found: {MSK_DIR}"

bad = []
all_vals = set()

for p in sorted(MSK_DIR.glob("*.png")):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        print("Failed to read:", p)
        continue
    vals = set(np.unique(m).tolist())
    all_vals |= vals
    if any(v < 0 or v > 5 for v in vals):
        bad.append((p.name, sorted(list(vals))[:50]))

print("Global unique values (sample):", sorted(list(all_vals))[:80])
print("Bad masks count:", len(bad))

for name, vals in bad[:30]:
    print("BAD:", name, "values:", vals)
