import json
import numpy as np
import pandas as pd
import os

CSV = "../data/labels/p11_tg_color.csv"
OUT = "../checkpoints/p11/p11_color_norm.json"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(CSV)
features = [c for c in df.columns if c != "SID"]

Y = df[features].to_numpy(dtype=np.float32)
mean = Y.mean(axis=0)
std = Y.std(axis=0)
std[std < 1e-6] = 1.0  # 防止除 0

norm = {
    "features": features,
    "mean": mean.tolist(),
    "std": std.tolist()
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(norm, f, indent=2)

print("Saved:", OUT)
print("Num features:", len(features))
