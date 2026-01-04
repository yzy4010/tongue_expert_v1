# 目录创建脚本
import os

BASE_DIR = "data"

DIRS = [
    "images/tongue",
    "masks/tongue",
    "labels",
    "splits",
]

for d in DIRS:
    path = os.path.join(BASE_DIR, d)
    os.makedirs(path, exist_ok=True)
    print(f"created: {path}")

# python scripts/prepare_dirs.py