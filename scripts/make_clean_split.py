import os
import random

IMG_DIR = "../data/images/tongue"
MASK_DIR = "../data/masks/tongue"

ids = []
for name in os.listdir(IMG_DIR):
    if name.endswith(".jpg"):
        sid = name.replace(".jpg", "")
        if os.path.exists(os.path.join(MASK_DIR, sid + ".png")):
            ids.append(sid)

print("Total valid samples:", len(ids))

random.seed(42)   # 重要：保证可复现
random.shuffle(ids)

n = len(ids)
n_train = int(0.7 * n)
n_val = int(0.85 * n)

train_ids = ids[:n_train]
val_ids = ids[n_train:n_val]
test_ids = ids[n_val:]

os.makedirs("data/splits", exist_ok=True)

def save(path, data):
    with open(path, "w") as f:
        f.write("\n".join(data))

save("data/splits/train.txt", train_ids)
save("data/splits/val.txt", val_ids)
save("data/splits/test.txt", test_ids)

print(f"Train: {len(train_ids)}")
print(f"Val:   {len(val_ids)}")
print(f"Test:  {len(test_ids)}")
