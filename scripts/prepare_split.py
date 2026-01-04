# 生成 split（训练 / 验证 / 测试）
import os
import random

IMG_DIR = "../data/images/tongue"
OUT_DIR = "../data/splits"


files = sorted([f.replace(".jpg", "") for f in os.listdir(IMG_DIR)])
random.seed(42)
random.shuffle(files)

n = len(files)
train = files[:int(0.7 * n)]
val   = files[int(0.7 * n):int(0.85 * n)]
test  = files[int(0.85 * n):]

def save(name, data):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for x in data:
            f.write(x + "\n")

save("train.txt", train)
save("val.txt", val)
save("test.txt", test)

print("Split prepared.")
