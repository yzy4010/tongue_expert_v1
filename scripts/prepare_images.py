import os
import shutil

SRC_RAW = r"D:\PROJECTS\TONGUE_EXPERT_V1\TONGUEXPERTDATABASE\TongueImage\Raw"
SRC_MASK = r"D:\PROJECTS\TONGUE_EXPERT_V1\TONGUEXPERTDATABASE\TongueImage\Mask"

DST_IMG = "../data/images/tongue"
DST_MASK = "../data/masks/tongue"

# 确保目标目录存在
os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_MASK, exist_ok=True)

# 拷贝 Raw 图像文件
for name in os.listdir(SRC_RAW):
    if name.lower().endswith(".jpg"):
        shutil.copy(
            os.path.join(SRC_RAW, name),
            os.path.join(DST_IMG, name)
        )

# 拷贝 Mask 文件并将其转换为 png 格式
for name in os.listdir(SRC_MASK):
    if name.lower().endswith(".jpg"):
        src = os.path.join(SRC_MASK, name)
        dst = os.path.join(DST_MASK, name.replace(".jpg", ".png"))
        shutil.copy(src, dst)

print("Images & masks prepared.")
