# 先转成干净 CSV
import pandas as pd

SRC = r"D:\PROJECTS\TONGUE_EXPERT_V1\TONGUEXPERTDATABASE\Phenotypes\P11_Tg_Color.txt"
# DST = "data/labels/tg_color.csv"
DST = "D:/projects/tongue_expert_v1/data/labels/tg_color.csv"

df = pd.read_csv(SRC, sep="\t")

# 确保 SID 是字符串
df["SID"] = df["SID"].astype(str)

df.to_csv(DST, index=False)
print(f"Saved: {DST}, shape={df.shape}")
