# 把 P11_Tg_Color.txt 转成 CSV（统一读取）
import os
import pandas as pd

SRC = r"D:\PROJECTS\TONGUE_EXPERT_V1\TONGUEXPERTDATABASE\Phenotypes\P11_Tg_Color.txt"
DST = "../data/labels/p11_tg_color.csv"

os.makedirs("data/labels", exist_ok=True)

# 以 tab 分隔读取
df = pd.read_csv(SRC, sep="\t")
# 第一列应该是 SID
df.rename(columns={df.columns[0]: "SID"}, inplace=True)

df.to_csv(DST, index=False)
print("Saved:", DST)
print("Rows:", len(df), "Cols:", len(df.columns))
