import os
import pandas as pd

# 解析 P12_Tg_Shape.txt → CSV

SRC = r"D:\PROJECTS\TONGUE_EXPERT_V1\TONGUEXPERTDATABASE\Phenotypes\P12_Tg_Shape.txt"
DST = "../data/labels/p12_tg_shape.csv"

os.makedirs("data/labels", exist_ok=True)

df = pd.read_csv(SRC, sep="\t")
df.rename(columns={df.columns[0]: "SID"}, inplace=True)

df.to_csv(DST, index=False)
print("Saved:", DST)
print("Rows:", len(df), "Cols:", len(df.columns))
print("Columns:", df.columns.tolist())
