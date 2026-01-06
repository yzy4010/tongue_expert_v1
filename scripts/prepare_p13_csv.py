import os
import pandas as pd

SRC = r"D:\projects\tongue_expert_v1\TonguExpertDatabase\Phenotypes\P13_Tg_Texture.txt"
DST = "../data/labels/p13_tg_texture.csv"

os.makedirs("data/labels", exist_ok=True)

df = pd.read_csv(SRC, sep="\t")
df.rename(columns={df.columns[0]: "SID"}, inplace=True)

df.to_csv(DST, index=False)
print("Saved:", DST)
print("Rows:", len(df), "Cols:", len(df.columns))
print("Columns preview:", df.columns[:20].tolist())
