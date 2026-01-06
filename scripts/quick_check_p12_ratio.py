import pandas as pd
import numpy as np

TEST_CSV = "../outputs/p12_eval_original_test.csv"  # 注意：就是 eval 脚本生成的那个，不是 calibrated 文件

def r2(y, p):
    ss_res = np.sum((y-p)**2)
    ss_tot = np.sum((y-y.mean())**2)
    return 1 - ss_res/(ss_tot + 1e-12)

df = pd.read_csv(TEST_CSV)

y = df["gt_tg_w_div_h"].to_numpy()
p = df["pred_tg_w_div_h"].to_numpy()

print("GT  mean/min/max:", y.mean(), y.min(), y.max())
print("Pred mean/min/max:", p.mean(), p.min(), p.max())

mae = np.mean(np.abs(y-p))
rmse = np.sqrt(np.mean((y-p)**2))
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2(y,p))
print("R:", np.corrcoef(y,p)[0,1])
