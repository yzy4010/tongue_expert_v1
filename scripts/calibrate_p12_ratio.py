import numpy as np
import pandas as pd

VAL_CSV = "../outputs/p12_eval_original_val.csv"   # 先生成 val 的对齐文件（下面会告诉你怎么生成）
TEST_CSV = "../outputs/p12_eval_original_test.csv"

# 校准 tg_w_div_h”的脚本（val 拟合 → test 评估 + 画图）

def fit_affine(x, y):
    # y ≈ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def main():
    dfv = pd.read_csv(VAL_CSV)
    dft = pd.read_csv(TEST_CSV)

    x = dfv["pred_tg_w_div_h"].to_numpy()
    y = dfv["gt_tg_w_div_h"].to_numpy()

    a, b = fit_affine(x, y)
    print("Fitted affine: gt ≈ a*pred + b")
    print("a =", a, "b =", b)

    # apply to test
    xt = dft["pred_tg_w_div_h"].to_numpy()
    yt = dft["gt_tg_w_div_h"].to_numpy()
    pt = a * xt + b

    mae = float(np.mean(np.abs(yt - pt)))
    rmse = float(np.sqrt(np.mean((yt - pt) ** 2)))
    r2v = r2(yt, pt)
    corr = float(np.corrcoef(yt, pt)[0,1])

    print("\nCalibrated tg_w_div_h on TEST:")
    print("MAE =", mae)
    print("RMSE =", rmse)
    print("R2  =", r2v)
    print("R   =", corr)

    # save calibrated
    dft["pred_tg_w_div_h_cal"] = pt
    out = "../outputs/p12_eval_original_test_calibrated.csv"
    dft.to_csv(out, index=False)
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
