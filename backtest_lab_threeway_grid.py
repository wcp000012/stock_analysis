import glob, os, itertools
import numpy as np
import pandas as pd

os.chdir(r"C:\Users\User\stock_analysis")
TOP_K = 20
ANCHORS = ["2025-06-02", "2025-09-01", "2025-10-15", "2025-12-01", "2026-01-15",
           "2026-03-02", "2026-04-15", "2026-06-01", "2026-07-09"]
STEP = 0.1

data = {}
for a in ANCHORS:
    D = pd.read_csv(f"lab_scores_{a}.csv", index_col=0)
    D = D.dropna(subset=["pred_max_raw", "foreign", "mom60"])
    D["z_raw"] = D["pred_max_raw"].rank(pct=True)
    D["z_for"] = D["foreign"].rank(pct=True)
    D["z_mom"] = D["mom60"].rank(pct=True)
    data[a] = D

# 權重網格：w_raw + w_for + w_mom = 1，間隔 0.1
grid = []
n = int(round(1 / STEP))
for i in range(n + 1):
    for j in range(n + 1 - i):
        k = n - i - j
        grid.append((i * STEP, j * STEP, k * STEP))

rows = []
for w_raw, w_for, w_mom in grid:
    excs = []
    hits = []
    for a, D in data.items():
        score = w_raw * D["z_raw"] + w_for * D["z_for"] + w_mom * D["z_mom"]
        top = D.loc[score.nlargest(TOP_K).index]
        excs.append(top["ret_end"].mean() - D["ret_end"].mean())
        hits.append((top["ret_end"] > 0).mean())
    excs = np.array(excs)
    rows.append({
        "w_raw": w_raw, "w_for": w_for, "w_mom": w_mom,
        "mean_excess": excs.mean(), "std_excess": excs.std(),
        "min_excess": excs.min(), "frac_positive": (excs > 0).mean(),
        "mean_hit": np.mean(hits),
    })

res = pd.DataFrame(rows)

# 先篩「9 個錨點全部正超額」的組合，再依平均超額排序
all_pos = res[res["frac_positive"] == 1.0].sort_values("mean_excess", ascending=False)
print(f"9 個錨點全部正超額的權重組合數: {len(all_pos)} / {len(res)}")
print("\nTop 10（全部正超額裡平均超額最高）：")
print(all_pos.head(10).round(3).to_string(index=False))

print("\n參考：平均超額最高的 10 個組合（不管是否全正）：")
print(res.sort_values("mean_excess", ascending=False).head(10).round(3).to_string(index=False))

print("\n參考：純模型 (1,0,0) 與之前的二訊號最佳 (0.7,0.3,0) / (0.8,0.2,0)：")
ref = res[((res.w_raw == 1.0) & (res.w_for == 0.0) & (res.w_mom == 0.0)) |
          ((res.w_raw.round(1) == 0.7) & (res.w_for.round(1) == 0.3) & (res.w_mom == 0.0)) |
          ((res.w_raw.round(1) == 0.8) & (res.w_for.round(1) == 0.2) & (res.w_mom == 0.0))]
print(ref.round(3).to_string(index=False))

res.to_csv("three_way_blend_grid.csv", index=False, encoding="utf-8-sig")
