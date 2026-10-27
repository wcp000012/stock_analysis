"""
簡單基準比較：PatchTST（複雜模型）vs 幾個不需要訓練的簡單規則。

用跟 backtest_riskadj.py 完全一樣的錨點、同一套評分方式（超額報酬、命中率、IC、
Top-20 波動 vs 全市場），只是把「模型預測分數排序」換成幾個簡單規則：

  mom60   : 過去 60 個交易日的報酬（純動能）排序，取最高 20 檔
  foreign : 過去 20 個交易日外資淨買超佔成交量比例排序，取最高 20 檔
  random  : 隨機挑 20 檔（固定種子，下限基準）

只需要 Close/Volume/foreign_net 三個欄位，不需要訓練，跑起來是秒級的。
跑完會自動讀 backtest_riskadj_results.csv 裡的 B_raw/C_raw（PatchTST）結果，
印出一張並排比較表。
"""
import glob
import os
import numpy as np
import pandas as pd

ANCHORS = ["2025-12-01", "2026-03-02", "2026-06-01", "2026-07-09"]
PARQUET_GLOB = "parquet_data/*.parquet"
TIME_STEP = 126       # 跟 backtest_riskadj.py 一樣的「至少要有這麼多天歷史」門檻，
                      # 確保比較的是同一個股票universe，不是因為規則不同就多篩了幾檔
EVAL_DAYS = 60
MIN_EVAL_DAYS = 40
VOL_WIN = 60
TOP_K = 20
FOREIGN_WIN = 20
MOM_WIN = 60
RANDOM_SEED = 42
RESULT_CSV = "backtest_baseline_results.csv"
COLS = ["Date", "Close", "Volume", "foreign_net"]


def load_stock(pq):
    try:
        df = pd.read_parquet(pq, columns=COLS)
    except Exception:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ["Close", "Volume", "foreign_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["foreign_net"] = df["foreign_net"].fillna(0)
    df["Volume"] = df["Volume"].fillna(0)
    return df


def build_case(df, anchor):
    hist = df[df["Date"] <= anchor]
    fut = df[df["Date"] > anchor].head(EVAL_DAYS)
    if len(hist) < TIME_STEP or len(fut) < MIN_EVAL_DAYS:
        return None
    a_close = float(hist["Close"].iloc[-1])
    if not np.isfinite(a_close) or a_close <= 0:
        return None

    logr = np.diff(np.log(hist["Close"].to_numpy(np.float64)), prepend=np.nan)
    vol = pd.Series(logr).rolling(VOL_WIN).std().iloc[-1]

    # mom60：過去 60 個交易日報酬
    if len(hist) > MOM_WIN:
        prev_close = float(hist["Close"].iloc[-1 - MOM_WIN])
        mom60 = float(np.log(a_close / prev_close)) if prev_close > 0 else np.nan
    else:
        mom60 = np.nan

    # foreign：過去 20 個交易日外資淨買超 / 同期成交量總和
    tail = hist.tail(FOREIGN_WIN)
    vol_sum = tail["Volume"].sum()
    foreign_score = float(tail["foreign_net"].sum() / vol_sum) if vol_sum > 0 else np.nan

    return {
        "ret_end": float(fut["Close"].iloc[-1] / a_close - 1),
        "ret_max": float(fut["Close"].max() / a_close - 1),
        "vol_anchor": float(vol * np.sqrt(252)) if np.isfinite(vol) else np.nan,
        "mom60": mom60,
        "foreign": foreign_score,
    }


def run_anchor(anchor_str, rng):
    anchor = pd.Timestamp(anchor_str)
    cases = {}
    for pq in glob.glob(PARQUET_GLOB):
        df = load_stock(pq)
        if df is None:
            continue
        c = build_case(df, anchor)
        if c is None:
            continue
        cases[os.path.basename(pq).replace(".parquet", "")] = c

    D = pd.DataFrame.from_dict(cases, orient="index")
    D = D.dropna(subset=["ret_end", "ret_max", "vol_anchor"])
    print(f"\n{'=' * 60}\n錨點 {anchor_str}　可評分股票數: {len(D)}\n{'=' * 60}", flush=True)

    univ_end, univ_vol = D["ret_end"].mean(), D["vol_anchor"].mean()
    D["random"] = rng.permutation(len(D))

    rows = []
    for name in ["mom60", "foreign", "random"]:
        valid = D.dropna(subset=[name])
        top = valid.nlargest(TOP_K, name)
        ic_end = valid[name].corr(valid["ret_end"], method="spearman")
        ic_max = valid[name].corr(valid["ret_max"], method="spearman")
        row = {
            "anchor": anchor_str, "variant": name, "n_universe": len(D),
            "top_ret_end": top["ret_end"].mean(), "universe_ret_end": univ_end,
            "excess_end": top["ret_end"].mean() - univ_end,
            "hit_end": (top["ret_end"] > 0).mean(),
            "top_vol": top["vol_anchor"].mean(), "universe_vol": univ_vol,
            "ic_max": ic_max, "ic_end": ic_end,
        }
        rows.append(row)
        print(f"  {name:8s}: 期末 {row['top_ret_end']:+.2%}（全市場 {univ_end:+.2%}，"
              f"超額 {row['excess_end']:+.2%}）｜命中 {row['hit_end']:.0%}｜"
              f"Top波動 {row['top_vol']:.2f}（全市場 {univ_vol:.2f}）｜IC_end {ic_end:+.3f}", flush=True)
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    for a in ANCHORS:
        all_rows += run_anchor(a, rng)
    res = pd.DataFrame(all_rows)
    res.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已存檔：{RESULT_CSV}")

    # ── 跟現有的 PatchTST 回測結果並排比較 ──────────────────────────────────
    try:
        pt = pd.read_csv("backtest_riskadj_results.csv")
        pt = pt[pt["variant"].isin(["B_raw", "C_raw"])]
        combo = pd.concat([res, pt], ignore_index=True)
        combo = combo[combo["anchor"].isin(ANCHORS)]
        print("\n" + "=" * 70)
        print("並排比較（超額報酬 excess_end，%）")
        print("=" * 70)
        piv = combo.pivot_table(index="anchor", columns="variant", values="excess_end")
        order = [c for c in ["random", "mom60", "foreign", "B_raw", "C_raw"] if c in piv.columns]
        print((piv[order] * 100).round(2).to_string())
        print("\n平均（%）：")
        print((piv[order].mean() * 100).round(2).to_string())
        print("\n⚠️ 注意：B_raw/C_raw 是 backtest_riskadj.py 之前（2026-10-13 bfill 修復前）跑的結果，"
              "跟這裡的簡單規則不是完全同一次資料快照跑出來的，但錨點、評分方式、universe 篩選邏輯一致，"
              "可以看量級跟方向，不要看到小數點後第二位的差異就下結論。")
    except Exception as e:
        print(f"（找不到或無法比對 backtest_riskadj_results.csv：{e}）")
