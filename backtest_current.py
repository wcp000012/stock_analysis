"""
現行模型（topK_predict.py 的特徵與標籤）的 hold-out 回測。

對每個錨點日 A：
  - 特徵、標準化、訓練標籤只用 <= A 的資料，A 之後的股價只用來評分
  - 季報衍生特徵延遲 50 天、月營收延遲 15 天才看得到（模擬公告時點，避免前視偏誤）
  - 評分：A 之後 60 個交易日的實際報酬（期末報酬、期間最高報酬）
  - 比較 Model B（純 long Top-20）與 Model C（long + reversal 過濾 Top-20）跟全市場平均
"""
import os
import gc
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume", "MA10", "RSI",
    "margin_buy", "margin_sell", "short_buy", "short_sell",
    "foreign_net", "invest_net", "dealer_net", "foreign_holding_pct",
    "pe_ratio", "pb_ratio", "div_yield",
    "rev_yoy", "rev_mom",
    "contract_liability_qoq", "inventory_qoq",
    "inventory_turnover", "gross_margin_q", "gross_margin_qoq",
]
QCOLS = ["contract_liability_qoq", "inventory_qoq", "inventory_turnover", "gross_margin_q", "gross_margin_qoq"]
MCOLS = ["rev_yoy", "rev_mom"]
Q_LAG_DAYS = 50       # 粗略值：Q1-Q3 季末後 45 天內申報。注意：這裡讀的是 parquet 裡已經
                      # 展開成每日重複值的季報欄位，已經遺失原始季末日期，無法像
                      # downLoad_Tech_and_merge_with_Chips.py 那樣準確判斷 Q4(年報，90天)，
                      # 所以 Q4 期間這個延遲仍然偏短（低估延遲，比沒延遲好但不是最準）。
                      # 2026-10-13 起 merge 腳本已在源頭修好完整版延遲，重新跑過 merge 產生
                      # 新 parquet 後，這裡的延遲邏輯其實是重複修正，可以評估拿掉。
M_LAG_DAYS = 15

PARQUET_GLOB = "parquet_data/*.parquet"
TIME_STEP = 126
FUTURE_DAYS = 60
WINDOW_STRIDE = 5
BATCH_SIZE = 512
LR = 5e-4
NUM_EPOCHS = 10
TOP_K = 20
LONG_POOL = TOP_K * 2
N_RUNS = 3
TRAIN_YEARS = 3
EVAL_DAYS = 60
MIN_EVAL_DAYS = 40
ANCHORS = ["2026-06-01", "2026-07-09"]
RESULT_CSV = "backtest_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stock(pq):
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return None
    if "Date" not in df.columns or any(c not in df.columns for c in FEATURE_COLS + ["Close"]):
        return None
    df = df[["Date"] + FEATURE_COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    # 公告延遲：季報值在 Date+50 天之後才看得到，月營收在 Date+15 天之後才看得到
    for cols, lag in [(QCOLS, Q_LAG_DAYS), (MCOLS, M_LAG_DAYS)]:
        src = df[["Date"] + cols].copy()
        src["Date"] = src["Date"] + pd.Timedelta(days=lag)
        m = pd.merge_asof(df[["Date"]], src, on="Date", direction="backward")
        df[cols] = m[cols].to_numpy()
    # 2026-10-13 修復：原本還有 .bfill()，會在 hist 被按錨點裁切之前，先把未來值往回填進
    # 缺值的早期日期，等於訓練窗口看到了未來。改成只 ffill + fillna(0)。
    df[FEATURE_COLS] = (df[FEATURE_COLS].astype(np.float32)
                        .replace([np.inf, -np.inf], np.nan).ffill().fillna(0))
    return df


def build_case(df, anchor):
    hist = df[df["Date"] <= anchor]
    fut = df[df["Date"] > anchor].head(EVAL_DAYS)
    if len(hist) < TIME_STEP or len(fut) < MIN_EVAL_DAYS:
        return None
    a_close = float(hist["Close"].iloc[-1])
    if not np.isfinite(a_close) or a_close <= 0:
        return None
    return {
        "hist": hist,
        "ret_end": float(fut["Close"].iloc[-1] / a_close - 1),
        "ret_max": float(fut["Close"].max() / a_close - 1),
    }


def windows_for(hist, scaler, train_start):
    d = hist[hist["Date"] >= train_start]
    n = len(d)
    max_i = n - TIME_STEP - FUTURE_DAYS
    if max_i <= 0:
        return None
    feats = np.nan_to_num(d[FEATURE_COLS].to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    feats_s = np.nan_to_num(scaler.transform(feats), nan=0, posinf=0, neginf=0).astype(np.float32)
    close = d["Close"].to_numpy(np.float32)
    wins = sliding_window_view(feats_s, (TIME_STEP, feats_s.shape[1])).squeeze(1)
    wins = wins[:max_i:WINDOW_STRIDE]
    ymax, ymin, keep = [], [], []
    for k, i in enumerate(range(0, max_i, WINDOW_STRIDE)):
        ap = close[i + TIME_STEP - 1]
        fut = close[i + TIME_STEP: i + TIME_STEP + FUTURE_DAYS]
        if len(fut) == 0:
            continue
        ymax.append(float(np.log((np.max(fut) + 1e-12) / (ap + 1e-12))))
        ymin.append(float(np.log((np.min(fut) + 1e-12) / (ap + 1e-12))))
        keep.append(k)
    if not keep:
        return None
    return wins[keep], np.array(ymax, np.float32), np.array(ymin, np.float32)


class PatchTSTRegression(nn.Module):
    def __init__(self, input_size, patch_len=32, stride=16, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len * input_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        B, T, F = x.shape
        patches = [x[:, i:i + self.patch_len, :].reshape(B, -1)
                   for i in range(0, T - self.patch_len + 1, self.stride)]
        h = self.patch_embed(torch.stack(patches, dim=1))
        return self.head(self.transformer(h)[:, -1, :]).squeeze(-1)


def train_model(X, Y):
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    model = PatchTSTRegression(input_size=X.shape[2]).to(DEVICE)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(NUM_EPOCHS):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = crit(model(bx), by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict(model, W):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(W), BATCH_SIZE):
            out.append(model(torch.from_numpy(W[i:i + BATCH_SIZE]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def run_anchor(anchor_str):
    anchor = pd.Timestamp(anchor_str)
    train_start = anchor - pd.DateOffset(years=TRAIN_YEARS)
    print(f"\n{'=' * 60}\n錨點 {anchor_str}（訓練起點 {train_start.date()}）\n{'=' * 60}")

    cases = {}
    for pq in tqdm(glob.glob(PARQUET_GLOB), desc="load"):
        df = load_stock(pq)
        if df is None:
            continue
        c = build_case(df, anchor)
        if c is None:
            continue
        cases[os.path.basename(pq).replace(".parquet", "")] = c
    print(f"可評分股票數: {len(cases)}")

    gmn = gmx = None
    for c in cases.values():
        h = c["hist"][c["hist"]["Date"] >= train_start]
        if h.empty:
            continue
        arr = np.nan_to_num(h[FEATURE_COLS].to_numpy(np.float32), nan=0, posinf=0, neginf=0)
        mn, mx = arr.min(0), arr.max(0)
        gmn = mn if gmn is None else np.minimum(gmn, mn)
        gmx = mx if gmx is None else np.maximum(gmx, mx)
    scaler = MinMaxScaler().fit(np.vstack([gmn, gmx]))

    Xs, Yxs, Yns = [], [], []
    for c in cases.values():
        w = windows_for(c["hist"], scaler, train_start)
        if w is not None:
            Xs.append(w[0]); Yxs.append(w[1]); Yns.append(w[2])
    X = np.concatenate(Xs).astype(np.float32)
    Ymax = np.concatenate(Yxs)
    Ymin = np.concatenate(Yns)
    print(f"訓練資料: X={X.shape}")
    del Xs, Yxs, Yns
    gc.collect()

    pred_sids, W = [], []
    for sid, c in cases.items():
        h = c["hist"]
        if len(h) < TIME_STEP:
            continue
        feats = np.nan_to_num(scaler.transform(h[FEATURE_COLS].to_numpy(np.float32)),
                              nan=0, posinf=0, neginf=0)
        W.append(feats[-TIME_STEP:].astype(np.float32))
        pred_sids.append(sid)
    W = np.stack(W)

    long_p, rev_p = [], []
    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}")
        m = train_model(X, Ymax)
        long_p.append(predict(m, W))
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        m = train_model(X, Ymin)
        rev_p.append(predict(m, W))
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    P = pd.DataFrame({"pred_max": np.mean(long_p, 0), "pred_min": np.mean(rev_p, 0)}, index=pred_sids)
    R = pd.DataFrame.from_dict({s: {"ret_end": cases[s]["ret_end"], "ret_max": cases[s]["ret_max"]}
                                for s in pred_sids}, orient="index")
    D = P.join(R, how="inner")

    univ_end = D["ret_end"].mean()
    univ_max = D["ret_max"].mean()
    ic_max = D["pred_max"].corr(D["ret_max"], method="spearman")
    ic_end = D["pred_max"].corr(D["ret_end"], method="spearman")
    q = D["pred_max"].quantile([0.1, 0.9])
    spread = (D.loc[D["pred_max"] >= q[0.9], "ret_end"].mean()
              - D.loc[D["pred_max"] <= q[0.1], "ret_end"].mean())

    B = D.nlargest(TOP_K, "pred_max")
    pool = D.nlargest(LONG_POOL, "pred_max")
    drop = set(pool.nsmallest(LONG_POOL - TOP_K, "pred_min").index)
    C = pool[~pool.index.isin(drop)].nlargest(TOP_K, "pred_max")

    rows = []
    for name, S in [("B 純long Top20", B), ("C long+反轉過濾 Top20", C)]:
        rows.append({
            "anchor": anchor_str, "model": name, "n_universe": len(D),
            "top_ret_end": S["ret_end"].mean(), "top_ret_max": S["ret_max"].mean(),
            "universe_ret_end": univ_end, "universe_ret_max": univ_max,
            "excess_end": S["ret_end"].mean() - univ_end,
            "hit_end": (S["ret_end"] > 0).mean(),
            "ic_max": ic_max, "ic_end": ic_end, "decile_spread_end": spread,
        })
        print(f"\n  {name}: 期末報酬 {S['ret_end'].mean():+.2%}（全市場 {univ_end:+.2%}，超額 {S['ret_end'].mean() - univ_end:+.2%}）"
              f"｜期間最高 {S['ret_max'].mean():+.2%}｜上漲命中 {(S['ret_end'] > 0).mean():.0%}")
        print("    " + ", ".join(f"{s}({r:+.1%})" for s, r in S["ret_end"].items()))
    print(f"\n  全市場 Spearman IC（預測 vs 期間最高）={ic_max:+.3f}，（預測 vs 期末）={ic_end:+.3f}")
    print(f"  預測前10% 與 後10% 的期末報酬差 = {spread:+.2%}")
    return rows


if __name__ == "__main__":
    all_rows = []
    for a in ANCHORS:
        all_rows += run_anchor(a)
    res = pd.DataFrame(all_rows)
    res.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n結果已存 {RESULT_CSV}")
    print(res[["anchor", "model", "top_ret_end", "universe_ret_end", "excess_end", "hit_end", "ic_max", "ic_end"]]
          .to_string(index=False))
