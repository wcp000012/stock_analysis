"""
風險調整標籤回測：比較「原始 60 日最高報酬標籤」與「除以前期波動的風險調整標籤」。

同一批錨點、同一份特徵、同一套延遲與評分方式，只差 long model 的訓練標籤：
  raw : y = log(最高價 / 錨點價)
  adj : y = log(最高價 / 錨點價) / (前60日日報酬標準差 * sqrt(60))   （無前視，波動只用錨點前的資料）
reversal model 兩者都用原始標籤（與現行 Model C 一致），只改 long 部分。

輸出 backtest_riskadj_results.csv，並記錄各組 Top-20 的平均前期波動，用來檢驗波動偏好是否下降。
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
Q_LAG_DAYS = 50       # 粗略值：見 backtest_current.py 同名常數註解（Q4/年報期間仍偏短，
                      # downLoad_Tech_and_merge_with_Chips.py 的源頭修復更準確）
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
VOL_WIN = 60
ANCHORS = ["2025-12-01", "2026-03-02", "2026-06-01", "2026-07-09"]
RESULT_CSV = "backtest_riskadj_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stock(pq):
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return None
    if "Date" not in df.columns or any(c not in df.columns for c in FEATURE_COLS):
        return None
    df = df[["Date"] + FEATURE_COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
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
    logr = np.diff(np.log(hist["Close"].to_numpy(np.float64)), prepend=np.nan)
    vol = pd.Series(logr).rolling(VOL_WIN).std().iloc[-1]
    return {
        "hist": hist,
        "ret_end": float(fut["Close"].iloc[-1] / a_close - 1),
        "ret_max": float(fut["Close"].max() / a_close - 1),
        "vol_anchor": float(vol * np.sqrt(252)) if np.isfinite(vol) else np.nan,
    }


def windows_for(hist, scaler, train_start):
    d = hist[hist["Date"] >= train_start]
    n = len(d)
    max_i = n - TIME_STEP - FUTURE_DAYS
    if max_i <= 0:
        return None
    feats = np.nan_to_num(d[FEATURE_COLS].to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    feats_s = np.nan_to_num(scaler.transform(feats), nan=0, posinf=0, neginf=0).astype(np.float32)
    close = d["Close"].to_numpy(np.float64)
    logr = np.diff(np.log(close), prepend=np.nan)
    sig = pd.Series(logr).rolling(VOL_WIN).std().to_numpy()
    wins = sliding_window_view(feats_s, (TIME_STEP, feats_s.shape[1])).squeeze(1)
    wins = wins[:max_i:WINDOW_STRIDE]
    ymax_raw, ymin, ymax_adj, keep = [], [], [], []
    for k, i in enumerate(range(0, max_i, WINDOW_STRIDE)):
        e = i + TIME_STEP - 1
        ap = close[e]
        fut = close[e + 1: e + 1 + FUTURE_DAYS]
        if len(fut) == 0 or not np.isfinite(sig[e]) or sig[e] <= 0:
            continue
        ymr = float(np.log((np.max(fut) + 1e-12) / (ap + 1e-12)))
        ymax_raw.append(ymr)
        ymin.append(float(np.log((np.min(fut) + 1e-12) / (ap + 1e-12))))
        ymax_adj.append(ymr / (sig[e] * np.sqrt(FUTURE_DAYS)))
        keep.append(k)
    if not keep:
        return None
    return (wins[keep], np.array(ymax_raw, np.float32),
            np.array(ymin, np.float32), np.array(ymax_adj, np.float32))


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


def pick_c(D, long_col):
    pool = D.nlargest(LONG_POOL, long_col)
    drop = set(pool.nsmallest(LONG_POOL - TOP_K, "pred_min").index)
    return pool[~pool.index.isin(drop)].nlargest(TOP_K, long_col)


def run_anchor(anchor_str):
    anchor = pd.Timestamp(anchor_str)
    train_start = anchor - pd.DateOffset(years=TRAIN_YEARS)
    print(f"\n{'=' * 60}\n錨點 {anchor_str}（訓練起點 {train_start.date()}）\n{'=' * 60}", flush=True)

    cases = {}
    for pq in tqdm(glob.glob(PARQUET_GLOB), desc="load"):
        df = load_stock(pq)
        if df is None:
            continue
        c = build_case(df, anchor)
        if c is None or not np.isfinite(c["vol_anchor"]):
            continue
        cases[os.path.basename(pq).replace(".parquet", "")] = c
    print(f"可評分股票數: {len(cases)}", flush=True)

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

    Xs, Yr, Yn, Ya = [], [], [], []
    for c in cases.values():
        w = windows_for(c["hist"], scaler, train_start)
        if w is not None:
            Xs.append(w[0]); Yr.append(w[1]); Yn.append(w[2]); Ya.append(w[3])
    X = np.concatenate(Xs).astype(np.float32)
    Ymax_raw, Ymin, Ymax_adj = np.concatenate(Yr), np.concatenate(Yn), np.concatenate(Ya)
    print(f"訓練資料: X={X.shape}", flush=True)
    del Xs, Yr, Yn, Ya
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

    raw_p, rev_p, adj_p = [], [], []
    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}", flush=True)
        m = train_model(X, Ymax_raw); raw_p.append(predict(m, W))
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = train_model(X, Ymin); rev_p.append(predict(m, W))
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = train_model(X, Ymax_adj); adj_p.append(predict(m, W))
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    P = pd.DataFrame({
        "pred_raw": np.mean(raw_p, 0),
        "pred_adj": np.mean(adj_p, 0),
        "pred_min": np.mean(rev_p, 0),
    }, index=pred_sids)
    R = pd.DataFrame.from_dict({s: {"ret_end": cases[s]["ret_end"], "ret_max": cases[s]["ret_max"],
                                    "vol": cases[s]["vol_anchor"]} for s in pred_sids}, orient="index")
    D = P.join(R, how="inner")

    univ_end, univ_max, univ_vol = D["ret_end"].mean(), D["ret_max"].mean(), D["vol"].mean()
    rows = []
    variants = {
        "B_raw": D.nlargest(TOP_K, "pred_raw"),
        "C_raw": pick_c(D.rename(columns={"pred_raw": "pred_max"}), "pred_max"),
        "B_adj": D.nlargest(TOP_K, "pred_adj"),
        "C_adj": pick_c(D.rename(columns={"pred_adj": "pred_max"}), "pred_max"),
    }
    for name, S in variants.items():
        ic_max = D["pred_raw" if name.endswith("raw") else "pred_adj"].corr(D["ret_max"], method="spearman")
        ic_end = D["pred_raw" if name.endswith("raw") else "pred_adj"].corr(D["ret_end"], method="spearman")
        rows.append({
            "anchor": anchor_str, "variant": name, "n_universe": len(D),
            "top_ret_end": S["ret_end"].mean(), "universe_ret_end": univ_end,
            "excess_end": S["ret_end"].mean() - univ_end,
            "hit_end": (S["ret_end"] > 0).mean(),
            "top_vol": S["vol"].mean(), "universe_vol": univ_vol,
            "ic_max": ic_max, "ic_end": ic_end,
        })
        print(f"  {name}: 期末 {S['ret_end'].mean():+.2%}（全市場 {univ_end:+.2%}，超額 {S['ret_end'].mean() - univ_end:+.2%}）"
              f"｜命中 {(S['ret_end'] > 0).mean():.0%}｜Top 平均波動 {S['vol'].mean():.2f}（全市場 {univ_vol:.2f}）"
              f"｜IC_end {ic_end:+.3f}", flush=True)
    return rows


if __name__ == "__main__":
    all_rows = []
    for a in ANCHORS:
        all_rows += run_anchor(a)
        pd.DataFrame(all_rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n結果已存 {RESULT_CSV}", flush=True)
