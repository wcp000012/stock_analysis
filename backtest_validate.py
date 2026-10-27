"""
Hold-out Backtest：驗證特徵與反轉過濾是否改善預測準確率

流程：
  錨點 = ANCHOR_DATE（該日之後 60 天我們有實際股價）
  Model A：17 特徵，無過濾
  Model B：19 特徵（+月營收），無過濾
  Model C：19 特徵 + Reversal Filter（long model + reversal model 交叉過濾）
"""
import os, gc, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ── 設定 ─────────────────────────────────────────────────────────────────────
ANCHOR_DATE  = "2025-12-01"
EVAL_DAYS    = 60
PARQUET_GLOB = "parquet_data/*.parquet"
TIME_STEP    = 126
FUTURE_DAYS  = 60
WINDOW_STRIDE= 5
BATCH_SIZE   = 512
LR           = 5e-4
NUM_EPOCHS   = 10
TOP_K        = 20
LONG_POOL    = TOP_K * 2   # reversal filter 候選池大小
N_RUNS       = 3
TRAIN_YEARS  = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS_A = [
    "Close", "High", "Low", "Open", "Volume", "MA10", "RSI",
    "margin_buy", "margin_sell", "short_buy", "short_sell",
    "foreign_net", "invest_net", "dealer_net",
    "pe_ratio", "pb_ratio", "div_yield"
]

FEATURE_COLS_B = FEATURE_COLS_A + [
    "rev_yoy", "rev_mom",
]

parquet_files = glob.glob(PARQUET_GLOB)
anchor        = pd.Timestamp(ANCHOR_DATE)
train_start   = anchor - pd.DateOffset(years=TRAIN_YEARS)


# ── 共用工具 ──────────────────────────────────────────────────────────────────
def safe_read(pq, feature_cols):
    try:
        df = pd.read_parquet(pq)
        if df is None or df.empty or "Date" not in df.columns:
            return None
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return None
        df = df[["Date"] + feature_cols].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[feature_cols] = (
            df[feature_cols].astype(np.float32)
            .replace([np.inf, -np.inf], np.nan)
            .ffill().bfill().fillna(0)
        )
        return df if not df.empty else None
    except Exception:
        return None


def fit_scaler(pq_files, feature_cols):
    gmn, gmx = None, None
    for pq in tqdm(pq_files, desc="Fit scaler", leave=False):
        df = safe_read(pq, feature_cols)
        if df is None:
            continue
        df = df[(df["Date"] >= train_start) & (df["Date"] <= anchor)]
        if df.empty:
            continue
        arr = np.nan_to_num(df[feature_cols].to_numpy(np.float32), nan=0)
        mn, mx = arr.min(0), arr.max(0)
        gmn = mn if gmn is None else np.minimum(gmn, mn)
        gmx = mx if gmx is None else np.maximum(gmx, mx)
    sc = MinMaxScaler()
    sc.fit(np.vstack([gmn, gmx]))
    return sc


def build_dataset(pq_files, scaler, feature_cols):
    """Returns X, Y_max, Y_min"""
    Xl, Ymaxl, Yminl = [], [], []
    for pq in tqdm(pq_files, desc="Build dataset", leave=False):
        df = safe_read(pq, feature_cols)
        if df is None:
            continue
        df = df[(df["Date"] >= train_start) & (df["Date"] <= anchor)].reset_index(drop=True)
        if len(df) < TIME_STEP + FUTURE_DAYS:
            continue
        feats     = np.nan_to_num(df[feature_cols].to_numpy(np.float32), nan=0)
        close_raw = df["Close"].to_numpy(np.float32)
        date_raw  = df["Date"].to_numpy()
        n = len(feats)
        max_i = n - TIME_STEP - FUTURE_DAYS
        if max_i <= 0:
            continue
        feats_s = np.nan_to_num(scaler.transform(feats), nan=0)
        wins = sliding_window_view(feats_s, (TIME_STEP, feats_s.shape[1])).squeeze(1)
        wins = wins[:max_i:WINDOW_STRIDE].astype(np.float32, copy=False)
        ymax_l, ymin_l, keep = [], [], []
        for k, i in enumerate(range(0, max_i, WINDOW_STRIDE)):
            lei = i + TIME_STEP + FUTURE_DAYS - 1
            if lei >= n or pd.Timestamp(date_raw[lei]) >= anchor:
                continue
            ap  = close_raw[i + TIME_STEP - 1]
            fut = close_raw[i + TIME_STEP: i + TIME_STEP + FUTURE_DAYS]
            if not len(fut):
                continue
            ymax_l.append(float(np.log((np.max(fut) + 1e-12) / (ap + 1e-12))))
            # max drawdown（負值：越負 = 回撤越大）
            running_max = np.maximum.accumulate(fut)
            max_dd = np.max((running_max - fut) / (running_max + 1e-12))
            ymin_l.append(-float(max_dd))
            keep.append(k)
        if not keep:
            continue
        Xl.append(wins[keep])
        Ymaxl.append(np.array(ymax_l, np.float32))
        Yminl.append(np.array(ymin_l, np.float32))
    if not Xl:
        raise ValueError("訓練資料集為空")
    return (np.concatenate(Xl, 0),
            np.concatenate(Ymaxl, 0),
            np.concatenate(Yminl, 0))


class PatchTSTRegression(nn.Module):
    def __init__(self, input_size, patch_len=32, stride=16, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.patch_len   = patch_len
        self.stride      = stride
        self.patch_embed = nn.Linear(patch_len * input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):
        B, T, F = x.shape
        patches = [x[:, i:i + self.patch_len, :].reshape(B, -1)
                   for i in range(0, T - self.patch_len + 1, self.stride)]
        h = self.patch_embed(torch.stack(patches, dim=1))
        return self.head(self.transformer(h)[:, -1, :]).squeeze(-1)


def train_model(X, Y, input_size):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    model = PatchTSTRegression(input_size=input_size).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.SmoothL1Loss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = crit(model(bx), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def predict_scores(pq_files, model, scaler, feature_cols, col="score"):
    records = []
    for pq in pq_files:
        df = safe_read(pq, feature_cols)
        if df is None:
            continue
        df = df[df["Date"] <= anchor]
        if len(df) < TIME_STEP:
            continue
        sid = os.path.basename(pq).replace(".parquet", "")
        feats   = np.nan_to_num(df[feature_cols].to_numpy(np.float32), nan=0)
        feats_s = np.nan_to_num(scaler.transform(feats), nan=0)
        records.append({"sid": sid, "window": feats_s[-TIME_STEP:]})
    X = torch.from_numpy(np.stack([r["window"] for r in records]).astype(np.float32))
    preds = []
    model.eval()
    with torch.no_grad():
        for (bx,) in DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False):
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    pred_arr = np.concatenate(preds)
    return pd.DataFrame({"sid": [r["sid"] for r in records], col: pred_arr})


def actual_return(pq, anchor_date, eval_days):
    try:
        df = pd.read_parquet(pq)
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        after     = df[df["Date"] > anchor_date].head(eval_days)
        at_anchor = df[df["Date"] <= anchor_date]
        if after.empty or at_anchor.empty:
            return None
        anchor_close = float(at_anchor["Close"].iloc[-1])
        max_future   = float(after["Close"].max())
        return (max_future - anchor_close) / anchor_close
    except Exception:
        return None


def eval_topk(top_sids, pq_map):
    actual = {}
    for sid in top_sids:
        pq = pq_map.get(sid)
        if pq:
            r = actual_return(pq, anchor, EVAL_DAYS)
            if r is not None:
                actual[sid] = r
    return pd.Series(actual)


def print_result(label, top_actual):
    if top_actual is None or top_actual.empty:
        print(f"  {label}: 無資料")
        return None, None
    avg = top_actual.mean()
    hit = (top_actual > 0).sum()
    print(f"\n  實際報酬 (Top {TOP_K} 中有資料的 {len(top_actual)} 支):")
    for sid, ret in top_actual.sort_values(ascending=False).items():
        print(f"    {sid}: {ret:+.1%}")
    print(f"\n  平均實際報酬: {avg:+.2%}")
    print(f"  命中率（實際上漲）: {hit}/{len(top_actual)} = {hit/len(top_actual):.0%}")
    return avg, hit / len(top_actual)


# ── Model A / B：純 long model，無 reversal filter ───────────────────────────
def run_model(label, feature_cols, pq_files):
    print(f"\n{'='*55}")
    print(f"  Model {label}  ({len(feature_cols)} features, no filter)")
    print(f"{'='*55}")
    scaler    = fit_scaler(pq_files, feature_cols)
    X, Y_max, _ = build_dataset(pq_files, scaler, feature_cols)
    print(f"  Dataset: X={X.shape}")

    all_preds = []
    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}")
        model  = train_model(X, Y_max, len(feature_cols))
        df_pred = predict_scores(pq_files, model, scaler, feature_cols, col="pred_max")
        all_preds.append(df_pred.set_index("sid")["pred_max"])
        del model; gc.collect(); torch.cuda.empty_cache()

    avg_pred = pd.concat(all_preds, axis=1).mean(axis=1)
    top_sids = list(avg_pred.nlargest(TOP_K).index)
    print(f"\n  Top {TOP_K} predicted: {top_sids[:10]}")

    pq_map     = {os.path.basename(p).replace(".parquet", ""): p for p in pq_files}
    top_actual = eval_topk(top_sids, pq_map)
    return print_result(label, top_actual)


# ── Model C：long + reversal filter ──────────────────────────────────────────
def run_model_with_filter(label, feature_cols, pq_files):
    print(f"\n{'='*55}")
    print(f"  Model {label}  ({len(feature_cols)} features + reversal filter)")
    print(f"{'='*55}")
    scaler         = fit_scaler(pq_files, feature_cols)
    X, Y_max, Y_min = build_dataset(pq_files, scaler, feature_cols)
    print(f"  Dataset: X={X.shape}")

    long_preds, rev_preds = [], []
    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}")

        model_long = train_model(X, Y_max, len(feature_cols))
        df_long    = predict_scores(pq_files, model_long, scaler, feature_cols, col="pred_max")
        long_preds.append(df_long.set_index("sid")["pred_max"])
        del model_long; gc.collect(); torch.cuda.empty_cache()

        model_rev = train_model(X, Y_min, len(feature_cols))
        df_rev    = predict_scores(pq_files, model_rev, scaler, feature_cols, col="pred_min")
        rev_preds.append(df_rev.set_index("sid")["pred_min"])
        del model_rev; gc.collect(); torch.cuda.empty_cache()

    avg_max = pd.concat(long_preds, axis=1).mean(axis=1).rename("pred_max")
    avg_min = pd.concat(rev_preds,  axis=1).mean(axis=1).rename("pred_min")
    combined = pd.concat([avg_max, avg_min], axis=1).dropna()

    # 候選池：long model 前 LONG_POOL
    pool    = combined.nlargest(LONG_POOL, "pred_max")
    # 剔除 reversal model 最差（最負）的一半
    n_drop  = LONG_POOL - TOP_K
    drop_set = set(pool.nsmallest(n_drop, "pred_min").index)
    filtered = pool[~pool.index.isin(drop_set)]
    top_sids = list(filtered.nlargest(TOP_K, "pred_max").index)

    print(f"\n  Top {TOP_K} predicted (after filter): {top_sids[:10]}")
    pq_map     = {os.path.basename(p).replace(".parquet", ""): p for p in pq_files}
    top_actual = eval_topk(top_sids, pq_map)
    return print_result(label, top_actual)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Anchor date  : {ANCHOR_DATE}")
    print(f"Eval period  : +{EVAL_DAYS} days")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"Device       : {DEVICE}")

    valid_pq = []
    for pq in parquet_files:
        try:
            df = pd.read_parquet(pq, columns=["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].max() > anchor:
                valid_pq.append(pq)
        except Exception:
            pass
    print(f"Valid stocks (data after anchor): {len(valid_pq)}")

    avg_a, hit_a = run_model("A (17特徵)", FEATURE_COLS_A, valid_pq)
    avg_b, hit_b = run_model("B (19特徵)", FEATURE_COLS_B, valid_pq)
    avg_c, hit_c = run_model_with_filter("C (19特徵+反轉過濾)", FEATURE_COLS_B, valid_pq)

    print(f"\n{'='*55}")
    print(f"  最終比較 (錨點: {ANCHOR_DATE}, 評估: +{EVAL_DAYS}d)")
    print(f"{'='*55}")
    print(f"  Model A (17特徵)          平均報酬: {avg_a:+.2%}  命中率: {hit_a:.0%}")
    print(f"  Model B (19特徵)          平均報酬: {avg_b:+.2%}  命中率: {hit_b:.0%}")
    print(f"  Model C (19特徵+反轉過濾) 平均報酬: {avg_c:+.2%}  命中率: {hit_c:.0%}")
    print(f"  B→C 改善幅度: {avg_c - avg_b:+.2%}")
