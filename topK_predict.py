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

# ── 超參數 ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # 技術面（7）
    "Close", "High", "Low", "Open", "Volume", "MA10", "RSI",
    # 籌碼面（8）
    "margin_buy", "margin_sell", "short_buy", "short_sell",
    "foreign_net", "invest_net", "dealer_net", "foreign_holding_pct",
    # 每日基本面（3）
    "pe_ratio", "pb_ratio", "div_yield",
    # 月營收特徵（2）
    "rev_yoy", "rev_mom",
    # 季報特徵（5）— 覆蓋率不保證全市場，缺值沿用既有 ffill/fillna(0) 邏輯，
    # 不因缺值剔除股票（與 pe_ratio/rev_yoy 等既有稀疏特徵同一套處理方式）
    "contract_liability_qoq", "inventory_qoq",
    "inventory_turnover", "gross_margin_q", "gross_margin_qoq",
]
PARQUET_GLOB     = "parquet_data/*.parquet"
TIME_STEP        = 126
FUTURE_DAYS      = 60
WINDOW_STRIDE    = 5
BATCH_SIZE       = 512
LR               = 5e-4
NUM_EPOCHS       = 10
TOP_K            = 20
LONG_POOL        = TOP_K * 2   # 先從 long model 取 2x 候選，再用 reversal 過濾

# ── Model D（觀察用，2026-10-20 新增）──────────────────────────────────────────
# 不取代 Model B/C。回測（RESEARCH_LOG.md E9/E9b）發現 Model B 的分數跟「外資流向」
# 風險特性互補，混合排名在 9 個錨點回測裡比純模型更穩定；但還沒有 out-of-sample 驗證，
# 這裡只是把混合分數也算出來、另存一份 Top-20 做觀察，不影響 B/C 的產出或排程的成功判斷。
MOM_WIN          = 60   # 跟 E9 的 mom60 定義一致
FOREIGN_WIN      = 20   # 跟 E9 的 foreign 定義一致
ENS_W_RAW, ENS_W_FOREIGN, ENS_W_MOM = 0.7, 0.2, 0.1   # 見 E9b 網格搜尋，21/66 組全正超額裡的代表點
N_RUNS           = 5           # 訓練幾次取平均
TRAIN_YEARS      = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parquet_files = glob.glob(PARQUET_GLOB)
print(f"Found parquet files: {len(parquet_files)}")
print(f"Device: {DEVICE}")

TRAIN_START = pd.Timestamp.today().normalize() - pd.DateOffset(years=TRAIN_YEARS)
print(f"Training data range: {TRAIN_START.date()} -> today")


# ── 資料讀取 ─────────────────────────────────────────────────────────────────
def safe_read_parquet(pq, feature_cols):
    try:
        df = pd.read_parquet(pq)
        if df is None or df.empty:
            return None
        if "Date" not in df.columns:
            return None
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return None

        df = df[["Date"] + feature_cols].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # 2026-10-13 修復：原本這裡還有 .bfill()，會把「未來才出現的值」往回填進缺值的
        # 早期日期，對切出來的訓練窗口來說等於看到了未來。改成只 ffill（用過去已知值延續到
        # 現在）+ fillna(0)（還沒有任何已知值之前，視為未知，不用未來值回填）。
        df[feature_cols] = (
            df[feature_cols]
            .astype(np.float32)
            .replace([np.inf, -np.inf], np.nan)
            .ffill().fillna(0)
        )
        return df if not df.empty else None
    except Exception as e:
        print(f"  [WARN] {pq}: {e}")
        return None


# ── Global Scaler ─────────────────────────────────────────────────────────────
def fit_global_scaler(parquet_files, feature_cols, train_start):
    global_min, global_max = None, None
    for pq in tqdm(parquet_files, desc="Fit scaler"):
        df = safe_read_parquet(pq, feature_cols)
        if df is None:
            continue
        df = df[df["Date"] >= train_start]
        if df.empty:
            continue
        arr = np.nan_to_num(df[feature_cols].to_numpy(dtype=np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        mn, mx = arr.min(axis=0), arr.max(axis=0)
        if not (np.isfinite(mn).all() and np.isfinite(mx).all()):
            continue
        global_min = mn if global_min is None else np.minimum(global_min, mn)
        global_max = mx if global_max is None else np.maximum(global_max, mx)

    if global_min is None:
        raise ValueError("無法建立 scaler，沒有有效資料")
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([global_min, global_max]))
    return scaler


# ── 訓練資料集（同時產生 max/min label）─────────────────────────────────────
def build_train_dataset(parquet_files, scaler, feature_cols, train_start,
                        time_step, future_days, window_stride, chunk_size=20):
    """
    Returns: X, Y_max, Y_min
      Y_max = log(max_future_close / anchor_close)  → long model label
      Y_min = log(min_future_close / anchor_close)  → reversal model label
    """
    today = pd.Timestamp.today().normalize()
    X_chunks, Ymax_chunks, Ymin_chunks = [], [], []

    for start in range(0, len(parquet_files), chunk_size):
        chunk = parquet_files[start:start + chunk_size]
        Xl, Ymaxl, Yminl = [], [], []

        for pq in chunk:
            df = safe_read_parquet(pq, feature_cols)
            if df is None:
                continue

            df = df[df["Date"] >= train_start].reset_index(drop=True)
            if df.empty:
                continue

            feats     = df[feature_cols].to_numpy(dtype=np.float32)
            close_raw = df["Close"].to_numpy(dtype=np.float32)
            date_raw  = df["Date"].to_numpy()
            n = len(feats)
            max_i = n - time_step - future_days
            if max_i <= 0:
                continue

            feats_s = np.nan_to_num(
                scaler.transform(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)),
                nan=0.0, posinf=0.0, neginf=0.0
            )

            wins = sliding_window_view(feats_s, (time_step, feats_s.shape[1])).squeeze(1)
            wins = wins[:max_i:window_stride].astype(np.float32, copy=False)

            indices = list(range(0, max_i, window_stride))
            ymax_list, ymin_list, keep = [], [], []
            for k, i in enumerate(indices):
                label_end_idx = i + time_step + future_days - 1
                if label_end_idx >= n:
                    continue
                if pd.Timestamp(date_raw[label_end_idx]) >= today:
                    continue
                anchor_price  = close_raw[i + time_step - 1]
                future_prices = close_raw[i + time_step: i + time_step + future_days]
                if len(future_prices) == 0:
                    continue
                ymax_list.append(float(np.log((np.max(future_prices) + 1e-12) / (anchor_price + 1e-12))))
                ymin_list.append(float(np.log((np.min(future_prices) + 1e-12) / (anchor_price + 1e-12))))
                keep.append(k)

            if not keep:
                continue
            Xl.append(wins[keep])
            Ymaxl.append(np.array(ymax_list, dtype=np.float32))
            Yminl.append(np.array(ymin_list, dtype=np.float32))

            del df, feats, feats_s, close_raw, date_raw, wins
            gc.collect()

        if Xl:
            X_chunks.append(np.concatenate(Xl, axis=0))
            Ymax_chunks.append(np.concatenate(Ymaxl, axis=0))
            Ymin_chunks.append(np.concatenate(Yminl, axis=0))
            del Xl, Ymaxl, Yminl
            gc.collect()

    if not X_chunks:
        raise ValueError("訓練資料集為空，請確認 parquet 資料日期範圍")

    X    = np.concatenate(X_chunks, axis=0).astype(np.float32)
    Ymax = np.concatenate(Ymax_chunks, axis=0).astype(np.float32)
    Ymin = np.concatenate(Ymin_chunks, axis=0).astype(np.float32)
    print(f"Train dataset: X={X.shape}, Y_max={Ymax.shape}, Y_min={Ymin.shape}")
    return X, Ymax, Ymin


# ── Model ─────────────────────────────────────────────────────────────────────
class PatchTSTRegression(nn.Module):
    def __init__(self, input_size=14, patch_len=32, stride=16,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.patch_embed = nn.Linear(patch_len * input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B, T, F = x.shape
        patches = [x[:, i:i + self.patch_len, :].reshape(B, -1)
                   for i in range(0, T - self.patch_len + 1, self.stride)]
        h = self.patch_embed(torch.stack(patches, dim=1))
        return self.head(self.transformer(h)[:, -1, :]).squeeze(-1)


# ── 訓練 ──────────────────────────────────────────────────────────────────────
def train_model(X_train, Y_train, input_size, device,
                num_epochs=10, batch_size=512, lr=5e-4):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    model = PatchTSTRegression(input_size=input_size).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss, count = 0.0, 0
        for bx, by in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            count += bx.size(0)
        print(f"[Epoch {epoch+1}] loss={total_loss / max(count, 1):.6f}")

    return model


# ── Live Prediction ───────────────────────────────────────────────────────────
def predict_live(parquet_files, model, scaler, feature_cols,
                 time_step, device, col_name="pred_log_return", batch_size=512):
    records = []
    for pq in tqdm(parquet_files, desc=f"Predict ({col_name})"):
        df = safe_read_parquet(pq, feature_cols)
        if df is None or len(df) < time_step:
            continue
        stock_code  = os.path.basename(pq).replace(".parquet", "")
        anchor_date = df["Date"].iloc[-1]
        feats = np.nan_to_num(df[feature_cols].to_numpy(dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        feats_s = np.nan_to_num(scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0)
        records.append({
            "stock_code":  stock_code,
            "anchor_date": anchor_date,
            "window":      feats_s[-time_step:]
        })

    if not records:
        raise ValueError("沒有可預測的股票")

    X = torch.from_numpy(
        np.stack([r["window"] for r in records], axis=0).astype(np.float32)
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for (bx,) in DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False):
            preds.append(model(bx.to(device)).cpu().numpy())

    pred_arr = np.concatenate(preds)
    return pd.DataFrame({
        "stock_code":  [r["stock_code"]  for r in records],
        "anchor_date": [r["anchor_date"] for r in records],
        col_name:      pred_arr,
    })


# ── Model D 用的輔助訊號（觀察用，見上方常數說明）───────────────────────────────
def compute_aux_signals(parquet_files):
    """算每檔股票的過去 60 日動能（mom60）跟過去 20 日外資流向強度（foreign），
    定義跟 backtest_lab.py 的 build_case 一致，只用 Close/Volume/foreign_net 三欄，
    不需要訓練。任何單一檔案失敗都跳過，不影響其他股票。"""
    rows = []
    for pq in tqdm(parquet_files, desc="Aux signals (mom60/foreign，Model D 用)"):
        try:
            df = pd.read_parquet(pq, columns=["Date", "Close", "Volume", "foreign_net"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            if len(df) < MOM_WIN + 1:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce")
            a_close, prev_close = float(close.iloc[-1]), float(close.iloc[-1 - MOM_WIN])
            mom60 = float(np.log(a_close / prev_close)) if a_close > 0 and prev_close > 0 else np.nan

            tail = df.tail(FOREIGN_WIN)
            vol_sum = pd.to_numeric(tail["Volume"], errors="coerce").fillna(0).sum()
            for_sum = pd.to_numeric(tail["foreign_net"], errors="coerce").fillna(0).sum()
            foreign = float(for_sum / vol_sum) if vol_sum > 0 else np.nan

            rows.append({"stock_code": os.path.basename(pq).replace(".parquet", ""),
                         "mom60": mom60, "foreign": foreign})
        except Exception as e:
            print(f"  [WARN][Model D aux] {pq}: {e}")
            continue
    return pd.DataFrame(rows, columns=["stock_code", "mom60", "foreign"])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_feats = len(FEATURE_COLS)

    # 1) Scaler
    print("\n[1/3] Fitting global scaler ...")
    scaler = fit_global_scaler(parquet_files, FEATURE_COLS, TRAIN_START)

    # 2) 訓練資料（同時產生 max/min label）
    print("\n[2/3] Building training dataset ...")
    X_train, Y_max, Y_min = build_train_dataset(
        parquet_files, scaler, FEATURE_COLS, TRAIN_START,
        time_step=TIME_STEP, future_days=FUTURE_DAYS,
        window_stride=WINDOW_STRIDE,
    )

    # 3) 多次訓練：long model（預測最大報酬）+ reversal model（預測最大跌幅）
    print(f"\n[3/3] Training long & reversal models ({N_RUNS} runs each) ...")
    long_preds, rev_preds = [], []
    ref_df = None

    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")

        # Long model
        print(f"  [Long model]")
        model_long = train_model(X_train, Y_max, n_feats, DEVICE, NUM_EPOCHS, BATCH_SIZE, LR)
        torch.save(model_long.state_dict(), f"model_long_run{run}.pt")
        df_long = predict_live(parquet_files, model_long, scaler, FEATURE_COLS,
                               TIME_STEP, DEVICE, col_name="pred_max")
        if ref_df is None:
            ref_df = df_long[["stock_code", "anchor_date"]].copy()
        long_preds.append(df_long.set_index("stock_code")["pred_max"])
        del model_long
        gc.collect(); torch.cuda.empty_cache()

        # Reversal model
        print(f"  [Reversal model]")
        model_rev = train_model(X_train, Y_min, n_feats, DEVICE, NUM_EPOCHS, BATCH_SIZE, LR)
        torch.save(model_rev.state_dict(), f"model_rev_run{run}.pt")
        df_rev = predict_live(parquet_files, model_rev, scaler, FEATURE_COLS,
                              TIME_STEP, DEVICE, col_name="pred_min")
        rev_preds.append(df_rev.set_index("stock_code")["pred_min"])
        del model_rev
        gc.collect(); torch.cuda.empty_cache()

    # 平均各 run 的預測
    avg_max = pd.concat(long_preds, axis=1).mean(axis=1).rename("pred_max")
    avg_min = pd.concat(rev_preds,  axis=1).mean(axis=1).rename("pred_min")

    result_df = ref_df.set_index("stock_code").join(avg_max).join(avg_min).reset_index()
    result_df["pred_max_simple"] = np.expm1(result_df["pred_max"])
    result_df["pred_min_simple"] = np.expm1(result_df["pred_min"])

    # ── Model B：純 long model Top-K（主要結果）────────────────────────────
    top_b = result_df.nlargest(TOP_K, "pred_max").reset_index(drop=True)

    print(f"\n=== [Model B] Top {TOP_K} — long model only ({N_RUNS} runs avg) ===")
    print(top_b[["stock_code", "anchor_date", "pred_max_simple"]].to_string(index=False))

    # ── Model C：long pool → reversal filter（參考）────────────────────────
    long_pool = result_df.nlargest(LONG_POOL, "pred_max")
    n_drop    = LONG_POOL - TOP_K
    drop_set  = set(long_pool.nsmallest(n_drop, "pred_min")["stock_code"])
    filtered  = long_pool[~long_pool["stock_code"].isin(drop_set)]
    top_c     = filtered.nlargest(TOP_K, "pred_max").reset_index(drop=True)

    print(f"\n=== [Model C] Top {TOP_K} — long + reversal filter (參考) ===")
    print(top_c[["stock_code", "anchor_date",
                 "pred_max_simple", "pred_min_simple"]].to_string(index=False))

    # 儲存（Model B/C 永遠先存，不受下面 Model D 的計算影響）
    result_df.sort_values("pred_max", ascending=False).to_csv(
        "live_prediction_all.csv", index=False)
    top_b.to_csv("live_prediction_top_B.csv", index=False)
    top_c.to_csv("live_prediction_top_C.csv", index=False)
    print("\nSaved -> live_prediction_all.csv / live_prediction_top_B.csv / live_prediction_top_C.csv")

    # ── Model D（觀察用，見上方常數說明）：失敗也不能影響 B/C 已經存好的結果 ──────
    try:
        aux = compute_aux_signals(parquet_files)
        ens_df = result_df.merge(aux, on="stock_code", how="left")
        ens_df = ens_df.dropna(subset=["pred_max", "foreign", "mom60"]).copy()
        if len(ens_df) < TOP_K:
            raise ValueError(f"可用於 Model D 的股票數太少（{len(ens_df)} < {TOP_K}）")
        ens_df["ens_score"] = (
            ENS_W_RAW     * ens_df["pred_max"].rank(pct=True)
            + ENS_W_FOREIGN * ens_df["foreign"].rank(pct=True)
            + ENS_W_MOM     * ens_df["mom60"].rank(pct=True)
        )
        top_d = ens_df.nlargest(TOP_K, "ens_score")[
            ["stock_code", "anchor_date", "pred_max_simple", "mom60", "foreign", "ens_score"]
        ].reset_index(drop=True)

        print(f"\n=== [Model D，觀察用，尚無 out-of-sample 驗證] Top {TOP_K} — "
              f"pred_max/foreign/mom60 = {ENS_W_RAW}/{ENS_W_FOREIGN}/{ENS_W_MOM} 混合排名（見 RESEARCH_LOG.md E9/E9b）===")
        print(top_d.to_string(index=False))

        # all 檔案也附上 mom60/foreign/ens_score，方便之後直接核對、不用重算
        result_df.merge(aux, on="stock_code", how="left").merge(
            ens_df[["stock_code", "ens_score"]], on="stock_code", how="left"
        ).sort_values("pred_max", ascending=False).to_csv("live_prediction_all.csv", index=False)
        top_d.to_csv("live_prediction_top_D.csv", index=False)
        print("Saved -> live_prediction_top_D.csv（觀察用，不影響 Model B/C）")
    except Exception as e:
        print(f"\n[WARN] Model D（觀察用）計算失敗，不影響 Model B/C 已存檔的結果：{e}")
