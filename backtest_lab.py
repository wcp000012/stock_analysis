"""
持續迭代用的回測實驗室。

設計原則：
  - 固定隨機種子（E4 發現同資料訓練兩次 Top-5 只重疊 1 檔，不固定種子的比較沒有意義）
  - 每個錨點只「訓練」一次（Y_max 跟 Y_end 兩種標籤各訓 N_RUNS 次取平均），把每檔股票的
    原始分數 + 動能/外資流向/波動輔助欄位存成 lab_scores_<anchor>.csv，事後可以在不重新
    訓練的情況下，用這些欄位試各種排序/過濾/混合規則。
  - 讀的 parquet 是 2026-10-19 merge 修復過（E6 公告延遲 + E7 月營收單位）之後的版本，
    這裡不再重複套用延遲（parquet 已經內建），只做 ffill+fillna(0)（不做 bfill，見 E6）。

標籤：
  Y_max : log(錨點後 60 日內最高價 / 錨點價)   —— 現行 Model B 的標籤
  Y_end : log(錨點後第 60 個交易日收盤 / 錨點價) —— 直接預測「60 日後的實際報酬」，
          不特別看中途最高點，理論上比較不會被引導去賭高波動股

變體（全部從同一次訓練的分數事後算出來，不用重新訓練）：
  raw        : Y_max 模型原始分數排序
  term       : Y_end 模型原始分數排序
  vol_resid  : raw 分數對 vol_anchor 做線性回歸，取殘差排序（去掉跟波動的線性相關）
  volmom_resid: raw 分數對 vol_anchor + mom60 做線性回歸，取殘差排序
  vol_filter : 從 raw 前 40 名裡，先濾掉波動最高的 25%，剩下的裡面取前 20
  ens_foreign: raw 分數跟 foreign 流向分數的排名各半混合
  mom60/foreign/random : 跟 E8 一樣的免費規則，同一批資料重新算一次，當對照
"""
import gc
import glob
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── 固定隨機性 ────────────────────────────────────────────────────────────────
SEED_BASE = 20261020
random.seed(SEED_BASE)
np.random.seed(SEED_BASE)
torch.manual_seed(SEED_BASE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_BASE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume", "MA10", "RSI",
    "margin_buy", "margin_sell", "short_buy", "short_sell",
    "foreign_net", "invest_net", "dealer_net", "foreign_holding_pct",
    "pe_ratio", "pb_ratio", "div_yield",
    "rev_yoy", "rev_mom",
    "contract_liability_qoq", "inventory_qoq",
    "inventory_turnover", "gross_margin_q", "gross_margin_qoq",
]
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
MOM_WIN = 60
FOREIGN_WIN = 20
ANCHORS = ["2025-06-02", "2025-09-01", "2025-10-15", "2026-01-15", "2026-04-15"]
# 註：2025-12-01/2026-03-02/2026-06-01/2026-07-09 這 4 個錨點已經跑過（結果在
# backtest_lab_results_anchors1-4.csv、lab_scores_<anchor>.csv），這裡只跑新增的
# 5 個錨點，避免重複訓練浪費時間；事後分析時把兩批 lab_scores_*.csv 合併看。
RESULT_CSV = "backtest_lab_results.csv"
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
    # parquet 已經是延遲修復過的版本，這裡不再重複加延遲；只 ffill+fillna(0)（不做 bfill，見 E6）
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

    mom60 = np.nan
    if len(hist) > MOM_WIN:
        prev_close = float(hist["Close"].iloc[-1 - MOM_WIN])
        if prev_close > 0:
            mom60 = float(np.log(a_close / prev_close))

    tail = hist.tail(FOREIGN_WIN)
    vsum = tail["Volume"].sum()
    foreign = float(tail["foreign_net"].sum() / vsum) if vsum > 0 else np.nan

    return {
        "hist": hist,
        "ret_end": float(fut["Close"].iloc[-1] / a_close - 1),
        "ret_max": float(fut["Close"].max() / a_close - 1),
        "vol_anchor": float(vol * np.sqrt(252)) if np.isfinite(vol) else np.nan,
        "mom60": mom60,
        "foreign": foreign,
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
    ymax, yend, keep = [], [], []
    for k, i in enumerate(range(0, max_i, WINDOW_STRIDE)):
        ap = close[i + TIME_STEP - 1]
        fut = close[i + TIME_STEP: i + TIME_STEP + FUTURE_DAYS]
        if len(fut) == 0:
            continue
        ymax.append(float(np.log((np.max(fut) + 1e-12) / (ap + 1e-12))))
        yend.append(float(np.log((fut[-1] + 1e-12) / (ap + 1e-12))))
        keep.append(k)
    if not keep:
        return None
    return wins[keep], np.array(ymax, np.float32), np.array(yend, np.float32)


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


def train_model(X, Y, seed):
    torch.manual_seed(seed)
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


def resid(score, factors):
    """score 對 factors（DataFrame）做線性回歸，回傳殘差（分數裡跟 factors 無關的部分）。"""
    ok = factors.notna().all(axis=1) & score.notna()
    r = pd.Series(np.nan, index=score.index)
    if ok.sum() > 30:
        reg = LinearRegression().fit(factors[ok].to_numpy(), score[ok].to_numpy())
        r[ok] = score[ok].to_numpy() - reg.predict(factors[ok].to_numpy())
    return r


def train_anchor(anchor_str, seed_base):
    anchor = pd.Timestamp(anchor_str)
    train_start = anchor - pd.DateOffset(years=TRAIN_YEARS)
    print(f"\n{'=' * 60}\n錨點 {anchor_str}（訓練起點 {train_start.date()}）\n{'=' * 60}", flush=True)

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

    Xs, Ymaxs, Yends = [], [], []
    for c in cases.values():
        w = windows_for(c["hist"], scaler, train_start)
        if w is not None:
            Xs.append(w[0]); Ymaxs.append(w[1]); Yends.append(w[2])
    X = np.concatenate(Xs).astype(np.float32)
    Ymax = np.concatenate(Ymaxs)
    Yend = np.concatenate(Yends)
    print(f"訓練資料: X={X.shape}")
    del Xs, Ymaxs, Yends
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

    max_preds, end_preds = [], []
    for run in range(N_RUNS):
        seed = seed_base + run
        print(f"  [Y_max] run {run + 1}/{N_RUNS} (seed={seed})", flush=True)
        m = train_model(X, Ymax, seed)
        max_preds.append(predict(m, W))
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed2 = seed_base + 1000 + run
        print(f"  [Y_end] run {run + 1}/{N_RUNS} (seed={seed2})", flush=True)
        m = train_model(X, Yend, seed2)
        end_preds.append(predict(m, W))
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    D = pd.DataFrame({
        "pred_max_raw": np.mean(max_preds, axis=0),
        "pred_term_raw": np.mean(end_preds, axis=0),
    }, index=pred_sids)
    aux = pd.DataFrame.from_dict(
        {s: {"ret_end": cases[s]["ret_end"], "ret_max": cases[s]["ret_max"],
             "vol_anchor": cases[s]["vol_anchor"], "mom60": cases[s]["mom60"],
             "foreign": cases[s]["foreign"]} for s in pred_sids},
        orient="index")
    D = D.join(aux, how="inner")
    D.to_csv(f"lab_scores_{anchor_str}.csv", encoding="utf-8-sig")
    del X, Ymax, Yend, W
    gc.collect()
    return D


def evaluate(D, anchor_str):
    univ_end, univ_vol = D["ret_end"].mean(), D["vol_anchor"].mean()
    rng = np.random.default_rng(42)
    D = D.copy()
    D["random"] = rng.permutation(len(D))

    D["vol_resid"] = resid(D["pred_max_raw"], D[["vol_anchor"]])
    D["volmom_resid"] = resid(D["pred_max_raw"], D[["vol_anchor", "mom60"]])

    rows = []

    def add_row(name, top, score_col):
        ic_end = D[score_col].corr(D["ret_end"], method="spearman") if score_col in D.columns else np.nan
        ic_max = D[score_col].corr(D["ret_max"], method="spearman") if score_col in D.columns else np.nan
        row = {
            "anchor": anchor_str, "variant": name, "n_universe": len(D),
            "top_ret_end": top["ret_end"].mean(), "universe_ret_end": univ_end,
            "excess_end": top["ret_end"].mean() - univ_end,
            "hit_end": (top["ret_end"] > 0).mean(),
            "top_vol": top["vol_anchor"].mean(), "universe_vol": univ_vol,
            "ic_max": ic_max, "ic_end": ic_end,
        }
        rows.append(row)
        print(f"  {name:14s}: 期末 {row['top_ret_end']:+.2%}（全市場 {univ_end:+.2%}，"
              f"超額 {row['excess_end']:+.2%}）｜命中 {row['hit_end']:.0%}｜"
              f"Top波動 {row['top_vol']:.2f}（全市場 {univ_vol:.2f}）｜IC_end {row['ic_end']:+.3f}", flush=True)

    add_row("raw", D.nlargest(TOP_K, "pred_max_raw"), "pred_max_raw")
    add_row("term", D.nlargest(TOP_K, "pred_term_raw"), "pred_term_raw")
    add_row("vol_resid", D.dropna(subset=["vol_resid"]).nlargest(TOP_K, "vol_resid"), "vol_resid")
    add_row("volmom_resid", D.dropna(subset=["volmom_resid"]).nlargest(TOP_K, "volmom_resid"), "volmom_resid")

    pool = D.nlargest(LONG_POOL, "pred_max_raw")
    cutoff = pool["vol_anchor"].quantile(0.75)
    filtered = pool[pool["vol_anchor"] <= cutoff]
    top_vf = filtered.nlargest(TOP_K, "pred_max_raw") if len(filtered) >= TOP_K else pool.nlargest(TOP_K, "pred_max_raw")
    add_row("vol_filter", top_vf, "pred_max_raw")

    valid_f = D.dropna(subset=["foreign"])
    z1 = valid_f["pred_max_raw"].rank(pct=True)
    z2 = valid_f["foreign"].rank(pct=True)
    ens = (0.5 * z1 + 0.5 * z2)
    top_ens = valid_f.loc[ens.nlargest(TOP_K).index]
    add_row("ens_foreign", top_ens, "pred_max_raw")

    add_row("mom60", D.dropna(subset=["mom60"]).nlargest(TOP_K, "mom60"), "mom60")
    add_row("foreign", valid_f.nlargest(TOP_K, "foreign"), "foreign")
    add_row("random", D.nlargest(TOP_K, "random"), "random")

    return rows


if __name__ == "__main__":
    all_rows = []
    for idx, a in enumerate(ANCHORS):
        seed_base = SEED_BASE + idx * 10000
        D = train_anchor(a, seed_base)
        all_rows += evaluate(D, a)
        pd.DataFrame(all_rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n=== ALL DONE === 已存檔：{RESULT_CSV}")

    res = pd.DataFrame(all_rows)
    piv = res.pivot_table(index="anchor", columns="variant", values="excess_end")
    order = [c for c in ["random", "mom60", "foreign", "raw", "term", "vol_resid",
                          "volmom_resid", "vol_filter", "ens_foreign"] if c in piv.columns]
    print("\n並排比較（超額報酬 excess_end，%）")
    print((piv[order] * 100).round(2).to_string())
    print("\n平均（%）：")
    print((piv[order].mean() * 100).round(2).to_string())
