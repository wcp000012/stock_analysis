"""
財務特徵下載，存入 chip.db 的 monthly_revenue / quarterly_financials：

  1. 月營收 → TWSE + TPEx OpenAPI 整批下載。免金鑰、無請求配額，但官方只提供
     「最新一期」快照，沒有歷史區間查詢，所以做法是每次執行都 upsert 一次
     最新快照，長期執行下來自然累積出歷史序列。

  2. 資產負債表明細（存貨、合約負債、應收/應付帳款、預付款項）+
     現金流量表（營業現金流、資本支出）+ 損益表（營收/成本/毛利/營業利益/淨利）
     → 全部改用 FinMind 逐股下載（`TaiwanStockBalanceSheet` /
       `TaiwanStockCashFlowsStatement` / `TaiwanStockFinancialStatements`）。

     2026-09-10 修正：損益表原本用 TWSE/TPEx OpenAPI 整批抓「最新一期」，
     但發現 OpenAPI 的金額單位（仟元）跟 FinMind 資產負債表的金額單位（元）
     差了 1000 倍，導致每檔股票「最新一季」的 revenue_q/cogs/gross_profit
     等欄位跟同一檔股票自己的歷史資料（FinMind 來源）對不上——不是只有缺
     歷史的股票才有問題，連原本有完整 14 季歷史的 314 檔，其最新一季也被
     OpenAPI 覆蓋壞掉。改成損益表也統一用 FinMind，確保同一張表裡的數字
     全部同源、單位一致。FinMind 免費版有請求配額限制，用 `income_fetched`
     表記錄每檔股票上次抓過的時間，80 天內不重抓（比照資產負債表的做法）。
"""
import requests
import sqlite3
import pandas as pd
import time
import glob
import os
from threading import Semaphore, Lock

DB_PATH      = "chip.db"
PARQUET_GLOB = "parquet_data/*.parquet"
SLEEP_SEC    = 0.35   # FinMind free tier: ~2 req/sec 安全起見放慢
MAX_WORKERS  = 4
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-analysis/1.0)"}


def _load_finmind_token():
    try:
        with open("APIKey.txt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("FinMind APIKey"):
                    return line.split("FinMind APIKey", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


FINMIND_TOKEN = _load_finmind_token()

TWSE_BASE = "https://openapi.twse.com.tw/v1/opendata"
TPEX_BASE = "https://www.tpex.org.tw/openapi/v1"

# ── 欄位對應 ─────────────────────────────────────────────────────────────────
# FinMind：只保留 TWSE/TPEx OpenAPI 沒有明細的資產負債表 + 現金流量表欄位
BS_FIELDS = {
    "Inventories":                   "inventory",
    "CurrentContractLiabilities":    "contract_liability",
    "AccountsReceivableNet":         "accounts_receivable",
    "AccountsPayable":               "accounts_payable",
    "Prepayments":                   "prepayments",
}
CF_FIELDS = {
    "CashFlowsFromOperatingActivities": "operating_cf",
    "PropertyAndPlantAndEquipment":     "capex",   # 負值 = 支出
}

# 損益表：改用 FinMind TaiwanStockFinancialStatements（跟資產負債表同源，單位一致）
IS_FIELDS_FINMIND = {
    "Revenue":          "revenue_q",
    "CostOfGoodsSold":  "cogs",
    "GrossProfit":      "gross_profit",
    "OperatingIncome":  "operating_income",
    "IncomeAfterTaxes": "net_income",
}

QF_ALL_COLS = (list(BS_FIELDS.values()) + list(CF_FIELDS.values())
               + list(IS_FIELDS_FINMIND.values()))


# ── DB 初始化 ────────────────────────────────────────────────────────────────
def init_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS monthly_revenue (
            Date    TEXT NOT NULL,
            Ticker  TEXT NOT NULL,
            revenue REAL,
            PRIMARY KEY (Date, Ticker)
        )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_mr_ticker ON monthly_revenue (Ticker, Date)")

    col_defs = ",\n            ".join(f"{c} REAL" for c in QF_ALL_COLS)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS quarterly_financials (
            Date    TEXT NOT NULL,
            Ticker  TEXT NOT NULL,
            {col_defs},
            PRIMARY KEY (Date, Ticker)
        )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_qf_ticker ON quarterly_financials (Ticker, Date)")

    # 記錄每檔股票上次用 FinMind 補損益表歷史的時間，80 天內不重抓
    con.execute("""
        CREATE TABLE IF NOT EXISTS income_fetched (
            Ticker     TEXT PRIMARY KEY,
            fetched_at TEXT
        )""")
    # 同上，資產負債表明細 + 現金流量表（2026-09-22 新增，修正舊版用季末日期誤判新鮮度的 bug）
    con.execute("""
        CREATE TABLE IF NOT EXISTS bs_cf_fetched (
            Ticker     TEXT PRIMARY KEY,
            fetched_at TEXT
        )""")
    con.commit()


# ── 共用小工具 ───────────────────────────────────────────────────────────────
def to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    if v in ("", "-", "N/A"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def roc_yyyymm_to_date(s):
    """'11507' (民國115年07月) → '2026-07-01'"""
    s = str(s or "").strip()
    if len(s) < 4:
        return None
    try:
        year  = int(s[:-2]) + 1911
        month = int(s[-2:])
    except ValueError:
        return None
    if not (1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}-01"


def fetch_bulk_json(base, dataset):
    """打 TWSE / TPEx OpenAPI，回傳 list[dict]（無金鑰、無配額，失敗重試 2 次）"""
    url = f"{base}/{dataset}"
    for attempt in range(1, 3):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] fetch {dataset} failed: {e}")
            else:
                time.sleep(1.5)
    return []


# ── Phase 1：TWSE/TPEx OpenAPI 整批（月營收 + 損益表基本項）────────────────────
# 2026-10-19 發現：2026-07 ~ 2026-09 這三個月，TWSE/TPEx OpenAPI 回傳的「營業收入-當月營收」
# 突然全市場性地變成原本的 1/1000（1072/1082 檔同時出現，7→8→9 月彼此之間比例又是正常的，
# 代表是同一批資料被套上錯誤單位，不是個別股票的真實營收暴跌）。原因不明（懷疑官方資料格式
# 曾經短暫變動），已經用 chip_backup_before_revenue_fix_20261019.db 備份後手動校正回來。
# 這裡加一個防線：新抓到的值如果跟資料庫裡「這檔股票、上一期」的值差了 50 倍以上，視為可疑，
# 記錄下來但不寫入，避免同一種問題再次悄悄污染 rev_yoy / rev_mom 特徵。
SUSPICIOUS_JUMP_RATIO = 50


def update_monthly_revenue_bulk(con, valid_tickers):
    rows = []
    for base, dataset in [(TWSE_BASE, "t187ap05_L"), (TPEX_BASE, "mopsfin_t187ap05_O")]:
        for r in fetch_bulk_json(base, dataset):
            code = r.get("公司代號")
            if code not in valid_tickers:
                continue
            date = roc_yyyymm_to_date(r.get("資料年月"))
            rev  = to_float(r.get("營業收入-當月營收"))
            if date is None or rev is None:
                continue
            rows.append((date, code, rev))

    if not rows:
        return 0

    # 可疑值檢查：跟這檔股票資料庫裡最近一筆已存在的值比，差超過 SUSPICIOUS_JUMP_RATIO 倍
    # （無論暴增或暴跌）就先擋下來，印出來讓人決定，不要默默寫進去。
    clean_rows, suspicious = [], []
    for date, code, rev in rows:
        last = con.execute(
            "SELECT revenue FROM monthly_revenue WHERE Ticker=? AND Date<? AND revenue IS NOT NULL "
            "ORDER BY Date DESC LIMIT 1", (code, date)
        ).fetchone()
        if last and last[0] not in (0, None) and rev not in (0, None):
            ratio = rev / last[0]
            if ratio < 1 / SUSPICIOUS_JUMP_RATIO or ratio > SUSPICIOUS_JUMP_RATIO:
                suspicious.append((date, code, rev, last[0], ratio))
                continue
        clean_rows.append((date, code, rev))

    if suspicious:
        print(f"  [可疑月營收，已跳過未寫入] {len(suspicious)} 筆（跟上一期差超過 {SUSPICIOUS_JUMP_RATIO} 倍）：")
        for date, code, rev, last, ratio in suspicious[:20]:
            print(f"    {code} {date}: 新值={rev:,.0f} 上期={last:,.0f} 比值={ratio:.4f}")
        if len(suspicious) > 20:
            print(f"    ...還有 {len(suspicious) - 20} 筆")

    if not clean_rows:
        return 0
    con.executemany("""
        INSERT INTO monthly_revenue (Date, Ticker, revenue)
        VALUES (?, ?, ?)
        ON CONFLICT(Date, Ticker) DO UPDATE SET revenue=excluded.revenue
    """, clean_rows)
    con.commit()
    return len(clean_rows)


# ── Phase 2：FinMind 逐股（資產負債表明細 + 現金流量表 + 損益表）───────────────
_semaphore = Semaphore(MAX_WORKERS)

# 全域配額冷卻狀態：一旦任何一個 worker 被 FinMind 判定為超過配額（429/403），
# 就記下要等到什麼時候才能再打，之後所有 worker 在發送前都先檢查、共同等待，
# 而不是各自傻傻重試浪費時間。
_cooldown_lock = Lock()
_cooldown_until = 0.0


def _wait_for_cooldown():
    while True:
        with _cooldown_lock:
            remaining = _cooldown_until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5))


def _set_cooldown(seconds):
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, time.time() + seconds)


def finmind_get(dataset, stock_id, start="2023-01-01", end=None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    params = {"dataset": dataset, "data_id": stock_id,
              "start_date": start, "end_date": end}
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN
    with _semaphore:
        for attempt in range(1, 4):
            _wait_for_cooldown()
            try:
                resp = requests.get(
                    FINMIND_BASE,
                    params=params,
                    timeout=20
                )
                j = resp.json()
                if j.get("msg") == "success":
                    return pd.DataFrame(j["data"]) if j["data"] else pd.DataFrame()
                if resp.status_code in (429, 403):
                    retry_after = j.get("retry_after")
                    wait_sec = float(retry_after) + 5 if retry_after else 30 * attempt
                    print(f"  [配額暫停] {dataset} status={resp.status_code} "
                          f"msg={j.get('msg')!r} → 等待 {wait_sec:.0f} 秒", flush=True)
                    _set_cooldown(wait_sec)
                    continue
            except Exception:
                time.sleep(SLEEP_SEC * attempt * 3)
        return pd.DataFrame()


def pivot_quarterly(df, field_map):
    if df.empty:
        return pd.DataFrame()
    df = df[df["type"].isin(field_map)].copy()
    if df.empty:
        return pd.DataFrame()
    df["field"] = df["type"].map(field_map)
    df["date"]  = df["date"].astype(str).str[:10]
    df = df[~df["type"].str.endswith("_per")]
    pivot = df.pivot_table(index="date", columns="field", values="value", aggfunc="first")
    return pivot.reset_index().rename(columns={"date": "Date"})


def upsert_quarterly(con, stock_id, df):
    if df.empty:
        return 0
    all_cols = list(BS_FIELDS.values()) + list(CF_FIELDS.values())
    for c in all_cols:
        if c not in df.columns:
            df[c] = None

    rows = []
    for _, r in df.iterrows():
        vals = [str(r["Date"])[:10], stock_id] + [
            float(r[c]) if pd.notna(r.get(c)) else None
            for c in all_cols
        ]
        rows.append(tuple(vals))

    placeholders = ", ".join(["?"] * (2 + len(all_cols)))
    update_set   = ", ".join(f"{c}=excluded.{c}" for c in all_cols)
    col_names    = "Date, Ticker, " + ", ".join(all_cols)
    con.executemany(f"""
        INSERT INTO quarterly_financials ({col_names})
        VALUES ({placeholders})
        ON CONFLICT(Date, Ticker) DO UPDATE SET {update_set}
    """, rows)
    con.commit()
    return len(rows)


def download_quarterly(stock_id, con):
    # 跳過近期抓過 BS/CF 明細的股票（用「我們上次去抓的時間」判斷，不是「最新一季的季末日期」——
    # 2026-09-22 發現舊版寫法拿季末日期當基準，只要離下一季公布還有一段時間，
    # 每次執行都會被誤判成「太久沒更新」而重抓全部股票，白白浪費配額。跟 income_fetched 同一套邏輯）
    row = con.execute(
        "SELECT fetched_at FROM bs_cf_fetched WHERE Ticker=?", (stock_id,)
    ).fetchone()
    if row:
        days_ago = (pd.Timestamp.today() - pd.Timestamp(row[0])).days
        if days_ago < 80:
            return 0

    dfs = []
    for dataset, field_map in [
        ("TaiwanStockBalanceSheet",       BS_FIELDS),
        ("TaiwanStockCashFlowsStatement", CF_FIELDS),
    ]:
        raw = finmind_get(dataset, stock_id)
        time.sleep(SLEEP_SEC)
        dfs.append(pivot_quarterly(raw, field_map))

    merged = None
    for d in dfs:
        if d.empty:
            continue
        merged = d if merged is None else pd.merge(merged, d, on="Date", how="outer")

    con.execute(
        "INSERT OR REPLACE INTO bs_cf_fetched (Ticker, fetched_at) VALUES (?, ?)",
        (stock_id, pd.Timestamp.today().strftime("%Y-%m-%d"))
    )
    con.commit()

    if merged is None or merged.empty:
        return 0
    return upsert_quarterly(con, stock_id, merged)


# 2026-10-27 發現：2026-09-10 損益表來源切換成 FinMind 那次，大量股票（~300 檔）
# 最新一季 revenue_q 被寫成只有真實值 1/500 左右（例如 3037 寫成 8033 萬，實際 410.7 億），
# 原因不明（疑似當時 FinMind 回應或解析過程出過短暫問題），但完全沒被發現——因為
# download_income_history() 不管資料對不對，都無條件把 income_fetched 標記成完成，
# 80 天內不會重抓。這裡加一個跟 SUSPICIOUS_JUMP_RATIO（月營收）同樣精神的交叉驗證：
# 用獨立來源（TWSE/TPEx 月營收，Phase 1 抓的）驗證同一季的 revenue_q 量級對不對，
# 不對就整季跳過不寫入（寧可留空值等下次重抓，也不要寫入會騙過下游模型的錯誤數字）。
INCOME_SUSPICIOUS_RATIO = 3


def _quarter_month_starts(qdate):
    q = pd.Timestamp(qdate)
    return [(q - pd.DateOffset(months=i)).replace(day=1).strftime("%Y-%m-%d") for i in range(3)]


def _revenue_sanity_check(con, stock_id, date, revenue_q):
    if revenue_q in (None, 0):
        return True
    months = _quarter_month_starts(date)
    placeholders = ",".join("?" * len(months))
    row = con.execute(
        f"SELECT SUM(revenue) FROM monthly_revenue WHERE Ticker=? AND Date IN ({placeholders})",
        [stock_id] + months
    ).fetchone()
    msum = row[0] if row else None
    if not msum:
        return True  # 沒有月營收可比對（例如金融股本來就不公告月營收），放行
    ratio = revenue_q / msum
    if ratio < 1 / INCOME_SUSPICIOUS_RATIO or ratio > INCOME_SUSPICIOUS_RATIO:
        print(f"  [可疑損益表，整季跳過未寫入] {stock_id} {date}: revenue_q={revenue_q:,.0f} "
              f"同期月營收合計={msum:,.0f} 比值={ratio:.4f}")
        return False
    return True


def upsert_income(con, stock_id, df):
    if df.empty:
        return 0, 0
    cols = list(IS_FIELDS_FINMIND.values())
    for c in cols:
        if c not in df.columns:
            df[c] = None

    rows, skipped = [], 0
    for _, r in df.iterrows():
        rev_q = float(r["revenue_q"]) if pd.notna(r.get("revenue_q")) else None
        if not _revenue_sanity_check(con, stock_id, r["Date"], rev_q):
            skipped += 1
            continue
        vals = [str(r["Date"])[:10], stock_id] + [
            float(r[c]) if pd.notna(r.get(c)) else None
            for c in cols
        ]
        rows.append(tuple(vals))

    if not rows:
        return 0, skipped
    placeholders = ", ".join(["?"] * (2 + len(cols)))
    update_set   = ", ".join(f"{c}=excluded.{c}" for c in cols)
    col_names    = "Date, Ticker, " + ", ".join(cols)
    con.executemany(f"""
        INSERT INTO quarterly_financials ({col_names})
        VALUES ({placeholders})
        ON CONFLICT(Date, Ticker) DO UPDATE SET {update_set}
    """, rows)
    con.commit()
    return len(rows), skipped


def download_income_history(stock_id, con):
    # 跳過 80 天內已用 FinMind 補過損益表歷史的股票
    row = con.execute(
        "SELECT fetched_at FROM income_fetched WHERE Ticker=?", (stock_id,)
    ).fetchone()
    if row:
        days_ago = (pd.Timestamp.today() - pd.Timestamp(row[0])).days
        if days_ago < 80:
            return 0

    raw = finmind_get("TaiwanStockFinancialStatements", stock_id)
    time.sleep(SLEEP_SEC)
    pivoted = pivot_quarterly(raw, IS_FIELDS_FINMIND)
    n, skipped = upsert_income(con, stock_id, pivoted)
    if skipped:
        print(f"  [{stock_id}] 損益表寫入 {n} 季，跳過 {skipped} 季可疑資料")

    con.execute(
        "INSERT OR REPLACE INTO income_fetched (Ticker, fetched_at) VALUES (?, ?)",
        (stock_id, pd.Timestamp.today().strftime("%Y-%m-%d"))
    )
    con.commit()
    return n


# ── 主流程 ───────────────────────────────────────────────────────────────────
def get_stock_list():
    files = glob.glob(PARQUET_GLOB)
    if files:
        return [os.path.basename(f).replace(".parquet", "").replace(".TW", "")
                for f in files]
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT DISTINCT Ticker FROM chip_daily WHERE Ticker GLOB '[0-9][0-9][0-9][0-9]'"
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def run_download(stock_ids):
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_tables(con)
    valid_tickers = set(stock_ids)
    total = len(stock_ids)

    print(f"Stocks to process: {total}")
    print("[Phase 1] 月營收 — TWSE/TPEx OpenAPI 整批（無配額限制）...")
    n_mr = update_monthly_revenue_bulk(con, valid_tickers)
    print(f"  monthly_revenue upserted: {n_mr} rows")

    print("\n[Phase 2] 資產負債表明細 + 現金流量表 + 損益表 — FinMind 逐股（配額限制，只補缺的）...")
    done_qf, done_is = 0, 0
    for i, sid in enumerate(stock_ids, 1):
        done_qf += download_quarterly(sid, con)
        done_is += download_income_history(sid, con)
        if i % 50 == 0:
            print(f"  {i}/{total}  BS/CF rows inserted: {done_qf}，損益表 rows inserted: {done_is}")

    con.close()
    print(f"\n完成：monthly_revenue +{n_mr}，BS/CF 明細 +{done_qf}，損益表 +{done_is} rows")


if __name__ == "__main__":
    stock_ids = get_stock_list()
    print(f"Found {len(stock_ids)} stocks from parquet_data/")
    run_download(stock_ids)
