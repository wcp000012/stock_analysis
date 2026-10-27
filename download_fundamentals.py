"""
從 TWSE BWIBBU_d 下載每日本益比、殖利率、股價淨值比，
存入 chip.db 的 fundamental_daily 表。
"""
import requests
import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta

DB_PATH = "chip.db"
SLEEP_SEC = 0.8
TRAIN_YEARS = 3


def init_fundamental_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_daily (
            Date   DATE NOT NULL,
            Ticker TEXT NOT NULL,
            pe_ratio  REAL,
            pb_ratio  REAL,
            div_yield REAL,
            PRIMARY KEY (Date, Ticker)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_fund_ticker_date ON fundamental_daily (Ticker, Date)")
    con.commit()


def fetch_bwibbu(date_str: str, max_retry: int = 3) -> pd.DataFrame:
    """
    抓單日全市場本益比/殖利率/股價淨值比。
    回傳 DataFrame（index=Ticker），無資料則回傳空 DataFrame。
    """
    url = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    for attempt in range(1, max_retry + 1):
        try:
            j = requests.get(url, timeout=20).json()
        except Exception as e:
            time.sleep(SLEEP_SEC * attempt)
            continue

        if j.get("stat") != "OK":
            return pd.DataFrame()   # 非交易日

        cols = j.get("fields", [])
        data = j.get("data", [])
        if not cols or not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=cols)
        df["證券代號"] = df["證券代號"].astype(str).str.strip()
        df = df[df["證券代號"].str.match(r"^\d{4}$")]  # 只留 4 位數股票

        def to_float(s):
            try:
                v = float(str(s).replace(",", "").strip())
                return v if v != 0 else None
            except Exception:
                return None

        df["pe_ratio"]  = df["本益比"].apply(to_float)
        df["pb_ratio"]  = df["股價淨值比"].apply(to_float)
        df["div_yield"] = df["殖利率(%)"].apply(to_float)

        return df[["證券代號", "pe_ratio", "pb_ratio", "div_yield"]].rename(
            columns={"證券代號": "Ticker"}
        ).set_index("Ticker")

    return pd.DataFrame()


def upsert_fundamental(con, df: pd.DataFrame, trade_date: str):
    if df is None or df.empty:
        return
    df = df.reset_index()
    df["Date"] = trade_date
    rows = [
        (r["Date"], r["Ticker"],
         None if pd.isna(r["pe_ratio"])  else r["pe_ratio"],
         None if pd.isna(r["pb_ratio"])  else r["pb_ratio"],
         None if pd.isna(r["div_yield"]) else r["div_yield"])
        for _, r in df.iterrows()
    ]
    con.executemany("""
        INSERT INTO fundamental_daily (Date, Ticker, pe_ratio, pb_ratio, div_yield)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(Date, Ticker) DO UPDATE SET
            pe_ratio=excluded.pe_ratio,
            pb_ratio=excluded.pb_ratio,
            div_yield=excluded.div_yield
    """, rows)
    con.commit()


def daterange(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    while d <= end_d:
        yield d.isoformat()
        d += timedelta(days=1)


if __name__ == "__main__":
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - pd.DateOffset(years=TRAIN_YEARS)).strftime("%Y-%m-%d")
    print(f"Downloading fundamentals: {start_date} -> {end_date}")

    con = sqlite3.connect(DB_PATH)
    init_fundamental_table(con)

    skipped, saved, failed = 0, 0, 0
    for ds in daterange(start_date, end_date):
        # 跳過已入庫的日期
        exists = pd.read_sql_query(
            "SELECT 1 FROM fundamental_daily WHERE Date=? LIMIT 1", con, params=(ds,)
        )
        if not exists.empty:
            skipped += 1
            continue

        df = fetch_bwibbu(ds.replace("-", ""))
        if df.empty:
            print(f"[SKIP] {ds} 非交易日或無資料")
        else:
            upsert_fundamental(con, df, ds)
            print(f"[OK]   {ds}  rows={len(df)}")
            saved += 1
        time.sleep(SLEEP_SEC)

    con.close()
    print(f"\n完成：新增 {saved} 天，略過 {skipped} 天（已入庫）")
