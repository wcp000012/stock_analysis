# 這區程式碼是用來下載籌碼資料到sqllite方便之後跟技術資料合併
import requests
import io
import re
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
#這段程式碼是會成功保留適當的股票代號(用JASON)，並依此merge融資融卷和外資買賣超數據
def fetch_t86_json(date_str: str, max_retry: int = 3, sleep_sec: float = 0.8) -> pd.DataFrame:
    # 建議使用 RWD JSON 介面（UTF-8）
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALLBUT0999"

    # 強化抽取：三大法人在「真交易日」理應有資料，整天缺多半是暫時性故障。
    # 因此在這裡自帶重試 —— 拿到 stat=OK 且有資料才往下解析；
    # 重試用盡仍拿不到才回傳空 DataFrame（交給呼叫端把法人欄位存成 NULL）。
    cols, data = [], []
    last_state = None
    for attempt in range(1, max_retry + 1):
        try:
            j = requests.get(url, timeout=30).json()
        except Exception as e:
            last_state = f"網路錯誤：{e}"
            time.sleep(sleep_sec * attempt)
            continue

        stat = j.get("stat")
        cols = j.get("fields", [])
        data = j.get("data", [])
        if stat == "OK" and cols and data:
            break  # 成功拿到資料

        last_state = f"stat={stat!r}, fields={len(cols)}, data={len(data)}"
        time.sleep(sleep_sec * attempt)
    else:
        # 重試用盡：可能是該日 T86 真的無資料，回傳空表，法人欄位將存為 NULL
        print(f"  [T86 WARN] {date_str} 重試 {max_retry} 次仍無資料（{last_state}），法人欄位留空")
        return pd.DataFrame()

    if not cols or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=cols)

    # 清理欄位
    df["證券代號"] = df["證券代號"].astype(str).str.strip()
    df = df[~df["證券代號"].isin(["合計", "總計"])]

    # 需要的欄位盡量彈性處理（不同日期，欄名可能有差）
    # 1) 外資兩欄分項 → 相加
    col_fx_net = None
    fx_net_no_prop = "外陸資買賣超股數(不含外資自營商)"
    fx_prop_net   = "外資自營商買賣超股數"
    fx_buy_no_prop = "外陸資買進股數(不含外資自營商)"
    fx_sell_no_prop= "外陸資賣出股數(不含外資自營商)"
    fx_prop_buy    = "外資自營商買進股數"
    fx_prop_sell   = "外資自營商賣出股數"
    #print(df.columns)
    # 轉數字（把千分位逗號拿掉）
    def to_num(s):
        return pd.to_numeric(pd.Series(s).astype(str).str.replace(',', '', regex=False), errors='coerce')

    num_cands = [c for c in df.columns if any(k in c for k in ["買進股數", "賣出股數", "買賣超股數"])]
    for c in num_cands:
        df[c] = to_num(df[c])

    if fx_net_no_prop in df.columns and fx_prop_net in df.columns:
        df["外資買賣超股數"] = df[fx_net_no_prop] + df[fx_prop_net]
        col_fx_net = "外資買賣超股數"
    elif "外資買賣超股數" in df.columns:
        # 有些日期可能直接提供彙總欄
        col_fx_net = "外資買賣超股數"
    elif all(c in df.columns for c in [fx_buy_no_prop, fx_sell_no_prop, fx_prop_buy, fx_prop_sell]):
        # 從買進/賣出去推回淨額
        df["外資買賣超股數"] = (df[fx_buy_no_prop] - df[fx_sell_no_prop] +
                            df[fx_prop_buy] - df[fx_prop_sell])
        col_fx_net = "外資買賣超股數"
    else:
        # 最少也保留個別欄位供你後處理
        pass

    # 最終只留你要的幾欄
    keep = ["證券代號", col_fx_net, "投信買賣超股數", "自營商買賣超股數"]
    keep = [c for c in keep if c and c in df.columns]
    df = df[keep].copy().set_index("證券代號")
    
    return df


def fetch_qfiis_json(date_str: str, max_retry: int = 3, sleep_sec: float = 0.8) -> pd.DataFrame:
    """外資及陸資持股比率（MI_QFIIS，免金鑰無配額），回傳 index=證券代號，欄位 foreign_holding_pct。"""
    url = f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?response=json&date={date_str}&selectType=ALLBUT0999"

    cols, data = [], []
    last_state = None
    for attempt in range(1, max_retry + 1):
        try:
            j = requests.get(url, timeout=30).json()
        except Exception as e:
            last_state = f"網路錯誤：{e}"
            time.sleep(sleep_sec * attempt)
            continue

        stat = j.get("stat")
        cols = j.get("fields", [])
        data = j.get("data", [])
        if stat == "OK" and cols and data:
            break
        last_state = f"stat={stat!r}, fields={len(cols)}, data={len(data)}"
        time.sleep(sleep_sec * attempt)
    else:
        print(f"  [QFIIS WARN] {date_str} 重試 {max_retry} 次仍無資料（{last_state}），外資持股比率留空")
        return pd.DataFrame()

    if not cols or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=cols)
    df["證券代號"] = df["證券代號"].astype(str).str.strip()
    df = df[df["證券代號"].str.match(r"^\d{4}$")]  # 只留 4 位數個股，排除 ETF/受益憑證等

    df["foreign_holding_pct"] = pd.to_numeric(
        df["全體外資及陸資持股比率"].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    return df[["證券代號", "foreign_holding_pct"]].set_index("證券代號")


def download_chip_data(date):
    """
    回傳值語意（讓呼叫端能區分三種狀況）：
      - DataFrame ：成功抓到當日籌碼資料
      - None      ：確認的非交易日／當日無資料（不需重試）
      - 拋出例外   ：真正的下載／解析失敗（網路錯誤交給外層重試，其餘記為失敗）
    """
    date_str = date.replace('-', '')

    # ---------- 【融資融券 MI_MARGN】 ----------
    # 修正 URL：改用 & 而不是 &amp;
    twse_margin_url = (
        "https://www.twse.com.tw/exchangeReport/MI_MARGN"
        f"?response=json&date={date_str}&selectType=ALL"
    )
    # 網路錯誤（逾時／連線失敗）不在這裡吞掉，讓它往上拋給外層重試迴圈
    j = requests.get(twse_margin_url, timeout=30).json()

    # 非交易日／無資料：TWSE 會回 stat != "OK"（例：「很抱歉，沒有符合條件的資料」）
    # 此時直接視為當日無資料，回傳 None，外層就不會重試也不會誤判為失敗
    if j.get("stat") != "OK":
        return None

    # 走到這裡代表 stat == OK，應有資料；若結構不如預期則視為真正的解析失敗（拋出）
    tables = j.get("tables", [])
    if len(tables) < 2:
        raise ValueError(f"MI_MARGN 結構異常（stat=OK 但 tables 數量={len(tables)}）")
    target_table = tables[1]  # 第二個表格（融資融券彙總）

    cols = target_table.get("fields", [])
    data = target_table.get("data", [])

    idx_margin_buy = cols.index("買進")
    idx_margin_sell = cols.index("賣出")
    idx_short_buy = cols.index("買進", idx_margin_buy + 1)
    idx_short_sell = cols.index("賣出", idx_margin_sell + 1)

    need_idx = [cols.index("代號"), idx_margin_buy, idx_margin_sell, idx_short_buy, idx_short_sell]
    margin_data = pd.DataFrame([[row[i] for i in need_idx] for row in data],
                      columns=["證券代號", "margin_buy", "margin_sell", "short_buy", "short_sell"])

    margin_data['證券代號'] = margin_data['證券代號'].astype(str).str.strip()
    margin_data = margin_data[
        margin_data['證券代號'].notna() &
        ~margin_data['證券代號'].isin(['合計', '總計'])
    ]

    for c in ['margin_buy', 'margin_sell', 'short_buy', 'short_sell']:
        margin_data[c] = pd.to_numeric(
            margin_data[c].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
    margin_data.set_index('證券代號', inplace=True)

    if margin_data.empty:
        # stat=OK 卻沒有任何有效個股資料，視為解析異常
        raise ValueError("MI_MARGN stat=OK 但解析後無任何個股資料")

    # ---------- 【三大法人 T86】（改走 JSON） ----------
    # T86 偶爾單獨缺資料：不讓它中斷整天，法人欄位留空即可（後續 upsert 會補成 NULL）
    try:
        twse_data = fetch_t86_json(date_str)  # ← 用上面那段 JSON 版本
    except Exception as e:
        print(f"{date}: T86 抓取失敗，法人欄位將留空：{e}")
        twse_data = pd.DataFrame()

    # ---------- 【外資及陸資持股比率 MI_QFIIS】 ----------
    try:
        qfiis_data = fetch_qfiis_json(date_str)
    except Exception as e:
        print(f"{date}: QFIIS 抓取失敗，外資持股比率將留空：{e}")
        qfiis_data = pd.DataFrame()

    # ---------- 合併 ----------
    out = margin_data.join(twse_data, how='left').join(qfiis_data, how='left')
    out = out.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    print(f"Download data of {date} successfully. shape={out.shape}")
    return out
    

def upsert_chip(con, df: pd.DataFrame, trade_date, ticker_col='證券代號'):
    
    """
    將籌碼 DataFrame 寫入 SQLite 的 chip_daily。
    - 支援 index 是「證券代號」的情況
    - 自動把中文欄位名稱對應到資料庫英文字段
    - 自動補齊缺欄位、轉數值型別
    - trade_date: 可傳入 datetime/date 或 'YYYY-MM-DD' 字串
    """

    if df is None or df.empty:
        return

    # 若 index 名稱是「證券代號」，先轉回欄位
    if df.index.name == ticker_col and ticker_col not in df.columns:
        df = df.reset_index()

    # 欄位對應：把資料來源的欄位名稱映射到 DB 欄位
    rename_map = {
        ticker_col: 'Ticker',
        'margin_buy': 'margin_buy',
        'margin_sell': 'margin_sell',
        'short_buy': 'short_buy',
        'short_sell': 'short_sell',
        '外資買賣超股數': 'foreign_net',
        '投信買賣超股數': 'invest_net',
        '自營商買賣超股數': 'dealer_net',
        'foreign_holding_pct': 'foreign_holding_pct',
    }
    df = df.rename(columns=rename_map)

    # 加入 Date 欄位（全表同一天）
    df['Date'] = pd.to_datetime(trade_date).date()
    
    # 需要的欄位
    required_cols = [
        'Date', 'Ticker',
        'margin_buy', 'margin_sell',
        'short_buy', 'short_sell',
        'foreign_net', 'invest_net', 'dealer_net',
        'foreign_holding_pct',
    ]

    # 補齊缺漏欄位（例如某天有融資融券、但三大法人 T86 個股層級缺資料時，
    # foreign_net/invest_net/dealer_net 可能不存在）。補成 NA 存為 NULL，
    # 才不會因為 df[required_cols] KeyError 把整個交易日的資料丟掉。
    # 注意：缺值一律存 NULL（忠實表示「未知」），不補 0，避免對籌碼資料注入假訊號；
    #       補值留到訓練前的特徵工程階段再依語意處理。
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # 只保留需要的欄位順序，同時可以確保dataframe的順序有和sqllite insert的順序一樣
    df = df[required_cols]


    # 組 rows 時把所有缺值（np.nan / pd.NA）統一轉成 Python None，
    # 否則 pandas 的 NAType 無法被 sqlite3 綁定（會 raise "type 'NAType' is not supported"），
    # 而 None 才會正確存成 SQL NULL。
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO chip_daily (Date, Ticker, margin_buy, margin_sell, short_buy, short_sell, foreign_net, invest_net, dealer_net, foreign_holding_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Date, Ticker) DO UPDATE SET
            margin_buy=excluded.margin_buy,
            margin_sell=excluded.margin_sell,
            short_buy=excluded.short_buy,
            short_sell=excluded.short_sell,
            foreign_net=excluded.foreign_net,
            invest_net=excluded.invest_net,
            dealer_net=excluded.dealer_net,
            foreign_holding_pct=excluded.foreign_holding_pct;
        """
    , rows)
    con.commit()

def daterange(start_date: str, end_date: str):
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)

def init_sqlite(db_path='chip.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chip_daily (
            Date DATE NOT NULL,
            Ticker TEXT NOT NULL,
            margin_buy REAL,
            margin_sell REAL,
            short_buy REAL,
            short_sell REAL,
            foreign_net REAL,
            invest_net REAL,
            dealer_net REAL,
            foreign_holding_pct REAL,
            PRIMARY KEY (Date, Ticker)
        );
        """
    )
    # 相容舊資料庫：欄位是後來才加的，舊的 chip.db 需要補上
    existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(chip_daily)").fetchall()]
    if 'foreign_holding_pct' not in existing_cols:
        cur.execute("ALTER TABLE chip_daily ADD COLUMN foreign_holding_pct REAL")
    cur.execute('CREATE INDEX IF NOT EXISTS idx_chip_ticker_date ON chip_daily (Ticker, Date);')
    con.commit()
    return con


def backfill_foreign_holding_pct(start_date: str, end_date: str, db_path='chip.db', sleep_sec=0.8, max_retry=3):
    """幫已經存在 chip_daily 裡、但還沒有 foreign_holding_pct 的歷史交易日補上這個欄位。"""
    con = init_sqlite(db_path)
    dates_needing = pd.read_sql_query(
        "SELECT DISTINCT Date FROM chip_daily WHERE Date BETWEEN ? AND ? "
        "AND foreign_holding_pct IS NULL ORDER BY Date",
        con, params=(start_date, end_date)
    )['Date'].tolist()
    print(f"共 {len(dates_needing)} 個交易日需要補外資持股比率")

    cur = con.cursor()
    for ds in dates_needing:
        ds_str = str(ds).replace('-', '')
        df = fetch_qfiis_json(ds_str, max_retry=max_retry, sleep_sec=sleep_sec)
        if df.empty:
            print(f'[SKIP] {ds} 無資料')
            time.sleep(sleep_sec)
            continue
        rows = [
            (float(v) if pd.notna(v) else None, str(ds), ticker)
            for ticker, v in df['foreign_holding_pct'].items()
        ]
        cur.executemany(
            "UPDATE chip_daily SET foreign_holding_pct=? WHERE Date=? AND Ticker=?",
            rows
        )
        con.commit()
        print(f'[OK] {ds} 更新 {len(rows)} 檔')
        time.sleep(sleep_sec)
    con.close()


def download_chip_range_to_sqlite(start_date: str, end_date: str, db_path='chip.db', sleep_sec=0.8, max_retry=3):
    con = init_sqlite(db_path)
    failed_dates = []   # 真正失敗（重試用盡仍失敗）的交易日，結尾彙總
    try:
        for ds in daterange(start_date, end_date):
            # 略過已入庫的日期（簡化：查是否存在任一筆）
            existed = pd.read_sql_query('SELECT 1 FROM chip_daily WHERE Date=? LIMIT 1', con, params=(ds,))
            if not existed.empty:
                continue

            handled = False      # 這天是否已妥善處理（成功入庫或確認非交易日）
            last_err = None
            for attempt in range(1, max_retry+1):
                try:
                    df = download_chip_data(ds)
                    if df is None:
                        # 確認的非交易日／當日無資料：不需重試，也不算失敗
                        print(f'[SKIP] {ds} 非交易日或當日無資料')
                    elif not df.empty:
                        upsert_chip(con, df, ds)
                        print(f'[OK] {ds} rows={len(df)}')
                    else:
                        print(f'[WARN] {ds} 回傳空資料，未入庫')
                    handled = True
                    break
                except (requests.exceptions.RequestException, HTTPError, URLError, TimeoutError) as nete:
                    # 網路類錯誤：值得重試
                    last_err = nete
                    print(f'[RETRY {attempt}/{max_retry}] {ds} 網路錯誤：{nete}')
                    time.sleep(sleep_sec * attempt)
                except Exception as e:
                    # 解析／結構類錯誤：重試通常無效，但仍記錄並結束重試
                    last_err = e
                    print(f'[RETRY {attempt}/{max_retry}] {ds} 解析錯誤：{e}')
                    time.sleep(sleep_sec * attempt)

            if not handled:
                # 重試用盡仍失敗 → 記錄下來，結尾統一回報（DB 沒這天，重跑會自動補）
                failed_dates.append((ds, repr(last_err)))
                print(f'[FAIL] {ds} 重試 {max_retry} 次仍失敗：{last_err}')
            time.sleep(sleep_sec)
    finally:
        con.close()

    # ---------- 結尾彙總：明確列出真正失敗、需要關注的交易日 ----------
    print('\n' + '=' * 50)
    if failed_dates:
        print(f'⚠ 有 {len(failed_dates)} 個日期重試後仍下載失敗（DB 未入庫，重跑本程式會自動補）：')
        for d, err in failed_dates:
            print(f'   {d}  {err}')
    else:
        print('✓ 全部日期皆已妥善處理，無重試後仍失敗的交易日。')
    print('=' * 50)
    return failed_dates
#download_chip_range_to_sqlite('2015-01-01','2024-12-31')
#download_chip_range_to_sqlite('2023-06-08','2026-06-26')

#download_chip_data('2017-04-10')
#download_chip_data('2024-12-31')


