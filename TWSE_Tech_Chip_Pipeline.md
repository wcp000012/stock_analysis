# 台股「技術＋籌碼」資料整合管線（Markdown 筆記）

> 作者：Chang-Pei（由 M365 Copilot 協作產出）  
> 版本：v1.0  
> 日期：2025-09-29

---

## 0) 這份筆記會帶你完成什麼？

- 從 **TWSE ISIN** 頁面取得 **上市/上櫃普通股清單**（正確排除權證、ETF、ETN 等）。
- 從 **TWSE CSV** 來源下載 **籌碼資料**：
  - **MI_MARGN**（融資融券）
  - **T86**（三大法人買賣超）
- 以 `Date + Ticker` 為鍵，將**籌碼**與 **yfinance 價格/技術指標**做 **left join**。
- 以 **SQLite**（或 Parquet）為儲存層，支援**增量下載、索引查詢、逐檔合併**，避免爆記憶體。
- 最後產出可供機器學習使用的特徵表（每檔股票一份，或合併為大表）。

> **為什麼要這樣做？**  
> TWSE 的 ISIN 與籌碼頁面屬於**人看為主**的 HTML/CSV，沒有乾淨的純 API 分類或過濾參數；因此推薦做法是**先下載→本地清洗→持久化**，然後在合併階段**只查需要的股票與日期範圍**，以降低時間與記憶體成本。

---

## 1) 環境需求

- Python 3.9+  
- 推薦套件：`pandas`, `requests`, `beautifulsoup4`（若要以 BS4 解析 ISIN），`yfinance`, `sqlite3`（標準庫）
- （選配）`pyarrow`/`fastparquet` 以寫入 Parquet

> **注意**：本文件中的程式碼段多半會連網，請在可連網的環境下執行。

---

## 2) 股票清單：只取「普通股」

TWSE 的 ISIN 頁面：
- `strMode=2`：**上市**（TWSE）
- `strMode=4`：**上櫃**（TPEx/櫃買）
- `strMode=5`：**興櫃**

這些頁面把**股票、ETF、ETN、權證**等商品**混放在同一張表**，用表內的**分類抬頭列**（例如「股票」「上市認購(售)權證」）來區隔。因此我們採用：

1. 下載整張表。
2. 偵測分類抬頭列，forward-fill 到每一列，產生 `category` 欄。
3. 過濾 `category == '股票'` 即可得到**普通股清單**。

```python
import re
import requests
import pandas as pd

def fetch_isin_table(str_mode=2):
    """抓取 TWSE ISIN 表（上市=2、上櫃=4、興櫃=5）並回傳整理後的 DataFrame。"""
    url = f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={str_mode}'
    res = requests.get(url, timeout=30)
    res.encoding = 'big5'  # TWSE ISIN 頁面編碼
    df = pd.read_html(res.text)[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:].copy()
    df = df.dropna(how='all')

    # 嘗試以欄名關鍵字對應
    col_isin = [c for c in df.columns if 'ISIN' in str(c)][0]
    col_list_date = [c for c in df.columns if ('上市' in str(c) and '日' in str(c))][0]
    col_name_code = [c for c in df.columns if '有價證券代號及名稱' in str(c)][0]

    # 分類抬頭列：常見為只有「有價證券代號及名稱」有值、其他欄位空白
    is_category_row = df[col_isin].isna() & df[col_list_date].isna() & df[col_name_code].notna()

    df.loc[is_category_row, 'category'] = df.loc[is_category_row, col_name_code]
    df['category'] = df['category'].ffill()

    # 去掉分類列，保留商品資料列
    df = df[~is_category_row].copy()

    # 拆代號與名稱（處理全形/半形空白）
    code_name = df[col_name_code].astype(str).str.split(r'\s+', n=1, expand=True)
    df['股票代號'] = code_name[0]
    df['股票名稱'] = code_name[1]

    # 只保留純四碼數字（排除權證等）
    df = df[df['股票代號'].str.match(r'^\d{4}$', na=False)].copy()

    # 市場別/產業別欄名不同日期可能略異，保留可用欄位
    keep_cols = ['股票代號', '股票名稱', 'category']
    for k in df.columns:
        if any(x in str(k) for x in ['市場別', '產業別', 'CFI']):
            keep_cols.append(k)

    return df[keep_cols].reset_index(drop=True)

# 取上市普通股
listed_common = fetch_isin_table(2)
listed_common = listed_common[listed_common['category'] == '股票']
listed_common = listed_common.drop_duplicates(subset=['股票代號']).reset_index(drop=True)

# 取上櫃普通股
otc_common = fetch_isin_table(4)
otc_common = otc_common[otc_common['category'] == '股票']
otc_common = otc_common.drop_duplicates(subset=['股票代號']).reset_index(drop=True)

print('上市普通股數量 =', len(listed_common))
print('上櫃普通股數量 =', len(otc_common))
```

> 註：過去你用正則 `r'\d{4}\s+'` 直接抓，會把**權證**（多數也是 4 位數代碼）一起撈進來，所以數量才會極大。

---

## 3) 籌碼資料：單日下載與清洗（MI_MARGN + T86）

這兩個報表常見特性：
- 編碼為 **Big5/Big5-HKSCS**。
- 會有**千分位逗號**、**合計/總計**列、及偶發的 **Excel 風格字串**（例如 `="2330"`）。
- 表頭位置偶爾變動，`skiprows` 不穩定，因此建議：
  1) 先抓原始文字，
  2) 自動偵測含「證券代號/代號」的標頭行，
  3) 再讀入為 DataFrame。

```python
import io
import re
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError

# 將 '="2330"' -> '2330'
def normalize_ticker(x: str) -> str:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r'^="?([0-9A-Z]+)"?$', r'', s)
    return s

# 偵測標頭行（尋找包含「證券代號」或「代號」的行）
def find_header_start_line(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines[:80]):
        if ('證券代號' in line) or ('代號' in line and '買進' in line):
            return i
    return None

# 強韌讀取 TWSE CSV
def robust_read_twse_csv(url: str, encoding='big5-hkscs'):
    try:
        return pd.read_csv(url, encoding=encoding, engine='python')
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode(encoding, errors='ignore')
        start = find_header_start_line(raw)
        if start is None:
            raise ValueError('找不到含「證券代號/代號」的標頭行，TWSE 報表格式可能變更。')
        text = "
".join(raw.splitlines()[start:])
        df = pd.read_csv(io.StringIO(text), engine='python')
        return df

# 下載單日籌碼（回傳 long format：Date, Ticker, 欄位們）
def download_chip_one_day(date_str: str) -> pd.DataFrame:
    dcompact = date_str.replace('-', '')

    # --- MI_MARGN 融資融券 ---
    try:
        url_m = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=csv&date={dcompact}&selectType=ALL"
        df_m = robust_read_twse_csv(url_m)
        # 欄位正規化
        # 可能欄名：['代號','買進','賣出','買進.1','賣出.1']
        colmap = {}
        for target in ['代號','買進','賣出','買進.1','賣出.1']:
            if target in df_m.columns:
                colmap[target] = target
            else:
                cands = [c for c in df_m.columns if target in str(c)]
                if cands:
                    colmap[target] = cands[0]
        req = ['代號','買進','賣出','買進.1','賣出.1']
        if not all(c in colmap for c in req):
            raise ValueError('MI_MARGN 欄位不齊')
        df_m = df_m[[colmap[c] for c in req]].copy()
        df_m.columns = req

        df_m['Ticker'] = df_m['代號'].map(normalize_ticker)
        df_m = df_m[df_m['Ticker'].notna() & ~df_m['Ticker'].isin(['合計','總計'])]
        for c in ['買進','賣出','買進.1','賣出.1']:
            df_m[c] = pd.to_numeric(df_m[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
        df_m.rename(columns={'買進':'margin_buy','賣出':'margin_sell','買進.1':'short_buy','賣出.1':'short_sell'}, inplace=True)
        df_m = df_m[['Ticker','margin_buy','margin_sell','short_buy','short_sell']]
    except Exception:
        df_m = pd.DataFrame(columns=['Ticker','margin_buy','margin_sell','short_buy','short_sell'])

    # --- T86 三大法人 ---
    try:
        url_t = f"https://www.twse.com.tw/fund/T86?response=csv&date={dcompact}&selectType=ALLBUT0999"
        df_t = robust_read_twse_csv(url_t)
        raw_cols = ['證券代號','外陸資買賣超股數(不含外資自營商)','外資自營商買賣超股數','投信買賣超股數','自營商買賣超股數']
        cmap = {}
        for target in raw_cols:
            if target in df_t.columns:
                cmap[target] = target
            else:
                cands = [c for c in df_t.columns if target in str(c)]
                if cands:
                    cmap[target] = cands[0]
        if len(cmap) < len(raw_cols):
            raise ValueError('T86 欄位不齊')
        df_t = df_t[[cmap[c] for c in raw_cols]].copy()
        df_t.columns = raw_cols

        df_t['Ticker'] = df_t['證券代號'].map(normalize_ticker)
        df_t = df_t[df_t['Ticker'].notna() & ~df_t['Ticker'].isin(['合計','總計'])]
        for c in raw_cols[1:]:
            df_t[c] = pd.to_numeric(df_t[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
        df_t['foreign_net'] = df_t['外陸資買賣超股數(不含外資自營商)'] + df_t['外資自營商買賣超股數']
        df_t.rename(columns={'投信買賣超股數':'invest_net','自營商買賣超股數':'dealer_net'}, inplace=True)
        df_t = df_t[['Ticker','foreign_net','invest_net','dealer_net']]
    except Exception:
        df_t = pd.DataFrame(columns=['Ticker','foreign_net','invest_net','dealer_net'])

    if df_m.empty and df_t.empty:
        return pd.DataFrame(columns=['Date','Ticker','margin_buy','margin_sell','short_buy','short_sell','foreign_net','invest_net','dealer_net'])

    out = pd.merge(df_m, df_t, on='Ticker', how='outer')
    out.insert(0, 'Date', pd.to_datetime(date_str))

    # dtype 壓縮
    for c in ['margin_buy','margin_sell','short_buy','short_sell','foreign_net','invest_net','dealer_net']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32')

    out['Ticker'] = out['Ticker'].astype(str)
    return out[['Date','Ticker','margin_buy','margin_sell','short_buy','short_sell','foreign_net','invest_net','dealer_net']]
```

---

## 4) SQLite 儲存（增量、UPSERT、索引）

我們以 `(Date, Ticker)` 做主鍵，並建立 `(Ticker, Date)` 索引，以利查詢單檔股票在日期區間內的籌碼資料。

```python
def init_sqlite(db_path='chip.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chip_daily (
            Date TEXT NOT NULL,
            Ticker TEXT NOT NULL,
            margin_buy REAL,
            margin_sell REAL,
            short_buy REAL,
            short_sell REAL,
            foreign_net REAL,
            invest_net REAL,
            dealer_net REAL,
            PRIMARY KEY (Date, Ticker)
        );
        """
    )
    cur.execute('CREATE INDEX IF NOT EXISTS idx_chip_ticker_date ON chip_daily (Ticker, Date);')
    con.commit()
    return con


def upsert_chip(con, df: pd.DataFrame):
    if df.empty:
        return
    rows = df[['Date','Ticker','margin_buy','margin_sell','short_buy','short_sell','foreign_net','invest_net','dealer_net']].values.tolist()
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO chip_daily (Date, Ticker, margin_buy, margin_sell, short_buy, short_sell, foreign_net, invest_net, dealer_net)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Date, Ticker) DO UPDATE SET
            margin_buy=excluded.margin_buy,
            margin_sell=excluded.margin_sell,
            short_buy=excluded.short_buy,
            short_sell=excluded.short_sell,
            foreign_net=excluded.foreign_net,
            invest_net=excluded.invest_net,
            dealer_net=excluded.dealer_net;
        """
    , rows)
    con.commit()


def query_chip(con, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
    SELECT Date, Ticker, margin_buy, margin_sell, short_buy, short_sell,
           foreign_net, invest_net, dealer_net
    FROM chip_daily
    WHERE Ticker = ? AND Date BETWEEN ? AND ?
    ORDER BY Date
    """
    df = pd.read_sql_query(q, con, params=(ticker, start_date, end_date), parse_dates=['Date'])
    return df
```

---

## 5) 批次下載籌碼（含重試與節流）

```python
def daterange(start_date: str, end_date: str):
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)


def download_chip_range_to_sqlite(start_date: str, end_date: str, db_path='chip.db', sleep_sec=0.8, max_retry=3):
    con = init_sqlite(db_path)
    try:
        for ds in daterange(start_date, end_date):
            # 略過已入庫的日期（簡化：查是否存在任一筆）
            existed = pd.read_sql_query('SELECT 1 FROM chip_daily WHERE Date=? LIMIT 1', con, params=(ds,))
            if not existed.empty:
                continue

            for attempt in range(1, max_retry+1):
                try:
                    df = download_chip_one_day(ds)
                    if not df.empty:
                        upsert_chip(con, df)
                        print(f'[OK] {ds} rows={len(df)}')
                    else:
                        print(f'[WARN] {ds} 無資料或非交易日')
                    break
                except (HTTPError, URLError, TimeoutError) as nete:
                    print(f'[RETRY {attempt}] {ds} 網路錯誤：{nete}')
                    time.sleep(sleep_sec * attempt)
                except Exception as e:
                    print(f'[FAIL] {ds} 解析失敗：{e}')
                    break
            time.sleep(sleep_sec)
    finally:
        con.close()
```

> 建議先跑：
>
> ```bash
> python -c "from your_module import download_chip_range_to_sqlite; download_chip_range_to_sqlite('2015-01-01','2024-12-31')"
> ```
>
> 可中斷、可重跑（已存在日期會被略過）。

---

## 6) 價格與技術指標（yfinance）

- 上市（TWSE）代碼：`{ticker}.TW`  
- 上櫃（TPEx）代碼：`{ticker}.TWO`

```python
import yfinance as yf

# 簡易 RSI 實作

def compute_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rsi = 100 - (100 / (1 + (ma_up / (ma_down + 1e-9))))
    return rsi


def download_price_features(ticker_num: str, market: str, start_date='2015-01-01', end_date='2024-12-31'):
    suffix = '.TW' if market == '上市' else '.TWO'
    t = f"{ticker_num}{suffix}"
    px = yf.download(t, start=start_date, end=end_date, progress=False)
    if px.empty:
        return None
    px = px.rename_axis('Date').reset_index()
    px['Ticker'] = ticker_num
    # 常用技術指標
    px['MA10'] = px['Close'].rolling(window=10).mean()
    px['MA20'] = px['Close'].rolling(window=20).mean()
    px['RSI14'] = compute_rsi(px['Close'], 14)

    # dtype 壓縮
    for c in ['Open','High','Low','Close','Adj Close','MA10','MA20','RSI14']:
        if c in px.columns:
            px[c] = pd.to_numeric(px[c], errors='coerce').astype('float32')
    if 'Volume' in px.columns:
        px['Volume'] = pd.to_numeric(px['Volume'], errors='coerce').astype('int32')

    return px
```

---

## 7) 合併：技術 × 籌碼（以股票為單位，省記憶體）

推薦流程：逐檔股票處理 → `query_chip()` 只撈該檔在日期區間內的資料 → `merge` on `['Date','Ticker']` → 寫檔。

```python
def merge_price_chip(ticker_num: str, market: str, start_date='2015-01-01', end_date='2024-12-31', db_path='chip.db'):
    # 價格與技術
    px = download_price_features(ticker_num, market, start_date, end_date)
    if px is None or px.empty:
        return None

    # 籌碼
    con = sqlite3.connect(db_path)
    chip = query_chip(con, ticker_num, start_date, end_date)
    con.close()

    # 整併
    df = pd.merge(px, chip, on=['Date','Ticker'], how='left')

    # 缺值處理（範例：以 0 補）
    for c in ['margin_buy','margin_sell','short_buy','short_sell','foreign_net','invest_net','dealer_net']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype('float32')

    # 節省記憶體
    df['Ticker'] = df['Ticker'].astype('category')
    return df
```

---

## 8) 記憶體與效能注意事項

- **逐檔處理**：避免把全市場同時載入記憶體。
- **型態壓縮**：價格/指標用 `float32`、成交量用 `int32`、`Ticker` 用 `category`。
- **索引**：SQLite 建立 `(Ticker, Date)` 索引，加速查詢。
- **Parquet 分割**（選配）：以年/月分割路徑存檔，查詢時只讀必要檔案。
- **非交易日**：以**價格日期**為準做 left join，籌碼缺值可視策略用 0 或保留 NaN。
- **節流與重試**：避免被官方限流；遇格式異動要容錯。

---

## 9) 標籤（Label）與訓練集切分（範例）

例如：用今日特徵預測**隔日**或 **N 日後**的報酬。

```python
def add_labels(df: pd.DataFrame, horizon=1):
    df = df.sort_values('Date').copy()
    df['ret_future'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    # 二元分類標籤：上漲=1、下跌=0（可視需求調整門檻）
    df['label_up'] = (df['ret_future'] > 0).astype('int8')
    return df
```

---

## 10) 一鍵示例：從建立資料庫 → 產生單檔特徵

```python
# 1) 先建立/補齊籌碼資料庫
# download_chip_range_to_sqlite('2015-01-01','2024-12-31', db_path='chip.db')

# 2) 取得股票清單（只要上市普通股）
listed = fetch_isin_table(2)
listed = listed[listed['category']=='股票'].drop_duplicates(subset=['股票代號'])

# 3) 示範：TSMC（2330）
tsdf = merge_price_chip('2330', market='上市', start_date='2015-01-01', end_date='2024-12-31', db_path='chip.db')
if tsdf is not None:
    tsdf = add_labels(tsdf, horizon=1)
    print(tsdf.head())
    # 可選：存檔
    # tsdf.to_parquet('features_2330.parquet', index=False)
```

---

## 11) 擴充建議

- **加入更多技術指標**：EMA、MACD、K/D、ATR、布林通道。
- **加入 TPEx（櫃買）籌碼來源**：本筆記示範的是 TWSE 的 MI_MARGN/T86；櫃買中心有對應檔案與 API，欄位需再正規化。
- **自動排程**：以 cron/Task Scheduler 每日收盤後更新昨日籌碼與技術資料。
- **DuckDB**：若偏好檔案型 OLAP，可把 SQLite 改成 DuckDB 並保留同樣的鍵值設計。
- **多工下載**：注意官方限流、善用 sleep 與重試機制。

---

## 12) FAQ

**Q1. 我能直接從伺服器只下載「普通股」嗎？**  
目前不行。ISIN 頁面沒有提供只回傳普通股的查詢參數；只能**整張表先抓下來**，再在本地按分類抬頭做過濾。

**Q2. 上市 vs 上櫃的 yfinance 代碼尾綴？**  
上市：`.TW`；上櫃：`.TWO`。請依 `市場別` 決定尾綴。

**Q3. ETN 是什麼？跟 ETF 有何不同？**  
ETN 是**指數投資證券**（交易所交易票據），由券商發行、無擔保、通常有到期日；不持有資產，理論上無追蹤誤差，但有**發行人信用風險**。ETF 為投信發行、實際持有資產、通常無到期日、可能有追蹤誤差。

---

## 13) 參考來源（延伸閱讀）

- **ISIN/上市上櫃清單**：
  - TWSE 證券編碼公告（含 ISIN 一覽表與鏈結）  
    https://www.twse.com.tw/zh/products/code/announcement.html
  - 社群文章對 `strMode` 的整理與示例：  
    https://docsaid.org/blog/get-taiwan-all-stocks-info/  
    https://jerrynest.io/python-twse-stock-list/

- **籌碼資料**（TWSE CSV）：
  - MI_MARGN（融資融券）：`https://www.twse.com.tw/exchangeReport/MI_MARGN?response=csv&date=YYYYMMDD&selectType=ALL`  
  - T86（三大法人）：`https://www.twse.com.tw/fund/T86?response=csv&date=YYYYMMDD&selectType=ALLBUT0999`

- **ETN 與 ETF 差異**（入門解釋）：
  - 市場先生：ETN 是什麼？  
    https://rich01.com/what-is-etn/
  - E大成長股教學：ETN 與 ETF 差異  
    https://efrontrade.com/2025/09/stock-162.html

> 以上外部連結僅作概念與定位參考，實務上以官方文件與頁面為準；TWSE/TPEx 的頁面格式可能隨時間異動，請於錯誤處理中做好容錯。

---

## 14) 版權與使用

- 本筆記僅供學術與技術研究，**不構成投資建議**。  
- 使用任何資料前請確認符合來源網站的使用條款與法規。
# 台股「技術＋籌碼」資料整合管線（Markdown 筆記）

