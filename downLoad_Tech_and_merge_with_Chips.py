#2026/1/20 為了使用machine上的GPU，必須回去使用default的python environment，所以先把資料下載下來，但這次會先存成parquet格式，因為之後可以直接讀取成dataframe。現在遇到的困難是我需要在自己的kernel和machine上的kernel安裝一個librery，下次再弄
#2026/1/25 安裝pyarrow解決第一個問題，但遇到另一個"而 PyArrow 會在第一次寫檔時動態註冊某些擴充型別（例如 pandas.period）。這個註冊表是全域而且非 thread-safe，因此不同 thread 幾乎同時註冊或查找時，就可能一個 thread 尚未註冊完成、另一個卻偵測到「已註冊」、或找不到而報錯，於是你就看到了這兩種互斥的錯誤訊息"，拿程式碼焊error問GPT即可
#2026/2/16 用自己寫的write_parquet_via_pyarrow可以避免多重註冊問題，但之後讀取還是要跟copilot確認一下。針對yfinance遇到快速的request會回傳多重stock種類的table的問題也有檢查機制了，目前已經大部分可成功下載，但還是有小部分會失敗，可能有些太新的股票並沒有辦法追朔到2015，之後可以考慮先用目前的data讀取然後做training
import time
import random
import os
import json
import yfinance as yf
import pandas as pd
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import requests
from datetime import datetime
from io import StringIO

def fetch_stock_codes():
    """抓全市場股票代號清單。2026-10-01 發現這段原本放在 module 最上層，
    Windows 用 spawn 啟動 multiprocessing 時，每個 worker process 都會重新 import
    這個檔案，等於每個 worker 都各自重打一次這個請求（原本也沒設 timeout，isin.twse.com.tw
    一旦連不上，每個 worker 都會卡住直到被外層 60 秒逾時機制強制終止，造成當晚整批 0 檔成功）。
    改成包成函式、只在 __main__ 裡呼叫一次，worker 不會再重複呼叫。"""
    url = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
    response = requests.get(url, timeout=30)
    response.encoding = 'big5'
    dfs = pd.read_html(StringIO(response.text))
    stock_list = dfs[0]
    stock_list.columns = stock_list.iloc[0]
    stock_list = stock_list[1:]
    stock_list['股票代號'] = stock_list['有價證券代號及名稱'].str.split().str[0]
    stock_list['股票名稱'] = stock_list['有價證券代號及名稱'].str.split().str[1]
    stock_list = stock_list[['股票代號', '股票名稱']]
    # 只保留剛好 4 位數字的代號（排除 ETF、權證等 5~6 位代號）
    # 另外排除 00 開頭（ETF）與 91 開頭（TDR）代碼：這批在 yfinance 上會整批卡死不回應，
    # 且本模型本來就是設計給個股用的，排除它們對訓練樣本量影響可忽略。
    stock_list = stock_list[stock_list['股票代號'].str.match(r'^(?!00)(?!91)\d{4}$')]
    return stock_list['股票代號'].tolist()

##################################################################


def debug_dump_df(df: pd.DataFrame, tag: str, out_dir: str = "debug_snapshots"):
    """把當下 df 的結構與前幾列存檔，不用 print（避免時序變化）。"""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(out_dir, f"{tag}_{ts}")

    # 1) 欄位資訊（支援 MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        cols = [tuple(map(str, lvl)) for lvl in df.columns.tolist()]
        cols_info = {"type": "MultiIndex", "columns": cols}
    else:
        cols_info = {"type": "Index", "columns": list(map(str, df.columns.tolist()))}

    # 2) 索引資訊
    idx_info = {
        "is_multiindex": isinstance(df.index, pd.MultiIndex),
        "index_name": str(df.index.name),
        "len": len(df.index)
    }

    # 3) dtypes
    dtypes_info = {str(c): str(t) for c, t in df.dtypes.items()}

    meta = {
        "cols_info": cols_info,
        "index_info": idx_info,
        "dtypes": dtypes_info,
        "shape": list(df.shape)
    }

    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4) 前幾列（不 print，存 CSV）
    df.head(10).to_csv(base + "_head.csv", index=True, encoding="utf-8-sig")

    # 5) 也可整份存一份 parquet/feather 以便離線重現
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, base + "_full.parquet", compression="zstd")
    except Exception:
        # 若環境一時缺套件，至少保住 CSV
        df.to_csv(base + "_full_fallback.csv", index=True, encoding="utf-8-sig")



def _get_close_series_strict(df: pd.DataFrame, tag: str) -> pd.Series:
    """
    穩健地擷取單一 Close 欄位；若取得結果不是 Series（代表多欄或不一致），
    先做一次 debug_dump 再拋例外。
    """
    # MultiIndex 欄位：取 level=0 == 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        try:
            sub = df.xs('Close', axis=1, level=0)
        except KeyError:
            debug_dump_df(df, tag + "_NoCloseLevel0")
            raise KeyError("DataFrame 沒有 'Close' (level=0) 欄位")

        if isinstance(sub, pd.DataFrame):
            if sub.shape[1] != 1:
                # 多個 'Close' 欄，存檔再報
                debug_dump_df(df, tag + "_MultiCloseMI")
                raise ValueError(f"'Close' 在 MultiIndex 下有 {sub.shape[1]} 欄，非單一 Series")
            return sub.iloc[:, 0]
        else:
            return sub

    # 單層欄位：可能有重複 'Close'
    close_cols = [c for c in df.columns if c == 'Close']
    if not close_cols:
        debug_dump_df(df, tag + "_NoClose")
        raise KeyError("DataFrame 不包含 'Close' 欄位")

    if len(close_cols) > 1:
        # 多個同名 'Close'，存檔再報
        debug_dump_df(df, tag + "_MultiCloseFlat")
        raise ValueError(f"發現 {len(close_cols)} 個 'Close' 欄，非單一 Series")

    # 一個 'Close'，但保險檢查取值型態
    sub = df['Close']
    if isinstance(sub, pd.DataFrame):
        if sub.shape[1] != 1:
            debug_dump_df(df, tag + "_CloseIsDF_MultiCols")
            raise ValueError("'Close' 取到的是多欄 DataFrame")
        return sub.iloc[:, 0]
    return sub

##################################################################

def _is_single_close_series(df: pd.DataFrame) -> bool:
    """檢查 df 的 Close 是否為單一 Series（無論原本是否是 MultiIndex）。"""
    if df is None or df.empty:
        return False

    # 情況一：單層欄位
    if not isinstance(df.columns, pd.MultiIndex):
        # 是否有且僅有一個 'Close' 欄位
        close_cols = [c for c in df.columns if c == 'Close']
        if len(close_cols) != 1:
            return False
        # 確保取到的是 Series（不是 DataFrame）
        return not isinstance(df['Close'], pd.DataFrame)

    # 情況二：雙層欄位
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # (Field, Ticker)
    if 'Close' in lvl0:
        sub = df.xs('Close', axis=1, level=0)
        # 只在寬度為 1（單一 ticker）時判定為單一 Series
        return isinstance(sub, pd.Series) or (isinstance(sub, pd.DataFrame) and sub.shape[1] == 1)

    # (Ticker, Field) 的話，第 1 層才是 Field
    if 'Close' in lvl1:
        sub = df.xs('Close', axis=1, level=1)
        return isinstance(sub, pd.Series) or (isinstance(sub, pd.DataFrame) and sub.shape[1] == 1)

    return False



def _normalize_to_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    將 yfinance 回傳結果規範為單一 ticker、單層欄位。
    若本來就是單層就原樣返回；若是 MultiIndex，嘗試 (Field, Ticker) 與 (Ticker, Field) 兩種切法。
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)

    # (Field, Ticker)
    if ticker in lvl1:
        sub = df.xs(ticker, axis=1, level=1)
        sub.columns = sub.columns.get_level_values(0)
        return sub

    # (Ticker, Field)
    if ticker in lvl0:
        sub = df.xs(ticker, axis=1, level=0)
        sub.columns = sub.columns.get_level_values(1)
        return sub

    # 如果都不是，原樣回傳，交給上層判斷重抓
    return df


def download_single_ticker_until_ok(
    ticker: str,
    start: str,
    end: str,
    max_tries: int = 4,
    backoff_base: float = 0.5,
    auto_adjust: bool = False
) -> pd.DataFrame:
    """
    while 迴圈：只要檢測到「Close 不是單一 Series」就重抓，直到成功或超過 max_tries。
    最後一次失敗時 fallback 走 Ticker.history()。
    """
    last_err = None

    for attempt in range(1, max_tries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False,        # 外面如果有 ThreadPool，這裡必須 False
                group_by="column",
                auto_adjust=auto_adjust,
                timeout=20            # 避免網路卡住時整個 process 掛死
            )

            if df is None or df.empty:
                raise RuntimeError("yfinance 無資料")

            # 規範欄位為單一 ticker、單層欄位
            df = _normalize_to_single_ticker(df, ticker)

            # 驗證 Close 是否為單一 Series
            if not _is_single_close_series(df):
                raise ValueError("Close 非單一 Series（可能是多檔、多欄或 MultiIndex 未切乾淨）")

            # 到這裡表示「可安全計算 MA / RSI」
            df.index = pd.to_datetime(df.index)
            if df.index.has_duplicates:
                df = df[~df.index.duplicated(keep='last')]
            return df

        except Exception as e:
            last_err = e
            # 最後一次嘗試 → fallback
            if attempt == max_tries:
                try:
                    ssn = requests.Session()
                    t = yf.Ticker(ticker, session=ssn)
                    df = t.history(start=start, end=end, auto_adjust=auto_adjust, actions=False, timeout=20)
                    if df is None or df.empty:
                        raise RuntimeError("Ticker.history() 亦無資料")
                    # 規範欄位
                    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
                    df = df[keep]
                    # 再驗證一次
                    if 'Close' not in df.columns:
                        raise ValueError("Fallback history() 無 Close 欄位")
                    df.index = pd.to_datetime(df.index)
                    if df.index.has_duplicates:
                        df = df[~df.index.duplicated(keep='last')]
                    return df
                except Exception as e2:
                    last_err = e2
                    break  # 跳出 for，交給呼叫端處理
            else:
                # 退避 + 抖動
                sleep_s = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.05, 0.3)
                time.sleep(sleep_s)

    raise RuntimeError(f"下載 {ticker} 失敗：{last_err}")


def _warm_up_pyarrow():
    import pyarrow as pa  # 觸發初始化
    tmp = pd.DataFrame({
        "Date": pd.to_datetime(["2000-01-01"]),
        "x": [1.0]
    })
    tmp.to_parquet("_warmup.parquet", engine="pyarrow", index=False)
    os.remove("_warmup.parquet")

_warm_up_pyarrow()


def write_parquet_via_pyarrow(df: pd.DataFrame, out_path: str, compression: str = "zstd"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    # 將 pandas DataFrame 轉成 Arrow Table，不保留 index
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression=compression)


def query_chip(con, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
    SELECT Date, Ticker, margin_buy, margin_sell, short_buy, short_sell,
           foreign_net, invest_net, dealer_net, foreign_holding_pct
    FROM chip_daily
    WHERE Ticker = ? AND Date BETWEEN ? AND ?
    ORDER BY Date
    """
    df = pd.read_sql_query(q, con, params=(ticker, start_date, end_date), parse_dates=['Date'])
    return df

def query_fundamental(con, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
    SELECT Date, pe_ratio, pb_ratio, div_yield
    FROM fundamental_daily
    WHERE Ticker = ? AND Date BETWEEN ? AND ?
    ORDER BY Date
    """
    try:
        df = pd.read_sql_query(q, con, params=(ticker, start_date, end_date), parse_dates=['Date'])
    except Exception:
        df = pd.DataFrame(columns=['Date', 'pe_ratio', 'pb_ratio', 'div_yield'])
    return df


def query_monthly_revenue(con, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    抓月營收並計算 rev_yoy / rev_mom。
    回傳 DataFrame: Date(月份第一天), rev_yoy, rev_mom
    """
    try:
        # 多抓 14 個月（計算 YoY 需要去年同月）
        extended_start = (pd.Timestamp(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        q = """
        SELECT Date, revenue FROM monthly_revenue
        WHERE Ticker = ? AND Date BETWEEN ? AND ?
        ORDER BY Date
        """
        df = pd.read_sql_query(q, con, params=(ticker, extended_start, end_date), parse_dates=['Date'])
        if df.empty:
            return pd.DataFrame(columns=['Date', 'rev_yoy', 'rev_mom'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['rev_mom'] = df['revenue'].pct_change(1)
        df['rev_yoy'] = df['revenue'].pct_change(12)
        # 只回傳 start_date 之後的
        df = df[df['Date'] >= pd.Timestamp(start_date)][['Date', 'rev_yoy', 'rev_mom']]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['Date', 'rev_yoy', 'rev_mom'])


def query_quarterly_financials(con, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    抓季報並計算 contract_liability_qoq / inventory_qoq /
    inventory_turnover / gross_margin_q / gross_margin_qoq。
    回傳 DataFrame: Date(季末日), 各衍生指標
    """
    try:
        extended_start = (pd.Timestamp(start_date) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        q = """
        SELECT Date, inventory, contract_liability, accounts_receivable,
               accounts_payable, revenue_q, gross_profit, cogs, operating_income
        FROM quarterly_financials
        WHERE Ticker = ? AND Date BETWEEN ? AND ?
        ORDER BY Date
        """
        df = pd.read_sql_query(q, con, params=(ticker, extended_start, end_date), parse_dates=['Date'])
        if df.empty:
            return pd.DataFrame(columns=['Date', 'contract_liability_qoq', 'inventory_qoq',
                                         'inventory_turnover', 'gross_margin_q', 'gross_margin_qoq'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 季增率
        df['contract_liability_qoq'] = df['contract_liability'].pct_change(1)
        df['inventory_qoq']          = df['inventory'].pct_change(1)

        # 存貨周轉率（年化）= cogs*4 / 平均存貨
        avg_inv = (df['inventory'] + df['inventory'].shift(1)) / 2
        df['inventory_turnover'] = (df['cogs'] * 4) / avg_inv.replace(0, float('nan'))

        # 毛利率及其季變化
        df['gross_margin_q']    = df['gross_profit'] / df['revenue_q'].replace(0, float('nan'))
        df['gross_margin_qoq']  = df['gross_margin_q'].diff(1)

        cols = ['Date', 'contract_liability_qoq', 'inventory_qoq',
                'inventory_turnover', 'gross_margin_q', 'gross_margin_qoq']
        df = df[df['Date'] >= pd.Timestamp(start_date)][cols]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['Date', 'contract_liability_qoq', 'inventory_qoq',
                                     'inventory_turnover', 'gross_margin_q', 'gross_margin_qoq'])


# 公告延遲（2026-10-13 修復）：quarterly_financials/monthly_revenue 表裡的 Date 是「季末/月份」
# 日期，不是公布日期。原本 merge_asof 直接拿季末/月份日期去跟交易日比對，等於訓練時在財報
# 真正公布前就讓模型看到這一季/這個月的數字（前視偏誤）。回測腳本（backtest_current.py /
# backtest_riskadj.py）已經有加這個延遲，這裡補上讓正式訓練 pipeline 跟回測一致。
# 月營收：依法次月 10 日前公布，留 5 天緩衝 → 15 天。
REV_LAG_DAYS = 15
# 季報：Q1-Q3 季末後 45 天內申報，留 5 天緩衝 → 50 天；Q4(年報) 季末後 90 天內申報 → 90 天。
Q_LAG_DAYS_NORMAL = 50
Q_LAG_DAYS_Q4 = 90


def merge_monthly_to_daily(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    """將月頻資料 forward-fill 到日頻（每日使用最新「已公布」月份的值）"""
    if monthly_df.empty:
        for c in ['rev_yoy', 'rev_mom']:
            daily_df[c] = None
        return daily_df
    monthly_df = monthly_df.sort_values('Date').copy()
    # 公告延遲：月營收要等 REV_LAG_DAYS 天之後才看得到，見上方說明
    monthly_df['Date'] = monthly_df['Date'] + pd.Timedelta(days=REV_LAG_DAYS)
    # merge_asof：找每個交易日對應的最近「已公布」月份資料
    result = pd.merge_asof(
        daily_df.sort_values('Date'),
        monthly_df,
        on='Date', direction='backward'
    )
    return result


def merge_quarterly_to_daily(daily_df: pd.DataFrame, quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """將季頻資料 forward-fill 到日頻（每日使用最新「已公布」季報的值）"""
    qcols = ['contract_liability_qoq', 'inventory_qoq',
             'inventory_turnover', 'gross_margin_q', 'gross_margin_qoq']
    if quarterly_df.empty:
        for c in qcols:
            daily_df[c] = None
        return daily_df
    quarterly_df = quarterly_df.sort_values('Date').copy()
    # 公告延遲：季末日期 (Date) 要先往後推才是「市場看得到」的日期，見上方說明
    # Q4（月份=12，即年報）申報期限比 Q1-Q3 長，延遲天數也要跟著拉長
    lag_days = quarterly_df['Date'].dt.month.map(
        lambda m: Q_LAG_DAYS_Q4 if m == 12 else Q_LAG_DAYS_NORMAL
    )
    quarterly_df['Date'] = quarterly_df['Date'] + pd.to_timedelta(lag_days, unit='D')
    result = pd.merge_asof(
        daily_df.sort_values('Date'),
        quarterly_df,
        on='Date', direction='backward'
    )
    return result

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_stock_to_parquet(
    stock_code,
    DB_PATH="chip.db",
    OUT_DIR="parquet_data",
    start_date=(datetime.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d"),
    end_date=datetime.today().strftime("%Y-%m-%d")
):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)

        stock_code_tw = stock_code + ".TW"

        # 1️⃣ yfinance
        
        stock_data = download_single_ticker_until_ok(stock_code_tw, start_date, end_date)
        
        if stock_data.empty or len(stock_data) < (120 + 126):
            return None

        # 2️⃣ 技術指標    
        stock_data['MA10'] = stock_data['Close'].rolling(10).mean()
        stock_data['RSI'] = compute_rsi(stock_data['Close'], 14)

        # index → Date
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data = stock_data.reset_index()

        # 3️⃣ chip data + fundamental + financial features
        with sqlite3.connect(DB_PATH) as sql_con:
            chip    = query_chip(sql_con, stock_code, start_date, end_date)
            fund    = query_fundamental(sql_con, stock_code, start_date, end_date)
            monthly = query_monthly_revenue(sql_con, stock_code, start_date, end_date)
            quarterly = query_quarterly_financials(sql_con, stock_code, start_date, end_date)

        chip['Date'] = pd.to_datetime(chip['Date'])
        fund['Date'] = pd.to_datetime(fund['Date'])
        if not monthly.empty:
            monthly['Date'] = pd.to_datetime(monthly['Date'])
        if not quarterly.empty:
            quarterly['Date'] = pd.to_datetime(quarterly['Date'])

        # pandas 3.x 各來源 datetime 解析度可能不同（s / us / ns），merge 會因此失敗，統一成 ns
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).astype('datetime64[ns]')
        chip['Date'] = chip['Date'].astype('datetime64[ns]')
        fund['Date'] = fund['Date'].astype('datetime64[ns]')
        if not monthly.empty:
            monthly['Date'] = monthly['Date'].astype('datetime64[ns]')
        if not quarterly.empty:
            quarterly['Date'] = quarterly['Date'].astype('datetime64[ns]')

        # 4️⃣ merge：tech + chip（inner），再 left join 各類特徵
        NULLABLE_COLS = ("pe_ratio", "pb_ratio", "div_yield",
                         "rev_yoy", "rev_mom",
                         "contract_liability_qoq", "inventory_qoq",
                         "inventory_turnover", "gross_margin_q", "gross_margin_qoq",
                         "foreign_holding_pct")

        df = pd.merge(
            stock_data,
            chip.drop(columns=["Ticker"]),
            on="Date",
            how="inner"
        )
        if not fund.empty:
            df = pd.merge(df, fund, on="Date", how="left")
        else:
            df["pe_ratio"] = df["pb_ratio"] = df["div_yield"] = None

        # 月營收特徵（forward-fill）
        df = merge_monthly_to_daily(df, monthly)

        # 季報特徵（forward-fill）
        df = merge_quarterly_to_daily(df, quarterly)

        df = df.dropna(subset=[c for c in df.columns if c not in NULLABLE_COLS])
        
        
        # 5️⃣ Parquet（只存「處理後可訓練」資料）
        out_path = os.path.join(OUT_DIR, f"{stock_code_tw}.parquet")

        write_parquet_via_pyarrow(df, out_path, compression="zstd")

        print(f"✅ Saved {stock_code_tw}: {out_path}")
        return out_path

    except Exception as e:
        print(f"❌ {stock_code_tw}: {e}")
        return None



def _worker_entry(stock_code, db_path, out_dir):
    prepare_stock_to_parquet(stock_code, DB_PATH=db_path, OUT_DIR=out_dir)


if __name__ == "__main__":
    import multiprocessing as mp
    import glob as _glob

    MAX_CONCURRENT = 5        # 2026-10-01 從 10 降到 5：原本 10 個 worker 完全沒有節流、
                              # 一有空位就馬上打下一檔，等於全速高併發轟炸 yfinance，
                              # 懷疑是當天多次被限流（Too Many Requests）的主因之一
    STAGGER_SEC = 0.5         # 每次派發新 worker 之間至少間隔這麼久，降低瞬間併發打過去的速率
    PER_TICKER_TIMEOUT = 60  # 秒；yfinance 少數股票在 curl_cffi 底層會整個掛死不回應，
                              # 一般的 timeout 參數對這種情況沒用，只能用 process 層級強制 kill

    try:
        stock_codes = fetch_stock_codes()
    except Exception as e:
        # isin.twse.com.tw 偶爾會連不上（2026-10-01 發生過），退回用現有 parquet_data 的股票清單，
        # 不要讓整次 merge 直接失敗——頂多漏掉當天新上市的個股，隔天這個清單來源恢復正常就會補上。
        print(f"⚠️ 抓股票清單失敗（{e}），改用 parquet_data/ 既有檔名當清單")
        stock_codes = [os.path.basename(f).replace(".TW.parquet", "")
                       for f in _glob.glob("parquet_data/*.TW.parquet")]
        print(f"   退回清單共 {len(stock_codes)} 檔")

    pending = list(stock_codes)
    active = {}  # pid -> (Process, ticker, start_time)

    while pending or active:
        while len(active) < MAX_CONCURRENT and pending:
            code = pending.pop(0)
            p = mp.Process(target=_worker_entry, args=(code, "chip.db", "parquet_data"))
            p.start()
            active[p.pid] = (p, code, time.time())
            time.sleep(STAGGER_SEC)

        time.sleep(1)
        for pid in list(active.keys()):
            p, code, start_t = active[pid]
            if not p.is_alive():
                p.join()
                del active[pid]
            elif time.time() - start_t > PER_TICKER_TIMEOUT:
                print(f"⏱️ {code}.TW: 超過 {PER_TICKER_TIMEOUT}s 無回應，強制終止", flush=True)
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                del active[pid]

    print("=== ALL DONE ===", flush=True)
