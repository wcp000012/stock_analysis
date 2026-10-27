"""排程用：每天自動更新資料，並在有新交易日資料時重新 merge + 訓練 + 存檔預測結果。
由 Windows 工作排程器在平日晚上 7 點觸發（見 memory/project-complete-guide.md 的排程說明）。
不需要人在旁邊：FinMind 配額冷卻、TWSE WAF 失敗都會自動處理或安全跳過。

步驟：
  1. chip_daily + 外資持股比例（TWSE，免配額）
  2. 月營收 + FinMind 資產負債表/現金流量表/損益表
  3. 基本面 PE/PB/殖利率（偶爾被 WAF 擋，可接受）
  4. merge：重新用 yfinance 產生 parquet_data（~25 分鐘）
  5. 訓練 + 預測：topK_predict.py（本機 GPU，~12 分鐘），完成後把結果 CSV 另存到 predictions_archive/

步驟 4、5 只有在 chip_daily 出現「上次訓練之後的新交易日」時才會執行（避免假日/重複執行浪費 GPU）。
用法：python _daily_update.py [--force-train] [--only-train]
"""
import argparse
import ctypes
import glob
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta

NETWORK_WAIT_MAX_SEC = 180   # 排程喚醒睡眠中的電腦後，網路/DNS 通常要幾秒到一分鐘才會恢復


def wait_for_network(log_fn=print, max_wait=NETWORK_WAIT_MAX_SEC):
    """排程可能在電腦剛從睡眠喚醒時就被觸發，此時網路/DNS 還沒就緒，
    直接打 TWSE/FinMind 會整批 DNS resolve 失敗（2026-09-25、09-28 兩晚都撞到這個問題，
    當晚 chip_daily 整天沒補到）。開跑前先確認能連上網路，連不上就等幾秒再試。"""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            socket.setdefaulttimeout(5)
            socket.getaddrinfo("www.twse.com.tw", 443)
            log_fn(f"網路已就緒（等待 {time.time() - start:.0f} 秒）")
            return True
        except OSError:
            time.sleep(5)
    log_fn(f"等了 {max_wait} 秒網路仍未就緒，繼續執行（讓各步驟自己的重試機制處理）")
    return False

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on):
    """告訴 Windows 這段期間不要自動睡眠（否則背景程式跑到一半電腦會因閒置計時器睡著，工作被凍結）。"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def pid_alive(pid):
    """用 tasklist 檢查（不能用 os.kill(pid, 0)：Windows 上會直接把該程序終止）。"""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out and "python" in out.lower()
    except Exception:
        return False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_FILE = os.path.join(BASE_DIR, "_daily_update.lock")
LOG_DIR = os.path.join(BASE_DIR, "daily_update_logs")
ARCHIVE_DIR = os.path.join(BASE_DIR, "predictions_archive")
MARKER_FILE = os.path.join(BASE_DIR, "_last_train_anchor.txt")
PRED_FILES = ["live_prediction_all.csv", "live_prediction_top_B.csv", "live_prediction_top_C.csv"]
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

MAX_LOCK_HOURS = 24           # lock 裡的 PID 還活著、且建立不到 24 小時 → 視為仍在執行；超過視為卡死，清掉重跑
LOG_KEEP_DAYS = 30
MIN_REGENERATED_PARQUET = 950  # merge 之後至少要有這麼多檔案是今天新產生的，否則不訓練（避免拿舊資料假裝新預測）


def chip_max_date():
    with sqlite3.connect(os.path.join(BASE_DIR, "chip.db")) as con:
        return con.execute("SELECT MAX(Date) FROM chip_daily").fetchone()[0]


def last_trained_anchor():
    try:
        with open(MARKER_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def cleanup_old_logs():
    cutoff = time.time() - LOG_KEEP_DAYS * 86400
    for p in glob.glob(os.path.join(LOG_DIR, "*.log")):
        if os.path.getmtime(p) < cutoff and p != LOG_FILE:
            os.remove(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-train", action="store_true", help="即使沒有新交易日資料也重新 merge + 訓練")
    parser.add_argument("--only-train", action="store_true", help="跳過步驟 1~3（資料更新），只做 merge + 訓練")
    args = parser.parse_args()

    if os.path.exists(LOCK_FILE):
        age_hours = (datetime.now().timestamp() - os.path.getmtime(LOCK_FILE)) / 3600
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
        except Exception:
            old_pid = None
        if old_pid and pid_alive(old_pid) and age_hours < MAX_LOCK_HOURS:
            print(f"上一次排程還在執行（PID {old_pid} 存活，lock 建立於 {age_hours:.1f} 小時前），這次跳過，避免重疊執行。")
            return
        print(f"發現孤兒 lock file（PID {old_pid} 已不存在或 lock 超過 {MAX_LOCK_HOURS} 小時，"
              f"建立於 {age_hours:.1f} 小時前），自動清除後繼續執行。")
        os.remove(LOCK_FILE)

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

    keep_awake(True)
    try:
        cleanup_old_logs()
        with open(LOG_FILE, "w", encoding="utf-8") as log:
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["TQDM_MININTERVAL"] = "30"   # 訓練/預測的 tqdm 進度條不要把 log 灌爆
            ca_bundle = os.path.join(BASE_DIR, "ca_bundle.pem")
            if os.path.exists(ca_bundle):
                child_env.setdefault("CURL_CA_BUNDLE", ca_bundle)  # yfinance(curl_cffi) 過這台電腦的 TLS 攔截需要

            def say(msg):
                log.write(f"{msg}\n")
                log.flush()

            wait_for_network(say)

            def run(cmd, desc, timeout=None):
                """回傳 exit code；逾時回傳 None。"""
                say(f"\n{'=' * 60}\n{desc} -- {datetime.now()}\n{'=' * 60}")
                try:
                    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", cwd=BASE_DIR,
                                           timeout=timeout, env=child_env)
                    say(f"\n[exit code {proc.returncode}]")
                    return proc.returncode
                except subprocess.TimeoutExpired:
                    say(f"\n[TIMEOUT after {timeout}s，已終止此步驟]")
                    return None

            if not args.only_train:
                # 1) 籌碼（融資融券/三大法人）+ 外資持股比例：補最近 10 天，skip-if-exists 成本很低
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
                run(
                    [sys.executable, "-c",
                     "from stock_data_down_load_chips import download_chip_range_to_sqlite, backfill_foreign_holding_pct; "
                     f"download_chip_range_to_sqlite('{start}', '{end}'); "
                     f"backfill_foreign_holding_pct('{start}', '{end}')"],
                    "Step 1: chip_daily + foreign_holding_pct（TWSE，免配額）"
                )

                # 2) 月營收 + FinMind 資產負債表/現金流量表/損益表（有配額冷卻機制，最多讓它跑 1.5 小時）
                run(
                    [sys.executable, "download_financial_features.py"],
                    "Step 2: monthly_revenue + FinMind BS/CF/損益表",
                    timeout=int(1.5 * 3600)
                )

                # 3) 基本面 PE/PB/殖利率（偶爾被 TWSE WAF 擋，失敗沒關係，之後排程會自動補）
                run(
                    [sys.executable, "download_fundamentals.py"],
                    "Step 3: fundamental_daily（PE/PB/殖利率）"
                )

            # 4)+5) 只有出現新的交易日資料才重新 merge + 訓練
            anchor = chip_max_date()
            trained = last_trained_anchor()
            say(f"\nchip_daily 最新日期 = {anchor}，上次訓練所用的最新日期 = {trained}")
            if anchor == trained and not args.force_train:
                say("沒有新的交易日資料，略過 merge + 訓練。")
            else:
                merge_start = time.time()
                rc = run([sys.executable, "-u", "downLoad_Tech_and_merge_with_Chips.py"],
                         "Step 4: merge（yfinance 股價 + chip.db → parquet_data）", timeout=int(1.5 * 3600))
                regenerated = sum(1 for p in glob.glob(os.path.join(BASE_DIR, "parquet_data", "*.parquet"))
                                  if os.path.getmtime(p) >= merge_start)
                say(f"merge 結束（exit code {rc}），本次新產生的 parquet 檔數 = {regenerated}"
                    f"（門檻 {MIN_REGENERATED_PARQUET}）")
                if rc != 0 or regenerated < MIN_REGENERATED_PARQUET:
                    say("merge 失敗或新產生的檔案太少（可能是 yfinance 連不上），不訓練，避免用舊資料產生假的新預測。")
                else:
                    train_start = time.time()
                    rc = run([sys.executable, "-u", "topK_predict.py"],
                             "Step 5: 訓練 + 預測（本機 GPU）", timeout=int(1.5 * 3600))
                    outputs_fresh = all(
                        os.path.exists(os.path.join(BASE_DIR, n)) and os.path.getmtime(os.path.join(BASE_DIR, n)) >= train_start
                        for n in PRED_FILES)
                    if rc == 0 and outputs_fresh:
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        for n in PRED_FILES:
                            shutil.copy2(os.path.join(BASE_DIR, n),
                                         os.path.join(ARCHIVE_DIR, f"{stamp}_anchor{anchor}_{n}"))
                        # Model D（觀察用，見 RESEARCH_LOG.md E9/E9b）：純粹多存一份，可有可無，
                        # 不列進 PRED_FILES、不影響上面的成功判斷跟 anchor 是否推進。
                        d_name = "live_prediction_top_D.csv"
                        d_path = os.path.join(BASE_DIR, d_name)
                        if os.path.exists(d_path) and os.path.getmtime(d_path) >= train_start:
                            shutil.copy2(d_path, os.path.join(ARCHIVE_DIR, f"{stamp}_anchor{anchor}_{d_name}"))
                        with open(MARKER_FILE, "w", encoding="utf-8") as f:
                            f.write(anchor)
                        say(f"訓練完成，預測結果已存檔到 predictions_archive/（{stamp}_anchor{anchor}_*），"
                            f"並記錄 anchor={anchor}。")
                    else:
                        say(f"訓練失敗或沒有產出新的結果檔（exit code {rc}, outputs_fresh={outputs_fresh}），"
                            f"不更新 anchor 標記，下次排程會重試。")

        print(f"完成，log 存在 {LOG_FILE}")
    finally:
        keep_awake(False)
        os.remove(LOCK_FILE)


if __name__ == "__main__":
    main()
