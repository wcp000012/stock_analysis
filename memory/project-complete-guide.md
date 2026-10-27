---
name: project-complete-guide
description: 完整專案轉移指南 — 換電腦或重啟時所需的所有資訊，包含腳本說明、資料庫、模型、Paperspace 環境、工作流程
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a972ca5-853c-467f-afd4-229a9fcc7620
---

## 專案目標
用 PatchTST Transformer 預測台灣股票未來 60 天最大報酬率，每週更新資料後重訓，輸出 Top-20 買進候選名單（Model B 主要結果 + Model C 參考）。

---

## 本機環境
- **工作目錄**：`C:\Users\User\stock_analysis`（2026-07-29 換新機器後的路徑；舊機器路徑 `C:\Users\cpwang\PERC_Working\script\stock_analysis` 已不再使用）
- **Python**：3.11.9（winget 裝的），已裝 `pip-system-certs` 解決這台機器的 TLS 攔截問題
- **Claude 會話 ID**：`9a972ca5-853c-467f-afd4-229a9fcc7620`（舊機器會話，新機器上不一定能 `--resume`，找不到就直接把這份指南交給新會話）

---

## API 金鑰與帳號
- **Paperspace API Key**：存放在本機未進版控的檔案/環境變數（**不要寫在這份文件裡** —— 2026-10-27 曾經在這裡明文外洩過，已撤銷重置，往後一律用 `<PAPERSPACE_API_KEY>` 佔位符）。Bearer 格式：`{'authorization': 'Bearer <key>'}`
- **FinMind**：免費 API，無需 token，URL: `https://api.finmindtrade.com/api/v4/data`
- **TWSE**：公開 API（不需金鑰），用於籌碼、融資融券下載
- **yfinance**：K 線資料，免費無需金鑰

---

## Paperspace 環境
- **Notebook 顯示名稱**：`nm0k7kyrma`（PyTorch 1.12）—— 這只是 name 欄位，不是真正的 notebook id！
- **Notebook 真正 ID**：**每次手動 Start 都會變**（2026-08-27 是 `nfg460g21w`，2026-09-07 重啟後變成 `nj36kn4zia`）；查 `GET /v1/notebooks/{id}` 要用這個 id，用顯示名稱會 404。每次重啟後務必用 `GET /v1/notebooks`（列表 API）重新查最新的 id/fqdn/token，並同步更新 `_run_remote.py`/`_run_merge.py`/`_check_merge_log.py`/`_check_predict_log.py`/`_upload_db.py` 這 5 支腳本裡的連線資訊，還要重新 `POST /api/kernels` 建新 kernel 拿新 kid。
- **Project ID**：`po7hremeio2`
- **Machine**：Free-A4000（GPU）
- **查 state / FQDN / token**：free tier 沒有 `/notebooks/{id}` 單筆查詢可靠，改用列表 API：
  ```python
  import requests
  r = requests.get('https://api.paperspace.com/v1/notebooks',
                   headers={'authorization': 'Bearer <PAPERSPACE_API_KEY>'})
  nb = r.json()['items'][0]   # 目前帳號下只有這一個 notebook
  fqdn  = nb['fqdn']
  token = nb['token']
  state = nb['state']         # 'Running' / 'Cancelled' 等
  ```
- **啟動 Notebook**：free tier **沒有 API 可以啟動**（試過 `/start`、`/fork` 都是 404），state 不是 Running 時必須到網頁 UI 手動按 Start，啟動後再重新查一次列表 API 拿新的 fqdn/token。

### Paperspace Python 環境（重要！）
| 用途 | Python 路徑 |
|------|------------|
| **Merge 腳本**（yfinance、資料處理） | `/storage/conda_envs/stock_env/bin/python` |
| **topK_predict.py**（PyTorch + CUDA GPU） | `/usr/local/bin/python3` |

conda env 設定方式參見 `/notebooks/conda_env.txt`（在 Paperspace 上）。

### Kernel 建立
```python
import requests
r = requests.post(f'https://{fqdn}/api/kernels',
    headers={'Authorization': f'token {token}'},
    json={'name': 'python3'}, timeout=30)
kid = r.json()['id']
```

---

## 本機腳本說明

### 主要腳本
| 檔案 | 用途 |
|------|------|
| `stock_data_down_load_chips.py` | 從 TWSE 下載籌碼資料（融資融券 + 三大法人）存入 chip.db |
| `download_fundamentals.py` | 從 FinMind 下載基本面（PE/PB/殖利率）存入 chip.db |
| `download_financial_features.py` | 月營收+損益表用 TWSE/TPEx OpenAPI 整批下載（免配額），資產負債表明細+現金流量表仍用 FinMind 逐股（2026-08-27 改版，見下方「資料來源」章節） |
| `downLoad_Tech_and_merge_with_Chips.py` | 用 yfinance 抓 K 線，合併 chip.db 資料，生成 parquet_data/*.parquet |
| `topK_predict.py` | PatchTST 訓練 + 預測，輸出 live_prediction_top_B/C.csv |
| `backtest_validate.py` | 回測驗證 Model A/B/C 歷史績效 |

### Paperspace 遙控腳本
| 檔案 | 用途 |
|------|------|
| `_run_remote.py` | 在 Paperspace kernel 執行 topK_predict.py（持續開著 WebSocket 到跑完為止，本機端會一直等，適合用 PowerShell background 執行避免卡住終端機） |
| `_run_merge.py` | 在 Paperspace kernel 執行 merge 腳本（用 `subprocess.Popen` + `start_new_session=True` 讓遠端背景執行，本機端只等 30 秒確認啟動成功就回傳，之後要用 `_check_merge_log.py` 另外查進度） |
| `_upload_db.py` | 把本機 chip.db 用 WebSocket 分塊上傳到 Paperspace（256KB/塊，取代不可靠的 REST PUT，2026-09-07 新增） |
| `_check_merge_log.py` | 查 merge.log 進度 + parquet 數量 |
| `_check_predict_log.py` | 查 predict.log 進度 |
| `_check_backtest_log.py` | 查 backtest.log 進度 |

> **注意**：每次 Paperspace 重啟，`_run_remote.py`、`_run_merge.py`、`_check_*.py` 裡的 `ws_base`、`token`、`kid` 都要更新。

---

## 資料庫：chip.db（本機 + Paperspace 各一份）

### 表格結構
```
chip_daily        Date, Ticker, margin_buy, margin_sell, short_buy, short_sell, foreign_net, invest_net, dealer_net
fundamental_daily Date, Ticker, pe_ratio, pb_ratio, div_yield
monthly_revenue   Date, Ticker, revenue
quarterly_financials Date, Ticker, inventory, contract_liability, accounts_receivable, accounts_payable,
                     prepayments, revenue_q, gross_profit, cogs, operating_income, net_income,
                     operating_cf, capex
```

### download_financial_features.py 資料來源（2026-08-27 改版，2026-09-03 追加 token + 自動冷卻，2026-09-10 損益表改用 FinMind）
FinMind 免費版有請求配額，之前卡在 314/1094。改成兩階段：
- **Phase 1（月營收）**：**TWSE + TPEx OpenAPI**（`openapi.twse.com.tw`／`www.tpex.org.tw/openapi`，免金鑰、無配額）整批下載，一次 API call 拿全市場最新一期快照，upsert 進 DB。缺點是官方只給「最新一期」，沒有歷史區間查詢，所以歷史序列要靠每次執行累積，不能一次回補到 2023。月營收：`t187ap05_L`（TWSE）+ `mopsfin_t187ap05_O`（TPEx）。
- **Phase 2（資產負債表明細 + 現金流量表 + 損益表）**：全部改用 **FinMind** 逐股下載（`TaiwanStockBalanceSheet` / `TaiwanStockCashFlowsStatement` / `TaiwanStockFinancialStatements`）。
- 試過直接 scrape MOPS（`mops.twse.com.tw/mops/web/ajax_t164sb03`）想拿明細，被官方 WAF 直接擋掉（"因為安全性考量"），這條路不可行，別再嘗試。

**⚠️ 2026-09-10 修正一個嚴重 bug（單位不一致）**：損益表原本（8/27 改版）是用 TWSE/TPEx OpenAPI 整批抓「最新一期」（`t187ap06_L_ci` 等 dataset），跟 FinMind 資產負債表（`inventory` 等）混用。使用者問「華邦電存貨週轉天數」時查證發現：**OpenAPI 損益表金額單位是「仟元」，FinMind 資產負債表金額單位是「元」，差了 1000 倍**，且 OpenAPI 那段是每次執行都無條件覆蓋「最新一期」（`COALESCE(excluded.c, c)`，新值不為空就覆蓋），導致**全市場 1080 檔股票的「最新一季」revenue_q/cogs/gross_profit 等欄位全部跟自己的歷史資料（FinMind 來源）差 1000 倍**——不是只有缺歷史的股票有事，連原本靠舊版 FinMind 全量拉過、有完整 14 季歷史的 314 檔，其最新一季也被 OpenAPI 覆蓋壞掉。實測案例：1101（台泥）2026Q1 revenue_q=331.68 億，2026Q2 卻只有 71,289,957（=7128萬，少了近500倍）。
  - **修法**：損益表全部改用 FinMind `TaiwanStockFinancialStatements`（跟資產負債表同源，單位保證一致），完全移除 TWSE/TPEx OpenAPI 損益表那段程式碼（`IS_FIELDS_CI`/`update_income_statement_bulk`/`_extract_is_row`/`roc_quarter_to_date` 等相關函式與常數都刪掉了，只留月營收還用 OpenAPI，月營收沒有這個單位問題）。
  - 新增 `income_fetched` 表（Ticker + fetched_at）追蹤每檔股票上次補損益表歷史的時間，80 天內不重抓，邏輯跟資產負債表的 skip 機制一致。`download_income_history()` 這個新函式跟 `download_quarterly()`（BS/CF）並列在同一個 Phase 2 迴圈裡，共用同一套配額冷卻機制。
  - **這是一次性全庫回填**：因為 `income_fetched` 表是全新的、一開始所有 1080 檔都會被視為「沒抓過」，所以第一次跑這個修好的版本會對全部 1080 檔各多打 1 次 FinMind request（原本只有 BS/CF 兩個 dataset，現在變 3 個），配額用量會比之前的 session 更快打滿，需要多花 1-2 次 600/hour 的 session 才能全部跑完，之後才會轉為只有真正過期的才重抓。
  - **驗證方式**：`upsert_income()` 用 FinMind 抓到的 5 個欄位（revenue_q/cogs/gross_profit/operating_income/net_income）**直接覆蓋**（不是 COALESCE），因為 FinMind 現在是唯一權威來源。已用 1101、2344 驗證過修復後數字量級前後一致（不再有 1000 倍斷層）。
  - **2026-09-10 首次跑這個修復版本**：一次 session 打到 2 次配額暫停（各 30 分鐘冷卻，這次沒有中途手動停掉，讓它自動睡等繼續跑，累積跑了約 900 檔），進度：`income_fetched` 表 920/1080 檔已補過。
  - **2026-09-11 補完**：再跑一次 `python download_financial_features.py`，這次**沒有觸發配額暫停就自然跑完**（`income_fetched` **1094/1094 全部補齊**），損益表單位不一致的 bug 已經在全市場修復完成。同一次也順便把 `inventory` 覆蓋率從 959 推進到 **1026/1094**（剩 68 檔可能是 FinMind 本身沒有資產負債表資料）。
- **2026-09-03**：使用者有註冊 FinMind 帳號，token 存在專案根目錄 `APIKey.txt`（格式：一行 `FinMind APIKey <jwt>`）。腳本開頭新增 `_load_finmind_token()` 自動讀取並帶入每次請求的 `token` 參數，免費會員配額是 **600 次/小時**。
- **2026-09-03**：`finmind_get()` 新增全域冷卻機制（`_cooldown_lock`/`_cooldown_until`）——一旦收到 429 或 403（FinMind 超額會回 403 `{"msg":"ip banned","retry_after":N}`），會讀取伺服器回傳的 `retry_after` 秒數，讓所有後續請求先睡那麼久再繼續，log 會印 `[配額暫停] ... 等待 N 秒`。**這是預期行為不是掛掉**，不要看到這行就殺掉 process。因為 `run_download()` 其實是單執行緒依序跑（`MAX_WORKERS`/`_semaphore` 目前是沒真的平行化的遺留設計），配額冷卻對它來說就是單純的睡眠等待。
- 實測效率：600 次配額中，大約只有 70-80% 能換到「有新資料」的股票（部分股票在 FinMind 本來就沒有資產負債表資料，一樣耗配額但拿不到東西）。

**⚠️ 2026-09-22 修正另一個嚴重 bug（`download_quarterly()` 的 80 天 skip 判斷邏輯錯誤）**：BS/CF 那段原本是拿「資料庫裡最新一筆季報的『季末日期』」跟今天比，小於 80 天就跳過。這個假設在剛抓到資料時是對的，但**只要今天離最新一次公布的季報（例如 2026-06-30，Q3 要等 11 月才公布）超過 80 天，全市場每一檔股票都會被誤判成「太久沒更新」，導致每次排程都重新對全部 1097 檔打 FinMind request**（不是只有缺資料的股票，是每一檔）。這正是 2026-09-21 晚上 21:30 排程配額用得特別兇、跑到 timeout 的原因。
  - **修法**：仿照 `income_fetched` 的做法，新增 `bs_cf_fetched`（Ticker + fetched_at）表，`download_quarterly()` 改成判斷「我們上次去抓的時間」而不是「季末日期」，80 天內不重抓。
  - **這也是一次性全庫回填**（`bs_cf_fetched` 一開始是空的），2026-09-22 晚上到 2026-09-24 之間分幾次跑完，**2026-09-24 確認 `bs_cf_fetched`／`income_fetched` 都 1097/1097 全部補齊**。順便修掉了另一個小副作用：原本完全沒有資產負債表資料的 ~70 檔股票，因為 `last` 永遠是 `None`，每次都不會被跳過、每晚都白白浪費配額重打；現在這些股票也會被記錄 `fetched_at`，80 天內不會再重打。
  - **驗證結果（2026-09-24）**：排程重跑一次，Step 2（FinMind）從原本動輒燒完整個 1.5 小時 timeout，**縮短到約 90 秒**（全部 1097 檔都印出 `BS/CF 明細 +0，損益表 +0 rows`，正確判斷不用重抓），修復完全生效。**FinMind 配額問題到這裡才算真正解決**，之後每天排程只有真的過期（超過 80 天沒抓過）或有新股票時才會用到配額。

### 2026-09-24：yfinance 大量限流導致當天 merge 失敗，屬正常安全機制
FinMind 那段修好、Step 2 飛快跑完之後，緊接著 Step 4 merge 卻在**一開始就被 yfinance 大量 rate limit**（`Too Many Requests`），退避重試、`Ticker.history()` fallback 全部無效，甚至開始出現大量 60 秒逾時（`⏱️ ...強制終止`），跑了 20+ 分鐘 **0 檔成功、167 個失敗**，判斷是 yfinance 端對這個 IP 的持續性限流（不是我們程式的問題，也不是之前處理過的個別股票卡死），手動停掉沒有再空等到 1.5 小時 timeout。
- **`_daily_update.py` 的 `MIN_REGENERATED_PARQUET=950` 新鮮度門檻這次正確發揮作用**：因為新產生的 parquet 遠低於門檻，`_last_train_anchor.txt` 沒有被更新（仍是 2026-09-17），沒有拿不完整/舊資料去訓練產生假的新預測，符合設計預期。
- **結論**：yfinance 限流會不定期發生，不只是先前手動測試時遇到過，往後任何一次排程都可能撞上。**這種情況不用特別修，讓門檻機制擋下來就好**，之後排程（今晚 21:30 或之後手動重跑）會自動重試；如果連續好幾天都失敗才需要進一步調查（例如降低 `MAX_CONCURRENT`、加大 retry 間隔，或換一個沒被限流過的網路環境）。
- **同一天發生過一次誤判，記錄下來避免以後重蹈覆轍**：使用者當時說「網路重連了」，一度讓人以為前一次失敗是本機網路斷線（不是 yfinance 限流），於是立刻重跑一次——結果**連 2330.TW（台積電）都被回報「可能已下市查無資料」**，這種指標股不可能下市，說明還是限流，不是本機網路問題。直接測 `https://query1.finance.yahoo.com/v8/finance/chart/2330.TW` 拿到 **HTTP 429**、回應只花 0.7 秒（代表連線本身完全正常，是對方主動拒絕），確認就是 Yahoo/yfinance 端對這個 IP 的限流，跟使用者的網路重連只是時間上巧合、無關。**下次再遇到大量 `Too Many Requests` 或「possibly delisted」洗版時，先用這個方式直接測一支绝對還在市的大型股（如 2330.TW）驗證是不是真的限流，不要因為使用者說網路狀況變化就急著重跑，重跑只會讓限流視窗延長。**當下處理方式：停掉、不空等，等之後（例如今晚排程或隔一段時間）自然恢復再試。

### 2026-10-01~04：找到並修好 yfinance 限流的根本原因——其實是自己打太猛，不只是對方的問題
一直到 10/4 才真正解決，中間還交織了另外兩個獨立問題：

1. **架構 bug：`isin.twse.com.tw` 股票清單抓取沒有 `if __name__=='__main__'` 保護、也沒設 timeout**。Windows 的 `multiprocessing` 用 spawn 方式啟動 worker，每個 worker process 都會重新 import 整個檔案，等於每個 worker 都各自重打一次這個請求。2026-10-01 晚上這個網站剛好暫時連不上，結果**全部 ~1097 個 worker 同時卡在這個請求上，被外層 60 秒逾時機制強制終止，造成當次 merge 100% 失敗（0 檔成功）**。
   - **修法**：改成 `fetch_stock_codes()` 函式，只在 `if __name__ == "__main__":` 裡呼叫一次（worker 不需要也不會再呼叫它），請求加 `timeout=30`，失敗時退回用 `parquet_data/*.TW.parquet` 既有檔名當股票清單（頂多漏掉當天新上市的個股）。
2. **使用者直接點出的關鍵問題**：「是不是你呼叫太頻繁？」——查證後確認**merge 腳本原本就沒有做過任何基礎節流**：10 個 worker 一有空位就立刻打下一檔、完全沒有間隔，只有「單一檔案失敗後」才有退避重試。對照 `stock_data_down_load_chips.py`（有 `sleep_sec=0.8`）跟 `download_financial_features.py`（有完整 FinMind 配額冷卻機制），**只有這支抓 yfinance 股價的腳本從一開始設計就缺這塊**，不是哪次修改時被拿掉的。
   - **修法**：`MAX_CONCURRENT` 從 10 降到 5，新增 `STAGGER_SEC=0.5`，每派發一個新 worker 就睡 0.5 秒再派下一個，大幅降低瞬間打過去的併發速率。
   - **驗證結果（2026-10-04）**：套用節流後重跑，**1077 檔成功、0 個限流失敗**，跟之前好幾次 0 檔成功、上百個失敗形成明顯對比，確認「自己打太猛」確實是主因（或至少是可控的那部分主因），不是單純運氣不好。
3. 這段期間還發生過一次跟程式無關的插曲：**2026-09-30 23:00 使用者自己手動把電腦睡眠**，當時 merge 跑到一半被直接砍斷（Windows 事件記錄確認「關機原因：Kernel API」正好吻合時間點）。這不是 bug，只是提醒之後排程期間（21:30~22:30 左右）盡量避免手動睡眠電腦；討論過加「訓練完成跳出 Windows 通知」的功能但使用者還沒決定要不要加。
- **教訓**：往後任何「對同一個外部服務短時間內高併發、零節流」的下載邏輯，都要預先加基礎節流（固定間隔或降併發數），不要等撞到限流才回頭查——這次拖了快兩週（9/17 到 10/4 預測都沒更新）才真正解決，中間還疊加了好幾個其他問題一起干擾判斷。
- **小插曲（除錯過程的教訓）**：2026-10-04 晚上曾經誤判一個其實還活著的背景程式已經死掉（`Get-Process python` 剛好有一次查詢落空），因此意外重複啟動了第二個 instance，後來發現並砍掉重複的那個才沒釀成問題。**之後遇到「看起來沒在跑」的情況，除了 `Get-Process` 之外也要對照 log 檔案是否仍有進度在更新，不要憑單次查詢結果就斷定程式已經結束。**

### chip_daily / fundamental_daily 補資料筆記（2026-09-08）
- **anchor_date 只被 `chip_daily` 卡住，`fundamental_daily` 不會**：`downLoad_Tech_and_merge_with_Chips.py` 合併時 `chip` 是跟 yfinance 股價 **inner join**（會截斷日期範圍），`fund`（fundamental_daily）是 **left join** 且 `pe_ratio`/`pb_ratio`/`div_yield` 本來就在 `NULLABLE_COLS` 白名單裡，缺值不會被 `dropna()` 砍掉、也不會截斷日期。所以只要 `chip_daily` 更新到最新，anchor_date 就能推進，`fundamental_daily` 沒跟上不影響。
- `stock_data_down_load_chips.py`（TWSE 籌碼）呼叫方式：`python -c "from stock_data_down_load_chips import download_chip_range_to_sqlite; download_chip_range_to_sqlite('YYYY-MM-DD','YYYY-MM-DD')"`，有 skip-if-exists，可以放心重跑補區間。
- **`download_fundamentals.py` 的 `fetch_bwibbu()`（TWSE BWIBBU_d API）2026-09-08 發現會被 TWSE WAF 間歇性擋掉**，回傳 307 導向一個 HTML 頁面（「因為安全性考量，您所執行的頁面無法呈現」），跟先前爬 MOPS 被擋是同一套 WAF 機制（見下方 `download_financial_features.py` 段落）。`fetch_bwibbu()` 目前的例外處理會把「WAF 擋掉（拋例外）」跟「真的非交易日（stat!=OK）」都吃掉、統一印成 `[SKIP] ... 非交易日或無資料`，兩者無法從 log 分辨，容易誤判成「資料源沒有這天」。這次觸發原因懷疑是同一個 session 短時間內對 twse.com.tw 打太多請求（chip_daily 回補 + 兩次 fundamentals 嘗試 + 手動測試）累積觸發限流。**因為不影響 anchor_date（見上一點），這個問題可以先擱置，之後有空再補 `fundamental_daily`，不用急著解決或修程式碼。**

### 目前資料範圍（2026-09-08）
- `chip_daily`：已更新到 **2026-09-07**（2026-09-08 這次補的）
- `fundamental_daily`：卡在 **2026-08-07** 沒動（被上述 WAF 問題擋住，2026-09-08 兩次嘗試都失敗），不影響 anchor_date 推進，之後有空再補
- `monthly_revenue`：1078/1094（TWSE/TPEx OpenAPI 整批，當次執行即完成，之後每次執行都會自動更新最新一期）
- `quarterly_financials`：**revenue_q 等損益表欄位 1021/1094**（TWSE/TPEx OpenAPI，覆蓋率已經很完整），**inventory 等資產負債表明細 959/1094**（FinMind，剩 135 檔，2026-09-03 為 644 → 2026-09-04 補到 798 → 2026-09-06 這次 session 一次補到 959，這次因為剩餘待補股票數少，2 次請求/檔沒有超過 600 次配額，跑完整個 `run_download()` 迴圈都沒觸發配額暫停就自然結束了）

**使用者決定（2026-09-03）**：不會一次把 FinMind 補滿，改成**每次想到就跑一段時間**慢慢補（`python download_financial_features.py`，新版會自動衝配額、遇到限流自動暫停再繼續）。**2026-09-04 進一步明確**：不需要跑滿一小時，只要打到 600 次/小時配額上限（log 出現 `[配額暫停] ... status=403 msg='ip banned'`）就代表這次額度用完，應立刻 `Stop-Process` 停掉，不必等 `retry_after`（通常 1800 秒）自動恢復——因為反正下次執行會用同一組 `(Ticker, dataset)` skip 邏輯自動接續，逐檔 commit 不會遺失進度。**注意（2026-09-06）**：待補股票數少於 ~300 檔時，一次 session 可能不會觸發配額暫停就直接跑完全部 1094 檔（正常結束、process 自然退出，非錯誤），這代表當次已經把「FinMind 有資料的股票」都補齊了；剩下沒補到的股票有可能是 FinMind 本身就沒有資產負債表資料（下市/ETF/特殊產業別等），不是配額問題，之後重跑不會再增加。

**下次先做**：
1. （持續）不定期跑 `python download_financial_features.py`，看到 log 出現 `[配額暫停]`（403 ip banned）就代表 600 次配額用完，立刻停掉即可（不用等自動恢復）；如果 process 自然跑完結束（沒有配額暫停訊息），代表這次待補股票都在配額內處理完了，把 `inventory`（FinMind BS/CF）覆蓋率從 959/1094 繼續往上補，但要留意剩下的 135 檔可能是 FinMind 本身沒資料，不一定補得起來
2. 到 Paperspace 網頁手動 Start notebook（見下方「Paperspace 環境」，free tier 不能用 API 啟動）
3. 上傳新 chip.db → 跑 merge（重新產生 parquet，讓新加的 5 個季報衍生特徵欄位有實際數值，目前 parquet 裡這 5 欄還是舊資料、幾乎全是 NaN）→ 跑 `topK_predict.py`（Model C，24 特徵）→ 下載新結果
   （這一步 2026-08-27 已經評估過，使用者因為預估 2.5-3 小時太長選擇先不做，下次接手時直接從「Start notebook」開始，覆蓋率不用等到 1094/1094 全滿才能跑，現有 959/1094 已經非常接近完整，可以視情況先跑一輪）

### 新機器環境筆記（2026-07-29，C:\Users\User\stock_analysis）
- 這台機器裝的是 Python 3.11.9（winget 裝的，路徑會自動加進 PATH，新開 PowerShell 應該就抓得到 `python`/`py`）
- **這台機器有 TLS 攔截（防毒/資安軟體），Python 內建憑證清單認不出攔截用的根憑證**：
  - 現象：`pip install` 和 `requests.get()` 都會丟 `CERTIFICATE_VERIFY_FAILED`，但 PowerShell/.NET 的 HTTPS 完全正常（因為它們走 Windows 憑證存放區）
  - 已裝 `pip-system-certs` 套件解決（讓 Python 的 ssl 改用 Windows 憑證存放區），之後 `requests`/`pip` 在這台機器上都應該正常
  - 如果換一台新機器又遇到一樣的 SSL 錯誤，直接 `pip install pip-system-certs`（如果連 pip 本身都連不上，先加 `--trusted-host pypi.org --trusted-host files.pythonhosted.org` 繞過裝這個套件，裝完之後其他套件就不用再加這個參數了）
- `stock_data_down_load_chips.py` 檔案最底部原本有一行沒包在 `if __name__=="__main__"` 裡的 `download_chip_range_to_sqlite('2023-06-08','2026-06-26')`，只要 `import` 這個檔案就會觸發整段重跑（幸好有 skip-if-exists 邏輯，正常情況下只是浪費時間，但這次剛好撞上 SSL 問題導致洗出一堆假失敗訊息）。已經註解掉，之後這行不會再自動執行。
- Paperspace notebook 的 free tier 閒置 **約 1 分鐘**就會自動關機（`shutdownTimeout` 雖然寫 1，實際上是 1 分鐘不是 1 小時），所以要連續操作、不要中間停太久再回來查

### 本機 GPU 訓練環境（2026-09-19 建好，取代/備案 Paperspace）
起因：Paperspace 免費 A4000 尖峰時段常「沒有可用 machine」，而且每次都要手動 Start、會被閒置關機、要上傳 172MB chip.db。發現使用者這台電腦本身有 **GTX 1080 Ti（11GB，Pascal sm_61）**，於是設定成本機訓練環境：
- **雙系統注意**：這台電腦是 Windows + Linux 雙系統（GRUB 2.06），使用者已把 GRUB 預設開機改成 Windows（原本沒按鍵會直接進 Linux，導致排程被中斷過）。
- **驅動**：NVIDIA 560.94，最高支援 CUDA 12.6 → 只能用 **cu126** 版 PyTorch（cu128 需要驅動 ≥570）。
- **安裝過程**：(1) 原本裝的是 `torch 2.13.0+cpu`，且缺 **Microsoft Visual C++ Redistributable** 導致 `c10.dll` 載入失敗 → 用 `winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget --silent` 安裝（**必須加 `--source winget`**，因為 msstore 來源會因 TLS 攔截憑證錯誤失敗；安裝需要使用者在 UAC 視窗按允許，視窗可能藏在工作列）。(2) 換成 `pip install torch --index-url https://download.pytorch.org/whl/cu126`（2.6GB）→ `torch 2.14.0+cu126`，arch list 含 `sm_61`，已用實際 matmul/Linear 驗證 GPU 運算正常。
- **yfinance 在本機的 TLS 問題**：`curl_cffi`（yfinance 使用）不走 Windows 憑證存放區，會報 `curl: (60) SSL certificate problem`（跟先前 pip/requests 的問題同源，`pip-system-certs` 只修得了 Python ssl，修不了 curl_cffi）。解法：`_make_ca_bundle.py` 把 certifi + Windows ROOT/CA 憑證合併成 `ca_bundle.pem`，並用 `setx CURL_CA_BUNDLE "C:\Users\User\stock_analysis\ca_bundle.pem"` 存成使用者環境變數（新開的 shell/排程都會吃到）。已驗證 yfinance 可正常下載。若之後 Windows 憑證更新或換電腦，重跑 `_make_ca_bundle.py` 即可。
- **腳本不用改**：`downLoad_Tech_and_merge_with_Chips.py`（DB_PATH="chip.db"、OUT_DIR="parquet_data" 皆相對路徑）與 `topK_predict.py`（`DEVICE = cuda if available`、輸出 CSV 到當前目錄）都可以直接在本機專案資料夾執行。依賴（yfinance/pyarrow/numpy/pandas/tqdm/curl_cffi/sklearn）皆已安裝。
- **本機 `parquet_data/` 是 2026-07-29 的舊版**，要先跑一次 merge 重新產生才能訓練。本機跑完整 merge 需要重新下載 ~1100 檔股價（曾在 Paperspace 首次從零跑要 60-90 分鐘，有快取時只要 1 分鐘）。
- 本機化的最大好處：不用手動 Start notebook、不會被閒置關機、不用上傳 chip.db，整條流程理論上可以完全自動化（甚至排程）。

### 缺值政策
**存 NULL，不補 0**（籌碼缺值語意是「不明」，補 0 會汙染訓練資料）。

### 每日自動更新排程（2026-09-17 新增）
使用者問「能不能每天固定時間自動更新」，討論後結論：Claude session 本身無法做到真正每天自動觸發（CronCreate 只在 session 存活期間有效且 7 天過期；雲端排程 agent 又存取不到本機檔案），所以改用 **Windows 工作排程器（Task Scheduler）直接排程執行既有腳本**，不需要 Claude 介入。

- **新增 `_daily_update.py`**：整合三個步驟依序執行，每次都寫一份帶時間戳記的 log 到 `daily_update_logs\`：
  1. `stock_data_down_load_chips.download_chip_range_to_sqlite()` + `backfill_foreign_holding_pct()` 補最近 10 天（TWSE，免配額，skip-if-exists 成本很低）
  2. `python download_financial_features.py`（FinMind 資產負債表/現金流量表/損益表 + 月營收，本來就有配額冷卻機制，這裡包一層 3 小時 timeout 安全上限）
  3. `python download_fundamentals.py`（TWSE PE/PB/殖利率，偶爾被 WAF 擋掉會失敗，這是已知、可接受的行為，之後排程會自動補）
  - 用 `_daily_update.lock` 檔案防止排程重疊執行（前一次還沒跑完，這次直接跳過）。
  - **重要 bug 修復**：一開始用 `subprocess.run(cmd, stdout=log, ...)` 直接把 Python 檔案物件當 `stdout` 傳給子程序，子程序寫入時用的是 Windows 主控台預設編碼（cp950），跟父程序開檔案指定的 UTF-8 不一致，導致 log 裡中文全部亂碼（數字英文正常）。修法：`subprocess.run` 額外傳 `encoding="utf-8"` 且透過 `env=` 設定子程序的 `PYTHONIOENCODING=utf-8`，統一編碼後中文就正常了。**之後任何用 `subprocess.run(..., stdout=<file>)` 寫中文 log 的地方都要記得這樣做，不然會踩到一樣的坑。**
- **Windows 工作排程器設定**：工作名稱 `StockAnalysis_DailyUpdate`，觸發時間**平日（一~五）晚上 21:30**（2026-09-19 由 19:00 改成 21:30，原因：19:00 時 TWSE 當天融資融券還沒公布，19:00 跑會永遠慢一個交易日；21:30 才能吃到當天資料。下次執行 2026-09-21 21:30），用 PowerShell `Register-ScheduledTask` 建立，之後用 `Set-ScheduledTask -TaskName ... -Trigger/-Settings` 修改：
  - `-WakeToRun`：允許把睡眠中的電腦叫醒執行（**前提是電腦要是睡眠狀態，不是完全關機**——關機的話 Task Scheduler 無法叫醒，只能靠下面這條補跑）
  - `-StartWhenAvailable`：如果排程時間電腦沒開，下次開機/上線時會自動補跑一次
  - `-ExecutionTimeLimit 4小時`、`-MultipleInstances IgnoreNew`（避免跟下一次排程重疊）
  - 查看/管理：工作排程器 App 搜尋 `StockAnalysis_DailyUpdate`，或 PowerShell `Get-ScheduledTask -TaskName "StockAnalysis_DailyUpdate"`；要移除用 `Unregister-ScheduledTask -TaskName "StockAnalysis_DailyUpdate" -Confirm:$false`。
- **使用者的電源習慣**：討論過睡眠 vs 關機的差異（睡眠才能被排程器喚醒；關機的話只能靠開機後補跑），使用者決定先讓電腦保持睡眠模式試試看排程效果。
- **2026-09-17 手動測試一次確認整條資料更新流程可以正常跑完**（exit code 0，lock file 正常清除）。
- **2026-09-19 擴充：`_daily_update.py` 現在還包含 Step 4 merge + Step 5 本機 GPU 訓練 + 結果存檔**（取代 Paperspace 手動流程，Paperspace 只當備案）：
  - **Step 4 merge**：`downLoad_Tech_and_merge_with_Chips.py`（~25 分鐘）；跑完檢查「本次新產生的 parquet 檔數 ≥ 950」（`MIN_REGENERATED_PARQUET`），不夠（例如 yfinance 連不上）就**不訓練**，避免拿舊資料產生假的新預測。**Step 5 訓練**：`topK_predict.py`（~12.5 分鐘），成功且 3 個結果 CSV 都是新的，才把它們複製到 `predictions_archive\<時間戳>_anchor<日期>_live_prediction_*.csv`（**歷史預測從此有留存**，之前每次都被覆蓋，9/11 那次還因此遺失），並把 anchor 寫進 `_last_train_anchor.txt`。
  - **Step 4/5 的閘門**：只有 `MAX(Date) FROM chip_daily` ≠ `_last_train_anchor.txt` 才會執行（沒有新交易日就略過，避免假日/重複執行浪費 GPU）；`--force-train` 可忽略閘門，`--only-train` 可跳過步驟 1~3。各重步驟 timeout 1.5 小時，每次 log 存在 `daily_update_logs\`（30 天後自動清除），子程序環境變數：`CURL_CA_BUNDLE`（`ca_bundle.pem`，yfinance TLS）、`TQDM_MININTERVAL=30`（避免進度條灌爆 log）、`PYTHONIOENCODING=utf-8`。排程工作執行上限放寬為 6 小時。
  - **排程時間 2026-09-19 由 19:00 改為 21:30**（19:00 時 TWSE 當天融資融券還沒公布，會永遠慢一個交易日）。
  - **⚠️ 2026-09-21 發現並修復的漏洞（睡眠會凍結背景工作）**：測試時使用者出門、電腦在 merge 結束後、訓練跑到一半（9/19 ~11:26）被 Windows 的閒置睡眠計時器（AC 30 分鐘）睡著，訓練直到 9/21 晚上電腦被喚醒才繼續跑完——**CPU 密集的背景程式不會阻止自動睡眠**。修法：`_daily_update.py` 啟動後呼叫 `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` 告訴 Windows 這段期間不要自動睡眠（`keep_awake()`，結束時解除）。這個問題在「排程喚醒電腦跑 ~45 分鐘」的情境本來就會發生，不修的話排程會在半途被凍結。
  - **同時修了 lock 判斷**：原本 lock 超過 7 小時就當孤兒清掉，但上面那次睡眠讓一個「仍活著、只是被凍結」的程序被誤判成孤兒，會導致新排程跟舊程序**同時**跑（兩個 merge/訓練互相覆蓋 parquet）。現在改成 lock 內記錄 PID，用 `tasklist` 檢查該 PID 是否還活著（**不能用 `os.kill(pid, 0)`：Windows 上會直接終止該程序**），活著且不到 24 小時就跳過這次，PID 已死才清除 lock。
- **使用者的電源設定**：目前 AC 電源閒置 30 分鐘自動睡眠（`powercfg` 的 STANDBYIDLE = 0x708 秒）。**這個排程只涵蓋資料更新＋本機重訓＋預測存檔**，不含 Paperspace（Paperspace free tier 沒有 API 可以啟動 notebook，現在只當備案）。

---

## 模型：PatchTST Transformer

### 特徵（Model C，25 個，2026-09-16 更新）
```python
FEATURE_COLS = [
    # 技術面（7）
    "Close", "High", "Low", "Open", "Volume", "MA10", "RSI",
    # 籌碼面（8，2026-09-16 加入 foreign_holding_pct）
    "margin_buy", "margin_sell", "short_buy", "short_sell",
    "foreign_net", "invest_net", "dealer_net", "foreign_holding_pct",
    # 每日基本面（3）
    "pe_ratio", "pb_ratio", "div_yield",
    # 月營收（2）
    "rev_yoy", "rev_mom",
    # 季報衍生指標（5，使用者建議的存貨周轉率/合約負債領先指標，2026-08-27 加入）
    "contract_liability_qoq", "inventory_qoq",
    "inventory_turnover", "gross_margin_q", "gross_margin_qoq",
]
```
這幾組稀疏特徵（季報衍生指標 5 個 + 新加的 `foreign_holding_pct`）覆蓋率不保證全市場（見上方 chip.db 現況），但不會因此讓股票被排除訓練：
`downLoad_Tech_and_merge_with_Chips.py` 的 `merge_quarterly_to_daily()` 一定會把這幾欄 merge 進每支股票的 parquet（沒資料就整欄 `None`），而且都在 `NULLABLE_COLS` 白名單裡不會被 `dropna()` 刪列；`topK_predict.py` 的 `safe_read_parquet()` 對缺值走既有的 `.ffill().bfill().fillna(0)`，跟 `pe_ratio`/`rev_yoy` 這些本來就會缺值的欄位處理方式一致。實測過 1094 個 parquet 檔只有 3 個是舊版產出、缺這 5 欄，重新跑一次 merge 就會補上。

**2026-09-16 新增 `foreign_holding_pct`（外資及陸資持股比率，存量指標）**：使用者建議跟現有的 `foreign_net`（外資當日買賣超，流量指標）並存當作互補特徵。
- **資料源**：TWSE `MI_QFIIS`（免金鑰、無配額，逐日查詢，跟 `MI_MARGN`/`BWIBBU_d` 同等級的公開 API），欄位「全體外資及陸資持股比率」。
- **實作**：`stock_data_down_load_chips.py` 新增 `fetch_qfiis_json()`，整合進 `download_chip_data()` 一起抓（跟融資融券/三大法人同一天一起下載），`upsert_chip()` 寫進 `chip_daily.foreign_holding_pct` 新欄位（`init_sqlite()` 有做 `ALTER TABLE` 相容舊資料庫）。新增 `backfill_foreign_holding_pct(start_date, end_date)` 專門回補這個功能上線前就已經存在的歷史交易日（因為 `download_chip_range_to_sqlite()` 原本的 skip-if-exists 邏輯是看「這天有沒有任何資料」，不會自動幫舊資料補新欄位，需要這支獨立的回補函式，UPDATE 而不是 INSERT）。
- **回補結果（2026-09-16）**：793 個交易日（2023-06-08 ~ 2026-09-10）全部跑過，**85.0% 資料列（802,183/943,814）有值**，覆蓋率跟其他既有稀疏欄位相近，剩下的多半是 ETF/新上市/當天無外資申報異動的個股，正常現象。
- **合併流程**：`downLoad_Tech_and_merge_with_Chips.py` 的 `query_chip()` 已加入這欄，`NULLABLE_COLS` 也加了，不會截斷日期或砍掉整列。
- **⚠️ 下次重跑 Paperspace 流程前提醒**：這個新特徵要等**下一次 merge** 才會真正進到 parquet 檔案、topK_predict.py 才吃得到（`FEATURE_COLS` 已經改成 25 個），現有的 parquet_data（Paperspace 上跑到 2026-09-11 那次的）還沒有這欄，直接拿舊 parquet 訓練會被 `safe_read_parquet()` 的 missing-column 檢查整檔排除。

### 超參數
```python
TIME_STEP   = 126   # 用 126 天歷史窗口
FUTURE_DAYS = 60    # 預測未來 60 天最大報酬
TRAIN_YEARS = 3     # 訓練資料回推 3 年
N_RUNS      = 5     # 同一組資料訓練 5 次取平均
TOP_K       = 20    # 輸出 Top-20
LONG_POOL   = 40    # reversal filter 從 Top-40 開始篩
NUM_EPOCHS  = 10
BATCH_SIZE  = 512
LR          = 5e-4
```

### 訓練標籤
- **Y_max**（Long model）：`log(max_future_price / anchor_price)`
- **Y_min**（Reversal model）：`log(min_future_price / anchor_price)`

### 輸出
- `live_prediction_top_B.csv`：Top-20 純 Long model（主要參考）
- `live_prediction_top_C.csv`：Top-20 Long + reversal filter（參考用）
- `live_prediction_all.csv`：全部股票預測分數

---

## 完整工作流程（每週更新）

### 步驟 1：更新本機資料
```bash
# 更新籌碼（TWSE）
py -c "from stock_data_down_load_chips import download_chip_range_to_sqlite; download_chip_range_to_sqlite('YYYY-MM-DD', 'YYYY-MM-DD')"

# 更新基本面（FinMind）
py download_fundamentals.py

# 更新月營收（FinMind，每月月底後）
py download_financial_features.py
```

### 步驟 2：啟動 Paperspace
1. 瀏覽器或 API 確認 notebook `nm0k7kyrma` 是 Running
2. 查詢最新 FQDN + token
3. 建立新 kernel，取得 kid
4. 更新 `_run_remote.py`、`_run_merge.py`、`_check_*.py` 裡的連線資訊

### 步驟 3：上傳更新後的 chip.db
```python
import requests, base64
with open('chip.db', 'rb') as f:
    content = base64.b64encode(f.read()).decode('utf-8')
requests.put(f'https://{fqdn}/api/contents/chip.db',
    headers={'Authorization': f'token {token}'},
    json={'type': 'file', 'format': 'base64', 'content': content}, timeout=300)
```

### 步驟 4：在 Paperspace 執行 merge（用 conda env Python）
```python
# 透過 kernel WebSocket 執行
code = """
import subprocess
log = open('merge.log', 'w', buffering=1)
proc = subprocess.Popen(
    ['/storage/conda_envs/stock_env/bin/python', '-u', 'downLoad_Tech_and_merge_with_Chips.py'],
    stdout=log, stderr=log, start_new_session=True)
print(f'Launched merge PID={proc.pid}')
with open('merge.pid', 'w') as f: f.write(str(proc.pid))
"""
# 約 60-90 分鐘，用 _check_merge_log.py 監控
```

### 步驟 5：執行 topK_predict.py（用 GPU Python）
```python
code = """
import subprocess
log = open('predict.log', 'w', buffering=1)
proc = subprocess.Popen(
    ['/usr/local/bin/python3', '-u', 'topK_predict.py'],
    stdout=log, stderr=log, start_new_session=True)
print(f'Launched predict PID={proc.pid}')
with open('predict.pid', 'w') as f: f.write(str(proc.pid))
"""
# 約 60-90 分鐘，用 _check_predict_log.py 監控
```

### 步驟 6：下載結果
```python
import requests, base64
for fname in ['live_prediction_top_B.csv', 'live_prediction_top_C.csv', 'live_prediction_all.csv']:
    r = requests.get(f'https://{fqdn}/api/contents/{fname}',
        headers={'Authorization': f'token {token}'})
    content = base64.b64decode(r.json()['content'])
    with open(fname, 'wb') as f: f.write(content)
```

---

## ⚠️ 重要發現（2026-09-21）：同樣資料重訓兩次，Top-5 名單變動大，多半是隨機雜訊
用「完全相同的資料」（anchor 2026-09-17，同一批 parquet）連續訓練兩次（`predictions_archive\20260919_104246_...` vs `20260921_192329_...`），比對結果：
- **Top-5 只重疊 1/5、Top-10 重疊 5/10，但 Top-20 重疊 19/20，全市場 1026 檔的預測排名 Spearman 相關 = 0.998**。也就是說**模型整體是穩定的**（大多數股票的相對排序幾乎一樣，「有潛力的那一群」很一致），但**群組內部的精確排名（誰是第 1 名、誰是第 5 名）非常吵**——第一次的前 3 名（群創 0.538 / 友達 0.523 / 聯電 0.521）在第二次裡分數都掉到 0.38 附近、不在前 8，第二次的第 1 名是欣興 3037（0.431）。
- 原因：`topK_predict.py` 沒有固定隨機種子，N_RUNS=5 平均後 Top 端仍有明顯波動。
- **影響**：這幾週我們一直在解讀「加了某特徵/修了某 bug 之後名單怎麼變」（例如 9/11 「華邦電掉到第 7、聯電掉出 Top-20」被當成修 bug 的證據；9/19 的變動被拿來對比 9/11），**這類 Top-5 排名變動有很大一部分其實是隨機雜訊，不能單靠名單變動下結論**。之前 memory 裡「證實舊排名部分是壞資料造成」這句話需要打折扣看待。
- **建議（尚未做）**：(1) 固定隨機種子讓結果可重現，或把 N_RUNS 加大（例如 5→15+）降低 Top 端變異；(2) 看名單時以「Top-20 集合」而不是精確名次為主，或改用多次訓練的排名平均；(3) 想評估特徵/修 bug 有沒有幫助，要靠**回測**（`predictions_archive\` 已開始累積歷史預測），不要用名單長相判斷。

## 回測（2026-10-05，`backtest_current.py`，結果 `backtest_results.csv`）
- **方法**：錨點 2026-06-01 與 2026-07-09；特徵、標準化、訓練標籤都只用 ≤ 錨點的資料；季報衍生特徵延遲 50 天、月營收延遲 15 天才看得到（避免前視偏誤）；評分 = 錨點後 60 個交易日的實際報酬；N_RUNS=3。Model B = 純 long Top-20，Model C = long + 反轉過濾 Top-20。
- **結果（期末報酬 vs 全市場平均）**：
  - 2026-06-01：B 超額 -3.95%（命中 40%），C 超額 -3.79%（命中 30%），期末 IC -0.23，十分位價差 -7.4%
  - 2026-07-09：B 超額 +10.94%（命中 70%），C 超額 -1.85%（命中 50%），期末 IC +0.03，十分位價差 +8.7%
  - 期間最高報酬的 IC 兩個錨點都是正的（+0.08 / +0.10），但期末報酬的方向在兩個錨點之間翻轉。
- **解讀**：結果混合，不能當成模型有效的證據。只有 2 個錨點、N_RUNS=3，變異很大。反轉過濾（Model C）沒有穩定帶來改善，07-09 還明顯變差。
- **已知限制**：延遲只能降低、不能消除前視偏誤；這是粗略檢查，不是獲利證明。
- **6/1 失準原因（2026-10-05 查證）**：6/1 之後市場偏弱（59% 股票下跌），而且**高波動股大跌**：依前 60 日波動分五組，最高波動組期末報酬中位數 -16.8%，最低組 +1.2%，兩者 Spearman = -0.36。模型 Top-20 全部都是高波動股（前 60 日年化波動中位數 0.84，全市場中位數 0.39，20 檔全部 > 0.6）。原因推測：模型的訓練標籤是「60 日內最高報酬」，這個目標天然偏好高波動股（最高點離起點越遠）；在風險趨避的市場裡這就變成系統性虧損。7/9 錨點沒有這個效應（波動與報酬 Spearman ≈ -0.05，高波動組反而略好），所以 7/9 的超額報酬不是模型選股能力的證據，而是當時剛好沒有波動懲罰。**結論：模型目前學到的主要可能是「波動暴露」，不是穩定的選股訊號。** 若要改進，應考慮波動中性化或風險調整後的標籤（例如報酬除以波動），並在多個不同市場環境的錨點上驗證。

## 最新預測結果（2026-10-04 訓練完成，anchor 2026-10-02——結束了從 9/17 拖到 10/4 快三週沒更新的窘境）
這次是排程/手動重跑 `_daily_update.py`（含 Step 4 merge + Step 5 訓練）首次真正順利跑完，用的是套用節流修復（`MAX_CONCURRENT=5`/`STAGGER_SEC=0.5`）後的 merge：1077 檔新產生、0 個限流失敗。21:49 啟動、22:27 完成，總共約 38 分鐘。

**Model B（純多方）Top-5**：
1. 2409.TW 友達 (0.4505)
2. 1303.TW 南亞 (0.4397)
3. 6451.TW 訊芯-KY (0.4363)
4. 6446.TW 藥華藥 (0.4271)
5. 3037.TW 欣興 (0.4229)

**Model C（多方+反轉篩選）Top-5**：1303 南亞、6446 藥華藥、3037 欣興、6770 力積電、6442 光聖

結果已存進 `predictions_archive\20261004_222720_anchor2026-10-02_*`，`_last_train_anchor.txt` 已更新為 `2026-10-02`。**這次中斷了近三週才恢復更新**，完整起因鏈見上方「2026-10-01~04：找到並修好 yfinance 限流的根本原因」段落：FinMind 80 天 skip 邏輯 bug（9/22 修好，9/24 補完種子資料）→ yfinance 自己打太猛造成的限流（10/1~10/4 才診斷出並修好）→ `isin.twse.com.tw` 架構 bug 疊加（10/1 修好）→ 9/30 使用者自己手動睡眠電腦中斷一次（非 bug）。

## 最新預測結果（2026-09-19，**本機 GPU 訓練**，含 foreign_holding_pct，25 個特徵，anchor 2026-09-17，供對照）
第一次完全在本機（GTX 1080 Ti）跑完整流程：merge → 訓練 → 產出 CSV，不需要 Paperspace。本機 `live_prediction_all/top_B/top_C.csv` 已是這次的新版本（時間戳 2026-09-19 10:42）。

**Model B（純多方）Top-5**：
1. 3481.TW 群創 (0.538091)
2. 2409.TW 友達 (0.522848)
3. 2303.TW 聯電 (0.520777)
4. 2344.TW 華邦電 (0.492016)
5. 1303.TW 南亞 (0.447674)
（6-10：6770 力積電 0.441、6451 訊芯-KY 0.424、3037 欣興 0.423、2408 南亞科 0.413、6446 藥華藥 0.413；國巨 2327 排第 11，0.396）

**Model C（多方＋反轉篩選，參考）Top-5**：2303 聯電 (0.521/-0.183)、1303 南亞 (0.448/-0.174)、6770 力積電 (0.441/-0.175)、6451 (0.424/-0.181)、3037 欣興 (0.423/-0.174)

**跟 2026-09-11（Paperspace，未含外資持股比例、anchor 09-07）比**：群創升到第 1（原第 2）、友達降到第 2（原第 1）、聯電回到第 3（9/11 曾掉出 Top-20，這次 0.521）、華邦電回到第 4（9/11 掉到第 7）、南亞維持前 5、南電 8046 從第 5 掉到第 12。**變動來源有三個混在一起，無法單獨歸因**：(1) 新增 foreign_holding_pct 特徵；(2) anchor 從 09-07 推進到 09-17（多了 8 個交易日新資料）；(3) ETF（如 0050）不再出現（本機 parquet 舊檔缺新欄位被自動略過，這是預期行為）。

**本機執行時間（1080 Ti）**：merge 約 25 分鐘（1073 檔，0 失敗）；訓練（5 runs × long+reversal = 10 個模型）約 12.5 分鐘（每個 epoch ~5 秒，GPU 使用率 ~62%，顯存只用 ~570MB）。**比 Paperspace（merge 60-90 分鐘、訓練 60-90 分鐘）快很多**，且不用手動 Start、不會被閒置關機、不用上傳 172MB chip.db。

**這次在本機踩到的坑（已修）**：
- 本機 pandas 3.0.5 / numpy 2.4.6，比 Paperspace 的 pandas 2.x 新很多，舊腳本有相容問題：(1) `pd.read_html(response.text)` 不再接受純 HTML 字串 → 改成 `pd.read_html(StringIO(response.text))`；(2) merge 時 yfinance 的 Date 是 `datetime64[s]`、chip 的 Date 是 `[us]`，`pd.merge` 報 `incompatible merge keys` 導致**每一檔都失敗** → 在 merge 前把所有來源的 `Date` 統一轉成 `datetime64[ns]`（`downLoad_Tech_and_merge_with_Chips.py`）；(3) 本機缺 `lxml`（`pd.read_html` 需要）→ `pip install lxml`。
- pandas 3.0 移除了 `pct_change` 預設的 forward-fill（pandas 2.x 預設 `fill_method='pad'`，Paperspace log 裡的 FutureWarning 就是在講這個），所以本機算出的 `inventory_qoq`/`contract_liability_qoq` 可能比之前 Paperspace 的結果多一些 NaN；影響應該很小但**跟舊結果不是完全同口徑**。
- 24 個舊 parquet 沒被更新（ETF 00xx、TDR 91xx、權證 03xxxx，以及 2867/3432/4766/5515/6806/6921/7711/7791/7803/7823 等已下市或歷史不足的個股），它們缺 `foreign_holding_pct` 欄位，`topK_predict.py` 的 `safe_read_parquet()` 會自動略過，符合預期（ETF/TDR 本來就不適合這個個股模型）。之前預測名單裡會出現 0050.TW（anchor 停在 8/7），就是這類舊檔造成的。

**（2026-09-19 補充）**：使用者一度要求「訓練跑完後留存紀錄並幫他關機」，但在關機指令送出前取消了（改為要求把 merge + 訓練接進每日排程），所以**電腦沒有被關機**。若電腦是完全關機的狀態，排程工作 `StockAnalysis_DailyUpdate` 要等使用者開機並進入 Windows 才會補跑（`StartWhenAvailable`；GRUB 預設已由使用者改為 Windows）；建議讓電腦保持睡眠而不是關機，排程器才能喚醒它執行。

## 上一輪預測結果（2026-09-11 重訓完成，anchor 2026-09-07，損益表單位 bug 修復後，供對照）
用修好損益表單位 bug（見上方「download_financial_features.py 資料來源」段落，income_fetched 1094/1094 全補齊）的 chip.db 重跑一次 Paperspace 流程，anchor_date 維持 2026-09-07（這次目的不是推進日期，chip_daily 沒動）。**merge 這次只花了約 1 分鐘就跑完**（比平常的 60-90 分鐘快很多，推測是 yfinance 股價資料已有快取，不用重新下載）。

**Model B（純多方）Top-5**：
1. 2409.TW 友達 (0.593452)
2. 3481.TW 群創 (0.501373)
3. 6770.TW 力積電 (0.497478)
4. 1303.TW 南亞 (0.427734)
5. 8046.TW 南電 (0.424489)

**跟修 bug 前（2026-09-09）比較，變化明顯**：
- 前三名排序沒變（友達/群創/力積電），但分數普遍降低
- **2344 華邦電從第 4 名（0.494）掉到第 7 名（0.408）**——這正是之前用壞掉的存貨周轉率/毛利率資料時被特別強調的股票，修正後排名下滑，印證舊排名有部分是壞資料造成的假訊號
- **2303 聯電直接掉出 Model B Top-20**（原第 5 名 0.467 → Model C 裡才排到第 10 名 0.344）
- 新進 Top-5：1303 南亞、8046 南電

**⚠️ 待辦**：這次 predict 跑完後、還沒來得及下載 3 個結果 CSV 前，notebook 又被 idle 關機了（下載時 API 連線失敗），**Top-5/Top-20 數字是從 `predict_run_log4.txt` 訓練時印出的 log 直接摘錄的，還沒有正式的 `live_prediction_top_B/C/all.csv` 檔案**（本機那 3 個檔案還是 2026-09-09 的舊版本）。下次開 notebook 時記得先把這次的 CSV 重新下載回來覆蓋舊檔（chip.db/parquet_data 應該還在同一份持久化 storage 上，不用重跑 merge/predict，直接下載檔案就好，除非發現檔案也隨關機消失了才需要重跑）。

**經驗更新**：`_monitor_merge.py`（新增於 2026-09-11，全程佔住一條 WebSocket、每 20 秒 poll 一次 merge.log 直到看到 `=== ALL DONE ===`）這次證實有效防止 merge 被 idle 關機打斷。但 **predict 跑完、`_run_remote.py` 的連線一關閉，notebook 馬上又進入 idle 倒數**，如果沒有立刻在同一個連線或很短時間內下載結果檔案，一樣會被關機打斷下載——之後應該讓下載結果 CSV 這個動作也緊接著用同一精神處理（例如 predict 腳本本身也可以在跑完訓練後、印出 exit code 之前，順便把結果檔案內容直接印出或用同一條連線讀出，不要依賴額外的 REST API call）。

## 最新預測結果（2026-09-09 重訓完成，anchor **2026-09-07**——終於推進了！，供對照）
2026-09-08 補了 chip_daily 到 9/7 後重跑 Paperspace 全流程（上傳→merge→topK_predict.py），這次 anchor_date 成功從卡住兩次的 2026-08-07 推進到 **2026-09-07**，證實「chip_daily inner join 卡住 anchor_date」的診斷正確，修法也有效。過程：
1. 2026-09-08：上傳新 chip.db（含 chip_daily 到 9/7）→ 跑 merge。**中途 notebook 被 idle timeout 自動關機一次**（`_run_merge.py` 觸發後就斷線沒維持連線，關機判斷只看有沒有開著的 WebSocket），使用者手動重開 notebook，還好 merge 在關機前已經跑完（`=== ALL DONE ===`，1100 個 parquet，chip.db 完整），沒有遺失進度。當天决定 predict 明天再跑。
2. 2026-09-09：重開 notebook（免費版每次都要重開，之前的 parquet_data/chip.db 用同一份持久化 storage 都還在，不用重新上傳/重跑 merge），跑 `python -u _run_remote.py *>predict_run_log3.txt`（全程佔住 WebSocket，這次沒有中途被關機）完成訓練。

Model B（純多方）Top-5：
1. 2409.TW 友達 (0.644434)
2. 3481.TW 群創 (0.600549)
3. 6770.TW 力積電 (0.566345)
4. 2344.TW 華邦電 (0.493947)
5. 2303.TW 聯電 (0.466538)

Model C（多方+反轉篩選，參考）Top-5：同上排序（0.644/0.601/0.566/0.494/0.467）

注意：`0050.TW`（元大台灣50 ETF）在 Top-20 名單裡 anchor_date 仍顯示 2026-08-07（跟其他個股不同），這支不受這次 chip_daily 更新影響，先留意但不影響其他個股結果正確性。

Paperspace notebook 這幾天每次重啟 id 都不同（`nj36kn4zia` → `n8m7px0qip` → `n3yaox1v7d` → `ns6alkxyd9`，最新是 `ns6alkxyd9`），每次都要重新查 fqdn/token/kid 更新進 6 支腳本（`_upload_db.py`/`_run_merge.py`/`_run_remote.py`/`_check_merge_log.py`/`_check_predict_log.py`/`_check_remote_files.py`，最後一支是 2026-09-08 新增，用來確認 chip.db/parquet/merge 產物是否在重啟後還存在）。這是正常現象不用特別記錄每個 id，只要記最新的。

### 上次預測（2026-09-07 重訓，anchor 卡在 2026-08-07，供對照——已用上面新結果取代）
Model B Top-5：2409.TW(0.620) / 3481.TW(0.573) / 2303.TW(0.536) / 8046.TW(0.469) / 3037.TW(0.446)

### 更早一次（2026-08-11 重訓完成，anchor 2026-08-07，供對照）
Model B Top-5：
1. 2303.TW 聯電 (0.841)
2. 3481.TW 群創 (0.725)
3. 2409.TW 友達 (0.722)
4. 6770.TW 力積電 (0.678)
5. 2344.TW 華邦電 (0.608)

跌幅 Top10（用 pred_min 排序，數值越負代表預測跌幅越深，存成 `live_prediction_top10_decline.csv`）：
1. 6141.TW 柏承 (-36.9%) 2. 6834.TW 天二科技 (-34.0%) 3. 6715.TW 嘉基 (-32.3%) 4. 6658.TW 聯策 (-29.7%) 5. 6902.TW GOGOLOOK (-29.2%)
6. 6225.TW 天瀚 (-28.3%) 7. 8033.TW 雷虎 (-27.7%) 8. 4989.TW 榮科 (-27.1%) 9. 2231.TW 為升 (-27.1%) 10. 4764.TW 雙鍵 (-26.5%)

上次預測（anchor: 2026-07-29）Model B Top-5（供對照）：
1. 2303.TW 聯電 (0.727) 2. 2409.TW 友達 (0.688) 3. 3481.TW 群創 (0.644) 4. 6770.TW 力積電 (0.540) 5. 8046.TW 南電 (0.505)

### 這次重訓踩到的坑（2026-08-11，重要）
- **merge 腳本（`downLoad_Tech_and_merge_with_Chips.py`）用 ThreadPoolExecutor 時，少數股票在 yfinance/curl_cffi 底層會整批掛死不回應**，加 `timeout=` 參數對 curl_cffi 沒用（Python thread 卡在 C 層呼叫，無法從外部中斷）。現象：merge.log 卡住不動、parquet 檔案數量幾十分鐘不增加，一律卡在同一批代碼（觀察到兩批：00開頭ETF/91開頭TDR、以及 8487/8162/2258/9958/7823/9942 這類 -KY 外國企業代碼）。
  - **永久修法**：把並行機制從 `ThreadPoolExecutor` 改成 `multiprocessing.Process`，每檔股票一個獨立 process，外部用 `time.time()-start_t > 60` 偵測逾時後 `p.terminate()`/`p.kill()` 強制砍掉，因為 OS process 可以被強制殺掉，thread 不行。已經改好在腳本裡（`_worker_entry` + `if __name__=="__main__"` 那段迴圈）。
  - 另外把 `stock_list` 過濾正則從 `^\d{4}$` 改成 `^(?!00)(?!91)\d{4}$`，直接排除 ETF(00開頭)和TDR(91開頭)，這兩類本來就不適合這個個股模型。
- **Paperspace notebook 的閒置關機判斷，看的是「有沒有開著的 kernel WebSocket 連線」，不是「有沒有進度」**：只要連線不斷開（哪怕遠端在跑的 subprocess 卡死沒有任何輸出），notebook 就不會被關機；反之只要连线一斷開（就算只斷 1-2 分鐘），幾乎必定被關機。所以監控腳本要用「單一 WebSocket 連線 + 每 15-25 秒送一次 execute_request + 斷線自動重連」的寫法全程不斷開，不要每次檢查都重新開關連線。**2026-09-08 再次踩到**：`_run_merge.py` 本身只負責觸發遠端背景執行、送出後就斷線（設計上就是 fire-and-forget，靠之後另外開連線用 `_check_merge_log.py` 查進度），這中間的空檔如果沒有其他連線頂著，notebook 就會被 idle 關機——這次剛好在關機前 merge 已經跑完所以沒事，但不能保證每次都這麼幸運。**之後跑 merge 時，建議跟 predict 一樣，用一個會全程佔住連線的方式監控（例如寫一支輪詢腳本，每 15-25 秒查一次 `_check_merge_log.py` 那種 code 但共用同一條連線不要斷開），而不是丟著讓它自己跑、隔一段時間才重新開連線查一次。**
- 同一個 kernel 若同時有兩條 WebSocket 連線在用（例如監控連線還開著、又另外開一條連線送指令），新開的那條有時會完全收不到回應（RESULT 是空的），原因不明；穩妥做法是要送新指令時，直接沿用「舊監控連線同一個 kernel id」而不是另外創建新 kernel，或乾脆先確認舊連線沒在忙。
- merge 腳本（含 subprocess 完成後）偶爾不會乾淨結束 process（`os.kill(pid,0)` 一直回傳存活），但實際工作已經做完（log 顯示 `=== ALL DONE ===` 或 predict.log 顯示 `Saved -> ...csv`），不用等 `STATUS_RUNNING=False`，看到完成標記就可以直接進下一步。

---

## 待完成事項
- [x] 換新機器：Python 環境建好、SSL 憑證問題解決、from_gradient 全部檔案（含 chip.db）下載回本機
- [x] chip_daily/fundamental_daily/monthly_revenue/quarterly_financials 全部更新到 2026-08-07 前後（monthly_revenue 1091/1094、quarterly_financials 314/1094，季報仍受 FinMind 額度限制）
- [x] 新 chip.db 上傳回 Paperspace（改用 WebSocket 分塊上傳，因為檔案太大 REST API 的 PUT 會被 proxy 斷線；細節見下方「上傳大檔案的坑」）
- [x] 修掉 merge 腳本並行卡死問題（ThreadPoolExecutor→multiprocessing + 逾時強制 kill；排除 ETF/TDR），Paperspace 執行 merge → topK_predict.py 重訓完成，anchor 2026-08-07 的新預測已下載回本機
- [x] 下跌 Top10（pred_min 排序）+ FinMind 中文名稱，存成 `live_prediction_top10_decline.csv`（anchor 2026-08-07）
- [x] `download_financial_features.py` 改版（2026-08-27）：月營收+損益表改用 TWSE/TPEx OpenAPI 整批（免配額），FinMind 只剩資產負債表明細+現金流量表兩個 dataset
- [x] 加入合約負債 qoq、存貨周轉率等 5 個季報特徵到 `topK_predict.py` FEATURE_COLS（Model C，24 特徵，2026-08-27），已確認缺值處理方式與既有稀疏特徵一致、不會排除股票
- [x] `download_financial_features.py` 接上使用者的 FinMind 帳號 token（`APIKey.txt`，600 次/小時）+ `finmind_get()` 加全域配額冷卻（收到 429/403 時讀 `retry_after` 自動睡等再繼續，不用人工重啟）（2026-09-03）
- [x] 2026-09-04：跑一次 FinMind session，看到 `[配額暫停]`（403 ip banned）就立刻停掉，不等自動恢復；inventory 覆蓋率 644 → 798/1094，花費約 9-10 分鐘
- [x] 2026-09-06：再跑一次 FinMind session，這次待補股票數已少於配額上限，process 自然跑完沒觸發配額暫停；inventory 覆蓋率 798 → 959/1094，剩 135 檔可能是 FinMind 本身沒有資產負債表資料
- [x] 2026-09-07：完整跑一次 Paperspace 重訓流程（手動 Start notebook → 新增 `_upload_db.py` 上傳 172MB chip.db → 跑 merge → 跑 `topK_predict.py` Model B/C）。新 chip.db 是 959/1094 inventory 覆蓋率。anchor_date 仍是 2026-08-07（已確認是 yfinance 真實資料進度限制，不是流程 bug，見上方「最新預測結果」段落說明），Top-5 結果已下載並記錄。Paperspace notebook 這次重啟 id 變成 `nj36kn4zia`（原 `nfg460g21w`），4 支連線腳本 + 新的 `_upload_db.py` 都已同步更新。
- [x] 2026-09-08：重跑 `stock_data_down_load_chips.py` 把 `chip_daily` 補到 2026-09-07（`fundamental_daily` 被 TWSE WAF 間歇擋掉沒補成，但確認不影響 anchor_date，見上方筆記）
- [x] 2026-09-08~09：用補好的 chip_daily 重跑一次完整 Paperspace 流程，**anchor_date 成功從 2026-08-07 推進到 2026-09-07**，驗證診斷正確。過程中 merge 被 idle 關機打斷一次但已跑完沒事，已記錄「跑 merge 也要全程佔住連線」的教訓
- [x] 2026-09-10：用 `_explain_top5.py` 對這次 Top-5 做特徵解釋分析（近 30 日平均 vs 全市場百分位），發現主因是成交量/融資融券熱度，2344 華邦電額外有基本面轉強佐證
- [x] 2026-09-10：使用者問華邦電存貨週轉天數時，發現並修好損益表資料源單位不一致的嚴重 bug（TWSE OpenAPI 仟元 vs FinMind 元，全市場 1080 檔最新一季損益表數字全部錯誤），改成損益表統一用 FinMind `TaiwanStockFinancialStatements`
- [x] 2026-09-11：再跑一次補完，**損益表 `income_fetched` 1094/1094 全部補齊**，bug 修復完成；`inventory` 順便推進到 1026/1094（剩 68 檔）
- [x] 2026-09-11：用修好的損益表資料重跑一次 Paperspace 流程，確認 Top-5 排名有明顯變化（2344/2303 排名下滑，證實舊排名部分是壞資料造成）。這次驗證 `_monitor_merge.py` 全程佔線監控有效防止 merge 被閒置關機打斷；但 predict 跑完到下載 CSV 之間notebook 又被關機，3 個結果 CSV 還沒補下載（本機仍是 9/9 舊版），Top-5/Top-20 數字目前只有從訓練 log 摘錄，見上方「最新預測結果」段落
- [x] 2026-09-16：使用者建議新增「外資持股比例」特徵，用 TWSE `MI_QFIIS`（免配額）實作 `foreign_holding_pct`，加進 `chip_daily`/`query_chip`/`NULLABLE_COLS`/`FEATURE_COLS`（24→25 個特徵），歷史回補 793 個交易日完成，85% 資料列有值。細節見上方「模型：PatchTST Transformer」段落
- [x] 2026-09-17：設定 Windows 工作排程器 `StockAnalysis_DailyUpdate`，平日晚上 7 點自動跑 `_daily_update.py`（chip_daily/外資持股比例/FinMind 財報/基本面），手動測試過一次成功。細節見上方「每日自動更新排程」段落
- [x] 2026-09-19：Paperspace 免費機器尖峰時段沒有可用 machine，改在**本機 GTX 1080 Ti 建好 GPU 訓練環境**（VC++ 執行庫 + cu126 版 PyTorch + yfinance TLS 憑證修正，見「本機 GPU 訓練環境」段落），並完成第一次本機完整流程（merge ~25 分鐘 + 訓練 ~12.5 分鐘），含 `foreign_holding_pct` 的 25 特徵新預測（anchor 2026-09-17）已產出，見上方「最新預測結果（2026-09-19）」。**本機流程取代 Paperspace 手動流程**（不用 Start、不用上傳、不會被閒置關機）。
- [ ] **下次接手直接做**：（1）**之後重訓一律優先用本機**：在專案資料夾先設好環境變數（`CURL_CA_BUNDLE` 已用 `setx` 永久設定，新開的 shell 才會生效；當前 shell 要手動 `$env:CURL_CA_BUNDLE="C:\Users\User\stock_analysis\ca_bundle.pem"`），依序 `python -u downLoad_Tech_and_merge_with_Chips.py *> local_merge_log.txt`（~25 分鐘）→ `python -u topK_predict.py *> local_predict_log.txt`（~12.5 分鐘）→ 直接產出 `live_prediction_*.csv`。Paperspace 只當備案。（2）✅ **已完成（2026-09-19~21）**：「chip 更新 → merge → 訓練 → 存檔」已串進每日排程（`_daily_update.py`，平日 21:30），端到端測試成功（merge 1072 檔新產生、訓練完成、結果封存到 `predictions_archive\`，`_last_train_anchor.txt`=2026-09-17；原本「待驗證」項目已驗證完畢，同樣資料兩次訓練的比較結果見「⚠️ 重要發現（2026-09-21）」）。測試過程發現並修好「睡眠會凍結背景工作」與「lock 誤判」兩個漏洞，見「每日自動更新排程」段落。（3）`inventory` 覆蓋率繼續跑 `python download_financial_features.py` 補（1026/1094 起，剩 68 檔可能補不起來，排程每天也會順便跑）。（4）找時間補 `fundamental_daily`（卡在 2026-08-07，被 TWSE WAF 擋，之後再試，不急）。（5）**建議優先做**：降低訓練隨機性（固定 seed 或加大 N_RUNS，見「⚠️ 重要發現（2026-09-21）」段落）以及做回測（`backtest_validate.py`；anchor 2026-07-09 的預測已滿 60 天；`predictions_archive\` 從 2026-09-19 開始累積歷史預測）——這幾週所有「名單變動」的解讀都可能被隨機雜訊污染。（6）2026-09-21 21:30 是新版排程第一次「無人值守」的完整執行（含 merge + 訓練 + keep-awake），**下次接手先看 `daily_update_logs\` 最新的 log**（`Step 1~5`、`exit code`、`訓練完成`）與 `predictions_archive\` 有沒有新的一組（anchor 應為 2026-09-21 或最近的交易日）。
- [x] 2026-09-22~24：修好 `download_quarterly()`（BS/CF）的 80 天 skip 判斷 bug（誤用季末日期，導致全市場每晚重抓），新增 `bs_cf_fetched` 表，2026-09-24 確認 `bs_cf_fetched`/`income_fetched` 皆 1097/1097、Step 2 從燒完整個 timeout 縮短到 ~90 秒，**FinMind 配額問題徹底解決**。細節見「download_financial_features.py 資料來源」段落。
- [x] **2026-10-04 解決**：從 9/17 到 10/4 近三週訓練沒更新的連環問題（FinMind skip bug、yfinance 自己打太猛造成限流、`isin.twse.com.tw` 架構 bug、9/30 使用者手動睡眠中斷），全部修好並驗證成功，anchor 推進到 2026-10-02。細節見上方「2026-10-01~04：找到並修好 yfinance 限流的根本原因」跟「最新預測結果（2026-10-04）」段落。
- [ ] 60 天後回測：anchor 2026-07-09 約 2026-09-07 可回測；anchor 2026-09-07 約 2026-11-06。`predictions_archive\` 從 2026-09-19 開始累積每次訓練的預測結果，之後回測可直接使用

### 上傳大檔案的坑（2026-07-30 新發現）
- 用 `requests.put(.../api/contents/chip.db, json={'format':'base64',...})` 直接傳 ~160MB+ 的 base64 內容會失敗（proxy 層會斷線：write timeout 或 SSLEOFError "message too big"），REST API 這條路對這麼大的檔案不可靠。
- **解法**：改用已建立的 Jupyter kernel，透過 WebSocket 把檔案切成小塊（256KB 原始資料一塊，太大會被 WS 直接拒絕並回 "message too big"）逐塊 `base64.b64decode` 後用 `'ab'` append 寫到遠端暫存檔，全部傳完後用 `os.replace()` 原地覆蓋成 `chip.db`。166MB 檔案切成 256KB 一塊大約 650 塊，全程約 6 分鐘。
- 下載方向（GET）用 REST API 沒問題，可以到 200+MB（chip_backup_20260613.db 222MB 測試過沒問題），只有上傳（PUT）方向會被卡。
- 下載回來的 base64 content 每 76 字元會夾雜 `\n` 跳脫符號（Python `base64.encodebytes` 的行為），decode 前要先用 regex 拿掉（`re.sub(r'\\[nr]', '', b64)`），否則 `base64.b64decode` 會噴 "Invalid base64" 錯。

---

## 換電腦轉移步驟
1. 安裝：Python、`pip install requests websocket-client pandas numpy yfinance`
2. 從 Paperspace `/notebooks/` 下載所有 `.py` 檔案和 `chip.db`
3. 更新各腳本的連線資訊（fqdn/token/kid）
4. 用 `claude --resume 9a972ca5-853c-467f-afd4-229a9fcc7620` 嘗試恢復對話（同帳號下有效）
5. 若無法恢復，把這份指南交給新 Claude 會話即可接手

**Why:** 保存完整轉移資訊，確保研究不因硬體問題中斷。
**How to apply:** 每次有重大進展後更新本文件的「最新預測結果」和「待完成事項」。
