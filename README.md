# stock_analysis — 台股 PatchTST 預測與回測專案

用 PatchTST Transformer 模型預測台股（上市+上櫃）未來 60 個交易日的最大報酬率，
每天自動更新資料、重新訓練、產出 Top-20 選股名單，並持續用回測實驗驗證/改進策略。

> 這不是投資建議，純屬個人研究用途。

## 核心方法

- **模型**：PatchTST（`TIME_STEP=126`，`FUTURE_DAYS=60`），用過去 3 年滑動視窗訓練，
  每個 anchor 日期訓練多次（`N_RUNS`）取平均分數。
- **標籤**：`Y_max`（未來 60 天最大報酬，log）、`Y_min`（最大回撤）。
- **三個模型版本**（見 `topK_predict.py`）：
  - **Model B**：純做多 Top-20（依 `Y_max` 預測分數排序）。
  - **Model C**：Model B 再疊加反轉濾網。
  - **Model D**（觀察用，不影響 B/C）：`pred_max` + 外資流向 + 動能 三訊號依百分位排名加權混合
    （`0.7 / 0.2 / 0.1`），經多次回測驗證後比純 raw 分數更穩定，目前以旁觀方式累積 out-of-sample 資料。
- 詳細實驗記錄（含失敗的嘗試）全部在 **`RESEARCH_LOG.md`**，是這個專案最重要的文件，
  任何人（包含未來的 Claude Code session）接手前應該先讀這份。

## 資料來源與管線

- **股價**：yfinance（需要 `ca_bundle.pem`，本機自行用 `_make_ca_bundle.py` 產生，未進版控）。
- **籌碼面**：TWSE 融資融券/三大法人、外資持股比例（`stock_data_down_load_chips.py`）。
- **基本面**：FinMind（月營收、資產負債表、現金流量表、損益表，`download_financial_features.py`）、
  TWSE/TPEx 本益比/淨值比/殖利率（`download_fundamentals.py`）。
- 所有籌碼/基本面資料存在 `chip.db`（SQLite，用 **Git LFS** 追蹤）。
- 合併股價 + 籌碼 + 基本面成每檔股票的特徵檔：`downLoad_Tech_and_merge_with_Chips.py` →
  產出 `parquet_data/<代號>.TW.parquet`。
- 處理財報公告延遲（季報 +50/+90 天、月營收 +15 天）避免 look-ahead bias，細節見
  `RESEARCH_LOG.md` 的 E6 實驗。

## 每日自動化排程

`_daily_update.py` 由 Windows 工作排程器在平日晚上 21:30 觸發：
1. 補籌碼/外資持股資料（免配額）
2. 補月營收 + FinMind 財報（有配額冷卻機制）
3. 補 PE/PB/殖利率
4. 若出現新交易日資料 → 重新 merge 產生 parquet_data（~25 分鐘）
5. 重新訓練 + 預測 `topK_predict.py`（本機 GPU，~12 分鐘）→ 結果存到 `predictions_archive/`

手動跑：`python _daily_update.py`（加 `--force-train` 強制重訓、`--only-train` 跳過資料更新）。

## 重要檔案索引

| 檔案 | 用途 |
| --- | --- |
| `topK_predict.py` | 正式訓練 + 預測（production，B/C/D） |
| `downLoad_Tech_and_merge_with_Chips.py` | 股價+籌碼+基本面 → parquet_data |
| `download_financial_features.py` | FinMind 月營收/財報下載 |
| `download_fundamentals.py` | PE/PB/殖利率下載 |
| `stock_data_down_load_chips.py` | 籌碼資料下載 |
| `_daily_update.py` | 每日排程主程式 |
| `backtest_lab.py` | 固定種子的回測實驗室，訓練一次可重複測多種變體 |
| `backtest_current.py` / `backtest_riskadj.py` / `backtest_baseline.py` | 各版本回測腳本 |
| `chip.db` | 籌碼+財報核心資料庫（Git LFS） |
| `parquet_data/` | 各股票合併後的特徵檔 |
| `predictions_archive/` | 歷史每日預測結果存檔 |
| `RESEARCH_LOG.md` | **研究主記錄**：時間線、實驗、失敗教訓、未解決問題 |
| `memory/project-complete-guide.md` | 操作手冊（環境、排程設定細節） |

## 環境設定

- Python 套件見 `requirements.txt`，conda 環境細節見 `conda_env.txt`。
- 需要本機 GPU（CUDA）執行訓練；也可選擇用 Paperspace Gradient 遠端 GPU
  （見 `memory/project-paperspace-ml-pipeline.md`，**未進版控**，含帳號資訊）。

### 密鑰（未進版控，需自行補上）

以下檔案故意排除在 repo 外，換電腦要自己建立：

- `APIKey.txt` / `API_key.txt`：FinMind 等服務的 API 金鑰。
- Paperspace 相關帳密/token：`memory/project-paperspace-ml-pipeline.md`
  （含 API Key）、`gradient_volumes.json`、`_projects.json`、`running_notebook.json`，
  以及 18 個寫死 Jupyter kernel token 的 `_check_*.py` / `_run_*.py` / `_explain_top*.py` /
  `_launch_backtest.py` / `_monitor_merge.py` / `_upload_db.py` / `_verify_*.py` 輔助腳本
  （這些 token 本來就會隨 notebook 重啟失效，要用的話得重新建立連線）。

## 模型權重

`model_run*.pt`、`model_long_run*.pt`、`model_rev_run*.pt`、`model_final.pt`、
`best_autoencoder.pth`、`best_contrastive_encoder.pth`、`best_model.pth` 都已進版控
（總共約 46MB，不需要 LFS），可以直接拿來預測，不一定要重新訓練。
