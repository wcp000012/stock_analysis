---
name: chip-data-for-ml-null-policy
description: stock_analysis 籌碼資料是要餵 AI 訓練用的；缺值一律存 NULL 不補 0
metadata: 
  node_type: memory
  type: project
  originSessionId: 52d5b5a0-bfb9-4ef4-bd02-b1243e394e9d
---

`stock_analysis` 下載的台股籌碼資料（chip.db / chip_daily）最終要拿去**訓練 AI 模型**。

**Why:** 對籌碼資料，`foreign_net=0`（法人持平）是真實且有意義的值，和 `NULL`（未知）語意完全不同；在儲存層補 0 會注入假訊號污染訓練資料。

**How to apply:**
- 儲存層一律存 NULL/NaN 忠實表示「未知」，**絕不補 0**。
- 補值是建模決策，留到訓練前的特徵工程階段依語意處理（流量型欄位勿做 forward-fill）。
- 區分缺漏層級：整天法人缺＝暫時性故障要重試（見 [[twse-stat-field-trading-day]]）；個股層級缺＝真實 NaN。
