---
name: twse-stat-field-trading-day
description: 用 TWSE JSON 的 stat 欄位區分交易日 vs 非交易日（避免把真失敗當無資料）
metadata: 
  node_type: memory
  type: reference
  originSessionId: 52d5b5a0-bfb9-4ef4-bd02-b1243e394e9d
---

TWSE 的 JSON 介面（MI_MARGN 融資融券、T86 三大法人等）回傳的頂層 `stat` 欄位可可靠區分：

- **交易日**：`stat == "OK"`，有 fields/data（MI_MARGN 會有 2 個 tables）
- **非交易日／當日無資料**（週末、國定假日、颱風停市）：`stat == "很抱歉，沒有符合條件的資料!"`，fields/data 為空

**How to apply:** 抓取籌碼資料時先看 `stat`：非 OK→當日無資料（回 None，不重試也不算失敗）；OK 卻解析不出→視為真失敗要重試/記錄。這樣才不會把真正的下載失敗默默當成「非交易日」吞掉。相關：[[chip-data-for-ml-null-policy]]
