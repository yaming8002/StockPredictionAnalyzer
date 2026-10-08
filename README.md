# StockPredictionAnalyzer

台股量化的**公開教學鏡像**：從「取得資料 → 計算指標 → 用 vectorbt 回測 → 分析結果」一條龍，搭配部落格系列文章。
資料一律落地成 parquet（或 csv），回測引擎統一用 **vectorbt**，策略只需「記錄買賣條件」。

📝 **文章都在這裡：[stockanalyzer.sailforthlab.dev](https://stockanalyzer.sailforthlab.dev/)** ——每個資料夾對應哪幾篇，見下方「[文章對照](#文章對照)」。

> 所有 Python 執行建議前綴 UTF-8（避免 Windows cp950 中文出錯）：
> `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python ...`
> 安裝相依：`pip install -r requirements.txt`（vectorbt / pandas / numpy）

---

## 這個專案在做什麼

網路上關於技術指標的說法很多——「黃金交叉要買」「KD 低檔鈍化」「MACD 背離是轉折訊號」——
但幾乎都停在「怎麼看圖」，很少有人拿**整個市場、整段歷史**去驗，也很少把口徑（費用、稅、
成交時點、流動性）交代清楚。

這個專案就是做這件事：**把常見的技術分析說法寫成可執行的規則，在全台股歷史上一項一項跑，
然後把結果原樣攤開**——包含大量「這樣做並沒有比較好」的情形。

- **程式碼在這裡，數據與結論在文章裡。** 這個 repo 負責「怎麼算」，文章負責「算出什麼」。
- **口徑優先於結論。** 每個數字都附帶樣本、期間、成交價取法、費用與稅、流動性門檻；
  換了口徑數字就不同，所以口徑講不清楚的績效沒有意義。
- **可重現。** 任何人 clone 下來、照註解切換規則，就能跑出文章裡的同一組數字。

### 這個專案不做什麼

- ❌ 不是投資建議，也不提供選股、訊號或代操服務。
- ❌ 不做「找出最賺的參數」——那多半是對歷史過度配適；本專案關心的是**同一條規則換個
  情境還成不成立**。
- ❌ 不放回測結果檔（見「[回測結果在哪裡](#回測結果在哪裡)」）。

---

## 目前完成了什麼

### 資料層

- 從 TWSE / TPEx 官方 ISIN 表建立乾淨的台股清單（含名稱、產業、上市日）。
- yfinance 下載 OHLCV，支援單檔／批次／可續傳的全史下載（checkpoint、失敗重試）。
- 落地成 parquet，全市場掃描時直接讀檔，不依賴資料庫。

### 指標庫（4 個模組、17 個純函式）

輸入含 OHLCV 的 DataFrame、回傳多了指標欄位的 DataFrame，依「衡量什麼」分類：

| 模組 | 指標 |
|---|---|
| `indicators_trend.py` | SMA / EMA / MACD / 布林通道 / BIAS / ADX / 拋物線 SAR |
| `indicators_momentum_volume.py` | RSI / KD / CMF / OBV |
| `indicators_volatility.py` | ATR / ATR% / 報酬率波動率 / 唐奇安通道 / Supertrend |
| `indicators_pattern.py` | ZigZag 擺動高低點 |

### 回測框架

- **單股**：`VbtSingleStrategy` 基底——子類只覆寫「哪天判定買、哪天判定賣」，
  隔日開盤成交、台股費用與稅、tick 進位、summary 全由基底處理。
- **多股**：`VbtMultiStrategy` 基底——單一共用現金池，現金不足時擋單，可自訂買入優先序。
- **台股成本精確重建**：手續費 0.1425%（最低 20 元）＋ 賣方證交稅 0.3%，成交價過升降單位進位。
  （vectorbt 的 `fees` 只吃比例，表達不了「最低 20 元」，故在產出交易明細後重算。）

### 策略

| 家族 | 內容 |
|---|---|
| 均線 | 雙均線交叉（含多股組合版）、單一均線突破；多種進出場濾網以旗標／註解切換 |
| KD | KD 交叉（黃金／死亡交叉、超買超賣區）＋ 各式優化 |
| MACD | 三個本質不同的進場基礎（交叉／零軸／背離），外加九條進場濾網與十條出場方案 |

各策略「有哪些條件、怎麼算、怎麼開」的完整清單見
**[`docs/strategy_condition_reference.md`](docs/strategy_condition_reference.md)**。

### 分析層

持有天數分布、逐年勝率與損益、對逐筆損益做 bootstrap 蒙地卡羅（含破產機率）、
vectorbt 內建投組統計。

### 回測口徑（各策略共用）

| 項目 | 設定 |
|---|---|
| 樣本 | 全台股 2,258 檔（排除少數價格 glitch 壞資料） |
| 期間 | 2002-01-01 ~ 2025-12-31 |
| 成交 | 訊號日**收盤判定**、**隔日開盤**成交（無 look-ahead） |
| 成本 | 台股手續費 0.1425%（最低 20 元）＋ 賣方證交稅 0.3%，價格過 tick 進位 |
| 流動性 | 可選的可成交門檻（5 日均量 × 股價），只 gate 進場 |

### 怎麼確保數字可信

回測最容易出的錯不是公式寫錯，而是**偷看未來**與**靜默失敗**。本專案的做法：

- **時點責任集中在基底**：子類只描述「哪天判定」，位移與成交價由框架統一處理；
  需要逐根掃描的路徑相依規則（停損、移動停損、進場以來最高）才覆寫，並在文件明確標示
  時點責任已轉移。
- **失敗要出現在報表裡**：單檔讀取或回測失敗會被計數並列進彙總，不會靜默跳過——
  否則「整批失敗」會長得像「跑完但沒訊號」。
- **恆等與回歸檢查**：數學上應該相等的兩個變體（例如 MACD 柱狀圖轉負 ⇔ 死亡交叉）
  必須跑出逐字相同的結果；重構後也用舊版逐筆對帳，確認行為沒被動到。

---

## 回測結果在哪裡

**這個 repo 只放程式碼，不放回測結果。** 各策略的 `result/` 目錄已在 `.gitignore` 排除
（全市場掃描一次動輒數萬到數十萬筆交易，不適合進版控）。

完整的數據、對照表與結論都寫在文章裡——每一篇都會列出交易次數、勝率、平均持有天數、
獲利平均／虧損平均、中位數、期望值、獲利因子與總獲利，並說明口徑。想看某個策略跑出什麼，
照下表找對應文章：

👉 **[stockanalyzer.sailforthlab.dev](https://stockanalyzer.sailforthlab.dev/)**

---

## 文章對照

這個 repo 是文章的程式碼面；每篇文章的完整回測數據、圖表與結論都在部落格。
**系列連結會自動跟著新文章更新**，不必回頭改這裡。

| 程式 | 對應文章 |
|---|---|
| `_01_data/fetch_stock_list.py` | [台股清單怎麼抓：用官方 ISIN 表建立一份乾淨的股票池](https://stockanalyzer.sailforthlab.dev/posts/2026/06/fetch-tw-stock-list/) |
| `_01_data/indicators_trend.py` | [常見技術指標（一）趨勢](https://stockanalyzer.sailforthlab.dev/posts/2026/06/indicators-trend/) |
| `_01_data/indicators_momentum_volume.py` | [常見技術指標（二）量能與動能](https://stockanalyzer.sailforthlab.dev/posts/2026/06/indicators-momentum-volume/) |
| `_01_data/indicators_volatility.py` | [常見技術指標（三）波動與突破](https://stockanalyzer.sailforthlab.dev/posts/2026/06/indicators-volatility/) |
| `_02_strategy/base/vbt/` | [回測引擎 vectorbt（一）最有效率的回測工具](https://stockanalyzer.sailforthlab.dev/posts/2026/06/vbt-intro/)、[（二）包成只想策略的框架](https://stockanalyzer.sailforthlab.dev/posts/2026/06/vbt-framework/) |
| `_03_multi_strategy/base/vbt/` | [回測引擎 vectorbt（三）多檔股票共用一筆資金](https://stockanalyzer.sailforthlab.dev/posts/2026/07/vbt-multi-framework/) |
| `_02_strategy/ma_strategy/`、`_03_multi_strategy/ma_cross/` | [均線交叉系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=%E5%9D%87%E7%B7%9A%E4%BA%A4%E5%8F%89) |
| `_02_strategy/kd_strategy/` | [KD 交叉系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=KD%20%E4%BA%A4%E5%8F%89) |
| `_02_strategy/macd_strategy/`、`_03_multi_strategy/macd/` | [MACD 系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=MACD) 的單股掃描、矩陣、多股回測與對帳驗證 |
| `_04_analysis/analyze_vbt.py` | [回測統計指標怎麼看：每一欄到底在說什麼](https://stockanalyzer.sailforthlab.dev/posts/2026/06/backtest-metrics-guide/) |
| `_04_analysis/macd/` | [MACD 系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=MACD) 的蒙地卡羅、對 0050 的總結、文章出表與數字驗證 |
| `_04_analysis/ma_cross/` | [均線交叉系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=%E5%9D%87%E7%B7%9A%E4%BA%A4%E5%8F%89) 的範例交易圖 |
| `_04_analysis/kd/` | [KD 交叉系列](https://stockanalyzer.sailforthlab.dev/archives/?subcategory=KD%20%E4%BA%A4%E5%8F%89) 的示範圖 |
| `_04_analysis/reference/` | [蒙地卡羅模擬](https://stockanalyzer.sailforthlab.dev/posts/2026/07/monte-carlo-streak-and-ruin/)、[風險與資金分配](https://stockanalyzer.sailforthlab.dev/posts/2026/07/risk-and-position-sizing/) 等參考資料類文章的觀念圖 |

其他分類：[資料處理](https://stockanalyzer.sailforthlab.dev/archives/?category=%E8%B3%87%E6%96%99%E8%99%95%E7%90%86)、[回測架構](https://stockanalyzer.sailforthlab.dev/archives/?category=%E5%9B%9E%E6%B8%AC%E6%9E%B6%E6%A7%8B)、[參考資料](https://stockanalyzer.sailforthlab.dev/archives/?category=%E5%8F%83%E8%80%83%E8%B3%87%E6%96%99)。

---

## 目錄結構

| 路徑 | 用途 |
|---|---|
| `_01_data/` | 取得股票清單、下載股價、計算技術指標 |
| `_02_strategy/` | **單股回測**：vbt 框架、策略，以及跑單股全市場回測的腳本 |
| `_03_multi_strategy/` | **多股回測**：同一本金、共用資金的 vbt 框架、策略，以及跑多股回測的腳本 |
| `_04_analysis/` | **分析**：讀回測結果做統計、蒙地卡羅、跟 0050 比、出文章表格與配圖；**不跑回測**。依策略主題分資料夾，另有 0050 基準線 `benchmark/` 與概念文用的 `reference/` |

分層的判斷依據是「這支程式在做什麼」：跑回測的放 `_02`（單股）或 `_03`（多股），讀回測結果再加工的放 `_04`。
需要「先回測、再分析」的流程拆成兩段：回測段把逐筆交易或權益曲線存成 parquet，分析段讀檔計算。
| `docs/` | 指標與策略條件的完整清單 |

---

## `_01_data/` — 資料取得與指標

- **`fetch_stock_list.py`**：從 TWSE/TPEx 官方 ISIN 表抓最新台股清單 → `stock_list.csv`（含名稱/產業/上市日）。
- **`download_stock.py`**：用 yfinance 下載 OHLCV，存 csv 或 parquet。
  - `download_stock_data(symbol, ...)`：單檔下載（穩定）。
  - `download_stock_data_multi(stock_list_file, ...)`：批次下載（快，適合每日增量）。
  - `save_prices(df, save_path, symbol, fmt)`：存檔（`csv` / `parquet`）。
- **`download_full_history.py`**：大量、可續傳的全史下載（checkpoint、失敗重試、進度估算），沿用 `download_stock.py`。
- **技術指標**（純函式，輸入含 OHLCV 的 DataFrame，回傳多了指標欄位的 DataFrame），依「衡量什麼」分組：
  - `indicators_trend.py` — 趨勢：SMA / EMA / MACD / 布林 / BIAS / ADX / 拋物線 SAR
  - `indicators_momentum_volume.py` — 量能動能：RSI / KD / CMF / OBV
  - `indicators_volatility.py` — 波動：ATR / ATR% / 報酬率波動率 / 唐奇安通道 / Supertrend
  - `indicators_pattern.py` — 型態：ZigZag 擺動高低點（結構判斷用）
  - `stock_technical.py` — 聚合入口：re-export 全部函式 + `add_all_indicators()` 一次套用
    （ADX / SAR / Supertrend 需逐根掃描，僅 re-export、不列入 `add_all_indicators`）

## `_02_strategy/` — 單股策略（vbt）

把 vectorbt 包成「繼承基底、只覆寫買賣條件」的開發手感。

- **`base/vbt/`** — vbt 策略套件（框架）
  - `common.py`：台股 tick 進位、精確費用重建（手續費 min 20 + 賣方證交稅）、summary 組裝；
    以及跨策略共用的單一定義——標準區間、GLITCH 排除集、資料路徑、全市場讀檔（`load_market`／`iter_market`）、
    輸出目錄（`result_dir`）、畫圖用中文字型（`chinese_font`）。
  - `single.py`：`VbtSingleStrategy` 基底。子類**只覆寫** `add_columns` / `buy_signal` / `sell_signal`（可選 `exec_price` / `build_signals`），引擎 / 費用 / 後處理由基底處理。
- **`ma_strategy/`** — 均線相關策略
  - `ma_cross_strategy.py`：雙均線交叉（2 日確認），`MACross_20_50` / `MACross_50_200`。
  - `single_ma_strategy.py`：單一均線突破（上穿買、下穿賣），附「測試資料中所有 MA 期數」的分析。
  - `extract_trades.py`：撈出 MA 60/200 交叉的全部逐筆交易（挑文章案例用）。
- **`kd_strategy/`** — KD 相關策略
  - `single_kd_strategy.py`：KD 交叉（黃金/死亡交叉、超買超賣區），以「註解切換」管理各種優化。
- **`macd_strategy/`** — MACD 相關策略
  - `single_macd_strategy.py`：MACD 三個本質不同的進場基礎（交叉／零軸／背離），
    外加九條進場濾網與十條出場方案，全部以「註解切換」管理（見檔頭說明）。
  - `macd_variants.py`：變體參數化子類（條件式沿用註解切換行原文，改由類別屬性選；掃描用）。
  - `macd_sweep.py`：掃描共用執行層（讀檔＋算指標只做一次，所有變體共用）。
  - `macd_exit_replace.py`、`macd_combo.py`：出場替換全表、拼裝矩陣。
  - `macd_mc_trades.py`：蒙地卡羅那五組的逐筆交易（回測段；分析段在 `_04_analysis/macd/macd_montecarlo.py`）。
  - `verify_exit_switches.py`：驗證策略檔的出場切換可用。

寫新策略範式：
```python
from _02_strategy.base.vbt.single import VbtSingleStrategy

class MyStrat(VbtSingleStrategy):
    def add_columns(self, df):
        df["sma20"] = df["close"].rolling(20).mean(); return df
    def buy_signal(self, df):  return df["close"] > df["sma20"]
    def sell_signal(self, df): return df["close"] < df["sma20"]

res = MyStrat(split_cash=10_000).run(df, stock_id="2330.TW")
# res = {"trades": DataFrame, "summary": dict}
```

## `_03_multi_strategy/` — 多股組合（vbt）

- **`base/vbt/multi.py`** — `VbtMultiStrategy` 基底：單一共用現金池（`cash_sharing`），現金不足時擋單，可覆寫 `priority` 自訂買入優先序。台股費用沿用 `_02` 的 `common`（依賴方向 `_03 → _02`）。
- 輸入 `data_dict = {stock_id: df}`，輸出 `{trades, summary, failed_orders_approx}`。
- **`ma_cross/`**
  - `multi_ma_cross.py`：多股雙均線交叉（類別＋執行入口）。
- **`kd/`**
  - `multi_kd.py`：多股 KD 類別。
- **`macd/`**
  - `multi_macd.py`：多股 MACD 類別（五組交易策略）。
  - `macd_multi_driver.py`：五組 × 兩種投法 × 低價／高價／流動性三種買入排序。
  - `macd_multi_random.py`：隨機買入順序基準線（1,000 次，多進程）。
  - `macd_conclusion_equity.py`：結論篇用的 2015 起權益曲線（回測段；跟 0050 比在 `_04_analysis/macd/macd_multi_conclusion.py`）。
  - `verify_multi_macd.py`：多股引擎的正確性錨點（關掉資金限制後必須等於單股回測）。

## `_04_analysis/` — 數據分析

讀 `_02`／`_03` 回測跑出來的結果再加工，**這一層不跑回測**。依策略主題分資料夾：

| 路徑 | 內容 |
|---|---|
| `analyze_vbt.py` | 跨策略共用的分析層（不屬任何主題） |
| `benchmark/` | 0050 買進持有（含息）基準線：各系列結論篇的最終比較對象，含對照區間與錨點驗算 |
| `macd/` | MACD：五組交易策略的蒙地卡羅、對 0050 的總結、母體候選表 |
| `macd/article/` | MACD 文章專用：從結果 CSV 產文章表格、把文章數字對回 CSV 驗證（需設 `BLOG_DIR`） |
| `ma_cross/charts/`、`kd/charts/` | 各系列的範例交易圖 |
| `reference/charts/` | 不屬單一策略的概念文用圖（指標字典、蒙地卡羅、資金分配） |

各主題下的 `charts/` 是**產文章配圖的一次性腳本**（`_draw_*.py`），不是給人 import 的模組。

**輸出落點**：回測與分析的結果都寫到 `_02_strategy/<策略>/result/<任務>/`（由 `common.result_dir` 統一決定，
不進版控），跟單股優化流程的 `result/` 同一棵樹；配圖則寫到 `CHART_OUT_DIR`。
**兩段式的分析要先跑回測段**（例：先 `_02_strategy/macd_strategy/macd_mc_trades.py`，再 `_04_analysis/macd/macd_montecarlo.py`），
分析段找不到回測輸出會直接提示要先跑哪一支。

- **`analyze_vbt.py`** — 吃 `VbtSingleStrategy.run()` 的輸出（trades / summary）：
  - `hold_days_stats(trades)`：持有天數分布
  - `yearly_performance(trades)`：依買入年份的勝率 / 損益
  - `monte_carlo(trades, ...)`：對 `real_pnl` 做 bootstrap 蒙地卡羅（含破產機率）
  - `portfolio_stats(pf)`：取 vbt `Portfolio` 內建統計
  - `full_report(result, pf=)`：一次印全部

---

## 快速開始

```bash
pip install -r requirements.txt

# 0.（選用）指定資料與輸出位置；不設就用 repo 內的預設目錄
#    STOCK_DATA_DIR  股價 parquet 全史所在目錄（預設 <repo>/stock_data）
#    CHART_OUT_DIR   產圖腳本的輸出目錄（預設 <repo>/result/charts）
#    BLOG_DIR        只有「對照已發佈文章」的驗證腳本需要，指向 blog 專案根目錄
#    DIVIDEND_FILE   只有含息計算（0050 基準線）需要，指向 dividend_actions.parquet
export STOCK_DATA_DIR=/path/to/stock_data

# 1. 取得最新股票清單
python _01_data/fetch_stock_list.py

# 2. 下載單檔（台積電）存 parquet
python -c "from _01_data.download_stock import download_stock_data; download_stock_data('2330.TW', save_path='_01_data/data', fmt='parquet')"

# 3. 跑一支均線交叉回測
python _02_strategy/ma_strategy/ma_cross_strategy.py _01_data/data/2330.TW.parquet
```

要重現文章裡的某個變體：打開對應策略檔，照檔頭「註解切換」的說明開啟該條規則，
再用 `--variant` 指定輸出資料夾即可。條件清單見
[`docs/strategy_condition_reference.md`](docs/strategy_condition_reference.md)。

---

## 已知限制

寫在前面，免得把回測結果讀得太重：

- **未模擬滑價**。成交價取隔日開盤，實務上大單、低流動性標的會有價差。
- **停損停利以收盤判定**，不模擬盤中觸價——日線資料做盤中觸價很容易變成偷看未來。
- **生存者偏差**：資料來源以現存標的為主，已下市公司的涵蓋不完整，長期報酬會偏樂觀。
- **除權息還原**依賴資料來源的處理方式，跨越大額配股配息的個股需另外留意。
- **回測是一條歷史**。同一組規則換一段期間、換一個市場都可能得到不同結果；
  本專案用蒙地卡羅與跨情境比較來看穩健度，但那也只是估計。

---

## 未來可以延伸的方向

以下都**尚未實作**，列出來當作這個框架還能往哪走（也歡迎拿去自己試）：

- **參數梯度掃描**：目前多數規則只取單一代表值（例如某個門檻、某個天期），
  尚未系統性地掃描參數面，看效果是隨參數單調變化還是只在某一點成立。
- **多股組合版的覆蓋**：目前只有均線交叉有多股版；其餘策略仍是「每檔獨立、不競爭資金」，
  和真實帳戶的資金排擠行為不同。
- **成本模型**：加入滑價假設、不同券商費率、當沖稅率。
- **更多指標與型態**：例如成交量分布、型態辨識、籌碼面資料。
- **指標實作對帳**：Supertrend 與拋物線 SAR 目前是依原始定義自行實作，
  尚未與獨立的 TA 套件逐值比對。
- **內部整理**：ADX 已收進 `_01_data`，但 `ma_strategy` 內仍有早期的 inline 實作，
  可統一改用共用函式。

---

*本專案為技術分享，所列指標、策略與程式碼僅供學習參考，非投資建議。*
