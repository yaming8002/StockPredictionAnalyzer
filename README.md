# StockPredictionAnalyzer

台股量化的**公開教學鏡像**：從「取得資料 → 計算指標 → 用 vectorbt 回測 → 分析結果」一條龍，搭配部落格系列文章。
資料一律落地成 parquet（或 csv），回測引擎統一用 **vectorbt**，策略只需「記錄買賣條件」。

> 所有 Python 執行建議前綴 UTF-8（避免 Windows cp950 中文出錯）：
> `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python ...`
> 安裝相依：`pip install -r requirements.txt`（vectorbt / pandas / numpy）

---

## 目錄結構

| 路徑 | 用途 |
|---|---|
| `_01_data/` | 取得股票清單、下載股價、計算技術指標 |
| `_02_strategy/` | **單股**策略（vbt 框架 + 策略） |
| `_03_multi_strategy/` | **多股組合**策略（同一本金、共用資金；vbt 框架） |
| `_04_analysis/` | 回測輸出的數據分析 |

---

## `_01_data/` — 資料取得與指標

- **`fetch_stock_list.py`**：從 TWSE/TPEx 官方 ISIN 表抓最新台股清單 → `stock_list.csv`（含名稱/產業/上市日）。
- **`download_stock.py`**：用 yfinance 下載 OHLCV，存 csv 或 parquet。
  - `download_stock_data(symbol, ...)`：單檔下載（穩定）。
  - `download_stock_data_multi(stock_list_file, ...)`：批次下載（快，適合每日增量）。
  - `save_prices(df, save_path, symbol, fmt)`：存檔（`csv` / `parquet`）。
- **`download_full_history.py`**：大量、可續傳的全史下載（checkpoint、失敗重試、進度估算），沿用 `download_stock.py`。
- **技術指標**（純函式，輸入含 OHLCV 的 DataFrame，回傳多了指標欄位的 DataFrame），依「衡量什麼」分三組：
  - `indicators_trend.py` — 趨勢：SMA / EMA / MACD / 布林 / BIAS
  - `indicators_momentum_volume.py` — 量能動能：RSI / KD / CMF / OBV
  - `indicators_volatility.py` — 波動：ATR% / 報酬率波動率 / 唐奇安通道
  - `stock_technical.py` — 聚合入口：re-export 上三組全部函式 + `add_all_indicators()` 一次套用

## `_02_strategy/` — 單股策略（vbt）

把 vectorbt 包成「繼承基底、只覆寫買賣條件」的開發手感。

- **`base/vbt/`** — vbt 策略套件（框架）
  - `common.py`：台股 tick 進位、精確費用重建（手續費 min 20 + 賣方證交稅）、summary 組裝。
  - `single.py`：`VbtSingleStrategy` 基底。子類**只覆寫** `add_columns` / `buy_signal` / `sell_signal`（可選 `exec_price` / `build_signals`），引擎 / 費用 / 後處理由基底處理。
- **`ma_strategy/`** — 均線相關策略
  - `ma_cross_strategy.py`：雙均線交叉（2 日確認），`MACross_20_50` / `MACross_50_200`。
  - `single_ma_strategy.py`：單一均線突破（上穿買、下穿賣），附「測試資料中所有 MA 期數」的分析。

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

## `_04_analysis/` — 數據分析

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

# 1. 取得最新股票清單
python _01_data/fetch_stock_list.py

# 2. 下載單檔（台積電）存 parquet
python -c "from _01_data.download_stock import download_stock_data; download_stock_data('2330.TW', save_path='_01_data/data', fmt='parquet')"

# 3. 跑一支均線交叉回測
python _02_strategy/ma_strategy/ma_cross_strategy.py _01_data/data/2330.TW.parquet
```

---

*本專案為技術分享，所列指標、策略與程式碼僅供學習參考，非投資建議。*
