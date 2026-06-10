# 股票資料取得與處理（_01_data）

這個資料夾示範「從零取得台股資料」的完整流程，搭配系列文章使用。
分成三個主題，照順序跑即可：

1. **取得股票清單** — 從證交所官方資料抓出全台股代號
2. **取得股票資料** — 用 yfinance 下載歷史 OHLCV
3. **資料優化 + 指標計算** — 把原始價格轉成技術指標

## 安裝

```bash
pip install -r requirements.txt
```

## 主題 1：取得股票清單

`fetch_stock_list.py` — 從 TWSE/TPEx 的官方 ISIN 表抓上市 / 上櫃 / 興櫃清單，
輸出 `stock_list.csv`（含代號、名稱、市場別、產業、ISIN、上市日）。代號會自動加上
`.TW`（上市）或 `.TWO`（上櫃 / 興櫃）後綴，與 yfinance 一致。

```bash
python fetch_stock_list.py
```

## 主題 2：取得股票資料

**基礎篇** `download_stock.py` — 用 yfinance 下載單檔或批次股票，可存成 CSV 或 parquet。

```bash
python download_stock.py
```

- `download_stock_data()`：單檔下載，最穩定，適合教學示範
- `download_stock_data_multi()`：批次下載，較快但較易被限流
- `save_prices(..., fmt="csv" | "parquet")`：兩種格式都示範。CSV 人眼可讀、好上手；
  parquet 體積小、讀寫快、保留型別，適合大量長期儲存

**進階篇** `download_full_history.py` — 大量、可續傳的全史下載。多了實務上必備的機制：
checkpoint 續傳（中斷後接著跑）、失敗重試、失敗清單、進度與剩餘時間估算。

```bash
# 需先用 fetch_stock_list.py 產生 stock_list.csv
python download_full_history.py
```

## 主題 3：資料優化 + 指標計算

技術指標依「衡量什麼」分三組，各一個模組（對應指標系列文章三篇）。都是純函式：
輸入含 OHLCV 的 DataFrame，回傳加上指標欄位的結果。

- `indicators_trend.py` — **趨勢**：SMA / EMA / MACD / 布林通道 / BIAS（乖離率）
- `indicators_momentum_volume.py` — **量能動能**：RSI / KD / CMF / OBV
- `indicators_volatility.py` — **波動**：ATR% / 報酬率波動率 / 唐奇安通道
- `stock_technical.py` — **聚合入口**：re-export 上三組全部函式，並提供 `add_all_indicators()` 一次套用

```bash
# 需先下載好 2330.TW 的資料；stock_technical.py 會一次套用全部指標
python stock_technical.py
```

## 檔案一覽

| 檔案 | 對應主題 | 說明 |
|---|---|---|
| `fetch_stock_list.py` | 1 | 抓全台股清單 → `stock_list.csv` |
| `download_stock.py` | 2 基礎 | 單檔 / 批次下載，CSV / parquet |
| `download_full_history.py` | 2 進階 | 可續傳的全史大量下載 |
| `indicators_trend.py` | 3 | 趨勢指標：SMA / EMA / MACD / 布林 / BIAS |
| `indicators_momentum_volume.py` | 3 | 量能動能：RSI / KD / CMF / OBV |
| `indicators_volatility.py` | 3 | 波動：ATR% / 報酬率波動率 / 唐奇安通道 |
| `stock_technical.py` | 3 | 聚合入口：re-export 三模組 + `add_all_indicators()` |
