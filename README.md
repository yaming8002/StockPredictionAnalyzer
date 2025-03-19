# 股票回測專案說明

## modules
存放通用的項目
### 1. `process_mongo.py`
**功能：**
- 取得 MongoDB 連線設定。

**主要函數：**
- `get_mongo_client()`：從 `config.json` 讀取 MongoDB 設定，返回 MongoDB 資料庫對象。

---

### 2. `config_loader.py`
**功能：**
- 負責讀取 `config.json` 配置檔案。

**主要函數：**
- `load_config(config_file=None)`：讀取並返回設定值。

---

### 3. `logger.py`
**功能：**
- 建立 `logger`，顯示在終端並寫入 log 檔案。

**主要函數：**
- `setup_logger(log_file=None)`：建立 `log` 記錄，避免重複添加 handler，並輸出到終端與檔案。



## 01_data
資料的取得及建置
### 1. `download_stock.py`
**功能：**
- 依照股票清單下載對應的股票價格。
- 下載的資料來自 Yahoo Finance，並轉換為 pandas DataFrame 格式。
- 下載的數據包括 `date`, `open`, `high`, `low`, `close`, `volume`。
- 下載後存入指定的資料夾，或批次下載所有股票。

**主要函數：**
- `download_stock_data(symbol, start_date, end_date, save_path)`：下載單個股票的數據。
- `process_stock(symbol, save_path)`：下載並存入對應資料夾。
- `process_all_stocks(stock_list_file, save_path)`：批次下載股票數據。
- `download_all_stocks()`：從 `stock_List.csv` 讀取股票清單並下載所有股票數據。

---

### 2. `stock_technical.py`
**功能：**
- 提供技術分析的計算函數，如移動平均線 (SMA, EMA)、MACD、RSI、布林通道。

**主要函數：**
- `calculate_sma(data, window=20)`：計算簡單移動平均線 (SMA)。
- `calculate_ema(data, span=20)`：計算指數移動平均線 (EMA)。
- `calculate_macd(data, short_period=12, long_period=26, signal_period=9)`：計算 MACD 指標。
- `calculate_rsi(data, period=14)`：計算 RSI 指標。
- `calculate_bollinger_bands(data, window=20, num_std=2)`：計算布林通道 (Bollinger Bands)。

---

### 3. `to_mongoDB.py`
**功能：**
- 將 CSV 檔案處理後匯入 MongoDB。
- 增加技術指標 (SMA, EMA, Bollinger Bands)。
- 清空 MongoDB 中的指定資料庫。

**主要函數：**
- `process_csv_files()`：讀取 CSV 檔案，計算技術指標後存入 MongoDB。
- `calculate_working(df)`：計算技術指標並回傳處理後的 DataFrame。
- `remove_mongoDB()`：清空 MongoDB 中的 `stock_analysis` 集合。

---

## 02_strategy
回測功能
### 1. `single_strategy.py`
**功能：**
- 提供單一股票價格回測的抽象類別。
- 負責設定回測範圍、手續費、風險管理。
- 記錄交易資訊，計算勝率及最終資產。

**主要函數：**
- `buy_signal(self, data, index)`：判斷買入條件。
- `sell_signal(self, data, index)`：判斷賣出條件。
- `run_backtest()`：執行回測，統計交易結果。

---
