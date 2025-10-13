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
## 03_deeplearning
深度學習訓練

### 1. DpTrainerBisis
**功能：**

- 提供一個通用的深度學習訓練框架，適合 LSTM 類型的模型進行訓練、驗證、儲存以及繪圖顯示。
- 支援 One-Hot 編碼的多類別分類，以及標量類型的二元分類。

**主要屬性：**

- file_path：訓練資料檔案的路徑。
- save_path：訓練後儲存模型的路徑。
- input_shape：輸入資料的形狀 (time_steps, features)。
- output_shape：輸出的類別數目（One-Hot 時為 2）。
- optimizer：優化器（預設為 Adam）。
- is_onehot：是否使用 One-Hot 編碼。
- batch_size：訓練的批次大小。
- epochs：最大訓練回合數。
- model：Keras 模型實例。
- history：儲存每個 Epoch 的訓練與驗證記錄。

**主要函數：**
- __init__(self, file_path, save_path, input_shape, output_shape, optimizer=None, is_onehot=False, batch_size=64, epochs=100)
初始化模型的基本參數。
根據 is_onehot 設定 loss function：
如果是 True，則使用 categorical_crossentropy。
如果是 False，則使用 sparse_categorical_crossentropy。
初始化歷史記錄 self.history。
呼叫 build_model() 建立模型並進行編譯。

- build_model(self)
抽象方法，必須在子類別中實作。

負責構建實際的 LSTM 模型結構。

- tranging_model(self)
    負責完整的模型訓練過程，包括：
    載入資料並切割為訓練集與驗證集（95% / 5%）。
    設定 class_weight 平衡正負樣本。
    呼叫 fit() 執行模型訓練。
    呼叫 evaluate() 進行評估。
    呼叫 save_model() 儲存最佳模型。
    呼叫 plot_history() 顯示學習曲線。

- fit(self, train_df, val_df, class_weight, patience=5)，負責模型的迭代訓練。
    每次 Epoch 重新打亂資料。
    使用 class_weight 來平衡正負類別。
    執行 Early Stopping，若 patience 次數內沒有改善則終止。
    儲存最佳模型權重。
    早期停止條件：
    如果 val_loss 無法在 patience 次數內降低，則恢復到最佳模型。

- create_data_generator(self, dataframe)：建立資料生成器，支援：
    - One-Hot 編碼模式
    - 標量類型模式
    - 使用 tf.data.Dataset.from_generator 創建資料流。
        資料會經過：
        - 分批讀取 (batch)
        - 提前載入 (prefetch)

- evaluate(self, val_df)
    負責評估模型的效能。
    輸出測試集的準確率與損失。

- save_model(self)
    將模型儲存至指定路徑。
    使用 .h5 格式進行儲存，包含優化器參數。

- plot_history(self)
    繪製模型在訓練過程中的學習曲線。
    包含：
    - 訓練準確率 (Train Accuracy)
    - 驗證準確率 (Validation Accuracy)
    - 訓練損失 (Train Loss)
    - 驗證損失 (Validation Loss)

**使用範例 **
```
from tensorflow.keras.optimizers import Adam
from custom_lstm_model import CustomLSTMModel
# 建構自訂義的model
class CustomLSTMModel(DpTrainerBisis):
    def build_model(self):
        inputs = Input(shape=self.input_shape, name="Input_Layer")
        x = LSTM(64, return_sequences=True, name="LSTM_1")(inputs)
        x = Dropout(0.5, name="Dropout_1")(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu", name="Dense_Layer1")(x)
        x = BatchNormalization(name="BatchNorm")(x)
        outputs = Dense(self.output_shape, activation="softmax", name="Output_Layer")(x)
        return Model(inputs=inputs, outputs=outputs, name="LSTM_Model")

file_path = "./stock_data/leaning_label/turtle_trades.csv"
model_path = "run_turtle_deep_01.h5"
optimizer = Adam(learning_rate=0.0001)
# 載入設定
trainer = CustomLSTMModel(
    file_path=file_path,
    save_path=model_path,
    is_onehot=True,
    input_shape=(30, 19),
    output_shape=2,
    optimizer=optimizer
)
# 訓練
trainer.tranging_model()


```
//