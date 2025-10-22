import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from _02_strategy.base.single_strategy import StockBacktest
from _04_deeplearning.model_process import DpTrainerBisis
from _04_deeplearning.unit import DEFAULT_FEATURE_COLUMNS
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import clear_old_connections, close_mongo_client, get_mongo_client
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report


config = load_config()


class CustomLSTMModel(DpTrainerBisis):
    def build_model(self):
        inputs = Input(shape=self.input_shape, name="Input_Layer")
        x = LSTM(64, return_sequences=True, name="LSTM_1")(inputs)
        x = Flatten()(x)
        x = Dense(512, activation="relu", name="Dense_Layer1")(x)
        x = Dropout(0.6, name="Dropout_1")(x)
        x = Dense(256, activation="relu", name="Dense_Layer2")(x)
        x = Dense(64, activation="relu", name="Dense_Layer3")(x)
        x = BatchNormalization(name="BatchNorm")(x)
        outputs = Dense(self.output_shape, activation="sigmoid", name="Output_Layer")(x)
        return Model(inputs=inputs, outputs=outputs, name="LSTM_Model")


class CustomCNNModel(DpTrainerBisis):
    def build_model(self):
        # 輸入層，形狀與原本一致
        inputs = Input(shape=self.input_shape, name="Input_Layer")

        # 卷積層組合
        x = Conv1D(64, kernel_size=3, activation="relu", padding="same", name="Conv1D_1")(inputs)
        x = MaxPooling1D(pool_size=2, name="MaxPool_1")(x)

        x = Conv1D(128, kernel_size=3, activation="relu", padding="same", name="Conv1D_2")(x)
        x = MaxPooling1D(pool_size=2, name="MaxPool_2")(x)

        x = Conv1D(256, kernel_size=3, activation="relu", padding="same", name="Conv1D_3")(x)
        x = MaxPooling1D(pool_size=2, name="MaxPool_3")(x)

        # 全局平均池化，將時序列展平成固定向量
        x = GlobalAveragePooling1D(name="GlobalAveragePooling")(x)

        # 全連接層
        x = Dense(512, activation="relu", name="Dense_Layer1")(x)
        x = Dropout(0.6, name="Dropout_1")(x)
        x = Dense(256, activation="relu", name="Dense_Layer2")(x)
        x = Dense(64, activation="relu", name="Dense_Layer3")(x)

        # 批次正規化
        x = BatchNormalization(name="BatchNorm")(x)

        # 輸出層
        outputs = Dense(self.output_shape, activation="sigmoid", name="Output_Layer")(x)

        # 模型構建
        return Model(inputs=inputs, outputs=outputs, name="CNN_Model")


class TurtleDeepStrategy(StockBacktest):

    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        model_path,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        loglevel=logging.INFO,
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)
        self.model = load_model(model_path)

    def predict_with_model(self, i, columns=DEFAULT_FEATURE_COLUMNS, normalize=True):
        if i < 30:
            return 0  # 資料不足

        # 取得前 30 天資料
        feature_data = self.data.iloc[i - 30 : i].copy()

        # 移除非數值欄位
        feature_data = feature_data.drop(columns=["_id", "date"], errors="ignore")

        # 僅保留指定欄位
        try:
            feature_data = feature_data[columns]
        except KeyError:
            return 0  # 缺欄位

        if feature_data.shape[0] != 30:
            return 0  # 天數不夠

        # ✅ 正規化
        if normalize:
            if "volume" in feature_data.columns:
                volume_values = feature_data[["volume"]].values
                feature_data["volume"] = MinMaxScaler().fit_transform(volume_values)

            other_cols = [col for col in feature_data.columns if col != "volume"]
            if other_cols:
                other_values = feature_data[other_cols].values
                feature_data[other_cols] = MinMaxScaler().fit_transform(other_values)

        # 模型預測
        X = np.expand_dims(feature_data.values.astype(np.float32), axis=0)
        y_pred = self.model.predict(X, verbose=0)
        return float(y_pred[0][0]) < 0.3 and float(y_pred[0][1]) > 0.5

    def buy_signal(self, i):
        if i > 20:
            max_value = max(self.data.iloc[i - 20 : i]["high"])
            cond_ma = max_value < self.data.iloc[i]["high"]
            if not cond_ma:
                return False

            # 加入模型判斷（用 i-1 對應買入當日）

            # cond_model = self.predict_with_model(i) > 0.6  # ✅ 模型預測為正收益
            return self.predict_with_model(i)
        return False

    def sell_signal(self, i):
        if i > 10:
            min_value = max(self.data.iloc[i - 10 : i]["low"])

            return min_value > self.data.iloc[i]["low"]
        return False

    def buy_price_select(self, i):
        max_value = max(self.data.iloc[i - 20 : i]["high"])
        return self.tw_ticket_gap(max_value)

    def sell_price_select(self, i):
        min_value = max(self.data.iloc[i - 10 : i]["low"])
        return self.tw_ticket_gap(min_value)


def run_turtle_deep_list(
    start_date="2012-01-01", end_date="2015-12-31", model_path="run_turtle_deep.h5", initial_cash=100000
) -> None:
    # clear_old_connections()
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []
    label = "turtle_deep"
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}_cnn0.6.0.4.log"
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"✅ 檔案已刪除: {log_file_path}")
    else:
        print(f"⚠️ 檔案不存在: {log_file_path}")
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    for stock_id in collections:
        backtest = TurtleDeepStrategy(
            stock_id=stock_id,
            model_path=model_path,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            label=label,
        )
        backtest.run_backtest()
        profit = backtest.cash - initial_cash
        log.info(
            f"{stock_id}: 初始金額{initial_cash} ,最終金額 {backtest.cash} 獲利:{math.floor(profit)}, 勝率 {backtest.win_rate:.2%}"
        )
        total_win += backtest.win_count
        total_lose += backtest.lose_count
        total_profit += profit
        hold_days.extend(backtest.hold_days)
        trade_records.extend(backtest.trade_records)

    buy_count = total_win + total_lose
    win_rate = total_win / buy_count if buy_count > 0 else 0
    avg_profit = total_profit / buy_count if buy_count > 0 else 0

    log.info(
        f"總計:總營利{total_profit}, 股票數量{len(collections)},總下注量:{buy_count},每注獲利 {avg_profit:.2f}, 獲勝次數{total_win}, 總勝率 {win_rate:.2%}"
    )
    if len(trade_records) > 0:
        df = pd.DataFrame(trade_records)
        output_folder = config.get("leaning_folder", "./stock_data/leaning_label")
        os.makedirs(output_folder, exist_ok=True)
        # 檔名格式可用 label 或加日期時間
        filename = f"{label}_trades.csv"
        filepath = os.path.join(output_folder, filename)
        # 儲存 CSV 檔案（避免 Excel 打不開加 utf-8-sig）
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

    if hold_days:
        max_days = max(hold_days)
        min_days = min(hold_days)
        avg_days = np.mean(hold_days)
        std_days = np.std(hold_days, ddof=1)
        try:
            mode_days = statistics.mode(hold_days)
        except statistics.StatisticsError:
            mode_days = "無唯一眾數"

        log.info(
            f"持有天數統計: 最大 {max_days}, 最小 {min_days}, 平均 {avg_days:.2f}, 標準差 {std_days:.2f}, 眾數 {mode_days}"
        )
    else:
        log.info("無持有天數數據")
    close_mongo_client()
