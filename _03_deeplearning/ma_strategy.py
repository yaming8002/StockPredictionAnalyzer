import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from _02_strategy.single_strategy import StockBacktest
from _03_deeplearning.unit import DEFAULT_FEATURE_COLUMNS, get_stock_features
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

config = load_config()


class DualMovingAverageStrategyDeeplean(StockBacktest):

    def __init__(self, stock_id, model_path, start_date, end_date, initial_cash=100000, split_cash=0, label="backtest", ma_low="sma_50", ma_high="ema_200", loglevel=logging.INFO):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)  # 繼承父類初始化
        self.ma_low = ma_low
        self.ma_high = ma_high
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
        return float(y_pred[0][0])  # 回傳機率（可由呼叫方決定是否 > 0.7）

    def buy_signal(self, i):
        if i > 2:
            cond_ma = self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high]
            if not cond_ma:
                return False

            # 加入模型判斷（用 i-1 對應買入當日）
            cond_model = self.predict_with_model(i) > 0.7  # ✅ 模型預測為正收益

            return cond_model
        return False

    def sell_signal(self, i):
        if i > 2:
            return self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


def run_ma_deep(start_date="2020-01-01", end_date="2024-12-31", initial_cash=100000, model_path="model_20_50_stream_cnn.h5"):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")
    ma_low = "sma_20"
    ma_high = "sma_50"

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []
    label = f"{ma_low}_{ma_high}_deep"
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    for stock_id in collections:
        backtest = DualMovingAverageStrategyDeeplean(stock_id=stock_id, model_path=model_path, start_date=start_date, end_date=end_date, initial_cash=initial_cash, label=label, ma_low=ma_low, ma_high=ma_high)
        backtest.run_backtest()
        profit = backtest.cash - initial_cash
        log.info(f"{stock_id}: 初始金額{initial_cash} ,最終金額 {backtest.cash} 獲利:{math.floor(profit)}, 勝率 {backtest.win_rate:.2%}")
        total_win += backtest.win_count
        total_lose += backtest.lose_count
        total_profit += profit
        hold_days.extend(backtest.hold_days)
        trade_records.extend(backtest.trade_records)

    buy_count = total_win + total_lose
    win_rate = total_win / buy_count if buy_count > 0 else 0
    avg_profit = total_profit / buy_count if buy_count > 0 else 0

    log.info(f"總計:總營利{total_profit}, 股票數量{len(collections)},總下注量:{buy_count},每注獲利 {avg_profit:.2f}, 獲勝次數{total_win}, 總勝率 {win_rate:.2%}")
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

        log.info(f"持有天數統計: 最大 {max_days}, 最小 {min_days}, 平均 {avg_days:.2f}, 標準差 {std_days:.2f}, 眾數 {mode_days}")
    else:
        log.info("無持有天數數據")
    close_mongo_client()
