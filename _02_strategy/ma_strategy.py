import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from _02_strategy.single_strategy import StockBacktest
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client

config = load_config()


class DualMovingAverageStrategy(StockBacktest):

    def __init__(self, stock_id, start_date, end_date, initial_cash=100000, split_cash=0, label="backtest", ma_low="sma_50", ma_high="ema_200", loglevel=logging.INFO):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)  # 繼承父類初始化
        self.ma_low = ma_low
        self.ma_high = ma_high

    def buy_signal(self, i):
        if i > 2:
            return self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high] and self.data.iloc[i - 1]["volume"] > 500000
        return False

    def sell_signal(self, i):
        if i > 2:
            return self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


def run_ma_list(ma_labs: list, start_date="2015-01-01", end_date="2019-12-31", initial_cash=100000):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")

    for i in range(len(ma_labs)):
        for j in range(i + 1, len(ma_labs)):
            total_win = 0
            total_lose = 0
            total_profit = 0.0
            hold_days = []
            trade_records = []
            label = f"{ma_labs[i]}_{ma_labs[j]}_volume"
            log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
            log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

            for stock_id in collections:
                backtest = DualMovingAverageStrategy(stock_id=stock_id, start_date=start_date, end_date=end_date, initial_cash=initial_cash, label=label, ma_low=ma_labs[i], ma_high=ma_labs[j])
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


def run_ma_backtest(start_date="2012-01-01", end_date="2019-12-31", initial_cash=100000):
    sma_labs = ["sma_5", "sma_20", "sma_50", "sma_60", "sma_120", "sma_200"]
    ema_labs = ["ema_5", "ema_20", "ema_50", "ema_60", "ema_120", "ema_200"]

    # run_ma_list(sma_labs, start_date=start_date, end_date=end_date, initial_cash=initial_cash)
    run_ma_list(ema_labs, start_date=start_date, end_date=end_date, initial_cash=initial_cash)
