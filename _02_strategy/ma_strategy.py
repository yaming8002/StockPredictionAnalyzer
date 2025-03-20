import math
import sys
import os
from _02_strategy.single_strategy import StockBacktest
from modules.logger import setup_logger
from modules.process_mongo import get_mongo_client


class DualMovingAverageStrategy(StockBacktest):

    def __init__(self, stock_id, start_date, end_date, initial_cash=100000, split_cash=0, logger_file="backtest.log", ma_low="sma_50", ma_high="ema_200"):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, logger_file)  # 繼承父類初始化
        self.ma_low = ma_low
        self.ma_high = ma_high

    def buy_signal(self, i):
        if i > 2:
            return self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high]
        return False

    def sell_signal(self, i):
        if i > 2:
            return self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high] and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
        return False

    def process_buy(self, i):
        self.buy_price = self.data.iloc[i]["open"]
        self.position = self.split_cash // self.buy_price
        if self.position <= 0:
            return
        tax = self.count_tax(self.buy_price, self.position)
        self.cash -= self.position * self.buy_price + tax
        self.log_transaction("BUY", i, self.buy_price, self.position, tax)

    def process_sell(self, i):
        sell_price = self.data.iloc[i]["open"]
        tax = self.count_tax(sell_price, self.position, is_sell=True)
        profit = (sell_price - self.buy_price) * self.position - tax
        self.cash += sell_price * self.position - tax
        self.win_count += 1 if profit > 0 else 0
        self.lose_count += 1 if profit <= 0 else 0
        self.log_transaction("SELL", i, sell_price, self.position, tax)
        self.position = 0
        self.buy_price = None


def run_ma_backtest(start_date="2015-01-01", end_date="2023-12-31", initial_cash=100000):

    sma_labs = ["sma_5", "sma_20", "sma_50", "sma_60", "sma_120", "sma_200"]
    ema_labs = ["ema_5", "ema_20", "ema_50", "ema_60", "ema_120", "ema_200"]
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    total_win = 0
    total_lose = 0
    total_profit = 0.0  # 計算總獲利
    for i in range(len(sma_labs)):
        for j in range(i + 1, len(sma_labs)):
            log = setup_logger(f"./strategy_log/{sma_labs[i]}_{sma_labs[j]}_total_summary.log")
            for stock_id in collections:
                backtest = DualMovingAverageStrategy(stock_id=stock_id, start_date=start_date, end_date=end_date, initial_cash=initial_cash, logger_file=f"./strategy_log/{sma_labs[i]}_{sma_labs[j]}_total_summary.log")
                backtest.run_backtest()
                profit = backtest.cash - initial_cash
                log.info(f"{stock_id}: 初始金額{initial_cash} ,最終金額 {backtest.cash} 獲利:{math.floor(profit)}, 勝率 {backtest.win_rate:.2%}")
                total_win += backtest.win_count
                total_lose += backtest.lose_count
                total_profit += profit
            buy_count = total_win + total_lose
            win_rate = total_win / buy_count if buy_count > 0 else 0
            avg_profit = total_profit / buy_count

            log.info(f"總計:總營利{total_profit}, 股票數量{len(collections)},總下注量:{buy_count},每注獲利 {avg_profit:.2%}, 獲勝次數{total_win}, 總勝率 {win_rate:.2%}")

    total_win = 0
    total_lose = 0
    total_profit = 0.0  # 計算總獲利
    for i in range(len(ema_labs)):
        for j in range(i + 1, len(ema_labs)):
            log = setup_logger(f"./strategy_log/{ema_labs[i]}_{ema_labs[j]}/total_summary.log")
            for stock_id in collections:
                backtest = DualMovingAverageStrategy(stock_id=stock_id, start_date=start_date, end_date=end_date, initial_cash=initial_cash, logger_file=f"./strategy_log/{sma_labs[i]}_{sma_labs[j]}/{stock_id}.log")
                backtest.run_backtest()
                profit = backtest.cash - initial_cash
                log.info(f"{stock_id}: 初始金額{initial_cash} ,最終金額 {backtest.cash} 獲利:{profit}, 勝率 {backtest.win_rate:.2%}")
                total_win += backtest.win_count
                total_lose += backtest.lose_count
                total_profit += profit

            win_rate = total_win / (total_win + total_lose) if ((total_win + total_lose)) > 0 else 0
            avg_profit = total_profit / len(collections)

            log.info(f"總計:總營利{total_profit}, 股票數量{len(collections)}, 平均獲利 {avg_profit:.2%}, 總勝率 {win_rate:.2%}")
