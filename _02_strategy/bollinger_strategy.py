import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from _02_strategy.base.single_strategy import StockBacktest

from _02_strategy.base.strategy_runner import StrategyRunner
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client

config = load_config()


class BollingerBandStrategy(StockBacktest):

    def __init__(self, stock_id, start_date, end_date, initial_cash=100000, split_cash=0, label="backtest", loglevel=logging.INFO):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)

    def buy_signal(self, i):
        if i > 2:
            prev_close = self.data.iloc[i - 1]["close"]
            prev_lower = self.data.iloc[i - 1]["bollinger_Lower"]
            prev_hight = max(self.data.iloc[i - 1]["close"], self.data.iloc[i - 1]["open"])

            return prev_close < prev_lower and self.data.iloc[i - 1]["high"] < self.data.iloc[i]["close"] and self.data.iloc[i - 1]["bollinger_Lower"] < self.data.iloc[i]["low"]
        return False

    def sell_signal(self, i):
        if i > 2:
            prev_close = self.data.iloc[i - 1]["close"]

            return prev_close > self.data.iloc[i]["close"]
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["close"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["close"])



def run_bolling_backtest(initial_cash=100000):
    folder = config.get("strategy_log_folder", "./strategy_log")
    # 整體回測起訖年
    start_year = 2015
    end_year = 2018  # 你可以依實際資料調整
    # start_year = 2021
    # end_year = 2025  # 你可以依實際資料調整

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"


    runner = StrategyRunner(
        strategy_cls=BollingerBandStrategy,
        label="dow_04_strategy_stopValue_OBV_40_80_value_10M",
        log_folder="./strategy_log"
    )
    runner.run(start_date="2010-01-01", end_date="2021-12-31")
