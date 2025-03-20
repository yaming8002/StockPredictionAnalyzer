import random
import backtrader as bt
import pandas as pd

from _01_data.to_mongoDB import get_mongo_client
from _02_strategy.single_strategy import StockBacktest
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.myStockPandasData import CustomPandasData

config = load_config()
logger = setup_logger("total_summary")


def run_backtest_one_by_one():

    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    log_end_msg = "ma_count_total.log"
    total_win = 0
    total_lose = 0
    cash = 100000
    for stock_id in collections:
        backtest = StockBacktest(stock_id=stock_id, start_date="2019-01-01", end_date="2023-12-31", initial_cash=cash, logger_file=f"./{stock_id}_{log_end_msg}")
        backtest.run_backtest()
