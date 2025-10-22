import logging
import numpy as np
import pandas as pd
from _02_strategy.base.single_strategy import StockBacktest
from modules.config_loader import load_config
from modules.process_mongo import close_mongo_client
from _02_strategy.base.strategy_runner import StrategyRunner


config = load_config()


# ===============================
# 唐奇安通道策略 (Turtle Strategy)
# ===============================
class TurtleStrategy(StockBacktest):
    """
    唐奇安通道突破策略：
    - 買入條件：突破過去20日高點
    - 賣出條件：跌破過去10日低點
    """

    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        loglevel=logging.INFO,
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)

    # ---------------------------
    # 買入條件：創 20 日新高
    # ---------------------------
    def buy_signal(self, i):
        if i > 20:
            max_value = max(self.data.iloc[i - 20 : i]["high"])
            return self.data.iloc[i]["high"] > max_value
        return False

    # ---------------------------
    # 賣出條件：跌破 10 日低點
    # ---------------------------
    def sell_signal(self, i):
        if i > 10:
            min_value = min(self.data.iloc[i - 10 : i]["low"])
            return self.data.iloc[i]["low"] < min_value
        return False

    # ---------------------------
    # 買入價格選擇（取突破價）
    # ---------------------------
    def buy_price_select(self, i):
        max_value = max(self.data.iloc[i - 20 : i]["high"])
        return self.tw_ticket_gap(max_value)

    # ---------------------------
    # 賣出價格選擇（取跌破價）
    # ---------------------------
    def sell_price_select(self, i):
        min_value = min(self.data.iloc[i - 10 : i]["low"])
        return self.tw_ticket_gap(min_value)


# ===============================
# 使用 StrategyRunner 執行整批回測
# ===============================
def run_turtle_strategy(start_date="2015-01-01", end_date="2019-12-31", initial_cash=100000):
    """
    使用 StrategyRunner 執行唐奇安通道策略
    """
    label = "turtle_channel_breakout_TW"

    runner = StrategyRunner(
        strategy_cls=TurtleStrategy,
        label=label,
        log_folder=config.get("strategy_log_folder", "./strategy_log"),
        initial_cash=initial_cash,
        show_each_stock=True,  # ✅ 若只想看總體績效，可改 False
    )

    runner.run(start_date=start_date, end_date=end_date)
    close_mongo_client()
