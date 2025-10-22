import logging
import numpy as np
import pandas as pd
from _02_strategy.base.single_strategy import StockBacktest
from modules.config_loader import load_config
from modules.process_mongo import close_mongo_client
from _03_strategy_runner import StrategyRunner  # ✅ 這是剛才建立的通用回測器


config = load_config()


# ===============================
# 雙均線交易策略 (Dual Moving Average Strategy)
# ===============================
class DualMovingAverageStrategy(StockBacktest):
    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        ma_low="sma_20",
        ma_high="sma_50",
        loglevel=logging.INFO,
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)
        self.ma_low = ma_low
        self.ma_high = ma_high
        self.sell_price = 0.0

    def calc_choppiness_index(self, df, n=14):
        if len(df) < n:
            return None
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(n).sum()
        hh = df["high"].rolling(n).max()
        ll = df["low"].rolling(n).min()
        ci = 100 * np.log10(atr / (hh - ll)) / np.log10(n)
        return ci.iloc[-1]

    def buy_signal(self, i):
        if i > 2:
            sub_df = self.data.iloc[i - 14 : i]
            choppiness_now = self.calc_choppiness_index(sub_df)
            ma_condition = (
                self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high]
            )
            if choppiness_now is not None:
                return ma_condition and choppiness_now < 31
            return ma_condition
        return False

    def sell_signal(self, i):
        if self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i]["low"]:
            self.sell_price = self.data.iloc[i - 1][self.ma_low]
            return True
        if i > 2:
            return (
                self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
            )
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        if self.sell_price > 0.0:
            temp = self.sell_price
            self.sell_price = 0.0
            return self.tw_ticket_gap(temp)
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


# ===============================
# 使用 StrategyRunner 執行回測
# ===============================
def run_dual_ma_strategy(start_date="2011-01-01", end_date="2023-12-31", initial_cash=100000):
    """
    使用通用回測器 StrategyRunner 執行雙均線策略
    """
    label = "dual_MA_choppiness_31_TW"
    runner = StrategyRunner(
        strategy_cls=DualMovingAverageStrategy,
        label=label,
        log_folder=config.get("strategy_log_folder", "./strategy_log"),
        initial_cash=initial_cash,
        show_each_stock=True,  # 若只想看總體績效可改為 False
    )
    runner.run(start_date=start_date, end_date=end_date)
    close_mongo_client()
