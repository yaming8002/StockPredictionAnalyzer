import logging
import numpy as np
import pandas as pd
from _02_strategy.base.single_strategy import StockBacktest
from _02_strategy.base.strategy_runner import StrategyRunner  # ✅ 通用回測器
from modules.config_loader import load_config
from modules.process_mongo import close_mongo_client

config = load_config()


# ===============================
# 成交量 MA5 觀察策略
# ===============================
class VolumeMA5Strategy(StockBacktest):
    """
    VolumeMA5 策略邏輯：
    - 下跌出量 → 進入觀察期（10日內）
    - 若 3 日內走平且站上 sma_120 → 隔日買入
    - 停損：跌破觀察低點
    - 停利：突破5MA後再跌破5MA
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
        self.check_low = False
        self.wait_to_sell = False
        self.check_data = None
        self.stop_loss_price = None
        self.sell_price_local = 0.0

    # ---------------------------
    # 計算 Choppiness Index（可用於確認盤整）
    # ---------------------------
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

    # ---------------------------
    # 買入條件：偵測下跌出量 + 盤整後反彈
    # ---------------------------
    def buy_signal(self, i):
        if i < 11:
            return False

        volume_avg_period = 5
        lookback_period = 3

        # 若進入觀察期
        if self.check_low:
            current_date = self.data.index[i]
            days_diff = (current_date - self.check_data).days

            # 超過觀察期或跌破停損價 → 結束觀察
            if days_diff > 10 or self.stop_loss_price > self.data.iloc[i]["low"]:
                self.check_low = False
            else:
                # 最近三日走平
                recent_range = (
                    self.data.iloc[i - 1 - lookback_period : i - 1]["close"].max()
                    - self.data.iloc[i - 1 - lookback_period : i - 1]["close"].min()
                )
                flat_condition = (
                    recent_range / self.data.iloc[i - 1 - lookback_period : i - 1]["close"].mean() < 0.01
                )

                # 若走平且站上 sma_120 → 隔日買入
                if flat_condition and self.data.iloc[i - 1]["close"] > self.data.iloc[i - 1]["sma_120"]:
                    self.check_low = False
                    return True
            return False

        # --- 否則偵測下跌出量 ---
        prev_vol_avg = self.data.iloc[i - volume_avg_period : i]["volume"].mean()
        prev_volume = self.data.iloc[i]["volume"]
        prev_close = self.data.iloc[i]["close"]
        prev_open = self.data.iloc[i]["open"]

        prev_down = prev_close < prev_open
        volume_surge = prev_volume > (prev_vol_avg * 2)

        if prev_down and volume_surge and not self.check_low:
            self.check_low = True
            self.check_data = self.data.index[i]
            self.stop_loss_price = self.data.iloc[i]["low"]

        return False

    # ---------------------------
    # 賣出條件：即時停損 + 突破後跌破5MA
    # ---------------------------
    def sell_signal(self, i):
        if i < 10:
            return False

        close_now = self.data.iloc[i - 1]["close"]
        ma5 = self.data.iloc[i - 1]["sma_5"]

        # 即時停損
        if self.stop_loss_price is not None and close_now < self.stop_loss_price:
            self.sell_price_local = self.stop_loss_price
            self.stop_loss_price = None
            return True

        # 若已突破後等待跌破5MA
        if self.wait_to_sell and close_now < ma5:
            self.wait_to_sell = False
            self.stop_loss_price = None
            self.sell_price_local = ma5
            return True

        # 若站上5MA，開始等待跌破
        if close_now > ma5:
            self.wait_to_sell = True
            return False

        # 若上漲出量 → 直接獲利了結
        volume_avg_period = 5
        prev_vol_avg = self.data.iloc[i - volume_avg_period : i]["volume"].mean()
        prev_volume = self.data.iloc[i]["volume"]
        prev_close = self.data.iloc[i]["close"]
        prev_open = self.data.iloc[i]["open"]
        prev_up = prev_close > prev_open
        volume_surge = prev_volume > (prev_vol_avg * 2)

        if prev_up and volume_surge and not self.wait_to_sell:
            self.sell_price_local = prev_close
            return True

        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["close"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.sell_price_local)


# ===============================
# 使用 StrategyRunner 執行整批回測
# ===============================
def run_volume_ma5_strategy(start_date="2015-01-01", end_date="2019-12-31", initial_cash=100000):
    label = "volumeMA20_8_5_choppiness"

    runner = StrategyRunner(
        strategy_cls=VolumeMA5Strategy,
        label=label,
        log_folder=config.get("strategy_log_folder", "./strategy_log"),
        initial_cash=initial_cash,
        show_each_stock=True,  # 若只想看總體績效可設 False
    )

    runner.run(start_date=start_date, end_date=end_date)
    close_mongo_client()
