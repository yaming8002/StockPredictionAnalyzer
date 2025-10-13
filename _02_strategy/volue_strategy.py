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


class VolumeMA5Strategy(StockBacktest):

    def __init__(
        self, stock_id, start_date, end_date, initial_cash=100000, split_cash=0, label="backtest", loglevel=logging.INFO
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)
        self.check_low = False
        self.wait_to_sell = False
        self.check_data = None
        self.stop_loss_price = None
        self.sell_price_local = 0.0

    def calc_choppiness_index(self, df, n=14):
        """
        傳入 n 筆資料的 DataFrame，回傳最新一筆的 Choppiness Index 值
        """
        if len(df) < n:
            return None  # 不足 n 筆資料無法計算

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
        """
        偵測下跌出量，進入觀察；六個交易日內走平，隔日買進
        """
        if i < 11:
            return False

        volume_avg_period = 5
        lookback_period = 3
        # 有啟動觀察，檢查是否六個交易日內
        if self.check_low:
            current_date = self.data.index[i]
            days_diff = (current_date - self.check_data).days

            if days_diff > 10 or self.stop_loss_price > self.data.iloc[i]["low"]:
                self.check_low = False  # 觀察期結束
            else:
                # 近三日走平判斷
                recent_range = (
                    self.data.iloc[i - 1 - lookback_period : i - 1]["close"].max()
                    - self.data.iloc[i - 1 - lookback_period : i - 1]["close"].min()
                )
                flat_condition = recent_range / self.data.iloc[i - 1 - lookback_period : i - 1]["close"].mean() < 0.01

                if flat_condition and self.data.iloc[i - 1]["close"] > self.data.iloc[i - 1]["sma_120"]:
                    self.check_low = False  # 走平確認，結束觀察
                    return True
                # sub_df = self.data.iloc[i - 13 : i + 1]  # 取 i-13 到 i 共14筆
                # choppiness_now = self.calc_choppiness_index(sub_df)

                # if choppiness_now is not None and choppiness_now > 60:
                #     self.check_low = False
                #     return True
            return False

        # 取得資料
        prev_vol_avg = self.data.iloc[i - volume_avg_period : i]["volume"].mean()
        prev_volume = self.data.iloc[i]["volume"]
        prev_close = self.data.iloc[i]["close"]
        prev_open = self.data.iloc[i]["open"]

        # 確認下跌出量，進入觀察期
        prev_down = prev_close < prev_open
        volume_surge = prev_volume > (prev_vol_avg * 2)

        if prev_down and volume_surge and not self.check_low:
            self.check_low = True
            self.check_data = self.data.index[i]  # 用 index 存日期
            self.stop_loss_price = self.data.iloc[i]["low"]

        return False

    def sell_signal(self, i):
        """
        即時停損：跌破出量低點
        隔日賣出：突破5MA後再跌破5MA
        """
        if i < 10:
            return False

        close_now = self.data.iloc[i - 1]["close"]
        ma5 = self.data.iloc[i - 1]["sma_5"]
        # print(ma5, self.stop_loss_price)
        # 即時停損
        if self.stop_loss_price is not None and close_now < self.stop_loss_price:
            self.sell_price_local = self.stop_loss_price
            self.stop_loss_price = None
            return True

        prev_close = self.data.iloc[i - 1]["close"]

        if self.wait_to_sell and close_now < ma5:
            self.wait_to_sell = False
            self.stop_loss_price = None
            self.sell_price_local = ma5
            return True

        if prev_close > ma5:
            self.wait_to_sell = True
            return False

        volume_avg_period = 5
        # 取得資料
        prev_vol_avg = self.data.iloc[i - volume_avg_period : i]["volume"].mean()
        prev_volume = self.data.iloc[i]["volume"]
        prev_close = self.data.iloc[i]["close"]
        prev_open = self.data.iloc[i]["open"]

        # 確認下跌出量，進入觀察期
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


def run_VolumeMA5_list(start_date="2015-01-01", end_date="2019-12-31", initial_cash=100000):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []
    label = "volumeMA20_8_5_choppiness"
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    for stock_id in collections:
        backtest = VolumeMA5Strategy(
            stock_id=stock_id, start_date=start_date, end_date=end_date, initial_cash=initial_cash, label=label
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
