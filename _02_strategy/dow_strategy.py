from datetime import timedelta
import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from scipy import stats
from _02_strategy.base.single_strategy import StockBacktest

from _02_strategy.base.strategy_runner import StrategyRunner
from _04_analysis.hold_days import analyze_hold_days
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client

config = load_config()


# --- 1) 以百分比 ZigZag 偵測轉折 ---
def compute_swings_zigzag_inline(df, pct=0.02):
    """
    即時版 ZigZag：直接在 df 上標記最近轉折點。
    新增欄位：
      - 'turn_high': 最近的高點價
      - 'turn_low': 最近的低點價

    規則：
      - 當上漲段回撤超過 pct → 記錄高點轉折
      - 當下跌段反彈超過 pct → 記錄低點轉折
    """

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    n = len(df)
    if n == 0:
        df["turn_high"] = np.nan
        df["turn_low"] = np.nan
        return df

    # 初始化
    turn_highs = np.full(n, np.nan)
    turn_lows = np.full(n, np.nan)
    direction = 0  # 0 未定, +1 上漲, -1 下跌
    ext_hi = highs[0]
    ext_hi_i = 0
    ext_lo = lows[0]
    ext_lo_i = 0
    last_turn_high = np.nan
    last_turn_low = np.nan

    for i in range(1, n):
        h, l, c = highs[i], lows[i], closes[i]

        # 更新當前極值
        if direction >= 0 and h >= ext_hi:
            ext_hi = h
            ext_hi_i = i
        if direction <= 0 and l <= ext_lo:
            ext_lo = l
            ext_lo_i = i

        made_pivot = False

        # 上漲段：檢查是否回撤超過門檻（出現高點轉折）
        if direction >= 0:
            retrace = (ext_hi - l) / ext_hi
            if retrace >= pct:
                last_turn_high = ext_hi
                turn_highs[ext_hi_i] = ext_hi
                direction = -1
                ext_lo = l
                ext_lo_i = i
                made_pivot = True

        # 下跌段：檢查是否反彈超過門檻（出現低點轉折）
        if not made_pivot and direction <= 0:
            bounce = (h - ext_lo) / ext_lo
            if bounce >= pct:
                last_turn_low = ext_lo
                turn_lows[ext_lo_i] = ext_lo
                direction = +1
                ext_hi = h
                ext_hi_i = i
                made_pivot = True

        # 特殊情況：同一根K棒同時創更高與更低
        if highs[i] > ext_hi and lows[i] < ext_lo:
            prev_close = closes[i - 1]
            if c > prev_close:
                ext_hi = h
                ext_hi_i = i
                direction = +1
            elif c < prev_close:
                ext_lo = l
                ext_lo_i = i
                direction = -1

        # 即時更新目前為止的最近轉折
        if not np.isnan(last_turn_high):
            turn_highs[i] = last_turn_high
        if not np.isnan(last_turn_low):
            turn_lows[i] = last_turn_low

    # 最後更新 DataFrame
    df["turn_high"] = turn_highs
    df["turn_low"] = turn_lows
    return df


def compute_swings_zigzag_close_inline(df, hi_pct=0.02, low_pct=None):
    """
    🔹 以「收盤價」為主的即時 ZigZag 偵測。

    功能：
      - 直接在 df 上標記最近的高/低轉折。
    新增欄位：
      - 'turn_high': 最近的高點價
      - 'turn_low': 最近的低點價

    規則：
      - 當上漲段下跌超過 pct → 記錄高點轉折
      - 當下跌段上漲超過 pct → 記錄低點轉折
    """

    if not low_pct:
        low_pct = hi_pct

    n = len(df)
    if n == 0:
        df["turn_high"] = np.nan
        df["turn_low"] = np.nan
        return df
    closes = df["close"].to_numpy()

    # 初始化
    turn_highs = np.full(n, np.nan)
    turn_lows = np.full(n, np.nan)
    direction = 0  # 0 未定, +1 上漲, -1 下跌
    ext_hi = closes[0]
    ext_hi_i = 0
    ext_lo = closes[0]
    ext_lo_i = 0
    last_turn_high = np.nan
    last_turn_low = np.nan

    for i in range(1, n):
        h, l, c = closes[i], closes[i], closes[i]

        # 更新當前極值
        if direction >= 0 and h >= ext_hi:
            ext_hi = h
            ext_hi_i = i
        if direction <= 0 and l <= ext_lo:
            ext_lo = l
            ext_lo_i = i

        made_pivot = False

        # 上漲段：檢查是否回撤超過門檻（出現高點轉折）
        if direction >= 0:
            retrace = (ext_hi - l) / ext_hi
            if retrace >= hi_pct:
                last_turn_high = ext_hi
                turn_highs[ext_hi_i] = ext_hi
                direction = -1
                ext_lo = l
                ext_lo_i = i
                made_pivot = True

        # 下跌段：檢查是否反彈超過門檻（出現低點轉折）
        if not made_pivot and direction <= 0:
            bounce = (h - ext_lo) / ext_lo
            if bounce >= low_pct:
                last_turn_low = ext_lo
                turn_lows[ext_lo_i] = ext_lo
                direction = +1
                ext_hi = h
                ext_hi_i = i
                made_pivot = True

        # 即時更新目前為止的最近轉折
        if not np.isnan(last_turn_high):
            turn_highs[i] = last_turn_high
        if not np.isnan(last_turn_low):
            turn_lows[i] = last_turn_low

    # 最後更新 DataFrame
    df["turn_high"] = turn_highs
    df["turn_low"] = turn_lows
    return df


class DowStrategy(StockBacktest):
    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        loglevel=logging.INFO,
        show_logger=False,
    ):
        super().__init__(
            stock_id,
            start_date,
            end_date,
            initial_cash,
            split_cash,
            label,
            loglevel,
            show_logger,
        )
        # ✅ 在初始化時計算 ZigZag 轉折
        # self.data = compute_swings_zigzag_inline(self.data, pct=0.06)
        self.data = compute_swings_zigzag_close_inline(self.data, hi_pct=0.04)

        self.data["rsi"] = self.compute_rsi(self.data["close"], period=14)
        # self.data = compute_swings_zigzag_close_inline_multip(self.data, hi_pct=0.06,lookahead=1)
        self.sell_value = 0.0
        self.buy_index = 0

    def compute_rsi(self, close_series: pd.Series, period: int = 14):
        """
        計算 RSI（相對強弱指標）
        回傳一個與 close_series 等長的 Series。
        """
        delta = close_series.diff()

        # 漲跌分開計算
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # 使用 EMA 平滑平均
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def ervey_date_work(self):
        pass

    def compute_obv(self, i, window=20):
        """即時計算 OBV（On-Balance Volume），標準化至 0～100"""
        if i < window:
            return None

        df = self.data.iloc[i - window : i].copy()
        if "close" not in df.columns or "volume" not in df.columns:
            return None

        df["direction"] = np.sign(df["close"].diff()).fillna(0)
        df["OBV_raw"] = (df["direction"] * df["volume"]).cumsum()

        obv_min, obv_max = df["OBV_raw"].min(), df["OBV_raw"].max()
        if obv_max == obv_min:
            return 50.0

        obv_value = df["OBV_raw"].iloc[-1]
        return (obv_value - obv_min) / (obv_max - obv_min) * 100

    def compute_choppiness_index(self, i, window=14):
        """計算 CHOP 指數，<25 視為有趨勢"""
        if i < window:
            return False

        df = self.data.iloc[i - window : i].copy()

        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_close"] = (df["low"] - df["close"].shift(1)).abs()
        df["TR"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(window=window).mean()

        high_n, low_n = df["high"].max(), df["low"].min()
        sum_atr = df["ATR"].sum()
        if high_n == low_n:
            return False

        chop = 100 * np.log10(sum_atr / (high_n - low_n)) / np.log10(window)
        return chop < 35

    # --- 買進條件 ---
    def buy_signal(self, i):
        if i <= 20:
            return False

        close = self.data["close"].iloc[i - 1]
        open = self.data["open"].iloc[i - 1]
        turn_high = self.data["turn_high"].iloc[i - 1]
        turn_low = self.data["turn_low"].iloc[i - 1]
        open_price = self.data["open"].iloc[i]
        volume = self.data["volume"].iloc[i - 1]
        total_value = close * volume
        open_close_max = max(open, close)
        is_long_high = abs(self.data["high"].iloc[i - 1] - open_close_max) * 3 > abs(close - open)
        # 計算 5 日平均成交量
        volume_ma5 = self.data["volume"].iloc[i - 6 : i - 1].mean()

        if not np.isnan(turn_high) and close > turn_high and open_price > turn_high:
            obv = self.compute_obv(i - 1)
            # 設定停損價為最近低點（下修1%防止被掃）
            if (
                obv
                and 40 <= obv <= 80
                and volume_ma5 * 1.5 < volume
                # and self.data["volume"].iloc[i-1] > 1000_000):
                and total_value > 10_000_000
                and not is_long_high
                # and self.data["rsi"].iloc[i - 1] > 40
            ):
                self.stop_value = turn_low
                self.trade_value = (open_price - turn_low) * 4 + open_price
                self.buy_index = i
                return True

        return False

    # --- 賣出條件 ---
    def sell_signal(self, i):
        if i <= 20:
            return False
        if self.stop_value and self.data["low"].iloc[i - 1] < self.stop_value:
            # self.sell_value = self.stop_value
            self.sell_index = i
            return True  # 停損

        close = self.data["close"].iloc[i - 1]

        turn_low = self.data["turn_low"].iloc[i - 1]
        if not np.isnan(turn_low) and close < turn_low:
            self.sell_index = i
            return True
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        sell_number = self.sell_value if self.sell_value != 0.0 else self.data.iloc[i]["open"]
        self.sell_value = 0.0
        return self.tw_ticket_gap(sell_number)

    # def sell_price_select(self, i):
    #     return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def insert_trade_record(
        self, buy_date, buy_price, buy_tax, quantity, sell_date, sell_price, sell_tax, days, profit
    ) -> None:
        # 取得當時的 ZigZag 資料
        buy_turn_high = float(self.data["turn_high"].iloc[self.buy_index])
        buy_turn_low = float(self.data["turn_low"].iloc[self.buy_index])
        sell_turn_high = float(self.data["turn_high"].iloc[self.sell_index])
        sell_turn_low = float(self.data["turn_low"].iloc[self.sell_index])

        # 建立交易紀錄
        record = {
            "stock_id": self.stock_id,
            "buy_date": buy_date.strftime("%Y-%m-%d"),
            "buy_price": round(buy_price, 2),
            "buy_tax": round(buy_tax, 2),
            "buy_turn_high": round(buy_turn_high, 2) if not np.isnan(buy_turn_high) else None,
            "buy_turn_low": round(buy_turn_low, 2) if not np.isnan(buy_turn_low) else None,
            "quantity": int(quantity),
            "sell_date": sell_date.strftime("%Y-%m-%d"),
            "sell_price": round(sell_price, 2),
            "sell_tax": round(sell_tax, 2),
            "sell_turn_high": round(sell_turn_high, 2) if not np.isnan(sell_turn_high) else None,
            "sell_turn_low": round(sell_turn_low, 2) if not np.isnan(sell_turn_low) else None,
            "hold_days": int(days),
            "profit": round(profit, 2),
        }

        self.trade_records.append(record)


def run_dow_backtest(initial_cash=100000):
    folder = config.get("strategy_log_folder", "./strategy_log")
    # 整體回測起訖年
    start_year = 2015
    end_year = 2018  # 你可以依實際資料調整
    # start_year = 2021
    # end_year = 2025  # 你可以依實際資料調整

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    runner = StrategyRunner(
        strategy_cls=DowStrategy,
        label="dow_zigzag_breakout_obv40_80_volx1_5_nolengthywick_rsi30",
        log_folder=folder,
    )
    runner.run(start_date="2010-01-01", end_date="2021-12-31")
