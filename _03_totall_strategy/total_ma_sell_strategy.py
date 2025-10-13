import logging
import math
import sys
import os
import numpy as np
import statistics
import pandas as pd

from _03_totall_strategy.total_strategy import MultiStockBacktest
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client

# 載入配置檔
config = load_config()


class TotalDualMovingAverageStrategy(MultiStockBacktest):
    """
    雙均線策略 (Dual Moving Average Strategy)
    - 搭配 Choppiness Index 判斷震盪狀態
    - 繼承 MultiStockBacktest (多股票回測框架)
    """

    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        ma_low="sma_20",   # 短均線
        ma_high="sma_50",  # 長均線
        loglevel=logging.INFO,
    ):
        # 呼叫父類別初始化 (設定資金、標籤、股票代號…等)
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)
        self.ma_low = ma_low
        self.ma_high = ma_high
        self.sell_price = 0.0  # 紀錄賣出價格

    def calc_choppiness_index(self, df, n=14):
        """
        計算 Choppiness Index (震盪指標)
        - 用 ATR 與最高/最低價區間計算
        - 回傳最後一筆 CI 值
        """
        if len(df) < n:
            return None  # 資料不足

        # 計算真實波幅 TR
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = n 日 TR 總和
        atr = tr.rolling(n).sum()
        hh = df["high"].rolling(n).max()  # n 日最高
        ll = df["low"].rolling(n).min()   # n 日最低

        # CI 計算公式
        ci = 100 * np.log10(atr / (hh - ll)) / np.log10(n)

        return ci.iloc[-1]  # 回傳最新值

    def buy_signal(self, df):
        """
        買進條件：
        1. 短均線黃金交叉 (由下往上突破長均線)
        2. Choppiness Index < 31 (市場非震盪區間)
        """
        if len(df) < 15:
            return False

        sub_df = df.iloc[-15:-1]  # 最近 15 天資料
        choppiness_now = self.calc_choppiness_index(sub_df)

        # 均線交叉條件
        ma_condition = (
            df.iloc[-3][self.ma_low] < df.iloc[-3][self.ma_high] and
            df.iloc[-2][self.ma_low] > df.iloc[-2][self.ma_high]
        )

        # 若 CI 存在，則需同時符合 CI < 31
        if choppiness_now is not None:
            return ma_condition and choppiness_now < 31
        else:
            return ma_condition

    def sell_signal(self, df):
        """
        賣出條件：
        1. 當前價格跌破短均線
        2. 或短均線與長均線死亡交叉 (短線由上往下跌破長線)
        """
        if len(df) < 15:
            return False

        # 跌破短均線
        if df.iloc[-2][self.ma_low] > df.iloc[-1]["low"]:
            self.sell_price = df.iloc[-2][self.ma_low]
            return True

        # 均線死亡交叉
        return (
            df.iloc[-3][self.ma_low] > df.iloc[-3][self.ma_high] and
            df.iloc[-2][self.ma_low] < df.iloc[-2][self.ma_high]
        )

    def buy_price_select(self, df):
        """
        買入價格選擇 (使用開盤價並經過跳空處理)
        """
        return self.tw_ticket_gap(df.iloc[-1]["open"])

    def sell_price_select(self, df):
        """
        賣出價格選擇：
        - 若之前已記錄 sell_price，優先使用
        - 否則使用最新開盤價
        """
        if self.sell_price > 0.0:
            temp = self.sell_price
            self.sell_price = 0.0
            return self.tw_ticket_gap(temp)
        return self.tw_ticket_gap(df.iloc[-1]["open"])


def run_ma_sell_list(start_date="2011-01-01", end_date="2023-12-31", initial_cash=100000):
    """
    執行雙均線策略回測
    - 從 MongoDB 抓取股票資料
    - 對每支股票跑回測
    - 統計總獲利、勝率與持有天數
    """
    db = get_mongo_client()

    # 過濾台股 (排除 "TWO")
    collections = [col for col in db.list_collection_names() if "TW" in col and "TWO" not in col]

    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []
    label = "volumeMA50_8_5_choppiness_31_TW"

    # 建立 log 檔
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    # 對每一檔股票執行回測
    for stock_id in collections:
        backtest = TotalDualMovingAverageStrategy(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            label=label,
            split_cash=5000,
        )
        backtest.run_backtest()

        # 計算盈虧
        profit = backtest.cash - initial_cash
        log.info(
            f"{stock_id}: 初始金額{initial_cash} , 最終金額 {backtest.cash} 獲利:{math.floor(profit)}, 勝率 {backtest.win_rate:.2%}"
        )

        total_win += backtest.win_count
        total_lose += backtest.lose_count
        total_profit += profit
        hold_days.extend(backtest.hold_days)
        trade_records.extend(backtest.trade_records)

    # 總體統計
    buy_count = total_win + total_lose
    win_rate = total_win / buy_count if buy_count > 0 else 0
    avg_profit = total_profit / buy_count if buy_count > 0 else 0

    log.info(
        f"總計: 總營利 {total_profit}, 股票數量 {len(collections)}, 總交易次數 {buy_count}, "
        f"平均單筆獲利 {avg_profit:.2f}, 獲勝次數 {total_win}, 總勝率 {win_rate:.2%}"
    )

    # 輸出交易紀錄
    if trade_records:
        df = pd.DataFrame(trade_records)
        output_folder = config.get("leaning_folder", "./stock_data/leaning_label")
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{label}_trades.csv"
        filepath = os.path.join(output_folder, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

    # 統計持有天數
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
            f"持有天數統計: 最大 {max_days}, 最小 {min_days}, 平均 {avg_days:.2f}, "
            f"標準差 {std_days:.2f}, 眾數 {mode_days}"
        )
    else:
        log.info("無持有天數數據")

    # 關閉 MongoDB 連線
    close_mongo_client()
