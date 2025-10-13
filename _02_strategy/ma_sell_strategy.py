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

# 載入自訂的設定檔，內含路徑與參數設定
config = load_config()


# ===============================
# 雙均線交易策略 (Dual Moving Average Strategy)
# 繼承 StockBacktest 基底類別
# ===============================
class DualMovingAverageStrategy(StockBacktest):
    def __init__(
        self,
        stock_id,          # 股票代碼 (對應 MongoDB 的 collection 名稱)
        start_date,        # 回測開始日期
        end_date,          # 回測結束日期
        initial_cash=100000,   # 初始資金
        split_cash=0,          # 是否分散下注金額 (例如每次固定 5000 元)
        label="backtest",      # 策略標籤名稱 (方便紀錄)
        ma_low="sma_20",       # 短期移動平均線欄位名稱
        ma_high="sma_50",      # 長期移動平均線欄位名稱
        loglevel=logging.INFO, # log 紀錄等級
    ):
        # 呼叫父類別的初始化，把參數傳入基底 StockBacktest
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)

        # 記錄本策略專屬參數
        self.ma_low = ma_low
        self.ma_high = ma_high
        self.sell_price = 0.0  # 紀錄「提前觸發」的賣出價 (避免當天盤中價過低時錯過)


    # ------------------------------------
    # 計算 Choppiness Index (震盪指標)
    # n = 14 代表用 14 天的資料計算
    # 回傳值愈高代表市場盤整愈嚴重
    # ------------------------------------
    def calc_choppiness_index(self, df, n=14):
        if len(df) < n:
            return None  # 資料不足，無法計算

        # 計算真實波幅 (True Range)
        tr1 = df["high"] - df["low"]             # 當日最高 - 當日最低
        tr2 = abs(df["high"] - df["close"].shift())  # 當日最高 - 前日收盤
        tr3 = abs(df["low"] - df["close"].shift())   # 當日最低 - 前日收盤
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # 取三者最大值

        # ATR (Average True Range) 計算 → n 日 TR 的加總
        atr = tr.rolling(n).sum()
        hh = df["high"].rolling(n).max()  # n 日最高價
        ll = df["low"].rolling(n).min()   # n 日最低價

        # CI 公式：100 * log10(ATR / (最高 - 最低)) / log10(n)
        ci = 100 * np.log10(atr / (hh - ll)) / np.log10(n)

        return ci.iloc[-1]  # 回傳最新一筆 CI 值


    # ------------------------------------
    # 判斷是否觸發買入信號
    # 條件：
    #   1. 短均線突破長均線 (黃金交叉)
    #   2. 若 CI 存在，需同時 < 31 (表示盤整結束，行情可能有趨勢)
    # ------------------------------------
    def buy_signal(self, i):
        if i > 2:
            # 取前 14 天資料計算 CI
            sub_df = self.data.iloc[i - 14 : i]
            choppiness_now = self.calc_choppiness_index(sub_df)

            # 黃金交叉條件：前天短均線 < 長均線，昨天短均線 > 長均線
            ma_condition = (
                self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high]
            )

            # 若有 CI → 必須同時滿足
            if choppiness_now is not None:
                return ma_condition and choppiness_now < 31
            else:
                return ma_condition
        return False


    # ------------------------------------
    # 判斷是否觸發賣出信號
    # 條件：
    #   1. 當日低點跌破昨天短均線 → 提前設定賣出
    #   2. 均線死亡交叉 (短均線跌破長均線)
    # ------------------------------------
    def sell_signal(self, i):
        # 條件1：若今天最低價 < 昨天短均線 → 提前設賣價
        if self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i]["low"]:
            self.sell_price = self.data.iloc[i - 1][self.ma_low]
            return True

        # 條件2：死亡交叉
        if i > 2:
            return (
                self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
            )
        return False


    # ------------------------------------
    # 選擇實際買入價格
    # 預設：開盤價，並調整為台股最小跳動單位
    # ------------------------------------
    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


    # ------------------------------------
    # 選擇實際賣出價格
    # 若有「提前設定的賣價」 → 直接使用
    # 否則：使用開盤價
    # ------------------------------------
    def sell_price_select(self, i):
        if self.sell_price > 0.0:
            temp = self.sell_price
            self.sell_price = 0.0  # 用過一次就清空
            return self.tw_ticket_gap(temp)
        return self.tw_ticket_gap(self.data.iloc[i]["open"])



# ===============================
# 批次執行回測並統計結果
# ===============================
def run_ma_sell_list(start_date="2011-01-01", end_date="2023-12-31", initial_cash=100000):
    # 連線 MongoDB，取得股票資料庫
    db = get_mongo_client()

    # 篩選出「上市股票」集合 (名稱含 TW，但排除 OTC TWO)
    collections = [col for col in db.list_collection_names() if "TW" in col and "TWO" not in col]

    # 讀取策略 log 檔路徑
    strategy_log_folder = config.get("strategy_log_folder", "./strategy_log")

    # 初始化統計數據
    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []      # 紀錄持有天數
    trade_records = []  # 紀錄交易細節
    label = "volumeMA50_8_5_choppiness_31_TW"

    # 建立 log 檔
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    # ------------------------------------
    # 逐一股票進行回測
    # ------------------------------------
    for stock_id in collections:
        backtest = DualMovingAverageStrategy(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            label=label,
            split_cash=5000  # 每次下注金額 5000 元
        )
        # 執行回測流程
        backtest.run_backtest()
        profit = backtest.cash - initial_cash  # 總獲利

        # log 輸出單檔績效
        log.info(
            f"{stock_id}: 初始金額 {initial_cash}, 最終金額 {backtest.cash}, "
            f"獲利 {math.floor(profit)}, 勝率 {backtest.win_rate:.2%}"
        )

        # 累加到總體績效
        total_win += backtest.win_count
        total_lose += backtest.lose_count
        total_profit += profit
        hold_days.extend(backtest.hold_days)
        trade_records.extend(backtest.trade_records)

    # ------------------------------------
    # 總體統計
    # ------------------------------------
    buy_count = total_win + total_lose  # 總交易次數
    win_rate = total_win / buy_count if buy_count > 0 else 0
    avg_profit = total_profit / buy_count if buy_count > 0 else 0

    log.info(
        f"總計: 總營利 {total_profit}, 股票數量 {len(collections)}, 總下注量 {buy_count}, "
        f"每注平均獲利 {avg_profit:.2f}, 獲勝次數 {total_win}, 總勝率 {win_rate:.2%}"
    )

    # ------------------------------------
    # 輸出所有交易紀錄至 CSV 檔
    # ------------------------------------
    if len(trade_records) > 0:
        df = pd.DataFrame(trade_records)
        output_folder = config.get("leaning_folder", "./stock_data/leaning_label")
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{label}_trades.csv"
        filepath = os.path.join(output_folder, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")  # utf-8-sig 避免 Excel 亂碼

    # ------------------------------------
    # 統計持有天數
    # ------------------------------------
    if hold_days:
        max_days = max(hold_days)
        min_days = min(hold_days)
        avg_days = np.mean(hold_days)
        std_days = np.std(hold_days, ddof=1)  # 標準差
        try:
            mode_days = statistics.mode(hold_days)  # 眾數
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
