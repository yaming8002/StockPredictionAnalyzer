from datetime import timedelta
import logging
import math
import sys
import os
import numpy as np
import statistics

import pandas as pd
from scipy import stats
from _02_strategy.single_strategy import StockBacktest
from modules.config_loader import load_config
from modules.logger import setup_logger
from modules.process_mongo import close_mongo_client, get_mongo_client

config = load_config()
# 一個簡單的農曆新年對照表（你可以加更多年份）
LUNAR_NEW_YEAR_DATES = {
    2010: pd.Timestamp("2010-02-14"),
    2011: pd.Timestamp("2011-02-03"),
    2012: pd.Timestamp("2012-01-23"),
    2013: pd.Timestamp("2013-02-10"),
    2014: pd.Timestamp("2014-01-31"),
    2015: pd.Timestamp("2015-02-19"),
    2016: pd.Timestamp("2016-02-08"),
    2017: pd.Timestamp("2017-01-28"),
    2018: pd.Timestamp("2018-02-16"),
    2019: pd.Timestamp("2019-02-05"),
    2020: pd.Timestamp("2020-01-25"),
    2021: pd.Timestamp("2021-02-12"),
    2022: pd.Timestamp("2022-02-01"),
    2023: pd.Timestamp("2023-01-22"),
    2024: pd.Timestamp("2024-02-10"),
    2025: pd.Timestamp("2025-01-29"),
}


def is_one_week_before_chinese_new_year(current_date):
    year = current_date.year
    if year in LUNAR_NEW_YEAR_DATES:
        new_year = LUNAR_NEW_YEAR_DATES[year]
        return (new_year - timedelta(days=14) <= current_date) and (current_date < new_year)
    return False


class DualMovingAverageStrategy(StockBacktest):
    def __init__(
        self,
        stock_id,
        start_date,
        end_date,
        initial_cash=100000,
        split_cash=0,
        label="backtest",
        ma_low="sma_50",
        ma_high="ema_200",
        loglevel=logging.INFO,
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)  # 繼承父類初始化
        self.ma_low = ma_low
        self.ma_high = ma_high

    def ervey_date_work(self):
        pass

    def compute_obv(self, df):
        df = df.copy()
        df["direction"] = np.sign(df["close"].diff())
        df["direction"].fillna(0, inplace=True)
        df["OBV_raw"] = (df["direction"] * df["volume"]).cumsum()

        # 正規化 OBV 到 0~100 範圍
        obv_min = df["OBV_raw"].min()
        obv_max = df["OBV_raw"].max()
        if obv_max != obv_min:
            df["OBV"] = (df["OBV_raw"] - obv_min) / (obv_max - obv_min) * 100
        else:
            df["OBV"] = 50  # 若 OBV 無變化，給中性值
        return df

    def is_consolidating(self, i, window=20, threshold=0.05):
        if i < window:
            return False
        window_data = self.data.iloc[i - window : i]
        high = window_data["high"].max()
        low = window_data["low"].min()
        return (high - low) / low <= threshold

    def compute_rsi(self, i, window=14):
        """
        計算從 i-window 到 i 這段區間的 RSI 值，
        回傳 RSI 數值 (float)，若資料不足則回傳 None
        """
        if i < window:
            return None  # 資料不足

        df = self.data.iloc[i - window : i].copy()

        # 計算價差
        delta = df["close"].diff()

        # 拆分漲跌
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # 平均漲跌幅
        avg_gain = gain.mean()
        avg_loss = loss.mean()

        if avg_loss == 0:  # 避免除以 0
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_obv(self, i, window=20):
        """
        即時計算 OBV（On-Balance Volume）
        - 使用最近 window 根 K 線的資料
        - 回傳標準化後 OBV 值 (0~100)
        - 若資料不足或 volume/close 缺失，回傳 None
        """
        if i < window:
            return None

        df = self.data.iloc[i - window : i].copy()
        if "close" not in df.columns or "volume" not in df.columns:
            return None

        # 計算收盤變化方向
        df["direction"] = np.sign(df["close"].diff())
        df["direction"] = df["direction"].fillna(0)  # ✅ 改這裡

        # 計算累積 OBV
        df["OBV_raw"] = (df["direction"] * df["volume"]).cumsum()

        # 標準化到 0～100
        obv_min = df["OBV_raw"].min()
        obv_max = df["OBV_raw"].max()
        if obv_max == obv_min:
            return 50.0

        obv_value = df["OBV_raw"].iloc[-1]
        obv_norm = (obv_value - obv_min) / (obv_max - obv_min) * 100

        return obv_norm
    
    def compute_choppiness_index(self, i, window=14):
        """
        計算從 i-window 到 i 這段區間的 CHOP 值，
        回傳 True 表示趨勢明顯 (CHOP < 45)
        """
        if i < window:
            return False  # 資料不足，無法計算

        df = self.data.iloc[i - window: i].copy()

        # 計算 True Range
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_close"] = (df["low"] - df["close"].shift(1)).abs()
        df["TR"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

        # 計算 ATR
        df["ATR"] = df["TR"].rolling(window=window).mean()

        # 計算 CHOP
        high_n = df["high"].max()
        low_n = df["low"].min()
        sum_atr = df["ATR"].sum()

        if (high_n - low_n) == 0:  # 防止除以 0
            return False

        chop_value = 100 * np.log10(sum_atr / (high_n - low_n)) / np.log10(window)

        # ✅ 回傳 True 表示「有趨勢」，False 表示「盤整」
        return chop_value < 35


    def buy_signal(self, i):
        if i > 20:
            current_date = self.data.index[i]
            if is_one_week_before_chinese_new_year(current_date):
                return False

            if self.data.iloc[i - 1]["close"] > self.data.iloc[i]["open"] :
                return False
            # 均線黃金交叉條件
            cross_signal = (
                self.data.iloc[i - 2][self.ma_low] < self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] > self.data.iloc[i - 1][self.ma_high]
            )

            if not cross_signal:
                return False

            # 趨勢濾網：CHOP < 35
            has_trend = self.compute_choppiness_index(i-1)
            if not has_trend:
                return False

            # 動能濾網：RSI 介於 45～70
            # rsi = self.compute_rsi(i-1)
            # if rsi is None or not (55 <= rsi <= 70):
            #     return False
            obv = self.compute_obv(i-1)
            if obv is None or not (40 <= obv <= 70):
                return False
            return True

        return False


    def sell_signal(self, i):
        if i > 2:

            return (
                self.data.iloc[i - 2][self.ma_low] > self.data.iloc[i - 2][self.ma_high]
                and self.data.iloc[i - 1][self.ma_low] < self.data.iloc[i - 1][self.ma_high]
                
            )
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


def ot_run_ma_list(ma_labs: list, start_date="2015-01-01", end_date="2019-12-31", folder="" , initial_cash=100000):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder =folder 
    collections.sort()
    for i in range(len(ma_labs)):
        for j in range(i + 1, len(ma_labs)):
            total_win = 0
            total_lose = 0
            total_profit = 0.0
            hold_days = []
            trade_records = []  # opt_sma_120_sma_200_3_sell_3_AVG_trades
            label = f"{ma_labs[i]}_{ma_labs[j]}_CHOP35_OBV_close_5000_anlzy"
            log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
            log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

            for stock_id in collections:
                try:
                    backtest = DualMovingAverageStrategy(
                        stock_id=stock_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_cash=initial_cash,
                        split_cash = 5000,
                        label=label,
                        ma_low=ma_labs[i],
                        ma_high=ma_labs[j],
                    )
                    backtest.run_backtest()
                except Exception as e:
                    print(f"⚠️ 忽略錯誤組合 {ma_labs[i]} / {ma_labs[j]}，錯誤原因：{e}")
                    continue
                buy_count = backtest.win_count + backtest.lose_count
                profit = backtest.cash - initial_cash
                log.info(
                    f"{stock_id}: 初始金額:{initial_cash:,} ,最終金額:{backtest.cash:,} ,交易次數:{buy_count} ,總獲利:{math.floor(profit):,} ,勝率:{backtest.win_rate:.2%}"
                )
                total_win += backtest.win_count
                total_lose += backtest.lose_count
                total_profit += profit
                hold_days.extend(backtest.hold_days)
                trade_records.extend(backtest.trade_records)
                
            buy_count = total_win + total_lose
            win_rate = total_win / buy_count if buy_count > 0 else 0
            avg_profit = total_profit / buy_count if buy_count > 0 else 0

            log.info("===============================================")
            log.info("📊 回測績效總結")
            log.info("-----------------------------------------------")
            log.info(f"總股票數量：{len(collections)}")
            log.info(f"總交易次數：{buy_count}")
            log.info(f"總勝率：{win_rate:.2%}")
            log.info(f"總獲利金額：{total_profit:,.0f}")
            log.info(f"平均每筆盈虧：{avg_profit:,.2f}")
            log.info("-----------------------------------------------")

            # ✅ 詳細績效分析
            if len(trade_records) > 0:
                df = pd.DataFrame(trade_records)

                # 計算報酬率與標記獲利／虧損
                df["profit_rate"] = (df["sell_price"] - df["buy_price"]) / df["buy_price"] * 100
                df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
                df = df[df["profit"] != 0]

                # 設定容忍誤差閾值（例如 ±1 元 或 ±0.1%）

                win_df = df[df["profit"] > 0]      # 明確正報酬
                lose_df = df[df["profit"] < 0]    # 明確負報酬

                avg_win = win_df["profit"].mean() if not win_df.empty else 0
                avg_lose = lose_df["profit"].mean() if not lose_df.empty else 0
                avg_win_rate = win_df["profit_rate"].mean() if not win_df.empty else 0
                avg_lose_rate = lose_df["profit_rate"].mean() if not lose_df.empty else 0

                max_win = df["profit"].max()
                max_lose = df["profit"].min()
                avg_hold_days = df["hold_days"].mean()

                # 計算期望報酬值（Expected Value）
                expect_value = win_rate * avg_win + (1 - win_rate) * avg_lose

                log.info("📈 詳細績效統計")
                log.info("-----------------------------------------------")
                log.info(f"平均獲利金額：{avg_win:,.2f}")
                log.info(f"平均虧損金額：{avg_lose:,.2f}")
                log.info(f"平均獲利報酬率：{avg_win_rate:.2f}%")
                log.info(f"平均虧損報酬率：{avg_lose_rate:.2f}%")
                log.info(f"最大單筆獲利：{max_win:,.2f}")
                log.info(f"最大單筆虧損：{max_lose:,.2f}")
                log.info(f"平均持有天數：{avg_hold_days:.1f} 天")
                log.info(f"期望報酬值（EV）：{expect_value:,.2f}")
                log.info("-----------------------------------------------")

                # =====================================================
                # 🔹 信賴區間 & IQR 排除極值分析
                # =====================================================

                def trim_outliers(series, mode="auto"):
                    """
                    使用 Median Absolute Deviation (MAD) 排除極值
                    """
                    if len(series) == 0:
                        return series, None, None

                    median = series.median()
                    mad = (abs(series - median)).median()
                    k = 3  # 相當於 ±3σ

                    lower = median - k * mad
                    upper = median + k * mad

                    if mode == "win":
                        lower = max(lower, 0)
                    elif mode == "lose":
                        upper = min(upper, 0)

                    filtered = series[(series >= lower) & (series <= upper)]
                    return filtered, lower, upper

                def get_confidence_interval(series):
                    """取得 95% 信賴區間"""
                    if len(series) < 2:
                        return (np.nan, np.nan)
                    mean = np.mean(series)
                    std_err = stats.sem(series)
                    ci = stats.t.interval(0.95, len(series)-1, loc=mean, scale=std_err)
                    return (round(ci[0], 2), round(ci[1], 2))

                # IQR 排除極值
                if not win_df.empty:
                    win_filtered, win_low, win_high = trim_outliers(win_df["profit"])
                    avg_win_trim = win_filtered.mean()
                else:
                    avg_win_trim, win_low, win_high = avg_win, None, None

                if not lose_df.empty:
                    lose_filtered, lose_low, lose_high = trim_outliers(lose_df["profit"])
                    avg_lose_trim = lose_filtered.mean()
                else:
                    avg_lose_trim, lose_low, lose_high = avg_lose, None, None

                # 95% 信賴區間
                win_ci_low, win_ci_high = get_confidence_interval(win_df["profit"]) if not win_df.empty else (np.nan, np.nan)
                lose_ci_low, lose_ci_high = get_confidence_interval(lose_df["profit"]) if not lose_df.empty else (np.nan, np.nan)

                # 排除極值後期望報酬值
                expect_trim = win_rate * avg_win_trim + (1 - win_rate) * avg_lose_trim

                log.info("📊 信賴區間與極值分析")
                log.info("-----------------------------------------------")
                log.info(f"IQR 獲利區間: [{win_low:,.2f} ~ {win_high:,.2f}]")
                log.info(f"IQR 虧損區間: [{lose_low:,.2f} ~ {lose_high:,.2f}]")
                log.info(f"排除極值後平均獲利金額: {avg_win_trim:,.2f} (原本: {avg_win:,.2f})")
                log.info(f"排除極值後平均虧損金額: {avg_lose_trim:,.2f} (原本: {avg_lose:,.2f})")
                log.info(f"95% 獲利信賴區間: [{win_ci_low:,.2f}, {win_ci_high:,.2f}]")
                log.info(f"95% 虧損信賴區間: [{lose_ci_low:,.2f}, {lose_ci_high:,.2f}]")
                log.info(f"排除極值後期望報酬值 (EV,Trim): {expect_trim:,.2f}")
                log.info("-----------------------------------------------")

                # =====================================================
                # 輸出交易紀錄與完整統計摘要
                # =====================================================
                output_folder = config.get("leaning_folder", "./stock_data/leaning_label")
                os.makedirs(output_folder, exist_ok=True)
                trades_path = os.path.join(output_folder, f"{label}_trades.csv")
                df.to_csv(trades_path, index=False, encoding="utf-8-sig")

                summary_path = os.path.join(output_folder, f"{label}_summary.csv")
                summary_df = pd.DataFrame([{
                    "回測起始日": start_date,
                    "回測結束日": end_date,
                    "股票數量": len(collections),
                    "交易次數": buy_count,
                    "勝率(%)": round(win_rate * 100, 2),
                    "平均獲利金額": round(avg_win, 2),
                    "平均虧損金額": round(avg_lose, 2),
                    "平均獲利報酬率(%)": round(avg_win_rate, 2),
                    "平均虧損報酬率(%)": round(avg_lose_rate, 2),
                    "最大獲利": round(max_win, 2),
                    "最大虧損": round(max_lose, 2),
                    "平均持有天數": round(avg_hold_days, 2),
                    "期望報酬值(EV)": round(expect_value, 2),
                    "總獲利": round(total_profit, 2),
                    # 新增：IQR 與信賴區間分析結果
                    "IQR獲利下限": round(win_low, 2) if win_low is not None else np.nan,
                    "IQR獲利上限": round(win_high, 2) if win_high is not None else np.nan,
                    "IQR虧損下限": round(lose_low, 2) if lose_low is not None else np.nan,
                    "IQR虧損上限": round(lose_high, 2) if lose_high is not None else np.nan,
                    "排除極值後平均獲利金額": round(avg_win_trim, 2),
                    "排除極值後平均虧損金額": round(avg_lose_trim, 2),
                    "排除極值後期望報酬值(EV,Trim)": round(expect_trim, 2),
                    "獲利信賴區間下限(95%)": win_ci_low,
                    "獲利信賴區間上限(95%)": win_ci_high,
                    "虧損信賴區間下限(95%)": lose_ci_low,
                    "虧損信賴區間上限(95%)": lose_ci_high,
                }])
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

                log.info(f"✅ 已輸出交易記錄：{trades_path}")
                log.info(f"✅ 已輸出完整統計摘要（含信賴區間）：{summary_path}")

   
    close_mongo_client()



def ot_run_ma_backtest(initial_cash=100000):
    sma_labs = ["sma_20","sma_50", "sma_200"]
    folder = config.get("strategy_log_folder", "./strategy_log")
    # 整體回測起訖年
    start_year = 2010
    end_year = 2021  # 你可以依實際資料調整

    for low, high in [("sma_20", "sma_50"), ("sma_50", "sma_200")]:

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        print(f"📈 回測區間：{start_date} ～ {end_date}, 使用均線組合 {low} / {high}")
        sma_labs = [low, high]
        ot_run_ma_list(
            sma_labs,
            start_date=start_date,
            end_date=end_date,
            folder=folder,
            initial_cash=initial_cash,
        )
