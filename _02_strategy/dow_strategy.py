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
    ext_hi = highs[0]; ext_hi_i = 0
    ext_lo = lows[0];  ext_lo_i = 0
    last_turn_high = np.nan
    last_turn_low = np.nan

    for i in range(1, n):
        h, l, c = highs[i], lows[i], closes[i]

        # 更新當前極值
        if direction >= 0 and h >= ext_hi:
            ext_hi = h; ext_hi_i = i
        if direction <= 0 and l <= ext_lo:
            ext_lo = l; ext_lo_i = i

        made_pivot = False

        # 上漲段：檢查是否回撤超過門檻（出現高點轉折）
        if direction >= 0:
            retrace = (ext_hi - l) / ext_hi
            if retrace >= pct:
                last_turn_high = ext_hi
                turn_highs[ext_hi_i] = ext_hi
                direction = -1
                ext_lo = l; ext_lo_i = i
                made_pivot = True

        # 下跌段：檢查是否反彈超過門檻（出現低點轉折）
        if not made_pivot and direction <= 0:
            bounce = (h - ext_lo) / ext_lo
            if bounce >= pct:
                last_turn_low = ext_lo
                turn_lows[ext_lo_i] = ext_lo
                direction = +1
                ext_hi = h; ext_hi_i = i
                made_pivot = True

        # 特殊情況：同一根K棒同時創更高與更低
        if highs[i] > ext_hi and lows[i] < ext_lo:
            prev_close = closes[i - 1]
            if c > prev_close:
                ext_hi = h; ext_hi_i = i
                direction = +1
            elif c < prev_close:
                ext_lo = l; ext_lo_i = i
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

def compute_swings_zigzag_close_inline(df, hi_pct=0.02,low_pct=None):
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

    if not low_pct :
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
    ext_hi = closes[0]; ext_hi_i = 0
    ext_lo = closes[0];  ext_lo_i = 0
    last_turn_high = np.nan
    last_turn_low = np.nan

    for i in range(1, n):
        h, l, c = closes[i], closes[i], closes[i]

        # 更新當前極值
        if direction >= 0 and h >= ext_hi:
            ext_hi = h; ext_hi_i = i
        if direction <= 0 and l <= ext_lo:
            ext_lo = l; ext_lo_i = i

        made_pivot = False

        # 上漲段：檢查是否回撤超過門檻（出現高點轉折）
        if direction >= 0:
            retrace = (ext_hi - l) / ext_hi
            if retrace >= hi_pct:
                last_turn_high = ext_hi
                turn_highs[ext_hi_i] = ext_hi
                direction = -1
                ext_lo = l; ext_lo_i = i
                made_pivot = True

        # 下跌段：檢查是否反彈超過門檻（出現低點轉折）
        if not made_pivot and direction <= 0:
            bounce = (h - ext_lo) / ext_lo
            if bounce >= low_pct:
                last_turn_low = ext_lo
                turn_lows[ext_lo_i] = ext_lo
                direction = +1
                ext_hi = h; ext_hi_i = i
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

def compute_swings_zigzag_close_inline_multip(df, hi_pct=0.02, low_pct=None, lookahead=2):
    import numpy as np
    if not low_pct:
        low_pct = hi_pct

    n = len(df)
    if n == 0:
        df["turn_high"] = np.nan
        df["turn_low"] = np.nan
        return df

    closes = df["close"].to_numpy()
    turn_highs = np.full(n, np.nan)
    turn_lows = np.full(n, np.nan)
    direction = 0
    ext_hi = closes[0]; ext_hi_i = 0
    ext_lo = closes[0]; ext_lo_i = 0
    last_turn_high = np.nan
    last_turn_low = np.nan

    for i in range(1, n):
        # 🔹 回看 lookahead 範圍的高低價
        look_hi = max(closes[max(0, i - lookahead):i + 1])
        look_lo = min(closes[max(0, i - lookahead):i + 1])

        if direction >= 0 and look_hi >= ext_hi:
            ext_hi = look_hi; ext_hi_i = i
        if direction <= 0 and look_lo <= ext_lo:
            ext_lo = look_lo; ext_lo_i = i

        made_pivot = False

        # 上漲段：檢查 retrace
        if direction >= 0:
            retrace = (ext_hi - closes[i]) / ext_hi
            if retrace >= hi_pct:
                last_turn_high = ext_hi
                turn_highs[ext_hi_i] = ext_hi
                direction = -1
                ext_lo = closes[i]; ext_lo_i = i
                made_pivot = True

        # 下跌段：檢查反彈
        if not made_pivot and direction <= 0:
            bounce = (closes[i] - ext_lo) / ext_lo
            if bounce >= low_pct:
                last_turn_low = ext_lo
                turn_lows[ext_lo_i] = ext_lo
                direction = +1
                ext_hi = closes[i]; ext_hi_i = i

        # 即時更新轉折
        if not np.isnan(last_turn_high):
            turn_highs[i] = last_turn_high
        if not np.isnan(last_turn_low):
            turn_lows[i] = last_turn_low

    df["turn_high"] = np.round(turn_highs, 2)
    df["turn_low"] = np.round(turn_lows, 2)
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
    ):
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)
        # ✅ 在初始化時計算 ZigZag 轉折
        # self.data = compute_swings_zigzag_inline(self.data, pct=0.06)
        self.data = compute_swings_zigzag_close_inline(self.data, hi_pct=0.04)
        # self.data = compute_swings_zigzag_close_inline_multip(self.data, hi_pct=0.06,lookahead=1)
        self.buy_index = 0

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
        turn_high = self.data["turn_high"].iloc[i - 1]
        turn_low = self.data["turn_low"].iloc[i - 1]
        open_price = self.data["open"].iloc[i]

        # 計算 5 日平均成交量
        volume_ma5 = self.data["volume"].iloc[i - 6:i-1].mean()

        if not np.isnan(turn_high) and close > turn_high and open_price > turn_high:
            obv = self.compute_obv(i - 1)
                # 設定停損價為最近低點（下修1%防止被掃）
            if obv and (45<= obv<=70 ) and volume_ma5*1.5 < self.data["volume"].iloc[i-1] and self.data["volume"].iloc[i-1] > 1000_000:
                self.stop_value = turn_low 
                self.trade_value = (open_price-turn_low)*4 + open_price
                self.buy_index = i
                return True

        return False

    # --- 賣出條件 ---
    def sell_signal(self, i):
        if i <= 20:
            return False
        if self.stop_value and self.data["close"].iloc[i - 1] < self.stop_value:
            self.sell_index = i
            return True # 停損

        # if self.trade_value and self.data["close"].iloc[i - 1] > self.trade_value:
        #     return True # 停利
        close = self.data["close"].iloc[i - 1]
        turn_low = self.data["turn_low"].iloc[i - 1]
        if not np.isnan(turn_low) and close < turn_low:
            self.sell_index = i
            return True
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])
    
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


def run_dow_list(start_date="2015-01-01", end_date="2019-12-31", folder="", initial_cash=100000):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder = folder
    collections.sort()

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []
    label = f"dow_04_strategy_stopValue_OBV_volumn_1000"
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    for stock_id in collections:
        try:
            backtest = DowStrategy(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                split_cash=5000,
                label=label,
            )
            backtest.run_backtest()
        except Exception as e:
            print(f"⚠️ 忽略錯誤，錯誤原因：{e}")
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

    analyze_hold_days(hold_days, log)
    close_mongo_client()



def run_dow_backtest(initial_cash=100000):
    folder = config.get("strategy_log_folder", "./strategy_log")
    # 整體回測起訖年
    start_year = 2010
    end_year = 2021  # 你可以依實際資料調整
    # start_year = 2021
    # end_year = 2025  # 你可以依實際資料調整

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    run_dow_list(
        start_date=start_date,
        end_date=end_date,
        folder=folder,
        initial_cash=initial_cash,
    )
