from datetime import timedelta
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


# --- 1) 以百分比 ZigZag 偵測轉折 ---
def compute_swings_zigzag(df, pct=0.02):
    """
    改良版 ZigZag：
    - 追蹤 high/low 極值
    - 若同一根K棒同時創更高與更低，則依據 close 相對前一收盤決定方向
    """
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    idxs  = df.index

    n = len(df)
    if n == 0:
        return []

    pivots = []
    direction = 0  # 0 未定, +1 上漲, -1 下跌
    ext_hi = highs[0]; ext_hi_i = 0
    ext_lo = lows[0];  ext_lo_i = 0

    for i in range(1, n):
        h, l, c = highs[i], lows[i], closes[i]

        # 1️⃣ 更新當前趨勢極值
        if direction >= 0:
            if h >= ext_hi:
                ext_hi = h; ext_hi_i = i
        if direction <= 0:
            if l <= ext_lo:
                ext_lo = l; ext_lo_i = i

        made_pivot = False

        # 2️⃣ 判斷是否反轉（優先依據趨勢 + 回撤閾值）
        if direction >= 0:  # 上漲段檢查是否回撤超過門檻
            retrace = (ext_hi - l) / ext_hi
            if retrace >= pct:
                pivots.append({
                    "idx": int(ext_hi_i),
                    "date": idxs[ext_hi_i],
                    "price": float(ext_hi),
                    "type": "H"
                })
                direction = -1
                ext_lo = l; ext_lo_i = i
                made_pivot = True

        if not made_pivot and direction <= 0:  # 下跌段檢查是否反彈超過門檻
            bounce = (h - ext_lo) / ext_lo
            if bounce >= pct:
                pivots.append({
                    "idx": int(ext_lo_i),
                    "date": idxs[ext_lo_i],
                    "price": float(ext_lo),
                    "type": "L"
                })
                direction = +1
                ext_hi = h; ext_hi_i = i
                made_pivot = True

        # 3️⃣ 特殊情況：同一根K棒創更高與更低
        # 用收盤價相對前一收盤決定方向
        if highs[i] > ext_hi and lows[i] < ext_lo:
            prev_close = closes[i - 1]
            if c > prev_close:
                # 收盤比前一高 → 判定上漲，更新高點
                ext_hi = h; ext_hi_i = i
                direction = +1
            elif c < prev_close:
                # 收盤比前一低 → 判定下跌，更新低點
                ext_lo = l; ext_lo_i = i
                direction = -1
            # 若收盤等於前一收盤 → 視為盤整，不改變方向

    return pivots



# --- 2) 標註 HH/HL/LH/LL 與主要趨勢 ---
def label_dow_structure(df, pivots):
    """
    依 pivots 標註每個 pivot 的結構與「目前主要趨勢」。
    回傳 DataFrame: columns=['idx','price','type','label','primary_trend']
    label: 'HH','HL','LH','LL' 之一
    primary_trend: 'UP','DOWN','UNKNOWN'
    """
    rows = []
    last_H = None
    last_L = None
    primary = "UNKNOWN"

    for p in pivots:
        t = p["type"]
        price = p["price"]
        label = None

        if t == "H":
            if last_H is None:
                label = "HH"  # 第一個高點先暫記為 HH
            else:
                label = "HH" if price > last_H else "LH"
            last_H = price
        else:  # 'L'
            if last_L is None:
                label = "HL"  # 第一個低點先暫記為 HL
            else:
                label = "HL" if price > last_L else "LL"
            last_L = price

        # 依序列判斷主要趨勢
        # 規則（簡化）：若最近一次高/低標籤組合出現 HH + HL → UP；LL + LH → DOWN
        if t == "H" and label == "HH" and last_L is not None:
            primary = "UP"
        elif t == "L" and label == "LL" and last_H is not None:
            primary = "DOWN"

        rows.append({"idx": p["idx"], "price": price, "type": t, "label": label, "primary_trend": primary})

    return pd.DataFrame(rows)

# --- 3) 依道氏規則產生訊號 ---
def dow_signals_from_swings(df, swings_df, confirm_with_volume=False):
    """
    當收盤「有效突破」上一個 Swing High（且 primary_trend 為 UP）→ buy
    當收盤「有效跌破」上一個 Swing Low（且 primary_trend 為 DOWN）→ sell
    回傳與 df 同長度的 signal 序列：'BUY','SELL',None
    """
    signals = [None] * len(df)
    last_SH = None  # 上一個有效 Swing High 價
    last_SL = None  # 上一個有效 Swing Low 價
    trend = "UNKNOWN"

    # 建立方便索引的 dict
    swing_by_idx = {int(r.idx): r for r in swings_df.itertuples()}

    for i in range(len(df)):
        if i in swing_by_idx:
            r = swing_by_idx[i]
            trend = r.primary_trend
            if r.type == 'H':
                last_SH = r.price
            else:
                last_SL = r.price

        close_i = df["close"].iloc[i]
        # 突破/跌破判斷可加入日內濾網：如連續2根收盤站上/跌破，或超過 0.5% 緩衝
        if trend == "UP" and last_SH is not None and close_i > last_SH:
            # 可選：量能確認（OBV/量均）
            if confirm_with_volume:
                if "OBV" in df.columns:
                    if df["OBV"].iloc[i] <= df["OBV"].iloc[i-1]:
                        continue
            signals[i] = "BUY"

        if trend == "DOWN" and last_SL is not None and close_i < last_SL:
            if confirm_with_volume:
                if "OBV" in df.columns:
                    if df["OBV"].iloc[i] >= df["OBV"].iloc[i-1]:
                        continue
            signals[i] = "SELL"

    return signals



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
        super().__init__(stock_id, start_date, end_date, initial_cash, split_cash, label, loglevel)  # 繼承父類初始化
        pivots = compute_swings_zigzag(self.data, pct=0.06)
        swings_df = label_dow_structure(self.data, pivots)
        # 2) 產生道氏訊號（可選擇是否加入量能確認）
        self.dow_signals = dow_signals_from_swings(self.data, swings_df, confirm_with_volume=False)

    def ervey_date_work(self):
        pass

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
        return chop_value < 25

    def buy_signal(self, i):
        # 前一日判斷為 BUY，今日開盤才買入
        if i > 21 and self.dow_signals[i - 1] == "BUY":
            obv = self.compute_obv(i-1)
            if obv and (40 <= obv <= 70):
                return True
        return False

    def sell_signal(self, i):
        if i > 21 :
            return self.dow_signals[i - 1] == "SELL"
     
        return False

    def buy_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])

    def sell_price_select(self, i):
        return self.tw_ticket_gap(self.data.iloc[i]["open"])


def run_dow_list( start_date="2015-01-01", end_date="2019-12-31", folder="" , initial_cash=100000):
    db = get_mongo_client()
    collections = [col for col in db.list_collection_names() if "TW" in col]
    strategy_log_folder =folder 
    collections.sort()

    total_win = 0
    total_lose = 0
    total_profit = 0.0
    hold_days = []
    trade_records = []  # opt_sma_120_sma_200_3_sell_3_AVG_trades
    label = f"dow_06_strategy_OBV"
    log_file_path = f"{strategy_log_folder}/{start_date}_to_{end_date}-{label}.log"
    log = setup_logger(log_file=log_file_path, loglevel=logging.INFO)

    for stock_id in collections:
        # try:
        backtest = DowStrategy(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            split_cash = 5000,
            label=label,
        )
        backtest.run_backtest()
    
        # except Exception as e:
        #     print(f"⚠️ 忽略錯誤，錯誤原因：{e}")
        #     continue
        
        buy_count = backtest.win_count + backtest.lose_count
        profit = backtest.cash - initial_cash
        log.info(
            f"{stock_id}: 初始金額:{initial_cash} ,最終金額:{backtest.cash} ,下注量:{buy_count} ,獲利:{math.floor(profit)}, 勝率:{backtest.win_rate:.2%}"
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



def run_dow_backtest(initial_cash=100000):
    folder = config.get("strategy_log_folder", "./strategy_log")
    # 整體回測起訖年
    start_year = 2010
    end_year = 2021  # 你可以依實際資料調整


    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    run_dow_list(
        start_date=start_date,
        end_date=end_date,
        folder=folder,
        initial_cash=initial_cash,
    )
