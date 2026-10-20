"""
多股海龜（共用資金）
====================
把海龜系統 1／系統 2 的純通道版搬到「單一本金、多檔共用現金」的組合回測：
繼承 `VbtMultiStrategy`，進出場沿用 `_02_strategy/turtle_strategy/single_turtle_strategy.py`
的 baseline（通道天數取同一份 SYSTEMS 對照表），執行時序一致：**收盤判定 → 隔日開盤成交**。

與單股 baseline 的差異只有一條：進場另加可成交門檻（成交金額 5日均量×股價 > 1,000 萬），
與 KD／MACD 多股同口徑——共用資金下真的要下單，低流動性小股買不進來。

本檔不做原版的 N 定倉與 ½N 加碼（每檔同時只抱一個部位），倉位由
VbtMultiStrategy 的 sizing_mode / invest_ratio / min_invest 決定（見 driver 的份數表）。

同根衝突（同一根同時有買訊與賣訊）在海龜不會發生：收盤 > 前 N 日高與收盤 < 前 M 日低
不可能同時成立，故不比照 MACD 多股另做互斥處理。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_volatility import calculate_donchian
from _02_strategy.turtle_strategy.single_turtle_strategy import SYSTEMS, TURNOVER_MIN
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy


class MultiTurtle(VbtMultiStrategy):
    """多股海龜：突破前 N 日高（＋可成交門檻）進、跌破前 M 日低出；隔日開盤成交、低價優先。"""

    SYSTEM = 1

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        entry_window, exit_window = SYSTEMS[self.SYSTEM]
        calculate_donchian(df, window=entry_window)
        calculate_donchian(df, window=exit_window)
        # 前 N 日極值不含當日（同單股 baseline）
        df["entry_high"] = df[f"donchian_upper_{entry_window}"].shift(1)
        df["exit_low"] = df[f"donchian_lower_{exit_window}"].shift(1)
        df["turnover"] = df["volume"].rolling(5).mean() * df["close"]
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] > df["entry_high"]) & (df["turnover"] > TURNOVER_MIN)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df["exit_low"]

    def build_signals(self, df: pd.DataFrame):
        """收盤判定 → 隔日成交：位移 +1（無 look-ahead）。進場取 entry_signal（區間前的判定日不算，與單股同一閘門）。"""
        entries = self.entry_signal(df).fillna(False).astype(bool)
        exits = self.sell_signal(df).fillna(False).astype(bool)
        return (entries.shift(1, fill_value=False),
                exits.shift(1, fill_value=False))

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["open"]                                        # 隔日開盤價成交

    def priority(self, df: pd.DataFrame, stock_id: str) -> pd.Series:
        return self.prio_by_low_price(df)                        # 低價優先（系列慣例）
