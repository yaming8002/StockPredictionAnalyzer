"""
多股 KD 交叉（共用資金）
========================
把 KD 系列「PF 前 6 進場 × 高檔死叉出場」搬到「單一本金、多檔共用現金」的組合回測：
繼承 `VbtMultiStrategy`，進場沿用單股 KD 的濾網、出場固定高檔死叉（K、D 都 > 80 的死叉）。
執行時序與 _02 一致：**收盤判定 → 隔日開盤成交**。

進場（self.ENTRY 擇一，皆＝黃金交叉 & 可成交門檻 & 該濾網）：
  breakout250 / breakout120 / breakout60（收盤創近 N 日新高）
  gap（今開 > 昨高）
  low_redk（K、D<20 的黃金交叉且當日收紅）
  divergence（底背離：價創 20 日新低但 K 未創）
出場：高檔死叉（死叉且 K、D 都 > 80）。

倉位 / 本金分割由 VbtMultiStrategy 的 sizing_mode / invest_ratio / min_invest 決定（見 driver）。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_momentum_volume import calculate_kd
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy

KD_N, KD_K, KD_D = 9, 3, 3
OVERSOLD, OVERBOUGHT = 20, 80
TURNOVER_MIN = 10_000_000        # 可成交門檻：5 日均量(股) × 收盤 > 1,000 萬

# 6 個進場（＋錨點），皆配高檔死叉出場
ENTRIES = ["breakout250", "breakout120", "gap", "low_redk", "breakout60", "divergence"]
# 錨點進場（2026-07-28 reference 另外納入矩陣／MC 的兩個系列策略）：不在文章多股表裡，
# 只供需要時比照多股；golden＝純黃金交叉（opt6）、low_zone＝低檔 K,D<20（opt1）
ANCHORS = ["golden", "low_zone"]


class MultiKD(VbtMultiStrategy):
    """多股 KD：進場 self.ENTRY（＋黃金交叉＋可成交門檻），出場高檔死叉；隔日開盤成交。"""

    ENTRY = "breakout250"
    PRIO = "low_price"          # 買入優先序：low_price(低價) / turnover(流動性) / high_price(高價)

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "k" not in df.columns or "d" not in df.columns:
            calculate_kd(df, n=KD_N, k_smooth=KD_K, d_smooth=KD_D)
        k, d = df["k"], df["d"]
        df["golden"] = (k > d) & (k.shift(1) <= d.shift(1))
        df["death"] = (k < d) & (k.shift(1) >= d.shift(1))
        df["high_zone"] = (k > OVERBOUGHT) & (d > OVERBOUGHT)     # 高檔死叉出場用
        df["turnover"] = df["volume"].rolling(5).mean() * df["close"]
        return df

    def _one_filter(self, df: pd.DataFrame, e: str) -> pd.Series:
        k, d, c, o, h = df["k"], df["d"], df["close"], df["open"], df["high"]
        if e == "breakout250":
            return c >= c.rolling(250).max()
        if e == "breakout120":
            return c >= c.rolling(120).max()
        if e == "breakout60":
            return c >= c.rolling(60).max()
        if e == "gap":
            return o > h.shift(1)
        if e == "low_redk":
            return (k < OVERSOLD) & (d < OVERSOLD) & (c >= o)
        if e == "divergence":
            return (c <= c.rolling(20).min()) & (k > k.rolling(20).min())
        if e == "low_zone":
            return (k < OVERSOLD) & (d < OVERSOLD)
        if e == "golden":
            return pd.Series(True, index=df.index)               # 不加濾網＝純黃金交叉
        raise ValueError(f"未知進場 {e}")

    def _entry_filter(self, df: pd.DataFrame) -> pd.Series:
        return self._one_filter(df, self.ENTRY)

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["golden"] & (df["turnover"] > TURNOVER_MIN) & self._entry_filter(df)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        return df["death"] & df["high_zone"]                     # 高檔死叉

    def build_signals(self, df: pd.DataFrame):
        # 收盤判定 → 隔日成交：位移 +1（無 look-ahead）
        entries = self.entry_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        exits = self.sell_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        return entries, exits

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["open"]                                        # 隔日開盤價成交

    def priority(self, df: pd.DataFrame, stock_id: str) -> pd.Series:
        # 買入優先序（擋單多時決定先買誰）；三選一：低價 / 流動性 / 高價
        if self.PRIO == "turnover":
            return self.prio_by_turnover(df)
        if self.PRIO == "high_price":
            return self.prio_by_high_price(df)
        return self.prio_by_low_price(df)                        # 預設低價優先
