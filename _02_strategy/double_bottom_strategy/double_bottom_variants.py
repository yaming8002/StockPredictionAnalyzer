"""
雙底進場優化的變體（掃描用）
============================
策略檔 `single_double_bottom_strategy.py` 用「註解切換」管理優化（buy_signal 內 # 優化 #2～#7）；
全組合掃描不能去改策略檔的註解狀態，所以這裡把每一行條件式**逐字照抄**成一個函式，
子類在 buy_signal 只把「被選中的那一行」疊在基本版上（比照 kd_variants／macd_variants 的做法）。
改條件時兩邊要一起改。

變體名：base（基本版，不加任何條件）／neck／gap／rise／ma200／align／volume／drop／height／volshrink／bullbar
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import (
    GAP_MIN, NECK_HEIGHT_MIN, PRIOR_DROP_MIN, RISE_MIN, TURNOVER_MIN, VOL_MULTIPLE,
    SingleDoubleBottomStrategy, neckline_breakout)

# 名稱 → (中文標籤, 條件)；條件收 (df, signal) 回傳新的 signal，內容與策略檔註解行逐字相同
ENTRY_FILTERS = {
    "base": ("基本版", lambda df, signal: signal),
    "neck": ("頸線突破", lambda df, signal: pd.Series(neckline_breakout(df["close"].to_numpy(dtype=float), signal.to_numpy(), df["zigzag_turn_low"].to_numpy(), df["db_neck"].to_numpy()), index=df.index)),  # 優化 #2
    "gap": ("兩底間隔 ≥ 10 天", lambda df, signal: signal & (df["db_gap"] >= GAP_MIN)),                       # 優化 #3
    "rise": ("第二底墊高 ≥ 2%", lambda df, signal: signal & (df["zigzag_turn_low"] >= df["db_l1"] * RISE_MIN)),  # 優化 #4
    "ma200": ("站上年線", lambda df, signal: signal & (df["close"] > df["ma200"])),                            # 優化 #5
    "align": ("長線多頭排列", lambda df, signal: signal & df["bull_align"]),                                   # 優化 #6
    "volume": ("放量 1.5 倍", lambda df, signal: signal & (df["volume"] > df["vol_ma20"] * VOL_MULTIPLE)),     # 優化 #7
    "drop": ("前段跌幅 ≥ 15%", lambda df, signal: signal & (df["db_drop1"] >= PRIOR_DROP_MIN)),               # 優化 #8
    "height": ("型態高度 ≥ 10%", lambda df, signal: signal & (df["db_neck"] >= df["zigzag_turn_low"] * NECK_HEIGHT_MIN)),  # 優化 #9
    "volshrink": ("第二底量縮", lambda df, signal: signal & (df["db_vol2"] < df["db_vol1"])),                  # 優化 #10
    "bullbar": ("確認日紅 K／長下影線", lambda df, signal: signal & df["db_bull_bar"]),                        # 優化 #11
}


class DoubleBottomEntryVariant(SingleDoubleBottomStrategy):
    """基本版進場＋ENTRY 指定的那一條（＋可成交門檻），出場不變。"""

    ENTRY = "base"

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = ENTRY_FILTERS[self.ENTRY][1](df, self.base_signal(df))
        if self.liquidity:
            signal = signal & (df["turnover"] > TURNOVER_MIN)
        return signal.fillna(False).astype(bool)


def make_variant(entry: str, pct: float, liquidity: bool) -> DoubleBottomEntryVariant:
    if entry not in ENTRY_FILTERS:
        raise ValueError(f"未知的進場變體 {entry}，可用：{list(ENTRY_FILTERS)}")
    v = DoubleBottomEntryVariant(pct=pct, liquidity=liquidity)
    v.ENTRY = entry
    return v
