"""
多股 MACD（共用資金）
======================
把 MACD 系列挑出來的五組交易策略搬到「單一本金、多檔共用現金」的組合回測：
繼承 `VbtMultiStrategy`，進場沿用單股 MACD 的基礎 ＋ 濾網，出場固定「跌破年線」。
執行時序與 _02 一致：**收盤判定 → 隔日開盤成交**。

五組交易策略（BASE, ENTRY）＝三個進場基礎 × 挑過的濾網，出場一律跌破 MA200（取代原生出場）：
  cross × align   黃金交叉 ＋ 均線多頭排列 5>20>60
  cross × adx25   黃金交叉 ＋ ADX>25
  cross × none    黃金交叉（不加濾網，當對照）
  zero  × ma200   DIF 上穿 0 ＋ 收盤>MA200
  div   × rsi     純柱狀圖背離 ＋ RSI<50 且上升

進場與出場的條件式全部逐字沿用 `_02_strategy/macd_strategy/single_macd_strategy.py`
的註解切換行（不照文字描述重寫，避免口徑飄掉）。出場是「取代」接法——原生出場
（死叉／DIF 下穿 0）整條拿掉，只留跌破年線。

買入優先序（PRIO）比單股多一種：除了低價／高價／流動性，另加 **random**。
`priority()` 是基底留的擴充點，隨機排序從這裡進來，不必動共用基底。隨機的用途是
當「亂買」基準線——重複多次取中位，用來檢驗低價優先到底有沒有優勢。

倉位 / 本金分割由 VbtMultiStrategy 的 sizing_mode / invest_ratio / min_invest 決定（見 driver）：
  固定金額投入＝sizing_mode="fixed"、min_invest=100 萬 ÷ 定額份數
  固定比例投入＝sizing_mode="percent_floor"、invest_ratio=1/比例份數、min_invest=1 萬
兩種投法的份數公式不同、份數也不同，不可共用（見 driver 的份數表）。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _01_data.indicators_momentum_volume import calculate_rsi
from _01_data.indicators_trend import calculate_adx, calculate_macd
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy

# 以下常數與 single_macd_strategy 一致（同一套口徑）
MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9
DIVERGENCE_WINDOW = 20
MA_LONG_TREND = 200
ADX_PERIOD, ADX_MIN = 14, 25
RSI_PERIOD, RSI_MAX = 14, 50
TURNOVER_MIN = 10_000_000        # 可成交門檻：5 日均量(股) × 收盤 > 1,000 萬，只 gate 進場

# 五組交易策略：(顯示名稱, 基礎, 濾網)
STRATEGIES = [
    ("交叉 × 均線多頭排列 × 跌破年線", "cross", "align"),
    ("交叉 × ADX>25 × 跌破年線", "cross", "adx25"),
    ("交叉 × 無濾網 × 跌破年線", "cross", "none"),
    ("零軸 × 收盤>MA200 × 跌破年線", "zero", "ma200"),
    ("背離 × RSI<50且上升 × 跌破年線", "div", "rsi"),
]
PRIOS = ["low_price", "high_price", "turnover", "random"]


class MultiMACD(VbtMultiStrategy):
    """多股 MACD：進場 BASE ＋ ENTRY ＋ 可成交門檻，出場跌破年線；隔日開盤成交。"""

    BASE = "cross"
    ENTRY = "none"
    PRIO = "low_price"
    SEED = 0                     # PRIO="random" 時用；同 seed 的排序可重現

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "macd" not in df.columns or "signal_line" not in df.columns:
            calculate_macd(df, short_period=MACD_SHORT, long_period=MACD_LONG,
                           signal_period=MACD_SIGNAL)
        macd, sig, close = df["macd"], df["signal_line"], df["close"]
        df["hist"] = macd - sig
        hist = df["hist"]
        df["golden"] = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        df["turnover"] = df["volume"].rolling(5).mean() * close
        df["ma_long"] = close.rolling(MA_LONG_TREND).mean()
        # 底背離：價創 20 日新低、但柱狀圖未創新低（＝價低但動能沒更低）
        df["bull_divergence"] = ((close <= close.rolling(DIVERGENCE_WINDOW).min())
                                 & (hist > hist.rolling(DIVERGENCE_WINDOW).min()))
        if self.ENTRY == "adx25" and "adx" not in df.columns:
            calculate_adx(df, period=ADX_PERIOD)
        if self.ENTRY == "rsi" and "rsi" not in df.columns:
            calculate_rsi(df, period=RSI_PERIOD)
        if self.ENTRY == "align":
            df["bull_align"] = ((close.rolling(5).mean() > close.rolling(20).mean())
                                & (close.rolling(20).mean() > close.rolling(60).mean()))
        return df

    # ── 進場 ────────────────────────────────────────────────
    def _base_signal(self, df: pd.DataFrame) -> pd.Series:
        macd = df["macd"]
        if self.BASE == "cross":
            return df["golden"]                                  # 基礎 A 交叉
        if self.BASE == "zero":
            return (macd > 0) & (macd.shift(1) <= 0)              # 基礎 B 零軸
        if self.BASE == "div":
            return df["bull_divergence"]                          # 基礎 C 純背離
        raise ValueError(f"未知基礎 {self.BASE}")

    def _entry_filter(self, df: pd.DataFrame) -> pd.Series:
        if self.ENTRY == "none":
            return pd.Series(True, index=df.index)
        if self.ENTRY == "adx25":
            return df["adx"] > ADX_MIN                            # ①趨勢強度
        if self.ENTRY == "rsi":                                   # ②RSI<50 且上升
            return (df["rsi"] < RSI_MAX) & (df["rsi"] > df["rsi"].shift(1))
        if self.ENTRY == "ma200":
            return df["close"] > df["ma_long"]                    # ⑥趨勢過濾
        if self.ENTRY == "align":
            return df["bull_align"]                               # ⑦均線多頭排列
        raise ValueError(f"未知濾網 {self.ENTRY}")

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        return (self._base_signal(df) & (df["turnover"] > TURNOVER_MIN)
                & self._entry_filter(df))

    # ── 出場：取代接法，原生出場整條拿掉，只用跌破年線 ──────────
    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        close, ma_long = df["close"], df["ma_long"]
        return (close < ma_long) & (close.shift(1) >= ma_long.shift(1))

    def build_signals(self, df: pd.DataFrame):
        """
        收盤判定 → 隔日成交：位移 +1（無 look-ahead）。

        **同一根同時有買訊與賣訊 → 兩邊都忽略**（系列共用口徑，見第十篇）。
        純背離的「價破底」與跌破年線常常同根成立；不處理的話多股引擎會在那一根
        「賣掉舊的、再買新的」，比單股多算交易（實測背離母體 632 → 651 筆）。
        單股引擎走 vbt from_signals，同根衝突本來就兩邊不動作，這裡補齊對齊。
        """
        entries = self.buy_signal(df).fillna(False).astype(bool)
        exits = self.sell_signal(df).fillna(False).astype(bool)
        both = entries & exits
        entries, exits = entries & ~both, exits & ~both
        return (entries.shift(1, fill_value=False),
                exits.shift(1, fill_value=False))

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["open"]                                        # 隔日開盤價成交

    def priority(self, df: pd.DataFrame, stock_id: str) -> pd.Series:
        if self.PRIO == "turnover":
            return self.prio_by_turnover(df)
        if self.PRIO == "high_price":
            return self.prio_by_high_price(df)
        if self.PRIO == "random":
            # 每檔一條獨立亂數序列；用 (SEED, 股票代號) 當種子，換 seed 就換一種買入順序
            rng = np.random.default_rng(
                abs(hash((self.SEED, stock_id))) % (2 ** 32))
            return pd.Series(rng.random(len(df)), index=df.index)
        return self.prio_by_low_price(df)                         # 預設低價優先
