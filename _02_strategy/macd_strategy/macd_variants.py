"""
MACD 變體參數化（給掃描 driver 用）
=====================================
`single_macd_strategy.py` 是教學主體，變體用**註解切換**管理——一次只開一行，
讀的人看得出「這一輪到底跑了什麼」。但要一口氣掃三十幾、上百組對照表時，
總不能改一行跑一次，所以這裡開一個參數化子類：同樣的條件式，改由類別屬性選。

**條件式一律沿用 `SingleMacdStrategy` 註解切換行的原文**（不照中文描述重寫），
改口徑時兩邊要一起改，否則掃描表與教學主體會各自走鐘。

用法：
    v = MacdVariant(); v.BASE, v.FILTER, v.EXIT, v.EXIT_MODE = "cross", "adx25", "ma200", "append"
    res = batch.run_folder(v, folder, start=..., end=..., exclude=GLITCH)

三個維度：
  BASE      三個進場基礎（cross 交叉／zero 零軸／div 背離），互斥擇一。
  FILTER    進場濾網，疊在基礎上（AND），一次一條；"none" ＝ 不加，當基準線。
  EXIT      出場規則；"native" ＝ 各基礎的自然出場，當基準線。
  EXIT_MODE 出場怎麼接：
              "append"  附加（主軸）：原出場留著、新規則疊上去，先觸發者算。
              "replace" 取代（附錄）：原出場整條拿掉，只用新規則。
            量「這條規則有沒有優化原策略」要用附加；取代量到的是「換一套技術分析」，
            是另一個問題。兩者不可混著比。
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from _02_strategy.macd_strategy.single_macd_strategy import (
    ADX_MIN, ATR_PERIOD, EXIT_ATR_STOP, EXIT_CHANDELIER, EXIT_LOWER_HIGH,
    EXIT_NONE, EXIT_TAKE_PROFIT, EXIT_TIME, EXIT_TRAIL_PCT, RSI_MAX,
    TURNOVER_MIN, VOL_MULTIPLE, SingleMacdStrategy, _scan_path_exits)

# 進場基礎：顯示名稱（表格用）
NAME_BASE = {"cross": "黃金交叉", "zero": "零軸上穿", "div": "純背離"}

# 進場濾網：顯示名稱。"none" 是基準線，務必擺在掃描清單第一個。
NAME_FILTER = {
    "none": "無濾網", "adx25": "ADX>25", "rsi": "RSI<50且上升",
    "volume": "放量1.5倍", "hist_rising": "柱狀圖連兩根遞增", "cmf": "CMF>0",
    "ma200": "收盤>MA200", "align": "均線多頭排列", "high250": "創250日新高",
    "gap": "跳空",
}

# 出場：顯示名稱。"native" 是基準線。
NAME_EXIT = {
    "native": "原生出場", "ma200": "跌破年線", "supertrend": "超級趨勢翻空",
    "psar": "SAR翻空", "donchian": "跌破二十日低", "lowerhigh": "波段高點走低",
    "beardiv": "頂背離", "chandelier": "吊燈3ATR", "trail10": "自最高點回落10%",
    "atrstop": "固定停損2ATR", "takeprofit": "固定停利+20%", "time60": "抱滿60天",
}

# 需要逐根掃描的出場（走 _scan_path_exits，不能寫成向量化條件）
_PATH_EXITS = {
    "lowerhigh": EXIT_LOWER_HIGH, "chandelier": EXIT_CHANDELIER,
    "trail10": EXIT_TRAIL_PCT, "atrstop": EXIT_ATR_STOP,
    "takeprofit": EXIT_TAKE_PROFIT, "time60": EXIT_TIME,
}


class MacdVariant(SingleMacdStrategy):
    """三維度（基礎 × 濾網 × 出場）參數化的 MACD 策略；條件式與註解切換行同一份。"""

    BASE = "cross"
    FILTER = "none"
    EXIT = "native"
    EXIT_MODE = "replace"
    LIQUIDITY = True        # 可成交門檻；2026-08-25 之後的所有輪次一律開

    # ── 進場 ────────────────────────────────────────────────
    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        macd = df["macd"]
        base = self.BASE
        if base == "cross":
            signal = df["golden"]                                 # 基礎 A 交叉
        elif base == "zero":
            signal = (macd > 0) & (macd.shift(1) <= 0)            # 基礎 B 零軸
        elif base == "div":
            signal = df["bull_divergence"]                        # 基礎 C 背離
        else:
            raise ValueError(f"未知基礎 {base}")

        f = self.FILTER
        if f == "adx25":
            signal = signal & (df["adx"] > ADX_MIN)                        # ①趨勢強度
        elif f == "rsi":
            signal = signal & (df["rsi"] < RSI_MAX) & (df["rsi"] > df["rsi"].shift(1))  # ②
        elif f == "volume":
            signal = signal & (df["volume"] > df["vol_ma20"] * VOL_MULTIPLE)  # ③放量
        elif f == "hist_rising":
            signal = signal & df["hist_rising"]                            # ④
        elif f == "cmf":
            signal = signal & (df["cmf"] > 0)                              # ⑤量能
        elif f == "ma200":
            signal = signal & (df["close"] > df["ma_long"])                # ⑥趨勢過濾
        elif f == "align":
            signal = signal & df["bull_align"]                             # ⑦均線多頭排列
        elif f == "high250":
            signal = signal & df["new_high"]                               # ⑧創250日新高
        elif f == "gap":
            signal = signal & df["gap_up"]                                 # ⑨跳空
        elif f != "none":
            raise ValueError(f"未知濾網 {f}")

        if self.LIQUIDITY:
            signal = signal & (df["turnover"] > TURNOVER_MIN)
        return signal.fillna(False).astype(bool)

    # ── 出場 ────────────────────────────────────────────────
    def _native_exit(self, df: pd.DataFrame) -> pd.Series:
        """各基礎的自然出場：交叉／背離配死叉，零軸配 DIF 下穿 0。"""
        macd = df["macd"]
        if self.BASE == "zero":
            return (macd < 0) & (macd.shift(1) >= 0)
        return df["death"]

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        native = self._native_exit(df)
        x = self.EXIT
        if x == "native":
            return native.fillna(False).astype(bool)
        if x in _PATH_EXITS:
            # 逐根掃描型：這裡只回原生出場，真正的規則在 build_signals 接
            return native.fillna(False).astype(bool)

        close, ma_long = df["close"], df["ma_long"]
        if x == "ma200":
            rule = (close < ma_long) & (close.shift(1) >= ma_long.shift(1))
        elif x == "supertrend":
            rule = self._ensure_supertrend(df)["supertrend_flip_down"]
        elif x == "psar":
            rule = self._ensure_psar(df)["psar_flip_down"]
        elif x == "donchian":
            rule = close < df["dc_low_prev"]
        elif x == "beardiv":
            rule = df["bear_divergence"]
        else:
            raise ValueError(f"未知出場 {x}")
        signal = (native | rule) if self.EXIT_MODE == "append" else rule
        return signal.fillna(False).astype(bool)

    def build_signals(self, df: pd.DataFrame):
        """
        非逐根掃描型出場 → 沿用基底（向量化、自動位移隔日成交）。
        逐根掃描型 → 自行掃描後 shift(1)（覆寫後位移責任轉移到子類，見基底 docstring）。
        """
        rule = _PATH_EXITS.get(self.EXIT, EXIT_NONE)
        if rule == EXIT_NONE:
            return super(SingleMacdStrategy, self).build_signals(df)
        if rule == EXIT_LOWER_HIGH:
            self._ensure_zigzag(df)
            turn_high = df["zigzag_turn_high"].to_numpy(dtype=np.float64)
        else:
            turn_high = np.full(len(df), np.nan)          # 其餘規則用不到，給占位陣列
        entries, exits = _scan_path_exits(
            self.buy_signal(df).to_numpy(),
            self.sell_signal(df).to_numpy(),
            df["open"].to_numpy(dtype=np.float64),
            df["high"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            df[f"atr_{ATR_PERIOD}"].to_numpy(dtype=np.float64),
            turn_high, rule, self._RULE_PARAM[rule],
            self.EXIT_MODE == "append")
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x

    def label(self) -> str:
        """表格用的組合名稱，例：黃金交叉 × ADX>25 × 跌破年線。"""
        return (f"{NAME_BASE[self.BASE]} × {NAME_FILTER[self.FILTER]}"
                f" × {NAME_EXIT[self.EXIT]}")
