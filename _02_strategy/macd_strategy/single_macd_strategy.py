"""
MACD 策略（single MACD）— macd_strategy 套件下的方案
========================================================

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）。
成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。
MACD 參數固定傳統值 12/26/9（引用 _01_data.calculate_macd，不在策略內自算）。
產出 macd（DIF 快線＝短 EMA − 長 EMA）與 signal_line（訊號線）；柱狀圖 hist = macd − signal_line 於策略內算。

【三個「基礎方案」— 本檔的骨架】
MACD 有三種「本質不同」的進場邏輯，各自獨立初探（每一種各有一篇文章，見檔頭的系列連結）：
  基礎 A 交叉（cross）：黃金交叉進場 / 死叉出場。最基礎的教學交叉。
  基礎 B 零軸（zero） ：DIF 上穿 0 進場 / 下穿 0 出場。穿越多空分界本身，不看訊號線。
  基礎 C 背離（div）  ：純柱狀圖背離進場（價創 20 日新低但 hist 未創新低），不靠交叉；死叉出場。
趨勢過濾 / 零軸位置 / 量能等「濾網」不是新的進場邏輯，而是「優化」——同一個濾網套在
不同基礎上結果不同，故優化按基礎分別產出（cross_* / zero_* / div_* 各自的優化），不混為一談。

【背景】網路主流規則調研：大規模實證（EdgeTools 1,430 萬次測試）顯示 MACD「交叉類」
（線交叉／零軸交叉／柱狀圖方向…）皆無統計 edge，唯一勉強有邊際的是「背離」。
本檔就是拿這個說法到台股全史上一項一項驗——**實測結果一律寫在文章裡，不寫在程式碼註解**。

以「註解切換」管理（比照 single_kd）：buy_signal / sell_signal 內三基礎行互斥擇一、優化行疊在其上，
一次只開一條同類行；--variant 只決定輸出資料夾、須與註解狀態一致。

【三處註解切換 — 動手前先看這裡】
  1. buy_signal   ：三基礎互斥擇一 ＋ 進場濾網（一次一條，疊在基礎上）。
  2. sell_signal  ：出場，一次一條（原生出場；新規則可「附加」疊上去，或「取代」整條換掉）。
  3. EXIT_RULE    ：類別屬性，路徑相依出場（頂頂低 ＋ 五條風控），一次一條；
     每條算疊加還是取代，由 _REPLACE_RULES 切換（預設：頂頂低取代、五條風控疊加）。
     這一區之所以不放在 sell_signal，是因為它們要看「進場價 / 進場以來最高 / 已持有幾根」，
     必須逐根掃描才算得出來——沿用 base/vbt/single.build_signals docstring 的責任轉移約定
     （覆寫後由子類自行位移成隔日成交），作法比照 ma_cross_strategy 的 CHOCH。

【對應文章】MACD 系列：https://stockanalyzer.sailforthlab.dev/archives/?subcategory=MACD
  每個變體實際跑出什麼（交易次數、勝率、持有天數、獲利因子、總獲利與完整口徑）都在文章裡；
  系列頁會隨新文章自動更新。本檔只放「怎麼算」，不放結論。

⚠️ 口徑提醒：有沒有開流動性門檻（見 buy_signal 最後一行）會讓數字差很多，
   比較不同變體時務必確認兩邊的門檻設定一致。

執行（全市場，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
    python _02_strategy/macd_strategy/single_macd_strategy.py <資料夾> --variant <名稱>
  （--variant 只決定輸出資料夾 result/single_macd/<variant>/；行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from numba import njit

from _01_data.indicators_trend import calculate_macd, calculate_adx, calculate_psar
from _01_data.indicators_momentum_volume import calculate_cmf, calculate_rsi
from _01_data.indicators_volatility import calculate_atr, calculate_donchian, calculate_supertrend
from _01_data.indicators_pattern import calculate_zigzag
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
# 資料品質排除集 GLITCH 與標準回測區間 DEFAULT_START/END：跨策略共用，統一由 base/vbt/common 取用（單一定義）。
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END


# 回測結果輸出目錄（策略同目錄底下 ./result，已於 .gitignore 排除）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# MACD 傳統參數（固定，不掃期數）
MACD_SHORT = 12     # 短 EMA
MACD_LONG = 26      # 長 EMA
MACD_SIGNAL = 9     # 訊號線 EMA

# 流動性門檻（優化·流動性）— 成交金額 = 5日均量(股) × 股價 > 1,000 萬。
# 用個股實際股價換算真實可入場金額，高價低量股不誤殺（對齊 single_kd、跨策略共用口徑）。
TURNOVER_MIN = 10_000_000

# CMF 視窗（優化·量能用）：Chaikin Money Flow 回看天數
CMF_WINDOW = 20
# 背離視窗（基礎 C 背離用）：價創新低但 hist 未創新低的回看天數；頂背離出場共用同一視窗
DIVERGENCE_WINDOW = 20
# 趨勢過濾均線（優化·趨勢過濾進場 / 趨勢跌破出場用）：MA200 年線＝長期趨勢方向
MA_LONG_TREND = 200

# ── 進場濾網參數 ──
ADX_PERIOD = 14                     # ADX 期數；門檻 25＝最常被引用的「有趨勢」分界
ADX_MIN = 25
RSI_PERIOD = 14
RSI_MAX = 50                        # RSI<50 且上升＝動能剛翻正、還沒過熱
VOL_WINDOW, VOL_MULTIPLE = 20, 1.5  # 放量：量 > 20 日均量 × 1.5
HIGH_WINDOW = 250                   # 創新高回看天數（沿用 KD 系列表現最好的 250 日）

# ── 出場參數 ──
ATR_PERIOD = 14
DONCHIAN_WINDOW = 20                # 跌破前 20 日最低＝海龜式出場
ZIGZAG_PCT = 0.02                   # 頂頂低用；沿用 ma_cross CHoCH 已驗門檻
SUPERTREND_PERIOD, SUPERTREND_MULT = 10, 3.0    # 各圖表軟體出廠值
SAR_AF_START, SAR_AF_STEP, SAR_AF_MAX = 0.02, 0.02, 0.2   # Wilder 原著參數
CHANDELIER_K = 3.0                  # 吊燈：進場後最高 − 3×ATR
TRAIL_PCT = 0.10                    # 移動停損：自最高點回落 10%
ATR_STOP_K = 2.0                    # 固定停損：進場價 − 2×ATR
TAKE_PROFIT = 0.20                  # 固定停利 +20%
TIME_BARS = 60                      # 時間出場：抱滿 60 根 K

# ── 路徑相依出場的代碼（給 EXIT_RULE 用）──
EXIT_NONE = 0
EXIT_LOWER_HIGH = 1                 # 頂頂低（進場以來出現較低的 ZigZag 擺動高點）
EXIT_CHANDELIER = 2
EXIT_TRAIL_PCT = 3
EXIT_ATR_STOP = 4
EXIT_TAKE_PROFIT = 5
EXIT_TIME = 6

# 每條規則是「疊加」還是「取代」由 _REPLACE_RULES 決定：列在裡面的走取代，其餘走疊加。
# 出場優化以「疊加（附加）」為主軸——原出場留著、新規則疊上去、先觸發者算。
# 這樣量到的才是「原策略有沒有被這條規則優化」；整條換掉量到的是「換一套技術分析」，
# 是另一個問題，故只作附錄。（進場優化同理：原訊號 AND 濾網，也是附加。）
_REPLACE_RULES = (EXIT_LOWER_HIGH,)
# 附錄用：把五條風控也當「唯一出場」（純取代）。開下面兩行、關上面那行；
# 一次仍只跑一條 EXIT_RULE，列進來的只是把該條的疊加改成取代。
# _REPLACE_RULES = (EXIT_LOWER_HIGH, EXIT_CHANDELIER, EXIT_TRAIL_PCT,
#                   EXIT_ATR_STOP, EXIT_TAKE_PROFIT, EXIT_TIME)


@njit(cache=True)
def _scan_path_exits(entry_raw, native_exit, open_, high_, close_, atr, turn_high,
                     rule, param, use_native):
    """
    路徑相依出場的逐根掃描，回傳「判定日」的 (entries, exits)。

    成交時序：第 i 根收盤判定進場 → 第 i+1 根開盤成交，故出場最早只能在第 i+1 根收盤判定。
    進場價取 open_[i+1]（＝實際成交價），且只在第 i+1 根以後才用到，故無 look-ahead。
    use_native=True → 疊加（原出場與本規則先觸發者算）；False → 取代（只用本規則）。
    """
    n = len(close_)
    entries = np.zeros(n, np.bool_)
    exits = np.zeros(n, np.bool_)
    in_pos = False
    entry_i = -1
    entry_px = 0.0
    atr_at_entry = 0.0
    peak_px = -1e18          # 進場以來最高價（吊燈 / 移動停損用）
    peak_th = -1e18          # 進場以來最高的擺動高點（頂頂低用）
    for i in range(n):
        if not in_pos:
            if entry_raw[i] and i + 1 < n:
                in_pos = True
                entries[i] = True
                entry_i = i
                entry_px = open_[i + 1]
                atr_at_entry = atr[i]
                peak_px = -1e18
                peak_th = -1e18
            continue
        # ── 持倉中：在第 i 根收盤判定是否出場 ──
        if high_[i] > peak_px:
            peak_px = high_[i]
        fired = False
        if rule == 1:                                    # 頂頂低
            t = turn_high[i]
            if not np.isnan(t):
                if t < peak_th:
                    fired = True
                else:
                    peak_th = t
        elif rule == 2:                                  # 吊燈（ATR 取當日，隨波動變動）
            a = atr[i]
            if not np.isnan(a) and close_[i] < peak_px - param * a:
                fired = True
        elif rule == 3:                                  # 自最高點回落 param
            if close_[i] < peak_px * (1.0 - param):
                fired = True
        elif rule == 4:                                  # 進場價 − param×ATR(進場判定日)
            if not np.isnan(atr_at_entry) and close_[i] < entry_px - param * atr_at_entry:
                fired = True
        elif rule == 5:                                  # 停利
            if close_[i] >= entry_px * (1.0 + param):
                fired = True
        elif rule == 6:                                  # 時間出場（i-entry_i＝已持有根數）
            if (i - entry_i) >= param:
                fired = True
        if fired or (use_native and native_exit[i]):
            exits[i] = True
            in_pos = False
    return entries, exits


class SingleMacdStrategy(VbtSingleStrategy):
    """
    MACD 策略。只描述「判定日」訊號，「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底處理。

    以「註解切換」管理（見 buy_signal / sell_signal / EXIT_RULE）：三個基礎方案（交叉／零軸／背離）
    互斥擇一，優化濾網（趨勢過濾／量能／流動性…）疊在選定基礎上、一次一條；--variant 只決定輸出資料夾。
    """

    # ── 出場優化·路徑相依（一次一條；維持 EXIT_NONE＝不啟用，走 sell_signal 的向量化路徑）──
    # 疊加／取代不在這裡決定，見 _REPLACE_RULES（預設：頂頂低取代、五條風控疊加）。
    # EXIT_RULE = EXIT_LOWER_HIGH    # 頂頂低：進場以來出現較低的 ZigZag(2%) 擺動高點
    # 五條風控（預設疊加；當唯一出場的純取代版，開 _REPLACE_RULES 的附錄那行）：
    # EXIT_RULE = EXIT_CHANDELIER    # 吊燈：收盤 < 進場後最高 − 3×ATR
    # EXIT_RULE = EXIT_TRAIL_PCT     # 移動停損：收盤自進場後最高回落 10%
    # EXIT_RULE = EXIT_ATR_STOP      # 固定停損：收盤 < 進場價 − 2×ATR
    # EXIT_RULE = EXIT_TAKE_PROFIT   # 固定停利：收盤 >= 進場價 × 1.20
    # EXIT_RULE = EXIT_TIME          # 時間出場：抱滿 60 根 K（對照組，不含市場判斷）
    EXIT_RULE = EXIT_NONE

    _RULE_PARAM = {EXIT_LOWER_HIGH: 0.0, EXIT_CHANDELIER: CHANDELIER_K,
                   EXIT_TRAIL_PCT: TRAIL_PCT, EXIT_ATR_STOP: ATR_STOP_K,
                   EXIT_TAKE_PROFIT: TAKE_PROFIT, EXIT_TIME: float(TIME_BARS)}

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD 引用 _01_data.calculate_macd（傳統 12/26/9）；另備交叉、柱狀圖、零軸、背離、趨勢輔助欄。

        只在這裡算「向量化、便宜」的欄位。ZigZag / Supertrend / SAR 要逐根掃描，
        改成用到才算（見 _ensure_*），沒開那些變體就不必付這筆成本。
        """
        # parquet 若已存 macd/signal_line 就沿用；缺欄才補算（不在策略內另寫 MACD 公式）
        if "macd" not in df.columns or "signal_line" not in df.columns:
            calculate_macd(df, short_period=MACD_SHORT, long_period=MACD_LONG, signal_period=MACD_SIGNAL)
        macd, sig = df["macd"], df["signal_line"]
        close = df["close"]
        # 柱狀圖 hist = DIF − 訊號線（動能）；zero-line = 0
        df["hist"] = macd - sig
        hist = df["hist"]
        # 黃金交叉：DIF 由下而上穿越訊號線（昨 DIF<=sig、今 DIF>sig）；死亡交叉相反
        df["golden"] = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        df["death"] = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        # 零軸之上（DIF、訊號線都 > 0）— 進場濾網用
        df["above_zero"] = (macd > 0) & (sig > 0)
        # 5 日均量（股數）→ 成交金額 = 5日均量 × 收盤價（用個股實際股價）— 流動性門檻用
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["turnover"] = df["vol_ma5"] * close
        # CMF（Chaikin Money Flow，20 日）— 優化·量能用；缺欄才補算。
        # 注意 calculate_cmf 產出欄名為 cmf_{window}（如 cmf_20），別名成 cmf 供 buy_signal 引用。
        if f"cmf_{CMF_WINDOW}" not in df.columns:
            calculate_cmf(df, window=CMF_WINDOW)
        df["cmf"] = df[f"cmf_{CMF_WINDOW}"]
        # 柱狀圖底背離（基礎 C 背離用）：今日 close 創 20 日新低、但今日 hist 未創 20 日新低＝價低動能不低。
        price_new_low = close <= close.rolling(DIVERGENCE_WINDOW).min()
        hist_not_new_low = hist > hist.rolling(DIVERGENCE_WINDOW).min()
        df["bull_divergence"] = price_new_low & hist_not_new_low
        # 柱狀圖頂背離（出場用，底背離的鏡像）：價創 20 日新高、但 hist 未創新高＝價高動能不高。
        price_new_high = close >= close.rolling(DIVERGENCE_WINDOW).max()
        hist_not_new_high = hist < hist.rolling(DIVERGENCE_WINDOW).max()
        df["bear_divergence"] = price_new_high & hist_not_new_high
        # MA200 年線（優化·趨勢過濾進場 / 趨勢跌破出場用）
        df["ma_long"] = close.rolling(MA_LONG_TREND).mean()

        # ── 進場濾網用欄位（皆於訊號判定日評估，不使用當日之後資訊）──
        if "adx" not in df.columns:
            calculate_adx(df, period=ADX_PERIOD)
        if "rsi" not in df.columns:
            calculate_rsi(df, period=RSI_PERIOD)          # calculate_rsi 產出欄名為 rsi
        df["vol_ma20"] = df["volume"].rolling(VOL_WINDOW).mean()
        df["hist_rising"] = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
        # 均線多頭排列 5>20>60：自算，不依賴 parquet 既有 sma 欄，避免不同來源口徑不一
        df["bull_align"] = ((close.rolling(5).mean() > close.rolling(20).mean())
                            & (close.rolling(20).mean() > close.rolling(60).mean()))
        df["new_high"] = close >= close.rolling(HIGH_WINDOW).max()
        df["gap_up"] = df["open"] > df["high"].shift(1)

        # ── 出場用欄位（向量化的部分）──
        if f"atr_{ATR_PERIOD}" not in df.columns:
            calculate_atr(df, window=ATR_PERIOD)
        if f"donchian_lower_{DONCHIAN_WINDOW}" not in df.columns:
            calculate_donchian(df, window=DONCHIAN_WINDOW)
        # 跌破「前 20 日」最低才算；含當日會讓「跌破」幾乎不可能成立
        df["dc_low_prev"] = df[f"donchian_lower_{DONCHIAN_WINDOW}"].shift(1)
        return df

    # ── 逐根掃描型指標：用到才算（避免沒開的變體白付成本）──
    def _ensure_zigzag(self, df: pd.DataFrame) -> pd.DataFrame:
        if "zigzag_turn_high" not in df.columns:
            calculate_zigzag(df, ZIGZAG_PCT)
        return df

    def _ensure_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        if "supertrend_flip_down" not in df.columns:
            calculate_supertrend(df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULT)
        return df

    def _ensure_psar(self, df: pd.DataFrame) -> pd.DataFrame:
        if "psar_flip_down" not in df.columns:
            calculate_psar(df, af_start=SAR_AF_START, af_step=SAR_AF_STEP, af_max=SAR_AF_MAX)
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        ── 三個「基礎方案」互斥擇一（本質不同的進場邏輯，各自一篇初探）──
          基礎 A 交叉：黃金交叉（DIF 上穿訊號線）。
          基礎 B 零軸：DIF 由下上穿 0 軸（穿越多空分界本身，不看訊號線）。
          基礎 C 背離：純柱狀圖背離（價創 20 日新低、但 hist 未創新低），不靠交叉。

        ── 進場濾網（疊在選定的基礎上，一次一條）──
          同一條濾網套在不同基礎上結果可以相反，故按基礎分別產出；各條的實測見文章系列。
        """
        macd = df["macd"]
        # ── 基礎方案（三選一，互斥）──
        signal = df["golden"]                                   # 基礎 A 交叉：黃金交叉
        # signal = (macd > 0) & (macd.shift(1) <= 0)            # 基礎 B 零軸：DIF 上穿 0
        # signal = df["bull_divergence"]                        # 基礎 C 背離：純背離（不靠交叉）

        # ── 進場濾網（疊在上面選定的基礎上，一次一條）──
        # signal = signal & (df["adx"] > ADX_MIN)                        # ①趨勢強度 ADX>25
        # signal = signal & (df["rsi"] < RSI_MAX) & (df["rsi"] > df["rsi"].shift(1))  # ②RSI<50 且上升
        # signal = signal & (df["volume"] > df["vol_ma20"] * VOL_MULTIPLE)  # ③放量 1.5 倍
        # signal = signal & df["hist_rising"]                            # ④柱狀圖連兩根遞增
        # signal = signal & (df["cmf"] > 0)                              # ⑤量能 CMF>0
        # signal = signal & (df["close"] > df["ma_long"])                # ⑥趨勢過濾 收盤>MA200
        # signal = signal & df["bull_align"]                             # ⑦均線多頭排列 5>20>60
        # signal = signal & df["new_high"]                               # ⑧創 250 日新高
        # signal = signal & df["gap_up"]                                 # ⑨跳空（開盤>昨日最高）
        # signal = signal & df["above_zero"]                             # （零軸之上：與基礎 B 概念重疊，已停用）

        # ── 流動性口徑（可成交門檻，只 gate 進場）──
        # 2026-08-25 之後的所有輪次一律開這一行；關掉＝無門檻口徑，與後續數字不可並列。
        # signal = signal & (df["turnover"] > TURNOVER_MIN)     # 優化：流動性（成交金額 > 1,000 萬）
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        ── 各基礎的自然出場（與進場基礎配對、互斥擇一）──
          基礎 A 交叉 / C 背離：死亡交叉（DIF 下穿訊號線）。
          基礎 B 零軸：DIF 由上下穿 0 軸。
          （註：hist 由正轉負 ⇔ DIF 下穿訊號線，與死叉「數學恆等」，故不另列。）

        ── 出場優化（一次一條）──
          附加（主軸）：原出場留著、新規則疊上去，先觸發者算 → 用 `signal = signal | (…)`。
          取代（附錄）：原出場整條拿掉，只用新規則         → 用 `signal = (…)`。
          兩種寫法的差別與為什麼以附加為主軸，見檔案上方 _REPLACE_RULES 的說明。
          頂頂低與五條風控要看進場價／進場以來最高／已持有幾根，必須逐根掃描，
          不能寫在這裡，改由 EXIT_RULE 那一區切換。各條的實測見文章系列。
        """
        macd = df["macd"]
        close, ma_long = df["close"], df["ma_long"]
        signal = df["death"]                                       # 基礎 A 交叉 / C 背離：死亡交叉
        # signal = (macd < 0) & (macd.shift(1) >= 0)               # 基礎 B 零軸：DIF 下穿 0
        # signal = df["bear_divergence"]                           # 矩陣用：頂背離出場

        # 附加型（主軸）：
        # signal = signal | ((close < ma_long) & (close.shift(1) >= ma_long.shift(1)))  # ＋跌破 MA200
        # signal = signal | self._ensure_supertrend(df)["supertrend_flip_down"]         # ＋Supertrend(10,3) 翻空
        # signal = signal | self._ensure_psar(df)["psar_flip_down"]                     # ＋拋物線 SAR 翻空
        # signal = signal | (close < df["dc_low_prev"])                                 # ＋跌破前 20 日最低
        # 取代型（附錄）：
        # signal = (close < ma_long) & (close.shift(1) >= ma_long.shift(1))  # 取代：跌破 MA200
        # signal = self._ensure_supertrend(df)["supertrend_flip_down"]       # 取代：Supertrend(10,3) 翻空
        # signal = self._ensure_psar(df)["psar_flip_down"]                   # 取代：拋物線 SAR 翻空
        # signal = close < df["dc_low_prev"]                                 # 取代：跌破前 20 日最低
        return signal.fillna(False).astype(bool)

    def build_signals(self, df: pd.DataFrame):
        """
        EXIT_RULE == EXIT_NONE → 沿用基底（向量化、自動位移隔日成交）。
        其餘 → 路徑相依：逐根掃描產「判定日」訊號後自行 shift(1) 成隔日開盤成交
               （覆寫後位移責任轉移到子類，見基底 docstring；作法比照 ma_cross 的 CHOCH）。
        """
        rule = self.EXIT_RULE
        if rule == EXIT_NONE:
            return super().build_signals(df)
        if rule == EXIT_LOWER_HIGH:
            self._ensure_zigzag(df)
            turn_high = df["zigzag_turn_high"].to_numpy(dtype=np.float64)
        else:
            turn_high = np.full(len(df), np.nan)          # 其餘規則用不到，給占位陣列
        entries, exits = _scan_path_exits(
            self.entry_signal(df).to_numpy(),
            self.sell_signal(df).to_numpy(),
            df["open"].to_numpy(dtype=np.float64),
            df["high"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            df[f"atr_{ATR_PERIOD}"].to_numpy(dtype=np.float64),
            turn_high, rule, self._RULE_PARAM[rule],
            rule not in _REPLACE_RULES)
        e = pd.Series(entries, index=df.index).shift(1, fill_value=False)
        x = pd.Series(exits, index=df.index).shift(1, fill_value=False)
        return e, x


def main(argv) -> int:
    """CLI：指定資料夾，MACD 參數固定 12/26/9，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="MACD 策略：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    # variant 只是輸出位置，不再用封閉 choices 限制（變體已擴到 40+：三基礎 × 九濾網 × 十出場）。
    # 命名慣例：{基礎}_{優化}_{口徑}，例 cross_baseline / div_adx25_amt / zero_exit_supertrend_amt。
    parser.add_argument("--variant", default="cross_baseline",
                        help="輸出資料夾分流（result/single_macd/<variant>/）；行為切換靠 buy_signal / "
                             "sell_signal / EXIT_RULE 的註解，--variant 只決定寫去哪，兩者請保持一致")
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"起始日 YYYY-MM-DD（預設標準區間 {DEFAULT_START}）")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"結束日 YYYY-MM-DD（預設標準區間 {DEFAULT_END}）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    strat = SingleMacdStrategy()

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit,
                              exclude=GLITCH)   # 排除 5 檔價格 glitch 壞股（跨策略同口徑）
    # 結果分流：各 variant 獨立子資料夾（對照用、互不覆蓋）；
    # 注意 variant 只是輸出位置，真正行為由 buy_signal / sell_signal / EXIT_RULE 的註解決定，務必一致
    out_dir = os.path.join(RESULT_DIR, "single_macd", args.variant)
    label = "single_macd"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場 MACD（variant={args.variant}，MACD={MACD_SHORT},{MACD_LONG},{MACD_SIGNAL}）===")
    print(f"參與股票數: {agg['參與股票數']}（失敗 {agg['失敗檔數']} 檔）")
    print(f"交易次數: {agg['交易次數']}")
    print(f"勝率(%): {agg['勝率(%)']}")
    print(f"中位數報酬率(%): {agg['中位數報酬率(%)']}")
    print(f"期望報酬值(EV): {agg['期望報酬值(EV)']}")
    print(f"獲利因子(PF): {agg['獲利因子(PF)']}")
    print(f"平均持有天數: {agg['平均持有天數']}")
    print(f"總獲利: {agg['總獲利']:.0f}")
    print(f"輸出目錄: {out_dir}")
    for path in written:
        print(f"  - {os.path.basename(path)}")
    if result["failed"]:
        print(f"失敗檔（前 10）: {result['failed'][:10]}")
    print(f"⚠️ 行為由 buy_signal / sell_signal / EXIT_RULE 的註解決定；"
          f"請確認與 --variant={args.variant} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
