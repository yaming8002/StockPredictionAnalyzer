"""
MACD 策略（single MACD）— macd_strategy 套件下的方案
========================================================

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）。
成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。
MACD 參數固定傳統值 12/26/9（引用 _01_data.calculate_macd，不在策略內自算）。
產出 macd（DIF 快線＝短 EMA − 長 EMA）與 signal_line（訊號線）；柱狀圖 hist = macd − signal_line 於策略內算。

【三個「基礎方案」— 本檔的骨架】
MACD 有三種「本質不同」的進場邏輯，各自獨立初探（見 reference macd/，各一篇）：
  基礎 A 交叉（cross）：黃金交叉進場 / 死叉出場。最基礎的教學交叉。
  基礎 B 零軸（zero） ：DIF 上穿 0 進場 / 下穿 0 出場。穿越多空分界本身，不看訊號線。
  基礎 C 背離（div）  ：純柱狀圖背離進場（價創 20 日新低但 hist 未創新低），不靠交叉；死叉出場。
趨勢過濾 / 零軸位置 / 量能等「濾網」不是新的進場邏輯，而是「優化」——同一個濾網套在
不同基礎上結果不同，故優化按基礎分別產出（cross_* / zero_* / div_* 各自的優化），不混為一談。

【背景】2026-08-09 網路主流規則調研（reference macd/2026-08-09）：大規模實證（EdgeTools 1,430 萬次
測試）顯示 MACD「交叉類」（線交叉／零軸交叉／柱狀圖方向…）皆無統計 edge，唯一勉強有邊際的是「背離」。
台股全史初探印證方向：三基礎裡背離勝率最高（47%、中位數唯一為正），交叉最平庸、零軸靠右尾大單衝總獲利。

以「註解切換」管理（比照 single_kd）：buy_signal / sell_signal 內三基礎行互斥擇一、優化行疊在其上，
一次只開一條同類行；--variant 只決定輸出資料夾、須與註解狀態一致。

三基礎（buy_signal / sell_signal 各一組互斥行）：
  cross_baseline：buy 黃金交叉、sell 死叉。
  zero_baseline ：buy DIF 上穿 0、sell DIF 下穿 0。
  div_baseline  ：buy 純背離、sell 死叉。

優化（疊在對應基礎上，按基礎分別跑；此輪尚未執行，待初探成績出來再指定）：
  趨勢過濾：且收盤 > MA200；　量能：且 CMF>0；　流動性：且成交金額 > 1,000 萬；　出場·趨勢跌破：收盤跌破 MA200。
  對應 variant 前綴：cross_trend / cross_cmf / cross_amt / cross_exit_ma、zero_* 、div_* …

執行（全市場，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
    python _02_strategy/macd_strategy/single_macd_strategy.py <資料夾> --variant <cross_baseline|zero_baseline|div_baseline|…>
  （--variant 只決定輸出資料夾 result/single_macd/<variant>/；行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_trend import calculate_macd
from _01_data.indicators_momentum_volume import calculate_cmf
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
# 背離視窗（基礎 C 背離用）：價創新低但 hist 未創新低的回看天數
DIVERGENCE_WINDOW = 20
# 趨勢過濾均線（優化·趨勢過濾進場 / 趨勢跌破出場用）：MA200 年線＝長期趨勢方向
MA_LONG_TREND = 200


class SingleMacdStrategy(VbtSingleStrategy):
    """
    MACD 策略。只描述「判定日」訊號，「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底處理。

    以「註解切換」管理（見 buy_signal / sell_signal）：三個基礎方案（交叉／零軸／背離）互斥擇一，
    優化濾網（趨勢過濾／量能／流動性）疊在選定基礎上、一次一條；--variant 只決定輸出資料夾。
    """

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 引用 _01_data.calculate_macd（傳統 12/26/9）；另備交叉、柱狀圖、零軸、背離、趨勢輔助欄。"""
        # parquet 若已存 macd/signal_line 就沿用；缺欄才補算（不在策略內另寫 MACD 公式）
        if "macd" not in df.columns or "signal_line" not in df.columns:
            calculate_macd(df, short_period=MACD_SHORT, long_period=MACD_LONG, signal_period=MACD_SIGNAL)
        macd, sig = df["macd"], df["signal_line"]
        # 柱狀圖 hist = DIF − 訊號線（動能）；zero-line = 0
        df["hist"] = macd - sig
        hist = df["hist"]
        # 黃金交叉：DIF 由下而上穿越訊號線（昨 DIF<=sig、今 DIF>sig）；死亡交叉相反
        df["golden"] = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        df["death"] = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        # 零軸之上（DIF、訊號線都 > 0）— 優化 #2 用
        df["above_zero"] = (macd > 0) & (sig > 0)
        # 5 日均量（股數）→ 成交金額 = 5日均量 × 收盤價（用個股實際股價）— 優化 #5 用
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["turnover"] = df["vol_ma5"] * df["close"]
        # CMF（Chaikin Money Flow，20 日）— 優化·量能用；缺欄才補算。
        # 注意 calculate_cmf 產出欄名為 cmf_{window}（如 cmf_20），別名成 cmf 供 buy_signal 引用。
        if f"cmf_{CMF_WINDOW}" not in df.columns:
            calculate_cmf(df, window=CMF_WINDOW)
        df["cmf"] = df[f"cmf_{CMF_WINDOW}"]
        # 柱狀圖底背離（基礎 C 背離用）：今日 close 創 20 日新低、但今日 hist 未創 20 日新低＝價低動能不低。
        price_new_low = df["close"] <= df["close"].rolling(DIVERGENCE_WINDOW).min()
        hist_not_new_low = hist > hist.rolling(DIVERGENCE_WINDOW).min()
        df["bull_divergence"] = price_new_low & hist_not_new_low
        # MA200 年線（優化·趨勢過濾進場 / 趨勢跌破出場用）
        df["ma_long"] = df["close"].rolling(MA_LONG_TREND).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        ── 三個「基礎方案」互斥擇一（本質不同的進場邏輯，各自一篇初探）──
          基礎 A 交叉：黃金交叉（DIF 上穿訊號線）。
          基礎 B 零軸：DIF 由下上穿 0 軸（穿越多空分界本身，不看訊號線）。
          基礎 C 背離：純柱狀圖背離（價創 20 日新低、但 hist 未創新低），不靠交叉。

        ── 優化濾網（疊在選定的基礎上，一次一條；優化按基礎分別產出）──
          趨勢過濾：且收盤 > MA200；　量能：且 CMF>0；　流動性：且成交金額 > 1,000 萬。
        """
        macd = df["macd"]
        # ── 基礎方案（三選一，互斥）──
        signal = df["golden"]                                   # 基礎 A 交叉：黃金交叉
        # signal = (macd > 0) & (macd.shift(1) <= 0)            # 基礎 B 零軸：DIF 上穿 0
        # signal = df["bull_divergence"]                        # 基礎 C 背離：純背離（不靠交叉）
        # ── 優化濾網（疊在上面選定的基礎上，一次一條）──
        # signal = signal & (df["close"] > df["ma_long"])       # 優化：趨勢過濾（收盤 > MA200）
        # signal = signal & (df["cmf"] > 0)                     # 優化：量能（CMF>0）
        # signal = signal & (df["turnover"] > TURNOVER_MIN)     # 優化：流動性（成交金額 > 1,000 萬）
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        ── 各基礎的自然出場（與進場基礎配對、互斥擇一）──
          基礎 A 交叉 / C 背離：死亡交叉（DIF 下穿訊號線）。
          基礎 B 零軸：DIF 由上下穿 0 軸。
          （註：hist 由正轉負 ⇔ DIF 下穿訊號線，與死叉「數學恆等」，故不另列。）

        ── 出場優化（進場固定對應基礎、各自取代自然出場，一次一條）──
          趨勢跌破：收盤跌破 MA200（close<MA200 且昨 close>=MA200）。
        """
        macd = df["macd"]
        close, ma_long = df["close"], df["ma_long"]
        signal = df["death"]                                       # 基礎 A 交叉 / C 背離：死亡交叉
        # signal = (macd < 0) & (macd.shift(1) >= 0)               # 基礎 B 零軸：DIF 下穿 0
        # signal = (close < ma_long) & (close.shift(1) >= ma_long) # 出場優化：趨勢跌破 MA200
        return signal.fillna(False).astype(bool)


def main(argv) -> int:
    """CLI：指定資料夾，MACD 參數固定 12/26/9，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="MACD 策略：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--variant",
                        choices=(  # 三基礎 baseline（本輪初探）
                                 "cross_baseline", "zero_baseline", "div_baseline",
                                 # 交叉方案的優化（後續章節，按基礎分別產出）
                                 "cross_trend", "cross_cmf", "cross_amt", "cross_exit_ma",
                                 # 零軸方案的優化
                                 "zero_trend", "zero_cmf", "zero_amt", "zero_exit_ma",
                                 # 背離方案的優化
                                 "div_trend", "div_amt", "div_exit_ma"),
                        default="cross_baseline",
                        help="輸出資料夾分流（result/single_macd/<variant>/）；行為切換靠 buy_signal / "
                             "sell_signal 內基礎/優化行的註解，--variant 只決定寫去哪，兩者請保持一致")
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
    # 注意 variant 只是輸出位置，真正行為由 buy_signal / sell_signal 的「# 優化 #N」註解決定，務必一致
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
    print(f"⚠️ 行為由 buy_signal / sell_signal 內『# 優化 #N』的註解決定；請確認與 --variant={args.variant} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
