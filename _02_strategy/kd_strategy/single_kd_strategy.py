"""
KD 交叉策略（single KD cross）— kd_strategy 套件下的方案
========================================================

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：KD 黃金交叉（K 由下而上穿越 D）。
  出場：KD 死亡交叉（K 由上而下穿越 D）。
  成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。

KD 參數固定傳統值 n=9、K/D 各 3 日平滑（引用 _01_data.calculate_kd，不在策略內自算）。

優化以「註解切換」管理（比照 single_ma_strategy）：baseline = 基本交叉；各優化只開自己那行，互斥不疊：
  # 優化 #1（低/高檔交叉）需同時改 buy_signal / sell_signal 兩行，兩行一起開／一起關：
    進場黃金交叉須落在低檔（K、D 都 < 20）、出場死亡交叉須落在高檔（K、D 都 > 80）。
  # 優化 #3（CMF 資金流確認）：黃金交叉且 CMF>0（近 20 日資金淨流入），濾掉無量假交叉。只改 buy_signal 一行。
  # 優化 #4（KD 底背離，簡化窗口）：黃金交叉且「今日 close 創 20 日新低、但今日 K 未創 20 日新低」＝價低 K 不低。只改 buy_signal 一行。
  # 優化 #5（長線均線多頭）：黃金交叉且 MA120 > MA200（半年線在年線之上＝長多排列，兩線才能定趨勢）。只改 buy_signal 一行。
  全部註解 = baseline。

優化紀錄：
  #1 2026-07-05 低/高檔交叉（買在深超賣、賣在超買）——翻正但長抱 ~320 天。
  #2 2026-07-05 流動性門檻（#2a 1000 張 / #2b 成交金額；2026-07-09 由 3000 萬放寬為 1000 萬），可疊 baseline 或 opt1。
  #3 2026-07-07 CMF>0 資金流確認（加在原始黃金交叉上，非 opt1）。
  #4 2026-07-07 KD 底背離簡化窗口版（價創 20 日新低但 K 未創新低，加在原始黃金交叉上）。
  #5 2026-07-08 長線均線多頭 MA120>MA200（半年線 > 年線，只在長多排列時承接黃金交叉；加在原始黃金交叉上）。
     全市場結果：PF 0.809、期望值 −45.07、中位數 −0.81%、交易 336,021 筆、−1,514 萬（vs baseline PF 0.822）
     ——濾掉半數交易但每筆品質未改善，長多濾網對進場點無效；與 CMF 同結論（病灶在出場）。
  #6 2026-07-09 出場優化·高檔死叉（K、D>80 才出，進場固定純黃金交叉）。PF 2.017、中位 +1.77%、+3,282 萬。
  #7 2026-07-09 出場優化·K 跌破 50（K 由上穿越中線往下）。PF 1.011、中位 −0.78%、+132 萬（勉強翻正）。
  #8 2026-07-09 出場優化·過熱後 K 下彎（昨 K>80 且今日 K 下彎）。PF 1.174、中位 +0.75%、+1,260 萬。
  #9 2026-07-09 出場優化·頂背離（價創 20 日新高但 K 未創新高）。PF 1.110、中位 +1.93%、+638 萬。
  ⭐ 關鍵發現：進場固定「爛」的純黃金交叉（單獨 PF 0.822、−2,885 萬），只換出場，#6~#9 全數翻正——
     證明 KD 純交叉的虧損來源在出場（死叉洗掉獲利）、不在進場。#6 甚至 +3,282 萬 > opt1 的 +1,913 萬。
     可成交宇宙覆核（2026-07-10，進場加 1,000 萬成交金額門檻）：只剩 #6（PF 1.635、+1,581 萬）與 opt1
     （PF 1.445）撐住；#7/#8/#9 全跌破 1.0（裸版正報酬多來自低流動性小股）。詳見
     blog/reference/single_kd/2026-07-10_kd_liquidity_1000w_evaluation.md。

⚠️ opt1 的高檔死亡交叉出場較嚴：個股若一路陰跌、K/D 未摸到 80 就死叉，可能長抱不出場，
   交易數會明顯少於 baseline。這是「正統低/高檔」定義的固有性質，照定義實作、由結果表反映。

流動性（優化 #2，兩基準並存、擇一疊上）：
  #2a 張數：5 日均量 > 1000 張；  #2b 金額：成交金額 5日均量×股價 > 1,000 萬
  （金額版用個股實際股價換算真實可入場金額，高價低量股不誤殺；>300 張門檻已作廢）。皆可套 baseline 或 opt1。

執行（全市場，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
  變體 → 註解切換（buy「#1」低檔、sell「#1」高檔、buy「#2a」1000張、buy「#2b」1000萬、
                    buy「#3」CMF、buy「#4」底背離、buy「#5」長多）：
    baseline      ：全註解
    opt1          ：#1 兩行
    baseline_lot  ：#2a
    opt1_lot      ：#1 兩行 + #2a
    baseline_amt  ：#2b
    opt1_amt      ：#1 兩行 + #2b
    opt3          ：#3（黃金交叉 + CMF>0）
    opt4          ：#4（黃金交叉 + 底背離）
    opt5          ：#5（黃金交叉 + MA120>MA200）
    python _02_strategy/kd_strategy/single_kd_strategy.py <資料夾> --variant <上列之一>
  （--variant 只決定輸出資料夾 result/single_kd/<variant>/；行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_momentum_volume import calculate_kd, calculate_cmf
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
# 資料品質排除集 GLITCH 與標準回測區間 DEFAULT_START/END：跨策略共用，統一由 base/vbt/common 取用（單一定義）。
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END


# 回測結果輸出目錄（策略同目錄底下 ./result，已於 .gitignore 排除）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# KD 傳統參數（固定，不掃期數）
KD_N = 9            # RSV 回看天數
KD_K_SMOOTH = 3     # K 平滑
KD_D_SMOOTH = 3     # D 平滑

# 超買 / 超賣門檻（opt1 低/高檔交叉、出場優化 #6/#8 用）
OVERSOLD = 20
OVERBOUGHT = 80
# 中線（出場優化 #7 用）：K 由上往下穿越 50 出場
MID_LINE = 50

# 流動性門檻（優化 #2）— 兩種基準並存，各自對照：
#   #2a 張數：5 日均量 > 1000 張（= 1,000,000 股）。單純看量能。
#   #2b 金額：成交金額 = 5日均量(股) × 股價 > 1,000 萬。用個股實際股價換算真實可入場金額，
#            高價低量股不被誤殺。（2026-07-09 由 3,000 萬放寬為 1,000 萬——覆蓋更多可成交股、
#            供 opt3~opt9 進出場優化做可成交宇宙評估；舊 3,000 萬數字見 2026-07-05 reference。）
# （註：>300 張門檻已作廢——對高價低量股不公平，不再使用。）
VOL_LOT_MIN = 1_000_000    # #2a：1000 張 = 100 萬股
TURNOVER_MIN = 10_000_000  # #2b：成交金額 1,000 萬

# CMF 視窗（優化 #3 用）：Chaikin Money Flow 回看天數
CMF_WINDOW = 20
# 底背離視窗（優化 #4 用）：價創新低但 K 未創新低的回看天數
DIVERGENCE_WINDOW = 20
# 長線均線多頭排列（優化 #5 用）：MA120（半年線）在 MA200（年線）之上＝長多
MA_MID = 120        # 半年線
MA_LONG = 200       # 年線

class SingleKDStrategy(VbtSingleStrategy):
    """
    KD 交叉策略。只描述「判定日」訊號，「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底處理。

    優化以「註解切換」管理（見 buy_signal / sell_signal）：baseline 為基本交叉，
      # 優化 #1（低/高檔交叉）兩行（買、賣各一）須一起開／一起關；全註解 = baseline。
    """

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """KD 引用 _01_data.calculate_kd（傳統 9,3,3）；另備交叉與高低檔區輔助布林欄。"""
        # parquet 若已存 k/d 就沿用；缺欄才補算（不在策略內另寫 KD 公式）
        if "k" not in df.columns or "d" not in df.columns:
            calculate_kd(df, n=KD_N, k_smooth=KD_K_SMOOTH, d_smooth=KD_D_SMOOTH)
        k, d = df["k"], df["d"]
        # 黃金交叉：K 由下而上穿越 D（昨 K<=D、今 K>D）；死亡交叉相反
        df["golden"] = (k > d) & (k.shift(1) <= d.shift(1))
        df["death"] = (k < d) & (k.shift(1) >= d.shift(1))
        # 低檔區 / 高檔區（K、D 都在門檻同側）— opt1 用
        df["low_zone"] = (k < OVERSOLD) & (d < OVERSOLD)
        df["high_zone"] = (k > OVERBOUGHT) & (d > OVERBOUGHT)
        # 5 日均量（股數）— 流動性金額門檻用；無對應 indicators 故自算
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        # 成交金額 = 5 日均量(股) × 當日收盤價（用個股實際股價，不是固定 30 元）— 優化 #2 用
        df["turnover"] = df["vol_ma5"] * df["close"]
        # CMF（Chaikin Money Flow，20 日）— 優化 #3 用；缺欄才補算（不在策略內另寫公式）。
        # 注意 calculate_cmf 產出欄名為 cmf_{window}（如 cmf_20），別名成 cmf 供 buy_signal 引用。
        if f"cmf_{CMF_WINDOW}" not in df.columns:
            calculate_cmf(df, window=CMF_WINDOW)
        df["cmf"] = df[f"cmf_{CMF_WINDOW}"]
        # 底背離（優化 #4 用）：今日 close 創 20 日新低、但今日 K 未創 20 日新低＝價低 K 不低。
        price_new_low = df["close"] <= df["close"].rolling(DIVERGENCE_WINDOW).min()
        k_not_new_low = k > k.rolling(DIVERGENCE_WINDOW).min()
        df["bull_divergence"] = price_new_low & k_not_new_low
        # 頂背離（出場優化 #9 用）：今日 close 創 20 日新高、但今日 K 未創 20 日新高＝價高 K 不高。
        price_new_high = df["close"] >= df["close"].rolling(DIVERGENCE_WINDOW).max()
        k_not_new_high = k < k.rolling(DIVERGENCE_WINDOW).max()
        df["top_divergence"] = price_new_high & k_not_new_high
        # 長線均線多頭排列（優化 #5 用）：MA120 > MA200＝半年線在年線之上。
        # 兩條均線比大小才能代表趨勢方向，單一均線不足以判定多頭（calculate_sma 為空殼故內算）。
        df["bull_trend"] = df["close"].rolling(MA_MID).mean() > df["close"].rolling(MA_LONG).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：KD 黃金交叉（K 上穿 D）。
        優化 #1（低檔交叉）：黃金交叉且落在低檔（K、D 都 < 20）。與 sell 的 #1 一起開／一起關。
        優化 #2（流動性門檻，兩基準擇一，套在 baseline 或 opt1 上）：
            #2a 張數：5 日均量 > 1000 張；  #2b 金額：成交金額 5日均量×股價 > 1,000 萬。
        優化 #3（CMF 資金流確認）：黃金交叉且 CMF>0（近 20 日資金淨流入），濾掉無量假交叉。
        優化 #4（KD 底背離）：黃金交叉且價創 20 日新低、但 K 未創 20 日新低（價低 K 不低）。
        優化 #5（長線均線多頭）：黃金交叉且 MA120 > MA200（半年線在年線之上＝長多排列）。
        （#3 / #4 / #5 各自加在原始黃金交叉上、互斥擇一開，非疊在 opt1；一次只開一行。）
        """
        signal = df["golden"]                        # baseline：黃金交叉
        # signal = signal & df["low_zone"]           # 優化 #1：低檔黃金交叉（K、D 都 < 20）
        # signal = signal & (df["vol_ma5"] > VOL_LOT_MIN)    # 優化 #2a：5 日均量 > 1000 張流動性門檻
        # signal = signal & (df["turnover"] > TURNOVER_MIN)  # 優化 #2b：成交金額 > 1,000 萬流動性門檻
        # signal = signal & (df["cmf"] > 0)          # 優化 #3：CMF>0 資金流確認
        # signal = signal & df["bull_divergence"]    # 優化 #4：底背離（價低 K 不低）
        # signal = signal & df["bull_trend"]         # 優化 #5：長線多頭排列（MA120 > MA200）
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：KD 死亡交叉（K 下穿 D）。
        優化 #1（高檔交叉）：死亡交叉且落在高檔（K、D 都 > 80）。與 buy 的 #1 一起開／一起關。

        出場優化 #6~#9（進場一律固定純黃金交叉＝buy 全註解；各自取代 baseline 死叉出場，一次只開一行）：
          #6 高檔死叉出場：死叉且高檔（K、D>80）——隔離「高檔出場」對純黃金進場的貢獻。
          #7 K 跌破 50 出場：K 由上往下穿越中線 50（不看死叉）。
          #8 過熱後 K 下彎出場：昨日 K>80 且今日 K 下彎（K < 昨 K）。
          #9 頂背離出場：價創 20 日新高、但 K 未創 20 日新高。
        """
        k = df["k"]
        signal = df["death"]                         # baseline：死亡交叉
        # signal = signal & df["high_zone"]          # 優化 #1：高檔死亡交叉（K、D 都 > 80）
        # signal = df["death"] & df["high_zone"]     # 出場優化 #6：高檔死叉出場（進場固定純黃金交叉）
        # signal = (k < MID_LINE) & (k.shift(1) >= MID_LINE)      # 出場優化 #7：K 跌破 50 出場
        # signal = (k.shift(1) > OVERBOUGHT) & (k < k.shift(1))   # 出場優化 #8：過熱後 K 下彎出場
        # signal = df["top_divergence"]              # 出場優化 #9：頂背離出場（價高 K 不高）
        return signal.fillna(False).astype(bool)


def main(argv) -> int:
    """CLI：指定資料夾，KD 參數固定 9,3,3，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="KD 交叉：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--variant",
                        choices=("baseline", "opt1",
                                 "baseline_lot", "opt1_lot",      # #2a：1000 張
                                 "baseline_amt", "opt1_amt",      # #2b：成交金額 1,000 萬
                                 "opt3", "opt4", "opt5",          # #3 CMF / #4 底背離 / #5 長多（進場）
                                 "opt6", "opt7", "opt8", "opt9"), # #6~#9 出場優化（進場固定黃金交叉）
                        default="baseline",
                        help="輸出資料夾分流（result/single_kd/<variant>/）；行為切換靠 buy_signal / "
                             "sell_signal 內『# 優化 #1』兩行的註解，--variant 只決定寫去哪，兩者請保持一致")
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"起始日 YYYY-MM-DD（預設標準區間 {DEFAULT_START}）")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"結束日 YYYY-MM-DD（預設標準區間 {DEFAULT_END}）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    strat = SingleKDStrategy()

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit,
                              exclude=GLITCH)   # 排除 5 檔價格 glitch 壞股（與 ma_cross 同口徑）
    # 結果分流：baseline / opt1 各自獨立子資料夾（對照用、互不覆蓋）；
    # 注意 variant 只是輸出位置，真正行為由 buy_signal / sell_signal 的「# 優化 #1」註解決定，務必一致
    out_dir = os.path.join(RESULT_DIR, "single_kd", args.variant)
    label = "single_kd"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場 KD 交叉（variant={args.variant}，KD={KD_N},{KD_K_SMOOTH},{KD_D_SMOOTH}）===")
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
    print(f"⚠️ 行為由 buy_signal / sell_signal 內『# 優化 #1』兩行的註解決定；請確認與 --variant={args.variant} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
