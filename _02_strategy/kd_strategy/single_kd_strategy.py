"""
KD 交叉策略（single KD cross）— kd_strategy 套件下的方案
========================================================

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：KD 黃金交叉（K 由下而上穿越 D）。
  出場：KD 死亡交叉（K 由上而下穿越 D）。
  成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。

KD 參數固定傳統值 n=9、K/D 各 3 日平滑（引用 _01_data.calculate_kd，不在策略內自算）。

優化以「註解切換」管理（比照 single_ma_strategy）：baseline = 基本交叉；
  # 優化 #1（低/高檔交叉）需同時改 buy_signal / sell_signal 兩行，兩行一起開／一起關：
    進場黃金交叉須落在低檔（K、D 都 < 20）、出場死亡交叉須落在高檔（K、D 都 > 80）。
  全部註解 = baseline。

⚠️ opt1 的高檔死亡交叉出場較嚴：個股若一路陰跌、K/D 未摸到 80 就死叉，可能長抱不出場，
   交易數會明顯少於 baseline。這是「正統低/高檔」定義的固有性質，照定義實作、由結果表反映。

流動性（優化 #2，兩基準並存、擇一疊上）：
  #2a 張數：5 日均量 > 1000 張；  #2b 金額：成交金額 5日均量×股價 > 3,000 萬
  （金額版用個股實際股價換算真實可入場金額，高價低量股不誤殺；>300 張門檻已作廢）。皆可套 baseline 或 opt1。

執行（全市場，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
  變體 → 註解切換（buy「#1」低檔、sell「#1」高檔、buy「#2a」1000張、buy「#2b」3000萬）：
    baseline      ：全註解
    opt1          ：#1 兩行
    baseline_lot  ：#2a
    opt1_lot      ：#1 兩行 + #2a
    baseline_amt  ：#2b
    opt1_amt      ：#1 兩行 + #2b
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

from _01_data.indicators_momentum_volume import calculate_kd
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

# 超買 / 超賣門檻（opt1 低/高檔交叉用）
OVERSOLD = 20
OVERBOUGHT = 80

# 流動性門檻（優化 #2）— 兩種基準並存，各自對照：
#   #2a 張數：5 日均量 > 1000 張（= 1,000,000 股）。單純看量能。
#   #2b 金額：成交金額 = 5日均量(股) × 股價 > 3,000 萬（= 1000張×1000股×30元）。
#            用個股實際股價換算真實可入場金額，高價低量股不被誤殺。
# （註：>300 張門檻已作廢——對高價低量股不公平，不再使用。）
VOL_LOT_MIN = 1_000_000    # #2a：1000 張 = 100 萬股
TURNOVER_MIN = 30_000_000  # #2b：成交金額 3,000 萬

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
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：KD 黃金交叉（K 上穿 D）。
        優化 #1（低檔交叉）：黃金交叉且落在低檔（K、D 都 < 20）。與 sell 的 #1 一起開／一起關。
        優化 #2（流動性門檻，兩基準擇一，套在 baseline 或 opt1 上）：
            #2a 張數：5 日均量 > 1000 張；  #2b 金額：成交金額 5日均量×股價 > 3,000 萬。
        """
        signal = df["golden"]                        # baseline：黃金交叉
        # signal = signal & df["low_zone"]           # 優化 #1：低檔黃金交叉（K、D 都 < 20）
        # signal = signal & (df["vol_ma5"] > VOL_LOT_MIN)    # 優化 #2a：5 日均量 > 1000 張流動性門檻
        # signal = signal & (df["turnover"] > TURNOVER_MIN)  # 優化 #2b：成交金額 > 3,000 萬流動性門檻
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：KD 死亡交叉（K 下穿 D）。
        優化 #1（高檔交叉）：死亡交叉且落在高檔（K、D 都 > 80）。與 buy 的 #1 一起開／一起關。
        """
        signal = df["death"]                         # baseline：死亡交叉
        # signal = signal & df["high_zone"]          # 優化 #1：高檔死亡交叉（K、D 都 > 80）
        return signal.fillna(False).astype(bool)


def main(argv) -> int:
    """CLI：指定資料夾，KD 參數固定 9,3,3，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="KD 交叉：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--variant",
                        choices=("baseline", "opt1",
                                 "baseline_lot", "opt1_lot",      # #2a：1000 張
                                 "baseline_amt", "opt1_amt"),     # #2b：成交金額 3,000 萬
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
