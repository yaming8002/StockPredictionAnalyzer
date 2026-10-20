"""
海龜交易法（single turtle）— turtle_strategy 套件下的方案
==========================================================

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）。

規則來源：Curtis Faith 公開的《The Original Turtle Trading Rules》。海龜原版有兩套系統，
買賣邏輯相同（Donchian 通道突破），只差在通道天數：
  系統 1（短線）：收盤價 > 前 20 日最高價 → 進場；收盤價 < 前 10 日最低價 → 出場。
  系統 2（長線）：收盤價 > 前 55 日最高價 → 進場；收盤價 < 前 20 日最低價 → 出場。
兩套系統是原典本身的兩個 baseline（不是優化），故以 --system 選擇通道天數，各自獨立輸出。

與原版的差異（文章需註明）：
  - 只做多：原版是期貨多空雙向。
  - 收盤判定、隔日開盤成交：原版是盤中超過前高一檔即進場（repo 統一口徑，與其他系列可比）。
  - 價格未還原現金股利：除息跳空可能提早觸發 N 日低點出場（與其他系列口徑一致）。
  - 單股階段每筆固定金額、不加碼：N 定倉與 ½N 加碼屬資金管理，留到多股階段。
  - baseline 只有通道本體：2N 停損、系統 1 的「上次獲利就跳過」與 55 日保險機制，留到調整篇逐項加上。

「前 N 日」不含當日：通道取 calculate_donchian（含當日）後再 shift(1)，
否則今日收盤永遠不可能大於含今日在內的最高價，訊號會被吃掉。

優化以「註解切換」管理（比照 single_kd_strategy）：baseline = 純通道；各優化只開自己那行。
  # 優化 #1（流動性門檻，可成交版）：成交金額 5日均量×股價 > 1,000 萬。只改 buy_signal 一行。
  全部註解 = baseline。

優化紀錄：
  （尚無；baseline 2026-10-06 建立）

執行（全市場，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
    python _02_strategy/turtle_strategy/single_turtle_strategy.py <資料夾> --system 1 --variant baseline
    python _02_strategy/turtle_strategy/single_turtle_strategy.py <資料夾> --system 2 --variant baseline
  輸出：result/single_turtle/s<系統>_<variant>/
  （--variant 只決定輸出資料夾；優化行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_volatility import calculate_donchian
from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy
# 資料品質排除集 GLITCH 與標準回測區間 DEFAULT_START/END：跨策略共用，統一由 base/vbt/common 取用（單一定義）。
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END


# 回測結果輸出目錄（策略同目錄底下 ./result，已於 .gitignore 排除）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# 海龜兩套系統的通道天數（原典固定值，不掃參數）：{系統: (進場通道天數, 出場通道天數)}
SYSTEMS = {
    1: (20, 10),   # 系統 1：20 日高進、10 日低出
    2: (55, 20),   # 系統 2：55 日高進、20 日低出
}

# 流動性門檻（優化 #1）：成交金額 = 5日均量(股) × 股價 > 1,000 萬（與 KD／MACD 系列同一基準）
TURNOVER_MIN = 10_000_000


class SingleTurtleStrategy(VbtSingleStrategy):
    """
    海龜通道突破策略。只描述「判定日」訊號，「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底處理。

    entry_window / exit_window 由 SYSTEMS 對照表帶入（系統 1 或 2），不是優化參數。
    """

    def __init__(self, entry_window: int, exit_window: int, **kwargs):
        super().__init__(**kwargs)
        self.entry_window = entry_window
        self.exit_window = exit_window

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """唐奇安通道引用 _01_data.calculate_donchian，再 shift(1) 成「前 N 日」極值。"""
        calculate_donchian(df, window=self.entry_window)
        calculate_donchian(df, window=self.exit_window)
        # 前 N 日最高 / 最低（不含當日），作為突破判定的比較基準
        df["entry_high"] = df[f"donchian_upper_{self.entry_window}"].shift(1)
        df["exit_low"] = df[f"donchian_lower_{self.exit_window}"].shift(1)
        # 成交金額 = 5 日均量(股) × 當日收盤價 — 優化 #1 用
        df["turnover"] = df["volume"].rolling(5).mean() * df["close"]
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：收盤價突破前 N 日最高價（N＝20 或 55，依系統）。
        優化 #1（流動性門檻）：再加成交金額 > 1,000 萬（可成交版）。
        """
        signal = df["close"] > df["entry_high"]                  # baseline：突破前 N 日高
        # signal = signal & (df["turnover"] > TURNOVER_MIN)      # 優化 #1：成交金額 > 1,000 萬
        return signal.fillna(False).astype(bool)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        出場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：收盤價跌破前 M 日最低價（M＝10 或 20，依系統）。
        """
        signal = df["close"] < df["exit_low"]                    # baseline：跌破前 M 日低
        return signal.fillna(False).astype(bool)


def main(argv) -> int:
    """CLI：指定資料夾與系統（1／2），掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="海龜交易法：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--system", type=int, choices=sorted(SYSTEMS), default=1,
                        help="海龜系統：1＝20 日高進／10 日低出；2＝55 日高進／20 日低出")
    parser.add_argument("--variant", choices=("baseline", "opt1"), default="baseline",
                        help="輸出資料夾分流（result/single_turtle/s<系統>_<variant>/）；行為切換靠 "
                             "buy_signal 內『# 優化 #1』那行的註解，--variant 只決定寫去哪，兩者請保持一致")
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"起始日 YYYY-MM-DD（預設標準區間 {DEFAULT_START}）")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"結束日 YYYY-MM-DD（預設標準區間 {DEFAULT_END}）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    entry_window, exit_window = SYSTEMS[args.system]
    strat = SingleTurtleStrategy(entry_window=entry_window, exit_window=exit_window)

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit,
                              exclude=GLITCH)   # 排除 5 檔價格 glitch 壞股（與其他系列同口徑）
    out_dir = os.path.join(RESULT_DIR, "single_turtle", f"s{args.system}_{args.variant}")
    label = f"single_turtle_s{args.system}"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場 海龜系統 {args.system}（{entry_window} 日高進／{exit_window} 日低出，"
          f"variant={args.variant}）===")
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
    print(f"⚠️ 行為由 buy_signal 內『# 優化 #1』那行的註解決定；請確認與 --variant={args.variant} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
