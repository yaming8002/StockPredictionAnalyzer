"""
單一均線突破策略（single MA breakout）— ma_strategy 套件下的方案

繼承 VbtSingleStrategy，只記錄「判定日」買賣條件（隔日開盤成交、費稅、tick 由基底處理）：
  進場：突破單一 MA。預設＝優化 #1（雙日確認：突破隔日收盤仍站上才判定）；
        把 buy_signal 內標「# 優化 #1」那行註解掉 → 回 baseline（突破當日判定）。
  出場：收盤「下穿」單一 MA（跌破）。
  成交：判定日的「隔日開盤」（基底統一，有訊號一律隔日成交、不在訊號當日收盤）。

附分析：對資料中所有 MA 期數各跑一次，列出比較。
  期數來源：自動偵測資料欄位 sma_<N>；偵測不到才用 DEFAULT_PERIODS。

優化紀錄（每次一行；N = 本策略檔自己的累計次數；切換靠「註解」非參數，各 #N 獨立可疊加）：
  #1 2026-06-15 雙日確認進場：突破當天不買，要隔日收盤仍站上 MA 才判定買進
     （過濾一日假突破）。切換 = buy_signal 內標「# 優化 #1」那行。
  #2 2026-06-15 量能濾網：判定日要 5 日均量 > 20 日均量（近期量增）且 5 日均量 > 100 萬股
     （流動性），過濾無量假突破與冷門股。切換 = buy_signal 內標「# 優化 #2」那行。

執行（全市場單一 MA，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
  優化 #1（預設，# 優化 #1 行不註解）：
    python _02_strategy/ma_strategy/single_ma_strategy.py <資料夾> --ma 20 --variant opt1
  baseline（把 # 優化 #1 行註解掉後）：
    python _02_strategy/ma_strategy/single_ma_strategy.py <資料夾> --ma 20 --variant baseline
  （--variant 只決定輸出資料夾 result/single_ma/<variant>/；行為切換一律靠註解，兩者請一致）
"""
import argparse
import os
import re
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import batch
from _02_strategy.base.vbt.single import VbtSingleStrategy

# 回測結果輸出目錄（策略同目錄底下 ./result，已於 .gitignore 排除）
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# 偵測不到資料欄位時的預設 MA 期數
DEFAULT_PERIODS = (5, 10, 20, 50, 60, 120, 200)

# 標準回測區間：後續測試一律以此為主（掐掉 2000 殘月與 2026 未滿年，資料較穩定）；可用 --start/--end 覆蓋
DEFAULT_START = "2001-01-01"
DEFAULT_END = "2025-12-31"


class SingleMAStrategy(VbtSingleStrategy):
    """
    單一均線突破。設定 MA_PERIOD 後即可用。只描述「判定日」訊號，
    「隔日開盤成交 + 台股費用 / 稅 / tick」全由基底 VbtSingleStrategy 處理。

    優化以「註解切換」管理（見 buy_signal）：各「# 優化 #N」獨立一行可疊加，
      取消註解＝啟用該優化、註解＝關閉；全部註解 = baseline。
    """

    MA_PERIOD = 20

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """加單一均線 + 量能均線欄位（量能濾網用）。"""
        df["ma"] = df["close"].rolling(self.MA_PERIOD).mean()
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        進場「判定日」訊號（基底會自動延到隔日開盤成交）。

        baseline：當日收盤上穿 MA（突破當天判定）。
        優化 #1（雙日確認）：突破隔日收盤仍站上 MA 才判定（多等一天，過濾一日假突破）。
        優化 #2（量能濾網）：判定日要 5 日均量 > 20 日均量（近期量增）且 5 日均量 > 100 萬股
                          （流動性門檻），過濾無量假突破與冷門股。

        ▶ 切換：每個「# 優化 #N」獨立一行，可各自註解／取消註解疊加；全註解 = baseline。
        """
        above = df["close"] > df["ma"]
        signal = above & ~above.shift(1, fill_value=False)   # baseline：突破當日判定
        signal = above & signal.shift(1, fill_value=False)   # 優化 #1：雙日確認（註解此行 → 不做隔日確認）
        signal = signal & (df["vol_ma5"] > df["vol_ma20"]) & (df["vol_ma5"] > 1_000_000)  # 優化 #2：量能濾網（5日均量>20日均量 且 >100萬股；註解此行 → 不套量能）
        return signal

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場判定日：今天收盤下穿 MA（昨收 >= MA、今收 < MA）。"""
        below = df["close"] < df["ma"]
        return below & ~below.shift(1, fill_value=False)


def detect_ma_periods(df: pd.DataFrame) -> list:
    """從資料欄位 sma_<N> 萃取 MA 期數（排序）；無則回 DEFAULT_PERIODS。"""
    periods = []
    for col in df.columns:
        match = re.fullmatch(r"sma_(\d+)", str(col))
        if match:
            periods.append(int(match.group(1)))
    return sorted(periods) or list(DEFAULT_PERIODS)


def run_all_periods(df: pd.DataFrame, periods=None,
                    initial_cash: float = 100_000, split_cash: float = 10_000) -> dict:
    """
    對每個 MA 期數跑單一均線突破，回傳 {period: summary}。
    periods 不給 → 自動偵測資料中的 MA 期數。
    """
    periods = periods or detect_ma_periods(df)
    results = {}
    for period in periods:
        strat = SingleMAStrategy(initial_cash=initial_cash, split_cash=split_cash)
        strat.MA_PERIOD = period
        results[period] = strat.run(df)["summary"]
    return results


def main(argv) -> int:
    """CLI：指定資料夾，固定單一 MA 期數，掃全部股票各自獨立回測並彙總。"""
    parser = argparse.ArgumentParser(description="單一均線突破：資料夾批次回測")
    parser.add_argument("folder", help="OHLCV parquet 資料夾路徑")
    parser.add_argument("--ma", type=int, default=20, help="單一 MA 期數（預設 20）")
    parser.add_argument("--variant", choices=("baseline", "opt1", "opt2"), default="opt1",
                        help="輸出資料夾分流（result/single_ma/<variant>/）；行為切換靠 buy_signal "
                             "內『# 優化 #1』那行的註解，--variant 只決定寫去哪，兩者請保持一致")
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"起始日 YYYY-MM-DD（預設標準區間 {DEFAULT_START}）")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"結束日 YYYY-MM-DD（預設標準區間 {DEFAULT_END}）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    strat = SingleMAStrategy()
    strat.MA_PERIOD = args.ma

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit)
    # 結果分流：baseline / opt1 各自獨立子資料夾（對照用、互不覆蓋）；
    # 注意 variant 只是輸出位置，真正行為由 buy_signal 的「# 優化 #1」註解決定，務必一致
    out_dir = os.path.join(RESULT_DIR, "single_ma", args.variant)
    label = f"single_ma_{args.ma}"
    written = batch.write_results(result, out_dir, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場單一均線突破 MA={args.ma}（variant={args.variant}）===")
    print(f"參與股票數: {agg['參與股票數']}（失敗 {agg['失敗檔數']} 檔）")
    print(f"交易次數: {agg['交易次數']}")
    print(f"勝率(%): {agg['勝率(%)']}")
    print(f"期望報酬值(EV): {agg['期望報酬值(EV)']}")
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
