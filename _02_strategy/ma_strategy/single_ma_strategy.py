"""
單一均線突破策略（single MA breakout）— ma_strategy 套件下的方案

繼承 VbtSingleStrategy，只記錄買賣條件：
  進場：收盤「上穿」單一 MA（今天突破）→ 買進
  出場：收盤「下穿」單一 MA（今天跌破）→ 賣出
成交價 = 訊號日收盤（「今天突破就買入」）。

附分析：對資料中所有 MA 期數各跑一次，列出比較。
  期數來源：自動偵測資料欄位 sma_<N>；偵測不到才用 DEFAULT_PERIODS。

執行（全市場單一 MA，掃整個資料夾、彙總；結果寫策略同目錄 ./result）：
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _02_strategy/ma_strategy/single_ma_strategy.py <資料夾> --ma 20
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


class SingleMAStrategy(VbtSingleStrategy):
    """單一均線突破。設定 MA_PERIOD 後即可用。"""

    MA_PERIOD = 20

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """加單一均線欄位。"""
        df["ma"] = df["close"].rolling(self.MA_PERIOD).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場：今天收盤上穿 MA（昨收 <= MA、今收 > MA）。"""
        above = df["close"] > df["ma"]
        return above & ~above.shift(1, fill_value=False)

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場：今天收盤下穿 MA（昨收 >= MA、今收 < MA）。"""
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
    parser.add_argument("--trades", action="store_true",
                        help="另存逐筆交易紀錄（預設不存，只出彙總）")
    parser.add_argument("--start", default=None, help="起始日 YYYY-MM-DD（可選）")
    parser.add_argument("--end", default=None, help="結束日 YYYY-MM-DD（可選）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（測試用）")
    args = parser.parse_args(argv[1:])

    strat = SingleMAStrategy()
    strat.MA_PERIOD = args.ma

    result = batch.run_folder(strat, args.folder,
                              start=args.start, end=args.end, limit=args.limit)
    label = f"single_ma_{args.ma}"
    written = batch.write_results(result, RESULT_DIR, label, write_trades=args.trades)

    agg = result["aggregate"]
    print(f"=== 全市場單一均線突破 MA={args.ma} ===")
    print(f"參與股票數: {agg['參與股票數']}（失敗 {agg['失敗檔數']} 檔）")
    print(f"交易次數: {agg['交易次數']}")
    print(f"勝率(%): {agg['勝率(%)']}")
    print(f"期望報酬值(EV): {agg['期望報酬值(EV)']}")
    print(f"總獲利: {agg['總獲利']:.0f}")
    print(f"輸出目錄: {RESULT_DIR}")
    for path in written:
        print(f"  - {os.path.basename(path)}")
    if result["failed"]:
        print(f"失敗檔（前 10）: {result['failed'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
