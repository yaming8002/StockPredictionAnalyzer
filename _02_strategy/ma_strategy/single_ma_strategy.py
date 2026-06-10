"""
單一均線突破策略（single MA breakout）— ma_strategy 套件下的方案

繼承 VbtSingleStrategy，只記錄買賣條件：
  進場：收盤「上穿」單一 MA（今天突破）→ 買進
  出場：收盤「下穿」單一 MA（今天跌破）→ 賣出
成交價 = 訊號日收盤（「今天突破就買入」）。

附分析：對資料中所有 MA 期數各跑一次，列出比較。
  期數來源：自動偵測資料欄位 sma_<N>；偵測不到才用 DEFAULT_PERIODS。

執行：
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _02_strategy/ma_strategy/single_ma_strategy.py <OHLCV parquet 路徑>
"""
import os
import re
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt.single import VbtSingleStrategy

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


def _load_prices(parquet_path: str) -> pd.DataFrame:
    """讀單檔 OHLCV parquet。index 須為 DatetimeIndex。"""
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index 必須是 DatetimeIndex（日期）")
    return df.sort_index()


def main(argv) -> int:
    """CLI：傳入 parquet 路徑，對資料所有 MA 期數各跑一次並列表比較。"""
    if len(argv) < 2:
        print("用法: python single_ma_strategy.py <OHLCV parquet 路徑>")
        return 1
    df = _load_prices(argv[1])
    results = run_all_periods(df)
    print(f"{'MA':>5} {'交易次數':>8} {'勝率(%)':>9} {'期望值(EV)':>12} {'總獲利':>14}")
    for period, summary in results.items():
        print(f"{period:>5} {summary['交易次數']:>8} {summary['勝率(%)']:>9} "
              f"{summary['期望報酬值(EV)']:>12} {summary['總獲利']:>14.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
