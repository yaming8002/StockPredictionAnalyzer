"""
雙均線交叉策略（MA cross）— ma_strategy 套件下的其中一支方案

繼承 VbtSingleStrategy，**只記錄買賣條件**，引擎 / 費用 / tick / 輸出皆由基底處理。

策略邏輯（2 日確認）：
  進場：short_ma 上穿 long_ma，且上穿後連 2 日維持 → 訊號 bar 開盤買進
  出場：鏡像（下穿後連 2 日確認）→ 訊號 bar 開盤賣出

執行：
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _02_strategy/ma_strategy/ma_cross_strategy.py <OHLCV parquet 路徑>
"""
import os
import sys

# 直接執行此檔時，把專案根目錄加進 sys.path（讓 _02_strategy.* 點號 import 可解析）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt.single import VbtSingleStrategy


class MACrossStrategy(VbtSingleStrategy):
    """雙均線交叉（2 日確認）。子類設定 SHORT_MA / LONG_MA。"""

    SHORT_MA = 20
    LONG_MA = 50

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """加短 / 長均線欄位。"""
        df["ma_short"] = df["close"].rolling(self.SHORT_MA).mean()
        df["ma_long"] = df["close"].rolling(self.LONG_MA).mean()
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """進場：上穿後連 2 日確認（i-3 時 short<=long、i-1 與 i-2 時 short>long）。"""
        above = df["ma_short"] > df["ma_long"]
        return (above.shift(1, fill_value=False)
                & above.shift(2, fill_value=False)
                & ~above.shift(3, fill_value=False))

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """出場：下穿後連 2 日確認（鏡像進場）。"""
        below = df["ma_short"] < df["ma_long"]
        return (below.shift(1, fill_value=False)
                & below.shift(2, fill_value=False)
                & ~below.shift(3, fill_value=False))

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        """成交價 = 訊號 bar 開盤（tick 進位由基底 _postprocess 處理）。"""
        return df["open"]


# 兩組常用參數（對應原 DualMA_20_50 / DualMA_50_200）
class MACross_20_50(MACrossStrategy):
    SHORT_MA, LONG_MA = 20, 50


class MACross_50_200(MACrossStrategy):
    SHORT_MA, LONG_MA = 50, 200


def _load_prices(parquet_path: str) -> pd.DataFrame:
    """讀單檔 OHLCV parquet（_01_data 下載產生）。index 須為 DatetimeIndex。"""
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index 必須是 DatetimeIndex（日期）")
    return df.sort_index()


def main(argv) -> int:
    """CLI：傳入 parquet 路徑，對兩組參數各跑一次並印 summary。"""
    if len(argv) < 2:
        print("用法: python ma_cross_strategy.py <OHLCV parquet 路徑>")
        return 1
    df = _load_prices(argv[1])
    for cls in (MACross_20_50, MACross_50_200):
        res = cls(initial_cash=100_000, split_cash=10_000).run(df, stock_id="")
        print(f"[{cls.__name__}] {res['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
