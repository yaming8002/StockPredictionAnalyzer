"""
多股雙均線交叉（共用資金）
==========================
把 _02 的雙均線交叉訊號（黃金/死亡交叉）搬到「單一本金、多檔共用現金」的組合回測：
繼承 `VbtMultiStrategy`，沿用 base 均線交叉的判定，差別在所有股票共用一個現金池、
依倉位模式分配資金、現金不足時真擋單。執行時序與 _02 一致：**收盤判定 → 隔日開盤成交**。

進場：短均線上穿長均線（黃金交叉）。出場：短均線下穿長均線（死亡交叉）。

執行（全市場、共用 100 萬、percent_floor、每筆佔已實現權益 1/30、下限 1 萬）：
  python _03_multi_strategy/ma_cross/multi_ma_cross.py <資料夾> --short 50 --long 200
"""
import argparse
import glob
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _01_data.indicators_trend import calculate_sma
from _03_multi_strategy.base.vbt.multi import VbtMultiStrategy

# 與 ma_cross 研究一致：價格 glitch 壞資料股事前剔除
GLITCH = {"3591.TW", "8039.TW", "8027.TWO", "6283.TW", "3666.TWO"}
DEFAULT_DATA = r"F:\stock-analyzer\data\stock_data"


def _ma(df: pd.DataFrame, window: int) -> pd.Series:
    """取 sma_{window}：優先用 parquet 既有欄位（全史 warmup），無則用 _01_data 補。"""
    col = f"sma_{window}"
    if col not in df.columns:
        calculate_sma(df, window)
    return df[col]


class MultiMACross(VbtMultiStrategy):
    """多股雙均線交叉。SHORT_MA < LONG_MA；隔日開盤成交。"""

    SHORT_MA = 50
    LONG_MA = 200

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_above"] = _ma(df, self.SHORT_MA) > _ma(df, self.LONG_MA)
        return df

    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        above = df["_above"]
        return above & ~above.shift(1, fill_value=False)   # 黃金交叉（上穿）

    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        above = df["_above"]
        return (~above) & above.shift(1, fill_value=False)  # 死亡交叉（下穿）

    def build_signals(self, df: pd.DataFrame):
        # 收盤判定 → 隔日成交：訊號位移 +1（無 look-ahead）
        entries = self.buy_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        exits = self.sell_signal(df).fillna(False).astype(bool).shift(1, fill_value=False)
        return entries, exits

    def exec_price(self, df: pd.DataFrame) -> pd.Series:
        return df["open"]   # 隔日開盤價成交


def load_data(folder: str, limit: int = None) -> dict:
    """讀資料夾 parquet 成 {stock_id: df}；只取需要的欄位、剔除 glitch 壞檔。"""
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    data = {}
    want = ["open", "high", "low", "close", "volume", "sma_50", "sma_200"]
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        if sid in GLITCH:
            continue
        try:
            cols = pd.read_parquet(f, columns=None).columns
            use = [c for c in want if c in cols]
            df = pd.read_parquet(f, columns=use)
        except Exception:
            continue
        if df.empty:
            continue
        data[sid] = df
        if limit and len(data) >= limit:
            break
    return data


def main(argv) -> int:
    p = argparse.ArgumentParser(description="多股雙均線交叉（共用資金）回測")
    p.add_argument("folder", nargs="?", default=DEFAULT_DATA, help="OHLCV parquet 資料夾")
    p.add_argument("--short", type=int, default=50)
    p.add_argument("--long", type=int, default=200)
    p.add_argument("--mode", choices=("percent_floor", "fixed"), default="percent_floor")
    p.add_argument("--ratio", type=float, default=1.0 / 30.0)
    p.add_argument("--min-invest", type=float, default=10_000.0)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--start", default="2001-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--limit", type=int, default=None, help="只讀前 N 檔（測試用）")
    args = p.parse_args(argv[1:])

    print(f"讀取資料夾 {args.folder} …")
    data = load_data(args.folder, limit=args.limit)
    print(f"載入 {len(data)} 檔（已剔除 glitch）。執行多股共用資金回測 "
          f"{args.short}/{args.long} mode={args.mode} …")

    strat = MultiMACross(sizing_mode=args.mode, invest_ratio=args.ratio,
                         min_invest=args.min_invest, initial_cash=args.cash)
    strat.SHORT_MA, strat.LONG_MA = args.short, args.long
    res = strat.run(data, start_date=args.start, end_date=args.end)

    s = res["summary"]
    print("\n=== 多股雙均線交叉（共用資金）結果 ===")
    print(f"短/長均線        : {args.short}/{args.long}")
    print(f"初始資金 / 模式   : {args.cash:,.0f} / {args.mode}"
          f"（ratio={args.ratio:.4f}, 下限={args.min_invest:,.0f}）")
    print(f"成交筆數         : {s['交易次數']}")
    print(f"擋單數(現金不足)  : {s.get('擋單數', 0)}")
    print(f"勝率(%)          : {s['勝率(%)']}")
    print(f"EV/筆            : {s['期望報酬值(EV)']:.0f}")
    print(f"PF / Trim PF     : {s.get('獲利因子(PF)')} / {s.get('排除極值後獲利因子(PF,Trim)')}")
    print(f"EV(Trim)         : {s['排除極值後期望報酬值(EV,Trim)']:.0f}")
    print(f"總獲利           : {s['總獲利']:,.0f}")
    print(f"平均持有天       : {s['平均持有天數']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
