"""
〈島型反轉（一）〉正文的輔助統計（不跑回測，只讀結果檔）：
持有天數分布（日曆天、拆成「碰到休市」與「個股有幾天沒成交」）、訊號／進場數、島根數分布、未平倉率。

日曆對照用 2330.TW 的交易日：持有期間大盤交易日數 > N 根＝個股有沒成交的日子；否則多出來的天數是休市。

執行：python _04_analysis/island/article/basic_stats.py <持有 N 根結果目錄> <頂部島型結果目錄>
  例：result/official_20261123/single_island  result/official_20261123/single_island_exit
"""
import json
import os
import sys

import numpy as np
import pandas as pd

CAL_STOCK = os.path.join(os.environ.get("STOCK_DATA_DIR", "F:/stock-analyzer/data/stock_data"), "2330.TW.parquet")


def load(folder: str, tag: str):
    """讀一組的逐筆交易與 diag.json。"""
    d = os.path.join(folder, tag)
    trades = pd.read_csv(os.path.join(d, f"single_island_{tag}_trades.csv"), parse_dates=["buy_date", "sell_date"])
    with open(os.path.join(d, "diag.json"), encoding="utf-8") as f:
        return trades, json.load(f)


def diag_text(dg: dict) -> str:
    return (f"signals {dg['signals']:,} entries {dg['entries']:,} "
            f"open {dg['open_at_end']} ({100 * dg['open_at_end'] / dg['entries']:.2f}%)")


def hold_stats(folder: str, tag: str, hold: int, cal: pd.DatetimeIndex) -> None:
    trades, dg = load(folder, tag)
    days = (trades.sell_date - trades.buy_date).dt.days
    mkt = np.searchsorted(cal, trades.sell_date.values) - np.searchsorted(cal, trades.buy_date.values)
    skip = pd.Series(mkt > hold, index=trades.index)
    normal, cut = (14, 16) if hold == 10 else (28, 31)
    long_ = days > cut
    print(f"[{tag}] n={len(trades):,} 平均 {days.mean():.1f} 中位 {days.median():.0f} 眾數 {days.mode()[0]} "
          f"｜={normal}天 {100 * (days == normal).mean():.1f}% ≤{cut}天 {100 * (days <= cut).mean():.1f}% "
          f">{cut}天 {100 * long_.mean():.1f}%（其中個股沒成交 {100 * skip[long_].mean():.1f}%、"
          f"純休市 {100 * (~skip[long_]).mean():.1f}%）最長 {days.max()} 天｜{diag_text(dg)}")
    if tag == "h20_raw":
        ks = {int(k[1:]): v for k, v in dg.items() if k.startswith("k") and k[1:].isdigit()}
        tot = sum(ks.values())
        print("  島根數：" + "、".join(f"{k} 根 {100 * v / tot:.1f}%" for k, v in sorted(ks.items()))
              + f"｜4～10 根合計 {100 * sum(v for k, v in ks.items() if k >= 4) / tot:.1f}%")


def top_stats(folder: str, tag: str) -> None:
    trades, dg = load(folder, tag)
    days = (trades.sell_date - trades.buy_date).dt.days
    print(f"[{tag}] n={len(trades):,} 平均 {days.mean():.0f} 中位 {days.median():.0f} "
          f"≤14天 {100 * (days <= 14).mean():.1f}% >365天 {100 * (days > 365).mean():.1f}% "
          f"最長 {days.max() / 365.25:.1f} 年｜{diag_text(dg)}")


def main() -> int:
    hold_dir, top_dir = sys.argv[1], sys.argv[2]
    idx = pd.read_parquet(CAL_STOCK).index
    cal = pd.DatetimeIndex(idx).normalize()
    for tag, hold in (("h10_raw", 10), ("h20_raw", 20), ("h10_liq1000", 10), ("h20_liq1000", 20)):
        hold_stats(hold_dir, tag, hold, cal)
    for tag in ("rep__none__topisland_raw", "rep__none__topisland_liq1000"):
        top_stats(top_dir, tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
