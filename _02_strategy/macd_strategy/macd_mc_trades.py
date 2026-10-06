"""
MACD（十）蒙地卡羅的回測段：五組交易策略的單股全市場逐筆交易
================================================================
只負責「跑回測、存逐筆交易」；蒙地卡羅在分析段 `_04_analysis/macd/macd_montecarlo.py`，
讀這裡的輸出。拆兩段是為了分層：_02 只做單股回測、_04 只做分析。
逐筆交易存 parquet（浮點數不失真），分析段固定種子重跑蒙地卡羅才會與合併版逐位一致。

五組的挑法（2026-09-27 改版）：**每一組都必須達到抽樣下限 T_LOW**（見分析段），否則
模擬出來的是「一個規模更小的操作」而不是策略本身的性格。在這個前提下，每個基礎取排名
最前的組合（黃金交叉前段班格子多，取前兩組），再加「無濾網」當基本版對照。
舊版六組裡有兩組不符、已移除：背離 × ADX>25（16,416 筆，排行第 9）改用排名次前但過得
了下限的 RSI<50；零軸 × 創250日新高（635 筆）全表獲利因子最高，但離下限差二十幾倍。

執行：python _02_strategy/macd_strategy/macd_mc_trades.py [--limit 300]
輸出：result/macd_mc/trades_<基礎>_<濾網>_<出場>.parquet（每組一檔）
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.macd_strategy.macd_sweep import prepare, variant_trades  # noqa: E402

OUT = common.result_dir("macd_strategy", "macd_mc")

# (顯示名稱, 基礎, 濾網, 出場)；出場一律「跌破年線」取代原生出場
CASES = [
    ("交叉 × 均線多頭排列 × 跌破年線", "cross", "align", "ma200"),
    ("交叉 × ADX>25 × 跌破年線", "cross", "adx25", "ma200"),
    ("交叉 × 無濾網 × 跌破年線", "cross", "none", "ma200"),
    ("零軸 × 收盤>MA200 × 跌破年線", "zero", "ma200", "ma200"),
    ("背離 × RSI<50且上升 × 跌破年線", "div", "rsi", "ma200"),
]


def trades_path(base: str, filt: str, exit_name: str) -> str:
    """某一組的逐筆交易檔路徑（回測段寫、分析段讀，兩邊共用這一支）。"""
    return os.path.join(OUT, f"trades_{base}_{filt}_{exit_name}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{len(CASES)} 組｜{time.time() - t0:.0f} 秒", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for label, base, filt, ex in CASES:
        trades, _ = variant_trades(data, base, filt, ex, "replace")
        path = trades_path(base, filt, ex)
        trades.to_parquet(path, index=False)
        print(f"  {label}：{len(trades):,} 筆 → {path}｜{time.time() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
