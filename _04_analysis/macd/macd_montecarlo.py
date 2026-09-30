"""
MACD（十）：五組交易策略的蒙地卡羅壓測
=========================================
口徑與均線、KD 兩個系列一致：

- **交易頻率基準**：每個交易日約 3–4 筆，乘上回測期間的交易日數得「部署總筆數區間 T」。
  2002–2025 約 5,949 個交易日 → T ∈ [17,847, 23,796]。
- **抽樣**：歷史筆數 N ≥ 區間下限 → 每條路徑先隨機抽一個 T，再有放回抽 T 筆；
  N < 下限 → T = N（全部交易納入），一樣有放回重抽，不硬灌到區間、避免虛增筆數。
- 每組跑 10,000 條路徑。
- 帳戶：起始 100 萬、定額累加（每筆固定金額、不複利）。
- **破產**：權益跌破 50 萬（本金的五成）。本金腰斬多半已經撐不住，這條線設早一點當警戒。

五組的挑法（2026-09-27 改版）：**每一組都必須達到抽樣下限 T_LOW**，否則模擬出來的是
「一個規模更小的操作」而不是策略本身的性格。在這個前提下，每個基礎取排名最前的組合
（黃金交叉前段班格子多，取前兩組），再加「無濾網」當基本版對照。
舊版六組裡有兩組不符、已移除：背離 × ADX>25（16,416 筆，排行第 9）改用排名次前但過得
了下限的 RSI<50；零軸 × 創250日新高（635 筆）全表獲利因子最高，但離下限差二十幾倍。

這張表的「最大連敗 P95」就是後續多股回測算份數用的 S（見 macd_multi_driver）。

執行：python _04_analysis/macd/macd_montecarlo.py [--limit 300]
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from _04_analysis.analyze_vbt import monte_carlo
from _04_analysis.macd.macd_sweep import prepare, variant_trades

OUT = os.path.join(_root, "_02_strategy", "macd_strategy", "result", "macd_mc")
PATHS = 10_000
INIT_CASH = 1_000_000.0
RUIN_RATIO = 0.5              # 破產＝權益跌破本金五成
TRADING_DAYS = 5_949          # 2002-01-01 ~ 2025-12-31 的交易日數
T_LOW, T_HIGH = TRADING_DAYS * 3, TRADING_DAYS * 4

# (顯示名稱, 基礎, 濾網, 出場)；出場一律「跌破年線」取代原生出場
CASES = [
    ("交叉 × 均線多頭排列 × 跌破年線", "cross", "align", "ma200"),
    ("交叉 × ADX>25 × 跌破年線", "cross", "adx25", "ma200"),
    ("交叉 × 無濾網 × 跌破年線", "cross", "none", "ma200"),
    ("零軸 × 收盤>MA200 × 跌破年線", "zero", "ma200", "ma200"),
    ("背離 × RSI<50且上升 × 跌破年線", "div", "rsi", "ma200"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{len(CASES)} 組 × {PATHS:,} 條路徑｜"
          f"T ∈ [{T_LOW:,}, {T_HIGH:,}]｜{time.time() - t0:.0f} 秒", flush=True)

    rows = []
    for label, base, filt, ex in CASES:
        trades, summary = variant_trades(data, base, filt, ex, "replace")
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=PATHS,
                         ruin_ratio=RUIN_RATIO, t_low=T_LOW, t_high=T_HIGH)
        rows.append({"交易策略": label, "歷史筆數": mc["歷史筆數"],
                     "抽樣筆數": mc["每次抽樣筆數"], **{
                         k: mc[k] for k in mc if k not in
                         ("模擬次數", "歷史筆數", "每次抽樣筆數")}})
        print(f"  {label}：歷史 {mc['歷史筆數']:,} 筆｜"
              f"最大連敗 P95 {mc['最大連敗_P95']}｜"
              f"最大回撤%中位 {mc['最大回撤%_中位']}｜"
              f"破產 {mc[f'破產機率(<{RUIN_RATIO:.0%})']}%｜"
              f"{time.time() - t0:.0f} 秒", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_montecarlo.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
