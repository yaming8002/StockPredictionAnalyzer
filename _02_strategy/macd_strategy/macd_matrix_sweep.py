"""
MACD（一）～（四）：三基礎基準線（無門檻）＋ 3 進場 × 3 出場矩陣（可成交）
==========================================================================
重建 2026-08-25 那輪遺失的 scratchpad driver，產出兩張表：

1. `_baselines.csv`：（一）（二）（三）篇第一張表——三個基礎各配自己的原生出場，
   **不開流動性門檻**（初探篇的「裸版」口徑；與後面所有可成交口徑的數字不可並列）。
2. `_matrix_3x3.csv`：（四）篇——進場 黃金交叉／DIF 上穿 0／底背離 × 出場 死叉／
   DIF 下穿 0／頂背離，一律**取代**接法（出場整條換成指定那一條），可成交門檻只 gate 進場。
   對角線以外的六格就是「進場與出場拆開重組」；矩陣的三格同時也是三篇的可成交基準線
   （交叉進×交叉出、零軸進×零軸出、背離進×交叉出），交叉進×零軸出＝第四個母體 mix。

9 格同一次執行（含三篇基準線那三格）——基準與新變體同次產出才可並排。

執行：
    python _02_strategy/macd_strategy/macd_matrix_sweep.py [--limit 30] [--out 目錄]
輸出：預設 result/single_macd/（--out 可改，冒煙時別覆蓋正式結果）
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _02_strategy.macd_strategy.macd_sweep import (  # noqa: E402
    legacy_row, prepare, variant_trades, write_csv)
from _02_strategy.macd_strategy.single_macd_strategy import RESULT_DIR  # noqa: E402

OUT = os.path.join(RESULT_DIR, "single_macd")

# 三篇基準線（無門檻）：(表格標籤, 基礎)；出場一律原生
BASELINES = [
    ("交叉（黃金交叉進、死叉出）", "cross"),
    ("零軸（上穿 0 進、下穿 0 出）", "zero"),
    ("背離（底背離進、死叉出）", "div"),
]

# 矩陣兩軸：(代碼, 表格用簡稱)
ENTRIES = [("cross", "交叉進"), ("zero", "零軸進"), ("div", "背離進")]
EXITS = [("death", "交叉出"), ("zero_down", "零軸出"), ("beardiv", "背離出")]


def _report(label: str, s: dict, t: float) -> None:
    print(f"  {label}：{s['交易次數']:,} 筆｜PF {s['獲利因子(PF)']}｜"
          f"未平倉 {s['未平倉%']}%｜{time.time() - t:.0f} 秒", flush=True)


def run_baselines(data: dict) -> list:
    """三基礎 × 原生出場，無流動性門檻。"""
    rows = []
    for label, base in BASELINES:
        t = time.time()
        _, s = variant_trades(data, base, "none", "native", "replace", liquidity=False)
        rows.append(legacy_row({"變體": label}, s, len(data)))
        _report(label, s, t)
    return rows


def run_matrix(data: dict) -> list:
    """3 進場 × 3 出場（取代），可成交門檻開。"""
    rows = []
    for base, en in ENTRIES:
        for ex, xn in EXITS:
            t = time.time()
            _, s = variant_trades(data, base, "none", ex, "replace")
            label = f"{en} × {xn}"
            rows.append(legacy_row({"組合": label}, s, len(data)))
            _report(label, s, t)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="MACD 三基礎基準線＋3×3 矩陣")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    ap.add_argument("--out", default=OUT, help=f"輸出目錄（預設 {OUT}）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)

    print("【三篇基準線（無門檻）】", flush=True)
    p1 = write_csv(run_baselines(data), a.out, "_baselines.csv")
    print("【3×3 矩陣（可成交、取代）】", flush=True)
    p2 = write_csv(run_matrix(data), a.out, "_matrix_3x3.csv")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {p1}\n{' ' * 10}→ {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
