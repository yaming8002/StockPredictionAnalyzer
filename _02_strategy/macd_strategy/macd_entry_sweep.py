"""
MACD（五）（六）：九條進場濾網 × 四個母體（可成交口徑）
=========================================================
重建 2026-08-25 那輪（v2）遺失的 scratchpad driver，產出 `_entry_sweep_v2.csv`。

- **母體（4）**：交叉（黃金交叉進×死叉出）／零軸（上穿 0 進×下穿 0 出）／
  背離（底背離進×死叉出）／交叉進×零軸出（mix）。出場一律各母體的原生出場。
  mix 與交叉的進場一字不差、只差出場，用來檢驗「濾網效果會不會被出場牽動」。
- **九條濾網**：疊在基礎上（AND），一次一條；條件式見 macd_variants.buy_signal。
- 可成交門檻（成交金額 > 1,000 萬）只 gate 進場。
- **基準線不在這張表**：四個母體的無濾網基準線＝`_matrix_3x3.csv` 的
  交叉進×交叉出／零軸進×零軸出／背離進×交叉出／交叉進×零軸出（與舊表同一個安排）。
- ⚠️ 背離 × 創250日新高 結構性 0 筆（底背離要收盤創 20 日新低，與創 250 日新高互斥）。

執行：
    python _02_strategy/macd_strategy/macd_entry_sweep.py [--limit 30] [--out 目錄]
輸出：預設 result/single_macd/_entry_sweep_v2.csv
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
from _02_strategy.macd_strategy.macd_variants import NAME_FILTER  # noqa: E402
from _02_strategy.macd_strategy.single_macd_strategy import RESULT_DIR  # noqa: E402

OUT = os.path.join(RESULT_DIR, "single_macd")

# (代碼, 表格標籤)；順序沿用舊表
POPULATIONS = [("cross", "交叉"), ("zero", "零軸"), ("div", "背離"), ("mix", "交叉進×零軸出")]
FILTERS = ["adx25", "rsi", "volume", "hist_rising", "cmf", "ma200", "align", "high250", "gap"]


def run_sweep(data: dict) -> list:
    """9 濾網 × 4 母體，原生出場、可成交門檻開。"""
    rows = []
    for base, pop in POPULATIONS:
        for f in FILTERS:
            t = time.time()
            _, s = variant_trades(data, base, f, "native", "replace")
            rows.append(legacy_row({"母體": pop, "濾網": NAME_FILTER[f]}, s, len(data)))
            print(f"  {pop} × {NAME_FILTER[f]}：{s['交易次數']:,} 筆｜PF {s['獲利因子(PF)']}｜"
                  f"未平倉 {s['未平倉%']}%｜{time.time() - t:.0f} 秒", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="MACD 九條進場濾網 × 四母體")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    ap.add_argument("--out", default=OUT, help=f"輸出目錄（預設 {OUT}）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    print(f"載入並備妥 {len(data)} 檔｜{len(POPULATIONS) * len(FILTERS)} 組｜"
          f"{time.time() - t0:.0f} 秒", flush=True)
    path = write_csv(run_sweep(data), a.out, "_entry_sweep_v2.csv")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
