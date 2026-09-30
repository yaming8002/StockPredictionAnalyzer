"""
MACD（十）：拼裝整合矩陣（3 基礎 × 5 進場 × 5 出場 ＝ 75 組）
================================================================
兩軸都按「排序取前五」篩，**基本版（無濾網／原生出場）不進矩陣**——（四）篇已經有
那組資料，這裡只拿它當比較基礎。

**進場（九條取五）**：按「原生出場下的三母體平均獲利因子」排序——創250日新高、
均線多頭排列(1.1627)、ADX>25(1.1424)、收盤>MA200(1.1271)、RSI<50且上升(1.1220)。
落選：跳空(1.1083)、放量1.5倍(1.0824)、柱狀圖連兩根遞增(1.0686)、CMF>0(1.0038)。
⚠️ **創250日新高在背離母體結構性 0 筆**（純背離要收盤創 20 日新低、創年度新高要收盤是
250 日最高，兩條件互斥），那五格是空的、不是漏跑，它的平均也不可與三母體組並列。

**出場（（九）篇八條取五）**：跌破年線(1.4673)、超級趨勢(1.2099)、抱滿60天(1.1670)、
跌破二十日低(1.1559)、波段高點走低(1.1095)。落選：自最高點回落10%(1.0786)、
吊燈3ATR(1.0583)、SAR(0.9786)——三母體平均皆 < 1.08。

出場一律用**取代**接法（原生出場整條拿掉），與（九）篇替換版同口徑。

執行：python _04_analysis/macd/macd_combo.py [--limit 300]
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _02_strategy.macd_strategy.macd_variants import (NAME_BASE, NAME_EXIT,
                                                      NAME_FILTER)
from _04_analysis.macd.macd_sweep import prepare, run_variant, spec_rows

OUT = os.path.join(_root, "_02_strategy", "macd_strategy", "result", "macd_combo")
BASES = ["cross", "zero", "div"]
# 依三母體平均獲利因子排序取前五；基本版（無濾網）不進矩陣
FILTERS = ["high250", "align", "adx25", "ma200", "rsi"]
# 同上，原生出場不進矩陣
EXITS = ["ma200", "supertrend", "time60", "donchian", "lowerhigh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    n = len(BASES) * len(FILTERS) * len(EXITS)
    print(f"載入並備妥 {len(data)} 檔｜{n} 組｜{time.time() - t0:.0f} 秒", flush=True)

    out = []
    for base in BASES:
        for f in FILTERS:
            for ex in EXITS:
                t = time.time()
                s = run_variant(data, base, f, ex, "replace")
                out.append(({"母體": NAME_BASE[base], "進場": NAME_FILTER[f],
                             "出場": NAME_EXIT[ex]}, s))
                print(f"  {NAME_BASE[base]} × {NAME_FILTER[f]} × {NAME_EXIT[ex]}："
                      f"{s['交易次數']:,} 筆｜PF {s['獲利因子(PF)']}｜"
                      f"未平倉 {s['未平倉%']}%｜{time.time() - t:.0f} 秒", flush=True)

    df = spec_rows(out)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_combo.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(df)} 列 → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
