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

**2026-10-08 起改跑 3 × 7 × 6 ＝ 126 組**（第十篇實際用的那張 6x6 寬表）：兩軸各補上基準
（無濾網、原生出場），進場另補「跳空」（排序第六、文章附錄有列）。矩陣與排行仍只取上面的前五；
基準列給「同列差距」與 MC 挑組用。原本這張表是 blog 私有 driver 跑的，這裡收回 SPA，欄名與名稱
沿用那份 CSV（母體＝交叉／零軸／背離；出場＝跌破MA200、Supertrend翻空、跌破20日低、頂頂低…），
文章的出表與驗證程式（_04_analysis/macd/article/）直接讀這份。

執行：python _02_strategy/macd_strategy/macd_combo.py [--limit 300] [--out 目錄]
輸出：result/macd_combo/macd_combo_6x6.csv
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _02_strategy.base.vbt import common
from _02_strategy.macd_strategy.macd_variants import (NAME_BASE, NAME_EXIT,
                                                      NAME_FILTER)
from _02_strategy.macd_strategy.macd_sweep import prepare, run_variant, spec_rows

OUT = common.result_dir("macd_strategy", "macd_combo")
BASES = ["cross", "zero", "div"]
# 依三母體平均獲利因子排序取前五；基本版（無濾網）不進矩陣
FILTERS = ["high250", "align", "adx25", "ma200", "rsi"]
# 同上，原生出場不進矩陣
EXITS = ["ma200", "supertrend", "time60", "donchian", "lowerhigh"]
# 寬表多跑的：兩軸基準＋排序第六的跳空（見檔頭）
GRID_FILTERS = ["none", "ma200", "gap", "align", "adx25", "rsi", "high250"]
GRID_EXITS = ["native", "ma200", "supertrend", "time60", "donchian", "lowerhigh"]
# 寬表沿用的名稱（第十篇出表／驗證程式認這套）
POP_NAME = {"cross": "交叉", "zero": "零軸", "div": "背離"}
EXIT_NAME = {"native": "原生出場", "ma200": "跌破MA200", "supertrend": "Supertrend翻空",
             "time60": "抱滿60天", "donchian": "跌破20日低", "lowerhigh": "頂頂低"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用）")
    ap.add_argument("--out", default=OUT, help="輸出目錄（冒煙用，避免蓋掉正式結果）")
    a = ap.parse_args()

    t0 = time.time()
    data = prepare(limit=a.limit)
    n = len(BASES) * len(GRID_FILTERS) * len(GRID_EXITS)
    print(f"載入並備妥 {len(data)} 檔｜{n} 組｜{time.time() - t0:.0f} 秒", flush=True)

    out = []
    for base in BASES:
        for f in GRID_FILTERS:
            for ex in GRID_EXITS:
                t = time.time()
                s = run_variant(data, base, f, ex, "replace")
                out.append(({"母體": POP_NAME[base], "進場濾網": NAME_FILTER[f],
                             "出場": EXIT_NAME[ex]}, s))
                print(f"  {NAME_BASE[base]} × {NAME_FILTER[f]} × {NAME_EXIT[ex]}："
                      f"{s['交易次數']:,} 筆｜PF {s['獲利因子(PF)']}｜"
                      f"未平倉 {s['未平倉%']}%｜{time.time() - t:.0f} 秒", flush=True)

    df = spec_rows(out)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "macd_combo_6x6.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒｜{len(df)} 列 → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
