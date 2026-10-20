"""
雙底進場優化掃描（vbt 單股）：ZigZag 3%／6%／9% × (基本版＋6 條進場條件) × 裸版／可成交 1,000 萬＝42 組
==================================================================================================
出場固定基本版（高點下移 或 收盤跌破最近確認波段低點）。每條條件單獨疊在基本版進場上，條件式見
double_bottom_variants.py（逐字抄策略檔 # 優化 #2～#7）。基本版同批次重跑，當同一批的比較基準。

診斷（diag.json）：signals＝區間內進場判定日數；生效率＝1 − 該條件 signals ÷ 同門檻同口徑基本版 signals
（被擋掉的判定比例；頸線突破是「沒有突破就作廢」的比例）；另有期末未平倉。

執行：python _02_strategy/double_bottom_strategy/entry_filter_sweep.py [--workers 4] [--limit N] [--out <根目錄>]
輸出：result/single_double_bottom_entry/p<門檻>_<raw|liq1000>__<變體>/、_sweep_summary.csv
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import batch, common
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH
from _02_strategy.double_bottom_strategy.double_bottom_variants import ENTRY_FILTERS, make_variant
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR

PCTS = (0.03, 0.06, 0.09)
SUB = "single_double_bottom_entry"


def _job(args):
    pct, liq, entry, limit, out_root = args
    t0 = time.time()
    strat = make_variant(entry, pct, liq)
    result = batch.run_folder(strat, common.DATA_DIR, start=DEFAULT_START, end=DEFAULT_END,
                              limit=limit, exclude=GLITCH)
    tag = f"p{int(round(pct * 100))}_{'liq1000' if liq else 'raw'}__{entry}"
    out_dir = os.path.join(out_root, SUB, tag)
    batch.write_results(result, out_dir, f"single_double_bottom_{tag}", write_trades=True)
    with open(os.path.join(out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump(strat.diag, f, ensure_ascii=False, indent=1)
    return {"tag": tag, "門檻%": int(round(pct * 100)), "可成交": liq, "變體": entry,
            **result["aggregate"], **strat.diag, "failed": len(result["failed"]),
            "秒": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=RESULT_DIR)
    a = ap.parse_args()
    jobs = [(p, liq, e, a.limit, a.out) for p in PCTS for liq in (False, True) for e in ENTRY_FILTERS]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜訊號 {r['signals']:,}｜{r['秒']} 秒",
                  flush=True)
    df = pd.DataFrame(rows)
    base = df[df["變體"] == "base"].set_index(["門檻%", "可成交"])["signals"]
    df["生效率%"] = [round((1 - r["signals"] / base.loc[(r["門檻%"], r["可成交"])]) * 100, 2)
                    for _, r in df.iterrows()]
    df["未平倉率%"] = (df["open_at_end"] / df["entries"] * 100).round(2)
    out = os.path.join(a.out, SUB, "_sweep_summary.csv")
    df.sort_values(["可成交", "門檻%", "變體"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
