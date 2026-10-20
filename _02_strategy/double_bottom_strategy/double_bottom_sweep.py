"""
雙底 baseline 全組合掃描（vbt 單股）：ZigZag 2%~10% × 裸版／可成交 1,000 萬＝18 組。

每組各自呼叫 single_double_bottom_strategy.run_config（同一支策略檔、同一個 vbt 引擎），
以多進程平行（每組一個進程、彼此獨立），結果寫 result/single_double_bottom/<tag>/（舊版含雙頂的 16 組保留在 single_double_bottom_v1_dtop/），
彙總寫 result/single_double_bottom/_sweep_summary.csv。

執行：python _02_strategy/double_bottom_strategy/double_bottom_sweep.py [--workers 4] [--limit N] [--nostop]
（--nostop＝優化 #1 不設停損，結果寫 result/single_double_bottom_nostop/）
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import (
    RESULT_DIR, run_config)

PCTS = tuple(p / 100 for p in range(2, 11))   # 2%～10%


def _job(args):
    pct, liq, limit, nostop, out_root = args
    t0 = time.time()
    res = run_config(common.DATA_DIR, pct, liq, limit=limit, nostop=nostop, out_root=out_root)
    res["秒"] = round(time.time() - t0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nostop", action="store_true", help="優化 #1 不設停損（只用 CHoCH 出場）")
    ap.add_argument("--out", default=None, help="輸出根目錄（預設 result/；重跑對帳請指到暫存資料夾）")
    a = ap.parse_args()
    jobs = [(p, liq, a.limit, a.nostop, a.out) for p in PCTS for liq in (False, True)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜{r['秒']} 秒", flush=True)
    out = os.path.join(a.out or RESULT_DIR, "single_double_bottom_nostop" if a.nostop else "single_double_bottom",
                       "_sweep_summary.csv")
    pd.DataFrame(rows).sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
