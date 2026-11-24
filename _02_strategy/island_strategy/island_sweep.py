"""
底部島型 baseline 全組合掃描（vbt 單股）：持有 5／10／20 根 × 裸版／可成交 1,000 萬＝6 組。

每組各自呼叫 single_island_strategy.run_config（同一支策略檔、同一個 vbt 引擎），
以多進程平行（每組一個進程、彼此獨立），結果寫 result/single_island/<tag>/，
彙總寫 result/single_island/_sweep_summary.csv。

母體：data_anomaly.island_exclude()（2026-11-23 起，2,143 檔）。

執行：python _02_strategy/island_strategy/island_sweep.py [--workers 4] [--limit N] [--out 目錄]
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
from _02_strategy.island_strategy.data_anomaly import island_exclude
from _02_strategy.island_strategy.single_island_strategy import RESULT_DIR, run_config

HOLDS = (5, 10, 20)


def _job(args):
    hold, liq, limit, out_root, extra = args
    t0 = time.time()
    res = run_config(common.DATA_DIR, hold, liq, limit=limit, out_root=out_root, extra_exclude=extra)
    res["秒"] = round(time.time() - t0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="輸出根目錄（預設 result/；冒煙測試請指到暫存資料夾）")
    a = ap.parse_args()
    extra = island_exclude()
    print(f"島型母體排除 {len(extra)} 檔", flush=True)
    jobs = [(h, liq, a.limit, a.out, extra) for h in HOLDS for liq in (False, True)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜{r['秒']} 秒", flush=True)
    out = os.path.join(a.out or RESULT_DIR, "single_island", "_sweep_summary.csv")
    pd.DataFrame(rows).sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
