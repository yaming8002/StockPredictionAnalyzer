"""
長黑出量後突破全組合掃描（vbt 單股）：CHoCH 的 ZigZag 2%／4%／6%／8% × 裸版／可成交 1,000 萬＝8 組。

每組各自呼叫 single_long_black_strategy.run_config，以多進程平行（每組一個進程、彼此獨立），
結果寫 result/single_long_black[_<variant>]/<tag>/，彙總寫同資料夾 _sweep_summary.csv。

執行：python _02_strategy/long_black_strategy/long_black_sweep.py [--workers 4] [--limit N] [--out 目錄]
                                                                  [--variant opt1] [--raw-only]
  --variant 只決定輸出資料夾，須與策略檔註解切換狀態一致；--raw-only 只跑裸版（不跑可成交版）。
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
from _02_strategy.long_black_strategy.single_long_black_strategy import RESULT_DIR, run_config

PCTS = (0.02, 0.04, 0.06, 0.08)   # 2026-11-06 用戶指定


def _job(args):
    pct, liq, limit, out_root, variant = args
    t0 = time.time()
    res = run_config(common.DATA_DIR, pct, liq, limit=limit, out_root=out_root, variant=variant)
    res["秒"] = round(time.time() - t0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="輸出根目錄（預設 result/；冒煙請指到暫存資料夾）")
    ap.add_argument("--variant", default=None, help="輸出資料夾後綴（例：opt1），須與註解狀態一致")
    ap.add_argument("--raw-only", action="store_true", help="只跑裸版，不跑可成交版")
    a = ap.parse_args()
    liqs = (False,) if a.raw_only else (False, True)
    jobs = [(p, liq, a.limit, a.out, a.variant) for p in PCTS for liq in liqs]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜{r['秒']} 秒", flush=True)
    sub = f"single_long_black_{a.variant}" if a.variant else "single_long_black"
    out = os.path.join(a.out or RESULT_DIR, sub, "_sweep_summary.csv")
    pd.DataFrame(rows).sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
