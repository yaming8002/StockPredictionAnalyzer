"""
海龜 baseline 掃描（vbt 單股）：系統 1／系統 2 × 裸版／可成交 1,000 萬＝4 組，多進程平行。

可成交版＝策略檔「# 優化 #1」那行（成交金額 5 日均量×股價 > 1,000 萬）。掃描時不去改策略檔的
註解狀態，而是用子類只在 buy_signal 多加這一條，條件與策略檔那行逐字相同。
結果寫 result/single_turtle/s<系統>_<baseline|opt1>/，彙總 result/single_turtle/_sweep_summary.csv。

執行：python _02_strategy/turtle_strategy/turtle_sweep.py [--workers 4] [--limit N]
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

from _02_strategy.base.vbt import batch, common
from _02_strategy.base.vbt.common import GLITCH, DEFAULT_START, DEFAULT_END
from _02_strategy.turtle_strategy.single_turtle_strategy import (
    RESULT_DIR, SYSTEMS, TURNOVER_MIN, SingleTurtleStrategy)


class _TurtleLiquidity(SingleTurtleStrategy):
    """可成交版：buy_signal 加上策略檔「# 優化 #1」那行的條件。"""

    def buy_signal(self, df):
        signal = super().buy_signal(df)
        return (signal & (df["turnover"] > TURNOVER_MIN)).fillna(False).astype(bool)


def _job(args):
    system, liq, limit = args
    t0 = time.time()
    ew, xw = SYSTEMS[system]
    cls = _TurtleLiquidity if liq else SingleTurtleStrategy
    strat = cls(entry_window=ew, exit_window=xw)
    result = batch.run_folder(strat, common.DATA_DIR, start=DEFAULT_START, end=DEFAULT_END,
                              limit=limit, exclude=GLITCH)
    variant = "opt1" if liq else "baseline"
    out_dir = os.path.join(RESULT_DIR, "single_turtle", f"s{system}_{variant}")
    batch.write_results(result, out_dir, f"single_turtle_s{system}", write_trades=True)
    return {"tag": f"s{system}_{variant}", **result["aggregate"],
            "failed": len(result["failed"]), "秒": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = [(s, liq, a.limit) for s in sorted(SYSTEMS) for liq in (False, True)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            rows.append(r)
            print(f"{r['tag']}｜{r['交易次數']:,} 筆｜PF {r['獲利因子(PF)']}｜{r['秒']} 秒", flush=True)
    out = os.path.join(RESULT_DIR, "single_turtle", "_sweep_summary.csv")
    pd.DataFrame(rows).sort_values("tag").to_csv(out, index=False, encoding="utf-8-sig")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
