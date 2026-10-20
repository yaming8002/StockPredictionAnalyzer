"""
新系列候選（海龜 vs 雙底）多股結果 → 每組最佳格的 2015–2025 vs 0050（分析段，不跑回測）
=====================================================================================
讀 `_03_multi_strategy/candidates/candidates_multi_driver.py` 的輸出：
  candidates_multi_result.csv  20 組 × 兩投法 × 三排序的全期（2002–2025）結果
  equity_2015/<key>.parquet    同 6 格的 2015–2025 逐日市值權益
每組挑「全期總獲利最高的投法×排序」，用它的 2015–2025 權益算資金倍數、年化、最大回撤、
報酬回撤比，與 0050 買進持有（含息，`_04_analysis/benchmark/benchmark_0050.py`）同一組交易日比。

執行：python _04_analysis/candidates/candidates_vs_0050.py
輸出：_02_strategy/double_bottom_strategy/result/candidates_multi/candidates_vs_0050.csv
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _03_multi_strategy.candidates.candidates_multi_driver import (INIT_CASH, OUT,  # noqa: E402
                                                                   RESULT_FILE, WIN_START)
from _04_analysis.benchmark.benchmark_0050 import BENCHMARK_START, bench_0050, curve_stats  # noqa: E402


def main() -> int:
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}")
    if not os.path.isfile(RESULT_FILE):
        raise SystemExit(f"找不到 {RESULT_FILE}，先跑 candidates_multi_driver.py")
    full = pd.read_csv(RESULT_FILE, encoding="utf-8-sig")
    best = full.loc[full.groupby("key", sort=False)["總獲利(萬)"].idxmax()]

    rows, cal = [], None
    for _, r in best.iterrows():
        eq = pd.read_parquet(os.path.join(OUT, "equity_2015", f"{r['key']}.parquet"))
        cal = eq.index if cal is None else cal
        if not eq.index.equals(cal):
            raise SystemExit(f"{r['key']} 的交易日曆與其他組不同，不能共用同一條 0050 基準")
        curve = eq[f"{r['投法']}｜{r['排序']}"].to_numpy(np.float64)
        years = (cal[-1] - cal[0]).days / 365.25
        mult, cagr, dd = curve_stats(curve, years)
        rows.append({"交易策略": r["交易策略"], "投法": r["投法"], "排序": r["排序"], "份數": r["份數"],
                     "全期總獲利(萬)": r["總獲利(萬)"], "資金倍數": mult, "年化報酬率%": cagr,
                     "最大回撤%": dd, "報酬回撤比": round(cagr / dd, 3) if dd > 0 else None})
    years = (cal[-1] - cal[0]).days / 365.25
    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    rows.append({"交易策略": "0050 買進持有（含息）", "投法": "—", "排序": "—", "份數": None,
                 "全期總獲利(萬)": None, "資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                 "報酬回撤比": round(cagr / dd, 3)})
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "candidates_vs_0050.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"（0050 錨點：×5.51／16.81%／33.8%）→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
