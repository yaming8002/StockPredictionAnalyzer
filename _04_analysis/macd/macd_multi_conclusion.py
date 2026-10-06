"""
MACD 系列結論：五組交易策略的多股回測（2015–2025）對 0050 買進持有含息（分析段）
==============================================================================
讀回測段 `_03_multi_strategy/macd/macd_conclusion_equity.py` 存下的逐日權益曲線與成交統計，
算資金倍數／年化／最大回撤，再跟 0050 比。這支不跑回測。

**為什麼期間是 2015–2025，不是回測用的 2002–2025**：見 0050 基準模組
`_04_analysis/benchmark/benchmark_0050.py`（對照區間、含息算法、錨點都定義在那裡）。

**判定基準＝報酬回撤比（年化報酬率% ÷ 最大回撤%）**：單看年化會偏好把風險放大的
做法，單看回撤會偏好不交易，兩者相除才問得出「這套值不值得取代長抱 0050」。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/macd/macd_conclusion_equity.py
    python _04_analysis/macd/macd_multi_conclusion.py
輸出：result/macd_multi/macd_multi_conclusion.csv ＋ 終端表。
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _03_multi_strategy.macd.macd_conclusion_equity import (EQUITY_FILE, RUNS_FILE,  # noqa: E402
                                                            WIN_START, equity_key)
from _03_multi_strategy.macd.macd_multi_driver import INIT_CASH, OUT  # noqa: E402
from _04_analysis.benchmark.benchmark_0050 import (BENCHMARK_START, bench_0050,  # noqa: E402
                                                   curve_stats)


def main():
    # 回測段的下單起點必須等於 0050 對照區間起點，否則兩邊比的不是同一段期間
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段下單起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}，"
                         "兩邊期間不一致，請先對齊。")
    for path in (EQUITY_FILE, RUNS_FILE):
        if not os.path.isfile(path):
            raise SystemExit(f"找不到回測輸出：{path}\n"
                             "先跑回測段：python _03_multi_strategy/macd/macd_conclusion_equity.py")
    equity = pd.read_parquet(EQUITY_FILE)
    runs = pd.read_csv(RUNS_FILE, encoding="utf-8-sig")
    cal = equity.index
    years = (cal[-1] - cal[0]).days / 365.25

    rows = []
    for r in runs.itertuples(index=False):
        eq = equity[equity_key(r.交易策略, r.投法)].to_numpy(np.float64)
        mult, cagr, dd = curve_stats(eq, years)
        rows.append({"交易策略": r.交易策略, "投法": r.投法, "份數": r.份數,
                     "交易次數": r.交易次數, "擋單": r.擋單, "資金倍數": mult,
                     "年化報酬率%": cagr, "最大回撤%": dd,
                     "報酬回撤比": round(cagr / dd, 3) if dd > 0 else None})
        print(f"{r.交易策略}｜{r.投法}({r.份數})：{r.交易次數:,} 筆｜擋單 {r.擋單:,}｜"
              f"×{mult}｜年化 {cagr}%｜回撤 {dd}%｜報酬回撤比 {rows[-1]['報酬回撤比']}")

    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    rows.append({"交易策略": "0050 買進持有（含息）", "投法": "—", "份數": None,
                 "交易次數": 1, "擋單": 0, "資金倍數": mult, "年化報酬率%": cagr,
                 "最大回撤%": dd, "報酬回撤比": round(cagr / dd, 3)})
    print(f"\n0050 買進持有（含息）：×{mult}｜年化 {cagr}%｜回撤 {dd}%｜"
          f"報酬回撤比 {rows[-1]['報酬回撤比']}")
    print("（錨點：×5.51／16.8%／33.8%；對不上就是含息或分割校準有問題）")

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_multi_conclusion.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n→ {path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
