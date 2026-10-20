"""
雙底三組（比例投入 × 三排序）：純費率 vs 台股實際費稅，2015–2025 對 0050（分析段，不跑回測）
=========================================================================================
讀 `_03_multi_strategy/candidates/candidates_tw_driver.py` 的輸出（tw/tw_runs.csv、tw/equity_2015_tw.parquet），
每格算 2015–2025 的資金倍數、年化、最大回撤、報酬回撤比，並附全期（2002–2025）總獲利與資金倍數；
0050 用同一組交易日。海龜系統 2 的結果留在檔案裡，這裡不列（2026-10-16 起暫停海龜）。

執行：python _04_analysis/candidates/candidates_tw_vs_0050.py
輸出：_02_strategy/double_bottom_strategy/result/candidates_multi/tw/candidates_tw_vs_0050.csv
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _03_multi_strategy.candidates.candidates_multi_driver import INIT_CASH, ORDERINGS, WIN_START  # noqa: E402
from _03_multi_strategy.candidates.candidates_tw_driver import OUT  # noqa: E402
from _04_analysis.benchmark.benchmark_0050 import BENCHMARK_START, bench_0050, curve_stats  # noqa: E402

KEYS = ("db_p7_stop", "db_p8_stop", "db_p9_nostop")
FEE_LABEL = {"rate": "純費率", "tw": "台股實際費稅"}


def main() -> int:
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}")
    runs = pd.read_csv(os.path.join(OUT, "tw_runs.csv"), encoding="utf-8-sig")
    eq = pd.read_parquet(os.path.join(OUT, "equity_2015_tw.parquet"))
    cal = eq.index
    years = (cal[-1] - cal[0]).days / 365.25
    rows = []
    for key in KEYS:
        for kind in ORDERINGS:
            for fees in ("rate", "tw"):
                full = runs[(runs["key"] == key) & (runs["費用"] == fees) & (runs["排序"] == kind)].iloc[0]
                mult, cagr, dd = curve_stats(eq[f"{key}｜{fees}｜{kind}"].to_numpy(np.float64), years)
                rows.append({"交易策略": full["交易策略"], "排序": kind, "費用": FEE_LABEL[fees],
                             "全期總獲利(萬)": full["總獲利(萬)"], "全期資金倍數": full["資金倍數"],
                             "全期最大回撤%": full["最大回撤%"], "2015–2025 資金倍數": mult,
                             "年化報酬率%": cagr, "最大回撤%": dd,
                             "報酬回撤比": round(cagr / dd, 3) if dd > 0 else None})
    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    rows.append({"交易策略": "0050 買進持有（含息）", "排序": "—", "費用": "—",
                 "2015–2025 資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                 "報酬回撤比": round(cagr / dd, 3)})
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "candidates_tw_vs_0050.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
