"""
0050 買進持有（含息）：持有期間越長，報酬為正的機率怎麼變
==========================================================
心得文〈消息面主導的市場，與我對技術指標的重新思考〉引用的數字由此產生。

做法：用 benchmark_0050.equity_0050 的逐日含息市值（2015~2025），對每一個交易日當起點，
往後持有 N 個交易日，算「期末市值 > 起點市值」的比例。為什麼用滾動起點而不是只看
一條路徑：同一段歷史，換一天進場結果就不同，把所有起點都算一次，才看得出「持有多久
才比較不怕買在錯的那一天」。相鄰起點高度重疊，這是描述性比例、不是獨立樣本的機率。

執行：python _04_analysis/benchmark/holding_period_winrate.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common  # noqa: E402
from _04_analysis.benchmark.benchmark_0050 import (  # noqa: E402
    BENCH, BENCHMARK_START, BENCHMARK_END, equity_0050)

# 交易日數：1 天、約 1 週、約 1 個月、約 1 季、約 1 年、約 3 年、約 5 年
HORIZONS = {"1 天": 1, "1 週": 5, "1 個月": 21, "1 季": 63,
            "1 年": 252, "3 年": 756, "5 年": 1260}


def winrate_table(eq: np.ndarray) -> pd.DataFrame:
    """每個持有期間：起點數、報酬為正的比例%、報酬中位數%、最差一次報酬%。"""
    rows = []
    for name, n in HORIZONS.items():
        if len(eq) <= n:
            continue
        ret = eq[n:] / eq[:-n] - 1.0
        rows.append({"持有期間": name, "交易日數": n, "起點數": len(ret),
                     "報酬為正%": round(float((ret > 0).mean() * 100), 1),
                     "報酬中位數%": round(float(np.median(ret) * 100), 1),
                     "最差一次%": round(float(ret.min() * 100), 1)})
    return pd.DataFrame(rows)


def main() -> int:
    px = pd.read_parquet(os.path.join(common.DATA_DIR, f"{BENCH}.parquet")).sort_index()
    cal = px.loc[BENCHMARK_START:BENCHMARK_END].index
    eq = equity_0050(cal)["市值"].to_numpy(np.float64)
    tbl = winrate_table(eq)
    print(f"0050 買進持有（含息）{cal[0].date()} ~ {cal[-1].date()}，滾動起點")
    print(tbl.to_string(index=False))
    out = os.path.join(_root, "result", "benchmark")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "0050_holding_period_winrate.csv")
    tbl.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
