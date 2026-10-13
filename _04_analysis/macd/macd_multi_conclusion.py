"""
MACD 系列結論：五組交易策略的多股回測（2015–2025）對 0050 買進持有含息（分析段）
==============================================================================
讀回測段 `_03_multi_strategy/macd/macd_conclusion_equity.py` 存下的逐日權益曲線與成交統計，
算資金倍數／年化／最大回撤，再跟 0050 比。這支不跑回測。

**跟 0050 比哪幾組**：（十一）（十二）全期（2002–2025）的 30 組（5 策略 × 2 投法 × 低價／高價／
流動性；隨機是基準線不列入）依總獲利取前 TOP_N 名，同一策略可重複入選。挑選用的是兩篇已發佈的
全期結果（`macd_multi_result.csv`），這裡只換成 2015–2025 的權益去跟 0050 同期比。

**為什麼期間是 2015–2025，不是回測用的 2002–2025**：見 0050 基準模組
`_04_analysis/benchmark/benchmark_0050.py`（對照區間、含息算法、錨點都定義在那裡）。

**判定基準＝報酬回撤比（年化報酬率% ÷ 最大回撤%）**：單看年化會偏好把風險放大的
做法，單看回撤會偏好不交易，兩者相除才問得出「這套值不值得取代長抱 0050」。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/macd/macd_conclusion_equity.py
    python _04_analysis/macd/macd_multi_conclusion.py
輸出：result/macd_multi/
  macd_multi_conclusion.csv      文章第二張表：前 TOP_N 名的 2015–2025 ＋ 0050（另附全期名次與總獲利）
  macd_multi_conclusion_all.csv  30 組的 2015–2025 全表（供查核）
  各列另附「<欄名>_精確」＝由權益曲線算的未四捨五入值（exact_curve；0050 列也有），文章出表從精確值一次進位。
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
                                                   curve_stats, equity_0050)

EXACT = "_精確"
TOP_N = 6
FULL_RESULT = os.path.join(OUT, "macd_multi_result.csv")   # （十一）（十二）的全期結果


def exact_curve(eq: np.ndarray, years: float) -> dict:
    """curve_stats 的未四捨五入版（同 KD 的 kd_multi_conclusion.exact_curve；文章出表一次進位用）。"""
    mult = eq[-1] / eq[0]
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return {"資金倍數" + EXACT: mult, "年化報酬率%" + EXACT: cagr, "最大回撤%" + EXACT: dd,
            "報酬回撤比" + EXACT: cagr / dd if dd > 0 else None}


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
    for _, r in runs.iterrows():
        eq = equity[equity_key(r["交易策略"], r["投法"], r["排序"])].to_numpy(np.float64)
        mult, cagr, dd = curve_stats(eq, years)
        rows.append({"交易策略": r["交易策略"], "投法": r["投法"], "排序": r["排序"],
                     "份數": r["份數"], "交易次數": r["交易次數"], "擋單": r["擋單"],
                     "資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                     "報酬回撤比": round(cagr / dd, 3) if dd > 0 else None}
                    | exact_curve(eq, years))
    all_2015 = pd.DataFrame(rows)

    # 前 TOP_N 名＝（十一）（十二）全期總獲利（精確值）由高到低
    full = pd.read_csv(FULL_RESULT, encoding="utf-8-sig")
    top = (full.sort_values("總獲利(萬)" + EXACT, ascending=False).head(TOP_N)
           [["交易策略", "投法", "排序", "總獲利(萬)" + EXACT]]
           .rename(columns={"總獲利(萬)" + EXACT: "全期總獲利(萬)" + EXACT}))
    top.insert(0, "全期名次", range(1, TOP_N + 1))
    picked = top.merge(all_2015, on=["交易策略", "投法", "排序"], how="left", validate="1:1")
    if picked["資金倍數"].isna().any():
        raise SystemExit("前幾名有組合在 2015–2025 權益裡找不到，請先重跑回測段")
    for _, r in picked.iterrows():
        print(f"#{r['全期名次']} {r['交易策略']}｜{r['投法']}｜{r['排序']}：{r['交易次數']:,} 筆｜"
              f"擋單 {r['擋單']:,}｜×{r['資金倍數']}｜年化 {r['年化報酬率%']}%｜"
              f"回撤 {r['最大回撤%']}%｜報酬回撤比 {r['報酬回撤比']}")
    rows = picked.to_dict("records")

    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    rows.append({"交易策略": "0050 買進持有（含息）", "投法": "—", "份數": None,
                 "交易次數": 1, "擋單": 0, "資金倍數": mult, "年化報酬率%": cagr,
                 "最大回撤%": dd, "報酬回撤比": round(cagr / dd, 3)}
                | exact_curve(equity_0050(cal, INIT_CASH)["市值"].to_numpy(np.float64), years))
    print(f"\n0050 買進持有（含息）：×{mult}｜年化 {cagr}%｜回撤 {dd}%｜"
          f"報酬回撤比 {rows[-1]['報酬回撤比']}")
    print("（錨點：×5.51／16.8%／33.8%；對不上就是含息或分割校準有問題）")

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_multi_conclusion.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    all_path = os.path.join(OUT, "macd_multi_conclusion_all.csv")
    all_2015.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"\n→ {path}、{all_path}")
    print(out[[c for c in out.columns if not c.endswith(EXACT)]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
