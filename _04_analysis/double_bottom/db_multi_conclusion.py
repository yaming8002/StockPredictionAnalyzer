"""
雙底系列結論：多股回測（2015–2025）對 0050 買進持有含息（分析段）
==================================================================
讀回測段 `_03_multi_strategy/double_bottom/db_conclusion_equity.py` 存下的逐日權益曲線與成交統計，
算資金倍數／年化／最大回撤，再跟 0050 比。這支不跑回測。結構同 MACD 的 macd_multi_conclusion。

**跟 0050 比哪幾組**：〈雙底（九）〉〈雙底（十）〉全期（2002–2025）的 30 組（5 組 × 2 投法 × 低價／高價／
成交金額；隨機是基準線不列入）依總獲利取前 TOP_N 名，同一組可重複入選。挑選用全期結果
（db_multi_result.csv），這裡只換成 2015–2025 的權益去跟 0050 同期比。

**為什麼期間是 2015–2025**：0050 在 2015 以前的價格與配息不可靠，對照區間、含息算法、錨點都定義在
`_04_analysis/benchmark/benchmark_0050.py`。策略本身的評估仍以（九）（十）的 2002–2025 全期為準。

**比較基準＝報酬回撤比（年化報酬率% ÷ 最大回撤%）**，另列總獲利、資金倍數、年化、最大回撤。

**兩邊都在 2025 年最後一個交易日收盤全部賣出**（用戶 2026-11-16 定）：策略把未平倉部位、0050 把全部持股
以收盤價賣出並扣手續費＋證交稅（0050 稅率也用 common.DUES）。總獲利＝清算淨額 − 本金、資金倍數＝清算淨額 ÷ 本金、
年化由資金倍數換算，三欄同一個口徑；最大回撤仍用逐日市值權益。0050 改成首日開盤買進（見 benchmark_0050.trade_0050），
原本的 equity_0050 與錨點 ×5.51／16.81%／33.8% 不動。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/double_bottom/db_conclusion_equity.py
    python _04_analysis/double_bottom/db_multi_conclusion.py
輸出：_02_strategy/double_bottom_strategy/result/double_bottom_multi/
  db_multi_conclusion.csv      前 TOP_N 名的 2015–2025 ＋ 0050（附全期名次與全期總獲利）
  db_multi_conclusion_all.csv  30 組的 2015–2025 全表（供查核）
  各列另附「<欄名>_精確」未四捨五入值，文章出表從精確值一次進位。
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _03_multi_strategy.double_bottom.db_conclusion_equity import (EQUITY_FILE, RUNS_FILE,  # noqa: E402
                                                                    WIN_START, equity_key)
from _03_multi_strategy.double_bottom.db_multi_driver import INIT_CASH, OUT, PATH  # noqa: E402
from _04_analysis.benchmark.benchmark_0050 import BENCHMARK_START, trade_0050  # noqa: E402

EXACT = "_精確"
TOP_N = 6
BENCH = "0050 買進持有（含息）"
CONC_FILE = os.path.join(OUT, "db_multi_conclusion.csv")
PROFIT = "2015總獲利(萬)" + EXACT   # 期末全部賣出後的總獲利（萬）
ALL_FILE = os.path.join(OUT, "db_multi_conclusion_all.csv")


def liquidated(eq: np.ndarray, net_end: float, years: float) -> dict:
    """期末全部賣出口徑：總獲利／資金倍數／年化由清算淨額算，最大回撤由逐日市值算（皆未四捨五入）。"""
    mult = net_end / INIT_CASH
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return {PROFIT: (net_end - INIT_CASH) / 10_000, "資金倍數" + EXACT: mult, "年化報酬率%" + EXACT: cagr,
            "最大回撤%" + EXACT: dd, "報酬回撤比" + EXACT: cagr / dd if dd > 0 else None}


def main() -> int:
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段下單起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}，兩邊期間不一致。")
    for path in (EQUITY_FILE, RUNS_FILE, PATH):
        if not os.path.isfile(path):
            raise SystemExit(f"找不到回測輸出：{path}")
    equity = pd.read_parquet(EQUITY_FILE)
    runs = pd.read_csv(RUNS_FILE, encoding="utf-8-sig")
    cal = equity.index
    years = (cal[-1] - cal[0]).days / 365.25

    rows = []
    for _, r in runs.iterrows():
        eq = equity[equity_key(r["交易策略"], r["投法"], r["排序"])].to_numpy(np.float64)
        rows.append({k: r[k] for k in ("交易策略", "投法", "排序", "份數", "交易次數", "擋單")}
                    | {"2015已平倉總獲利(萬)" + EXACT: r["總獲利(萬)" + EXACT]}
                    | liquidated(eq, float(r["期末清算淨額" + EXACT]), years))
    all_2015 = pd.DataFrame(rows)

    full = pd.read_csv(PATH)
    full = full[full["排序"] != "隨機"]
    if len(full) != 30:
        raise SystemExit(f"全期結果應有 30 組（不含隨機），實際 {len(full)}")
    top = (full.sort_values("總獲利(萬)" + EXACT, ascending=False).head(TOP_N)
           [["交易策略", "投法", "排序", "總獲利(萬)" + EXACT]]
           .rename(columns={"總獲利(萬)" + EXACT: "全期總獲利(萬)" + EXACT}))
    top.insert(0, "全期名次", range(1, TOP_N + 1))
    picked = top.merge(all_2015, on=["交易策略", "投法", "排序"], how="left", validate="1:1")
    if picked["資金倍數" + EXACT].isna().any():
        raise SystemExit("前幾名有組合在 2015–2025 權益裡找不到，請先重跑回測段")

    b = trade_0050(cal, INIT_CASH)
    bench = {"交易策略": BENCH, "投法": "—", "排序": "—"} | liquidated(b["市值"], b["期末清算淨額"], years)
    out = pd.concat([picked, pd.DataFrame([bench])], ignore_index=True)
    all_2015 = all_2015.merge(full[["交易策略", "投法", "排序", "總獲利(萬)" + EXACT]]
                              .rename(columns={"總獲利(萬)" + EXACT: "全期總獲利(萬)" + EXACT}),
                              on=["交易策略", "投法", "排序"], validate="1:1")
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(CONC_FILE, index=False, encoding="utf-8-sig")
    all_2015.to_csv(ALL_FILE, index=False, encoding="utf-8-sig")
    show = [c for c in out.columns if c not in ("份數",)]
    print(out[show].round(3).to_string(index=False))
    print(f"（0050 已改期末全部賣出口徑，不等於錨點 ×5.51／16.81%）\n→ {CONC_FILE}、{ALL_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
