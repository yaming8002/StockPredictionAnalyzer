"""
KD 交叉（八）結論：6 交易策略各挑一種投法，對 0050 買進持有含息（2015–2025，分析段）
=====================================================================================
讀回測段 `_03_multi_strategy/kd/kd_conclusion_equity.py` 存下的逐日權益曲線與成交統計，
算資金倍數／年化／最大回撤，每個交易策略在定額／比例中挑**報酬回撤比（年化 ÷ 最大回撤）**
較高者，再跟 0050 比。這支不跑回測。

**為什麼期間是 2015–2025**：見 0050 基準模組 `_04_analysis/benchmark/benchmark_0050.py`。
**為什麼用報酬回撤比挑**：單看年化會偏好把風險放大的比例投法，單看回撤會偏好不交易；
文章（八）「挑出兩種投法中『報酬 ÷ 回撤』較高的那一個」、2026-08-04 reference 訂正 3 同此。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/kd/kd_conclusion_equity.py
    python _04_analysis/kd/kd_multi_conclusion.py [--out <回測段輸出目錄>]
輸出（與回測段同目錄）：
    kd_conclusion_all.csv   12 組全表（規格 9 欄＋擋單＋資金倍數／年化／回撤／報酬回撤比＋選）
    kd_conclusion.csv       文章（八）第一張表：每策略取較優投法的 6 列，依報酬回撤比高到低
    kd_conclusion_vs_0050.csv  文章（八）第二張表：0050 ＋ 上面 6 列（資金倍數／年化／回撤／報酬回撤比）
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _03_multi_strategy.kd.kd_conclusion_equity import (EQUITY_NAME, RUNS_NAME,  # noqa: E402
                                                        WIN_START, equity_key)
from _03_multi_strategy.kd.kd_multi_driver import INIT_CASH, OUT  # noqa: E402
from _04_analysis.benchmark.benchmark_0050 import (BENCHMARK_START, bench_0050,  # noqa: E402
                                                   curve_stats, equity_0050)


EXACT = "_精確"


def exact_curve(eq: np.ndarray, years: float) -> dict:
    """curve_stats 的未四捨五入版（文章出表一次進位用；挑投法仍照 curve_stats 的存檔值）。"""
    mult = eq[-1] / eq[0]
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return {"資金倍數" + EXACT: mult, "年化報酬率%" + EXACT: cagr, "最大回撤%" + EXACT: dd,
            "報酬回撤比" + EXACT: cagr / dd if dd > 0 else None}


def ratio(cagr: float, dd: float):
    """報酬回撤比＝年化報酬率% ÷ 最大回撤%（回撤 0 無定義）。"""
    return round(cagr / dd, 3) if dd > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 結論：挑投法 × 對 0050")
    ap.add_argument("--out", default=OUT, help="回測段輸出目錄（本支也寫在這裡）")
    a = ap.parse_args()

    # 回測段的下單起點必須等於 0050 對照區間起點，否則兩邊比的不是同一段期間
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段下單起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}，"
                         "兩邊期間不一致，請先對齊。")
    eq_path, runs_path = os.path.join(a.out, EQUITY_NAME), os.path.join(a.out, RUNS_NAME)
    for path in (eq_path, runs_path):
        if not os.path.isfile(path):
            raise SystemExit(f"找不到回測輸出：{path}\n"
                             "先跑回測段：python _03_multi_strategy/kd/kd_conclusion_equity.py")
    equity = pd.read_parquet(eq_path)
    runs = pd.read_csv(runs_path, encoding="utf-8-sig")
    cal = equity.index
    years = (cal[-1] - cal[0]).days / 365.25

    stats = []
    for r in runs.itertuples(index=False):
        eq = equity[equity_key(r.交易策略, r.投法)].to_numpy(np.float64)
        mult, cagr, dd = curve_stats(eq, years)
        stats.append({"資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                      "報酬回撤比": ratio(cagr, dd)} | exact_curve(eq, years))
    full = pd.concat([runs.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    # 每個交易策略取報酬回撤比較高的投法（平手取定額：不複利、較不集中）
    full["_key"] = full["報酬回撤比"].fillna(-np.inf)
    best = full.sort_values(["進場", "_key", "投法"], ascending=[True, False, True],
                            kind="mergesort").groupby("進場", sort=False).head(1).index
    full["選"] = full.index.isin(best)
    full = full.drop(columns="_key")

    picked = full[full["選"]].sort_values("報酬回撤比", ascending=False, kind="mergesort")
    pick_cols = ["交易策略", "投法", "交易次數", "擋單", "勝率%", "平均持有天", "獲利平均%",
                 "虧損平均%", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)",
                 "資金倍數", "年化報酬率%", "最大回撤%", "報酬回撤比"]
    exact_cols = [c for c in full.columns if c.endswith(EXACT)]     # 精確值隨列帶著，供文章出表
    picked = picked[pick_cols + exact_cols].rename(columns={"投法": "投入"})

    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    b_exact = exact_curve(equity_0050(cal, INIT_CASH)["市值"].to_numpy(np.float64), years)
    curve_cols = ["資金倍數", "年化報酬率%", "最大回撤%", "報酬回撤比"]
    mine = picked[curve_cols + [c + EXACT for c in curve_cols]].reset_index(drop=True)
    vs = pd.concat([
        pd.DataFrame([{"做法": "0050 買進持有（含息）", "資金倍數": mult, "年化報酬率%": cagr,
                       "最大回撤%": dd, "報酬回撤比": ratio(cagr, dd)} | b_exact]),
        pd.concat([(picked["交易策略"] + " · " + picked["投入"]).rename("做法").reset_index(drop=True),
                   mine], axis=1),
    ], ignore_index=True)

    os.makedirs(a.out, exist_ok=True)
    full.to_csv(os.path.join(a.out, "kd_conclusion_all.csv"), index=False, encoding="utf-8-sig")
    picked.to_csv(os.path.join(a.out, "kd_conclusion.csv"), index=False, encoding="utf-8-sig")
    vs.to_csv(os.path.join(a.out, "kd_conclusion_vs_0050.csv"), index=False, encoding="utf-8-sig")
    print(f"{cal[0].date()} ~ {cal[-1].date()}（{years:.2f} 年）")
    print(full.drop(columns=["進場"] + exact_cols).to_string(index=False))
    print()
    print(vs[["做法"] + curve_cols].to_string(index=False))
    print("（0050 錨點：×5.51／16.81%／33.8%；對不上就是含息或分割校準有問題）")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
