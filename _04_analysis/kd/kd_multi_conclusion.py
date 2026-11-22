"""
KD 交叉（八）結論：（六）（七）全期總獲利前 TOP_N 名，對 0050 買進持有含息（2015–2025，分析段）
=========================================================================================
讀回測段 `_03_multi_strategy/kd/kd_conclusion_equity.py` 存下的逐日權益曲線與成交統計，
算資金倍數／年化／最大回撤，再跟 0050 比。這支不跑回測。

**跟 0050 比哪幾組**（2026-11-20 改，比照 MACD 結論篇 `_04_analysis/macd/macd_multi_conclusion.py`）：
（六）（七）全期（2002–2025）的 36 組（6 策略 × 2 投法 × 低價／流動性／高價；隨機是基準線不列入）
依總獲利取前 TOP_N 名，同一策略可重複入選。挑選用的是兩篇已發佈的全期結果（`kd_multi_result.csv`），
這裡只換成 2015–2025 的權益去跟 0050 同期比。舊版「每策略在兩投法中挑報酬回撤比較高者」已停用。

**為什麼期間是 2015–2025**：見 0050 基準模組 `_04_analysis/benchmark/benchmark_0050.py`。
**報酬回撤比（年化 ÷ 最大回撤）** 是跟 0050 比的判定基準：單看年化會偏好把風險放大的做法，
單看回撤會偏好不交易。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/kd/kd_conclusion_equity.py
    python _04_analysis/kd/kd_multi_conclusion.py [--out <回測段輸出目錄>]
輸出（與回測段同目錄）：
    kd_conclusion_all.csv      36 組的 2015–2025 全表（規格 9 欄＋擋單＋資金倍數／年化／回撤／報酬回撤比）
    kd_conclusion_top.csv      文章（八）第一張表：全期總獲利前 TOP_N 名（全期規格欄，取自 kd_multi_result.csv）
    kd_conclusion_vs_0050.csv  文章（八）第二張表：0050 ＋ 前 TOP_N 名的 2015–2025（附全期名次）
    各列另附「<欄名>_精確」＝未四捨五入值，文章出表從精確值一次進位。
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
TOP_N = 6
FULL_NAME = "kd_multi_result.csv"     # （六）（七）的全期結果（kd_multi_driver.py 輸出）
KEYS = ["交易策略", "投法", "排序"]


def exact_curve(eq: np.ndarray, years: float) -> dict:
    """curve_stats 的未四捨五入版（文章出表一次進位用）。"""
    mult = eq[-1] / eq[0]
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return {"資金倍數" + EXACT: mult, "年化報酬率%" + EXACT: cagr, "最大回撤%" + EXACT: dd,
            "報酬回撤比" + EXACT: cagr / dd if dd > 0 else None}


def ratio(cagr: float, dd: float):
    """報酬回撤比＝年化報酬率% ÷ 最大回撤%（回撤 0 無定義）。"""
    return round(cagr / dd, 3) if dd > 0 else None


def stats_2015(equity: pd.DataFrame, runs: pd.DataFrame, years: float) -> pd.DataFrame:
    """36 組的 2015–2025 成交統計＋權益曲線統計。"""
    rows = []
    for r in runs.itertuples(index=False):
        eq = equity[equity_key(r.交易策略, r.投法, r.排序)].to_numpy(np.float64)
        mult, cagr, dd = curve_stats(eq, years)
        rows.append({"資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                     "報酬回撤比": ratio(cagr, dd)} | exact_curve(eq, years))
    return pd.concat([runs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def top_full(path: str) -> pd.DataFrame:
    """（六）（七）全期 36 組（不含隨機）依總獲利精確值取前 TOP_N，附全期名次。"""
    full = pd.read_csv(path, encoding="utf-8-sig")
    full = full[full["排序"] != "隨機"]
    top = full.sort_values("總獲利(萬)" + EXACT, ascending=False, kind="mergesort").head(TOP_N).copy()
    top.insert(0, "全期名次", range(1, TOP_N + 1))
    return top.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 結論：全期前 6 名 × 對 0050")
    ap.add_argument("--out", default=OUT, help="回測段輸出目錄（本支也寫在這裡）")
    a = ap.parse_args()

    # 回測段的下單起點必須等於 0050 對照區間起點，否則兩邊比的不是同一段期間
    if WIN_START != BENCHMARK_START:
        raise SystemExit(f"回測段下單起點 {WIN_START} ≠ 0050 對照起點 {BENCHMARK_START}，"
                         "兩邊期間不一致，請先對齊。")
    eq_path, runs_path = os.path.join(a.out, EQUITY_NAME), os.path.join(a.out, RUNS_NAME)
    full_path = os.path.join(a.out, FULL_NAME)
    for path in (eq_path, runs_path, full_path):
        if not os.path.isfile(path):
            raise SystemExit(f"找不到輸出：{path}\n先跑 kd_multi_driver.py 與 kd_conclusion_equity.py")
    equity = pd.read_parquet(eq_path)
    runs = pd.read_csv(runs_path, encoding="utf-8-sig")
    if "排序" not in runs.columns:
        raise SystemExit(f"{runs_path} 沒有「排序」欄＝舊版只跑低價的輸出，請重跑回測段")
    cal = equity.index
    years = (cal[-1] - cal[0]).days / 365.25

    all_2015 = stats_2015(equity, runs, years)
    top = top_full(full_path)

    curve_cols = ["資金倍數", "年化報酬率%", "最大回撤%", "報酬回撤比"]
    picked = top[["全期名次"] + KEYS].merge(
        all_2015[KEYS + ["交易次數", "擋單"] + curve_cols + [c + EXACT for c in curve_cols]],
        on=KEYS, how="left", validate="1:1")
    if picked["資金倍數"].isna().any():
        raise SystemExit("前幾名有組合在 2015–2025 權益裡找不到，請先重跑回測段")

    mult, cagr, dd = bench_0050(cal, years, INIT_CASH)
    b_exact = exact_curve(equity_0050(cal, INIT_CASH)["市值"].to_numpy(np.float64), years)
    vs = pd.concat([
        pd.DataFrame([{"全期名次": None, "交易策略": "0050 買進持有（含息）", "投法": "—", "排序": "—",
                       "交易次數": 1, "擋單": 0, "資金倍數": mult, "年化報酬率%": cagr,
                       "最大回撤%": dd, "報酬回撤比": ratio(cagr, dd)} | b_exact]),
        picked], ignore_index=True)

    os.makedirs(a.out, exist_ok=True)
    all_2015.to_csv(os.path.join(a.out, "kd_conclusion_all.csv"), index=False, encoding="utf-8-sig")
    top.to_csv(os.path.join(a.out, "kd_conclusion_top.csv"), index=False, encoding="utf-8-sig")
    vs.to_csv(os.path.join(a.out, "kd_conclusion_vs_0050.csv"), index=False, encoding="utf-8-sig")

    show = [c for c in ["全期名次"] + KEYS + ["交易次數", "擋單", "勝率%", "平均持有天", "獲利平均%",
                                              "虧損平均%", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]]
    print("全期（2002–2025）總獲利前", TOP_N, "名：")
    print(top[show].to_string(index=False))
    print(f"\n{cal[0].date()} ~ {cal[-1].date()}（{years:.2f} 年）對 0050：")
    print(vs[["全期名次"] + KEYS + curve_cols].to_string(index=False))
    print("（0050 錨點：×5.51／16.81%／33.8%；對不上就是含息或分割校準有問題）")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
