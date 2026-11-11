"""
雙底 進場 × 出場 組合排名（分析段，不跑回測）
==============================================
讀 `_02_strategy/double_bottom_strategy/combo_sweep.py` 產出的 _sweep_summary.csv（924 組），
排除「不適用」組合後，每組在可成交 3 格（ZigZag 3/6/9% ＋ 5 日均量 × 收盤 > 1,000 萬）各自依獲利因子排名，
取 3 格平均名次排序；平均名次相同時以期望值的平均名次排前。

不適用（用戶 2026-11-06 定）：頸線突破進場的買點已在頸線之上，以頸線價或型態高度為依據的
附加出場（頸線停利、目標價兩種、跌破頸線、回到保本、跌破第二底）判斷基準不同，不列入排名。

只用可成交 3 格（用戶 2026-11-11：「直接以成交金額 > 1,000 萬的條件，原版的都要去掉」）；
裸版仍在 _sweep_summary.csv 裡，只是不進排名。

執行：python _04_analysis/double_bottom/combo_ranking.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.double_bottom_strategy.combo_sweep import SUB  # noqa: E402
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402

SUMMARY = os.path.join(RESULT_DIR, SUB, "_sweep_summary.csv")
NOT_APPLICABLE_ENTRY = "neck"
NOT_APPLICABLE_EXITS = ("add_neck", "add_target", "add_target65", "add_neckfail", "add_breakeven", "add_bolow")
RANK_LIQ = True                       # 排名只用可成交格（用戶 2026-11-11）
EXPECTED_CELLS = 3


def load_summary() -> pd.DataFrame:
    """讀全部 924 組，加上 cell 欄與 not_applicable 旗標。"""
    df = pd.read_csv(SUMMARY)
    if len(df) != 924:
        raise RuntimeError(f"{SUMMARY} 應有 924 組，實得 {len(df)}；先跑完 combo_sweep")
    df["cell"] = df["門檻%"].astype(str) + "_" + df["可成交"].map({True: "liq", False: "raw"})
    df["not_applicable"] = (df["進場"] == NOT_APPLICABLE_ENTRY) & df["出場"].isin(NOT_APPLICABLE_EXITS)
    return df


def rank_combos(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """回傳每個 (進場, 出場) 的平均名次表，名次 1 在最前。"""
    df = load_summary() if df is None else df
    ok = df[~df["not_applicable"] & (df["可成交"] == RANK_LIQ)].copy()
    ok["pf_rank"] = ok.groupby("cell")["獲利因子(PF)"].rank(ascending=False)
    ok["ev_rank"] = ok.groupby("cell")["期望報酬值(EV)"].rank(ascending=False)
    g = ok.groupby(["進場", "出場"]).agg(cells=("cell", "nunique"), pf_rank_mean=("pf_rank", "mean"),
                                          pf_rank_worst=("pf_rank", "max"), ev_rank_mean=("ev_rank", "mean"))
    if (g["cells"] != EXPECTED_CELLS).any():
        raise RuntimeError("有組合不滿 3 格，排名不成立")
    g = g.sort_values(["pf_rank_mean", "ev_rank_mean"]).reset_index()
    g.insert(0, "名次", range(1, len(g) + 1))
    return g


if __name__ == "__main__":
    r = rank_combos()
    print(f"可排名 {len(r)} 組")
    print(r.head(15).to_string(index=False))
