"""
雙底 進場 × 出場 組合：篩掉「持有期間常跌 20% 以上」的組合後重新排名（分析段，不跑回測）
=====================================================================================
用戶 2026-11-08：長期持有、帳面跌 20% 以上實際不容易抱住 → 找實際上沒有下跌 20% 的組合。
- 範圍：可排名的 148 種組合（排除不適用，見 combo_ranking），只看可成交的 3 格
  （ZigZag 3/6/9% ＋ 5 日均量 × 收盤 > 1,000 萬；用戶：「只要跑 1000 萬以上的資訊就好」）。
- 回檔口徑同〈雙底（六）〉build_article6_tables.worst_close_pct：已平倉交易（損益 ≠ 0），
  持有期間＝買進日～賣出日前一天的最低收盤，相對買進價。
- 判準：3 格的「曾跌 ≥20% 佔比」都 ≤ 同格基本版（基本進場 × 基本出場）。
- 排名：只在通過的組合之間，可成交 3 格各依獲利因子排名，取平均名次；同分以期望值平均名次
  （用戶 2026-11-11：「直接以成交金額 > 1,000 萬的條件，原版的都要去掉」→ 裸版不進排名；11-10 的 6 格排名作廢）。

執行：python _04_analysis/double_bottom/combo_drawdown_filter.py
輸出：result/single_double_bottom_combo/_drawdown_filter.csv（每組每格的回檔統計＋是否通過）
      result/single_double_bottom_combo/_drawdown_filter_rank.csv（通過者重新排名）
"""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.double_bottom_strategy.combo_sweep import SUB, _tag  # noqa: E402
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402
from _04_analysis.double_bottom.combo_ranking import RANK_LIQ, load_summary  # noqa: E402

PCTS = (3, 6, 9)
LIQ = True
RANK_CELLS = 3
BASE = ("base", "base")
DROP = -20.0
OUT = os.path.join(RESULT_DIR, SUB, "_drawdown_filter.csv")
OUT_RANK = os.path.join(RESULT_DIR, SUB, "_drawdown_filter_rank.csv")


def load_closes() -> dict:
    """{stock_id: (日期, 收盤＋尾端 inf)}；尾端 inf 讓 reduceat 的區間終點可以等於序列長度。"""
    return {sid: (df.index.to_numpy(), np.append(df["close"].to_numpy(dtype=float), np.inf))
            for sid, df in common.iter_market(columns=["close"])}


def worst_close_pct(tr: pd.DataFrame, px: dict) -> np.ndarray:
    """每筆交易持有期間最低收盤相對買進價的 %；結果與 build_article6_tables.worst_close_pct 相同，
    改成每檔股票一次 np.minimum.reduceat 向量化（888 → 444 格逐筆迴圈太慢）。"""
    out = np.empty(len(tr))
    for sid, idx in tr.groupby("stock_id").indices.items():
        days, close = px[sid]
        sub = tr.iloc[idx]
        a = np.searchsorted(days, sub["buy_date"].to_numpy(), "left")
        b = np.maximum(np.searchsorted(days, sub["sell_date"].to_numpy(), "left"), a + 1)
        pairs = np.column_stack([a, b]).ravel()
        lows = np.minimum.reduceat(close, pairs)[::2]
        out[idx] = (lows / sub["buy_price"].to_numpy() - 1) * 100
    return out


def cell_drawdown(p: int, e: str, x: str, px: dict) -> dict:
    tag = _tag(p / 100, LIQ, e, x)
    tr = pd.read_csv(os.path.join(RESULT_DIR, SUB, tag, f"single_double_bottom_{tag}_trades.csv"),
                     parse_dates=["buy_date", "sell_date"])
    tr["real_pnl"] = pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0)
    tr = tr[tr["real_pnl"] != 0].reset_index(drop=True)
    dd = worst_close_pct(tr, px)
    if np.isnan(dd).any():
        raise RuntimeError(f"{tag} 有交易算不出持有期間最低收盤")
    held = (tr["sell_date"] - tr["buy_date"]).dt.days.to_numpy()
    return {"門檻%": p, "可成交": LIQ, "進場": e, "出場": x, "交易次數": len(tr),
            "超過一年%": (held > 365).mean() * 100, "曾跌10%": (dd <= -10).mean() * 100,
            "曾跌20%": (dd <= DROP).mean() * 100}


def rerank(passed: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    """通過者之間重新排名：可成交 3 格 PF 名次平均，同分看 EV 名次平均。"""
    keys = passed[["進場", "出場"]].drop_duplicates()
    s = summ[summ["可成交"] == RANK_LIQ].merge(keys, on=["進場", "出場"])
    s["pf_rank"] = s.groupby("cell")["獲利因子(PF)"].rank(ascending=False)
    s["ev_rank"] = s.groupby("cell")["期望報酬值(EV)"].rank(ascending=False)
    g = s.groupby(["進場", "出場"]).agg(cells=("cell", "nunique"), pf_rank_mean=("pf_rank", "mean"),
                                         pf_rank_worst=("pf_rank", "max"), ev_rank_mean=("ev_rank", "mean"))
    if (g["cells"] != RANK_CELLS).any():
        raise RuntimeError("有通過的組合不滿 3 格")
    g = g.sort_values(["pf_rank_mean", "ev_rank_mean"]).reset_index()
    g.insert(0, "名次", range(1, len(g) + 1))
    return g


def main() -> int:
    t0 = time.time()
    summ = load_summary()
    combos = summ[~summ["not_applicable"]][["進場", "出場"]].drop_duplicates()
    px = load_closes()
    print(f"{len(combos)} 種組合 × {len(PCTS)} 格（可成交）｜讀價 {time.time() - t0:.0f} 秒", flush=True)
    rows = [cell_drawdown(p, e, x, px) for e, x in combos.itertuples(index=False) for p in PCTS]
    df = pd.DataFrame(rows)
    # 交叉核對：筆數必須與 combo_sweep 彙總一致
    chk = df.merge(summ, on=["門檻%", "可成交", "進場", "出場"], suffixes=("", "_summ"))
    if len(chk) != len(df) or (chk["交易次數"] != chk["交易次數_summ"]).any():
        raise RuntimeError("回檔統計的交易次數與 _sweep_summary 不一致")
    base = df[(df["進場"] == BASE[0]) & (df["出場"] == BASE[1])].set_index("門檻%")["曾跌20%"]
    df["基本版曾跌20%"] = df["門檻%"].map(base)
    df["本格通過"] = df["曾跌20%"] <= df["基本版曾跌20%"]
    df["通過"] = df.groupby(["進場", "出場"])["本格通過"].transform("all")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    rank = rerank(df[df["通過"]], summ)
    rank.to_csv(OUT_RANK, index=False, encoding="utf-8-sig")
    print(f"通過 {len(rank)} / {len(combos)} 種｜{time.time() - t0:.0f} 秒")
    print(f"→ {OUT}\n→ {OUT_RANK}")
    print(rank.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
