# -*- coding: utf-8 -*-
"""
產生〈雙底（七）：拼裝整合〉的表格 HTML（用戶 2026-11-10 拆兩篇：七＝組合＋回檔篩選，八＝挑選與蒙地卡羅）。

資料來源＝`_02_strategy/double_bottom_strategy/combo_sweep.py` 的 924 組（11 進場 × 14 出場 × 6 格，文章只用其中可成交 462 組），
排名＝`_04_analysis/double_bottom/combo_ranking.py`（排除不適用後 148 種組合、可成交 3 格 PF 平均名次），
文章只用可成交 3 格（用戶 2026-11-11：「直接以成交金額 > 1,000 萬的條件，原版的都要去掉」），裸版不出表。
蒙地卡羅＝`_04_analysis/double_bottom/double_bottom_montecarlo.py` 的 _montecarlo_top6.csv。
（文章表不含原始 14 條出場中的 add/rep 以外寫法；替換版只收 ATR／MA10／高檔 M 頭三條，見 combo_sweep。）

表（用戶 2026-11-11：七只放組合、排名、第 1 名，最後列出全部組合；跌 20% 篩選與蒙地卡羅移到〈雙底（八）〉）：
  1. 基本版（基本進場 × 基本出場）可成交 3 格，不上色——全篇底色的基準。
  2. 排名前 10：各組 3 格獲利因子＋平均名次＋最差名次（不上色）。
  3. 前 5 名各自 3 格，標準 10 欄、跟同格基本版比上色（用戶 2026-11-11）。
  4. 附錄：3%／6%／9% 各一張，148 種組合全列（基本版第一列、其餘依獲利因子由高到低），標準 10 欄、不上色。
  （drawdown_combo／drawdown_focus_table／mc_table／ranked_label 給〈雙底（八）〉用。）
數字從逐筆 trades 用 fast_multi.exact_stats 重算精確值、一次 half-up（同 build_article2_tables）。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article7_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_combo/_article7_stats.csv（verify 用）
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import functools  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.double_bottom_strategy.combo_sweep import SUB, _tag  # noqa: E402
from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats  # noqa: E402
from _04_analysis.double_bottom.combo_ranking import load_summary, rank_combos  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import DRAWDOWN_RANK, OUT_DRAWDOWN as MC_DRAWDOWN_FILE  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import OUT as MC_FILE  # noqa: E402
from build_article2_tables import COLS, HEAD, NOTE, PCTS, cell_class, fmt, half_up, row_label  # noqa: E402
from build_article2_tables import LABELS as ENTRY_LABELS  # noqa: E402
from build_article4_tables import LABELS as EXIT_RULE_LABELS  # noqa: E402
from build_article6_tables import drawdown_table, worst_close_pct  # noqa: E402

SRC = os.path.join(RESULT_DIR, SUB)
STATS_FILE = os.path.join(SRC, "_article7_stats.csv")
TOP_TABLE = 10
TOP_DETAIL = 5                        # 列完整 10 欄的名次數（用戶 2026-11-11：「前 5 名都列完整數字」）
TOP_MC = 5                            # 〈雙底（八）〉蒙地卡羅取篩選後前五（含基本版；用戶 2026-11-11）
FOCUS = (6, True)                     # 前六與附錄用的格子：6% 可成交（用戶 2026-11-07）
BASE = ("base", "base")


def entry_label(e: str) -> str:
    return "基本進場" if e == "base" else ENTRY_LABELS[e]


def exit_label(x: str) -> str:
    if x == "base":
        return "基本出場"
    kind, rule = x.split("_", 1)
    return ("附加：" if kind == "add" else "替換：") + EXIT_RULE_LABELS[rule]


def combo_label(e: str, x: str, sep: str = "<br>× ") -> str:
    return f"{entry_label(e)}{sep}{exit_label(x)}"


def _one(p: int, liq: bool, e: str, x: str) -> dict:
    tag = _tag(p / 100, liq, e, x)
    tr = pd.read_csv(os.path.join(SRC, tag, f"single_double_bottom_{tag}_trades.csv"))
    ex = exact_stats(tr)
    n = int((pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0).sum())
    return {"門檻%": p, "可成交": liq, "進場": e, "出場": x, "交易次數": n,
            **{c: ex[c + EXACT] for c in COLS if c != "交易次數"}}


@functools.lru_cache(maxsize=None)
def _one_cached(p: int, liq: bool, e: str, x: str) -> dict:
    return _one(p, liq, e, x)


def stats() -> pd.DataFrame:
    """全部可排名組合 × 可成交 3 格（附錄三張表全列）。"""
    summ = load_summary()
    rank = rank_combos(summ)
    keys = [(p, True, r["進場"], r["出場"]) for _, r in rank.iterrows() for p in PCTS]
    df = pd.DataFrame([_one_cached(*k) for k in keys])
    # 交叉核對：交易次數必須與 combo_sweep 的彙總一致
    chk = df.merge(summ, on=["門檻%", "可成交", "進場", "出場"], suffixes=("", "_summ"))
    if len(chk) != len(df) or (chk["交易次數"] != chk["交易次數_summ"]).any():
        raise RuntimeError("逐筆重算的交易次數與 _sweep_summary 不一致")
    return df


def _get(df, p, liq, e, x):
    return df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["進場"] == e) & (df["出場"] == x)].iloc[0]


def _row(label: str, r, b=None) -> str:
    tds = []
    for c in COLS:
        cls = "" if b is None else cell_class(c, r[c], b[c])
        tds.append(f'<td class="{cls}">{fmt(c, r[c])}</td>' if cls else f"<td>{fmt(c, r[c])}</td>")
    return f"<tr><td>{label}</td>{''.join(tds)}</tr>"


def _wrap(rows, note: bool) -> str:
    out = ['<div class="table-wrap"><table class="bt-table">', HEAD, "<tbody>", *rows, "</tbody></table></div>"]
    if note:
        out.append(NOTE)
    return "\n".join(out)


def cell_table(df, e, x, colored: bool) -> str:
    """一組的可成交 3 格（3/6/9%）；colored 時跟同格基本版比。"""
    rows = [_row(row_label(p, True), _get(df, p, True, e, x), _get(df, p, True, *BASE) if colored else None)
            for p in PCTS]
    return _wrap(rows, colored)


def ranking_table(rank: pd.DataFrame, summ: pd.DataFrame, top: int = TOP_TABLE) -> str:
    """排名前 top（預設 10）：可成交 3 格獲利因子（3 位）、平均名次（1 位）、最差名次。"""
    head = ("<thead><tr><th>#</th><th>進場 × 出場</th><th>3%<br>可成交</th>"
            "<th>6%<br>可成交</th><th>9%<br>可成交</th><th>平均<br>名次</th><th>最差<br>名次</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in rank.head(top).iterrows():
        # 從逐筆交易重算精確值再一次 half-up；彙總檔 PF 已四捨五入到 4 位，再進位會跟 10 欄表差 0.001
        pfs = [fmt("獲利因子", _one_cached(p, True, r["進場"], r["出場"])["獲利因子"]) for p in PCTS]
        worst = half_up(r["pf_rank_worst"], 1).normalize()
        out.append(f"<tr><td>{r['名次']}</td><td>{combo_label(r['進場'], r['出場'], ' × ')}</td>"
                   f"{''.join(f'<td>{v}</td>' for v in pfs)}<td>{half_up(r['pf_rank_mean'], 1)}</td>"
                   f"<td>{worst:f}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def top_focus_table(df, rank) -> str:
    p, liq = FOCUS
    b = _get(df, p, liq, *BASE)
    rows = [_row("基本版", b)]
    rows += [_row(f"#{r['名次']} {combo_label(r['進場'], r['出場'])}", _get(df, p, liq, r["進場"], r["出場"]), b)
             for _, r in rank.head(TOP_MC).iterrows()]
    return _wrap(rows, True)


def ranked_label(rank_no, e: str, x: str, sep: str) -> str:
    """回檔篩選後基本版本身也在名單內，直接寫「基本版」，不寫成「基本進場 × 基本出場」。"""
    return f"#{rank_no} " + ("基本版" if (e, x) == BASE else combo_label(e, x, sep))


def drawdown_focus_table(df) -> str:
    """回檔篩選後重排的前六，在 6% 可成交這一格；跟基本版比上色（基本版那列與自己相同、不上色）。"""
    p, liq = FOCUS
    b = _get(df, p, liq, *BASE)
    rows = [_row(ranked_label(r["名次"], r["進場"], r["出場"], "<br>× "), _get(df, p, liq, r["進場"], r["出場"]), b)
            for _, r in pd.read_csv(DRAWDOWN_RANK).head(TOP_MC).iterrows()]
    return _wrap(rows, True)


def mc_table(path: str = MC_FILE) -> str:
    mc = pd.read_csv(path)
    head = ("<thead><tr><th>組合</th><th>交易數</th><th>勝率%</th><th>賺賠比</th><th>報酬%<br>悲觀P5</th>"
            "<th>報酬%<br>中位</th><th>報酬%<br>樂觀P95</th><th>區間<br>寬度</th><th>本金大虧%<br>(跌破5成)</th>"
            "<th>連敗<br>P95</th><th>回撤%<br>P95</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in mc.iterrows():
        ret = {k: half_up(r[f"報酬% {k}_精確"], 0) for k in ("P5", "中位", "P95")}
        cells = [f"{int(r['交易數']):,}", f"{half_up(r['勝率%'], 1)}", f"{half_up(r['賺賠比'], 2)}",
                 *(f"+{ret[k]:,}" for k in ("P5", "中位", "P95")), f"{ret['P95'] - ret['P5']:,}",
                 f"{half_up(r['破產%'], 2)}", f"{int(r['最大連敗 P95'])}", f"{half_up(r['最大回撤% P95'], 1)}"]
        out.append(f"<tr><td>{ranked_label(r['名次'], r['進場'], r['出場'], ' × ')}</td>"
                   f"{''.join(f'<td>{c}</td>' for c in cells)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def appendix_table(df, p: int) -> str:
    """ZigZag p% 可成交，148 種組合全列：基本版第一列，其餘依獲利因子由高到低；不上色。"""
    cell = df[(df["門檻%"] == p) & df["可成交"]]
    rest = cell[~((cell["進場"] == "base") & (cell["出場"] == "base"))].sort_values("獲利因子", ascending=False)
    rows = [_row("基本版", _get(df, p, True, *BASE))]
    rows += [_row(combo_label(r["進場"], r["出場"]), r) for _, r in rest.iterrows()]
    return _wrap(rows, False)


def _closed_combo_trades(p: int, liq: bool, e: str, x: str) -> pd.DataFrame:
    """該組該格的已平倉交易；排掉損益剛好 0 的，與主表「交易次數」同一批。"""
    tag = _tag(p / 100, liq, e, x)
    tr = pd.read_csv(os.path.join(SRC, tag, f"single_double_bottom_{tag}_trades.csv"),
                     parse_dates=["buy_date", "sell_date"])
    tr["real_pnl"] = pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0)
    return tr[tr["real_pnl"] != 0].reset_index(drop=True)


@functools.lru_cache(maxsize=4)
def drawdown_combo(e: str, x: str) -> pd.DataFrame:
    """可成交 3 格回檔統計，欄位與口徑同〈雙底（六）〉build_article6_tables.drawdown（用戶 2026-11-08：完全照雙底（六））。
    持有期間＝買進日～賣出日前一天的最低收盤，相對買進價。"""
    rows = []
    for p in PCTS:
        tr = _closed_combo_trades(p, True, e, x)
        dd = worst_close_pct(tr)
        held = (tr["sell_date"] - tr["buy_date"]).dt.days.to_numpy()
        win = (tr["real_pnl"] > 0).to_numpy()
        win_dd = win & (dd <= -10)
        rows.append({"門檻%": p, "可成交": True, "交易次數": len(tr),
                     "超過一年%": (held > 365).mean() * 100,
                     "曾跌10%": (dd <= -10).mean() * 100, "曾跌20%": (dd <= -20).mean() * 100,
                     "賺錢曾跌10%": win_dd.sum() / win.sum() * 100,
                     "賺錢曾跌10%持有天中位數": float(np.median(held[win_dd]))})
    return pd.DataFrame(rows)


def all_tables(df: pd.DataFrame) -> dict:
    summ = load_summary()
    rank = rank_combos(summ)
    first = tuple(rank.iloc[0][["進場", "出場"]])
    return {"基本版": cell_table(df, *BASE, colored=False),
            "排名前10": ranking_table(rank, summ),
            **{f"第{n}名": cell_table(df, r["進場"], r["出場"], colored=True)
               for n, (_, r) in enumerate(rank.head(TOP_DETAIL).iterrows(), 1)},
            **{f"附錄{p}%": appendix_table(df, p) for p in PCTS}}


def main() -> int:
    df = stats()
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for k, html in all_tables(df).items():
        print(f"=== {k} ===")
        print(html)
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
