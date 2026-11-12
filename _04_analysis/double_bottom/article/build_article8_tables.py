# -*- coding: utf-8 -*-
"""
產生〈雙底（八）：挑選組合與蒙地卡羅壓測〉的表格 HTML。

結構（用戶 2026-11-11 附註）：
- 先拿〈雙底（七）〉排名前五，以 ZigZag 6% 為範例看持有期間的回檔（「不用把所有數據都列出來……先以 ZigZag 6% 為範例」）。
- 「門檻 × 進場 × 出場」444 格各自當一個策略（「幾% 買入條件跟賣出條件……應該是要完整的排列組合，並非 6%」）：
  先拿掉持有期間曾跌 ≥20% 比例超過同門檻基本版的格子，通過者依獲利因子排名
  （combo_drawdown_filter.py → _drawdown_filter_cells.csv）；交易數不到蒙地卡羅抽樣下限的格子跳過、順序往下遞補，
  取前五進蒙地卡羅（double_bottom_montecarlo.py --drawdown → _montecarlo_drawdown_top5.csv）。
全部只用可成交格（用戶 2026-11-11 拿掉裸版）。

表：
  1. 〈雙底（七）〉前五名，ZigZag 6% 可成交的回檔（欄位同〈雙底（六）〉，首欄改組合）。
  2. 篩選後排名前 10 格：門檻、組合、交易次數、獲利因子、期望值、曾跌 ≥20% 佔比、是否進蒙地卡羅（不上色）。
  組合欄只寫組合名稱，不加 #名次 前綴（用戶 2026-11-11）。
  3. 挑出的五組：標準 10 欄，各自那一格；跟同門檻基本版比上色（基本版那列與自己相同、不上色）。
  4. 挑出的五組：持有期間的回檔（持有超過一年%、曾跌 ≥10%／≥20% 佔比%）。
  5. 蒙地卡羅（欄位同 MACD（十））。
數字從逐筆 trades 重算（build_article7_tables）、一次 half-up。

執行：STOCK_DATA_DIR=... PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article8_tables.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from _04_analysis.double_bottom.combo_drawdown_filter import OUT_CELLS  # noqa: E402
from _04_analysis.double_bottom.combo_ranking import rank_combos  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import OUT_DRAWDOWN  # noqa: E402
from build_article2_tables import half_up  # noqa: E402
from build_article7_tables import (BASE, FOCUS, TOP_MC, _get, _row, _wrap, combo_or_base,  # noqa: E402
                                   drawdown_cell, fmt, mc_table)

TOP_TABLE = 10
SEVENTH_TOP = 5          # 〈雙底（七）〉前五名的回檔（用戶 2026-11-11）


def seventh_top() -> pd.DataFrame:
    """〈雙底（七）〉排名前五（148 種組合、可成交 3 格平均名次）。"""
    return rank_combos().head(SEVENTH_TOP)


def cells() -> pd.DataFrame:
    """逐格篩選後的排名（只含通過者）。"""
    return pd.read_csv(OUT_CELLS)


def picked() -> pd.DataFrame:
    """挑出的五組：逐格排名依序往下，跳過交易數不到抽樣下限的格子。"""
    c = cells()
    return c[c["夠筆數"]].head(TOP_MC).reset_index(drop=True)


def cell_label(p: int, e: str, x: str, sep: str) -> str:
    """首欄＝門檻＋組合：一列是一格（門檻 × 進場 × 出場），不寫門檻表格說明就不完整（用戶 2026-11-11）。"""
    return f"ZigZag {p}%<br>{combo_or_base(e, x, sep)}"


def seventh_drawdown_table() -> str:
    """〈雙底（七）〉前五名在 ZigZag 6% 可成交的回檔，每組一列。"""
    p, _ = FOCUS
    head = ("<thead><tr><th>組合</th><th>持有超過<br>一年%</th><th>曾跌 ≥10%<br>佔比%</th><th>曾跌 ≥20%<br>佔比%</th>"
            "<th>賺錢交易中<br>曾跌 ≥10% 佔比%</th><th>這些交易的<br>持有天中位數</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in seventh_top().iterrows():
        d = drawdown_cell(p, r["進場"], r["出場"])
        vals = [half_up(d["超過一年%"], 1), half_up(d["曾跌10%"], 1), half_up(d["曾跌20%"], 1),
                half_up(d["賺錢曾跌10%"], 1), half_up(d["賺錢曾跌10%持有天中位數"], 0)]
        out.append(f"<tr><td>{cell_label(p, r['進場'], r['出場'], ' × ')}</td>"
                   f"{''.join(f'<td>{v}</td>' for v in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cell_ranking_table() -> str:
    """篩選後排名前 10 格；最後一欄標「選入」或「筆數不足」。"""
    head = ("<thead><tr><th>#</th><th>門檻</th><th>進場 × 出場</th><th>交易次數</th><th>獲利因子</th>"
            "<th>期望值<br>/筆(元)</th><th>曾跌 ≥20%<br>佔比%</th><th>蒙地卡羅</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    chosen = set(zip(picked()["門檻%"], picked()["進場"], picked()["出場"]))
    for _, r in cells().head(TOP_TABLE).iterrows():
        key = (r["門檻%"], r["進場"], r["出場"])
        mark = "選入" if key in chosen else ("筆數不足" if not r["夠筆數"] else "—")
        name = combo_or_base(r["進場"], r["出場"], " × ")
        vals = [f"{int(r['交易次數']):,}", fmt("獲利因子", r["獲利因子_精確"]), fmt("期望值/筆", r["期望值_精確"]),
                f"{half_up(r['曾跌20%'], 2)}", mark]
        out.append(f"<tr><td>{r['名次']}</td><td>{int(r['門檻%'])}%</td><td>{name}</td>"
                   f"{''.join(f'<td>{v}</td>' for v in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def picked_table(df: pd.DataFrame) -> str:
    """挑出的五組，各自那一格的標準 10 欄；跟同門檻基本版比上色。"""
    rows = []
    for _, r in picked().iterrows():
        p = int(r["門檻%"])
        rows.append(_row(cell_label(p, r["進場"], r["出場"], " × "), _get(df, p, True, r["進場"], r["出場"]),
                         _get(df, p, True, *BASE)))
    return _wrap(rows, True)


def picked_drawdown_table() -> str:
    """挑出的五組，持有期間的回檔（數字取自 combo_drawdown_filter 的逐格統計）。"""
    head = ("<thead><tr><th>組合</th><th>持有超過一年%</th><th>曾跌 ≥10%<br>佔比%</th>"
            "<th>曾跌 ≥20%<br>佔比%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in picked().iterrows():
        vals = [half_up(r["超過一年%"], 1), half_up(r["曾跌10%"], 1), half_up(r["曾跌20%"], 2)]
        out.append(f"<tr><td>{cell_label(int(r['門檻%']), r['進場'], r['出場'], ' × ')}</td>"
                   f"{''.join(f'<td>{v}</td>' for v in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def mc_labels() -> list:
    """蒙地卡羅表的組合名稱；CSV 的列序必須與挑出的五組一致。"""
    mc = pd.read_csv(OUT_DRAWDOWN)
    pk = picked()
    if list(zip(mc["門檻%"], mc["進場"], mc["出場"])) != list(zip(pk["門檻%"], pk["進場"], pk["出場"])):
        raise RuntimeError("蒙地卡羅結果與挑出的五組不一致，先重跑 double_bottom_montecarlo.py --drawdown")
    return [cell_label(int(r["門檻%"]), r["進場"], r["出場"], " × ") for _, r in pk.iterrows()]


def all_tables(df: pd.DataFrame) -> dict:
    return {"七前五回檔": seventh_drawdown_table(),
            "篩選後排名前10": cell_ranking_table(),
            "挑出的五組": picked_table(df),
            "回檔": picked_drawdown_table(),
            "蒙地卡羅": mc_table(OUT_DRAWDOWN, mc_labels())}


def main() -> int:
    from build_article7_tables import stats
    for k, html in all_tables(stats()).items():
        print(f"=== {k} ===")
        print(html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
