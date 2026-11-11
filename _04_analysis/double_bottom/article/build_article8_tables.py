# -*- coding: utf-8 -*-
"""
產生〈雙底（八）：挑選組合與蒙地卡羅壓測〉的表格 HTML。

結構（用戶 2026-11-11）：〈雙底（七）〉只放組合與排名；這篇從七的第 1 名的回檔講起，加上「持有期間跌 20%」門檻
（combo_drawdown_filter.py → _drawdown_filter.csv／_drawdown_filter_rank.csv，可成交 3 格），
通過者重排後取前五（含基本版）進蒙地卡羅（double_bottom_montecarlo.py --drawdown → _montecarlo_drawdown_top5.csv，
6% 可成交、口徑同 MACD（十））。全部只用可成交格（用戶 2026-11-11 拿掉裸版）。

表：
  1. 〈雙底（七）〉第 1 名的回檔（可成交 3 格；欄位同〈雙底（六）〉）。
  2. 篩選後排名前 10：可成交 3 格獲利因子＋平均名次＋最差名次（格式同七的排名表，不上色）。
  3. 6% 可成交：五組標準 10 欄，跟基本版比上色（基本版那列與自己相同、不上色）。
  4. 6% 可成交：五組持有期間的回檔（持有超過一年%、曾跌 ≥10%／≥20% 佔比%）。
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

from _04_analysis.double_bottom.combo_drawdown_filter import OUT as DRAWDOWN_FILTER  # noqa: E402
from _04_analysis.double_bottom.combo_ranking import load_summary, rank_combos  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import DRAWDOWN_RANK, OUT_DRAWDOWN  # noqa: E402
from build_article2_tables import half_up  # noqa: E402
from build_article6_tables import drawdown_table  # noqa: E402
from build_article7_tables import (FOCUS, TOP_MC, drawdown_combo, drawdown_focus_table, mc_table,  # noqa: E402
                                   ranked_label, ranking_table, stats)


def picked() -> pd.DataFrame:
    """挑出的五組（篩選後可成交 3 格重排前五，含基本版）。"""
    return pd.read_csv(DRAWDOWN_RANK).head(TOP_MC)


def first_combo() -> tuple:
    """〈雙底（七）〉排名第 1 的組合。"""
    return tuple(rank_combos().iloc[0][["進場", "出場"]])


def drawdown_rows_table() -> str:
    """6% 可成交，五組各一列的回檔統計。"""
    p, liq = FOCUS
    flt = pd.read_csv(DRAWDOWN_FILTER)
    cell = flt[(flt["門檻%"] == p) & (flt["可成交"] == liq)].set_index(["進場", "出場"])
    head = ("<thead><tr><th>組合</th><th>持有超過一年%</th><th>曾跌 ≥10%<br>佔比%</th>"
            "<th>曾跌 ≥20%<br>佔比%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in picked().iterrows():
        d = cell.loc[(r["進場"], r["出場"])]
        cells = [half_up(d["超過一年%"], 1), half_up(d["曾跌10%"], 1), half_up(d["曾跌20%"], 2)]
        out.append(f"<tr><td>{ranked_label(r['名次'], r['進場'], r['出場'], ' × ')}</td>"
                   f"{''.join(f'<td>{x}</td>' for x in cells)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def all_tables(df: pd.DataFrame) -> dict:
    return {"第1名回檔": drawdown_table(drawdown_combo(*first_combo())),
            "篩選後排名前10": ranking_table(pd.read_csv(DRAWDOWN_RANK), load_summary()),
            "6%可成交": drawdown_focus_table(df),
            "回檔": drawdown_rows_table(),
            "蒙地卡羅": mc_table(OUT_DRAWDOWN)}


def main() -> int:
    for k, html in all_tables(stats()).items():
        print(f"=== {k} ===")
        print(html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
