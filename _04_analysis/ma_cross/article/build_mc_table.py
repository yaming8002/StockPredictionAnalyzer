# -*- coding: utf-8 -*-
"""
產生〈均線交叉（四）：單股評估〉蒙地卡羅表的 HTML 列（只排版）。

資料＝_04_analysis/ma_cross/ma_cross_montecarlo.py 的 mc_realistic.csv（夾角>20°＋ADX<25＋1000 張）。
上色：每筆期望與報酬三欄依正負（正＝粉紅 up、負＝淺綠 down），其餘欄不上色。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/build_mc_table.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import PAIRS, fmt, mc, mc_exact_wr  # noqa: E402


def sign_td(v, nd: int) -> str:
    return f'<td class="{"up" if v > 0 else "down"}">{fmt(v, nd)}</td>'


def row(pair: str, r) -> str:
    tds = [f"<td>{pair}</td>", f"<td>{fmt(r['交易數'])}</td>",
           f"<td>{str(r['抽樣模式']).split('(')[0]}</td>",
           f"<td>{fmt(mc_exact_wr(pair), 1)}</td>", sign_td(r["每筆淨期望%"], 2),
           sign_td(r["報酬%_P5"], 1), sign_td(r["報酬%_中位"], 1), sign_td(r["報酬%_P95"], 1),
           f"<td>{fmt(r['區間寬度%'], 1)}</td>",
           # 本金大虧只有 5/10 不是 0，CSV 存兩位小數（99.98）；0 照 CSV 寫成 0.0
           f"<td>{r['本金大虧%']}</td>",
           f"<td>{int(r['最大連敗_P95'])}</td>", f"<td>{fmt(r['最大回撤%_P95'], 1)}</td>"]
    return "<tr>" + "".join(tds) + "</tr>"


def main() -> int:
    t = mc()
    print("\n".join(row(p, t.loc[p]) for p in PAIRS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
