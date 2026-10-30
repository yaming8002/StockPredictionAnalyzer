# -*- coding: utf-8 -*-
"""
產生〈雙底（五）：出場的優化（下）〉的表格 HTML（基本版總覽、5 條一般出場各一張、十條出場的獲利因子總表）。
收 #17 保本停損、#18 ATR 移動停損、#19 自高點回落 20%、#20 跌破 MA10、#21 高檔 M 頭；
總表把〈雙底（四）〉的 #12～#16 也放進來，十條一起看。

算法、位數、底色沿用 build_article4_tables／build_article2_tables（同一批 result/single_double_bottom_exit/ 輸出），
這裡只換出場清單、另寫出場版的總表（多一欄搶先率）。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article5_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_exit/_article5_stats.csv（verify 用）
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from build_article2_tables import PCTS, fmt, half_up, table  # noqa: E402
from build_article4_tables import EXITS as EXITS4, LABELS, SRC, stats  # noqa: E402

EXITS = ("base", "breakeven", "atr", "trail20", "ma10", "mtop")
ALL_EXITS = EXITS4 + EXITS[1:]
STATS_FILE = os.path.join(SRC, "_article5_stats.csv")


def summary_table(df: pd.DataFrame, exits=ALL_EXITS) -> str:
    """總表：各出場 6 格的獲利因子＋比基本版高的格數＋生效率、搶先率範圍（不上色）。"""
    head = ("<thead><tr><th>出場</th><th>3%</th><th>6%</th><th>9%</th><th>3%<br>可成交</th>"
            "<th>6%<br>可成交</th><th>9%<br>可成交</th><th>比基本版高</th><th>生效率%</th><th>搶先率%</th>"
            "</tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for e in exits:
        pfs, ups = [], 0
        for liq in (False, True):
            for p in PCTS:
                v = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == e)].iloc[0]["獲利因子"]
                b = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == "base")].iloc[0]["獲利因子"]
                pfs.append(fmt("獲利因子", v))
                ups += v > b
        sub = df[df["變體"] == e]
        rng = lambda c: f"{half_up(sub[c].min(), 0)}～{half_up(sub[c].max(), 0)}"
        tail = ("<td>—</td><td>—</td><td>—</td>" if e == "base"
                else f"<td>{ups}／6</td><td>{rng('生效率%')}</td><td>{rng('搶先率%')}</td>")
        name = f"<b>{LABELS[e]}</b>" if e == "base" else LABELS[e]
        out.append(f"<tr><td>{name}</td>{''.join(f'<td>{x}</td>' for x in pfs)}{tail}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def main() -> int:
    df = stats(ALL_EXITS)
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for e in EXITS:
        print(f"=== {LABELS[e]} ===")
        print(table(df, e))
    print("=== 總表 ===")
    print(summary_table(df))
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
