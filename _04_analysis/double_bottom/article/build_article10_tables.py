# -*- coding: utf-8 -*-
"""
產生〈雙底（十）：多股共用資金 · 固定比例投入〉的表格 HTML。

份數表、完整結果表、資金曲線表與〈雙底（九）〉同一套產生器（build_article9_tables，--mode 比例）；
本檔另出一張「固定金額 vs 固定比例」對照表：五組的低價優先列，兩種投法並排。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article10_tables.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_article2_tables import half_up  # noqa: E402
from build_article9_tables import all_tables, block_order, label, load, random_note  # noqa: E402

MODE = "比例"


def compare_table() -> str:
    """低價優先列：定額與比例的總獲利、比例÷定額、交易次數、獲利因子、年化報酬、最大回撤。依比例的總獲利排。"""
    fixed, pct = load("定額"), load("比例")
    lo_f = fixed[fixed["排序"] == "低價"].set_index("交易策略")
    lo_p = pct[pct["排序"] == "低價"].set_index("交易策略")
    head = ("<thead><tr><th>交易策略<br>（低價優先）</th><th>定額<br>總獲利(萬)</th><th>比例<br>總獲利(萬)</th>"
            "<th>比例÷定額</th><th>定額<br>交易次數</th><th>比例<br>交易次數</th><th>定額<br>獲利因子</th>"
            "<th>比例<br>獲利因子</th><th>定額<br>最大回撤%</th><th>比例<br>最大回撤%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for name in block_order(pct):
        f, p = lo_f.loc[name], lo_p.loc[name]
        vals = [f"{half_up(f['總獲利(萬)'], 0):,}", f"{half_up(p['總獲利(萬)'], 0):,}",
                f"{half_up(p['總獲利(萬)'] / f['總獲利(萬)'], 2)}",
                f"{half_up(f['交易次數'], 0):,}", f"{half_up(p['交易次數'], 0):,}",
                f"{half_up(f['獲利因子'], 4)}", f"{half_up(p['獲利因子'], 4)}",
                f"{half_up(f['最大回撤(%)'], 1)}", f"{half_up(p['最大回撤(%)'], 1)}"]
        out.append(f"<tr><td>{label(name)}</td>{''.join(f'<td>{v}</td>' for v in vals)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def all_tables10() -> dict:
    return all_tables(MODE) | {"對照": compare_table()}


def main() -> int:
    for k, html in all_tables10().items():
        print(f"=== {k} ===\n{html}\n")
    print("=== 隨機列資金倍數中位（P5／P95）===\n" + random_note(load(MODE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
