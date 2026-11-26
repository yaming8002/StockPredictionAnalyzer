# -*- coding: utf-8 -*-
"""
產生〈雙底（十一）：一路走來的總結評估〉的兩張表 HTML（結構同〈MACD（十三）〉build_article13_tables）。

資料來源（回測程式全部在 SPA，這支只負責排版）：
  db_multi_result.csv       〈雙底（九）〉〈雙底（十）〉全期 2002–2025（盤點表；隨機列不列入）
  db_multi_conclusion.csv   前 TOP_N 名的 2015–2025 ＋ 0050（_04_analysis/double_bottom/db_multi_conclusion.py）

表一（盤點）：30 組依全期總獲利取前 TOP_N，另開「全期名次」欄（名次不塞進交易策略欄）。
表二（跟 0050 比）：0050 ＋ 同 TOP_N 組在 2015–2025 的總獲利／年化／回撤／報酬回撤比（期末全部賣出口徑），依總獲利排。
數值一律用「<欄>_精確」一次 half-up 進位。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article11_tables.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _04_analysis.double_bottom.db_multi_conclusion import ALL_FILE, BENCH, CONC_FILE, TOP_N  # noqa: E402
from build_article2_tables import half_up  # noqa: E402
from build_article9_tables import COLS, EXACT, HEAD, fmt, label, load  # noqa: E402

T2_COLS = ["資金倍數", "年化報酬率%", "最大回撤%", "報酬回撤比"]   # 讀檔時換成精確值的欄位
T2_SHOW = ["年化報酬率%", "最大回撤%", "報酬回撤比"]               # 表二實際列出的欄位
P15 = "2015總獲利(萬)" + EXACT     # 2015–2025 期末全部賣出後的總獲利（表二的排序依據）


def ranked() -> pd.DataFrame:
    """兩投法 × 三種固定排序共 30 組，依全期總獲利（精確值）由高到低，附名次。"""
    df = pd.concat([load("定額"), load("比例")])
    df = df[df["排序"] != "隨機"].sort_values("總獲利(萬)", ascending=False).reset_index(drop=True)
    df.insert(0, "全期名次", range(1, len(df) + 1))
    return df


def _exact(path: str) -> pd.DataFrame:
    df = common.read_result_csv(path, encoding="utf-8-sig")
    for c in T2_COLS:
        df[c] = df[c + EXACT]
    return df


def conclusion() -> pd.DataFrame:
    return _exact(CONC_FILE)


def all_2015() -> pd.DataFrame:
    return _exact(ALL_FILE)


def fmt2(col: str, v) -> str:
    if col in ("資金倍數", "年化報酬率%"):
        return f"{half_up(v, 2)}"
    if col == "最大回撤%":
        return f"{half_up(v, 1)}"
    return f"{half_up(v, 3)}"


def table1(rk: pd.DataFrame) -> str:
    head = ("<thead><tr><th>全期<br>名次</th><th>交易策略</th><th>投法</th><th>買入<br>排序</th>"
            + "".join(f"<th>{h}</th>" for h in HEAD) + "</tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for _, r in rk.head(TOP_N).iterrows():
        cells = "".join(f"<td>{fmt(c, r[c])}</td>" for c in COLS)
        out.append(f"<tr><td>{r['全期名次']}</td><td>{label(r['交易策略'])}</td><td>{r['投法']}</td>"
                   f"<td>{r['排序']}</td>{cells}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def picks_2015(conc: pd.DataFrame) -> pd.DataFrame:
    """前 TOP_N 組（不含 0050），依 2015–2025 總獲利（精確值）由高到低，附本表名次（1 起連號）。"""
    df = conc[conc["交易策略"] != BENCH].sort_values(P15, ascending=False).reset_index(drop=True)
    df.insert(0, "同期名次", range(1, len(df) + 1))
    return df


def table2(conc: pd.DataFrame) -> str:
    # 只放 2015–2025 的數字跟 0050 比（全期數字不同期間、不可比，用戶 2026-11-16 定）。
    # 名次欄＝本表排序（2015–2025 期末全部賣出後的總獲利）的名次，必須跟列順序一致
    # 有總獲利就不列資金倍數（兩欄是同一個數換算，並列反而讓人疑問；用戶 2026-11-16 定）
    head = ("<thead><tr><th>名次</th><th>做法</th><th>總獲利<br>(萬)</th>"
            "<th>年化<br>報酬率%</th><th>最大<br>回撤%</th><th>報酬<br>回撤比</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    b = conc[conc["交易策略"] == BENCH].iloc[0]
    out.append(f"<tr><td>—</td><td><b>0050 買進持有（含息）</b></td><td><b>{fmt('總獲利(萬)', b[P15])}</b></td>"
               + "".join(f"<td><b>{fmt2(c, b[c])}</b></td>" for c in T2_SHOW) + "</tr>")
    for _, r in picks_2015(conc).iterrows():
        cells = "".join(f"<td>{fmt2(c, r[c])}</td>" for c in T2_SHOW)
        out.append(f"<tr><td>{r['同期名次']}</td><td>{label(r['交易策略'])}<br>{r['投法']} · {r['排序']}</td>"
                   f"<td>{fmt('總獲利(萬)', r[P15])}</td>{cells}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def all_tables11() -> dict:
    return {"盤點": table1(ranked()), "對0050": table2(conclusion())}


def main() -> int:
    for k, html in all_tables11().items():
        print(f"=== {k} ===\n{html}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
