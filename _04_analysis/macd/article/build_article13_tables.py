# -*- coding: utf-8 -*-
"""
產生（十三）結論篇的兩張表 HTML 列。

資料來源（回測程式全部在 SPA，這支只負責排版）：
  result/macd_multi/macd_multi_result.csv      （十一）（十二）全期 2002–2025 的 30 組（盤點表）
  result/macd_multi/macd_multi_conclusion.csv  前 TOP_N 名的 2015–2025 ＋ 0050
                                               （_04_analysis/macd/macd_multi_conclusion.py）

表一（盤點）：30 組依全期總獲利取前 TOP_N，欄位沿用（十一）（十二）主表。
表二（跟 0050 比）：0050 ＋ 同 TOP_N 組在 2015–2025 的資金倍數／年化／回撤／報酬回撤比，依全期名次排。

數值一律用「<欄>_精確」未四捨五入值一次 half-up 進位（同 build_article11_tables）。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/macd/article/build_article13_tables.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from build_article11_tables import COLS, DET, SPA, fmt, half_up, with_exact  # noqa: E402
from _04_analysis.macd.macd_multi_conclusion import TOP_N  # noqa: E402

CONC = os.path.join(SPA, "macd_multi_conclusion.csv")
BENCH = "0050 買進持有（含息）"
T2_COLS = ["資金倍數", "年化報酬率%", "最大回撤%", "報酬回撤比"]


def fmt2(col: str, v) -> str:
    """表二的位數：同 KD（八）與先前的（十三）。"""
    if col in ("資金倍數", "年化報酬率%"):
        return f"{half_up(v, 2)}"
    if col == "最大回撤%":
        return f"{half_up(v, 1)}"
    return f"{half_up(v, 3)}"


def label(name: str) -> str:
    """表格裡的 < > 要轉義，「RSI<50且上升」補上文章慣用的空白；出場一律跌破年線，表裡省略。"""
    name = name.replace(" × 跌破年線", "").replace("RSI<50且上升", "RSI<50 且上升")
    return name.replace("<", "&lt;").replace(">", "&gt;")


def ranked() -> pd.DataFrame:
    """30 組依全期總獲利（精確值）由高到低，附名次。"""
    df = with_exact(pd.read_csv(DET, encoding="utf-8-sig"))
    df = df.sort_values("總獲利(萬)", ascending=False).reset_index(drop=True)
    df.insert(0, "全期名次", range(1, len(df) + 1))
    return df


def conclusion() -> pd.DataFrame:
    return with_exact(pd.read_csv(CONC, encoding="utf-8-sig"))


def table1(rk: pd.DataFrame) -> str:
    out = []
    for _, r in rk.head(TOP_N).iterrows():
        cells = "".join(f"<td>{fmt(c, r[c])}</td>" for c in COLS)
        out.append(f"<tr><td>{r['全期名次']}</td><td>{label(r['交易策略'])}</td>"
                   f"<td>{r['投法']}</td><td>{r['排序']}</td>{cells}</tr>")
    return "\n".join(out)


def table2(conc: pd.DataFrame) -> str:
    b = conc[conc["交易策略"] == BENCH].iloc[0]
    out = ["<tr><td><b>0050 買進持有（含息）</b></td>"
           + "".join(f"<td><b>{fmt2(c, b[c])}</b></td>" for c in T2_COLS) + "</tr>"]
    picks = conc[conc["交易策略"] != BENCH].sort_values("全期名次")
    for _, r in picks.iterrows():
        cells = "".join(f"<td>{fmt2(c, r[c])}</td>" for c in T2_COLS)
        out.append(f"<tr><td>{label(r['交易策略'])} · {r['投法']} · {r['排序']}</td>"
                   f"{cells}</tr>")
    return "\n".join(out)


def main():
    print(f"=== 表一：（十一）（十二）全期總獲利前 {TOP_N} ===")
    print(table1(ranked()))
    print("\n=== 表二：2015–2025 對 0050 ===")
    print(table2(conclusion()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
