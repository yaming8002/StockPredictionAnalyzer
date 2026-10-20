# -*- coding: utf-8 -*-
"""
產生〈雙底（一）〉兩張結果表的 HTML 列，並另存文章數字用的精確值。

資料來源＝SPA 單股回測輸出（回測程式在 `_02_strategy/double_bottom_strategy/`，這支只負責排版）：
  result/single_double_bottom/p<門檻>_raw/      無流動性門檻（表一）
  result/single_double_bottom/p<門檻>_liq1000/  成交金額 > 1,000 萬（表二）
每組讀逐筆 trades CSV 重算規格 9 欄的精確值（fast_multi.exact_stats，口徑同 summarize_trades：
排除淨損益 0、金額類用淨損益、報酬率類用毛報酬），再一次 half-up 進位；diag.json 給出場原因分布。

位數比照 MACD（一）：勝率／獲利平均／虧損平均／中位數 1 位、持有天與期望值與總獲利取整、獲利因子 2 位。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article1_tables.py
輸出：終端印兩張表的 <tr> 列；result/single_double_bottom/_article1_stats.csv（verify 用）
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.double_bottom_strategy.single_double_bottom_strategy import RESULT_DIR  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats  # noqa: E402

SRC = os.path.join(RESULT_DIR, "single_double_bottom")
STATS_FILE = os.path.join(SRC, "_article1_stats.csv")
PCTS = range(2, 11)
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]


def half_up(v, nd: int) -> Decimal:
    """四捨五入（.5 一律進位）；用 Decimal 避開銀行家進位與二進位誤差。"""
    return Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def fmt(col: str, v) -> str:
    """文章顯示位數；負號用全形減號（同系列其他文章）。"""
    if col == "交易次數":
        s = f"{half_up(v, 0):,}"
    elif col in ("平均持有天", "期望值/筆", "總獲利(萬)"):
        s = f"{half_up(v, 0):,}"
    elif col == "獲利因子":
        s = f"{half_up(v, 2)}"
    else:
        s = f"{half_up(v, 1)}"
    return s.replace("-", "−")


def stats() -> pd.DataFrame:
    """18 組（9 門檻 × 裸版／可成交）的精確統計＋出場原因分布。"""
    rows = []
    for liq in (False, True):
        for p in PCTS:
            tag = f"p{p}_{'liq1000' if liq else 'raw'}"
            folder = os.path.join(SRC, tag)
            tr = pd.read_csv(os.path.join(folder, f"single_double_bottom_{tag}_trades.csv"))
            ex = exact_stats(tr)
            n = int((pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0).sum())
            with open(os.path.join(folder, "diag.json"), encoding="utf-8") as f:
                dg = json.load(f)
            n_exit = dg["exit_choch"] + dg["exit_low"] + dg["exit_both"]
            rows.append({"tag": tag, "門檻%": p, "可成交": liq, "交易次數": n,
                         **{c: ex[c + EXACT] for c in COLS if c != "交易次數"},
                         "進場數": dg["entries"], "出場_CHoCH": dg["exit_choch"],
                         "出場_跌破低點": dg["exit_low"], "出場_同日": dg["exit_both"],
                         "CHoCH占比%": dg["exit_choch"] / n_exit * 100,
                         "跌破低點占比%": dg["exit_low"] / n_exit * 100,
                         "同日占比%": dg["exit_both"] / n_exit * 100,
                         "期末未平倉": dg["open_at_end"],
                         "未平倉率%": dg["open_at_end"] / dg["entries"] * 100})
    return pd.DataFrame(rows)


def label(p: int, liq: bool) -> str:
    return f"ZigZag {p}%" + ("<br>＋ 成交金額 &gt; 1,000 萬" if liq else "")


def table_rows(df: pd.DataFrame, liq: bool) -> str:
    out = []
    for _, r in df[df["可成交"] == liq].iterrows():
        cells = "".join(f"<td>{fmt(c, r[c])}</td>" for c in COLS)
        out.append(f"<tr><td>{label(int(r['門檻%']), liq)}</td>{cells}</tr>")
    return "\n".join(out)


def main() -> int:
    df = stats()
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    print("=== 表一：無流動性門檻 ===")
    print(table_rows(df, False))
    print("\n=== 表二：成交金額 > 1,000 萬 ===")
    print(table_rows(df, True))
    print(f"\n→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
