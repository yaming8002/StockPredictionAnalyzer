# -*- coding: utf-8 -*-
"""
產生〈雙底（二）：進場的優化〉的表格 HTML（基本版總覽、6 條條件各一張、最後的獲利因子總表）。

資料來源＝SPA 單股回測輸出（`_02_strategy/double_bottom_strategy/entry_filter_sweep.py`）：
  result/single_double_bottom_entry/p<3|6|9>_<raw|liq1000>__<base|neck|gap|rise|ma200|align|volume>/
每組讀逐筆 trades 重算規格 9 欄精確值（fast_multi.exact_stats）、一次 half-up；生效率讀 diag.json 的 signals。

位數同（一）（勝率／獲利平均／虧損平均／中位數 1 位，持有天、期望值、總獲利取整），**獲利因子改 3 位**：
條件和基本版的獲利因子常只差千分位（例 1.1929 對 1.1911），2 位會出現「顯示一樣卻上了底色」。
底色＝跟同門檻、同口徑的基本版比，只上勝率、獲利平均、中位數、期望值、獲利因子五欄；
比較用文章顯示的位數，顯示一樣就不上色（同 KD 系列 build_kd_single_tables 的做法）。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/article/build_article2_tables.py
輸出：終端印各表 HTML；result/single_double_bottom_entry/_article2_stats.csv（verify 用）
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

SRC = os.path.join(RESULT_DIR, "single_double_bottom_entry")
STATS_FILE = os.path.join(SRC, "_article2_stats.csv")
PCTS = (3, 6, 9)
ENTRIES = ("base", "neck", "gap", "rise", "ma200", "align", "volume")
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]
PAINT = ("勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子")
HEAD = ("<thead><tr><th>版本</th><th>交易次數</th><th>勝率%</th><th>平均持有天</th><th>獲利平均%</th>"
        "<th>虧損平均%</th><th>中位數%</th><th>期望值(EV)/筆(元)</th><th>獲利因子(PF)</th><th>總獲利(萬)</th>"
        "</tr></thead>")
NOTE = ('<div class="bt-note">勝率%、獲利平均%、中位數%、期望值、獲利因子 五欄以底色標方向：'
        '<span style="background:#fde4ec">&nbsp;粉紅＝比基本版好&nbsp;</span>、'
        '<span style="background:#e7f4ea">&nbsp;淺綠＝比基本版差&nbsp;</span>；其餘欄不上色。</div>')


def half_up(v, nd: int) -> Decimal:
    return Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def nd_of(col: str) -> int:
    if col in ("交易次數", "平均持有天", "期望值/筆", "總獲利(萬)"):
        return 0
    return 3 if col == "獲利因子" else 1


def fmt(col: str, v) -> str:
    q = half_up(v, nd_of(col))
    return (f"{q:,}" if nd_of(col) == 0 else f"{q}").replace("-", "−")


def stats() -> pd.DataFrame:
    rows = []
    for liq in (False, True):
        for p in PCTS:
            for e in ENTRIES:
                tag = f"p{p}_{'liq1000' if liq else 'raw'}__{e}"
                folder = os.path.join(SRC, tag)
                tr = pd.read_csv(os.path.join(folder, f"single_double_bottom_{tag}_trades.csv"))
                ex = exact_stats(tr)
                with open(os.path.join(folder, "diag.json"), encoding="utf-8") as f:
                    dg = json.load(f)
                n = int((pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0).sum())
                rows.append({"門檻%": p, "可成交": liq, "變體": e, "交易次數": n,
                             **{c: ex[c + EXACT] for c in COLS if c != "交易次數"},
                             "signals": dg["signals"], "entries": dg["entries"],
                             "未平倉率%": dg["open_at_end"] / dg["entries"] * 100})
    df = pd.DataFrame(rows)
    base = df[df["變體"] == "base"].set_index(["門檻%", "可成交"])["signals"]
    df["生效率%"] = [(1 - r["signals"] / base.loc[(r["門檻%"], r["可成交"])]) * 100 for _, r in df.iterrows()]
    return df


def row_label(p: int, liq: bool) -> str:
    return f"ZigZag {p}%" + ("<br>＋ 成交金額 &gt; 1,000 萬" if liq else "")


def cell_class(col: str, v, b) -> str:
    if col not in PAINT:
        return ""
    a, bb = half_up(v, nd_of(col)), half_up(b, nd_of(col))
    return "up" if a > bb else ("down" if a < bb else "")


def table(df: pd.DataFrame, entry: str) -> str:
    """一張表 6 列（裸版 3/6/9、可成交 3/6/9）；entry＝base 時不上色、也不附 bt-note。"""
    out = ['<div class="table-wrap"><table class="bt-table">', HEAD, "<tbody>"]
    for liq in (False, True):
        for p in PCTS:
            r = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == entry)].iloc[0]
            b = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == "base")].iloc[0]
            tds = []
            for c in COLS:
                cls = "" if entry == "base" else cell_class(c, r[c], b[c])
                tds.append(f'<td class="{cls}">{fmt(c, r[c])}</td>' if cls else f"<td>{fmt(c, r[c])}</td>")
            out.append(f"<tr><td>{row_label(p, liq)}</td>{''.join(tds)}</tr>")
    out.append("</tbody></table></div>")
    if entry != "base":
        out.append(NOTE)
    return "\n".join(out)


LABELS = {"base": "基本版", "neck": "頸線突破", "gap": "兩底間隔 ≥ 10 天", "rise": "第二底墊高 ≥ 2%",
          "ma200": "站上年線", "align": "均線多頭排列", "volume": "放量 1.5 倍"}


def summary_table(df: pd.DataFrame) -> str:
    """總表：各條件 6 格的獲利因子＋比基本版高的格數＋生效率範圍（不上色）。"""
    head = ("<thead><tr><th>進場條件</th><th>3%</th><th>6%</th><th>9%</th><th>3%<br>可成交</th>"
            "<th>6%<br>可成交</th><th>9%<br>可成交</th><th>比基本版高</th><th>生效率%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for e in ENTRIES:
        pfs, ups = [], 0
        for liq in (False, True):
            for p in PCTS:
                v = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == e)].iloc[0]["獲利因子"]
                b = df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["變體"] == "base")].iloc[0]["獲利因子"]
                pfs.append(fmt("獲利因子", v))
                ups += v > b
        eff = df[df["變體"] == e]["生效率%"]
        tail = ("<td>—</td><td>—</td>" if e == "base"
                else f"<td>{ups}／6</td><td>{half_up(eff.min(), 0)}～{half_up(eff.max(), 0)}</td>")
        name = f"<b>{LABELS[e]}</b>" if e == "base" else LABELS[e]
        out.append(f"<tr><td>{name}</td>{''.join(f'<td>{x}</td>' for x in pfs)}{tail}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def main() -> int:
    df = stats()
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for e in ENTRIES:
        print(f"=== {LABELS[e]} ===")
        print(table(df, e))
    print("=== 總表 ===")
    print(summary_table(df))
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
