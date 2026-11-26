# -*- coding: utf-8 -*-
"""
產生〈島型反轉（二）〉〈島型反轉（三）〉進場優化兩篇的表格 HTML（基本版總覽、每條條件一張、最後的獲利因子總表）。

兩篇的分法（2026-11-18）：
  （二）進場的優化（上）＝島型本身與量價：前段跌幅、向上缺口、島寬、量能、確認日 K 棒（#23～#30、#33）
  （三）進場的優化（下）＝趨勢與指標：週線 5 條、MA200 兩向、均線排列、島內 RSI／K 值（#14～#22、#31、#32）
出場沿用〈島型反轉（一）〉的三種：持有 10 根、持有 20 根、出現頂部島型（替換、無時間上限）。
只用可成交格（成交金額 > 1,000 萬；同雙底進場篇 2026-11-11 用戶定的做法）。

資料來源＝SPA 單股回測輸出（`_02_strategy/island_strategy/island_entry_sweep.py`）：
  result/single_island_entry/<tag>/single_island_<tag>_trades.csv、diag.json
每組讀逐筆 trades 重算通用回測表 11 個指標欄精確值（fast_multi.exact_stats）、一次 half-up；生效率、未平倉率讀 diag.json。
位數同雙底進場篇：勝率／獲利平均／虧損平均／中位數 1 位，持有天、期望值、總獲利取整，獲利因子 3 位；
日均獲利(元) 1 位、日化PF 3 位（同獲利因子）。兩欄天數用日曆天（common.time_cost_exact）。
底色＝跟同出場的基本版比，只上勝率、獲利平均、中位數、期望值、獲利因子五欄；顯示一樣就不上色。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/island/article/build_entry_tables.py [2|3] [--fill]
輸出：終端印各表 HTML；result/single_island_entry/_article_entry_stats.csv（verify 用）
--verify：文章裡每張表必須與重算的 HTML 逐字相同（數字、位數、底色 class）。
--fill：把文章 md 裡的佔位 {{T:<條件代號>}}／{{T:summary}} 換成表格 HTML（文章路徑由 BLOG_DIR 推得）
"""
import html
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.island_strategy.single_island_strategy import RESULT_DIR  # noqa: E402
from _03_multi_strategy.base.fast_multi import EXACT, exact_stats  # noqa: E402

SRC = os.path.join(RESULT_DIR, "single_island_entry")
STATS_FILE = os.path.join(SRC, "_article_entry_stats.csv")
# 出場代號 → (tag 主體, 表格列名)
EXITS = {"h10": ("h10__none__none", "持有 10 根"), "h20": ("h20__none__none", "持有 20 根"),
         "topisland": ("rep__none__topisland", "出現頂部島型")}
BASE_TAG = {"h10": "h10", "h20": "h20", "topisland": "rep__none__topisland"}
ARTICLE_ENTRIES = {
    2: ("base", "drop15", "drop25", "gap1", "gap3", "k1", "k3plus", "vol_break", "vol_shrink", "redk_upper"),
    3: ("base", "wk_up", "wk_down", "wk_above20", "wk_ma20_up", "wk_cross4", "above200", "below200",
        "align20", "ma60_120", "rsi30", "kd20"),
}
ALL_ENTRIES = tuple(dict.fromkeys(ARTICLE_ENTRIES[2] + ARTICLE_ENTRIES[3]))
LABELS = {"base": "基本版", "drop15": "前段跌幅 ≥ 15%", "drop25": "前段跌幅 ≥ 25%",
          "gap1": "向上缺口 ≥ 1%", "gap3": "向上缺口 ≥ 3%", "k1": "單日島（1 根）", "k3plus": "島寬 ≥ 3 根",
          "vol_break": "判定日放量 1.5 倍", "vol_shrink": "島內量縮", "redk_upper": "判定日紅 K 收上半部",
          "wk_up": "週 MA5 > 週 MA20", "wk_down": "週 MA5 < 週 MA20", "wk_above20": "週收盤 > 週 MA20",
          "wk_ma20_up": "週 MA20 上揚", "wk_cross4": "週 MA5 4 週內上穿週 MA20",
          "above200": "收盤 > MA200", "below200": "收盤 < MA200", "align20": "MA20 > MA60 > MA120",
          "ma60_120": "MA60 > MA120", "rsi30": "島內 RSI < 30", "kd20": "島內 K 值 < 20"}
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%", "期望值", "日均獲利(元)",
        "獲利因子(PF)", "日化PF", "總獲利(萬)"]
PAINT = ("勝率%", "獲利平均%", "中位數%", "期望值", "獲利因子(PF)")
HEAD = ("<thead><tr><th>版本</th><th>交易次數</th><th>勝率%</th><th>平均持有天</th><th>獲利平均%</th>"
        "<th>虧損平均%</th><th>中位數%</th><th>期望值(EV)/筆(元)</th><th>日均獲利(元)</th><th>獲利因子(PF)</th>"
        "<th>日化PF</th><th>總獲利(萬)</th>"
        "</tr></thead>")
NOTE = ('<div class="bt-note">勝率%、獲利平均%、中位數%、期望值、獲利因子 五欄以底色標方向：'
        '<span style="background:#fde4ec">&nbsp;粉紅＝比基本版好&nbsp;</span>、'
        '<span style="background:#e7f4ea">&nbsp;淺綠＝比基本版差&nbsp;</span>；其餘欄不上色。</div>')


def half_up(v, nd: int) -> Decimal:
    return Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def nd_of(col: str) -> int:
    if col in ("交易次數", "平均持有天", "期望值", "總獲利(萬)"):
        return 0
    return 3 if col in ("獲利因子(PF)", "日化PF") else 1


def fmt(col: str, v) -> str:
    q = half_up(v, nd_of(col))
    return (f"{q:,}" if nd_of(col) == 0 else f"{q}").replace("-", "−")


def tag_of(entry: str, ex: str) -> str:
    if entry == "base":
        return f"{BASE_TAG[ex]}_liq1000"
    return f"in-{entry}__{EXITS[ex][0]}_liq1000"


def stats(entries=ALL_ENTRIES) -> pd.DataFrame:
    rows = []
    for ex in EXITS:
        for e in entries:
            tag = tag_of(e, ex)
            folder = os.path.join(SRC, tag)
            tr = common.read_result_csv(os.path.join(folder, f"single_island_{tag}_trades.csv"))
            exs = exact_stats(tr)
            with open(os.path.join(folder, "diag.json"), encoding="utf-8") as f:
                dg = json.load(f)
            n = int((pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0).sum())
            rows.append({"出場": ex, "變體": e, "交易次數": n,
                         **{c: exs[c + EXACT] for c in COLS if c != "交易次數"},
                         "signals": dg["signals"], "entries": dg["entries"],
                         "未平倉率%": dg["open_at_end"] / dg["entries"] * 100})
    df = pd.DataFrame(rows)
    base = df[df["變體"] == "base"].set_index("出場")["signals"]
    df["生效率%"] = [(1 - r["signals"] / base.loc[r["出場"]]) * 100 for _, r in df.iterrows()]
    return df


def pick(df, e, ex):
    return df[(df["變體"] == e) & (df["出場"] == ex)].iloc[0]


def cell_class(col: str, v, b) -> str:
    if col not in PAINT:
        return ""
    a, bb = half_up(v, nd_of(col)), half_up(b, nd_of(col))
    return "up" if a > bb else ("down" if a < bb else "")


def table(df: pd.DataFrame, entry: str) -> str:
    """一張表 3 列（三種出場，可成交）；entry＝base 時不上色、也不附 bt-note。"""
    out = ['<div class="table-wrap"><table class="bt-table">', HEAD, "<tbody>"]
    for ex, (_, name) in EXITS.items():
        r, b = pick(df, entry, ex), pick(df, "base", ex)
        tds = []
        for c in COLS:
            cls = "" if entry == "base" else cell_class(c, r[c], b[c])
            tds.append(f'<td class="{cls}">{fmt(c, r[c])}</td>' if cls else f"<td>{fmt(c, r[c])}</td>")
        out.append(f"<tr><td>{name}<br>＋ 成交金額 &gt; 1,000 萬</td>{''.join(tds)}</tr>")
    out.append("</tbody></table></div>")
    if entry != "base":
        out.append(NOTE)
    return "\n".join(out)


def span(lo, hi, nd: int) -> str:
    a, b = half_up(lo, nd), half_up(hi, nd)
    return f"{a}" if a == b else f"{a}～{b}"


def summary_table(df: pd.DataFrame, entries) -> str:
    """總表：各條件三種出場的獲利因子＋比基本版高的出場數＋生效率（不上色）。"""
    head = ("<thead><tr><th>進場條件</th><th>持有<br>10 根</th><th>持有<br>20 根</th><th>頂部<br>島型</th>"
            "<th>比基本版高</th><th>生效率%</th></tr></thead>")
    out = ['<div class="table-wrap"><table class="bt-table">', head, "<tbody>"]
    for e in entries:
        pfs, ups = [], 0
        for ex in EXITS:
            v, b = pick(df, e, ex)["獲利因子(PF)"], pick(df, "base", ex)["獲利因子(PF)"]
            pfs.append(fmt("獲利因子(PF)", v))
            ups += half_up(v, 3) > half_up(b, 3)
        eff = df[df["變體"] == e]["生效率%"]
        tail = ("<td>—</td><td>—</td>" if e == "base"
                else f"<td>{ups}／{len(EXITS)}</td><td>{span(eff.min(), eff.max(), 0)}</td>")
        label = html.escape(LABELS[e], quote=False)          # < > 轉成實體，避免被當成標籤
        name = f"<b>{label}</b>" if e == "base" else label
        out.append(f"<tr><td>{name}</td>{''.join(f'<td>{x}</td>' for x in pfs)}{tail}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


SLUGS = {2: "island-reversal-entry-filters", 3: "island-reversal-entry-filters-2"}


def render(df: pd.DataFrame, art: int) -> dict:
    """佔位代號 → 表格 HTML。"""
    out = {e: table(df, e) for e in ARTICLE_ENTRIES[art]}
    out["summary"] = summary_table(df, ARTICLE_ENTRIES[art])
    return out


def article_path(art: int) -> str:
    blog = os.environ.get("BLOG_DIR", os.path.join(_root, "..", "blog"))
    return os.path.normpath(os.path.join(blog, "site", "content", "posts", f"{SLUGS[art]}.md"))


def fill(df: pd.DataFrame, art: int) -> str:
    """把文章裡的 {{T:代號}} 換成表格；剩下沒換到的佔位視為錯誤。"""
    path = article_path(art)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for key, htm in render(df, art).items():
        text = text.replace("{{T:" + key + "}}", htm)
    if "{{T:" in text:
        raise ValueError(f"{path} 還有沒替換的佔位")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def main() -> int:
    which = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [2, 3]
    df = stats()
    if "--fill" in sys.argv:
        for art in which:
            print(f"→ {fill(df, art)}")
        return 0
    if "--verify" in sys.argv:                        # 文章裡的表必須與重算結果逐字相同
        bad = 0
        for art in which:
            with open(article_path(art), encoding="utf-8") as f:
                text = f.read()
            for key, htm in render(df, art).items():
                if htm not in text:
                    bad += 1
                    print(f"✗ （{art}）{key} 表格與重算不同")
        print("表格全部一致" if bad == 0 else f"共 {bad} 張不一致")
        return 1 if bad else 0
    df.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    for art in which:
        for e in ARTICLE_ENTRIES[art]:
            print(f"=== （{art}）{LABELS[e]} ===")
            print(table(df, e))
        print(f"=== （{art}）總表 ===")
        print(summary_table(df, ARTICLE_ENTRIES[art]))
    print(f"→ {STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
