# -*- coding: utf-8 -*-
"""
驗〈雙底（三）：進場的優化（下）〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 7 張表（基本版總覽、5 條技術分析類條件、總表）必須與 build_article3_tables
   由逐筆交易重算的 HTML 逐字相同（數字、位數、底色 class 都在內）。
2. 正文主張：每條數字敘述在這裡重算（區間、六格方向、比例……），判斷式必須成立，
   格式化後的字串必須逐字出現在文章裡。輔助函式沿用 verify_article2。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article3.py
"""
import io
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

from _02_strategy.base.vbt import common  # noqa: E402
from build_article2_tables import fmt, half_up, stats, summary_table, table  # noqa: E402
from build_article3_tables import ENTRIES  # noqa: E402
from verify_article2 import cell, pct_range, six  # noqa: E402

SLUG = "double-bottom-entry-filters-2"


def claims(df) -> list:
    """只用可成交 3 格（用戶 2026-11-11 拿掉裸版）。"""
    pf = lambda e, p: fmt("獲利因子", cell(df, e, p, True, "獲利因子"))
    eff = lambda e: df[(df["變體"] == e) & df["可成交"]]["生效率%"]
    ups = lambda e, c="獲利因子": [a > b for a, b in six(df, e, c)]
    pfd = lambda e: [a - b for a, b in six(df, e, "獲利因子")]
    ratio = lambda e: [a / b for a, b in six(df, e, "交易次數")]
    rng = lambda xs, nd: f"{half_up(min(xs), nd)}～{half_up(max(xs), nd)}"
    pct = lambda xs: f"{half_up(min(xs) * 100, 0)}%～{half_up(max(xs) * 100, 0)}%"
    win = lambda e, p: half_up(cell(df, e, p, True, "勝率%"), 1)
    vs_d3 = half_up(pfd("volshrink")[0], 4)
    return [
        ("年線 三格皆降、3% 降最多", not any(ups("ma200")) and min(pfd("ma200")) == pfd("ma200")[0],
         f"三格的獲利因子都比基本版低（3%：{pf('base', 3)} → {pf('ma200', 3)}；9%：{pf('base', 9)} → {pf('ma200', 9)}）"),
        ("年線 生效率", True, f"生效率在 {pct_range(eff('ma200'))}"),
        ("長線排列 生效率", True, f"生效率在 {pct_range(eff('align'))}，約一半"),
        ("長線排列 三格皆降", not any(ups("align")),
         f"三格的獲利因子都比基本版低，差 {rng([-x for x in pfd('align')], 3)}（9%：{pf('base', 9)} → {pf('align', 9)}）"),
        ("長線排列 筆數與總獲利皆少", all(a < b for a, b in six(df, "align", "總獲利(萬)")),
         f"筆數只剩約一半（{pct(ratio('align'))}），總獲利每一格都比基本版少很多"),
        ("放量 生效率 3% 最多", eff("volume").idxmax() == df[(df["變體"] == "volume") & df["可成交"] & (df["門檻%"] == 3)].index[0],
         f"生效率在 {pct_range(eff('volume'))}，3% 擋得最多"),
        ("放量 只有 6% 升、3% 降最多", ups("volume") == [False, True, False] and min(pfd("volume")) == pfd("volume")[0],
         f"獲利因子只有 6% 比基本版高，3%、9% 兩格都比較低；3% 降得最多（{pf('base', 3)} → {pf('volume', 3)}）"),
        ("放量 勝率三格皆升", all(ups("volume", "勝率%")), f"勝率三格都略升（例：3%：{win('base', 3)}% → {win('volume', 3)}%）"),
        ("量縮 生效率", True, f"生效率約 {pct_range(eff('volshrink'))}"),
        ("量縮 6/9 升、3% 顯示相同", ups("volshrink")[1:] == [True, True] and pf("volshrink", 3) == pf("base", 3) and vs_d3 > 0,
         f"6%、9% 兩格的獲利因子比基本版高（9%：{pf('base', 9)} → {pf('volshrink', 9)}）；3% 跟基本版幾乎一樣，表上顯示相同（只高 {vs_d3}）"),
        ("量縮 筆數與總獲利皆少", all(a < b for a, b in six(df, "volshrink", "總獲利(萬)")),
         f"因為筆數只剩約五成多（{pct(ratio('volshrink'))}），總獲利每一格都比基本版少"),
        ("紅K 生效率", True, f"這條只擋掉約 {pct_range(eff('bullbar'))} 的訊號"),
        ("紅K 三格略降", not any(ups("bullbar")) and not any(ups("bullbar", "勝率%")),
         f"三格的獲利因子都比基本版略低（差 {rng([-x for x in pfd('bullbar')], 3)}），勝率也三格都略降"),
        ("總表 量縮 3 升", sum(ups("volshrink")) == 3, f"**第二底量縮**三格都比基本版高：6%、9% 比較明顯，3% 只高 {vs_d3}"),
        ("總表 放量 1 升", sum(ups("volume")) == 1, "**放量** 1 格高、2 格低，高的是 6%"),
        ("總表 三條全降", sum(ups("ma200")) == sum(ups("align")) == sum(ups("bullbar")) == 0,
         "**站上年線、長線多頭排列、確認日 K 棒**三格都比基本版低"),
    ]


def main() -> int:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{SLUG}.md")
    text = io.open(path, encoding="utf-8").read()
    df = stats(ENTRIES)
    errs = []
    want = [table(df, e) for e in ENTRIES] + [summary_table(df, ENTRIES)]
    for e, html in zip(list(ENTRIES) + ["總表"], want):
        if html not in text:
            errs.append(f"表格與重算不一致：{e}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    cl = claims(df)
    for name, ok, phrase in cl:
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    print(f"表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
