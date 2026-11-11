# -*- coding: utf-8 -*-
"""
驗〈雙底（二）：進場的優化〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 7 張表（基本版總覽、5 條型態條件、總表）必須與 build_article2_tables 由逐筆交易
   重算的 HTML 逐字相同（數字、位數、底色 class 都在內）。
2. 正文主張：每條數字敘述在這裡重算（區間、單調、六格方向、比例……），判斷式必須成立，
   格式化後的字串必須逐字出現在文章裡。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article2.py
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

from decimal import Decimal  # noqa: E402

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from build_article2_tables import ENTRIES, PCTS, fmt, half_up, stats, summary_table, table  # noqa: E402

SLUG = "double-bottom-entry-filters"


def cell(df, e, p, liq, col):
    return df[(df["變體"] == e) & (df["門檻%"] == p) & (df["可成交"] == liq)].iloc[0][col]


def six(df, e, col):
    """可成交 3 格（3/6/9）的 (條件值, 基本版值)；用戶 2026-11-11 拿掉裸版，函式名沿用。"""
    return [(cell(df, e, p, True, col), cell(df, "base", p, True, col)) for p in PCTS]


def pct_range(s) -> str:
    lo, hi = half_up(min(s), 0), half_up(max(s), 0)
    return f"{lo}%" if lo == hi else f"{lo}%～{hi}%"


def claims(df) -> list:
    pf = lambda e, p: fmt("獲利因子", cell(df, e, p, True, "獲利因子"))
    one = lambda e, p, c: fmt(c, cell(df, e, p, True, c))
    eff = lambda e: df[(df["變體"] == e) & df["可成交"]]["生效率%"]
    ups = lambda e, c="獲利因子": [a > b for a, b in six(df, e, c)]
    effp = lambda e, p: [cell(df, e, p, True, "生效率%")]
    win_diff = [a - b for a, b in six(df, "neck", "勝率%")]
    neck_n = [a / b for a, b in six(df, "neck", "交易次數")]
    d9 = half_up(cell(df, "height", 9, True, "獲利因子") - cell(df, "base", 9, True, "獲利因子"), 4)
    out = [
        # 頸線突破
        ("頸線 生效率範圍", True, f"生效率在 {pct_range(eff('neck'))}，也就是大約三分之一"),
        ("頸線 三格 PF 皆升", all(ups("neck")),
         f"（3%：{pf('base', 3)} → {pf('neck', 3)}；9%：{pf('base', 9)} → {pf('neck', 9)}）"),
        ("頸線 勝率三格皆升", all(ups("neck", "勝率%")),
         f"提高 {half_up(min(win_diff), 1)}～{half_up(max(win_diff), 1)} 個百分點"),
        ("頸線 虧損平均三格皆變深", all(a < b for a, b in six(df, "neck", "虧損平均%")),
         f"9% 從 {one('base', 9, '虧損平均%')}% 變成 {one('neck', 9, '虧損平均%')}%"),
        ("頸線 持有天數三格皆變長", all(a > b for a, b in six(df, "neck", "平均持有天")),
         f"9% 從 {one('base', 9, '平均持有天')} 天變成 {one('neck', 9, '平均持有天')} 天"),
        ("頸線 筆數約少四分之一、總獲利三格皆多",
         all(0.70 <= x <= 0.76 for x in neck_n) and all(a > b for a, b in six(df, "neck", "總獲利(萬)")),
         f"交易次數少了約四分之一，總獲利三格都比基本版多（9%：{one('base', 9, '總獲利(萬)')} 萬 → {one('neck', 9, '總獲利(萬)')} 萬）"),
        # 兩底間隔
        ("間隔 生效率", True, f"3% 擋掉 {pct_range(effp('gap', 3))}，6% 約 {pct_range(effp('gap', 6))}，到 9% 只剩 {pct_range(effp('gap', 9))}"),
        ("間隔 只有 3% 升、其餘差 < 0.02",
         ups("gap") == [True, False, False]
         and all(abs(a - b) < 0.02 for (a, b), u in zip(six(df, "gap", "獲利因子"), ups("gap")) if not u),
         f"獲利因子只有 3% 比基本版高（{pf('base', 3)} → {pf('gap', 3)}）"),
        # 第二底墊高
        ("墊高 生效率範圍且門檻越大越低", effp("rise", 3)[0] > effp("rise", 6)[0] > effp("rise", 9)[0],
         f"生效率在 {pct_range(eff('rise'))}，門檻越大越低"),
        ("墊高 只有 9% 升、約 0.01",
         ups("rise") == [False, False, True]
         and all(abs(a - b) < 0.015 for (a, b), u in zip(six(df, "rise", "獲利因子"), ups("rise")) if u),
         "獲利因子只有 9% 比基本版高"),
        # 前段跌幅
        ("跌幅 生效率", min(effp("drop", 3)) > 84.5,
         f"3% 擋掉 {pct_range(effp('drop', 3))}，6% 擋掉 {pct_range(effp('drop', 6))}，9% 擋掉 {pct_range(effp('drop', 9))}"),
        ("跌幅 只有 3% 升、勝率也升", ups("drop") == [True, False, False] and ups("drop", "勝率%")[0],
         f"獲利因子只有 3% 比基本版高（{pf('base', 3)} → {pf('drop', 3)}），勝率也跟著升"),
        ("跌幅 9%", True, f"（9%：{pf('base', 9)} → {pf('drop', 9)}）"),
        ("跌幅 3% 筆數", True,
         f"只剩下約 {half_up(cell(df, 'drop', 3, True, '交易次數') / 1e4, 1)} 萬筆"
         f"（基本版約 {half_up(cell(df, 'base', 3, True, '交易次數') / 1e4, 1)} 萬筆）"),
        # 型態高度
        ("高度 生效率", min(effp("height", 3)) > 84 and max(effp("height", 9)) < 2,
         f"3% 擋掉約 {pct_range(effp('height', 3))}，6% 約 {pct_range(effp('height', 6))}，"
         f"9% 只有 {half_up(effp('height', 9)[0], 1)}%"),
        ("高度 9% 門檻下頸線下限 = 1/0.91 < 1.10", 1 / 0.91 < 1.10 and str(half_up(1 / 0.91, 3)) == "1.099", "約 × 1.099"),
        ("高度 3%/6% 降、3% 跌破 1", ups("height")[:2] == [False, False] and cell(df, "height", 3, True, "獲利因子") < 1,
         f"3% 降得最多（{pf('base', 3)} → {pf('height', 3)}，跌破 1）"),
        ("高度 3% 勝率降", not ups("height", "勝率%")[0],
         f"3%：{half_up(cell(df, 'base', 3, True, '勝率%'), 1)}% → {half_up(cell(df, 'height', 3, True, '勝率%'), 1)}%"),
        ("高度 9% 只高一點且顯示相同", 0 < d9 < Decimal("0.001") and pf("height", 9) == pf("base", 9),
         f"9% 幾乎沒變（只高 {d9}，表上顯示一樣）"),
        # 總結
        ("總表：頸線 3 升", sum(ups("neck")) == 3, "**頸線突破**三格都比基本版高"),
        ("總表：間隔／墊高／跌幅 1 升",
         sum(ups("gap")) == sum(ups("rise")) == sum(ups("drop")) == 1, "都是 1 格高、2 格低"),
        ("總表：高度 1 升", sum(ups("height")) == 1, f"**型態高度** 1 格高、2 格低，高的那格只差 {d9}"),
        ("總表：跌幅／高度 3% 皆擋 > 84%",
         min(effp("drop", 3) + effp("height", 3)) > 84, "在 3% 都擋掉八成五左右的訊號"),
    ]
    return out


def main() -> int:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{SLUG}.md")
    text = io.open(path, encoding="utf-8").read()
    df = stats()
    errs = []
    want = [table(df, e) for e in ENTRIES] + [summary_table(df)]
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
