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

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from build_article2_tables import ENTRIES, PCTS, fmt, half_up, stats, summary_table, table  # noqa: E402

SLUG = "double-bottom-entry-filters"


def cell(df, e, p, liq, col):
    return df[(df["變體"] == e) & (df["門檻%"] == p) & (df["可成交"] == liq)].iloc[0][col]


def six(df, e, col):
    """六格（裸版 3/6/9、可成交 3/6/9）的 (條件值, 基本版值)。"""
    return [(cell(df, e, p, liq, col), cell(df, "base", p, liq, col)) for liq in (False, True) for p in PCTS]


def pct_range(s) -> str:
    return f"{half_up(min(s), 0)}%～{half_up(max(s), 0)}%"


def claims(df) -> list:
    pf = lambda e, p, liq: fmt("獲利因子", cell(df, e, p, liq, "獲利因子"))
    one = lambda e, p, liq, c: fmt(c, cell(df, e, p, liq, c))
    eff = lambda e: df[df["變體"] == e]["生效率%"]
    ups = lambda e, c="獲利因子": [a > b for a, b in six(df, e, c)]
    effp = lambda e, p: [cell(df, e, p, liq, "生效率%") for liq in (False, True)]
    win_diff = [a - b for a, b in six(df, "neck", "勝率%")]
    neck_n = [a / b for a, b in six(df, "neck", "交易次數")][:3]
    out = [
        # 頸線突破
        ("頸線 生效率範圍", True, f"生效率在 {pct_range(eff('neck'))}"),
        ("頸線 六格 PF 皆升", all(ups("neck")),
         f"（3%：{pf('base', 3, False)} → {pf('neck', 3, False)}；9% 可成交：{pf('base', 9, True)} → {pf('neck', 9, True)}）"),
        ("頸線 勝率六格皆升", all(ups("neck", "勝率%")),
         f"提高 {half_up(min(win_diff), 1)}～{half_up(max(win_diff), 1)} 個百分點"),
        ("頸線 虧損平均六格皆變深", all(a < b for a, b in six(df, "neck", "虧損平均%")),
         f"9% 可成交從 {one('base', 9, True, '虧損平均%')}% 變成 {one('neck', 9, True, '虧損平均%')}%"),
        ("頸線 持有天數六格皆變長", all(a > b for a, b in six(df, "neck", "平均持有天")),
         f"9% 可成交從 {one('base', 9, True, '平均持有天')} 天變成 {one('neck', 9, True, '平均持有天')} 天"),
        ("頸線 裸版總獲利三格皆少、筆數約少三成",
         all(a < b for a, b in six(df, "neck", "總獲利(萬)")[:3]) and all(0.6 <= x <= 0.75 for x in neck_n),
         "裸版三格都比基本版少（筆數少了約三成）"),
        ("頸線 可成交總獲利三格皆多", all(a > b for a, b in six(df, "neck", "總獲利(萬)")[3:]),
         f"9%：{one('base', 9, True, '總獲利(萬)')} 萬 → {one('neck', 9, True, '總獲利(萬)')} 萬"),
        # 兩底間隔
        ("間隔 3% 生效率", True, f"3% 擋掉 {pct_range(effp('gap', 3))}"),
        ("間隔 6% 生效率", True, f"6% 約 {pct_range(effp('gap', 6))}"),
        ("間隔 9% 生效率", True, f"9% 只剩 {pct_range(effp('gap', 9))}"),
        ("間隔 只有 3% 兩格升、其餘差 < 0.02",
         ups("gap") == [True, False, False, True, False, False]
         and all(abs(a - b) < 0.02 for (a, b), u in zip(six(df, "gap", "獲利因子"), ups("gap")) if not u),
         f"（裸版 {pf('base', 3, False)} → {pf('gap', 3, False)}）"),
        # 第二底墊高
        ("墊高 生效率範圍且門檻越大越低",
         all(effp("rise", 3)[i] > effp("rise", 6)[i] > effp("rise", 9)[i] for i in (0, 1)),
         f"生效率在 {pct_range(eff('rise'))}，門檻越大越低"),
        ("墊高 只有 9% 兩格升、約 0.01",
         ups("rise") == [False, False, True, False, False, True]
         and all(abs(a - b) < 0.015 for (a, b), u in zip(six(df, "rise", "獲利因子"), ups("rise")) if u),
         "只有 9% 兩種口徑比基本版高"),
        # 前段跌幅
        ("跌幅 3% 生效率", min(effp("drop", 3)) > 84.5, f"3% 擋掉 {pct_range(effp('drop', 3))}"),
        ("跌幅 6% 生效率", True, f"6% 擋掉 {pct_range(effp('drop', 6))}"),
        ("跌幅 9% 生效率", True, f"9% 擋掉 {pct_range(effp('drop', 9))}"),
        ("跌幅 只有 3% 兩格升、勝率也升",
         ups("drop") == [True, False, False, True, False, False]
         and ups("drop", "勝率%")[0] and ups("drop", "勝率%")[3],
         f"（裸版 {pf('base', 3, False)} → {pf('drop', 3, False)}、可成交 {pf('base', 3, True)} → {pf('drop', 3, True)}），勝率也跟著升"),
        ("跌幅 9% 可成交", True, f"（9% 可成交 {pf('base', 9, True)} → {pf('drop', 9, True)}）"),
        ("跌幅 3% 裸版筆數", True,
         f"只剩下約 {half_up(cell(df, 'drop', 3, False, '交易次數') / 1e4, 1)} 萬筆"
         f"（基本版約 {half_up(cell(df, 'base', 3, False, '交易次數') / 1e4, 1)} 萬筆）"),
        # 型態高度
        ("高度 3% 生效率", min(effp("height", 3)) > 84, f"3% 擋掉約 85%（{pct_range(effp('height', 3))}）"),
        ("高度 6% 生效率", True, f"6% 約 {pct_range(effp('height', 6))}"),
        ("高度 9% 生效率", max(effp("height", 9)) < 2,
         f"9% 只有 {half_up(min(effp('height', 9)), 1)}%～{half_up(max(effp('height', 9)), 1)}%"),
        ("高度 9% 門檻下頸線下限 = 1/0.91 < 1.10", 1 / 0.91 < 1.10 and str(half_up(1 / 0.91, 3)) == "1.099", "約 × 1.099"),
        ("高度 3%/6% 四格降、3% 降最多",
         ups("height")[:2] == [False, False] and ups("height")[3:5] == [False, False]
         and cell(df, "height", 3, True, "獲利因子") < 1,
         f"（裸版 {pf('base', 3, False)} → {pf('height', 3, False)}、可成交 {pf('base', 3, True)} → {pf('height', 3, True)}，可成交那格跌破 1）"),
        ("高度 3% 裸版勝率降", not ups("height", "勝率%")[0],
         f"3% 裸版 {half_up(cell(df, 'base', 3, False, '勝率%'), 1)}% → {half_up(cell(df, 'height', 3, False, '勝率%'), 1)}%"),
        ("高度 9% 可成交只高一點且顯示相同",
         0 < cell(df, "height", 9, True, "獲利因子") - cell(df, "base", 9, True, "獲利因子") < 0.001
         and pf("height", 9, True) == pf("base", 9, True),
         f"可成交只高 {half_up(cell(df, 'height', 9, True, '獲利因子') - cell(df, 'base', 9, True, '獲利因子'), 4)}，表上顯示一樣"),
        # 總結
        ("總表：頸線 6 升", sum(ups("neck")) == 6, "**頸線突破**六格都比基本版高"),
        ("總表：間隔／墊高／跌幅 2 升",
         sum(ups("gap")) == sum(ups("rise")) == sum(ups("drop")) == 2, "都是 2 格高、4 格低"),
        ("總表：高度 1 升", sum(ups("height")) == 1,
         f"**型態高度** 1 格高、5 格低，高的那格只差 "
         f"{half_up(cell(df, 'height', 9, True, '獲利因子') - cell(df, 'base', 9, True, '獲利因子'), 4)}"),
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
