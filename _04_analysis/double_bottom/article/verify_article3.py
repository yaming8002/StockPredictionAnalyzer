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
    pf = lambda e, p, liq: fmt("獲利因子", cell(df, e, p, liq, "獲利因子"))
    one = lambda e, p, liq, c: fmt(c, cell(df, e, p, liq, c))
    eff = lambda e: df[df["變體"] == e]["生效率%"]
    ups = lambda e, c="獲利因子": [a > b for a, b in six(df, e, c)]
    effp = lambda e, p: [cell(df, e, p, liq, "生效率%") for liq in (False, True)]
    pfd = lambda e: [a - b for a, b in six(df, e, "獲利因子")]
    ratio = lambda e: [a / b for a, b in six(df, e, "交易次數")]
    rng = lambda xs: f"{half_up(min(xs) * 100, 0)}%～{half_up(max(xs) * 100, 0)}%"
    return [
        # 站上年線
        ("年線 六格皆降、裸版降得多",
         not any(ups("ma200")) and all(
             (b1 - a1) > (b2 - a2) for (a1, b1), (a2, b2)
             in zip(six(df, "ma200", "獲利因子")[:3], six(df, "ma200", "獲利因子")[3:])),
         f"裸版降得比較多（3%：{pf('base', 3, False)} → {pf('ma200', 3, False)}），"
         f"可成交版降得比較少（9%：{pf('base', 9, True)} → {pf('ma200', 9, True)}）"),
        ("年線 生效率", True, f"生效率在 {pct_range(eff('ma200'))}"),
        # 長線多頭排列（MA20 > MA60 > MA120，2026-10-22 用戶改定）
        ("長線排列 生效率", True, f"生效率在 {pct_range(eff('align'))}"),
        ("長線排列 六格皆降", not any(ups("align")),
         f"六格的獲利因子都比基本版低（3% 裸版 {pf('base', 3, False)} → {pf('align', 3, False)}；"
         f"9% 裸版 {pf('base', 9, False)} → {pf('align', 9, False)}）"),
        ("長線排列 可成交降得比較少", max(-x for x in pfd("align")[3:]) < min(-x for x in pfd("align")[:3]),
         f"可成交版降得比較少（差 {half_up(-max(pfd('align')[3:]), 3)}～{half_up(-min(pfd('align')[3:]), 3)}）"),
        ("長線排列 筆數比例與總獲利皆少", all(a < b for a, b in six(df, "align", "總獲利(萬)")),
         f"筆數只剩約一半（{rng(ratio('align'))}）"),
        # 放量
        ("放量 生效率且 3% 最多", min(effp("volume", 3)) > max(effp("volume", 6) + effp("volume", 9)),
         f"生效率在 {pct_range(eff('volume'))}，3% 擋得最多"),
        ("放量 只有 6% 兩格升（總表：2 格高、門檻間方向相反）", ups("volume") == [False, True, False, False, True, False],
         f"3% 可成交那格降得最多（{pf('base', 3, True)} → {pf('volume', 3, True)}）"),
        ("放量 3% 可成交是降幅最大的一格", min(range(6), key=lambda i: pfd("volume")[i]) == 3, "降得最多"),
        ("放量 勝率六格皆升", all(ups("volume", "勝率%")),
         f"勝率六格都略升（例：3% 裸版 {one('base', 3, False, '勝率%')}% → {one('volume', 3, False, '勝率%')}%）"),
        # 第二底量縮
        ("量縮 生效率", True, f"生效率在 {pct_range(eff('volshrink'))}"),
        ("量縮 6/9% 四格升", [ups("volshrink")[i] for i in (1, 2, 4, 5)] == [True] * 4,
         f"6%、9% 四格的獲利因子都比基本版高（9% 裸版 {pf('base', 9, False)} → {pf('volshrink', 9, False)}）"),
        ("量縮 3% 兩格顯示相同、裸低可成交高",
         pf("volshrink", 3, False) == pf("base", 3, False) and pf("volshrink", 3, True) == pf("base", 3, True)
         and pfd("volshrink")[0] < 0 < pfd("volshrink")[3],
         f"（裸版低 {half_up(-pfd('volshrink')[0], 4)}、可成交高 {half_up(pfd('volshrink')[3], 4)}）"),
        ("量縮 筆數比例與總獲利皆少", all(a < b for a, b in six(df, "volshrink", "總獲利(萬)")),
         f"筆數只剩約六成（{rng(ratio('volshrink'))}）"),
        # 確認日 K 棒
        ("K 棒 生效率約 5%", 4.5 <= min(eff("bullbar")) and max(eff("bullbar")) < 6,
         f"只擋掉約 5% 的訊號（各格 {pct_range(eff('bullbar'))}）"),
        ("K 棒 九成五通過", 0.94 <= 1 - max(eff("bullbar")) / 100 and 1 - min(eff("bullbar")) / 100 <= 0.96,
         "本來就有九成五左右是紅 K 或長下影線"),
        ("K 棒 六格皆降、差距範圍", not any(ups("bullbar")),
         f"六格的獲利因子都比基本版略低（差 {half_up(-max(pfd('bullbar')), 3)}～{half_up(-min(pfd('bullbar')), 3)}）"),
        ("K 棒 勝率六格皆降", not any(ups("bullbar", "勝率%")), "勝率也六格都略降"),
        # 總結
        ("總表：量縮 5 升", sum(ups("volshrink")) == 5,
         f"**第二底量縮** 5 格高、1 格低：6%、9% 四格比較高，3% 兩格幾乎一樣（低的那格只差 "
         f"{half_up(-pfd('volshrink')[0], 4)}）"),
        ("總表：放量 2 升", sum(ups("volume")) == 2, "**放量** 2 格高、4 格低，高的是 6% 那兩格"),
        ("總表：年線／長線排列／K 棒 0 升", sum(ups("ma200")) == sum(ups("align")) == sum(ups("bullbar")) == 0,
         "**站上年線、長線多頭排列、確認日 K 棒**六格都比基本版低"),
        ("總表：三條擋掉一半上下", all(40 <= x <= 65 for e in ("align", "volume", "volshrink") for x in eff(e)),
         "都擋掉一半上下的訊號"),
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
