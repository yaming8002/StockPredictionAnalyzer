# -*- coding: utf-8 -*-
"""
驗〈雙底（五）：出場的優化（下）〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 7 張表（基本版總覽、5 條一般出場、十條總表）必須與 build_article5_tables
   由逐筆交易重算的 HTML 逐字相同（數字、位數、底色 class 都在內）。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。
   輔助函式沿用 verify_article2／verify_article4。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article5.py
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

from build_article2_tables import PCTS, fmt, half_up, table  # noqa: E402
from build_article4_tables import stats  # noqa: E402
from build_article5_tables import ALL_EXITS, EXITS, summary_table  # noqa: E402
from verify_article2 import cell, pct_range, six  # noqa: E402
from verify_article4 import article_path, arrow  # noqa: E402

SLUG = "double-bottom-exit-2"
EXIT_LIST = ALL_EXITS   # 給草稿填字工具用：stats 要算十條＋基本版


def claims(df) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。只用可成交 3 格（用戶 2026-11-11 拿掉裸版）。"""
    col = lambda e, c: df[df["變體"] == e][c]
    down = lambda e, c: all(a < b for a, b in six(df, e, c))
    up = lambda e, c: all(a > b for a, b in six(df, e, c))
    pfd = lambda e: [a - b for a, b in six(df, e, "獲利因子")]
    by_pct = lambda e, c: [[cell(df, e, p, liq, c) for p in PCTS] for liq in (True,)]
    rising = lambda rows: all(r[0] < r[1] < r[2] for r in rows)
    pfdrop_rows = lambda e: [[-x for x in pfd(e)]]
    pf2 = lambda e: f"{arrow(df, e, 3, True, '獲利因子')}；{arrow(df, e, 9, True, '獲利因子')}"
    drop_rng = lambda e: f"{half_up(-max(pfd(e)), 3)}～{half_up(-min(pfd(e)), 3)}"
    below1 = lambda e: sum(v < 1 for v, _ in six(df, e, "獲利因子"))
    mean_drop = {e: -sum(pfd(e)) / len(pfd(e)) for e in ALL_EXITS[1:]}
    ranked = sorted(mean_drop, key=mean_drop.get)
    return {
        # 保本停損
        "be_eff": (max(col("breakeven", "生效率%")) < 7, pct_range(col("breakeven", "生效率%"))),
        "be_pfd": (down("breakeven", "獲利因子") and up("breakeven", "虧損平均%") and down("breakeven", "勝率%"),
                   drop_rng("breakeven")),
        # ATR 移動停損
        "atr_eff": (rising(by_pct("atr", "生效率%")) and by_pct("atr", "生效率%")[0][2] > 90,
                    pct_range(col("atr", "生效率%"))),
        "atr_hold": (down("atr", "平均持有天"), arrow(df, "atr", 9, True, "平均持有天", " 天")),
        "atr_pf": (down("atr", "獲利因子") and rising(pfdrop_rows("atr")), pf2("atr")),
        # 自高點回落 20%
        "tr_eff": (rising(by_pct("trail20", "生效率%")) and by_pct("trail20", "生效率%")[0][0] < 5
                   and all(38 < r[2] < 42 for r in by_pct("trail20", "生效率%")),
                   pct_range(col("trail20", "生效率%"))),
        "tr_pf": (down("trail20", "獲利因子") and rising(pfdrop_rows("trail20")), pf2("trail20")),
        # 跌破 MA10
        "ma10_eff": (min(col("ma10", "生效率%")) > 94, pct_range(col("ma10", "生效率%"))),
        "ma10_pre": (True, pct_range(col("ma10", "搶先率%"))),
        "ma10_hold": (True, f"{half_up(min(col('ma10', '平均持有天')), 0)}～{half_up(max(col('ma10', '平均持有天')), 0)} 天"),
        "ma10_pf": (below1("ma10") == 3,
                    f"{fmt('獲利因子', min(col('ma10', '獲利因子')))}～{fmt('獲利因子', max(col('ma10', '獲利因子')))}"),
        # 高檔 M 頭
        "mt_eff": (all(55 < x < 65 for x in col("mtop", "生效率%")), pct_range(col("mtop", "生效率%"))),
        "mt_pre": (all(18 < x < 23 for x in col("mtop", "搶先率%")), pct_range(col("mtop", "搶先率%"))),
        "mt_pfd": (down("mtop", "獲利因子"), drop_rng("mtop")),
        # 十條合併
        "all_down": (all(down(e, "獲利因子") for e in ALL_EXITS[1:]), "兩篇十條、三十格的獲利因子都比基本版低"),
        "least2": (ranked[:2] == ["breakeven", "mtop"], "降得最少的是**保本停損**和**高檔 M 頭**"),
        "below1": (below1("neck") == 3 and max(below1(e) for e in ALL_EXITS[1:] if e not in ("ma10", "neck")) < 3,
                   "**頸線停利**也三格都在 1 以下"),
        "base_pf": (rising(by_pct("base", "獲利因子")) and rising(by_pct("base", "平均持有天")),
                    "（" + "、".join(fmt("獲利因子", v) for v in by_pct("base", "獲利因子")[0]) + "）"),
        "all_hold": (all(down(e, "平均持有天") for e in ALL_EXITS[1:]), "三十格的平均持有天數全部比基本版短"),
    }


def main() -> int:
    path = article_path(SLUG)
    text = io.open(path, encoding="utf-8").read()
    df = stats(ALL_EXITS)
    errs = []
    want = [table(df, e) for e in EXITS] + [summary_table(df)]
    for e, html in zip(list(EXITS) + ["總表"], want):
        if html not in text:
            errs.append(f"表格與重算不一致：{e}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    cl = claims(df)
    for name, (ok, phrase) in cl.items():
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    print(f"{path}\n表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
