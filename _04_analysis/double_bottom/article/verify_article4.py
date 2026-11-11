# -*- coding: utf-8 -*-
"""
驗〈雙底（四）：出場的優化（上）〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 6 張表（基本版總覽、5 條型態類出場）必須與 build_article4_tables
   由逐筆交易重算的 HTML 逐字相同（數字、位數、底色 class 都在內）。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。
   claims() 以 key 回傳，草稿裡的 {{C:key}} 也是用同一份字串填的（文章數字只有這一個來源）。

文章先找已發佈的 posts/<slug>.md，沒有就找 drafts/<slug>.md。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article4.py
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
from build_article2_tables import PCTS, fmt, half_up, table  # noqa: E402
from build_article4_tables import EXITS, SRC, stats  # noqa: E402
from verify_article2 import cell, pct_range, six  # noqa: E402

SLUG = "double-bottom-exit"
EARLY_DAYS = 4   # 頸線停利「買進後沒幾天就出場」的日曆天門檻（同 reference 2026-10-29 的近似）


def article_path(slug: str) -> str:
    blog = common.require_blog_dir()
    for path in (os.path.join(blog, "site", "content", "posts", f"{slug}.md"),
                 os.path.join(blog, "drafts", f"{slug}.md")):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"posts／drafts 都找不到 {slug}.md")


def early_share(rule: str, p: int, liq: bool) -> float:
    """持有日曆天 ≤ EARLY_DAYS 的交易佔比（%，排除淨損益 0，同 exact_stats 的母體）。"""
    tag = f"p{p}_{'liq1000' if liq else 'raw'}__{rule}"
    tr = pd.read_csv(os.path.join(SRC, tag, f"single_double_bottom_{tag}_trades.csv"))
    tr = tr[pd.to_numeric(tr["real_pnl"], errors="coerce").fillna(0) != 0]
    days = (pd.to_datetime(tr["sell_date"]) - pd.to_datetime(tr["buy_date"])).dt.days
    return float((days <= EARLY_DAYS).mean() * 100)


def arrow(df, e, p, liq, col, unit=""):
    """「9% −11.1% → −8.0%」這種前後對照字串（用戶 2026-11-11 拿掉裸版，只剩可成交格，不再標口徑）。"""
    tag = f"{p}%"
    return f"{tag} {fmt(col, cell(df, 'base', p, liq, col))}{unit} → {fmt(col, cell(df, e, p, liq, col))}{unit}"


def claims(df) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。"""
    col = lambda e, c: df[df["變體"] == e][c]
    down = lambda e, c: all(a < b for a, b in six(df, e, c))
    up = lambda e, c: all(a > b for a, b in six(df, e, c))
    diff = lambda e, c: [a - b for a, b in six(df, e, c)]
    pf2 = lambda e: f"{arrow(df, e, 3, True, '獲利因子')}；{arrow(df, e, 9, True, '獲利因子')}"
    rng = lambda xs, nd=1, unit="%": f"{half_up(min(xs), nd)}{unit}～{half_up(max(xs), nd)}{unit}"
    pfv = lambda e, p, liq: cell(df, e, p, liq, "獲利因子")
    neck_med = [cell(df, "neck", p, True, "中位數%") for p in PCTS]
    early = [early_share("neck", p, True) for p in (3, 9)]
    return {
        # 測幅滿足點
        "target_eff": (all(col("target", "生效率%") == col("target", "搶先率%")), pct_range(col("target", "生效率%"))),
        "target_pf": (down("target", "獲利因子") and pfv("target", 3, True) < 1, pf2("target")),
        "target_win": (up("target", "勝率%"), f"提高 {rng(diff('target', '勝率%'), unit='')} 個百分點"),
        "target_avgwin": (down("target", "獲利平均%") and min(diff("target", "勝率%")) > 0,
                          f"{arrow(df, 'target', 3, True, '獲利平均%', '%')}；{arrow(df, 'target', 9, True, '獲利平均%', '%')}"),
        # 打折測幅
        "target65_eff": (all(col("target65", "生效率%") == col("target65", "搶先率%"))
                         and min(col("target65", "生效率%")) > max(col("target", "生效率%")),
                         pct_range(col("target65", "生效率%"))),
        "target65_win": (all(a > b for a, b in zip(diff("target65", "勝率%"), diff("target", "勝率%"))),
                         f"提高 {rng(diff('target65', '勝率%'), unit='')} 個百分點"),
        "target65_pf": (down("target65", "獲利因子")
                        and all(a < b for (a, _), (b, _) in zip(six(df, "target65", "獲利因子"), six(df, "target", "獲利因子")))
                        and pfv("target65", 3, True) < 1 and abs(pfv("target65", 6, True) - 1) < 0.01,
                        pf2("target65")),
        # 頸線停利
        "neck_eff": (min(col("neck", "生效率%")) > max(col(e, "生效率%").max() for e in EXITS[1:] if e != "neck"),
                     pct_range(col("neck", "生效率%"))),
        "neck_win": (all(cell(df, "neck", p, True, "勝率%") >= 45 for p in PCTS),
                     "（" + "、".join(f"{fmt('勝率%', cell(df, 'neck', p, True, '勝率%'))}%" for p in PCTS) + "）"),
        "neck_median": (all(half_up(v, 1) >= 0 for v in neck_med) and all(b < 0 for _, b in six(df, "neck", "中位數%")),
                        "（" + "、".join(f"{p}% {fmt('中位數%', cell(df, 'neck', p, True, '中位數%'))}%" for p in PCTS) + "）"),
        "neck_avgwin": (max(col("neck", "獲利平均%")) < 10, rng(col("neck", "獲利平均%"))),
        "neck_hold": (max(col("neck", "平均持有天")) < 30, rng(col("neck", "平均持有天"), 0, " 天").replace(" 天～", "～")),
        "neck_pf": (all(v < 1 for v, _ in six(df, "neck", "獲利因子")) and pfv("neck", 9, True) == max(v for v, _ in six(df, "neck", "獲利因子")),
                    f"三格的獲利因子都在 1 以下（最高是 9% 的 {fmt('獲利因子', pfv('neck', 9, True))}）"),
        "neck_early": (True, f"3% 佔 {half_up(early[0], 0)}%、9% 佔 {half_up(early[1], 0)}%"),
        # 突破後跌回頸線
        "neckfail_eff": (True, pct_range(col("neckfail", "生效率%"))),
        "neckfail_pre": (all(b / a > 0.9 for a, b in zip(col("neckfail", "生效率%"), col("neckfail", "搶先率%"))),
                         pct_range(col("neckfail", "搶先率%"))),
        "neckfail_loss": (up("neckfail", "虧損平均%"), arrow(df, "neckfail", 9, True, "虧損平均%", "%")),
        "neckfail_median": (up("neckfail", "中位數%"), arrow(df, "neckfail", 9, True, "中位數%", "%")),
        "neckfail_avgwin": (down("neckfail", "獲利平均%"), arrow(df, "neckfail", 9, True, "獲利平均%", "%")),
        "neckfail_pf": (down("neckfail", "獲利因子"), pf2("neckfail")),
        # 跌破突破 K 棒低點
        "bolow_eff": (True, pct_range(col("bolow", "生效率%"))),
        "bolow_pre": (True, pct_range(col("bolow", "搶先率%"))),
        "bolow_loss": (up("bolow", "虧損平均%") and down("bolow", "獲利平均%"),
                       arrow(df, "bolow", 9, True, "虧損平均%", "%")),
        "bolow_pf": (down("bolow", "獲利因子"), pf2("bolow")),
        # 重點整理
        "all_down": (all(down(e, "獲利因子") and down(e, "平均持有天") for e in EXITS[1:])
                     and all(a > b > c for a, b, c in zip(*[[v for v, _ in six(df, e, "獲利因子")]
                                                            for e in ("target", "target65", "neck")]))
                     and all(a < b < c for a, b, c in zip(*[diff(e, "勝率%") for e in ("target", "target65", "neck")]))
                     and all(cell(df, "base", 3, l, "獲利因子") < cell(df, "base", 6, l, "獲利因子")
                             < cell(df, "base", 9, l, "獲利因子") for l in (True,)),
                     "五條、十五格的獲利因子都比基本版低"),
    }


def main() -> int:
    path = article_path(SLUG)
    text = io.open(path, encoding="utf-8").read()
    df = stats()
    errs = []
    want = [table(df, e) for e in EXITS]
    for e, html in zip(EXITS, want):
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
