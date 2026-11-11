# -*- coding: utf-8 -*-
"""
驗〈雙底（六）：出場的替換〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 11 張表（基本版、保留組 5 條＋加警告的 2 張回檔表、總表、移除組未賣出比例、未賣出概況）必須與
   build_article6_tables 由逐筆交易／open_positions.csv 重算的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。
   草稿裡的 {{C:key}} 也是用同一份字串填的。輔助函式沿用 verify_article2／verify_article4。
3. 回檔表（加警告的出場）要讀股價 parquet，執行前設 STOCK_DATA_DIR。

執行：
    BLOG_DIR=F:/stock-analyzer/blog STOCK_DATA_DIR=F:/stock-analyzer/data/stock_data PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article6.py
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

from build_article2_tables import fmt, half_up  # noqa: E402
from build_article4_tables import stats  # noqa: E402
from build_article6_tables import (ALL_EXITS, OPEN_MAX, SRC, WARN, all_tables, drawdown,  # noqa: E402
                                   open_positions, open_summary, split_exits)
from verify_article2 import six  # noqa: E402
from verify_article4 import article_path, arrow  # noqa: E402

SLUG = "double-bottom-exit-replace"


def claims(df) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。只用可成交 3 格（用戶 2026-11-11 拿掉裸版）。"""
    col = lambda e, c: df[df["變體"] == e][c]
    down = lambda e, c: all(a < b for a, b in six(df, e, c))
    up = lambda e, c: all(a > b for a, b in six(df, e, c))
    n_up = lambda e, c: sum(a > b for a, b in six(df, e, c))
    pf2 = lambda e: f"{arrow(df, e, 3, True, '獲利因子')}；{arrow(df, e, 9, True, '獲利因子')}"
    rng = lambda xs, nd, unit="%": f"{fmt_nd(min(xs), nd)}{unit}～{fmt_nd(max(xs), nd)}{unit}"
    days = lambda e: f"{half_up(col(e, '平均持有天').min(), 0)}～{half_up(col(e, '平均持有天').max(), 0)} 天"
    kept, dropped = split_exits(df)
    summ = open_summary(df, dropped).set_index("變體").loc["target"]
    oldest = open_positions("target").sort_values("held_days", ascending=False).iloc[0]
    buy = pd.Timestamp(oldest["buy_date"])
    base_open_max = col("base", "未平倉率%").max()
    warn = WARN
    dd = {e: drawdown(e) for e in warn}
    dcol = lambda e, c: dd[e][c]
    ddays = lambda e: (f"{half_up(dcol(e, '賺錢曾跌10%持有天中位數').min(), 0)}～"
                       f"{half_up(dcol(e, '賺錢曾跌10%持有天中位數').max(), 0)} 天")
    half_year = lambda e: dcol(e, "賺錢曾跌10%持有天中位數").min() > 365 / 2
    same_n = all(dd[e]["交易次數"].tolist() == col(e, "交易次數").tolist() for e in warn)
    return {
        # 移除組
        "target_open": (summ["帳面虧損%"] > 50,
                        f"{int(summ['未賣出']):,} 筆到 2025 年底還沒賣出，佔進場的 {half_up(summ['未賣出%'], 1)}%，"
                        f"其中 {half_up(summ['帳面虧損%'], 1)}% 帳面是虧損，已持有年數的中位數是 "
                        f"{half_up(summ['持有年中位數'], 1)} 年"),
        "oldest": (oldest["unrealized_pct"] < 0,
                   f"{buy.year} 年 {buy.month} 月買進，到 2025 年底已經持有約 {half_up(oldest['held_days'] / 365, 0)} 年，"
                   f"帳面 {fmt('中位數%', oldest['unrealized_pct'])}%"),
        "be_open": (open_summary(df, ("breakeven",)).iloc[0]["帳面虧損%"] < 50, "沒賣出的部位帳面多數是賺的"),
        "target_win": (min(col("target", "勝率%")) > 99, "勝率會接近百分之百"),
        "base_open": (True, f"基本版在 {rng(col('base', '未平倉率%'), 1)}"),
        # 頸線停利
        "neck_open": (col("neck", "未平倉率%").max() <= OPEN_MAX and col("neck", "未平倉率%").min() > 2 * base_open_max,
                      rng(col("neck", "未平倉率%"), 1)),
        "neck_x2": (col("neck", "未平倉率%").min() > 2 * base_open_max,
                    f"但是基本版（{rng(col('base', '未平倉率%'), 1)}）的兩倍以上"),
        "neck_win": (True, rng(col("neck", "勝率%"), 1)),
        "neck_pf": (min(col("neck", "獲利因子")) > 10, rng(col("neck", "獲利因子"), 3, "")),
        "neck_hold": (up("neck", "平均持有天"), days("neck")),
        # ATR
        "atr_hold": (col("atr", "平均持有天").max() - col("atr", "平均持有天").min() < 2, days("atr")),
        "atr_pf": (down("atr", "獲利因子"), pf2("atr")),
        # 回落 20%
        "trail_up": ([a > b for a, b in six(df, "trail20", "獲利因子")] == [True, True, False],
                     "3%、6% 比基本版高、9% 比基本版低"),
        "trail_pf": (True, pf2("trail20")),
        "trail_loss": (down("trail20", "虧損平均%"),                  # 六格都比基本版賠得深
                       rng(col("trail20", "虧損平均%"), 1)),
        "trail_hold": (up("trail20", "平均持有天"), days("trail20")),
        # MA10
        "ma10_hold": (True, days("ma10")),
        "ma10_pf": (max(col("ma10", "獲利因子")) < 1, rng(col("ma10", "獲利因子"), 3, "")),
        # 高檔 M 頭
        "mtop_pf": (up("mtop", "獲利因子") and up("mtop", "中位數%"), pf2("mtop")),
        "mtop_hold": (up("mtop", "平均持有天"), arrow(df, "mtop", 9, True, "平均持有天", " 天")),
        "mtop_open": (True, rng(col("mtop", "未平倉率%"), 1)),
        # 警告的兩條（逐條判定，不是門檻挑出來的）與回檔表的數字
        "warn": (warn == ("neck", "trail20") and set(warn) <= set(kept) and same_n,
                 "頸線停利和自高點回落 20% 標上 ⚠️"),
        "neck_dd_win": (True, rng(dcol("neck", "賺錢曾跌10%"), 1)),
        "neck_dd_days": (half_year("neck"), ddays("neck")),
        "neck_dd20": (True, rng(dcol("neck", "曾跌20%"), 1)),
        "trail_dd10": (dcol("trail20", "曾跌10%").min() > 50, f"過半（{rng(dcol('trail20', '曾跌10%'), 1)}）"),
        "trail_dd20": (True, rng(dcol("trail20", "曾跌20%"), 1)),
        "trail_dd_win": (True, rng(dcol("trail20", "賺錢曾跌10%"), 1)),
        "trail_dd_days": (half_year("trail20"), ddays("trail20")),
        # 分組本身
        "split": (kept == ("neck", "atr", "trail20", "ma10", "mtop")
                  and dropped == ("target", "target65", "neckfail", "bolow", "breakeven"),
                  "下面這五條在三格裡至少有一格超過 10%"),
    }


def fmt_nd(v, nd: int) -> str:
    return f"{half_up(v, nd)}".replace("-", "−")


def main() -> int:
    path = article_path(SLUG)
    text = io.open(path, encoding="utf-8").read()
    df = stats(ALL_EXITS, src=SRC)
    errs = []
    want = all_tables(df)
    for name, html in want:
        if html not in text:
            errs.append(f"表格與重算不一致：{name}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    cl = claims(df)
    for name, (ok, phrase) in cl.items():
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    if "{{" in text:
        errs.append("文章還有沒填的佔位符")
    print(f"{path}\n表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
