# -*- coding: utf-8 -*-
"""
驗 KD 交叉（一）kd-cross-golden-cross：兩張結果表逐格＋正文所有數字句。

表格：與 build_kd_single_tables.article1() 逐格比對（數字、正負號、千分位、紅綠字）。
正文：每句先由逐筆交易重算，再確認文章真的這樣寫；台積電 2015 年 4–9 月那段的
「進出 16 次、14 次小賠、2 次小賺不到 1.5%、毛報酬相加約 −17%」從無門檻版逐筆交易重數。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article1.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.build_kd_single_tables import article1  # noqa: E402
from _04_analysis.kd.article.kd_article_common import (SWEEP, Checker, fmt,  # noqa: E402
                                                       read_post, single_metrics)

RAW, AMT = "raw__golden__death", "amt__golden__death"


def tsmc_window(ck: Checker, text: str):
    """台積電 2015-04～09 無門檻版的實際交易（買、賣都落在區間內）。"""
    t = pd.read_parquet(os.path.join(SWEEP, RAW, "single_kd_trades.parquet"))
    w = t[(t["stock_id"] == "2330.TW") & (t["buy_date"] >= "2015-04-01")
          & (t["sell_date"] <= "2015-09-30")]
    gross = (w["sell_price"] - w["buy_price"]) / w["buy_price"] * 100
    n, lose = len(w), int((gross < 0).sum())
    small_win = gross[gross > 0]
    ck.contains(text, f"進出了 {n} 次，其中 {lose} 次是小賠、只有 {len(small_win)} 次小賺不到 1.5%",
                "台積電交易次數")
    ck.check(bool((small_win < 1.5).all()), f"台積電小賺都 < 1.5%：{small_win.round(2).tolist()}")
    ck.contains(text, f"毛報酬相加大約是 {fmt(gross.sum(), 0)}%", "台積電毛報酬相加")


def main() -> int:
    ck = Checker("KD 交叉（一）")
    text = read_post("kd-cross-golden-cross")
    ck.tables(text, article1())
    r, a = single_metrics(RAW), single_metrics(AMT)

    ck.contains(text, f"獲利因子 {fmt(r['獲利因子'], 2)}（小於 1", "無門檻 PF")
    ck.contains(text, f"每筆平均期望值 {fmt(r['期望值/筆'], 0)} 元、中位數 {fmt(r['中位數%'], 1)}%，"
                      f"總損益 {fmt(r['總獲利(萬)'], 0)} 萬", "無門檻 EV／中位／總損益")
    ck.check(650_000 <= r["交易次數"] < 700_000, f"「近七十萬筆」：實際 {r['交易次數']:,}")
    ck.contains(text, f"近七十萬筆交易、平均每筆只抱 {fmt(r['平均持有天'], 0)} 天", "持有天數")
    tsmc_window(ck, text)

    ck.contains(text, f"（{r['交易次數'] // 10_000} 萬 → {a['交易次數'] // 10_000} 萬筆）", "筆數萬位")
    ck.check(0.4 <= a["交易次數"] / r["交易次數"] <= 0.6,
             f"「少了一半」：比例 {a['交易次數'] / r['交易次數']:.3f}")
    ck.contains(text, f"獲利因子從 {fmt(r['獲利因子'], 2)} 再降到 {fmt(a['獲利因子'], 2)}、"
                      f"每筆期望值從 {fmt(r['期望值/筆'], 0)} 元變成 {fmt(a['期望值/筆'], 0)} 元、"
                      f"中位數從 {fmt(r['中位數%'], 1)}% 變成 {fmt(a['中位數%'], 1)}%", "加門檻前後對照")
    ck.contains(text, f"從 {fmt(r['總獲利(萬)'], 0)} 萬縮到 {fmt(a['總獲利(萬)'], 0)} 萬", "總損益縮小")
    ck.check(a["獲利因子"] < r["獲利因子"] and a["期望值/筆"] < r["期望值/筆"],
             "「結果並沒有變好、每筆平均更差」")
    ck.check(a["獲利因子"] < 1 and r["獲利因子"] < 1, "兩版都是負的（PF < 1）")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
