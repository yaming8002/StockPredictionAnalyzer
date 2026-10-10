# -*- coding: utf-8 -*-
"""
驗〈均線交叉（三）：調整測試（下）〉（ma-cross-adjustment-test-2）：三張「基準→調整」表＋正文統計句。

  基準＝liq1000；多頭排列＝align_liq1000、夾角>20 度＝angle20_liq1000、
  跌破短均線就賣＝exit_below_short_liq1000。驗法同（二）。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article3.py
"""
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import (PAIRS, Checker, all_metrics, fmt, post_text, section,  # noqa: E402
                            tables, verify_arrow_table)

SLUG = "ma-cross-adjustment-test-2"
BASE = "liq1000"
SECTIONS = [("## 多頭排列：只在「長線還在多頭」時才做", "align_liq1000"),
            ("## 夾角>20度：用角度衡量氣勢", "angle20_liq1000"),
            ("## 跌破短均線就賣", "exit_below_short_liq1000")]
END = "## 重點整理"


def verify_claims(ck: Checker, text: str) -> None:
    b = all_metrics(BASE)
    al, an, ex = (all_metrics(v) for _, v in SECTIONS)
    b6 = b["60/200"]
    ck.phrase(text, f"交易數 {fmt(b6['n'])}、勝率 {fmt(b6['wr'], 1)}%、獲利平均 {fmt(b6['aw'], 1)}%、"
                    f"中位數 {fmt(b6['med'], 1)}%、EV {fmt(b6['ev'])} 元、PF {fmt(b6['pf'], 2)}、"
                    f"總獲利 {fmt(b6['tot'])} 萬、持有 {fmt(b6['hold'])} 天")
    # 多頭排列
    a6 = al["60/200"]
    ck.phrase(text, f"**EV 從 {fmt(b6['ev'])} 衝到 {fmt(a6['ev'])}、PF 從 {fmt(b6['pf'], 2)} 升到 "
                    f"{fmt(a6['pf'], 2)}、獲利平均從 {fmt(b6['aw'], 1)}% 升到 {fmt(a6['aw'], 1)}%**")
    ck.phrase(text, f"60/200 從 {fmt(b6['n'])} 砍到只剩 **{fmt(a6['n'])}**（約 1/6）")
    ck.check(round(b6["n"] / a6["n"]) == 6, f"多頭排列 60/200 剩 1/{b6['n'] / a6['n']:.2f}（文：約 1/6）")
    ck.phrase(text, f"**總獲利也從 {fmt(b6['tot'])} 萬掉到 {fmt(a6['tot'])} 萬**")
    b12, a12 = b["120/200"], al["120/200"]
    ck.phrase(text, f"120/200 還留著 {fmt(a12['n'])} 筆（總獲利幾乎沒掉，{fmt(a12['tot'])} 萬 vs "
                    f"{fmt(b12['tot'])} 萬），60/200 卻只剩 {fmt(a6['n'])}")
    # 夾角：tan(20°) 換算每天多爬的幅度
    ck.phrase(text, f"短均每天比長均多爬約 {fmt(math.tan(math.radians(20)), 2)}% 以上")
    g6b, g6 = b["60/120"], an["60/120"]
    ck.phrase(text, f"EV 從 {fmt(g6b['ev'])} 升到 {fmt(g6['ev'])}、PF 從 {fmt(g6b['pf'], 2)} 升到 {fmt(g6['pf'], 2)}")
    n6 = an["60/200"]
    ck.phrase(text, f"EV 雖然從 {fmt(b6['ev'])} 升到 {fmt(n6['ev'])}，PF 卻只小幅上升"
                    f"（{fmt(b6['pf'], 2)}→{fmt(n6['pf'], 2)}）")
    ck.phrase(text, f"**中位數從 {fmt(b6['med'], 1)} 降到 {fmt(n6['med'], 1)}**")
    n12 = an["120/200"]
    ck.phrase(text, f"EV {fmt(b12['ev'])}→{fmt(n12['ev'])}、PF {fmt(n12['pf'], 2)}")
    ck.phrase(text, f"**交易數只剩 {fmt(n12['n'])} 筆**")
    # 跌破短均出場
    up_med = sum(ex[p]["med"] > b[p]["med"] for p in PAIRS)
    ck.phrase(text, f"21 組有 {up_med} 組")
    e6, e5 = ex["60/200"], ex["50/200"]
    ck.phrase(text, f"60/200 從 {fmt(b6['med'], 1)} 升到 {fmt(e6['med'], 1)}、50/200 從 "
                    f"{fmt(b['50/200']['med'], 1)} 升到 {fmt(e5['med'], 1)}")
    ck.check(fmt(-b["50/200"]["med"], 0) == "6" and fmt(-e5["med"], 0) == "3",
             "50/200 中位數不是「從賠 6% 變成只賠 3%」")
    ck.phrase(text, f"**獲利平均從 {fmt(b6['aw'], 1)}% 降到 {fmt(e6['aw'], 1)}%**")
    ck.phrase(text, f"**平均持有天數從 {fmt(b6['hold'])} 天大幅縮到 {fmt(e6['hold'])} 天**")
    ck.phrase(text, f"**EV 從 {fmt(b6['ev'])} 降到 {fmt(e6['ev'])}、PF 從 {fmt(b6['pf'], 2)} 掉到 "
                    f"{fmt(e6['pf'], 2)}、總獲利從 {fmt(b6['tot'])} 萬掉到 {fmt(e6['tot'])} 萬**")
    for key, name in (("aw", "獲利平均%"), ("ev", "EV"), ("pf", "獲利因子")):
        ck.check(all(ex[p][key] < b[p][key] for p in PAIRS), f"跌破短均：{name} 不是 21 組全降（文：整片淺綠）")
    ck.check(e6["aw"] / b6["aw"] < 0.55, "跌破短均 60/200 獲利平均不是「大約只剩一半」")


def main() -> int:
    ck = Checker("均線交叉（三）")
    text = post_text(SLUG)
    for i, (head, variant) in enumerate(SECTIONS):
        nxt = SECTIONS[i + 1][0] if i + 1 < len(SECTIONS) else END
        tb = tables(section(text, head, nxt))
        if ck.check(len(tb) == 1, f"{head} 段應有 1 張表，實際 {len(tb)}"):
            verify_arrow_table(ck, tb[0], BASE, variant, head.lstrip("# "))
    verify_claims(ck, text)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
