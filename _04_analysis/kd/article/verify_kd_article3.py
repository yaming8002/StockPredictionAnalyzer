# -*- coding: utf-8 -*-
"""
驗 KD 交叉（三）kd-cross-entry-filters：五張結果表逐格（含底色）＋正文所有數字與排名句。

排行表的列序由獲利因子重排（build_kd_single_tables.rank3_order），文章順序不對就會逐格報錯；
正文的「單調趨勢」「排第四」「濾掉約七成」「站上單一均線 0.76~0.78」等句子都由逐筆交易重算。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article3.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _04_analysis.kd.article.build_kd_single_tables import (BASE, article3, rank3_order,  # noqa: E402
                                                            updown, var3)
from _04_analysis.kd.article.kd_article_common import (Checker, fmt, read_post,  # noqa: E402
                                                       single_metrics)


def m(code):
    return single_metrics(var3(code))


def main() -> int:
    ck = Checker("KD 交叉（三）")
    text = read_post("kd-cross-entry-filters")
    ck.tables(text, article3())
    base = single_metrics(BASE)

    # 突破創新高：回看越長越好（獲利因子、期望值都單調）
    brk = [m(f"breakout{n}") for n in (20, 60, 120, 250)]
    ck.check(all(a["獲利因子"] < b["獲利因子"] for a, b in zip(brk, brk[1:])), "創新高 PF 單調上升")
    ck.check(all(a["期望值/筆"] < b["期望值/筆"] for a, b in zip(brk, brk[1:])), "創新高期望值單調上升")
    ck.contains(text, f"獲利因子從創 20 日的 {fmt(brk[0]['獲利因子'], 3)} 一路升到創 250 日（約一年新高）的 "
                      f"**{fmt(brk[3]['獲利因子'], 3)}**、期望值 {fmt(brk[0]['期望值/筆'], 0)} 一路縮到 "
                      f"{fmt(brk[3]['期望值/筆'], 0)}", "創新高梯度")
    ranked = rank3_order()
    best = [c for _, c in ranked if c is not None][0]
    ck.check(best == "breakout250", f"所有進場條件裡最好＝創 250 日新高（實際 {best}）")

    gap = m("gap")
    ck.contains(text, f"獲利因子 {fmt(gap['獲利因子'], 3)}、勝率上到 {fmt(gap['勝率%'], 1)}%，多數指標粉紅", "跳空")
    n_up = sum(updown(gap, base, c) == "up" for c in ("勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子"))
    ck.check(n_up >= 3, f"跳空「多數指標粉紅」：{n_up}/5")
    ck.check(gap["獲利因子"] < 1, "跳空仍在 1.0 以下")

    bull = m("bull_align_5_20_60")
    ck.contains(text, f"獲利因子 {fmt(bull['獲利因子'], 3)}、期望值從 {fmt(base['期望值/筆'], 0)} 縮到 "
                      f"{fmt(bull['期望值/筆'], 0)}", "均線多頭")
    ck.contains(text, f"**勝率反而略降**（{fmt(base['勝率%'], 1)}→{fmt(bull['勝率%'], 1)}）", "均線多頭勝率")
    ck.check(bull["勝率%"] < base["勝率%"], "均線多頭勝率確實下降")
    cut = 1 - bull["交易次數"] / base["交易次數"]
    ck.check(round(cut * 10) == 7, f"「濾掉約七成」：實際濾掉 {cut:.1%}")
    ck.contains(text, f"（{base['交易次數'] // 10_000} 萬→{fmt(bull['交易次數'] / 10_000, 1)} 萬筆）", "均線多頭筆數")

    div = m("divergence")
    ck.contains(text, f"只剩 {div['交易次數'] // 1000} 千筆", "底背離筆數")
    n_up = sum(updown(div, base, c) == "up" for c in ("勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子"))
    ck.check(n_up == 5, f"底背離「五個指標整片粉紅」：{n_up}/5")
    ck.contains(text, f"（獲利因子 {fmt(div['獲利因子'], 3)}）", "底背離 PF")

    # 排行表註：交叉強度勝率排第四、獲利因子低於基本版
    sp = m("kd_spread5")
    wr_rank = 1 + sum(m(c)["勝率%"] > sp["勝率%"] for _, c in ranked if c not in (None, "kd_spread5"))
    ck.contains(text, f"勝率 {fmt(sp['勝率%'], 1)}% 在全表排第{'一二三四五六'[wr_rank - 1]}、"
                      f"獲利因子卻只有 {fmt(sp['獲利因子'], 3)}、低於基本版", "交叉強度排名")
    ck.check(sp["獲利因子"] < base["獲利因子"], "交叉強度 PF 低於基本版")
    # 攤開來看三件事
    for code in ("vol_x2", "vol_x1_5", "vol_above_ma5", "cmf_pos"):
        ck.check(m(code)["獲利因子"] < base["獲利因子"], f"量能／CMF 類比不加還差：{code}")
    ma = [m(f"above_ma{n}")["獲利因子"] for n in (20, 60, 120, 200)]
    ck.check(0.755 <= min(ma) and max(ma) < 0.785, f"站上單一均線在 0.76~0.78：{[round(x, 3) for x in ma]}")
    ck.check(all(x < m("bull_align_5_20_60")["獲利因子"] for x in ma), "多頭排列略好於站上單一均線")
    pfs = [m(c)["獲利因子"] for _, c in ranked if c is not None]
    ck.check(max(pfs) < 1, "沒有一個進場條件把策略翻正")
    ck.contains(text, f"進場端最好的條件（創年度新高）也只到 **{fmt(brk[3]['獲利因子'], 3)}**", "天花板")
    ck.contains(text, f"把獲利因子推到 {fmt(brk[3]['獲利因子'], 3)}，卻仍停在 1.0 以下", "結語")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
