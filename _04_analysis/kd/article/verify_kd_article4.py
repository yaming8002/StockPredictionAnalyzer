# -*- coding: utf-8 -*-
"""
驗 KD 交叉（四）kd-cross-exit：四個出場小節表＋小結排行表逐格（含底色），以及正文所有數字與比較句。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article4.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from _04_analysis.kd.article.build_kd_single_tables import BASE, article4, updown  # noqa: E402
from _04_analysis.kd.article.kd_article_common import (Checker, fmt, read_post,  # noqa: E402
                                                       single_metrics)


def g(exit_):
    return single_metrics(f"amt__golden__{exit_}")


def main() -> int:
    ck = Checker("KD 交叉（四）")
    text = read_post("kd-cross-exit")
    ck.tables(text, article4())
    base = single_metrics(BASE)
    hd, lh, k80, cl, td = g("high_death"), g("lower_high"), g("k_down80"), g("climax"), g("top_div")
    combo = g("k_down50-or-k_down80")
    b250 = single_metrics("amt__breakout250__death")

    ck.contains(text, f"它把策略翻到獲利因子 {fmt(hd['獲利因子'], 2)}", "前言高檔死叉")
    ck.contains(text, f"最好也只到 {fmt(b250['獲利因子'], 3)}、沒能翻正", "前言進場天花板")

    ck.contains(text, f"獲利因子 {fmt(lh['獲利因子'], 2)}、每筆期望值 {fmt(lh['期望值/筆'], 0, True)} 元、"
                      f"總損益 {fmt(lh['總獲利(萬)'], 0, True)} 萬", "頂頂低")
    ck.contains(text, f"跟後面的 K 跌破 80（{fmt(k80['獲利因子'], 2)}）伯仲之間，還比不上第二篇的高檔死叉"
                      f"（{fmt(hd['獲利因子'], 2)}）", "頂頂低對照")
    ck.contains(text, f"持有 {fmt(lh['平均持有天'], 0)} 天不算長，虧損平均只有 {fmt(lh['虧損平均%'], 1)}%"
                      f"（比後面 K 跌破 80 的 {fmt(k80['虧損平均%'], 1)}%、頂背離的 {fmt(td['虧損平均%'], 1)}% 小很多）",
                "頂頂低虧損")
    ck.contains(text, f"勝率只有 {fmt(lh['勝率%'], 0)}%、中位數還是負的（{fmt(lh['中位數%'], 1)}%）", "頂頂低勝率")
    ck.check(lh["中位數%"] < 0 and lh["勝率%"] < 50, "頂頂低一半以上交易在小賠")

    ck.contains(text, f"獲利因子 {fmt(k80['獲利因子'], 2)}、每筆期望值 {fmt(k80['期望值/筆'], 0, True)} 元、"
                      f"總損益 **{fmt(k80['總獲利(萬)'], 0, True)} 萬**、勝率過半、中位數轉正", "K 跌破 80")
    ck.check(k80["勝率%"] > 50 and k80["中位數%"] > 0, "K 跌破 80 勝率過半、中位數轉正")
    ck.contains(text, f"持有 {fmt(k80['平均持有天'], 0)} 天，比高檔死叉的 {fmt(hd['平均持有天'], 0)} 天短",
                "K 跌破 80 持有")

    n_up = sum(updown(cl, base, c) == "up" for c in ("勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子"))
    ck.check(n_up == 5, f"過熱 K 下彎「五個指標整片粉紅」：{n_up}/5")
    ck.contains(text, f"獲利因子 {fmt(cl['獲利因子'], 3)}（表上四捨五入為 {fmt(cl['獲利因子'], 2)}）、"
                      f"每筆期望值 {fmt(cl['期望值/筆'], 0)} 元、中位數轉正、勝率過半", "過熱 K 下彎")
    ck.check(cl["獲利因子"] < 1 < k80["獲利因子"], "過熱 K 下彎差在 1.0 下面、K 跌破 80 在上面")

    ck.contains(text, f"**勝率最高（{fmt(td['勝率%'], 1)}%）、中位數也最高（{fmt(td['中位數%'], 1, True)}%）**",
                "頂背離")
    pool = [hd, lh, k80, cl, td, combo]
    ck.check(td["勝率%"] == max(x["勝率%"] for x in pool) and td["中位數%"] == max(x["中位數%"] for x in pool),
             "頂背離勝率、中位數在所有出場中最高")
    ck.contains(text, f"（虧損平均 {fmt(td['虧損平均%'], 1)}%），拖累了整體：獲利因子 {fmt(td['獲利因子'], 2)}、"
                      f"每筆期望值 {fmt(td['期望值/筆'], 0)} 元", "頂背離拖累")

    ck.contains(text, f"這篇的 **頂頂低（{fmt(lh['獲利因子'], 2)}）、K 跌破 80（{fmt(k80['獲利因子'], 2)}）** "
                      f"兩個勉強站上損益兩平，加上第二篇的高檔死叉（{fmt(hd['獲利因子'], 2)}、明顯最強）", "小結")
    ck.check(hd["獲利因子"] == max(x["獲利因子"] for x in pool), "高檔死叉最強")
    above = [x for x in pool if x["獲利因子"] > 1]
    ck.check(len(above) == 3, f"站上損益兩平的出場數（含高檔死叉）：{len(above)}")
    ck.contains(text, f"一般死叉出場只有 {fmt(base['勝率%'], 1)}% 勝率", "基本版勝率")
    hold = [hd, k80, cl, td]
    ck.check(all(50 <= x["勝率%"] <= 61 for x in hold),
             f"「抱到過熱」四個出場勝率在 50~60%：{[round(x['勝率%'], 1) for x in hold]}")
    ck.contains(text, "換成高檔死叉、K 跌破 80、過熱後 K 下彎、頂背離這幾個出場，勝率一口氣升到 50~60%",
                "勝率 50~60% 的範圍")
    ck.contains(text, f"頂背離勝率最高（{fmt(td['勝率%'], 1)}%），獲利因子卻只有 {fmt(td['獲利因子'], 2)}", "勝率≠賺錢")
    ck.check(all(-10.5 <= x["虧損平均%"] <= -9 for x in (k80, cl)), "K 跌破 80、過熱 K 下彎虧損平均在 −10% 上下")
    ck.contains(text, f"把獲利因子從 {fmt(base['獲利因子'], 2)} 推上損益兩平", "結語")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
