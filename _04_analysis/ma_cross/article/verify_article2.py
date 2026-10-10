# -*- coding: utf-8 -*-
"""
驗〈均線交叉（二）：調整測試（上）〉（ma-cross-adjustment-test-1）：三張「基準→調整」表＋正文統計句。

  基準＝黃金交叉＋5 日均量>1,000 張（result/ma_cross/liq1000）
  強 K＝strongk_liq1000、雙重確認＝confirm_liq1000、盤整 ADX<25＝vol_adx25_liq1000
  表格每格（箭頭兩側）由逐筆交易重算；方向五欄（勝率、獲利平均、中位數、期望值、獲利因子）
  依未捨入真值比較上色。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article2.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from article_common import (PAIRS, Checker, all_metrics, fmt, post_text, section,  # noqa: E402
                            tables, verify_arrow_table)

SLUG = "ma-cross-adjustment-test-1"
BASE = "liq1000"
SECTIONS = [("## 強 K：只在「氣勢夠」的交叉才進", "strongk_liq1000"),
            ("## 雙重確認：避開均線糾纏情況", "confirm_liq1000"),
            ("## 盤整過濾：盤整後是否會有好的趨勢?", "vol_adx25_liq1000")]
END = "## 重點整理"


def verify_claims(ck: Checker, text: str) -> None:
    b = all_metrics(BASE)
    sk, cf, adx = (all_metrics(v) for _, v in SECTIONS)
    b6, s6, c6, a6 = b["60/200"], sk["60/200"], cf["60/200"], adx["60/200"]

    # 強 K
    ck.phrase(text, f"**中位數反而變差了**（{fmt(b6['med'], 1)} → {fmt(s6['med'], 1)}")
    ck.check(s6["med"] < b6["med"], "強 K 60/200 中位數沒有變差")
    ck.phrase(text, f"獲利平均 {fmt(b6['aw'], 1)}%→{fmt(s6['aw'], 1)}%、EV {fmt(b6['ev'])}→{fmt(s6['ev'])} 元、"
                    f"PF {fmt(b6['pf'], 2)}→{fmt(s6['pf'], 2)}")
    ratio = sum(sk[p]["n"] for p in PAIRS) / sum(b[p]["n"] for p in PAIRS)
    ck.check(0.45 <= ratio <= 0.65, f"強 K 總交易數為基準的 {ratio:.3f}（文：約一半）")
    up_aw = sum(sk[p]["aw"] > b[p]["aw"] for p in PAIRS)
    ck.check(up_aw >= 19, f"強 K 獲利平均% 上升 {up_aw}/21（文：幾乎每組都上升）")
    ck.check(all(sk[p]["al"] < b[p]["al"] for p in PAIRS), "強 K 虧損平均% 不是每組都變大")
    ck.phrase(text, f"獲利平均 {fmt(b6['aw'], 1)}→{fmt(s6['aw'], 1)}、虧損平均 {fmt(b6['al'], 1)}→{fmt(s6['al'], 1)}")
    ck.check(all(sk[p]["tot"] < b[p]["tot"] for p in PAIRS if b[p]["tot"] > 0),
             "強 K 總獲利（基準為正的組）不是全部下降")
    # 雙重確認
    ck.phrase(text, f"結果（60/200）：中位數 {fmt(b6['med'], 1)} → {fmt(c6['med'], 1)}、EV {fmt(b6['ev'])} → "
                    f"{fmt(c6['ev'])}、PF {fmt(b6['pf'], 2)} → {fmt(c6['pf'], 2)}、交易數 {fmt(b6['n'])} → "
                    f"{fmt(c6['n'])}")
    # 盤整 ADX
    ck.phrase(text, f"60/200 的**獲利因子從 {fmt(b6['pf'], 2)} 升到 {fmt(a6['pf'], 2)}、"
                    f"中位數從 {fmt(b6['med'], 1)} 變 {fmt(a6['med'], 1)}**")
    ck.phrase(text, f"**EV 只小幅上升**（{fmt(b6['ev'])} → {fmt(a6['ev'])}）")
    ck.phrase(text, f"虧損平均從 {fmt(b6['al'], 1)}% 縮到 {fmt(a6['al'], 1)}%")
    ck.phrase(text, f"**總獲利更是從 {fmt(b6['tot'])} 萬掉到 {fmt(a6['tot'])} 萬**")
    ck.phrase(text, f"交易數砍到剩不到 1/3（{fmt(b6['n'])} → {fmt(a6['n'])}）")
    ck.check(a6["n"] / b6["n"] < 1 / 3, "ADX 60/200 交易數沒有砍到 1/3 以下")
    b12, a12 = b["120/200"], adx["120/200"]
    ck.phrase(text, f"120/200 的 EV 反而從 {fmt(b12['ev'])} 掉到 {fmt(a12['ev'])}、"
                    f"總獲利從 {fmt(b12['tot'])} 萬掉到 {fmt(a12['tot'])} 萬")
    ratio_adx = sum(adx[p]["n"] for p in PAIRS) / sum(b[p]["n"] for p in PAIRS)
    ck.check(a6["n"] / b6["n"] < 0.4, f"ADX 60/200 剩 {a6['n'] / b6['n']:.3f}（摘要表：交易砍 2/3）")
    print(f"  （參考）ADX 21 組合計交易數為基準的 {ratio_adx:.3f}")


def main() -> int:
    ck = Checker("均線交叉（二）")
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
