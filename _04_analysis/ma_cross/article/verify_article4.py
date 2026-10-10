# -*- coding: utf-8 -*-
"""
驗〈均線交叉（四）：單股評估〉（ma-cross-single-eval）：蒙地卡羅表每一格＋正文統計句。

  表＝mc_realistic.csv（夾角>20°＋ADX<25＋1000 張，10,000 條路徑）。勝率由逐筆交易重算
  （CSV 只到兩位小數，文章顯示一位）；其餘欄 CSV 位數與文章相同，直接比。
  另驗「代表配置是幾個優化方案裡整體相對穩的一組」：在單股 sweep 的 6 種配置裡，
  夾角＋ADX 的 21 組平均獲利因子與平均期望值都最高。

執行（BLOG_DIR 指向 blog 專案根目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/ma_cross/article/verify_article4.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np  # noqa: E402

from article_common import (MC_VARIANT, PAIRS, TRADING_DAYS, Checker, all_metrics,  # noqa: E402
                            body_rows, fmt, mc, mc_exact_wr, num, post_text, same, section,
                            tables)

SLUG = "ma-cross-single-eval"
# 單股 sweep 的 6 種配置（皆含 1000 張流動性），見 blog/drafts/_handoff_ma_cross_articles.md §十六
SIX = ["liq1000", "strongk_liq1000", "angle20_liq1000", "vol_adx25_liq1000",
       "strongk_adx25_liq1000", "angle20_adx25_liq1000"]


def verify_table(ck: Checker, table: str, t) -> None:
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, "蒙地卡羅表列順序／組數與 21 組不符")
    for row in rows:
        pair, r = row[0][1], t.loc[row[0][1]]
        cells = [c for _, c in row]
        cls = [c for c, _ in row]
        ck.check(same(cells[1], r["交易數"]), f"{pair} 交易數 {cells[1]}")
        ck.check(cells[2] == str(r["抽樣模式"]).split("(")[0], f"{pair} 抽樣模式 {cells[2]}")
        ck.check(same(cells[3], mc_exact_wr(pair)), f"{pair} 勝率 {cells[3]} vs {mc_exact_wr(pair):.4f}")
        for i, col in ((4, "每筆淨期望%"), (5, "報酬%_P5"), (6, "報酬%_中位"), (7, "報酬%_P95")):
            ck.check(same(cells[i], r[col]), f"{pair} {col} {cells[i]} vs {r[col]}")
            ck.check(cls[i] == ("up" if r[col] > 0 else "down"), f"{pair} {col} 底色 {cls[i]}")
        for i, col in ((8, "區間寬度%"), (9, "本金大虧%"), (10, "最大連敗_P95"), (11, "最大回撤%_P95")):
            ck.check(same(cells[i], r[col]), f"{pair} {col} {cells[i]} vs {r[col]}")
            ck.check(cls[i] == "", f"{pair} {col} 不應上色")
        ck.check(abs(r["區間寬度%"] - (r["報酬%_P95"] - r["報酬%_P5"])) < 0.051, f"{pair} 區間寬度≠P95−P5")
        ck.check((r["抽樣模式"] == "抽區間") == (r["交易數"] >= 3 * TRADING_DAYS), f"{pair} 抽樣模式與下限不符")


def verify_claims(ck: Checker, text: str, t) -> None:
    ck.phrase(text, f"2002–2025 共 {fmt(TRADING_DAYS)} 個交易日")
    ck.phrase(text, f"**{fmt(3 * TRADING_DAYS)}～{fmt(4 * TRADING_DAYS)} 筆**")
    ck.phrase(text, f"實際交易數還不到下限 {fmt(3 * TRADING_DAYS)}")
    n = t["交易數"]
    ck.phrase(text, f"120/200 僅 {fmt(n['120/200'])} 筆、50/200 與 60/200 分別 {fmt(n['50/200'])}、{fmt(n['60/200'])} 筆")
    ev = t["每筆淨期望%"]
    ck.check(list(ev[ev < 0].index) == ["5/10"], "每筆期望為負的不只 5/10")
    ck.phrase(text, f"**5/10 的每筆期望是負的（{fmt(ev['5/10'], 2)}%）**，其餘 20 組都為正")
    ck.phrase(text, f"本金幾乎全部大虧（{t.loc['5/10', '本金大虧%']}%）")
    ck.check(130 < t.loc["5/10", "最大回撤%_P95"] < 140, "5/10 回撤 P95 不是「破 130%」")
    ck.check(list(t[t["本金大虧%"] > 0].index) == ["5/10"], "本金大虧不只 5/10")
    s = t["最大連敗_P95"]
    ck.phrase(text, f"最大連敗 P95 約 {s.min()}～{s.max()} 筆")
    dd = t.drop("5/10")["最大回撤%_P95"]
    ck.check(((dd >= 3) & (dd <= 9)).sum() >= 15, "回撤 P95 落在 3%～9% 的不是多數")
    rel = t["區間寬度%"] / t["報酬%_中位"]
    ck.phrase(text, f"如 50/200 中位 {fmt(t.loc['50/200', '報酬%_中位'], 1)}、區間卻跨 {fmt(t.loc['50/200', '區間寬度%'], 1)}")
    longs = ["50/60", "50/120", "50/200", "60/120", "60/200", "120/200"]
    ck.check(all(0.5 <= rel[p] < 0.8 for p in longs), f"長天期區間相對中位不在五到八成：{rel[longs].round(3).to_dict()}")
    three = ["10/50", "10/60", "10/120"]
    ck.check(sorted(rel.drop("5/10").sort_values().index[:3]) == sorted(three),
             f"區間相對中位最窄的三組：{list(rel.drop('5/10').sort_values().index[:3])}")
    ck.phrase(text, f"（{fmt(n['10/50'])}／{fmt(n['10/60'])}／{fmt(n['10/120'])}）")
    ck.phrase(text, f"每筆期望都正（{fmt(ev[three].min(), 1)}%～{fmt(ev[three].max(), 1)}%）")
    ck.check(all(0.3 <= rel[p] < 0.45 for p in three), "三組區間相對中位不是約三到四成")
    ck.check(all(t.loc[p, "最大回撤%_P95"] < 8 for p in three), "三組回撤 P95 不是都在 8% 以內")
    ck.phrase(text, f"每筆期望其實很高（{fmt(ev['50/200'], 2)}%）")
    ck.phrase(text, f"**但交易數只有 {fmt(n['50/200'])} 筆")
    ck.phrase(text, f"信賴區間相對中位很寬（約 {fmt(rel['50/200'] * 100, 0)}%）")
    ck.check(min(n[p] for p in three) / n["50/200"] > 10, "三組交易筆數沒有多出十幾倍以上")
    # 代表配置：6 種配置裡 21 組平均獲利因子、平均期望值最高
    pf = {v: np.mean([m["pf"] for m in all_metrics(v).values()]) for v in SIX}
    evm = {v: np.mean([m["ev"] for m in all_metrics(v).values()]) for v in SIX}
    ck.check(max(pf, key=pf.get) == MC_VARIANT, f"6 配置平均獲利因子最高的是 {max(pf, key=pf.get)}")
    ck.check(max(evm, key=evm.get) == MC_VARIANT, f"6 配置平均期望值最高的是 {max(evm, key=evm.get)}")
    print("  （參考）6 配置 21 組平均獲利因子：" + "、".join(f"{v} {pf[v]:.3f}" for v in SIX))


def main() -> int:
    ck = Checker("均線交叉（四）")
    text = post_text(SLUG)
    t = mc()
    tb = tables(section(text, "## 全 21 組，原樣攤開", "## 這張表在說什麼"))
    if ck.check(len(tb) == 1, f"蒙地卡羅段應有 1 張表，實際 {len(tb)}"):
        verify_table(ck, tb[0], t)
    verify_claims(ck, text, t)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
