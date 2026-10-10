# -*- coding: utf-8 -*-
"""
驗 KD 交叉（五）kd-cross-combo：6×6 矩陣、高檔死叉完整表、蒙地卡羅表、附錄 36 組逐格，
以及正文的範圍句（「1.42～1.72」「0.92 以下」「回撤 P95 4～7%」…）與排名句。

蒙地卡羅表的報酬欄存檔只有一位小數，顯示取整數時若剛好落在 .5（例如 626.5），
build_kd_single_tables.mc_exact 會用同一 seed 重算該組未取整的分位數再進位。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article5.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.build_kd_single_tables import ENT5, EXIT5, article5, v5  # noqa: E402
from _04_analysis.kd.article.kd_article_common import (MC_DIR, Checker, fmt,  # noqa: E402
                                                       read_post, single_metrics)


def pf(e, x):
    return single_metrics(v5(e, x))["獲利因子"]


def main() -> int:
    ck = Checker("KD 交叉（五）")
    text = read_post("kd-cross-combo")
    ck.tables(text, article5())
    codes = [c for _, _, c in ENT5]
    hd = {c: single_metrics(v5(c, "high_death")) for c in codes}
    hd_pf = [hd[c]["獲利因子"] for c in codes]
    death_pf = [pf(c, "death") for c in codes]

    rng = f"{fmt(min(hd_pf), 2)} 到 {fmt(max(hd_pf), 2)}"
    ck.contains(text, f"「高檔死叉」整欄從 {rng}", "高檔死叉整欄範圍")
    ck.contains(text, f"六個進場從 {rng}", "高檔死叉六進場範圍")
    ck.contains(text, f"都在 {fmt(min(hd_pf), 2)}～{fmt(max(hd_pf), 2)}", "重點整理範圍")
    others = [pf(c, x) for c in codes for _, x in EXIT5 if x != "high_death"]
    ck.check(min(hd_pf) > max(others), "高檔死叉整欄壓過其他所有出場")
    ck.check(max(death_pf) <= 0.92, f"死叉整欄 0.92 以下：{max(death_pf):.4f}")
    ck.check(0.8 <= min(death_pf) and max(death_pf) < 1.0 and min(hd_pf) >= 1.4, "0.8～0.9 → 1.4 以上")
    lh = {c: pf(c, "lower_high") for c in ("breakout250", "breakout120", "breakout60", "gap")}
    ck.check(all(1.115 <= v < 1.275 for v in lh.values()), f"頂頂低配新高／跳空在 1.12～1.27：{lh}")
    ck.check(all(lh[c] > pf(c, "death") and lh[c] < hd[c]["獲利因子"] for c in lh), "頂頂低比死叉好、追不上高檔死叉")

    best_pf = max(codes, key=lambda c: hd[c]["獲利因子"])
    best_tot = max(codes, key=lambda c: hd[c]["總獲利(萬)"])
    ck.check(best_pf == "breakout120" and best_tot == "gap", f"PF 冠軍 {best_pf}、總獲利冠軍 {best_tot}")
    ck.contains(text, f"**獲利因子最高的（創 120 日新高，{fmt(hd['breakout120']['獲利因子'], 2)}）跟總損益最高的"
                      f"（跳空，{fmt(hd['gap']['總獲利(萬)'], 0, True)} 萬）不是同一個。**", "取捨句")
    ck.contains(text, f"創 120 只有 {fmt(hd['breakout120']['交易次數'], 0)} 筆、創 250 更只剩 "
                      f"{fmt(hd['breakout250']['交易次數'], 0)} 筆", "筆數")
    g = hd["gap"]
    ck.contains(text, f"留下的交易多（{fmt(g['交易次數'], 0)} 筆），單筆品質（PF {fmt(g['獲利因子'], 2)}）", "跳空")
    ck.contains(text, f"（{fmt(g['獲利平均%'], 1, True)}% / {fmt(g['虧損平均%'], 1)}%）", "跳空賺賠")
    news = [hd[c] for c in ("breakout250", "breakout120", "breakout60")]
    ck.check(all(g["獲利平均%"] > n["獲利平均%"] and g["虧損平均%"] < n["虧損平均%"] for n in news),
             "跳空平均獲利、平均虧損都比創新高類大")
    ck.contains(text, f"獲利因子最高的是創 120 日新高（{fmt(hd['breakout120']['獲利因子'], 2)}），總獲利最高的是"
                      f"跳空（{fmt(g['總獲利(萬)'], 0, True)} 萬）", "重點整理冠軍")

    mc = pd.read_csv(os.path.join(MC_DIR, "kd_montecarlo.csv"), encoding="utf-8-sig")
    mc = mc[mc["分組"] == "文章"]
    ck.check((mc["本金大虧%"] == 0).all(), "破產率全部 0")
    ck.contains(text, f"最大回撤 P95 只有 {fmt(mc['回撤% P95'].min(), 0)}～{fmt(mc['回撤% P95'].max(), 0)}%、"
                      f"最大連敗 P95 約 {mc['連敗 P95'].min()}～{mc['連敗 P95'].max()} 筆", "回撤／連敗範圍")
    ck.contains(text, f"回撤 P95 只有 {fmt(mc['回撤% P95'].min(), 0)}～{fmt(mc['回撤% P95'].max(), 0)}%，而且六組",
                "重點整理回撤")
    ck.check(mc["報酬% P5"].min() >= 120, f"P5 都在 +120% 以上：{mc['報酬% P5'].min()}")
    hi = mc[mc["進場"].isin(["breakout250", "breakout120", "breakout60", "gap"])]
    ck.check(hi["報酬% P5"].min() >= 600, f"新高、跳空類 P5 在 +600% 以上：{hi['報酬% P5'].min()}")
    ck.contains(text, f"每筆淨期望（{fmt(mc['每筆期望%'].min(), 1)}～{fmt(mc['每筆期望%'].max(), 1)}%）", "每筆期望範圍")

    samp = mc[mc["抽樣模式"] == "抽區間"].sort_values("交易數", ascending=False)
    full = mc[mc["抽樣模式"] == "全納入"]
    names = {c: s for _, s, c in ENT5}
    ck.check(samp["區間寬度"].min() > full["區間寬度"].max(), "抽區間組的區間都比全納入組寬")
    ck.contains(text, f"走「抽區間」的三組（{'、'.join(names[c] for c in samp['進場'])}）報酬區間往上抬也往外拉寬"
                      f"（區間寬度 {fmt(samp['區間寬度'].min(), 0)}～{fmt(samp['區間寬度'].max(), 0)}）", "抽區間三組")
    order_full = [c for c in ("breakout250", "low_redk", "divergence") if c in set(full["進場"])]
    ck.check(len(order_full) == len(full), "全納入組成員")
    ck.contains(text, f"走「全納入」的三組（{'、'.join(names[c] for c in order_full)}）報酬區間反而最窄"
                      f"（{fmt(full['區間寬度'].min(), 0)}～{fmt(full['區間寬度'].max(), 0)}）", "全納入三組")
    ck.contains(text, "K、D < 20 且收 ≥ 開", "紅K定義（程式＝close >= open）")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
