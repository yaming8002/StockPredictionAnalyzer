# -*- coding: utf-8 -*-
"""
驗 KD 交叉（八）kd-cross-conclusion：六種策略表、對 0050 表逐格，以及正文的挑選結果、分段範圍、
對 0050 的比較句、全期倍數句。

挑選依據＝kd_conclusion_all.csv 的「選」欄（每策略取報酬回撤比較高的投法，_04_analysis/kd/kd_multi_conclusion.py）；
全期（2002–2025）倍數取多股回測 kd_multi_result.csv 的最高資金倍數，並確認它的全期回撤比同策略同投法
2015–2025 那次更深（文中「過程的回撤也更深」）。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article8.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.build_kd_multi_tables import tables_for  # noqa: E402
from _04_analysis.kd.article.kd_article_common import (MULTI, Checker, fmt, read_post,  # noqa: E402
                                                       with_exact)

NEW_HIGH = ["創250日新高", "創120日新高", "創60日新高"]
OTHERS = ["跳空", "低檔＋紅K", "底背離"]
PCT_HIGHER = NEW_HIGH + ["底背離"]      # 比例版資金倍數較高（底背離 1.00 vs 0.96）
PCT_LOWER = ["跳空", "低檔＋紅K"]       # 比例版資金倍數略低


def read(name):
    """帶精確欄就用精確值（同 builder），正文範圍句才會和表格同一次進位。"""
    return with_exact(pd.read_csv(os.path.join(MULTI, name), encoding="utf-8-sig"))


def pick(df, name, mode):
    return df[(df["交易策略"] == name) & (df["投法"] == mode)].iloc[0]


def mode_claims(ck: Checker, text: str, allc: pd.DataFrame):
    for n in PCT_HIGHER:
        ck.check(pick(allc, n, "比例")["資金倍數"] > pick(allc, n, "定額")["資金倍數"], f"{n} 比例倍數較高")
    for n in PCT_LOWER:
        ck.check(pick(allc, n, "比例")["資金倍數"] < pick(allc, n, "定額")["資金倍數"], f"{n} 比例倍數略低")
    p, f = pick(allc, "創120日新高", "比例"), pick(allc, "創120日新高", "定額")
    ck.contains(text, f"創 120 的比例版深到 {fmt(p['最大回撤%'], 1)}%（定額 {fmt(f['最大回撤%'], 1)}%）", "創 120 回撤")
    diff = max(abs(pick(allc, n, "比例")["最大回撤%"] - pick(allc, n, "定額")["最大回撤%"])
               for n in NEW_HIGH + OTHERS if n != "創120日新高")
    ck.contains(text, f"其餘五種兩投法相差在 {fmt(diff, 1)} 個百分點以內", "其餘回撤差")
    sel = allc[allc["選"]]
    n_fix = int((sel["投法"] == "定額").sum())
    pct = sel[sel["投法"] == "比例"]["交易策略"].tolist()
    ck.check(n_fix == 3 and set(pct) == {"創250日新高", "創60日新高", "底背離"}, f"選出 3 定額＋比例 {pct}")
    ck.contains(text, "六個裡三個選了定額，創 250、創 60、底背離的比例版", "挑選結果")


def tier_claims(ck: Checker, text: str, conc: pd.DataFrame):
    nh = conc[conc["交易策略"].isin(NEW_HIGH)]
    lo, hi = int(nh["年化報酬率%"].min()), int(nh["年化報酬率%"].max())
    ck.contains(text, f"年化 {lo}～{hi}%、回撤 {fmt(nh['最大回撤%'].min(), 0)}～{fmt(nh['最大回撤%'].max(), 0)}%",
                "創新高段")
    gap = conc[conc["交易策略"] == "跳空"].iloc[0]
    ck.contains(text, f"跳空·定額年化約 {fmt(gap['年化報酬率%'], 0)}%", "跳空段")
    weak = conc[conc["交易策略"].isin(["低檔＋紅K", "底背離"])]
    ck.check((weak["年化報酬率%"].abs() < 1).all(), "低檔＋紅K、底背離年化在 0 上下")
    ck.contains(text, f"回撤卻 {fmt(weak['最大回撤%'].min(), 0)}～{fmt(weak['最大回撤%'].max(), 0)}%", "弱端回撤")


def bench_claims(ck: Checker, text: str, conc: pd.DataFrame, vs: pd.DataFrame):
    b = vs.iloc[0]
    ck.check(b["做法"].startswith("0050"), "對照表第一列是 0050")
    top2 = conc.sort_values("年化報酬率%", ascending=False).head(2)
    lbl = {"創250日新高": "創 250", "創120日新高": "創 120", "創60日新高": "創 60"}
    parts = [f"{lbl.get(r['交易策略'], r['交易策略'])}·{r['投入']} {fmt(r['年化報酬率%'], 2)}%"
             for _, r in top2.iterrows()]
    ck.contains(text, f"最好的（{'、'.join(parts)}）都追不上 0050 的 {fmt(b['年化報酬率%'], 2)}%", "年化全輸")
    ck.check((conc["年化報酬率%"] < b["年化報酬率%"]).all(), "六列年化皆低於 0050")
    ck.check((conc["報酬回撤比"] < b["報酬回撤比"]).all(), "六列報酬回撤比皆低於 0050")
    best = conc.sort_values("報酬回撤比", ascending=False).iloc[0]
    ck.contains(text, f"0050 的報酬回撤比約 {fmt(b['報酬回撤比'], 2)}（{fmt(b['年化報酬率%'], 2)} ÷ "
                      f"{fmt(b['最大回撤%'], 1)}），比六列裡最高的 {fmt(best['報酬回撤比'], 2)}"
                      f"（{lbl[best['交易策略']]}·{best['投入']}）還高", "報酬回撤比")
    ck.contains(text, f"最好的{lbl[best['交易策略']]}·{best['投入']}年化 {fmt(best['年化報酬率%'], 2)}%、"
                      f"報酬回撤比 {fmt(best['報酬回撤比'], 2)}，都低於 0050 的 {fmt(b['年化報酬率%'], 2)}% 與 "
                      f"{fmt(b['報酬回撤比'], 2)}", "結語")
    ck.check(best["年化報酬率%"] == conc["年化報酬率%"].max(), "報酬回撤比最高者同時年化最高")


def full_period_claims(ck: Checker, text: str, allc: pd.DataFrame):
    m = read("kd_multi_result.csv")
    top = m.sort_values("資金倍數", ascending=False).iloc[0]
    ck.check(10 <= top["資金倍數"] < 20, f"全期最高倍數「十幾倍」：{top['資金倍數']}（{top['交易策略']}·{top['投法']}）")
    win = pick(allc, top["交易策略"], top["投法"])
    ck.check(top["最大回撤%"] > win["最大回撤%"],
             f"全期回撤 {top['最大回撤%']}% 深於 2015–2025 的 {win['最大回撤%']}%")
    ck.contains(text, "拉回更長的 **2002–2025 全期**跑", "全期區間")


def main() -> int:
    ck = Checker("KD 交叉（八）")
    text = read_post("kd-cross-conclusion")
    ck.tables(text, tables_for(8))
    allc, conc, vs = read("kd_conclusion_all.csv"), read("kd_conclusion.csv"), read("kd_conclusion_vs_0050.csv")
    ck.check(list(conc["報酬回撤比"]) == sorted(conc["報酬回撤比"], reverse=True), "第一表依報酬回撤比高到低")
    mode_claims(ck, text, allc)
    tier_claims(ck, text, conc)
    bench_claims(ck, text, conc, vs)
    full_period_claims(ck, text, allc)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
