# -*- coding: utf-8 -*-
"""
驗 KD 交叉（八）kd-cross-conclusion（2026-11-20 改版，比照 MACD（十三））：
  表一＝（六）（七）全期總獲利前六名、表二＝0050 ＋ 同六組 2015–2025，逐格比對；
  另驗資料面主張（前六名的組成、同期全輸 0050、全期最高倍數「十幾倍」），
  以及正文多股數字句（ranking_text_claims／bench_text_claims：句子由 CSV 算出後逐字比對）。
單股段（（一）（三）的 0.76／0.916）沿用各篇已驗證數字，由 verify_kd_article1／3 負責，這裡不重驗。

資料來源：_04_analysis/kd/kd_multi_conclusion.py 的 kd_conclusion_top.csv／kd_conclusion_vs_0050.csv／
kd_conclusion_all.csv，與多股 driver 的 kd_multi_result.csv。

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
from _04_analysis.kd.kd_multi_conclusion import TOP_N  # noqa: E402

# 節標題（數字句由下面兩個 *_text_claims 從 CSV 算出再比對）
TEXT_CLAIMS = [
    ("全期總獲利前六名", "盤點節標題"),
    ("跟 0050 買進持有比一比", "對 0050 節標題"),
    ("重點整理", "系列結論節"),
]


def read(name):
    """帶精確欄就用精確值（同 builder）。"""
    return with_exact(pd.read_csv(os.path.join(MULTI, name), encoding="utf-8-sig"))


def data_claims(ck: Checker, top: pd.DataFrame, vs: pd.DataFrame, allc: pd.DataFrame):
    ck.check(len(top) == TOP_N and list(top["全期名次"]) == list(range(1, TOP_N + 1)), "表一 6 列、名次 1～6")
    ck.check(list(top["總獲利(萬)"]) == sorted(top["總獲利(萬)"], reverse=True), "表一依全期總獲利高到低")
    full = read("kd_multi_result.csv")
    full = full[full["排序"] != "隨機"]
    ck.check(len(full) == 36 and len(allc) == 36, f"全期 {len(full)} 組、2015–2025 {len(allc)} 組（應各 36）")
    b = vs.iloc[0]
    ck.check(str(b["交易策略"]).startswith("0050"), "表二第一列是 0050")
    picks = vs.iloc[1:]
    ck.check((picks["年化報酬率%"] < b["年化報酬率%"]).all(), "六組年化皆低於 0050")
    ck.check((picks["報酬回撤比"] < b["報酬回撤比"]).all(), "六組報酬回撤比皆低於 0050")
    ck.check((allc["年化報酬率%"] < b["年化報酬率%"]).all(), "36 組年化皆低於 0050")
    ck.check((allc["報酬回撤比"] < b["報酬回撤比"]).all(), "36 組報酬回撤比皆低於 0050")
    best = full.sort_values("資金倍數", ascending=False).iloc[0]
    ck.check(10 <= best["資金倍數"] < 20, f"全期最高倍數「十幾倍」：{best['資金倍數']}")


LONG = {"創250日新高": "創 250 日新高", "創120日新高": "創 120 日新高", "創60日新高": "創 60 日新高",
        "跳空": "跳空", "底背離": "底背離", "低檔＋紅K": "低檔＋紅K"}


def ranked_full() -> pd.DataFrame:
    """全期 36 組（不含隨機）依總獲利精確值排名，附「名次」。"""
    full = read("kd_multi_result.csv")
    full = full[full["排序"] != "隨機"].sort_values("總獲利(萬)", ascending=False, kind="mergesort")
    full = full.reset_index(drop=True)
    full["名次"] = range(1, len(full) + 1)
    return full


def one(df, name, mode, order):
    return df[(df["交易策略"] == name) & (df["投法"] == mode) & (df["排序"] == order)].iloc[0]


def ranking_text_claims(ck: Checker, text: str, rk: pd.DataFrame):
    """盤點節：排序、投法、策略組成與名次句。"""
    top = rk.head(TOP_N)
    ck.check((top["排序"] == "低價").all(), "前六名全是低價優先")
    for order, want in (("成交金額", 9), ("高價", 11)):
        b = rk[rk["排序"] == order].iloc[0]
        ck.check(b["名次"] == want, f"{order}最好排第 {want}（實際 {b['名次']}）")
        ck.contains(text, f"{order}{'優先' if order == '成交金額' else ''}最好的一組排第 {b['名次']}（{LONG[b['交易策略']]} · {b['投法']}，"
                          f"{fmt(b['總獲利(萬)'], 0)} 萬）", f"{order}最好名次")
    n_pct = int((top["投法"] == "比例").sum())
    ck.check(n_pct == 4, f"比例佔 {n_pct} 席")
    ck.contains(text, "比例投入佔四席、定額佔兩席", "投法席次")
    p, f = one(rk, "創120日新高", "比例", "低價"), one(rk, "創120日新高", "定額", "低價")
    ck.contains(text, f"創 120 日新高的比例版 {fmt(p['總獲利(萬)'], 0)} 萬、定額版 {fmt(f['總獲利(萬)'], 0)} 萬",
                "創 120 兩投法總獲利")
    r7 = rk.iloc[TOP_N]
    ck.contains(text, f"第 7 名是{LONG[r7['交易策略']]} · {r7['投法']} · {r7['排序']}（{fmt(r7['總獲利(萬)'], 0)} 萬）",
                "第 7 名")
    new_high = {"創250日新高", "創120日新高", "創60日新高"}
    ck.check(set(top.head(5)["交易策略"]) <= new_high and top.iloc[5]["交易策略"] == "跳空",
             "三種創新高包辦前五名、第六名是跳空")
    for name, phrase in (("底背離", "底背離最好的一組排第 {r}（{m} · {o}，{v} 萬）"),
                         ("低檔＋紅K", "低檔＋紅K 最好的一組排第 {r}（{v} 萬）")):
        b = rk[rk["交易策略"] == name].iloc[0]
        ck.contains(text, phrase.format(r=b["名次"], m=b["投法"], o=b["排序"], v=fmt(b["總獲利(萬)"], 0)),
                    f"{name}最好名次")
    ck.check(top["交易策略"].isin(["底背離", "低檔＋紅K"]).sum() == 0, "低檔＋紅K、底背離都沒進榜")


def bench_text_claims(ck: Checker, text: str, vs: pd.DataFrame, allc: pd.DataFrame):
    """對 0050 節與結語的數字句。"""
    b, picks = vs.iloc[0], vs.iloc[1:].reset_index(drop=True)
    lo, hi = picks["年化報酬率%"].min(), picks["年化報酬率%"].max()
    best_c = picks.sort_values("年化報酬率%", ascending=False).iloc[0]
    ck.contains(text, f"六組的年化落在 {fmt(lo, 2)}%～{fmt(hi, 2)}%，最好的{LONG[best_c['交易策略']]} · "
                      f"{best_c['投法']}（{fmt(hi, 2)}%）也追不上 0050 的 {fmt(b['年化報酬率%'], 2)}%", "年化全輸")
    best_r = picks.sort_values("報酬回撤比", ascending=False).iloc[0]
    ck.contains(text, f"0050 的報酬回撤比約 {fmt(b['報酬回撤比'], 2)}，比六組裡最高的 {fmt(best_r['報酬回撤比'], 2)}"
                      f"（{LONG[best_r['交易策略']]} · {best_r['投法']}）還高", "報酬回撤比")
    first = picks[picks["全期名次"] == 1].iloc[0]
    order_r = picks.sort_values("報酬回撤比", kind="mergesort").reset_index(drop=True)
    ck.check(first["年化報酬率%"] == hi, "全期第 1 名同期年化最高")
    ck.check(order_r.iloc[1]["全期名次"] == 1, "全期第 1 名報酬回撤比倒數第二")
    worst = order_r.iloc[0]
    ck.contains(text, f"最大回撤深到 {fmt(first['最大回撤%'], 1)}%，報酬回撤比只有 {fmt(first['報酬回撤比'], 2)}，"
                      f"是六組裡倒數第二（{LONG[worst['交易策略']]} · {worst['投法']} {fmt(worst['報酬回撤比'], 2)} 最低）",
                "全期第 1 名同期")
    ck.check(best_r["全期名次"] == 2, f"同期最穩是全期第 2 名（實際 {best_r['全期名次']}）")
    ck.contains(text, f"同期最穩的是全期第 2 名的{LONG[best_r['交易策略']]} · {best_r['投法']}", "同期最穩")
    ck.contains(text, f"36 組在 2015–2025 的年化最高 {fmt(allc['年化報酬率%'].max(), 2)}%、"
                      f"報酬回撤比最高 {fmt(allc['報酬回撤比'].max(), 2)}", "36 組上限")
    shallow = allc[allc["最大回撤%"] < b["最大回撤%"]]
    names = {f"{LONG[r['交易策略']]} · {r['投法']} · {r['排序']}" for _, r in shallow.iterrows()}
    ck.check(names == {"創 250 日新高 · 比例 · 低價", "創 120 日新高 · 定額 · 低價"}, f"回撤比 0050 淺：{names}")
    ck.check(set(shallow["最大回撤%"].map(lambda v: fmt(v, 1))) == {"32.5"}, "兩組回撤都是 32.5%")
    ck.contains(text, "回撤比 0050 淺的只有創 250 日新高 · 比例 · 低價與創 120 日新高 · 定額 · 低價兩組（都是 32.5%）",
                "回撤較淺兩組")
    ck.contains(text, f"年化最高的 {fmt(hi, 2)}%、報酬回撤比最高的 {fmt(best_r['報酬回撤比'], 2)}，都低於 0050 的 "
                      f"{fmt(b['年化報酬率%'], 2)}% 與 {fmt(b['報酬回撤比'], 2)}", "結語")
    p, f = one(allc, "創120日新高", "比例", "低價"), one(allc, "創120日新高", "定額", "低價")
    ck.contains(text, f"創 120 日新高在 2015–2025 的比例版最大回撤 {fmt(p['最大回撤%'], 1)}%，"
                      f"定額版是 {fmt(f['最大回撤%'], 1)}%", "創 120 兩投法回撤")


def main() -> int:
    ck = Checker("KD 交叉（八）")
    text = read_post("kd-cross-conclusion")
    ck.tables(text, tables_for(8))
    top, vs, allc = read("kd_conclusion_top.csv"), read("kd_conclusion_vs_0050.csv"), read("kd_conclusion_all.csv")
    data_claims(ck, top, vs, allc)
    rk = ranked_full()
    ranking_text_claims(ck, text, rk)
    bench_text_claims(ck, text, vs, allc)
    ck.contains(text, f"成交金額、高價優先最好的也只排到第 {rk[rk['排序'] == '成交金額'].iloc[0]['名次']}、"
                      f"第 {rk[rk['排序'] == '高價'].iloc[0]['名次']}", "重點整理排序名次")
    for phrase, what in TEXT_CLAIMS:
        ck.contains(text, phrase, what)
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
