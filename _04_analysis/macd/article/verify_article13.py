# -*- coding: utf-8 -*-
"""
驗 MACD（十三）macd-conclusion：兩張表逐格，以及正文裡盤點、0050 對照與多股觀察的每一條數字主張。

資料來源：
  result/macd_multi/macd_multi_result.csv         （十一）（十二）全期 30 組（盤點）
  result/macd_multi/macd_multi_conclusion.csv     前 TOP_N 名 2015–2025 ＋ 0050
  result/macd_multi/macd_multi_conclusion_all.csv 30 組 2015–2025（「放大到 30 組」那幾句）

表格：拿 build_article13_tables 產生的每一列，逐列比對文章裡同一列（整列字串一致才算過）。
正文：每條主張先由資料重算並用 assert 確認方向／排名成立，再組出「應出現的字串」檢查文章裡有沒有。
「重點整理」的單股段落引用（一）～（十）的數字，由各篇自己的驗證程式負責
（verify_article1_8／9／10），這裡不重驗。

執行（需設 BLOG_DIR 指向 blog 目錄）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/macd/article/verify_article13.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import pandas as pd  # noqa: E402

from build_article11_tables import SPA, half_up, with_exact  # noqa: E402
from build_article13_tables import BENCH, TOP_N, conclusion, ranked, table1, table2  # noqa: E402

_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402

SLUG = "macd-conclusion"
ALL_2015 = os.path.join(SPA, "macd_multi_conclusion_all.csv")


def read_post() -> str:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{SLUG}.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


def wan(v) -> str:
    return f"{half_up(v, 0):,}"


def inventory_claims(rk: pd.DataFrame) -> list:
    """第一節：盤點前六名。"""
    top = rk.head(TOP_N)
    c = []
    assert (top["投法"] == "比例").sum() == TOP_N - 1
    last_fix = top[top["投法"] == "定額"].iloc[0]
    assert last_fix["全期名次"] == TOP_N and last_fix["排序"] == "低價" and "ADX" in last_fix["交易策略"]
    nxt = rk.iloc[TOP_N]
    assert nxt["投法"] == "定額"
    c.append("**比例投入佔了六席裡的五席**")
    c.append(f"定額只有第 6 名（交叉 × ADX>25 · 低價，{wan(last_fix['總獲利(萬)'])} 萬）")
    c.append(f"第 7 名的定額組合只少它 {wan(last_fix['總獲利(萬)'] - nxt['總獲利(萬)'])} 萬")

    assert top["交易策略"].str.startswith("交叉").all()
    c.append("**六席全是交叉進場。**")
    for base, word in (("零軸", "零軸"), ("背離", "背離")):
        best = rk[rk["交易策略"].str.startswith(base)].iloc[0]
        assert best["全期名次"] > TOP_N
        c.append(f"{word}最好的一組排第 {best['全期名次']}（{wan(best['總獲利(萬)'])} 萬）")

    ma = top[top["交易策略"].str.contains("均線多頭")]
    assert set(ma["排序"]) == {"低價", "流動性", "高價"}
    vals = [ma[ma["排序"] == k]["總獲利(萬)"].iloc[0] for k in ("低價", "流動性", "高價")]
    c.append("它在低價、流動性、高價三種排序下都進了前六（" + "、".join(wan(v) for v in vals) + " 萬）")
    nf = rk[rk["交易策略"].str.contains("無濾網") & (rk["投法"] == "比例")].set_index("排序")
    c.append(f"無濾網換成流動性只剩 {wan(nf.loc['流動性', '總獲利(萬)'])} 萬、換成高價只剩 {wan(nf.loc['高價', '總獲利(萬)'])} 萬")

    first = top.iloc[0]
    higher = top[top["獲利因子"] > first["獲利因子"]]
    assert len(higher) == 2 and set(higher["投法"]) == {"比例"}
    nf_row = higher[higher["交易策略"].str.contains("無濾網")].iloc[0]
    adx_row = higher[higher["交易策略"].str.contains("ADX")].iloc[0]
    c.append(f"第 1 名的獲利因子是 {half_up(first['獲利因子'], 4)}，排在它後面的無濾網（{half_up(nf_row['獲利因子'], 4)}）"
             f"與 ADX>25（{half_up(adx_row['獲利因子'], 4)}）都比它高")
    assert first["交易次數"] > nf_row["交易次數"] and first["擋單"] < nf_row["擋單"]
    c.append(f"（{first['交易次數']:,} 筆、擋單 {first['擋單']:,}，無濾網是 {nf_row['交易次數']:,} 筆、擋單 {nf_row['擋單']:,}）")
    return c


def bench_claims(rk: pd.DataFrame, conc: pd.DataFrame, all15: pd.DataFrame) -> list:
    """第二節：2015–2025 對 0050，以及「放大到 30 組」。"""
    c = []
    b = conc[conc["交易策略"] == BENCH].iloc[0]
    six = conc[conc["交易策略"] != BENCH].copy()
    best_cagr = six.sort_values("年化報酬率%", ascending=False).iloc[0]
    assert best_cagr["全期名次"] == 4 and best_cagr["投法"] == "比例"
    assert best_cagr["年化報酬率%"] / b["年化報酬率%"] < 0.7
    c.append(f"六組裡最好的交叉 × ADX>25 · 比例 · 低價（{half_up(best_cagr['年化報酬率%'], 2)}%）連 0050 的 "
             f"{half_up(b['年化報酬率%'], 2)}% 的七成都不到")
    c.append(f"是 {half_up(best_cagr['資金倍數'], 2)} 倍對 {half_up(b['資金倍數'], 2)} 倍")
    c.append(f"0050 的報酬回撤比約 {half_up(b['報酬回撤比'], 2)}，高過六組裡最高的 {half_up(six['報酬回撤比'].max(), 2)}")
    assert (six["最大回撤%"] > b["最大回撤%"]).all()
    c.append(f"六組的最大回撤落在 {half_up(six['最大回撤%'].min(), 1)}%～{half_up(six['最大回撤%'].max(), 1)}%，"
             f"全都比 0050 的 {half_up(b['最大回撤%'], 1)}% 深")

    six["rdr_rank"] = six["報酬回撤比"].rank(ascending=False).astype(int)
    first = six[six["全期名次"] == 1].iloc[0]
    top_rdr = six.sort_values("報酬回撤比", ascending=False).iloc[0]
    assert top_rdr["全期名次"] == 4 and "ADX" in top_rdr["交易策略"] and top_rdr["投法"] == "比例"
    c.append(f"在 2015–2025 的報酬回撤比只排六組裡的第 {first['rdr_rank']}（{half_up(first['報酬回撤比'], 3)}）")
    c.append("同期最穩的是全期第 4 名的 ADX>25 · 比例")

    best30 = all15.sort_values("報酬回撤比", ascending=False).iloc[0]
    key = (best30["交易策略"], best30["投法"], best30["排序"])
    r30 = rk.set_index(["交易策略", "投法", "排序"]).loc[key, "全期名次"]
    assert best30["交易策略"].startswith("零軸") and best30["投法"] == "比例" and best30["排序"] == "低價"
    c.append(f"是全期第 {r30} 名的零軸 × 收盤>MA200 · 比例 · 低價（{half_up(best30['報酬回撤比'], 3)}）")
    shallow = all15[all15["最大回撤%"] < b["最大回撤%"]]
    assert set(shallow["交易策略"].str[:2]) == {"零軸"} and len(shallow) == 6
    c.append(f"六種組合的回撤（{half_up(shallow['最大回撤%'].min(), 1)}%～{half_up(shallow['最大回撤%'].max(), 1)}%）都比 0050 淺")
    assert all15["年化報酬率%"].max() < b["年化報酬率%"] and all15["報酬回撤比"].max() < b["報酬回撤比"]
    c.append(f"但年化最高也只有 {half_up(all15['年化報酬率%'].max(), 2)}%")

    top = rk.head(TOP_N)
    c.append(f"前六名的總獲利從 {wan(top['總獲利(萬)'].min())} 萬到 {wan(top['總獲利(萬)'].max())} 萬")
    return c


def summary_claims(rk: pd.DataFrame, all15: pd.DataFrame, conc: pd.DataFrame) -> list:
    """重點整理的多股段落。"""
    c = []
    nf = rk[rk["交易策略"].str.contains("無濾網") & (rk["投法"] == "比例")].set_index("排序")
    c.append(f"總獲利前六名有五名是比例投入，同一組交易策略換個買入排序，總獲利可以從 "
             f"{wan(nf.loc['低價', '總獲利(萬)'])} 萬掉到 {wan(nf.loc['高價', '總獲利(萬)'])} 萬")
    b = conc[conc["交易策略"] == BENCH].iloc[0]
    assert all15["年化報酬率%"].max() < b["年化報酬率%"]
    c.append("30 組裡沒有一組的年化或報酬回撤比追得上 0050；回撤比 0050 淺的只有零軸那一組交易策略")
    return c


def main():
    text = read_post()
    rk, conc = ranked(), conclusion()
    all15 = with_exact(pd.read_csv(ALL_2015, encoding="utf-8-sig"))
    errors = 0
    rows = table1(rk).splitlines() + table2(conc).splitlines()
    for row in rows:
        if row not in text:
            errors += 1
            print(f"✗ 表格列不符：{row[:120]}")
    claims = inventory_claims(rk) + bench_claims(rk, conc, all15) + summary_claims(rk, all15, conc)
    for cl in claims:
        if cl not in text:
            errors += 1
            print(f"✗ 正文缺：{cl}")
    print(f"表格 {len(rows)} 列、正文主張 {len(claims)} 條｜錯誤 {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
