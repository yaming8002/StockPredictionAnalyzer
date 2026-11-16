# -*- coding: utf-8 -*-
"""
驗〈雙底（十一）：一路走來的總結評估〉的表格與正文數字，全部由多股結果重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 2 張表必須與 build_article11_tables 產出的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。
（回顧〈雙底（一）～（八）〉的段落引用的是那幾篇已核對過的數字，不在這裡重算。）

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article11.py [--print]
"""
import io
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_article2_tables import half_up  # noqa: E402
from build_article11_tables import (BENCH, P15, TOP_N, all_2015, all_tables11, conclusion, fmt2,  # noqa: E402
                                    picks_2015, ranked)
from verify_article4 import article_path  # noqa: E402

SLUG = "double-bottom-conclusion"


def _name(r) -> str:
    return f"{r['交易策略']} · {r['投法']} · {r['排序']}"


def claims() -> dict:
    rk, conc, a15 = ranked(), conclusion(), all_2015()
    top = rk.head(TOP_N)
    b = conc[conc["交易策略"] == BENCH].iloc[0]
    picks = conc[conc["交易策略"] != BENCH].sort_values("全期名次").reset_index(drop=True)
    best_fixed = rk[rk["投法"] == "定額"].iloc[0]
    sixth = top.iloc[-1]
    best_cagr = a15.loc[a15["年化報酬率%"].idxmax()]
    best_rr = a15.loc[a15["報酬回撤比"].idxmax()]
    # 表二排序：依 2015–2025 總獲利嚴格遞減；資金倍數必須同向（同一段期間、同樣 100 萬起跑，兩者應一致）
    t2 = picks_2015(conc)
    t2_sorted = (t2[P15].is_monotonic_decreasing and t2["資金倍數"].is_monotonic_decreasing
                 and t2["同期名次"].tolist() == list(range(1, len(t2) + 1)))
    return {
        "比例在前": ((rk.head(15)["投法"] == "比例").all(), "比例投入的 15 組全部排在固定金額投入的 15 組前面"),
        "前六": ((top.head(5)["排序"] == "低價").all() and sixth["排序"] != "低價",
               f"前五名都是低價優先，第 6 名是{_name(sixth)}（{half_up(sixth['總獲利(萬)'], 0)} 萬）"),
        "定額最好": (True, f"固定金額投入最好的一組是{_name(best_fixed)}（{half_up(best_fixed['總獲利(萬)'], 0)} 萬），"
                         f"排第 {best_fixed['全期名次']}"),
        "年化最高": (best_cagr["年化報酬率%"] < b["年化報酬率%"],
                 f"年化最高的是{_name(best_cagr)}（{fmt2('年化報酬率%', best_cagr['年化報酬率%'])}%），"
                 f"0050 是 {fmt2('年化報酬率%', b['年化報酬率%'])}%"),
        "30年化": ((a15["年化報酬率%"] < b["年化報酬率%"]).all() and len(a15) == 30,
                 f"30 組的年化落在 {fmt2('年化報酬率%', a15['年化報酬率%'].min())}%～"
                 f"{fmt2('年化報酬率%', a15['年化報酬率%'].max())}%，沒有一組追上 0050"),
        "30回撤": ((a15["最大回撤%"] < b["最大回撤%"]).all(),
                 f"30 組的最大回撤落在 {fmt2('最大回撤%', a15['最大回撤%'].min())}%～"
                 f"{fmt2('最大回撤%', a15['最大回撤%'].max())}%，全都比 0050 的 {fmt2('最大回撤%', b['最大回撤%'])}% 淺"),
        "回撤比最高": (best_rr["報酬回撤比"] < b["報酬回撤比"],
                  f"報酬回撤比最高的是{_name(best_rr)}（{fmt2('報酬回撤比', best_rr['報酬回撤比'])}），"
                  f"0050 是 {fmt2('報酬回撤比', b['報酬回撤比'])}"),
        "表二排序": (t2_sorted, "本表依 2015–2025 總獲利由高到低排。"),
    }


def main() -> int:
    cl = claims()
    if "--print" in sys.argv:
        for name, (ok, phrase) in cl.items():
            print(f"[{name}] {'OK' if ok else '✗ 判斷不成立'}\n{phrase}\n")
        return 0
    text = io.open(article_path(SLUG), encoding="utf-8").read()
    errs = []
    want = all_tables11()
    for name, html in want.items():
        if html not in text:
            errs.append(f"表格與重算不一致：{name}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    for name, (ok, phrase) in cl.items():
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    print(f"表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
