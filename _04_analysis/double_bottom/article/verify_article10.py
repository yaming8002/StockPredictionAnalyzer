# -*- coding: utf-8 -*-
"""
驗〈雙底（十）：多股共用資金 · 固定比例投入〉的表格與正文數字，全部由多股結果重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 4 張表必須與 build_article10_tables 產出的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article10.py [--print]
（--print：只印出重算字串，不比對文章）
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

import pandas as pd  # noqa: E402

from build_article2_tables import half_up  # noqa: E402
from build_article9_tables import EXACT, load, random_note  # noqa: E402
from build_article10_tables import all_tables10  # noqa: E402
from verify_article4 import article_path  # noqa: E402
from verify_article9 import NAMES, _pct, _rng, per_strategy  # noqa: E402

SLUG = "double-bottom-multi-ratio"
FLOOR_AMT = 10_000


def _srng(xs, nd, unit="%"):
    """可能跨正負的區間，負號用全形減號。"""
    return _rng(xs, nd, unit).replace("-", "−")


def _beyond(df: pd.DataFrame, kind: str, side: str) -> bool:
    """每組的 kind 排序總獲利是否都在隨機 1,000 次的 P95 之上（side=hi）或 P5 之下（side=lo）。"""
    ok = []
    for _, b in df.groupby("交易策略"):
        b = b.set_index("排序")
        v, rnd = b.loc[kind, "總獲利(萬)"], b.loc["隨機"]
        ok.append(v > rnd["總獲利(萬)P95" + EXACT] if side == "hi" else v < rnd["總獲利(萬)P5" + EXACT])
    return all(ok)


def claims(df: pd.DataFrame, fixed: pd.DataFrame) -> dict:
    s, sf = per_strategy(df), per_strategy(fixed)
    lo = df[df["排序"] == "低價"].set_index("交易策略")
    lo_f = fixed[fixed["排序"] == "低價"].set_index("交易策略")
    units = df.drop_duplicates("交易策略").set_index("交易策略")["份數"]
    units_f = fixed.drop_duplicates("交易策略").set_index("交易策略")["份數"]
    start = 1_000_000 / units
    ratio = lo["總獲利(萬)"] / lo_f["總獲利(萬)"]
    n_ratio = lo["交易次數"] / lo_f["交易次數"]
    by_ratio = s.sort_values("擋單倍數")
    by_width = s.sort_values("寬度")
    ratio_lines = "\n".join(f"- {n}：{half_up(r, 2)} 倍（固定金額投入是 {half_up(sf.loc[n, '擋單倍數'], 2)} 倍）"
                            for n, r in by_ratio["擋單倍數"].items())
    width_lines = "\n".join(f"- {n}：{_pct(r['寬度'])}%（{half_up(r['P5'], 1)}～{half_up(r['P95'], 1)} 萬）"
                            for n, r in by_width.iterrows())
    base = s.loc["基本版"]
    return {
        "份數": (True, f"切出來是 {units.min()}～{units.max()} 份"),
        "起手": ((start < FLOOR_AMT).all(), f"100 萬 ÷ 份數只有 {half_up(start.min(), 0):,}～{half_up(start.max(), 0):,} 元，"
                                           "五組都低於 1 萬的下限"),
        "接手": (True, f"已實現權益要長到 {units.min()}～{units.max()} 萬"),
        "下限倍數": (True, f"固定金額投入每筆 {half_up(1e6 / units_f.max(), 0):,}～{half_up(1e6 / units_f.min(), 0):,} 元，"
                       f"比例投入起手就是 1 萬，是它的 {_rng(FLOOR_AMT / (1e6 / units_f), 2)} 倍"),
        "低價全贏": (s["低勝隨"].all() and s["高輸隨"].all(), "**低價優先五組都贏過隨機**，高價優先、成交金額優先五組都輸給隨機"),
        "總獲利最高": (lo["總獲利(萬)"].idxmax() == "基本版",
                    f"總獲利最高的是基本版（低價 {half_up(lo.loc['基本版', '總獲利(萬)'], 0)} 萬）"),
        "PF最高": (lo["獲利因子"].idxmax() == NAMES["rise_base"],
                 f"獲利因子最高的是第二底墊高 ≥ 2% × 基本出場（低價 {half_up(lo.loc[NAMES['rise_base'], '獲利因子'], 4)}）"),
        "中位數負": ((df["中位數%"] < 0).all(),
                 f"中位數報酬五組都是負的（{half_up(df['中位數%'].max(), 2)}%～{half_up(df['中位數%'].min(), 2)}%）"
                 .replace("-", "−")),
        "擋單倍數": ((s["擋單倍數"] > 1).all(), f"五組是 {_rng(s['擋單倍數'], 2)} 倍"),
        "定額擋單": (True, f"固定金額投入只有 {_rng(sf['擋單倍數'], 2)} 倍"),
        "擋單清單": (True, ratio_lines),
        "低對高": (True, f"同一組裡低價對高價的總獲利是 {_rng(s['低對高'], 2)} 倍"),
        "定額低對高": (True, f"固定金額投入是 {_rng(sf['低對高'], 2)} 倍"),
        "低對隨": (True, f"低價對隨機多 {_rng(s['低對隨'] * 100, 1, '%')}（固定金額投入是 "
                       f"{_rng(sf['低對隨'] * 100, 1, '%')}）"),
        "拆解五組": ((s["EV多"] > 0.2).all() and (s["筆數多"].abs() < 0.01).all(),
                 f"每筆期望值多 {_rng(s['EV多'] * 100, 1, '%')}，成交筆數則在 {_srng(s['筆數多'] * 100, 1)} 之間"),
        "拆解基本版": (True, f"基本版每筆期望值多 {_pct(base['EV多'])}%、成交筆數{'多' if base['筆數多'] >= 0 else '少'} "
                         f"{_pct(abs(base['筆數多']))}%"),
        "寬度清單": (True, width_lines),
        "寬度範圍": (True, f"固定金額投入是 {_rng(sf['寬度'] * 100, 1, '%')}，這裡變成 {_rng(s['寬度'] * 100, 1, '%')}"),
        "超出區間": (_beyond(df, "低價", "hi") and _beyond(df, "高價", "lo")
                 and _beyond(fixed, "低價", "hi") and _beyond(fixed, "高價", "lo"),
                 "五組的低價優先都高於隨機 1,000 次的 P95，高價優先都低於 P5"),
        "年化": (True, f"**年化報酬 {_rng(df['年化報酬(%)'], 2, '%')}**，最大回撤 {_rng(df['最大回撤(%)'], 1, '%')}"),
        "定額年化": (True, f"固定金額投入是年化 {_rng(fixed['年化報酬(%)'], 2, '%')}、最大回撤 "
                       f"{_rng(fixed['最大回撤(%)'], 1, '%')}"),
        "期末": (True, f"基本版低價優先的期末權益是 {half_up(lo.loc['基本版', '最終權益'] / 1e4, 0):,} 萬"),
        "防線": ((df["已實現權益最低(%)"] >= 80).all() and len(df) == 20,
               f"**已實現權益最低是本金的 {_rng(df['已實現權益最低(%)'], 1, '%')}**，二十列都沒有跌破八成"),
        "隨機附註": (True, random_note(df)),
        "比例倍數": ((ratio > 1).all(), f"比例的總獲利是定額的 {_rng(ratio, 2)} 倍"),
        "筆數比": (True, f"交易次數只有定額的 {_rng(n_ratio * 100, 0, '%')}"),
        "PF上升": ((lo["獲利因子"] > lo_f["獲利因子"]).all(), "獲利因子五組都比固定金額投入高"),
        "回撤加深": ((lo["最大回撤(%)"] > lo_f["最大回撤(%)"]).all(), "最大回撤五組都比固定金額投入深"),
        "整理": (True, f"**擋單變成成交筆數的 {_rng(s['擋單倍數'], 2)} 倍**"),
        "整理低價": (True, f"低價對隨機的總獲利多 {_rng(s['低對隨'] * 100, 1, '%')}，低價對高價是 {_rng(s['低對高'], 2)} 倍"),
        "整理年化": (True, f"**年化報酬 {_rng(df['年化報酬(%)'], 2, '%')}、最大回撤 {_rng(df['最大回撤(%)'], 1, '%')}**；"
                         f"已實現權益最低 {_rng(df['已實現權益最低(%)'], 1, '%')}"),
        "摘要": (True, f"擋單變成成交筆數的 {_rng(s['擋單倍數'], 2)} 倍，低價優先五組都贏過隨機，高價與成交金額優先五組都輸給隨機，"
                       f"低價對隨機的總獲利多 {_rng(s['低對隨'] * 100, 1, '%')}"),
    }


def main() -> int:
    df, fixed = load("比例"), load("定額")
    cl = claims(df, fixed)
    if "--print" in sys.argv:
        for name, (ok, phrase) in cl.items():
            print(f"[{name}] {'OK' if ok else '✗ 判斷不成立'}\n{phrase}\n")
        return 0
    text = io.open(article_path(SLUG), encoding="utf-8").read()
    errs = []
    want = all_tables10()
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
