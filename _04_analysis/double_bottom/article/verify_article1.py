# -*- coding: utf-8 -*-
"""
驗〈雙底（一）〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 兩張表：逐列逐欄 ＝ build_article1_tables 由逐筆交易重算、一次 half-up 的結果（列數、順序都要對）。
2. 正文主張：每一條數字敘述在這裡用資料重算一次（區間、單調、比例、倍數……），
   重算結果格式化後必須逐字出現在文章裡；判斷式（例「九組全部為正」「一路往上」）不成立也算錯。
3. 示意圖的數字（88、92、96.6）用圖檔腳本的同一組假想走勢重算。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article1.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from build_article1_tables import COLS, fmt, half_up, stats  # noqa: E402

SLUG = "double-bottom-basic"
ROW_RE = re.compile(r"<tr><td>(.*?)</td>(.*?)</tr>")
TD_RE = re.compile(r"<td>(.*?)</td>")


def table_errors(text: str, df: pd.DataFrame) -> list:
    """兩張表逐列比對（表一＝裸版、表二＝可成交，各 9 列，依門檻排序）。"""
    errs = []
    tables = re.findall(r'<table class="bt-table">.*?<tbody>(.*?)</tbody>', text, re.S)
    if len(tables) != 2:
        return [f"應有 2 張結果表，實際 {len(tables)}"]
    for liq, body in zip((False, True), tables):
        want = df[df["可成交"] == liq].reset_index(drop=True)
        rows = ROW_RE.findall(body)
        if len(rows) != len(want):
            errs.append(f"{'可成交' if liq else '裸版'}表列數 {len(rows)} ≠ {len(want)}")
            continue
        for (label, rest), (_, r) in zip(rows, want.iterrows()):
            if not label.startswith(f"ZigZag {int(r['門檻%'])}%"):
                errs.append(f"列名順序不符：{label}（應為 ZigZag {int(r['門檻%'])}%）")
            got = TD_RE.findall(rest)
            exp = [fmt(c, r[c]) for c in COLS]
            for c, g, e in zip(COLS, got, exp):
                if g != e:
                    errs.append(f"{label}｜{c}：文章 {g}，重算 {e}")
    return errs


def mono_up(s) -> bool:
    return bool(np.all(np.diff(np.asarray(s, dtype=float)) > 0))


def mono_down(s) -> bool:
    return bool(np.all(np.diff(np.asarray(s, dtype=float)) < 0))


def claims(df: pd.DataFrame) -> list:
    """(說明, 判斷式是否成立, 文章裡必須出現的字串)。"""
    raw = df[~df["可成交"]].reset_index(drop=True)
    liq = df[df["可成交"]].reset_index(drop=True)
    f1 = lambda v: f"{half_up(v, 1)}"
    f2 = lambda v: f"{half_up(v, 2)}"
    f0 = lambda v: f"{half_up(v, 0):,}"
    rng = lambda s: f"{half_up(s.min(), 1)}%～{half_up(s.max(), 1)}%"   # 九組的範圍（一位小數）
    out = [
        ("裸版九組總獲利皆正", bool((raw["總獲利(萬)"] > 0).all()), "九種門檻全部是正的"),
        ("裸版 PF 區間", True, f"獲利因子從 {f2(raw['獲利因子'].iloc[0])} 到 {f2(raw['獲利因子'].iloc[-1])}"),
        ("裸版總損益區間", True, f"總損益從 +{f0(raw['總獲利(萬)'].iloc[0])} 萬到 +{f0(raw['總獲利(萬)'].iloc[-1])} 萬"),
        ("裸版 PF 單調上升", mono_up(raw["獲利因子"]),
         f"{f2(raw['獲利因子'].iloc[0])} → {f2(raw['獲利因子'].iloc[1])} → {f2(raw['獲利因子'].iloc[2])} → … → {f2(raw['獲利因子'].iloc[-1])}"),
        ("裸版 EV 單調上升", mono_up(raw["期望值/筆"]),
         f"每筆期望值從 +{f0(raw['期望值/筆'].iloc[0])} 元一路到 +{f0(raw['期望值/筆'].iloc[-1])} 元"),
        ("裸版總獲利也單調（九組沒有一組回頭）", mono_up(raw["總獲利(萬)"]), "九組沒有一組回頭"),
        ("裸版勝率範圍 31~33", bool(raw["勝率%"].between(31, 33).all()), "31%～33%"),
        ("裸版獲利平均端點且單調", mono_up(raw["獲利平均%"]),
         f"獲利平均從 +{f1(raw['獲利平均%'].iloc[0])}% 拉到 +{f1(raw['獲利平均%'].iloc[-1])}%"),
        ("裸版虧損平均端點且單調", mono_down(raw["虧損平均%"]),
         f"虧損平均也從 −{f1(-raw['虧損平均%'].iloc[0])}% 擴大到 −{f1(-raw['虧損平均%'].iloc[-1])}%"),
        ("裸版中位數端點且單調", mono_down(raw["中位數%"]),
         f"中位數一路從 −{f1(-raw['中位數%'].iloc[0])}% 降到 −{f1(-raw['中位數%'].iloc[-1])}%"),
        ("裸版持有天數端點且單調", mono_up(raw["平均持有天"]),
         f"從 {f0(raw['平均持有天'].iloc[0])} 天變成 {f0(raw['平均持有天'].iloc[-1])} 天"),
        ("裸版交易次數端點（萬筆）", mono_down(raw["交易次數"]),
         f"交易次數則從 {f1(raw['交易次數'].iloc[0] / 1e4)} 萬筆降到 {f1(raw['交易次數'].iloc[-1] / 1e4)} 萬筆"),
        ("裸版跌破低點占比範圍", True, f"**跌破波段低點**：{rng(raw['跌破低點占比%'])}"),
        ("裸版高點下移占比範圍", True, f"**高點下移**：{rng(raw['CHoCH占比%'])}"),
        ("裸版同日占比範圍", True, f"**同一天兩條都成立**：{rng(raw['同日占比%'])}"),
        ("裸版未平倉率端點且單調", mono_up(raw["未平倉率%"]),
         f"佔全部進場的 {f1(raw['未平倉率%'].min())}%～{f1(raw['未平倉率%'].max())}%（門檻越大越多）"),
    ]
    drop = 1 - liq["交易次數"] / raw["交易次數"]
    pf_drop = raw["獲利因子"] - liq["獲利因子"]
    out += [
        ("可成交筆數少三到四成", bool(drop.between(0.3, 0.4).all()), "交易筆數少了三到四成"),
        ("可成交九組皆正且 PF 單調", bool((liq["總獲利(萬)"] > 0).all()) and mono_up(liq["獲利因子"]),
         f"獲利因子 {f2(liq['獲利因子'].iloc[0])} → {f2(liq['獲利因子'].iloc[-1])}"),
        ("PF 降幅區間", bool((pf_drop > 0).all()), f"獲利因子掉了 {f2(pf_drop.min())}～{f2(pf_drop.max())}"),
        ("總損益區間 裸版→可成交", True,
         f"總損益從 +{f0(raw['總獲利(萬)'].min())}～+{f0(raw['總獲利(萬)'].max())} 萬縮成 "
         f"+{f0(liq['總獲利(萬)'].min())}～+{f0(liq['總獲利(萬)'].max())} 萬"),
        ("可成交 2% 接近打平", True,
         f"獲利因子 {f2(liq['獲利因子'].iloc[0])}、每筆期望值 +{f0(liq['期望值/筆'].iloc[0])} 元"),
        ("可成交勝率 30~31", bool(liq["勝率%"].between(30, 31).all()), "落在 30%～31%"),
        ("可成交出場比例範圍", True,
         f"跌破波段低點 {rng(liq['跌破低點占比%'])}、高點下移 {rng(liq['CHoCH占比%'])}、同一天 {rng(liq['同日占比%'])}"),
        ("重點：裸版 PF 區間", True, f"獲利因子從 {f2(raw['獲利因子'].iloc[0])} 到 {f2(raw['獲利因子'].iloc[-1])}"),
        ("重點：可成交 PF 區間", True, f"獲利因子落在 {f2(liq['獲利因子'].iloc[0])} 到 {f2(liq['獲利因子'].iloc[-1])}"),
        ("重點：勝率三成出頭", bool(raw["勝率%"].between(30, 34).all()), "勝率一直停在三成出頭"),
    ]
    return out


def chart_claims() -> list:
    """示意圖數字：用圖檔腳本的同一組假想走勢重算第一、二個底與確認門檻。"""
    from _01_data.indicators_pattern import calculate_zigzag
    from _04_analysis.double_bottom.charts.double_bottom_concept import PCT, price_path
    zz = calculate_zigzag(pd.DataFrame({"close": price_path()}), PCT)
    lows = zz["zigzag_turn_low"].dropna().to_numpy()
    l1, l2 = lows[0], lows[1]
    return [("圖：第一個底", True, f"股價先跌到 {l1:g} 元（第一個底）"),
            ("圖：第二個底高於第一個", l2 > l1, f"只跌到 {l2:g} 元（第二個底），比 {l1:g} 元高"),
            ("圖：確認門檻", True, f"{l2:g} × 1.05 ＝ {half_up(l2 * (1 + PCT), 1)} 元")]


def main() -> int:
    path = os.path.join(common.require_blog_dir(), "site", "content", "posts", f"{SLUG}.md")
    text = io.open(path, encoding="utf-8").read()
    df = stats()
    errs = table_errors(text, df)
    n_claims = 0
    for name, ok, phrase in claims(df) + chart_claims():
        n_claims += 1
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    print(f"表格 2 張 × 9 列、正文主張 {n_claims} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
