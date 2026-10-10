# -*- coding: utf-8 -*-
"""
驗 KD 交叉（六）kd-cross-multi（多股・固定金額投入）：份數表、結果表逐格（含標色），
隨機列資金倍數註，以及正文所有數字句與「贏／輸隨機」「墊底」「差幾倍」等計數句。

結果表、份數表、隨機註都與 build_kd_multi_tables 的輸出逐字比對；落在 .5 又還原不了方向的格子
列為警告（見 kd_article_common.multi_cell）。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article6.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.build_kd_multi_tables import (load_multi, random_note,  # noqa: E402
                                                           tables_for)
from _04_analysis.kd.article.kd_article_common import MULTI, Checker, fmt, read_post  # noqa: E402

STRONG = ["創60日新高", "創120日新高", "創250日新高", "跳空"]
WEAK = ["底背離", "低檔＋紅K"]


def pick(df, name, order):
    return df[(df["交易策略"] == name) & (df["排序"] == order)].iloc[0]


def ordering_claims(ck: Checker, text: str, df: pd.DataFrame):
    names = df["交易策略"].unique()
    low_win = [n for n in names if pick(df, n, "低價")["總獲利(萬)"] > pick(df, n, "隨機")["總獲利(萬)"]
               and pick(df, n, "低價")["獲利因子"] > pick(df, n, "隨機")["獲利因子"]]
    ck.check(len(low_win) == 6, f"六種策略低價都贏過隨機：{low_win}")
    gap = {n: pick(df, n, "低價")["總獲利(萬)"] - pick(df, n, "隨機")["總獲利(萬)"] for n in names}
    ck.check(min(gap[n] for n in STRONG) > 100 and max(gap[n] for n in WEAK) < 20,
             f"前四種差距明顯、後兩種差距小：{ {k: round(v, 1) for k, v in gap.items()} }")
    c = pick(df, "創120日新高", "低價"), pick(df, "創120日新高", "隨機")
    ck.contains(text, f"創 120 新高：低價總獲利 {fmt(c[0]['總獲利(萬)'], 0)} 萬、獲利因子 {fmt(c[0]['獲利因子'], 2)}，"
                      f"高過隨機的 {fmt(c[1]['總獲利(萬)'], 0)} 萬／{fmt(c[1]['獲利因子'], 2)}", "創 120 例子")
    for order in ("流動性", "高價"):
        lose = sum(pick(df, n, order)["總獲利(萬)"] < pick(df, n, "隨機")["總獲利(萬)"] for n in names)
        ck.check(lose >= 4, f"{order}「多半輸給隨機」：{lose}/6")
    last = sum(df[df["交易策略"] == n].sort_values("總獲利(萬)").iloc[0]["排序"] == "高價" for n in names)
    ck.check(last == 6, f"高價「六種全數墊底」：{last}/6")
    ratios = []
    for n in STRONG:
        tot = df[df["交易策略"] == n]["總獲利(萬)"]
        ratios.append(tot.max() / tot.min())
    ck.check(1.5 <= min(ratios) < 2 and 9 <= max(ratios) < 10,
             f"前四種「差近兩倍到近十倍」：{[round(r, 2) for r in ratios]}")
    ck.contains(text, "前四種策略的總獲利就能差近兩倍到近十倍", "差幾倍")


def weak_claims(ck: Checker, text: str, df: pd.DataFrame):
    dv, lr = df[df["交易策略"] == "底背離"], df[df["交易策略"] == "低檔＋紅K"]
    ck.contains(text, f"底背離擋單少（{fmt(dv['擋單'].min(), 0)}～{fmt(dv['擋單'].max(), 0)}），"
                      f"四種排序（含隨機）的總獲利都落在 {fmt(dv['總獲利(萬)'].min(), 0)}～"
                      f"{fmt(dv['總獲利(萬)'].max(), 0)} 萬", "底背離範圍")
    others = df[~df["交易策略"].isin(WEAK)]
    ck.check(dv["擋單"].max() < others["擋單"].min() and dv["擋單"].max() < lr["擋單"].min(), "底背離擋單最少")
    ck.contains(text, f"低檔＋紅K 四種排序都黏在損益兩平上下（{fmt(lr['總獲利(萬)'].min(), 0)}～"
                      f"{fmt(lr['總獲利(萬)'].max(), 0)} 萬）", "低檔＋紅K 範圍")
    ck.check(lr["獲利因子"].between(0.95, 1.07).all(), f"低檔＋紅K PF 在 1 附近：{lr['獲利因子'].tolist()}")


def main() -> int:
    ck = Checker("KD 交叉（六）")
    text = read_post("kd-cross-multi")
    ck.tables(text, tables_for(6))
    ck.contains(text, f"隨機列的資金倍數中位（p5／p95）：{random_note('定額')}。", "隨機倍數註")
    units = pd.read_csv(os.path.join(MULTI, "units.csv"), encoding="utf-8-sig")  # 份數表（整數，無精確欄）
    ck.contains(text, f"估出的「最壞連敗」（{units['S'].min()}～{units['S'].max()} 次）", "S 範圍")
    ck.check((units["定額份數"] == (units["S"] / 0.2).round().astype(int)).all(), "定額份數＝S÷0.2")
    ck.contains(text, "（例如份數 75 → 13,333 元）", "每筆金額例")
    df = load_multi("定額")
    ordering_claims(ck, text, df)
    weak_claims(ck, text, df)
    ck.contains(text, "2,258 檔、2002–2025", "期間")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
