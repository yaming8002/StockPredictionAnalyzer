# -*- coding: utf-8 -*-
"""
驗 KD 交叉（七）kd-cross-multi-ratio（多股・固定比例投入）：份數表、結果表、投法對照表逐格（含標色），
隨機列資金倍數註，以及正文的贏輸計數、範圍、倍數與「交易次數只剩約三成」等句。

執行：
    BLOG_DIR=<blog 專案根目錄> PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/verify_kd_article7.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.build_kd_multi_tables import (block_order, load_multi,  # noqa: E402
                                                           random_note, tables_for)
from _04_analysis.kd.article.kd_article_common import MULTI, Checker, fmt, read_post  # noqa: E402

STRONG = ["創60日新高", "創120日新高", "創250日新高", "跳空"]
WEAK = ["底背離", "低檔＋紅K"]


def pick(df, name, order):
    return df[(df["交易策略"] == name) & (df["排序"] == order)].iloc[0]


def ordering_claims(ck: Checker, text: str, pct: pd.DataFrame):
    names = block_order(pct)
    win = [n for n in names if pick(pct, n, "低價")["總獲利(萬)"] > pick(pct, n, "隨機")["總獲利(萬)"]]
    ck.check(win == names[:5] and names[5] == "低檔＋紅K", f"低價在前五種贏過隨機、低檔＋紅K 例外：{win}")
    lr = pick(pct, "低檔＋紅K", "低價"), pick(pct, "低檔＋紅K", "隨機")
    ck.contains(text, f"只有低檔＋紅K 的低價（{fmt(lr[0]['總獲利(萬)'], 0)} 萬）輸給隨機"
                      f"（{fmt(lr[1]['總獲利(萬)'], 0)} 萬）", "低檔＋紅K 例外")
    c = pick(pct, "創120日新高", "低價"), pick(pct, "創120日新高", "隨機")
    ck.contains(text, f"創 120 低價總獲利 {fmt(c[0]['總獲利(萬)'], 0)} 萬／獲利因子 {fmt(c[0]['獲利因子'], 2)}，"
                      f"遠高於隨機的 {fmt(c[1]['總獲利(萬)'], 0)} 萬／{fmt(c[1]['獲利因子'], 2)}", "創 120 例子")
    for order in ("流動性", "高價"):
        lose = sum(pick(pct, n, order)["總獲利(萬)"] < pick(pct, n, "隨機")["總獲利(萬)"]
                   and pick(pct, n, order)["獲利因子"] < pick(pct, n, "隨機")["獲利因子"] for n in names)
        ck.check(lose == 6, f"{order}六種策略全部輸給隨機：{lose}/6")
    dv, lw = pct[pct["交易策略"] == "底背離"], pct[pct["交易策略"] == "低檔＋紅K"]
    ck.contains(text, f"底背離四種排序總獲利 {fmt(dv['總獲利(萬)'].min(), 0)}～{fmt(dv['總獲利(萬)'].max(), 0)} 萬、"
                      f"低檔＋紅K {fmt(lw['總獲利(萬)'].min(), 0)}～{fmt(lw['總獲利(萬)'].max(), 0)} 萬", "弱策略範圍")


def compare_claims(ck: Checker, text: str, fix: pd.DataFrame, pct: pd.DataFrame):
    f = {n: pick(fix, n, "低價") for n in STRONG + WEAK}
    p = {n: pick(pct, n, "低價") for n in STRONG + WEAK}
    share = {n: p[n]["交易次數"] / f[n]["交易次數"] for n in f}
    s_lo, s_hi = min(share[n] for n in STRONG), max(share[n] for n in STRONG)
    ck.contains(text, f"三種創新高與跳空的比例交易次數只剩定額的約三成（{s_lo * 100:.0f}～{s_hi * 100:.0f}%）",
                "強策略交易次數比例")
    ck.check(0.25 <= s_lo and s_hi < 0.4, f"約三成：{s_lo:.3f}～{s_hi:.3f}")
    cut = [1 - share[n] for n in WEAK]
    ck.check(all(0.15 <= c < 0.35 for c in cut), f"底背離、低檔＋紅K 少了兩到三成：{[round(c, 3) for c in cut]}")
    ck.check(all(p[n]["擋單"] > f[n]["擋單"] for n in f), "擋單六種都變多")
    ck.check(all(p[n]["最大回撤%"] > f[n]["最大回撤%"] for n in f), "比例回撤六種都更深（「回撤更大」）")
    up = [n for n in f if p[n]["獲利因子"] > f[n]["獲利因子"] and p[n]["總獲利(萬)"] > f[n]["總獲利(萬)"]]
    ck.check(set(up) == set(STRONG), f"強策略被放大、弱策略被拖累：{up}")
    ck.contains(text, f"（創 120 獲利因子 {fmt(p['創120日新高']['獲利因子'], 2)} vs "
                      f"{fmt(f['創120日新高']['獲利因子'], 2)}）", "創 120 對照")
    ck.contains(text, f"（底背離 {fmt(p['底背離']['獲利因子'], 2)} vs {fmt(f['底背離']['獲利因子'], 2)}、"
                      f"低檔＋紅K 總獲利 {fmt(p['低檔＋紅K']['總獲利(萬)'], 0)} 萬 vs "
                      f"{fmt(f['低檔＋紅K']['總獲利(萬)'], 0)} 萬）", "弱策略對照")
    rf, rp = pick(fix, "創120日新高", "隨機"), pick(pct, "創120日新高", "隨機")
    ck.contains(text, f"（創 120 隨機基準 {fmt(rp['資金倍數_P5'], 2)}–{fmt(rp['資金倍數_P95'], 2)} vs 金額 "
                      f"{fmt(rf['資金倍數_P5'], 2)}–{fmt(rf['資金倍數_P95'], 2)}）", "資金倍數區間")
    wider = [n for n in STRONG if (pick(pct, n, "隨機")["資金倍數_P95"] - pick(pct, n, "隨機")["資金倍數_P5"])
             > (pick(fix, n, "隨機")["資金倍數_P95"] - pick(fix, n, "隨機")["資金倍數_P5"])]
    ck.check(len(wider) == 4, f"強策略比例區間都更寬：{wider}")


def main() -> int:
    ck = Checker("KD 交叉（七）")
    text = read_post("kd-cross-multi-ratio")
    ck.tables(text, tables_for(7))
    ck.contains(text, f"隨機列的資金倍數中位（p5／p95）：{random_note('比例')}。", "隨機倍數註")
    units = pd.read_csv(os.path.join(MULTI, "units.csv"), encoding="utf-8-sig")  # 份數表（整數，無精確欄）
    ck.contains(text, f"估出的「最壞連敗」（{units['S'].min()}～{units['S'].max()} 次）", "S 範圍")
    ck.contains(text, f"（{units['比例份數'].min()}～{units['比例份數'].max()} 份）", "份數範圍")
    geo = [round(1 / (1 - 0.8 ** (1 / s))) for s in units["S"]]
    ck.check(units["比例份數"].tolist() == geo, "比例份數＝round(1 ÷ (1 − 0.8^(1/S)))")
    fix, pct = load_multi("定額"), load_multi("比例")
    ordering_claims(ck, text, pct)
    compare_claims(ck, text, fix, pct)
    ck.contains(text, "2,258 檔、2002–2025", "期間")
    return ck.done()


if __name__ == "__main__":
    raise SystemExit(main())
