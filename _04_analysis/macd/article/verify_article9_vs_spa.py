# -*- coding: utf-8 -*-
"""
第九篇（替換版全表）能不能用 SPA 的程式重現？

第九篇的數字當年是用 blog 底下的獨立 driver（不依賴 SPA）跑出來的，那批 driver 已於
2026-09-30 依「公開回測程式一律住 SPA」的規則從 blog 移除（tracked 的可用
`git show HEAD:blog/tools/macd_backtest.py` 取回）。**所以這支的任務是補上可重現性**：
證明公開 repo 跑得出同一組已發佈數字，不然文章的數字就沒有公開來源可查。

比對對象：
  文章 = `blog/site/content/posts/macd-exit-pure.md` 表二的「替換出場規則」與「未平倉」兩欄
  SPA  = `_02_strategy/macd_strategy/macd_exit_replace.py` 的輸出 CSV

容許值：獲利因子 0.005、未平倉率 0.1 個百分點。兩邊的下單股數算法有極小差異
（見 SPA `verify_multi_macd.py` 的說明），不會逐位元相同；但若時點或規則寫錯，
差距會是 0.1 以上、不會卡在容許值內。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _04_analysis/macd/article/verify_article9_vs_spa.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402

import pandas as pd

CSV = os.path.join(common.result_dir("macd_strategy", "macd_exit_replace"),
                   "macd_exit_replace.csv")
TOL_PF, TOL_UNCLOSED = 0.005, 0.1

# 文章表二（替換出場規則欄、未平倉欄）；(母體, 出場) -> (獲利因子, 未平倉%)
WANT = {
    ("黃金交叉", "跌破年線"): (1.5088, 2.57),
    ("黃金交叉", "超級趨勢翻空"): (1.2217, 1.57),
    ("黃金交叉", "SAR翻空"): (0.9882, 0.38),
    ("黃金交叉", "波段高點走低"): (1.1175, 0.69),
    ("黃金交叉", "跌破二十日低"): (1.2118, 1.14),
    ("黃金交叉", "吊燈3ATR"): (1.0849, 0.62),
    ("黃金交叉", "自最高點回落10%"): (1.0982, 0.89),
    ("黃金交叉", "抱滿60天"): (1.1885, 1.73),
    ("零軸上穿", "跌破年線"): (1.4863, 2.70),
    ("零軸上穿", "超級趨勢翻空"): (1.2110, 1.28),
    ("零軸上穿", "SAR翻空"): (0.9791, 0.40),
    ("零軸上穿", "波段高點走低"): (1.1748, 0.86),
    ("零軸上穿", "跌破二十日低"): (1.2550, 1.35),
    ("零軸上穿", "吊燈3ATR"): (1.1418, 0.70),
    ("零軸上穿", "自最高點回落10%"): (1.1613, 1.03),
    ("零軸上穿", "抱滿60天"): (1.1730, 1.76),
    ("純背離", "跌破年線"): (1.4069, 3.82),
    ("純背離", "超級趨勢翻空"): (1.1969, 2.38),
    ("純背離", "SAR翻空"): (0.9685, 0.68),
    ("純背離", "波段高點走低"): (1.0361, 0.88),
    ("純背離", "跌破二十日低"): (1.0008, 1.41),
    ("純背離", "吊燈3ATR"): (0.9483, 0.64),
    ("純背離", "自最高點回落10%"): (0.9764, 1.02),
    ("純背離", "抱滿60天"): (1.1394, 1.76),
}


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"找不到 SPA 的輸出：{CSV}\n先跑 _02_strategy/macd_strategy/macd_exit_replace.py")
    df = pd.read_csv(CSV)
    got = {(r["母體"], r["出場"]): (float(r["獲利因子"]), float(r["未平倉%"]))
           for _, r in df.iterrows()}

    bad = 0
    for key, (pf_want, un_want) in WANT.items():
        if key not in got:
            print(f"[缺] {key[0]} × {key[1]}：SPA 的表裡沒有這一格")
            bad += 1
            continue
        pf, un = got[key]
        dp, du = abs(pf - pf_want), abs(un - un_want)
        ok = dp <= TOL_PF and du <= TOL_UNCLOSED
        if not ok:
            print(f"[NG] {key[0]} × {key[1]}：獲利因子 {pf:.4f} vs 文章 {pf_want}"
                  f"（差 {dp:.4f}）｜未平倉 {un:.2f}% vs {un_want}%（差 {du:.2f}）")
            bad += 1
    print(f"\n比對 {len(WANT)} 格，不符 {bad} 格"
          f"（容許：獲利因子 {TOL_PF}、未平倉 {TOL_UNCLOSED} 個百分點）")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
