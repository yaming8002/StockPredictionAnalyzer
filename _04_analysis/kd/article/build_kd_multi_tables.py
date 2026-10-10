# -*- coding: utf-8 -*-
"""
產生 KD 交叉（六）（七）多股兩篇、（八）結論篇的表格 HTML 列（分析段，只排版不回測）。

資料來源（SPA 回測／分析段輸出）：
  result/kd_multi/units.csv、kd_multi_result.csv   ← _03_multi_strategy/kd/kd_multi_driver.py
  result/kd_multi/kd_conclusion.csv、kd_conclusion_vs_0050.csv ← _04_analysis/kd/kd_multi_conclusion.py
kd_multi_result.csv 的「隨機」列＝1,000 次隨機買入順序中**最終權益取中位那次**的完整規格
（＝文章表註「隨機列＝1000 次隨機順序取中位那次」），資金倍數中位／P5／P95 另由 1,000 次統計。

排法與標色比照（六）（七）原文：
  結果表：策略區塊依「低價」總獲利高到低、區塊內依總獲利高到低；獲利因子與總獲利兩欄對同組隨機列
          比，贏＝粉紅(up)、輸＝淺綠(down)，隨機列本身不標色。
  （七）投法對照表：低價優先的定額／比例兩列，比例列對定額列比高低。
  （八）第一表照 kd_conclusion.csv（報酬回撤比高到低），第二表＝0050 ＋ 同六列。
多股 CSV 沒有逐筆交易可重算，顯示值落在 .5 的格子見 kd_article_common.multi_cell。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/build_kd_multi_tables.py --article 6|7|8
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.kd_article_common import (MULTI, fmt, html_row, multi_cell,  # noqa: E402
                                                       with_exact)

MODE = {6: "定額", 7: "比例"}
# 文章用名：表格區塊名（緊湊）、正文名（全稱）
SHORT = {"創250日新高": "創250新高", "創120日新高": "創120新高", "創60日新高": "創60新高",
         "跳空": "跳空", "底背離": "底背離", "低檔＋紅K": "低檔+紅K"}
LONG = {"創250日新高": "創 250 日新高", "創120日新高": "創 120 日新高", "創60日新高": "創 60 日新高",
        "跳空": "跳空", "底背離": "底背離", "低檔＋紅K": "低檔＋紅K"}
NOTE = {"創250日新高": "創250新高", "創120日新高": "創120新高", "創60日新高": "創60新高",
        "跳空": "跳空", "底背離": "底背離", "低檔＋紅K": "低檔＋紅K"}
# 份數表列序（沿用原文）
UNIT_ORDER = ["創250日新高", "創120日新高", "創60日新高", "跳空", "低檔＋紅K", "底背離"]
COLS = [("交易次數", 0), ("擋單", 0), ("勝率%", 1), ("平均持有天", 0), ("獲利平均%", 1),
        ("虧損平均%", 1), ("中位數%", 2), ("期望值/筆", 0), ("獲利因子", 2), ("總獲利(萬)", 0)]
PAINT = ("獲利因子", "總獲利(萬)")


def load_multi(mode: str) -> pd.DataFrame:
    df = with_exact(pd.read_csv(os.path.join(MULTI, "kd_multi_result.csv"), encoding="utf-8-sig"))
    return df[df["投法"] == mode].copy()


def val(r: pd.Series, col: str, nd: int) -> str:
    where = f"{r['交易策略']}·{r['投法']}·{r['排序']}"
    return fmt(multi_cell(r, col, nd, where), nd)


def block_order(df: pd.DataFrame) -> list:
    low = df[df["排序"] == "低價"].sort_values("總獲利(萬)", ascending=False, kind="mergesort")
    return low["交易策略"].tolist()


def units_rows(mode: str) -> list:
    u = pd.read_csv(os.path.join(MULTI, "units.csv"), encoding="utf-8-sig").set_index("交易策略")
    rows = []
    for name in UNIT_ORDER:
        r = u.loc[name]
        if mode == "定額":
            cells = [str(r["S"]), str(r["定額份數"]), fmt(r["定額每筆"], 0)]
        else:
            cells = [str(r["S"]), str(r["比例份數"]), f"1/{r['比例份數']}"]
        rows.append([("", LONG[name])] + [("", c) for c in cells])
    return rows


def result_rows(mode: str) -> list:
    df = load_multi(mode)
    out = []
    for name in block_order(df):
        blk = df[df["交易策略"] == name].sort_values("總獲利(萬)", ascending=False, kind="mergesort")
        rnd = blk[blk["排序"] == "隨機"].iloc[0]
        for i, (_, r) in enumerate(blk.iterrows()):
            is_rnd = r["排序"] == "隨機"
            row = [("", f"<b>{SHORT[name]}</b>" if i == 0 else ""),
                   ("", f"<b>{r['排序']}</b>" if is_rnd else r["排序"])]
            for col, nd in COLS:
                cls = ""
                if col in PAINT and not is_rnd:
                    cls = "up" if r[col] > rnd[col] else "down"
                row.append((cls, val(r, col, nd)))
            out.append(row)
    return out


def random_note(mode: str) -> str:
    """隨機列的資金倍數中位（p5／p95），依中位高到低。"""
    df = load_multi(mode)
    rnd = df[df["排序"] == "隨機"].sort_values("資金倍數_中位", ascending=False, kind="mergesort")
    return "、".join(f"{NOTE[r['交易策略']]} {fmt(r['資金倍數_中位'], 2)}（{fmt(r['資金倍數_P5'], 2)}／"
                    f"{fmt(r['資金倍數_P95'], 2)}）" for _, r in rnd.iterrows())


def compare_rows() -> list:
    """（七）固定金額 vs 固定比例（皆低價優先）；比例列對定額列標色。區塊依比例低價總獲利排。"""
    fix, pct = load_multi("定額"), load_multi("比例")
    out = []
    for name in block_order(pct):
        f = fix[(fix["交易策略"] == name) & (fix["排序"] == "低價")].iloc[0]
        p = pct[(pct["交易策略"] == name) & (pct["排序"] == "低價")].iloc[0]
        for i, (lbl, r) in enumerate((("固定金額", f), ("固定比例", p))):
            row = [("", f"<b>{SHORT[name]}</b>" if i == 0 else ""), ("", lbl),
                   ("", val(r, "交易次數", 0)), ("", val(r, "擋單", 0))]
            for col, nd in (("獲利因子", 2), ("總獲利(萬)", 0)):
                cls = "" if i == 0 else ("up" if p[col] > f[col] else "down")
                row.append((cls, val(r, col, nd)))
            out.append(row)
    return out


CONC_COLS = COLS[:]
CONC_COLS[0], CONC_COLS[1] = ("交易次數", 0), ("擋單", 0)
CONC_EXTRA = [("資金倍數", 2), ("年化報酬率%", 2), ("最大回撤%", 1), ("報酬回撤比", 2)]


def read_conc(name: str) -> pd.DataFrame:
    """結論 CSV（帶精確欄就用精確值）。"""
    return with_exact(pd.read_csv(os.path.join(MULTI, name), encoding="utf-8-sig"))


def conclusion_rows() -> tuple:
    c = read_conc("kd_conclusion.csv")
    t1 = []
    for _, r in c.iterrows():
        row = [("", LONG[r["交易策略"]]), ("", r["投入"])]
        where = f"結論·{r['交易策略']}·{r['投入']}"
        row += [("", fmt(multi_cell(r, col, nd, where), nd)) for col, nd in CONC_COLS + CONC_EXTRA]
        t1.append(row)
    vs = read_conc("kd_conclusion_vs_0050.csv")
    t2 = []
    for i, r in vs.iterrows():
        name = r["做法"]
        for k, v in LONG.items():
            name = name.replace(k, v)
        cells = [name] + [fmt(r[col], nd) for col, nd in CONC_EXTRA]
        t2.append([("", f"<b>{x}</b>" if i == 0 else x) for x in cells])
    return t1, t2


def tables_for(article: int) -> list:
    if article in MODE:
        mode = MODE[article]
        out = [(f"（{'六七'[article - 6]}）份數表", units_rows(mode)),
               (f"（{'六七'[article - 6]}）結果表", result_rows(mode))]
        if article == 7:
            out.append(("（七）投法對照表", compare_rows()))
        return out
    t1, t2 = conclusion_rows()
    return [("（八）六種策略", t1), ("（八）對 0050", t2)]


def main() -> int:
    ap = argparse.ArgumentParser(description="KD（六）（七）（八）表格 HTML")
    ap.add_argument("--article", type=int, required=True, choices=[6, 7, 8])
    a = ap.parse_args()
    for name, rows in tables_for(a.article):
        print(f"\n=== {name} ===")
        print("\n".join(html_row(r) for r in rows))
    if a.article in MODE:
        print("\n=== 隨機列資金倍數中位（p5／p95）===")
        print(random_note(MODE[a.article]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
