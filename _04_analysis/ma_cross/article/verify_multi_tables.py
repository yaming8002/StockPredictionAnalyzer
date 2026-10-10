# -*- coding: utf-8 -*-
"""
均線交叉（五）（六）多股兩篇共用的表格驗證（被 verify_article5.py／verify_article6.py 呼叫）。

兩篇結構相同、各 7 張表（依出現順序）：
  0 份數表        S（單股蒙地卡羅最大連敗 P95）、份數、每筆投入
  1 分批對照表    等分 20 份 vs 公式，各自隨機 ×1,000：擋單、本金<80%、總獲利 P5／中位／P95、中位差
  2 排序對照表    公式分批下四排序的總獲利（萬）與期望值/筆（元），以隨機為基準上色
  3~6 附錄        流動性／低價／高價／隨機 四張完整表（10 欄＋擋單）
「隨機」列一律是 driver 的代表列＝1,000 次裡最終權益中位數那一次。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from build_multi_tables import INIT_CASH, ORDERS, sizing_names, tot_wan  # noqa: E402
from article_common import (PAIRS, Checker, body_rows, half_up, headers, mc, mrow,  # noqa: E402
                            multi_key, same, tables)

FLOOR = 0.80


def units(s: int):
    """S → (定額份數, 比例份數)；與 ma_cross_multi_driver.units 同公式（這裡重算一次當獨立驗證）。"""
    return int(round(s / (1.0 - FLOOR))), int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))


def shown(cell: str):
    """顯示值（Decimal），用於『顯示值相減／相比』的檢查。"""
    from article_common import num
    from decimal import Decimal
    return Decimal(repr(num(cell)))


def verify_units(ck: Checker, table: str, df, mode: str) -> None:
    s = mc()["最大連敗_P95"]
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, "份數表列順序／組數不符")
    for row in rows:
        p = row[0][1]
        r = mrow(df, p, sizing_names(mode)[0], "流動性")
        want_n = units(int(s[p]))[0 if mode == "定額" else 1]
        ck.check(int(row[1][1]) == int(s[p]), f"份數表 {p} S：文 {row[1][1]} vs {s[p]}")
        ck.check(int(row[2][1]) == int(r["份數"]) == want_n, f"份數表 {p} 份數：文 {row[2][1]} vs {r['份數']}/{want_n}")
        ck.check(same(row[3][1], INIT_CASH / r["份數"]), f"份數表 {p} 每筆：文 {row[3][1]}")


def verify_sizing(ck: Checker, table: str, df, mode: str) -> None:
    formula, even = sizing_names(mode)
    n_all = mc()["交易數"]
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, "分批對照表列順序／組數不符")
    for row in rows:
        p = row[0][1]
        c = [x for _, x in row]
        cl = [x for x, _ in row]
        ck.check(same(c[1], n_all[p]), f"分批表 {p} 總交易次數：文 {c[1]} vs {n_all[p]}")
        e, f = mrow(df, p, even, "隨機"), mrow(df, p, formula, "隨機")
        # 等分：擋單、本金<80%、P5、中位、P95；公式：份數、擋單、本金<80%、P5、中位、P95
        blocks = [(e, 2, None), (f, 8, 7)]
        for r, i0, i_units in blocks:
            tag = "等分" if r is e else "公式"
            if i_units is not None:
                ck.check(int(c[i_units]) == int(r["份數"]), f"分批表 {p} {tag} 份數 {c[i_units]}")
            ck.check(same(c[i0], r["擋單"]), f"分批表 {p} {tag} 擋單：文 {c[i0]} vs {r['擋單']}")
            risk = r["本金曾<80%_比例%"]
            ck.check(same(c[i0 + 1], risk), f"分批表 {p} {tag} 本金<80%：文 {c[i0 + 1]} vs {risk}")
            ck.check(("risk" in cl[i0 + 1]) == (risk > 0), f"分批表 {p} {tag} 本金<80% 底色")
            for k, col in enumerate(("總獲利(萬)_P5", "總獲利(萬)_中位", "總獲利(萬)_P95")):
                ck.check(same(c[i0 + 2 + k], r[col]), f"分批表 {p} {tag} {col}：文 {c[i0 + 2 + k]} vs {r[col]}")
        diff = shown(c[11]) - shown(c[5])
        ck.check(shown(c[13]) == diff, f"分批表 {p} 中位差：文 {c[13]} vs 顯示值相減 {diff}")
        ck.check(cl[13].split()[-1] == ("up" if diff > 0 else "down"), f"分批表 {p} 中位差底色 {cl[13]}")


def verify_orders(ck: Checker, table: str, df, mode: str) -> None:
    formula = sizing_names(mode)[0]
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, "排序對照表列順序／組數不符")
    for row in rows:
        p = row[0][1]
        rr = {o: mrow(df, p, formula, o) for o in ORDERS}
        ck.check(same(row[1][1], rr["隨機"]["擋單"]), f"排序表 {p} 擋單：文 {row[1][1]}")
        for j, getter in enumerate((tot_wan, lambda r: float(r["期望值/筆"]))):
            base = half_up(getter(rr["隨機"]), 0)
            for k, o in enumerate(ORDERS):
                cls, cell = row[2 + 4 * j + k]
                v = getter(rr[o])
                ck.check(same(cell, v), f"排序表 {p} {o} {'總獲利' if j == 0 else '期望值'}：文 {cell} vs {v}")
                d = half_up(v, 0)
                want = "" if o == "隨機" or d == base else ("up" if d > base else "down")
                ck.check(cls == want, f"排序表 {p} {o} 底色：文 {cls or '無'} vs 應 {want or '無'}")


def verify_appendix(ck: Checker, table: str, df, mode: str, order: str) -> None:
    keys = ["n", "wr", "hold", "aw", "al", "med", "ev", "pf", "tot", "blk"]
    ck.check(headers(table)[1:] == ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%",
                                    "期望值/筆(元)", "獲利因子", "總獲利(萬)", "擋單"], f"附錄 {order} 表頭不符")
    rows = body_rows(table)
    ck.check([r[0][1] for r in rows] == PAIRS, f"附錄 {order} 列順序／組數不符")
    for row in rows:
        p = row[0][1]
        r = mrow(df, p, sizing_names(mode)[0], order)
        m = multi_key(r)
        m["tot"] = tot_wan(r)
        for key, (_, cell) in zip(keys, row[1:]):
            ck.check(same(cell, m[key]), f"附錄 {order} {p} {key}：文 {cell} vs {m[key]}")


def verify_tables(ck: Checker, text: str, df, mode: str) -> list:
    tb = tables(text)
    if not ck.check(len(tb) == 7, f"應有 7 張表，實際 {len(tb)}"):
        return tb
    verify_units(ck, tb[0], df, mode)
    verify_sizing(ck, tb[1], df, mode)
    verify_orders(ck, tb[2], df, mode)
    for t, o in zip(tb[3:], ORDERS):
        verify_appendix(ck, t, df, mode, o)
    return tb
