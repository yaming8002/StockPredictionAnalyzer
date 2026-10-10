# -*- coding: utf-8 -*-
"""
均線交叉（五）（六）正文統計句的共用計算（被 verify_article5.py／verify_article6.py 呼叫）。

每個計數、比較都由多股結果重算，再組成文章裡應該出現的那段字去比對；
文章或數據任一邊變了都會報錯。「中位」一律用隨機 1,000 次總獲利的中位數（分批對照表那一欄），
排序比較一律用排序對照表的值（隨機＝最終權益居中那一次）。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from build_multi_tables import ORDERS, sizing_names, tot_wan  # noqa: E402
from article_common import PAIRS, half_up, mc, mrow  # noqa: E402


def sizing_stats(df, mode: str) -> dict:
    """分批對照：等分／公式的破產組、中位比較。"""
    formula, even = sizing_names(mode)
    E = {p: mrow(df, p, even, "隨機") for p in PAIRS}
    F = {p: mrow(df, p, formula, "隨機") for p in PAIRS}
    med = lambda r: half_up(r["總獲利(萬)_中位"], 0)  # noqa: E731
    e_hi = [p for p in PAIRS if med(E[p]) > med(F[p])]
    e_ruin = [p for p in PAIRS if E[p]["本金曾<80%_比例%"] > 0]
    e_zero = [p for p in PAIRS if E[p]["本金曾<80%_比例%"] == 0]
    return {"E": E, "F": F, "med": med, "e_hi": e_hi, "e_ruin": e_ruin,
            "e_ruin90": [p for p in PAIRS if E[p]["本金曾<80%_比例%"] >= 90],
            "e_zero": e_zero, "e_zero_hi": [p for p in e_zero if p in e_hi],
            "f_ruin": [p for p in PAIRS if F[p]["本金曾<80%_比例%"] > 0]}


def order_stats(df, mode: str) -> dict:
    """排序對照：四排序一樣的組、最大差距、各排序嚴格最高的組數。"""
    formula = sizing_names(mode)[0]
    tot = {p: {o: tot_wan(mrow(df, p, formula, o)) for o in ORDERS} for p in PAIRS}
    shown = {p: {o: half_up(v, 0) for o, v in tot[p].items()} for p in PAIRS}
    ev = {p: {o: half_up(mrow(df, p, formula, o)["期望值/筆"], 0) for o in ORDERS} for p in PAIRS}
    same = [p for p in PAIRS if len(set(shown[p].values())) == 1 and len(set(ev[p].values())) == 1]
    spread = {p: max(tot[p].values()) - min(tot[p].values()) for p in PAIRS}
    top = {o: [] for o in ORDERS}
    # 只算「顯示上看得出差別」的組：四排序顯示值都一樣的組，差在小數點後，讀者看不到
    for p in PAIRS:
        if p in same:
            continue
        vals = sorted(tot[p].values(), reverse=True)
        if vals[0] > vals[1]:
            top[max(tot[p], key=tot[p].get)].append(p)
    colored = sum((shown[p][o] != shown[p]["隨機"]) + (ev[p][o] != ev[p]["隨機"])
                  for p in PAIRS for o in ORDERS[:3])
    blk = {p: int(mrow(df, p, formula, "隨機")["擋單"]) for p in PAIRS}
    return {"tot": tot, "shown": shown, "same": same, "spread": spread, "top": top,
            "colored": colored, "blk": blk}


def single_ev(pair: str) -> float:
    return float(mc().loc[pair, "每筆淨期望%"])
