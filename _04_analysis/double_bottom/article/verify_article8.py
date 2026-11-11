# -*- coding: utf-8 -*-
"""
驗〈雙底（八）：挑選組合與蒙地卡羅壓測〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 5 張表必須與 build_article8_tables 由逐筆交易重算的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article8.py
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

from _04_analysis.double_bottom.combo_drawdown_filter import OUT as DRAWDOWN_FILTER  # noqa: E402
from _04_analysis.double_bottom.double_bottom_montecarlo import DRAWDOWN_RANK, OUT_DRAWDOWN, T_LOW  # noqa: E402
from build_article2_tables import PCTS, half_up  # noqa: E402
from build_article7_tables import BASE, FOCUS, TOP_MC, _one_cached, drawdown_combo, stats  # noqa: E402
from build_article8_tables import all_tables, first_combo, picked  # noqa: E402
from verify_article4 import article_path  # noqa: E402

SLUG = "double-bottom-montecarlo"


def _rng(xs, nd, unit=""):
    return f"{half_up(min(xs), nd)}{unit}～{half_up(max(xs), nd)}{unit}"


def claims(df: pd.DataFrame) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。全部只用可成交格（用戶 2026-11-11）。"""
    # 〈雙底（七）〉第 1 名的回檔
    first = first_combo()
    dd1 = drawdown_combo(*first)
    # 跌 20% 篩選與 3 格重排
    flt = pd.read_csv(DRAWDOWN_FILTER)
    flt_ok = flt.groupby(["進場", "出場"])["通過"].first()
    flt_rank = pd.read_csv(DRAWDOWN_RANK)
    flt_first = tuple(flt_rank.iloc[0][["進場", "出場"]])
    flt_base = flt[(flt["進場"] == "base") & (flt["出場"] == "base")].sort_values("門檻%")["曾跌20%"]
    by_exit = flt_ok.groupby(level="出場").sum()
    passed = [k for k, v in flt_ok.items() if v and k != BASE]
    pf_r = lambda p, e, x: half_up(_one_cached(p, True, e, x)["獲利因子"], 3)
    up = {k: sum(pf_r(p, *k) > pf_r(p, *BASE) for p in PCTS) for k in passed}
    down = {k: sum(pf_r(p, *k) < pf_r(p, *BASE) for p in PCTS) for k in passed}
    cmp = [(pf_r(p, *flt_first), pf_r(p, *BASE)) for p in PCTS]
    # 挑出的五組（6% 可成交）
    pk = picked()
    combos = [(r["進場"], r["出場"]) for _, r in pk.iterrows()]
    p, liq = FOCUS
    get = lambda e, x: df[(df["門檻%"] == p) & (df["可成交"] == liq) & (df["進場"] == e) & (df["出場"] == x)].iloc[0]
    rows = [get(*c) for c in combos]
    base = get(*BASE)
    vs = [r for (e, _), r in zip(combos, rows) if e == "volshrink"]
    fewer = [(1 - r["交易次數"] / base["交易次數"]) * 100 for r in vs]
    cell = flt[(flt["門檻%"] == p) & (flt["可成交"] == liq)].set_index(["進場", "出場"])
    dd = cell.loc[combos]
    mc = pd.read_csv(OUT_DRAWDOWN)
    med = mc["報酬% 中位_精確"]
    mc_keys = list(zip(mc["進場"], mc["出場"]))
    ev_order = sorted(combos, key=lambda c: -get(*c)["期望值/筆"])
    med_order = [k for _, k in sorted(zip(-med, mc_keys))]
    dd_mc = mc.set_index(["進場", "出場"])["最大回撤% P95"]
    entries = pd.Series([e for e, _ in combos]).value_counts()
    exits = {x for _, x in combos}
    pf_rng = _rng([r["獲利因子"] for r in rows], 3)
    med_rng = f"+{half_up(med.min(), 0)}%～+{half_up(med.max(), 0)}%"
    return {
        "第1名": (first == ("neck", "rep_mtop"), "排第 1 的頸線突破 × 替換：高檔 M 頭"),
        "回檔20": (len(dd1) == 3 and dd1.sort_values("門檻%")["曾跌20%"].is_monotonic_increasing,
                 f"曾經跌 20% 以上的有 {_rng(dd1['曾跌20%'], 1, '%')}，門檻越大越多"),
        "賺錢曾跌10": (True, f"有 {_rng(dd1['賺錢曾跌10%'], 1, '%')} 在賣出之前曾經先跌 10% 以上"),
        "曾跌10天數": (True, f"持有天數中位數是 {_rng(dd1['賺錢曾跌10%持有天中位數'], 0)} 天"),
        "回檔最多": (True, f"曾跌 20% 以上的交易最多到 {half_up(dd1['曾跌20%'].max(), 1)}%"),
        "篩選基準": (len(flt) == 148 * 3 and flt["可成交"].all(), "（" + "、".join(f"{half_up(v, 2)}%" for v in flt_base) + "）"),
        "篩選通過": (len(flt_rank) == flt_ok.sum() == 70 and len(flt_ok) == 148, "148 種組合裡有 70 種通過"),
        "篩選替換": (by_exit["rep_mtop"] == 0 and by_exit["rep_atr"] == 0 and by_exit["rep_ma10"] == 1,
                 "替換：高檔 M 頭、替換：ATR 移動停損全部沒有通過，替換：跌破 MA10 只通過 1 種"),
        "篩選排名格": ((flt_rank["cells"] == 3).all(), "再用三格的獲利因子平均名次重新排序"),
        "篩選基本第3": (tuple(flt_rank.iloc[2][["進場", "出場"]]) == BASE, "基本版本身排第 3"),
        "篩選無全高": (len(passed) == 69 and max(up.values()) < 3,
                  "69 種組合裡沒有一種三格的獲利因子都比基本版高"),
        "篩選全低": (True, f"有 {sum(v == 3 for v in down.values())} 種是三格都比基本版低"),
        "篩選第1": (flt_first == ("volshrink", "base"), "篩選後第 1 名是第二底量縮 × 基本出場"),
        "篩選第1格": ([a > b for a, b in cmp] == [False, True, True] and cmp[0][0] == cmp[0][1],
                   f"6% 與 9% 的獲利因子比基本版高，3% 取到小數三位跟基本版相同（{cmp[0][0]}）"),
        "五組含基本": (len(combos) == 5 and BASE in combos, "蒙地卡羅取篩選後排名的前五組（含基本版）"),
        "進場三種": (entries.to_dict() == {"volshrink": 3, "gap": 1, "base": 1},
                 "第二底量縮三組、兩底間隔 ≥ 10 天一組，另外一組是基本版本身"),
        "出場三種": (exits == {"base", "add_breakeven", "add_mtop"},
                 "基本出場、附加：保本停損、附加：高檔 M 頭"),
        "PF區間": (True, f"五組的獲利因子在 {pf_rng} 之間，基本版是 {half_up(base['獲利因子'], 3)}"),
        "EV區間": (True, f"每筆期望值 {_rng([r['期望值/筆'] for r in rows], 0)} 元，基本版是 "
                       f"{half_up(base['期望值/筆'], 0)} 元"),
        "交易少": (len(vs) == 3, f"交易次數比基本版少 {_rng(fewer, 0, '%')}"),
        "曾跌20": (True, f"五組曾跌 20% 以上的比例在 {_rng(dd['曾跌20%'], 2, '%')} 之間"),
        "超一年": (True, f"持有超過一年的在 {_rng(dd['超過一年%'], 1, '%')}"),
        "曾跌10": (True, f"曾跌 10% 以上的有 {_rng(dd['曾跌10%'], 1, '%')}"),
        "MC下限": ((mc["交易數"] >= T_LOW).all() and (mc["交易數"] > 20000).all(), "五組的交易數都在 2 萬筆以上"),
        "MC組合": (mc_keys == combos, "五組的破產率都是 0"),
        "MC破產": ((mc["破產%"] == 0).all(), "一萬條路徑沒有一條把本金虧掉一半"),
        "MC中位": (mc_keys[med.idxmax()] == ("volshrink", "base"),
                 f"報酬中位數在 {med_rng} 之間**，第二底量縮 × 基本出場最高，基本版是 "
                 f"+{half_up(med[mc_keys.index(BASE)], 0)}%"),
        "MC順序": (med_order == ev_order, "報酬中位數的高低順序跟每筆期望值的高低順序一樣"),
        "MC連敗": (True, f"最大連敗 P95 是 {mc['最大連敗 P95'].min()}～{mc['最大連敗 P95'].max()} 筆、回撤 P95 "
                       f"{_rng(mc['最大回撤% P95'], 1, '%')}"),
        "MC回撤": (dd_mc[combos[0]] < dd_mc[BASE] and dd_mc[combos[1]] < dd_mc[BASE]
                 and combos[0][0] == combos[1][0] == "volshrink",
                 f"第二底量縮的前兩組回撤 P95 是 {half_up(dd_mc[combos[0]], 1)}%、{half_up(dd_mc[combos[1]], 1)}%，"
                 f"比基本版的 {half_up(dd_mc[BASE], 1)}% 小"),
        "整理PF": (True, f"五組的獲利因子 {pf_rng}，跟基本版的 {half_up(base['獲利因子'], 3)} 很接近"),
        "整理MC": (True, f"報酬中位數 {med_rng}**，高低順序跟每筆期望值相同"),
        "摘要": (True, f"報酬中位數在 {med_rng} 之間，高低順序跟每筆期望值相同"),
    }


def main() -> int:
    text = io.open(article_path(SLUG), encoding="utf-8").read()
    df = stats()
    errs = []
    want = all_tables(df)
    for name, html in want.items():
        if html not in text:
            errs.append(f"表格與重算不一致：{name}")
    n_tables = len(re.findall(r'<table class="bt-table">', text))
    if n_tables != len(want):
        errs.append(f"文章表格數 {n_tables} ≠ {len(want)}")
    cl = claims(df)
    for name, (ok, phrase) in cl.items():
        if not ok:
            errs.append(f"判斷不成立：{name}")
        if phrase not in text:
            errs.append(f"文章找不到重算字串［{name}］：{phrase}")
    if "[[T:" in text:
        errs.append("文章還有沒填的佔位符")
    print(f"表格 {len(want)} 張（逐字比對 HTML）、正文主張 {len(cl)} 條")
    for e in errs:
        print("  ✗", e)
    print("錯誤 0，全部一致" if not errs else f"錯誤 {len(errs)}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
