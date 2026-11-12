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
from _04_analysis.double_bottom.double_bottom_montecarlo import OUT_DRAWDOWN, T_LOW  # noqa: E402
from build_article2_tables import half_up  # noqa: E402
from build_article7_tables import BASE, FOCUS, TOP_MC, drawdown_cell, stats  # noqa: E402
from build_article8_tables import all_tables, cells, picked, seventh_top  # noqa: E402
from verify_article4 import article_path  # noqa: E402

SLUG = "double-bottom-montecarlo"


def _rng(xs, nd, unit=""):
    lo, hi = half_up(min(xs), nd), half_up(max(xs), nd)
    return f"{lo}{unit}" if lo == hi else f"{lo}{unit}～{hi}{unit}"


def claims(df: pd.DataFrame) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。全部只用可成交格（用戶 2026-11-11）。"""
    # 〈雙底（七）〉前五名，6% 回檔
    p6, _ = FOCUS
    top7 = seventh_top()
    dd7 = pd.DataFrame([drawdown_cell(p6, r["進場"], r["出場"]) for _, r in top7.iterrows()])
    flt = pd.read_csv(DRAWDOWN_FILTER)
    base20 = flt[(flt["進場"] == "base") & (flt["出場"] == "base")].set_index("門檻%")["曾跌20%"]
    # 逐格篩選與排名
    c = cells()
    by_p = flt.groupby("門檻%")["本格通過"].sum()
    first_non9 = c[c["門檻%"] != 9]["名次"].min()
    short = c[~c["夠筆數"]]
    pk = picked()
    keys = [(int(r["門檻%"]), r["進場"], r["出場"]) for _, r in pk.iterrows()]
    get = lambda p, e, x: df[(df["門檻%"] == p) & df["可成交"] & (df["進場"] == e) & (df["出場"] == x)].iloc[0]
    rows = [get(*k) for k in keys]
    base9 = get(9, *BASE)
    mc = pd.read_csv(OUT_DRAWDOWN)
    med = mc["報酬% 中位_精確"]
    mc_keys = list(zip(mc["門檻%"], mc["進場"], mc["出場"]))
    ev_order = sorted(keys, key=lambda k: -get(*k)["期望值/筆"])
    med_order = [k for _, k in sorted(zip(-med, mc_keys))]
    entries = pd.Series([e for _, e, _ in keys]).value_counts()
    exits = {x for _, _, x in keys}
    pf_rng = _rng([r["獲利因子"] for r in rows], 3)
    med_rng = f"+{half_up(med.min(), 0)}%～+{half_up(med.max(), 0)}%"
    dd20_7 = _rng(dd7["曾跌20%"], 1, "%")
    # 報酬中位數跟基本版比（算法同〈MACD（十）〉：中位數 ÷ 基本版中位數 − 1）
    base_med = med[mc_keys.index((9, *BASE))]
    rel = [(m / base_med - 1) * 100 for m in med]
    lo, hi = half_up(min(rel), 0), half_up(max(rel), 0)
    assert lo < 0 < hi, "報酬中位數跟基本版比的字串假設一負一正"
    gap = f"少 {-lo}% 到多 {hi}%"
    base_rank = int(c[(c["門檻%"] == 9) & (c["進場"] == "base") & (c["出場"] == "base")]["名次"].iloc[0])
    return {
        "七前五出場": ((top7["出場"] == "rep_mtop").all(), "排名前五的組合，出場都是替換：高檔 M 頭"),
        "七前五曾跌20": (True, f"五組裡曾經跌 20% 以上的有 {dd20_7}，同一格的基本版是 {half_up(base20[p6], 2)}%"),
        "七前五賺錢曾跌10": (True, f"有 {_rng(dd7['賺錢曾跌10%'], 1, '%')} 在賣出之前曾經先跌 10% 以上"),
        "七前五天數": (True, f"持有天數中位數是 {_rng(dd7['賺錢曾跌10%持有天中位數'], 0)} 天"),
        "篩選基準": (len(flt) == 444 and flt["可成交"].all(), "（3% 是 " + f"{half_up(base20[3], 2)}%、6% 是 "
                 f"{half_up(base20[6], 2)}%、9% 是 {half_up(base20[9], 2)}%）"),
        "篩選通過": (len(c) == flt["本格通過"].sum(), f"444 種裡有 {len(c)} 種通過，3%、6%、9% 各 "
                 f"{by_p[3]}、{by_p[6]}、{by_p[9]} 種"),
        "替換M頭": (flt[flt["出場"] == "rep_mtop"]["本格通過"].sum() == 0, "替換：高檔 M 頭在三個門檻都沒有通過"),
        "前N都9": (True, f"前 {first_non9 - 1} 名都是 ZigZag 9%"),
        "6最前": ((c[c["名次"] == first_non9]["門檻%"] == 6).all(), f"6% 最前面的排第 {first_non9}"),
        "跳過": (list(short["名次"].head(2)) == [1, 2] and (short.head(2)["進場"] == "volshrink").all()
                 and short.head(2)["交易次數"].max() < 14000 and short.head(2)["交易次數"].min() > 13000
                 and list(pk["名次"]) == [3, 4, 5, 6, 7], "蒙地卡羅取的是第 3～7 名"),
        "下限": (T_LOW == 17847, "抽樣下限 17,847 筆"),
        "基本第7": (base_rank == 7 and (9, *BASE) in keys, "9% 的基本版本身排第 7，也在選入的五組裡"),
        "全9": (all(k[0] == 9 for k in keys) and len(keys) == TOP_MC and c[c["門檻%"] == 3]["名次"].min() > first_non9,
                "所以挑出來的五組都是 ZigZag 9%"),
        "PF區間": (True, f"五組的獲利因子在 {pf_rng} 之間，基本版是 {half_up(base9['獲利因子'], 3)}"),
        "EV區間": (True, f"每筆期望值 {_rng([r['期望值/筆'] for r in rows], 0)} 元，基本版是 "
                       f"{half_up(base9['期望值/筆'], 0)} 元"),
        "進場三種": (entries.to_dict() == {"rise": 3, "height": 1, "base": 1},
                 "第二底墊高 ≥ 2% 三組、型態高度 ≥ 10% 一組，另外一組是基本版本身"),
        "出場三種": (exits == {"base", "add_breakeven", "add_mtop"},
                 "基本出場、附加：保本停損、附加：高檔 M 頭"),
        "曾跌20": (True, f"五組曾跌 20% 以上的比例在 {_rng(pk['曾跌20%'], 2, '%')} 之間"),
        "超一年": (True, f"持有超過一年的在 {_rng(pk['超過一年%'], 1, '%')}"),
        "曾跌10": (True, f"曾跌 10% 以上的有 {_rng(pk['曾跌10%'], 1, '%')}"),
        "MC下限": ((mc["交易數"] >= T_LOW).all() and (mc["交易數"] > 20000).all(), "五組的交易數都在 2 萬筆以上"),
        "MC組合": (mc_keys == keys, "五組的破產率都是 0"),
        "MC破產": ((mc["破產%"] == 0).all(), "一萬條路徑沒有一條把本金虧掉一半"),
        "MC中位": (mc_keys[med.idxmax()] == (9, "rise", "base"),
                 f"報酬中位數在 {med_rng} 之間**，第二底墊高 ≥ 2% × 基本出場最高，基本版是 "
                 f"+{half_up(med[mc_keys.index((9, *BASE))], 0)}%"),
        "MC順序": (med_order == ev_order, "報酬中位數的高低順序跟每筆期望值的高低順序一樣"),
        "MC連敗": (True, f"最大連敗 P95 是 {mc['最大連敗 P95'].min()}～{mc['最大連敗 P95'].max()} 筆、回撤 P95 "
                       f"{_rng(mc['最大回撤% P95'], 1, '%')}"),
        "整理七": (True, f"以 6% 為例，持有期間曾跌 20% 以上的有 {dd20_7}"),
        "整理篩選": (True, f"加上持有期間跌 20% 的門檻後，444 種剩 {len(c)} 種"),
        "整理前N": (True, f"前 {first_non9 - 1} 名都是 ZigZag 9%"),
        "整理PF": (True, f"獲利因子 {pf_rng}，基本版是 {half_up(base9['獲利因子'], 3)}；"
                       f"報酬中位數比基本版{gap}"),
        "整理MC": (True, f"回撤 P95 {_rng(mc['最大回撤% P95'], 1, '%')}、最大連敗 P95 "
                       f"{mc['最大連敗 P95'].min()}～{mc['最大連敗 P95'].max()} 筆"),
        "摘要七": (True, f"在 ZigZag 6% 有 {dd20_7} 的交易在持有期間曾跌 20% 以上"),
        "摘要篩選": (True, f"剩 {len(c)} 種；依獲利因子排序"),
        "摘要MC": (True, f"報酬中位數在 {med_rng} 之間"),
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
