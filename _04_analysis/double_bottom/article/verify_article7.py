# -*- coding: utf-8 -*-
"""
驗〈雙底（七）：拼裝整合〉的表格與正文數字，全部由資料重算後比對；錯誤 0 才算過。

1. 表格：文章裡的 6 張表必須與 build_article7_tables 由逐筆交易重算的 HTML 逐字相同。
2. 正文主張：每條數字敘述在這裡重算，判斷式必須成立，格式化後的字串必須逐字出現在文章裡。

執行：
    BLOG_DIR=F:/stock-analyzer/blog PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python _04_analysis/double_bottom/article/verify_article7.py
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

from _04_analysis.double_bottom.combo_ranking import load_summary, rank_combos  # noqa: E402
from build_article2_tables import PCTS, half_up  # noqa: E402
from build_article7_tables import BASE, FOCUS, TOP_DETAIL, _one_cached, all_tables, combo_label, stats  # noqa: E402
from verify_article4 import article_path  # noqa: E402

SLUG = "double-bottom-combo"


def _rng(xs, nd, unit=""):
    return f"{half_up(min(xs), nd)}{unit}～{half_up(max(xs), nd)}{unit}"


def _span(xs, nd, unit=""):
    lo, hi = half_up(min(xs), nd), half_up(max(xs), nd)
    return f"{lo}{unit}" if lo == hi else f"{lo}{unit}～{hi}{unit}"


def detail_claim(df: pd.DataFrame, rank: pd.DataFrame, n: int, open_rate) -> tuple:
    """第 n 名（2～5）那一節的說明：五欄都比基本版高、持有變長、虧損變深；字串照文章原文組。"""
    e, x = rank.iloc[n - 1]["進場"], rank.iloc[n - 1]["出場"]
    get = lambda p, k: df[(df["門檻%"] == p) & df["可成交"] & (df["進場"] == k[0]) & (df["出場"] == k[1])].iloc[0]
    v = {p: get(p, (e, x)) for p in PCTS}
    b = {p: get(p, BASE) for p in PCTS}
    paint = ("勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子")
    ok = (all(half_up(v[p][c], 3) > half_up(b[p][c], 3) for p in PCTS for c in paint)
          and all(v[p]["平均持有天"] > b[p]["平均持有天"] and v[p]["虧損平均%"] < b[p]["虧損平均%"] for p in PCTS))
    hold = [v[p]["平均持有天"] / b[p]["平均持有天"] for p in PCTS]
    n_ratio = [v[p]["交易次數"] / b[p]["交易次數"] * 100 for p in PCTS]
    phrase = (f"三格的勝率、獲利平均、中位數、期望值、獲利因子都比基本版高。平均持有天數是基本版的 {_span(hold, 1)} 倍"
              f"（9% 從 {half_up(b[9]['平均持有天'], 0)} 天拉長到 {half_up(v[9]['平均持有天'], 0)} 天），虧損平均也比較深"
              f"（9% 從 {half_up(b[9]['虧損平均%'], 1)}% 變成 {half_up(v[9]['虧損平均%'], 1)}%）。交易次數是基本版的 "
              f"{_span(n_ratio, 0, '%')}，期末未賣出比例在 {_span(open_rate(e, x), 1, '%')}。").replace("-", "−")
    return ok, phrase


def claims(df: pd.DataFrame) -> dict:
    """key → (判斷式是否成立, 文章裡的字串)。只用可成交 3 格（用戶 2026-11-11 拿掉裸版）。"""
    summ = load_summary()
    rank = rank_combos(summ)
    first = tuple(rank.iloc[0][["進場", "出場"]])
    # 比基本版高／低：用逐筆重算的 PF 取到小數三位比（與表格底色同一口徑）
    pf_r = {(p, e, x): half_up(_one_cached(p, True, e, x)["獲利因子"], 3)
            for e, x in zip(rank["進場"], rank["出場"]) for p in PCTS}
    keys = list(zip(rank["進場"], rank["出場"]))
    up = pd.Series({k: sum(pf_r[(p, *k)] > pf_r[(p, *BASE)] for p in PCTS) for k in keys})
    down = pd.Series({k: sum(pf_r[(p, *k)] < pf_r[(p, *BASE)] for p in PCTS) for k in keys})
    others = up.drop(index=BASE)
    all_up = others[others == 3]
    mtop = rank[rank["出場"] == "rep_mtop"].set_index("進場")["名次"]
    rk = lambda x: rank[rank["出場"] == x]["名次"]
    get = lambda p, e, x: df[(df["門檻%"] == p) & df["可成交"] & (df["進場"] == e) & (df["出場"] == x)].iloc[0]
    three = lambda e, x, c: [get(p, e, x)[c] for p in PCTS]
    ratio = [a / b for a, b in zip(three(*first, "平均持有天"), three(*BASE, "平均持有天"))]
    open_rate = lambda e, x: summ[(summ["進場"] == e) & (summ["出場"] == x) & summ["可成交"]]["未平倉率%"]
    mtop_focus = [get(FOCUS[0], e, "rep_mtop")["獲利因子"] for e in mtop.index]
    in_band = [v for v in mtop_focus if 1.30 <= float(half_up(v, 3)) <= 1.35]
    pair = lambda x: summ[(summ["進場"] == "neck") & (summ["出場"] == x) & summ["可成交"]]["獲利因子(PF)"].tolist()
    neck_same = pair("add_mtop") == pair("base")
    neck_rank = sorted(rank[(rank["進場"] == "neck") & rank["出場"].isin(["add_mtop", "base"])]["名次"])
    up_extra = set(all_up.index) - {(e, "rep_mtop") for e in mtop.index}
    return {
        "組合數": (len(summ) == 924 and len(rank) == 148 and summ["not_applicable"].sum() == 36
                and summ["可成交"].sum() == 462, "共 462 次回測"),
        "可排名": (len(rank) == 148, "剩下 148 種"),
        "前十名": ((rank.head(10)["出場"] == "rep_mtop").all(), "前十名的出場都是「替換：高檔 M 頭」"),
        "型態第44": (len(mtop) == 11 and sorted(mtop)[:10] == list(range(1, 11)) and mtop["height"] == 44,
                 "有十種排在前十名，型態高度排第 44"),
        "第1最差": (rank.iloc[0]["pf_rank_worst"] == 2 and first == ("neck", "rep_mtop"), "三格裡最差也是第 2 名"),
        "其他最差": (rank.iloc[1:]["pf_rank_worst"].min() == 7, "其他組合的最差名次都在第 7 名以後"),
        "頸線同分": (neck_same and neck_rank == [11, 12], "排第 11、12 名的頸線突破 × 附加：高檔 M 頭與頸線突破 × 基本出場，數字完全一樣"),
        "ATR名次": (True, f"替換：ATR 移動停損落在第 {rk('rep_atr').min()}～{rk('rep_atr').max()} 名"),
        "MA10名次": (True, f"替換：跌破 MA10 落在第 {rk('rep_ma10').min()}～{rk('rep_ma10').max()} 名"),
        "三格全高": (len(others) == 147 and len(all_up) == 13,
                 f"147 種組合裡只有 {len(all_up)} 種三格的獲利因子都比基本版高"),
        "三格全高明細": (len(all_up) - len(up_extra) == 10 and up_extra == {("neck", "base"), ("neck", "add_mtop"), ("neck", "add_trail20")},
                   "10 種是替換：高檔 M 頭，另外 3 種是頸線突破 × 基本出場、頸線突破 × 附加：高檔 M 頭、頸線突破 × 附加：自高點回落 20%"),
        "三格全低": (True, f"有 {(down.drop(index=BASE) == 3).sum()} 種是三格都比基本版低"),
        "持有倍數": (half_up(min(ratio), 1) == half_up(max(ratio), 1), f"平均持有天數約是基本版的 {half_up(ratio[0], 1)} 倍"),
        "9%持有": (True, f"9% 從 {half_up(get(9, *BASE)['平均持有天'], 0)} 天拉長到 "
                       f"{half_up(get(9, *first)['平均持有天'], 0)} 天"),
        "9%虧損": (True, f"9% 從 {half_up(get(9, *BASE)['虧損平均%'], 1)}% 變成 "
                       f"{half_up(get(9, *first)['虧損平均%'], 1)}%".replace("-", "−")),
        "未平倉": (True, f"期末未賣出比例在三格裡是 {_rng(open_rate(*first), 1, '%')}，基本版是 {_rng(open_rate(*BASE), 1, '%')}"),
        **{f"第{n}名細節": detail_claim(df, rank, n, open_rate) for n in range(2, TOP_DETAIL + 1)},
        "整理區間": (len(in_band) == 9, f"{len(in_band)} 種落在 1.30～1.35 之間"),
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
