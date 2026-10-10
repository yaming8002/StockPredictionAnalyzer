# -*- coding: utf-8 -*-
"""
產生 KD 交叉（一）～（五）單股篇所有結果表的 HTML 列（分析段，只排版不回測）。

數字從 kd_sweep.py 存的逐筆交易重算精確值再四捨五入（見 kd_article_common 說明），
底色規則照各篇表註：
  （一）（二）：數字本身正＝紅字(pos)、負＝綠字(neg)。
  （三）（四）：勝率／獲利平均／中位數／期望值／獲利因子對「基本的 KD 交叉」比高低，
              好＝粉紅(up)、差＝淺綠(down)、一樣＝不上色；比較用兩位小數（獲利因子四位），
              所以顯示一位時看起來相同（如 −1.1 對 −1.1）的格子仍可能上色。
  （五）：矩陣／附錄的獲利因子 > 1 上粉紅；「高檔死叉」完整表對同一進場配一般死叉比；
         蒙地卡羅表報酬類 > 0 上粉紅。
每篇的表依文章出現順序排列，驗證腳本（verify_kd_article<N>.py）直接拿這裡的輸出逐格比對。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/kd/article/build_kd_single_tables.py --article 1|2|3|4|5
"""
import argparse
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _04_analysis.kd.article.kd_article_common import (MC_DIR, SPEC, fmt, half_up,  # noqa: E402
                                                       html_row, single_metrics)

BASE = "amt__golden__death"                 # 基本的 KD 交叉（含 1,000 萬門檻）
SIGNED = {"中位數%", "期望值/筆", "總獲利(萬)"}
POSNEG = ["獲利平均%", "虧損平均%", "中位數%", "期望值/筆", "總獲利(萬)"]
UPDOWN = ["勝率%", "獲利平均%", "中位數%", "期望值/筆", "獲利因子"]
_ND = {"交易次數": 0, "勝率%": 1, "平均持有天": 0, "獲利平均%": 1, "虧損平均%": 1,
       "中位數%": 1, "期望值/筆": 0, "總獲利(萬)": 0}


def cell(m: dict, col: str, pf_nd: int) -> str:
    nd = pf_nd if col == "獲利因子" else _ND[col]
    return fmt(m[col], nd, sign=col in SIGNED)


def posneg(m: dict, col: str) -> str:
    """（一）（二）：顯示值正＝pos、負＝neg。"""
    if col not in POSNEG:
        return ""
    q = half_up(m[col], 0 if col in ("期望值/筆", "總獲利(萬)") else 1)
    return "pos" if q > 0 else ("neg" if q < 0 else "")


def updown(m: dict, base: dict, col: str, cols=UPDOWN) -> str:
    """（三）（四）：對基本版比高低（兩位小數；獲利因子四位），相同不上色。"""
    if base is None or col not in cols:
        return ""
    nd = 4 if col == "獲利因子" else 2
    a, b = half_up(m[col], nd), half_up(base[col], nd)
    return "up" if a > b else ("down" if a < b else "")


def spec_row(label: str, variant: str, pf_nd: int, mode: str, base=None, cols=SPEC,
             paint=UPDOWN) -> list:
    m = single_metrics(variant)
    out = [("", label)]
    for c in cols:
        cls = posneg(m, c) if mode == "posneg" else updown(m, base, c, paint)
        out.append((cls, cell(m, c, pf_nd)))
    return out


# ── （一）最基本的 KD 交叉 ──────────────────────────────────────────────────────
def article1() -> list:
    raw, amt = "raw__golden__death", BASE
    return [("（一）表1 全市場", [spec_row("KD 黃金／死亡交叉", raw, 2, "posneg")]),
            ("（一）表2 加流動性門檻", [
                spec_row("KD 交叉<br>（無門檻）", raw, 2, "posneg"),
                spec_row("KD 交叉 ＋ 成交金額 > 1,000 萬", amt, 2, "posneg")])]


# ── （二）只在超賣區買、超買區賣 ────────────────────────────────────────────────
ZONE = "amt__low_zone__high_death"


def article2() -> list:
    return [
        ("（二）表1 限定之後", [
            spec_row("基本的 KD 交叉<br>（含 1000 萬門檻）", BASE, 2, "posneg"),
            spec_row("只在超賣區買、超買區賣<br>（20／80）", ZONE, 2, "posneg")]),
        ("（二）表2 拆開看", [
            spec_row("只有進場那半<br>（&lt;20＋死叉出）", "amt__low_zone__death", 2, "posneg"),
            spec_row("只有出場那半<br>（&gt;80＋黃金交叉買）", "amt__golden__high_death", 2, "posneg"),
            spec_row("兩半都要<br>（20／80）", ZONE, 2, "posneg")]),
        ("（二）表3 門檻", [
            spec_row("30/70<br>(較鬆)", ZONE + "__z30_70", 2, "posneg"),
            spec_row("20/80<br>(標準)", ZONE, 2, "posneg"),
            spec_row("10/90<br>(較嚴)", ZONE + "__z10_90", 2, "posneg")]),
    ]


# ── （三）進場的優化 ────────────────────────────────────────────────────────────
BASE_LABEL3 = "基本的 KD 交叉<br>（含 1000 萬門檻）"
# 小結排行表：文章列出的條件（名稱、變體）；列序由獲利因子決定，不照這裡
RANK3 = [
    ('<a href="#en-breakout">創 250 日新高</a>', "breakout250"),
    ("低檔黃金交叉＋紅K", "low_redk"),
    ('<a href="#en-gap">跳空</a>', "gap"),
    ('<a href="#en-divergence">底背離</a>', "divergence"),
    ("低檔黃金交叉（單獨）", "low_zone"),
    ('<a href="#en-mabull">均線多頭排列 5&gt;20&gt;60</a>', "bull_align_5_20_60"),
    ("均線多頭排列 20&gt;60&gt;120", "bull_align_20_60_120"),
    ("站上 MA20", "above_ma20"),
    ("長線均線多頭 MA120&gt;200", "ma120_gt_ma200"),
    ("K 線品質（非黑K）", "candle_not_black"),
    ("OBV 上升", "obv_up_1d"),
    ("<b>基本的 KD 交叉（不加）</b>", None),
    ("站上 MA200", "above_ma200"),
    ("站上 MA120", "above_ma120"),
    ("站上 MA60", "above_ma60"),
    ("帶量（量&gt;5日均量）", "vol_above_ma5"),
    ("量增 2 倍", "vol_x2"),
    ("交叉強度（K−D&gt;5）", "kd_spread5"),
    ("量增 1.5 倍", "vol_x1_5"),
    ("連續確認（隔日 K 仍&gt;D）", "confirm2"),
    ("資金流 CMF&gt;0", "cmf_pos"),
]


def var3(code) -> str:
    return BASE if code is None else f"amt__{code}__death"


def rank3_order() -> list:
    """小結排行：依獲利因子由高到低（精確值排序）。"""
    return sorted(RANK3, key=lambda x: -single_metrics(var3(x[1]))["獲利因子"])


def article3() -> list:
    base = single_metrics(BASE)

    def blk(rows):
        return [spec_row(BASE_LABEL3, BASE, 3, "updown")] + [
            spec_row(lbl, f"amt__{code}__death", 3, "updown", base) for lbl, code in rows]

    rank = [spec_row(lbl, var3(code), 3, "updown", None if code is None else base)
            for lbl, code in rank3_order()]
    return [
        ("（三）表1 突破創新高", blk([("＋創 20 日新高", "breakout20"), ("＋創 60 日新高", "breakout60"),
                                    ("＋創 120 日新高", "breakout120"), ("＋創 250 日新高", "breakout250")])),
        ("（三）表2 跳空", blk([("＋跳空", "gap")])),
        ("（三）表3 均線多頭", blk([("＋均線多頭<br>（5&gt;20&gt;60）", "bull_align_5_20_60")])),
        ("（三）表4 底背離", blk([("＋底背離", "divergence")])),
        ("（三）表5 排行", rank),
    ]


# ── （四）出場優化 ──────────────────────────────────────────────────────────────
BASE_LABEL4 = "基本的 KD 交叉<br>（一般死叉出）"
EXIT4 = [  # (小節表列名, 變體, 小結表列名)
    ("頂頂低 出場", "amt__golden__lower_high", '<a href="#ex-lowerhigh">頂頂低</a>'),
    ("K 跌破 80 出場", "amt__golden__k_down80", '<a href="#ex-k80">K 跌破 80</a>'),
    ("過熱後 K 下彎出場", "amt__golden__climax", '<a href="#ex-climax">過熱後 K 下彎</a>'),
    ("頂背離出場", "amt__golden__top_div", '<a href="#ex-topdiv">頂背離</a>'),
]
SUM4_EXTRA = [("高檔死叉<br>（K、D&gt;80 才賣）＊第二篇已見", "amt__golden__high_death"),
              ("K 跌破 50 或 80<br>（合併）", "amt__golden__k_down50-or-k_down80")]
SUM4_COLS = ["交易次數", "平均持有天", "勝率%", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]
SUM4_PAINT = ["勝率%", "中位數%", "期望值/筆", "獲利因子"]


def article4() -> list:
    base = single_metrics(BASE)
    out = []
    for lbl, var, _ in EXIT4:
        out.append((f"（四）{lbl}", [spec_row(BASE_LABEL4, BASE, 3, "updown"),
                                     spec_row(lbl, var, 2, "updown", base)]))
    rows = [(s, v) for _, v, s in EXIT4] + SUM4_EXTRA
    rows.sort(key=lambda x: -single_metrics(x[1])["獲利因子"])
    summ = [spec_row(s, v, 2, "updown", base, SUM4_COLS, SUM4_PAINT) for s, v in rows]
    summ.append(spec_row("<b>基本的 KD 交叉<br>（一般死叉）</b>", BASE, 2, "updown", None, SUM4_COLS))
    out.append(("（四）小結排行", summ))
    return out


# ── （五）拼裝整合與蒙地卡羅 ────────────────────────────────────────────────────
ENT5 = [("創 250 日新高", "創 250", "breakout250"), ("創 120 日新高", "創 120", "breakout120"),
        ("低檔＋紅K", "低檔＋紅K", "low_redk"), ("跳空", "跳空", "gap"),
        ("創 60 日新高", "創 60", "breakout60"), ("底背離", "底背離", "divergence")]
EXIT5 = [("死叉", "death"), ("高檔死叉", "high_death"), ("K跌破80", "k_down80"),
         ("過熱K下彎", "climax"), ("K跌破50", "k_down50"), ("頂頂低", "lower_high")]


def v5(entry: str, exit_: str) -> str:
    return f"amt__{entry}__{exit_}"


def ent5_order() -> list:
    """矩陣與附錄的進場列序＝進場本身（×一般死叉）的獲利因子由高到低（同（三）排行）。"""
    return sorted(ENT5, key=lambda e: -single_metrics(v5(e[2], "death"))["獲利因子"])


def best5() -> tuple:
    """36 組裡獲利因子最高、總獲利最高的（進場, 出場）——★／◆ 標記用。"""
    pf = {(e[2], x[1]): single_metrics(v5(e[2], x[1]))["獲利因子"] for e in ENT5 for x in EXIT5}
    tot = {k: single_metrics(v5(*k))["總獲利(萬)"] for k in pf}
    return max(pf, key=pf.get), max(tot, key=tot.get)


def matrix5(ents: list, best_pf: tuple) -> list:
    """6×6 獲利因子矩陣：> 1 上粉紅、全矩陣最高加 ★。"""
    rows = []
    for name, _, code in ents:
        row = [("", name)]
        for _, x in EXIT5:
            v = single_metrics(v5(code, x))["獲利因子"]
            row.append(("up" if v > 1 else "", fmt(v, 2) + (" ★" if (code, x) == best_pf else "")))
        rows.append(row)
    return rows


def high_death5(ents: list, best_pf: tuple, best_tot: tuple) -> list:
    """高檔死叉完整表：依獲利因子排；五欄對同一進場配一般死叉比。"""
    hd = sorted(ents, key=lambda e: -single_metrics(v5(e[2], "high_death"))["獲利因子"])
    rows = []
    for name, _, code in hd:
        m, d = single_metrics(v5(code, "high_death")), single_metrics(v5(code, "death"))
        r = [("", name)]
        for c in SPEC:
            txt = cell(m, c, 2)
            if c == "獲利因子" and (code, "high_death") == best_pf:
                txt += " ★"
            if c == "總獲利(萬)" and (code, "high_death") == best_tot:
                txt += " ◆"
            r.append((updown(m, d, c), txt))
        rows.append(r)
    return rows


def mc5() -> list:
    """蒙地卡羅表：依每筆期望高到低；報酬類 > 0 上粉紅。"""
    mc = pd.read_csv(os.path.join(MC_DIR, "kd_montecarlo.csv"), encoding="utf-8-sig")
    mc = mc[mc["分組"] == "文章"].sort_values("每筆期望%", ascending=False, kind="mergesort")
    label = {code: name for name, _, code in ENT5}
    return [[("", label[r["進場"]]), ("", fmt(r["交易數"], 0)), ("", r["抽樣模式"]),
             ("", fmt(r["勝率%"], 1)), ("up" if r["每筆期望%"] > 0 else "", fmt(r["每筆期望%"], 2)),
             *[("up" if r[c] > 0 else "", mc_cell(r, c, 0, sign=True))
               for c in ("報酬% P5", "報酬% 中位", "報酬% P95")],
             ("", mc_cell(r, "區間寬度", 0)), ("", fmt(r["本金大虧%"], 1)),
             ("", fmt(r["連敗 P95"], 0)), ("", fmt(r["回撤% P95"], 1))]
            for _, r in mc.iterrows()]


def appendix5(ents: list, best_pf: tuple, best_tot: tuple) -> list:
    """附錄 36 組：★＝高檔死叉列、◆＝全矩陣獲利因子／總獲利最高；獲利因子 > 1 上粉紅。"""
    rows = []
    for _, short, code in ents:
        for xname, x in EXIT5:
            m = single_metrics(v5(code, x))
            r = [("", f"{short} × {xname}{' ★' if x == 'high_death' else ''}")]
            for c in SPEC:
                txt, cls = cell(m, c, 2), ""
                if c == "獲利因子":
                    cls = "up" if m[c] > 1 else ""
                    txt += " ◆" if (code, x) == best_pf else ""
                if c == "總獲利(萬)" and (code, x) == best_tot:
                    txt += " ◆"
                r.append((cls, txt))
            rows.append(r)
    return rows


def article5() -> list:
    ents = ent5_order()
    best_pf, best_tot = best5()
    return [("（五）矩陣", matrix5(ents, best_pf)),
            ("（五）高檔死叉完整表", high_death5(ents, best_pf, best_tot)),
            ("（五）蒙地卡羅", mc5()), ("（五）附錄 36 組", appendix5(ents, best_pf, best_tot))]


def mc_cell(r: pd.Series, col: str, nd: int, sign: bool = False) -> str:
    """蒙地卡羅欄：存檔一位小數落在 .5 時改用精確值（重算該組路徑，seed 固定可重現）。"""
    from _04_analysis.kd.article.kd_article_common import is_tie
    v = float(r[col])
    if is_tie(v, nd):
        exact = mc_exact(r["變體"])[col]
        if float(half_up(exact, 1)) != v:          # 重算必須先重現存檔的一位小數
            raise SystemExit(f"{r['變體']} {col} 重算 {exact} 對不上存檔 {v}")
        v = exact
    return fmt(v, nd, sign=sign)


_MC_CACHE = {}


def mc_exact(variant: str) -> dict:
    """照 kd_montecarlo.run_case 的口徑重算（seed=42 同一套抽樣），回傳未四捨五入的報酬分位。"""
    if variant in _MC_CACHE:
        return _MC_CACHE[variant]
    import numpy as np
    from _04_analysis.analyze_vbt import _mc_core
    from _04_analysis.macd.macd_montecarlo import INIT_CASH, PATHS, RUIN_RATIO, T_HIGH, T_LOW
    from _04_analysis.kd.article.kd_article_common import SWEEP
    t = pd.read_parquet(os.path.join(SWEEP, variant, "single_kd_trades.parquet"))
    pnl = t.loc[t["real_pnl"] != 0, "real_pnl"].to_numpy(np.float64)
    finals, _, _, _ = _mc_core(pnl, PATHS, float(INIT_CASH), float(INIT_CASH * RUIN_RATIO),
                               42, T_LOW, T_HIGH)
    # 精確值＝未取整的分位數（存檔版先取整到元、再四捨五入到一位小數，兩次進位）
    q = {k: float(np.percentile(finals, p)) for k, p in (("P5", 5), ("中位", 50), ("P95", 95))}
    ret = {k: (v - INIT_CASH) / INIT_CASH * 100 for k, v in q.items()}
    out = {"報酬% P5": ret["P5"], "報酬% 中位": ret["中位"], "報酬% P95": ret["P95"],
           "區間寬度": ret["P95"] - ret["P5"]}
    _MC_CACHE[variant] = out
    return out


ARTICLES = {1: article1, 2: article2, 3: article3, 4: article4, 5: article5}


def main() -> int:
    ap = argparse.ArgumentParser(description="KD（一）～（五）表格 HTML")
    ap.add_argument("--article", type=int, required=True, choices=sorted(ARTICLES))
    a = ap.parse_args()
    for name, rows in ARTICLES[a.article]():
        print(f"\n=== {name} ===")
        print("\n".join(html_row(r) for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
