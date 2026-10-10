# -*- coding: utf-8 -*-
"""
驗第十篇（macd-combo）的表格與正文統計，全部對回 CSV。

文章裡的每一格都必須能在矩陣 CSV 或蒙地卡羅 CSV 找到來源，正文的統計句
（幾格高於基本版、排第幾名、範圍落在哪裡）也一起重算，避免憑印象寫的數字留在文章裡。

對應文章 2026-09-27 改版後的結構：
  ## 結果：三張矩陣           三張 5×5 獲利因子矩陣（底色＝高／低於基本版，灰＋†＝交易次數不足門檻）
  ## 排行：排除樣本不足之後    通過門檻的格子按獲利因子排序
  ## 蒙地卡羅壓測              五組（每個基礎取排名最前且達抽樣下限者，交叉取兩組，加無濾網對照）
  ## 附錄：完整 75 組數據       三個基礎的完整 10 欄＋未平倉%
開頭兩張「取前五」表的數字引用自（六）（九）篇，不在這兩份 CSV 裡，不在本檔驗證範圍。

兩份 CSV 都讀 SPA 回測／分析段的產出：矩陣＝result/macd_combo/macd_combo_6x6.csv
（_02_strategy/macd_strategy/macd_combo.py）、蒙地卡羅＝result/macd_mc/macd_montecarlo.csv
（_04_analysis/macd/macd_montecarlo.py）。

正文統計句的驗法：每一條先由 CSV 重算，再確認文章裡真的出現那段字；兩邊任何一邊變了都會報錯。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/macd/article/verify_article10.py
"""

import io
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402

POST = os.path.join(common.require_blog_dir(), "site", "content", "posts", "macd-combo.md")

# 文章為了好讀在名稱裡加了空格／改了說法，對回 CSV 前要還原
FILTER_MAP = {"創 250 日新高": "創250日新高", "均線多頭排列": "均線多頭排列",
              "ADX>25": "ADX>25", "收盤>MA200": "收盤>MA200",
              "RSI<50 且上升": "RSI<50且上升"}
EXIT_MAP = {"跌破年線": "跌破MA200", "超級趨勢": "Supertrend翻空",
            "抱滿 60 天": "抱滿60天", "跌破二十日低": "跌破20日低",
            "波段高點走低": "頂頂低"}
POP_MAP = {"黃金交叉": "交叉", "零軸上穿": "零軸", "純背離": "背離"}
POP_ORDER = ["黃金交叉", "零軸上穿", "純背離"]
THRESH = 5_949                     # 排序門檻＝平均每天至少成交一筆（2002–2025 交易日數）
T_LOW, T_HIGH = THRESH * 3, THRESH * 4     # 蒙地卡羅抽樣下限／上限
SECTIONS = ["## 結果：三張矩陣", "## 排行：排除樣本不足之後", "## 蒙地卡羅壓測",
            "## 重點整理", "## 附錄：完整 75 組數據"]

errs = []
checked = 0


def fail(msg):
    errs.append(msg)
    print("  X " + msg)


def check(ok, msg):
    """計數＋失敗時記錄；回傳 ok 方便呼叫端接著判斷。"""
    global checked
    checked += 1
    if not ok:
        fail(msg)
    return ok


def unescape(text):
    """表格裡的 < 要寫成 &lt;（否則瀏覽器把 <50 當標籤開頭），比對前還原。"""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def num(text):
    """文章用全形減號、千分位、正號與 † 註記，還原成可比的數字。"""
    t = re.sub(r"<sup>.*?</sup>", "", text)
    t = t.replace("−", "-").replace(",", "").replace("%", "").replace("+", "")
    return float(t.strip())


def same(cell, want):
    """
    文章數字是否等於 CSV 值四捨五入到「文章顯示的位數」。
    容許誤差依顯示位數決定（4 位小數的獲利因子只容 0.00005），不用一個固定值——
    固定 0.011 對 4 位小數的欄位太寬，改一個萬分位抓不到（突變測試踩過）。
    """
    shown = re.sub(r"<sup>.*?</sup>", "", cell).strip()
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return abs(num(cell) - float(want)) <= 0.5 * 10 ** -decimals + 1e-9


def rows_of(html):
    """回傳每一列的 [(td 屬性, td 內容), ...]。"""
    return [re.findall(r"<td([^>]*)>(.*?)</td>", tr)
            for tr in re.findall(r"<tr>(.*?)</tr>", html) if "<td" in tr]


def cls(attr):
    m = re.search(r'class="([^"]+)"', attr)
    return m.group(1) if m else ""


def split_sections(text):
    """依固定的五個二級標題切段；任何一個標題不見了就直接停（文章改版要先改這支）。"""
    missing = [h for h in SECTIONS if h not in text]
    if missing:
        raise SystemExit(f"文章結構與本檔不符，找不到標題：{missing}")
    parts = {}
    for i, h in enumerate(SECTIONS):
        body = text.split(h, 1)[1]
        if i + 1 < len(SECTIONS):
            body = body.split(SECTIONS[i + 1], 1)[0]
        parts[h] = body
    return parts


def lookup(mx, pop, filt, exit_):
    r = mx[(mx["母體"] == pop) & (mx["進場濾網"] == filt) & (mx["出場"] == exit_)]
    return r.iloc[0] if len(r) else None


def expect_class(r, base, thin_rule):
    """矩陣與排行的底色規則：交易次數不足門檻→thin，否則依高低於基本版。"""
    if thin_rule and r["交易次數"] < THRESH:
        return "thin"
    return "up" if r["獲利因子"] > base else "down"


def verify_matrices(sec, mx, base):
    """三張 5×5 獲利因子矩陣：數值、底色、† 與「—」。"""
    blocks = re.split(r"\*\*(黃金交叉|零軸上穿|純背離)\*\*（基本版 ([\d.]+)）", sec)[1:]
    check(len(blocks) == 9, f"矩陣段應有 3 張表，實際 {len(blocks) // 3}")
    for name, base_txt, html in zip(blocks[0::3], blocks[1::3], blocks[2::3]):
        pop = POP_MAP[name]
        check(same(base_txt, base[pop]),
              f"{name} 基本版：文 {base_txt} vs CSV {base[pop]:.4f}")
        exits = [unescape(x) for x in re.findall(r"<th>(.*?)</th>", html)[1:]]
        n = 0
        for row in rows_of(html):
            filt = unescape(row[0][1])
            for exit_, (attr, cell) in zip(exits, row[1:]):
                n += 1
                r = lookup(mx, pop, FILTER_MAP[filt], EXIT_MAP[exit_])
                if cell.strip() == "—":
                    check(r is not None and r["交易次數"] == 0,
                          f"矩陣 {name}/{filt}/{exit_}：文寫 — 但 CSV 有交易")
                    continue
                if not check(r is not None, f"矩陣 {name}/{filt}/{exit_}：CSV 查無"):
                    continue
                check(same(cell, r["獲利因子"]),
                      f"矩陣 {name}/{filt}/{exit_}：文 {cell} vs CSV {r['獲利因子']}")
                want = expect_class(r, base[pop], thin_rule=True)
                check(cls(attr) == want and (("†" in cell) == (want == "thin")),
                      f"矩陣 {name}/{filt}/{exit_} 底色：文 {cls(attr) or '無'} vs 應為 {want}")
        print(f"  矩陣 {name}：{n} 格")
        check(n == 25, f"矩陣 {name} 格數 {n} ≠ 25")


RANK_COLS = ["交易次數", "勝率%", "平均持有天", "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)"]


def verify_ranking(sec, passed, base):
    """排行表：列數、順序（獲利因子由高到低）、每格數值與底色。"""
    want = passed.sort_values("獲利因子", ascending=False).reset_index(drop=True)
    rows = rows_of(sec)
    print(f"  排行表：{len(rows)} 列（應為 {len(want)}）")
    check(len(rows) == len(want), f"排行列數 {len(rows)} ≠ {len(want)}")
    rev_pop = {v: k for k, v in POP_MAP.items()}
    for i, (row, (_, r)) in enumerate(zip(rows, want.iterrows()), start=1):
        cells = [c for _, c in row]
        key = (cells[1], FILTER_MAP.get(unescape(cells[2])), EXIT_MAP.get(unescape(cells[3])))
        if not check(cells[0] == str(i) and key == (rev_pop[r["母體"]], r["進場濾網"], r["出場"]),
                     f"排行第 {i} 名：文 {cells[:4]} vs 應為 {r['母體']}/{r['進場濾網']}/{r['出場']}"):
            continue
        for col, cell in zip(RANK_COLS, cells[4:]):
            check(same(cell, r[col]), f"排行第 {i} 名 {col}：文 {cell} vs CSV {r[col]}")
        check(cls(row[9][0]) == expect_class(r, base[r["母體"]], thin_rule=False),
              f"排行第 {i} 名 獲利因子底色")


MC_COLS = ["交易數", "抽樣模式", "勝率%", "賺賠比", "報酬% P5", "報酬% 中位", "報酬% P95",
           "信賴區間寬度", "破產%", "最大連敗 P95", "最大回撤% P95"]


def mc_key(label):
    """「交叉 × 均線多頭排列 × 跌破年線」→（母體, 濾網, 出場）的 CSV 名稱。"""
    pop, filt, exit_ = [x.strip() for x in unescape(label).split("×")]
    filt = FILTER_MAP.get(filt, filt.replace(" ", ""))
    return pop, filt, EXIT_MAP.get(exit_, exit_)


def verify_mc_table(sec, mc, mx):
    """蒙地卡羅表：每格對 MC CSV，交易數再對矩陣 CSV（兩份 CSV 必須是同一批交易）。"""
    rows = rows_of(sec)
    print(f"  蒙地卡羅表：{len(rows)} 列（應為 {len(mc)}）")
    check(len(rows) == len(mc), f"蒙地卡羅列數 {len(rows)} ≠ {len(mc)}")
    for row in rows:
        label = unescape(row[0][1])
        m = mc[mc["組合"].str.replace(" ", "") == label.replace(" ", "")]
        if not check(len(m) == 1, f"蒙地卡羅查無組合：{label}"):
            continue
        r = m.iloc[0]
        for col, (_, cell) in zip(MC_COLS, row[1:]):
            if col == "抽樣模式":
                check(cell.strip() == r[col], f"蒙地卡羅 {label} 抽樣模式：文 {cell} vs CSV {r[col]}")
                continue
            check(same(cell, r[col]), f"蒙地卡羅 {label} {col}：文 {cell} vs CSV {r[col]}")
        x = lookup(mx, *mc_key(label))
        check(x is not None and x["交易次數"] == r["交易數"],
              f"蒙地卡羅 {label} 交易數與矩陣 CSV 不一致")


APPX_COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%",
             "期望值/筆", "獲利因子", "總獲利(萬)", "未平倉%"]


def verify_appendix(sec, mx, base):
    """附錄三張完整表：每列 10 欄＋獲利因子底色（附錄不用灰底，只標高低於基本版）。"""
    parts = re.split(r"^### (黃金交叉|零軸上穿|純背離)\s*$", sec, flags=re.M)[1:]
    check(len(parts) == 6, f"附錄應有 3 張表，實際 {len(parts) // 2}")
    for name, html in zip(parts[0::2], parts[1::2]):
        pop, n = POP_MAP[name], 0
        for row in rows_of(html):
            cells = [c for _, c in row]
            filt, exit_ = unescape(cells[0]), unescape(cells[1])
            if exit_ not in EXIT_MAP:
                # 合併說明列（純背離 × 創 250 日新高，條件互斥）：CSV 裡這五格必須全是 0 筆
                zero = mx[(mx["母體"] == pop) & (mx["進場濾網"] == FILTER_MAP.get(filt))]
                check(len(zero) == 5 and (zero["交易次數"] == 0).all(),
                      f"附錄 {name}/{filt}：文寫「{exit_}」但 CSV 不是五格皆 0 筆")
                continue
            r = lookup(mx, pop, FILTER_MAP[filt], EXIT_MAP[exit_])
            if not check(r is not None, f"附錄 {name}/{filt}/{exit_}：CSV 查無"):
                continue
            n += 1
            for col, cell in zip(APPX_COLS, cells[2:]):
                check(same(cell, r[col]), f"附錄 {name}/{filt}/{exit_} {col}：文 {cell} vs CSV {r[col]}")
            check(cls(row[9][0]) == expect_class(r, base[pop], thin_rule=False),
                  f"附錄 {name}/{filt}/{exit_} 獲利因子底色")
        want = int(((mx["母體"] == pop) & (mx["交易次數"] > 0)).sum())
        print(f"  附錄 {name}：{n} 列（應為 {want}）")
        check(n == want, f"附錄 {name} 列數 {n} ≠ {want}")


def fmt_win_by_exit(passed):
    """五條出場的平均勝率，由高到低排成文章那句的寫法。"""
    rev = {v: k for k, v in EXIT_MAP.items()}
    s = passed.groupby("出場")["勝率%"].mean().sort_values(ascending=False)
    return "、".join(f"{rev[k]} {v:.2f}%" for k, v in s.items())


def filter_vs_none(passed, full):
    """每條濾網勝過「無濾網 × 同一出場」的格數／比較格數（只比通過門檻的格子）。"""
    out = {}
    for f, g in passed.groupby("進場濾網"):
        won = sum(r["獲利因子"] > lookup(full, r["母體"], "無濾網", r["出場"])["獲利因子"]
                  for _, r in g.iterrows())
        out[f] = f"{won}/{len(g)}"
    return out


def mc_selection(passed, full):
    """依文章規則重選蒙地卡羅五組：各基礎取排名最前且達抽樣下限者（交叉兩組），加無濾網對照。"""
    ok = passed[passed["交易次數"] >= T_LOW].sort_values("獲利因子", ascending=False)
    picks = []
    for pop, k in (("交叉", 2), ("零軸", 1), ("背離", 1)):
        picks += [tuple(r) for r in ok[ok["母體"] == pop][["母體", "進場濾網", "出場"]].values[:k]]
    picks.append(("交叉", "無濾網", "跌破MA200"))
    return sorted(picks)


def frac_above(passed, above, col, key):
    """某一群通過門檻的格子裡，高於基本版的「格數/總格數」。"""
    mask = passed[col] == key
    return f"{int(above[mask].sum())}/{int(mask.sum())}"


def half_up(v, nd):
    """四捨五入到 nd 位（文章口徑）；f-string 遇到 .5 邊界會受二進位誤差影響，改走 Decimal。"""
    return f"{Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)}"


def cross_filter_spread(passed):
    """黃金交叉表裡「同一條濾網換不同出場」的獲利因子高低差：回傳差距最小與最大的兩條濾網。"""
    rev = {v: k for k, v in FILTER_MAP.items()}
    g = passed[passed["母體"] == "交叉"].groupby("進場濾網")["獲利因子"].agg(["min", "max"])
    g["spread"] = g["max"] - g["min"]
    lo, hi = g["spread"].idxmin(), g["spread"].idxmax()

    def one(f):
        name = rev[f].replace(">", "&gt;").replace("<", "&lt;")
        r = g.loc[f]
        return f"{half_up(r['spread'], 2)}（{name}，{half_up(r['min'], 2)}～{half_up(r['max'], 2)}）"
    return f"獲利因子的高低差從 {one(lo)}到 {one(hi)}"


def all_or_frac(frac, text):
    """全部過關時回傳文章「全部高於」的寫法；沒全過就回傳分數本身（比對會失敗並印出實際值）。"""
    won, total = frac.split("/")
    return text.format(n=total) if won == total else frac


def claims_matrix(mx, full, passed, base):
    """「三個要先講清楚的讀法」與「結果：三張矩陣」兩段的統計句。"""
    thin = mx[(mx["交易次數"] > 0) & (mx["交易次數"] < THRESH)]
    z250 = mx[(mx["母體"] == "零軸") & (mx["進場濾網"] == "創250日新高")]
    hold0 = lookup(full, "交叉", "無濾網", "原生出場")["平均持有天"]
    hold1 = lookup(full, "交叉", "無濾網", "跌破MA200")["平均持有天"]
    above = passed.apply(lambda r: r["獲利因子"] > base[r["母體"]], axis=1)
    pop = {p: frac_above(passed, above, "母體", p) for p in ("交叉", "零軸", "背離")}
    n_combo = ["零", "一", "二", "三", "四"][thin.groupby(["母體", "進場濾網"]).ngroups]
    return [
        ("三個基本版", f"{base['交叉']:.4f}、{base['零軸']:.4f}、{base['背離']:.4f}",
         "0.9869、1.2388、1.0290"),
        ("交叉換跌破年線的持有天", f"{hold0:.2f} 天變成 {hold1:.2f} 天", "18.65 天變成 128.05 天"),
        ("不過門檻的格數", f"{len(mx)} 格裡有 {len(thin)} 格沒過", "75 格裡有 15 格沒過"),
        ("未平倉率最大值", f"最高只有 {mx['未平倉%'].max():.2f}%", "最高只有 3.79%"),
        ("零軸×創250 的獲利因子範圍",
         f"{z250['獲利因子'].min():.2f} 到 {z250['獲利因子'].max():.2f}", "1.51 到 2.20"),
        ("零軸×創250 的交易次數範圍",
         f"{z250['交易次數'].min()} 到 {z250['交易次數'].max()} 筆", "641 到 679 筆"),
        ("不及格格子來自幾個搭配", f"{len(thin)} 格只來自{n_combo}個搭配", "15 格只來自三個搭配"),
        ("黃金交叉高於基本版的格數", all_or_frac(pop["交叉"], "黃金交叉 {n} 格全部高於基本版"),
         "黃金交叉 25 格全部高於基本版"),
        ("零軸／背離高於基本版", f"零軸上穿 {pop['零軸']}、純背離 {pop['背離']}",
         "零軸上穿 9/15、純背離 12/15"),
        ("黃金交叉表同一濾網換出場的高低差", cross_filter_spread(passed),
         "獲利因子的高低差從 0.31（收盤&gt;MA200，1.14～1.45）到 0.46（創 250 日新高，1.16～1.63）"),
    ]


def claims_ranking(full, passed, base):
    """「排行：排除樣本不足之後」與「重點整理」裡跟排行有關的統計句。"""
    ranked = passed.sort_values("獲利因子", ascending=False).reset_index(drop=True)
    ma = passed[passed["出場"] == "跌破MA200"]
    above = passed.apply(lambda r: r["獲利因子"] > base[r["母體"]], axis=1)
    ex = {e: frac_above(passed, above, "出場", e) for e in EXIT_MAP.values()}
    fv = filter_vs_none(passed, full)
    n_ma_top11 = int((ranked["出場"].iloc[:11] == "跌破MA200").sum())
    return [
        ("通過門檻格數", f"{len(passed)} 個通過門檻的格子", "55 個通過門檻的格子"),
        ("前 11 名幾名是跌破年線", f"前 11 名裡剛好有 {n_ma_top11} 名是它", "前 11 名裡剛好有 9 名是它"),
        ("跌破年線過關格數", all_or_frac(ex["跌破MA200"], "通過門檻的 {n} 格全部高於基本版"),
         "通過門檻的 11 格全部高於基本版"),
        ("跌破年線平均", f"平均獲利因子 {ma['獲利因子'].mean():.4f}、平均抱 {ma['平均持有天'].mean():.1f} 天",
         "平均獲利因子 1.4659、平均抱 144.9 天"),
        ("其餘四條出場過關格數",
         f"抱滿 60 天 {ex['抱滿60天']}、超級趨勢 {ex['Supertrend翻空']}、"
         f"波段高點走低 {ex['頂頂低']}、跌破二十日低 {ex['跌破20日低']}",
         "抱滿 60 天 9/11、超級趨勢 9/11、波段高點走低 9/11、跌破二十日低 8/11"),
        ("五條出場平均勝率", fmt_win_by_exit(passed),
         "抱滿 60 天 45.64%、波段高點走低 39.55%、超級趨勢 38.82%、跌破年線 36.36%、跌破二十日低 30.89%"),
        ("濾網勝過無濾網（排行段）",
         f"收盤&gt;MA200 勝過無濾網 {fv['收盤>MA200']} 格、ADX&gt;25 {fv['ADX>25']}、"
         f"均線多頭排列 {fv['均線多頭排列']}、創 250 日新高 {fv['創250日新高']}",
         "收盤&gt;MA200 勝過無濾網 9/15 格、ADX&gt;25 11/15、均線多頭排列 8/10、創 250 日新高 3/5"),
        ("RSI 濾網勝過無濾網", f"RSI&lt;50 且上升只有 {fv['RSI<50且上升']}", "RSI&lt;50 且上升只有 3/10"),
        ("濾網勝過無濾網（重點整理）",
         f"收盤&gt;MA200 {fv['收盤>MA200']}、ADX&gt;25 {fv['ADX>25']}、均線多頭排列 {fv['均線多頭排列']}、"
         f"創 250 日新高 {fv['創250日新高']}、RSI&lt;50 且上升 {fv['RSI<50且上升']}",
         "收盤&gt;MA200 9/15、ADX&gt;25 11/15、均線多頭排列 8/10、創 250 日新高 3/5、RSI&lt;50 且上升 3/10"),
        ("排行前段的交易次數",
         f"第一名只有 {ranked.loc[0, '交易次數']:,} 筆、剛過門檻，第二名 {ranked.loc[1, '交易次數']:,} 筆、"
         f"第四名 {ranked.loc[3, '交易次數']:,} 筆",
         "第一名只有 6,989 筆、剛過門檻，第二名 24,824 筆、第四名 6,334 筆"),
    ]


def claims_mc_selection(mx, passed):
    """「蒙地卡羅壓測」裡關於抽樣區間與「哪幾組被挑中／被擋下」的統計句。"""
    ranked = passed.sort_values("獲利因子", ascending=False).reset_index(drop=True)
    top = mx[mx["交易次數"] > 0].sort_values("獲利因子", ascending=False).iloc[0]
    div_rank = ranked[(ranked["母體"] == "背離") & (ranked["交易次數"] >= T_LOW)].index[0] + 1
    # 被擋下＝交叉／零軸各自排在該母體入選組之前、但交易次數不到抽樣下限的格子
    # （不寫死名次，重跑後名次會動；背離另有「純背離的代表」一句交代）
    hit = []
    for pop, k in (("交叉", 2), ("零軸", 1)):
        sub = ranked[ranked["母體"] == pop]
        last_pick = sub[sub["交易次數"] >= T_LOW].index[k - 1]
        hit += list(sub.loc[:last_pick][sub.loc[:last_pick, "交易次數"] < T_LOW].index)
    blocked = ranked.loc[sorted(hit)]
    blocked_txt = "與".join(f"第 {i + 1}（{r['交易次數']:,} 筆）" for i, r in blocked.iterrows())
    # 純背離：排在入選組之前、但交易次數不到抽樣下限的全部格子（不只 ADX>25）
    rev_f = {v: k for k, v in FILTER_MAP.items()}
    div_sub = ranked.loc[:div_rank - 1]
    div_blocked = div_sub[(div_sub["母體"] == "背離") & (div_sub["交易次數"] < T_LOW)]
    n_word = ["零", "一", "兩", "三", "四", "五"][len(div_blocked)]
    div_blocked_txt = "；".join(
        f"第 {i + 1} 的{' ' if rev_f[r['進場濾網']][0].isascii() else ''}"   # 中英之間空一格，同文章寫法
        f"{rev_f[r['進場濾網']].replace('>', '&gt;').replace('<', '&lt;')}，{r['交易次數']:,} 筆"
        for i, r in div_blocked.iterrows())
    return [
        ("抽樣區間", f"約 {THRESH:,} 個交易日，一次完整部署落在 {T_LOW:,}～{T_HIGH:,} 筆",
         "約 5,949 個交易日，一次完整部署落在 17,847～23,796 筆"),
        ("被抽樣下限擋下的組", f"排行{blocked_txt}" if len(blocked) else "未被擋下",
         "排行第 1（6,989 筆）與第 4（6,334 筆）"),
        ("純背離的代表",
         f"排行第 {div_rank} 的 RSI&lt;50 且上升（{ranked.loc[div_rank - 1, '交易次數']:,} 筆），"
         f"排在它前面的{n_word}組純背離（{div_blocked_txt}）都沒到下限",
         "排行第 14 的 RSI&lt;50 且上升（18,775 筆），排在它前面的兩組純背離"
         "（第 5 的 ADX&gt;25，16,989 筆；第 13 的收盤&gt;MA200，13,689 筆）都沒到下限"),
        ("全表最高那格", f"全表第一的 {top['獲利因子']:.4f} 只成交 {top['交易次數']} 筆",
         "全表第一的 2.2045 只成交 641 筆"),
    ]


def claims_mc_results(mc):
    """「蒙地卡羅壓測」裡關於報酬、回撤、連敗的統計句。"""
    m = {mc_key(r["組合"]): r for _, r in mc.iterrows()}
    none_mc = m[("交叉", "無濾網", "跌破MA200")]["報酬% 中位"]
    gain = {f: m[("交叉", f, "跌破MA200")]["報酬% 中位"] for f in ("均線多頭排列", "ADX>25")}
    zero_mc = m[("零軸", "收盤>MA200", "跌破MA200")]["報酬% 中位"]
    div_mc = m[("背離", "RSI<50且上升", "跌破MA200")]["報酬% 中位"]
    longest, best_win = mc.loc[mc["最大連敗 P95"].idxmax()], mc.loc[mc["勝率%"].idxmax()]
    return [
        ("濾網的報酬差距",
         f"是 +{none_mc:.0f}%，加均線多頭排列變 +{gain['均線多頭排列']:.0f}%"
         f"（+{(gain['均線多頭排列'] / none_mc - 1) * 100:.0f}%）、加 ADX&gt;25 變 +{gain['ADX>25']:.0f}%"
         f"（+{(gain['ADX>25'] / none_mc - 1) * 100:.0f}%）",
         "是 +600%，加均線多頭排列變 +866%（+44%）、加 ADX&gt;25 變 +859%（+43%）"),
        ("換基礎的報酬",
         f"零軸 × 收盤&gt;MA200 +{half_up(zero_mc, 0)}%（比對照組少 {half_up((1 - zero_mc / none_mc) * 100, 0)}%）、"
         f"純背離 × RSI&lt;50 且上升 +{half_up(div_mc, 0)}%（少 {half_up((1 - div_mc / none_mc) * 100, 0)}%）",
         "零軸 × 收盤&gt;MA200 +558%（比對照組少 7%）、純背離 × RSI&lt;50 且上升 +393%（少 35%）"),
        ("蒙地卡羅回撤與連敗範圍",
         f"回撤 P95 從 {mc['最大回撤% P95'].min():.1f}% 到 {mc['最大回撤% P95'].max():.1f}%、"
         f"最大連敗 P95 從 {mc['最大連敗 P95'].min()} 筆到 {mc['最大連敗 P95'].max()} 筆",
         "回撤 P95 從 6.7% 到 11.4%、最大連敗 P95 從 19 筆到 43 筆"),
        ("連敗最長那組", f"勝率只有 {longest['勝率%']:.2f}%，靠 {longest['賺賠比']:.2f} 的賺賠比",
         "勝率只有 23.29%，靠 5.38 的賺賠比"),
        ("勝率最高那組", f"勝率 {best_win['勝率%']:.2f}% 是五組最高、連敗最短"
         f"（{best_win['最大連敗 P95']} 筆），但賺賠比只有 {best_win['賺賠比']:.2f}",
         "勝率 46.89% 是五組最高、連敗最短（19 筆），但賺賠比只有 1.61"),
    ]


def body_claims(mx, full, mc, passed, base):
    """
    正文統計句，依文章段落分四組；回傳 [(說明, 由 CSV 重算出的那段字, 文章裡應出現的那段字)]。
    文章某一段改版時，只要改對應的那一組。
    """
    return (claims_matrix(mx, full, passed, base) + claims_ranking(full, passed, base)
            + claims_mc_selection(mx, passed) + claims_mc_results(mc))


def main():
    mx_all = pd.read_csv(os.path.join(common.result_dir("macd_strategy", "macd_combo"), "macd_combo_6x6.csv"))
    # CSV 是 7×6 跑出來的，文章只用排序前五的兩軸，先裁掉落選的列
    mx = mx_all[mx_all["進場濾網"].isin(FILTER_MAP.values())
                & mx_all["出場"].isin(EXIT_MAP.values())].copy()
    mc = pd.read_csv(os.path.join(common.result_dir("macd_strategy", "macd_mc"), "macd_montecarlo.csv"))
    base = {p: lookup(mx_all, p, "無濾網", "原生出場")["獲利因子"] for p in ("交叉", "零軸", "背離")}
    passed = mx[mx["交易次數"] >= THRESH]
    text = io.open(POST, encoding="utf-8").read()
    sec = split_sections(text)

    verify_matrices(sec["## 結果：三張矩陣"], mx, base)
    verify_ranking(sec["## 排行：排除樣本不足之後"], passed, base)
    verify_mc_table(sec["## 蒙地卡羅壓測"], mc, mx_all)     # 含「無濾網」對照組，要用未裁切的
    check(sorted(mc_key(x) for x in mc["組合"]) == mc_selection(passed, mx_all),
          f"蒙地卡羅五組與挑選規則不符：應為 {mc_selection(passed, mx_all)}")
    verify_appendix(sec["## 附錄：完整 75 組數據"], mx, base)

    print("  正文統計：")
    for name, got, want in body_claims(mx, mx_all, mc, passed, base):
        ok = check(got == want, f"{name}：CSV 重算為「{got}」，本檔預期「{want}」")
        ok = check(want in text, f"{name}：文章裡找不到「{want}」") and ok
        print(f"    {'OK' if ok else 'NG'} {name}")

    print(f"\n對照 {checked} 個數值，錯誤 {len(errs)} 個")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
