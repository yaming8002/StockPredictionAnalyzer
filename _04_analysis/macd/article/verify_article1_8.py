# -*- coding: utf-8 -*-
"""
驗 MACD（一）～（八）篇的表格數字對得回 SPA 單股回測結果。

這八篇的表格不是由 builder 產生（當年手工排版），所以改用「逐列比對」：
- 標準結果表（交易次數／勝率／…／總獲利 這組欄位）：每一列的所有數字欄，
  必須能在回測結果池裡找到**同一列**、在文章顯示的位數下四捨五入（ROUND_HALF_UP）後全部相同。
  不靠列名對應，是因為文章列名（例「交叉（黃金交叉 ／ 死亡交叉）」）和 CSV 名稱寫法不同；
  十個欄位同時對上的機率極低，足以認定是同一列。
- 獲利因子格狀表（（四）矩陣、（五）～（八）的「四個母體同向？」彙總）：每一格 4 位小數的
  獲利因子必須存在於結果池；「同一列／同一欄的差距」重算；「N 升 M 降」依同欄基本版重數。
- （八）「虧損平均改善的格數」那張是衍生統計，由本檔依 _exit_sweep 的 B疊加組重算。

結果池＝result/single_macd 的 _baselines、_matrix_3x3、_entry_sweep_v2、_exit_parallel、
_exit_sweep、_exit_pure_risk（_02_strategy/macd_strategy 各 sweep driver 的產出）。
CSV 本身已存 2 位小數，文章顯示 1 位時會遇到「第 2 位剛好是 5」的雙重進位邊界：
這類格只要等於任一種進位結果就放行，並列為「邊界」提示（不計錯）。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/macd/article/verify_article1_8.py
（需設 BLOG_DIR 指向 blog 目錄；--pool-dir 可改用別的結果目錄比對）
"""

import argparse
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

SLUGS = ["macd-crossover", "macd-zero-line", "macd-divergence", "macd-matrix",
         "macd-entry-filters", "macd-entry-filters-2", "macd-exit-trend", "macd-exit-risk"]
POOL_FILES = ["_baselines", "_matrix_3x3", "_entry_sweep_v2", "_exit_parallel",
              "_exit_sweep", "_exit_pure_risk"]
# 表頭關鍵字 → CSV 欄名（依序比對，先中先贏）
HEAD_MAP = [("交易次數", "交易次數"), ("勝率", "勝率%"), ("持有", "平均持有天"),
            ("獲利平均", "獲利平均%"), ("虧損平均", "虧損平均%"), ("中位", "中位數%"),
            ("期望值", "期望值/筆"), ("獲利因子", "獲利因子"), ("總獲利", "總獲利(萬)"),
            ("未平倉", "未平倉%")]
NUM_RE = re.compile(r"^[+−-]?[\d,]+(\.\d+)?%?$")


def load_pool(pool_dir: str) -> pd.DataFrame:
    """把六份結果疊成一張池（只留比對用的數字欄）。"""
    frames = []
    for name in POOL_FILES:
        d = pd.read_csv(os.path.join(pool_dir, name + ".csv"))
        d["_src"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).replace("&gt;", ">").replace("&lt;", "<").strip()


def parse_num(text: str):
    """文章顯示字串 → (Decimal, 小數位數)；非數字回 None。"""
    t = text.strip()
    if not NUM_RE.match(t):
        return None
    t = t.replace("−", "-").replace(",", "").replace("%", "")
    dec = len(t.split(".")[1]) if "." in t else 0
    return Decimal(t), dec


def rounded(value: float, dec: int) -> set:
    """CSV 值在 dec 位下的可接受顯示值（含雙重進位邊界的兩種結果）。"""
    v = Decimal(repr(float(value)))
    q = Decimal(1).scaleb(-dec)
    out = {v.quantize(q, rounding=ROUND_HALF_UP)}
    # CSV 只存到第 2 位（獲利因子 4 位）：若被截掉的部分剛好是 5，真值可能在兩側
    tail = v - v.quantize(q, rounding="ROUND_DOWN")
    if abs(abs(tail) - q / 2) < Decimal("1e-9"):
        out.add(v.quantize(q, rounding="ROUND_DOWN"))
        out.add(v.quantize(q, rounding="ROUND_UP"))
    return out


def tables(md: str):
    """逐張回傳 (表頭文字串列, [列的 (原始 tr 字串, 儲存格文字串列)])。"""
    for m in re.finditer(r'<table class="bt-table">(.*?)</table>', md, re.S):
        body = m.group(1)
        heads = [strip(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", body, re.S)]
        rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
            if "<th" in tr:
                continue
            rows.append((tr, [strip(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]))
        yield heads, rows


def col_map(heads):
    """表頭 → {儲存格索引: CSV 欄名}；不是標準結果表回 None。"""
    out = {}
    for i, h in enumerate(heads):
        for key, col in HEAD_MAP:
            if key in h:
                out[i] = col
                break
    return out if "交易次數" in out.values() and "獲利因子" in out.values() else None


def match_row(pool, cells, cmap):
    """回傳池中所有數字欄都對得上的列索引（含是否落在邊界）。"""
    # 列名欄可能被 <br> 拆成兩格（（三）的對照表），數字欄從右邊對齊
    shift = len(cells) - (max(cmap) + 1)
    want = {}
    for i, col in cmap.items():
        j = i + shift
        if 0 <= j < len(cells):
            p = parse_num(cells[j])
            if p is not None:
                want[col] = p
    hits = []
    for idx, row in pool.iterrows():
        edge = False
        ok = True
        for col, (val, dec) in want.items():
            if pd.isna(row.get(col)):
                ok = False
                break
            acc = rounded(row[col], dec)
            if val not in acc:
                ok = False
                break
            if len(acc) > 1:
                edge = True
        if ok:
            hits.append((idx, edge))
    return hits, want


def check_pf_grid(heads, rows, pf4: set, report):
    """獲利因子格狀表：4 位小數必須在池內；差距與升降重算。"""
    errs = 0
    grid = []
    for _, cells in rows:
        grid.append(cells)
        for c in cells:
            p = parse_num(c)
            if p and p[1] == 4 and p[0] not in pf4:
                report(f"獲利因子 {c} 不在結果池（列：{cells[0]}）")
                errs += 1
    # （四）矩陣：「同一列的差距」＝該列三格 max−min（3 位）；「同一欄的差距」同理
    if any("差距" in h for h in heads):
        body = [r for r in grid if not r[0].startswith("同一欄")]
        for r in body:
            vals = [parse_num(c)[0] for c in r[1:4]]
            want = (max(vals) - min(vals)).quantize(Decimal("0.001"), ROUND_HALF_UP)
            if parse_num(r[4])[0] != want:
                report(f"矩陣「{r[0]}」同一列差距 {r[4]} ≠ {want}")
                errs += 1
        foot = [r for r in grid if r[0].startswith("同一欄")]
        for r in foot:
            for k in range(3):
                vals = [parse_num(b[1 + k])[0] for b in body]
                want = (max(vals) - min(vals)).quantize(Decimal("0.001"), ROUND_HALF_UP)
                if parse_num(r[1 + k])[0] != want:
                    report(f"矩陣第 {k + 1} 欄差距 {r[1 + k]} ≠ {want}")
                    errs += 1
    # 「四個母體同向？」「方向」：對同欄基本版數升降
    dir_idx = next((i for i, h in enumerate(heads) if "同向" in h or h == "方向"), None)
    if dir_idx is not None:
        pf_cols = [i for i, h in enumerate(heads) if h in ("交叉", "零軸", "背離", "交叉進 × 零軸出")]
        base = grid[0]
        # 著色：up＝高於同欄基本版、down＝低於（粉紅／淺綠由 CSS 決定）
        for tr, cells in rows[1:]:
            attrs = re.findall(r"<td([^>]*)>", tr)
            for i in pf_cols:
                cls = "up" if 'class="up"' in attrs[i] else "down" if 'class="down"' in attrs[i] else None
                a, b = parse_num(cells[i]), parse_num(base[i])
                if cls and a and b and cls != ("up" if a[0] > b[0] else "down" if a[0] < b[0] else "eq"):
                    report(f"「{cells[0]}」{heads[i]} {cells[i]} 著色 {cls} 與基本版 {base[i]} 不符")
                    errs += 1
        for r in grid[1:]:
            # 「—」＝該母體沒有這個條件（例：背離不疊創新高），記為「無」
            pairs = [(parse_num(r[i]), parse_num(base[i])) for i in pf_cols]
            up = sum(1 for a, b in pairs if a and a[0] > b[0])
            dn = sum(1 for a, b in pairs if a and a[0] < b[0])
            na = sum(1 for a, _ in pairs if a is None)
            want = " ".join(x for x in [f"{up} 升" if up else "", f"{dn} 降" if dn else "",
                                        f"{na} 無" if na else ""] if x)
            if r[dir_idx] != want:
                report(f"「{r[0]}」方向 {r[dir_idx]} ≠ {want}")
                errs += 1
    return errs


POP_KEYS = ["交叉進 × 零軸出", "交叉", "零軸", "背離"]


def check_colors(tabs, report):
    """
    標準結果表的 up／down 著色要和「同母體基本版」比較一致。
    基本版＝文章開頭那張四母體表（列名帶全名括號、本身不著色）。
    """
    base, errs = {}, 0
    for heads, rows in tabs:
        if not col_map(heads):
            continue
        for tr, cells in rows:
            k = next((k for k in POP_KEYS if cells[0].startswith(k)), None)
            if k and "（" in cells[0] and 'class="up"' not in tr and 'class="down"' not in tr:
                base.setdefault(k, cells)
    for heads, rows in tabs:
        if not col_map(heads):
            continue
        for tr, cells in rows:
            k = next((k for k in POP_KEYS if cells[0].startswith(k)), None)
            if not k or k not in base or "（" in cells[0]:
                continue
            for j, attr in enumerate(re.findall(r"<td([^>]*)>", tr)):
                cls = "up" if 'class="up"' in attr else "down" if 'class="down"' in attr else None
                a, b = parse_num(cells[j]), parse_num(base[k][j])
                if cls and a and b and cls != ("up" if a[0] > b[0] else "down" if a[0] < b[0] else "eq"):
                    report(f"「{cells[0]}」{heads[j]} {cells[j]} 著色 {cls} 與基本版 {base[k][j]} 不符")
                    errs += 1
    return errs


def check_risk_summary(rows, pool_dir, report):
    """（八）風控彙總：四個母體 B疊加 vs 原生出場的平均變化（pp／獲利因子）。"""
    sw = pd.read_csv(os.path.join(pool_dir, "_exit_sweep.csv"))
    nat = {}
    mx = pd.read_csv(os.path.join(pool_dir, "_matrix_3x3.csv"))
    # 四個母體的原生出場：三條對角線＋交叉進×零軸出
    for pop, combo in {"交叉": "交叉進 × 交叉出", "零軸": "零軸進 × 零軸出",
                       "背離": "背離進 × 交叉出", "交叉進×零軸出": "交叉進 × 零軸出"}.items():
        nat[pop] = mx[mx["組合"] == combo].iloc[0]
    names = {"固定停損 2×ATR": "停損2ATR", "吊燈 3×ATR": "吊燈3ATR", "自最高點回落 10%": "回落10%",
             "固定停利 +20%": "停利+20%", "抱滿 60 天": "抱滿60天"}
    errs = 0
    for _, cells in rows:
        key = names.get(cells[0])
        if key is None:
            continue
        sub = sw[(sw["組"] == "B疊加") & (sw["出場"] == key)].set_index("母體")
        d_loss = [sub.loc[p, "虧損平均%"] - nat[p]["虧損平均%"] for p in nat]
        d_win = [sub.loc[p, "獲利平均%"] - nat[p]["獲利平均%"] for p in nat]
        d_pf = [sub.loc[p, "獲利因子"] - nat[p]["獲利因子"] for p in nat]
        # CSV 每格已進位（% 到 0.01、獲利因子到 0.0001），差值最多偏 ±1 個末位：
        # 文章是用未進位值算的，所以這裡只要求落在「CSV 推得出的範圍」內
        sure = sum(x > 0.01 + 1e-9 for x in d_loss)
        maybe = sum(abs(x) <= 0.01 + 1e-9 for x in d_loss)
        ok_cnt = {f"{k} / 4" for k in range(sure, sure + maybe + 1)}
        checks = [(cells[1], ok_cnt, "改善格數"),
                  (cells[2], span(sum(d_loss) / 4, 0.01, 2, "pp"), "虧損平均變化"),
                  (cells[3], span(sum(d_win) / 4, 0.01, 2, "pp"), "獲利平均變化"),
                  (cells[4], span(sum(d_pf) / 4, 0.0001, 4, ""), "獲利因子變化")]
        for got, ok, lab in checks:
            if got not in ok:
                report(f"（八）彙總「{cells[0]}」{lab} {got} 不在 {sorted(ok)}")
                errs += 1
    return errs


def span(mid: float, delta: float, dec: int, suffix: str) -> set:
    """mid±delta 範圍內可能出現的顯示值（含兩端）。"""
    step = 10 ** -dec
    lo, hi = mid - delta, mid + delta
    out, x = set(), lo
    while x <= hi + step / 2:
        out.add(fmt_signed(min(x, hi), dec) + suffix)
        x += step / 2
    out.add(fmt_signed(hi, dec) + suffix)
    return out


def fmt_signed(x: float, dec: int) -> str:
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dec), ROUND_HALF_UP)
    s = f"{abs(v):.{dec}f}"
    return ("+" if v > 0 else "−" if v < 0 else "") + s


def fmt_pp(x: float) -> str:
    return fmt_signed(x, 2) + "pp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=common.result_dir("macd_strategy", "single_macd"))
    a = ap.parse_args()
    posts = os.path.join(common.require_blog_dir(), "site", "content", "posts")
    pool = load_pool(a.pool_dir)
    pf4 = {Decimal(repr(float(v))).quantize(Decimal("0.0001"), ROUND_HALF_UP)
           for v in pool["獲利因子"].dropna()}
    total = errs = edges = 0
    for slug in SLUGS:
        md = io.open(os.path.join(posts, slug + ".md"), encoding="utf-8").read()
        msgs = []
        tabs = list(tables(md))
        errs += check_colors(tabs, msgs.append)
        for heads, rows in tabs:
            cmap = col_map(heads)
            if cmap:
                for _, cells in rows:
                    hits, want = match_row(pool, cells, cmap)
                    total += len(want)
                    if not hits:
                        msgs.append(f"找不到對應列：{cells}")
                        errs += 1
                    elif all(e for _, e in hits):
                        edges += 1
            elif any("虧損平均改善" in h for h in heads):
                n = check_risk_summary(rows, a.pool_dir, msgs.append)
                total += 4 * len(rows)
                errs += n
            else:
                n = check_pf_grid(heads, rows, pf4, msgs.append)
                total += sum(len(c) for _, c in rows)
                errs += n
        print(f"[{'OK' if not msgs else 'NG'}] {slug}")
        for m in msgs:
            print("   X " + m)
    print(f"\n比對約 {total:,} 格｜錯誤 {errs}｜落在雙重進位邊界、放行的列 {edges}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
