# -*- coding: utf-8 -*-
"""
驗第十篇（macd-combo）的五張表與正文統計，全部對回 CSV。

文章裡的每一格都必須能在矩陣 CSV 或蒙地卡羅 CSV 找到來源，正文的統計句
（幾格高於基本版、排第幾名、範圍落在哪裡）也一起重算，避免憑印象寫的數字留在文章裡。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe \
        _04_analysis/macd/article/verify_article10.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import io
import re
import sys

import pandas as pd

DATA = os.path.join(common.require_blog_dir(), "reference", "macd", "data")
POST = os.path.join(common.require_blog_dir(), "site", "content", "posts", "macd-combo.md")
# 文章為了好讀在名稱裡加了空格，對回 CSV 前要還原
FILTER_MAP = {"創 250 日新高": "創250日新高", "均線多頭排列": "均線多頭排列",
              "ADX>25": "ADX>25", "收盤>MA200": "收盤>MA200",
              "RSI<50 且上升": "RSI<50且上升"}
EXIT_MAP = {"跌破年線": "跌破MA200", "超級趨勢": "Supertrend翻空",
            "抱滿 60 天": "抱滿60天", "跌破二十日低": "跌破20日低",
            "波段高點走低": "頂頂低"}
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%",
        "中位數%", "期望值/筆", "獲利因子", "總獲利(萬)", "未平倉%"]
BASE = {"交叉": 0.9869, "零軸": 1.2392, "背離": 1.0294}
POPS = {"黃金交叉": "交叉", "零軸上穿": "零軸", "純背離": "背離"}

errs = []


def fail(msg):
    errs.append(msg)
    print("  X " + msg)


def unescape(text):
    """表格裡的 < 要寫成 &lt;（否則瀏覽器把 <50 當標籤開頭），比對前還原。"""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def num(text):
    """文章用全形減號與千分位，還原成可比的數字。"""
    t = text.replace("−", "-").replace(",", "").replace("%", "").replace("+", "")
    return float(t.strip())


def cell_list(rest):
    return re.findall(r"<td[^>]*>(.*?)</td>", rest)


def main():
    mx = pd.read_csv(f"{DATA}/macd_combo_6x6.csv")
    # CSV 是 7×6 跑出來的，文章只用排序前五的兩軸，先裁掉落選的列
    mx = mx[mx["進場濾網"].isin(FILTER_MAP.values())
            & mx["出場"].isin(EXIT_MAP.values())]
    mc = pd.read_csv(f"{DATA}/macd_matrix_montecarlo.csv")
    text = io.open(POST, encoding="utf-8").read()
    # 跨基礎排行以後的表格欄位結構不同，先切開，避免被母體表的正則吃到
    body, rank_part = text.split("## 跨基礎排行", 1)
    checked = 0

    # ── 三張母體表 ──
    for sec in re.split(r"^### ", body, flags=re.M)[1:4]:
        pop = POPS[sec.split("\n", 1)[0].strip()]
        n_data = 0
        for filt, exit_, rest in re.findall(
                r"<tr><td>(.*?)</td><td>(.*?)</td>(.*?)</tr>", sec):
            filt, exit_ = unescape(filt), unescape(exit_)
            if filt not in FILTER_MAP or exit_ not in EXIT_MAP:
                continue                      # 合併說明列（創 250 日新高的 0 筆）
            r = mx[(mx["母體"] == pop) & (mx["進場濾網"] == FILTER_MAP[filt])
                   & (mx["出場"] == EXIT_MAP[exit_])]
            if not len(r):
                fail(f"{pop}/{filt}/{exit_}：CSV 查無此列")
                continue
            r = r.iloc[0]
            n_data += 1
            for col, cell in zip(COLS, cell_list(rest)):
                checked += 1
                got, want = num(cell), float(r[col])
                if col == "期望值/筆":
                    got, want = round(got), round(want)   # 文章這欄取整數
                tol = 0.011 if col != "交易次數" else 0.5
                if abs(got - want) > tol:
                    fail(f"{pop}/{filt}/{exit_} {col}: 文 {got} vs CSV {want}")
        print(f"  {pop} 表：{n_data} 列資料")
        want_rows = 20 if pop == "背離" else 25
        if n_data != want_rows:
            fail(f"{pop} 表列數 {n_data} ≠ {want_rows}")

    # ── 兩張排行表：三個基礎的獲利因子 ＋ 平均 ＋ 勝過基本版 ──
    rank_body, mc_part = rank_part.split("## 蒙地卡羅壓測", 1)
    a = mx[mx["交易次數"] > 0]
    n_rank = 0
    for filt, exit_, rest in re.findall(
            r"<tr><td>(.*?)</td><td>(.*?)</td>(.*?)</tr>", rank_body):
        filt, exit_ = unescape(filt), unescape(exit_)
        if filt not in FILTER_MAP or exit_ not in EXIT_MAP:
            continue
        cells = cell_list(rest)
        sub = a[(a["進場濾網"] == FILTER_MAP[filt]) & (a["出場"] == EXIT_MAP[exit_])]
        n_rank += 1
        pfs = []
        for pop, cell in zip(("交叉", "零軸", "背離"), cells[:3]):
            r = sub[sub["母體"] == pop]
            checked += 1
            if cell.strip() == "—":
                if len(r):
                    fail(f"排行 {filt}×{exit_} {pop}：文寫 0 筆但 CSV 有資料")
                continue
            if not len(r):
                fail(f"排行 {filt}×{exit_} {pop}：CSV 查無")
                continue
            pfs.append(float(r.iloc[0]["獲利因子"]))
            if abs(num(cell) - pfs[-1]) > 0.0001:
                fail(f"排行 {filt}×{exit_} {pop}: 文 {cell} vs CSV {pfs[-1]:.4f}")
        checked += 2
        if abs(num(cells[3]) - sum(pfs) / len(pfs)) > 0.0001:
            fail(f"排行 {filt}×{exit_} 平均: 文 {cells[3]}")
        won = sum(v > BASE[p] for v, p in zip(
            pfs, [p for p in ("交叉", "零軸", "背離") if len(sub[sub["母體"] == p])]))
        if cells[4].strip() != f"{won}/{len(pfs)}":
            fail(f"排行 {filt}×{exit_} 勝過基本版: 文 {cells[4]} vs 實際 {won}/{len(pfs)}")
    print(f"  排行表：{n_rank} 列（預期 25 ＝ 20 ＋ 5）")
    if n_rank != 25:
        fail(f"排行表列數 {n_rank} ≠ 25")

    # ── 蒙地卡羅表 ──
    mc_rows = re.findall(r"<tr><td>(.*?×.*?)</td>(.*?)</tr>", mc_part)
    print(f"  蒙地卡羅表：{len(mc_rows)} 列（預期 6）")
    if len(mc_rows) != 6:
        fail(f"蒙地卡羅列數 {len(mc_rows)} ≠ 6")
    MCCOL = ["交易數", None, "勝率%", "賺賠比", "報酬% P5", "報酬% 中位",
             "報酬% P95", "信賴區間寬度", "破產%", "最大連敗 P95", "最大回撤% P95"]
    for label, rest in mc_rows:
        key = unescape(label).replace(" ", "")
        m = mc[mc["組合"].str.replace(" ", "") == key]
        if not len(m):
            fail(f"蒙地卡羅查無組合：{label}")
            continue
        r = m.iloc[0]
        for col, cell in zip(MCCOL, cell_list(rest)):
            if col is None:                   # 抽樣模式是文字欄
                continue
            checked += 1
            got, want = num(cell), float(r[col])
            tol = 1.0 if col.startswith("報酬") or col == "信賴區間寬度" else 0.011
            if abs(got - want) > tol:
                fail(f"蒙地卡羅 {label} {col}: 文 {got} vs CSV {want}")

    # ── 正文的統計句 ──
    # 基本版與落選那一軸的數字仍要對（文章拿它們當比較基礎），所以另外讀一份未裁切的
    full = pd.read_csv(f"{DATA}/macd_combo_6x6.csv")
    d = mx[mx["交易次數"] > 0].copy()
    d["勝"] = d.apply(lambda r: r["獲利因子"] > BASE[r["母體"]], axis=1)
    g = (d.groupby(["進場濾網", "出場"])
          .agg(母體數=("母體", "count"), 平均=("獲利因子", "mean"))
          .reset_index().sort_values("平均", ascending=False))
    t3 = g[g["母體數"] == 3].reset_index(drop=True)

    def cell(pop, filt, exit_, col="獲利因子", df=None):
        df = full if df is None else df
        return float(df[(df["母體"] == pop) & (df["進場濾網"] == filt)
                        & (df["出場"] == exit_)].iloc[0][col])

    claims = [("全表格數＝75", len(mx), 75),
              ("有交易的格數＝70", len(d), 70),
              ("0 筆的格數＝5", int((mx["交易次數"] == 0).sum()), 5),
              ("三基礎都有交易的組合數＝20", len(t3), 20)]
    for pop, want in (("交叉", 25), ("零軸", 21), ("背離", 16)):
        claims.append((f"{pop} 高於基本版的格數＝{want}",
                       int(d[d["母體"] == pop]["勝"].sum()), want))
    for pop, want in (("交叉", 25), ("零軸", 25), ("背離", 20)):
        claims.append((f"{pop} 有交易的格數＝{want}", len(d[d["母體"] == pop]), want))
    for x, want in (("跌破MA200", 14), ("抱滿60天", 13), ("Supertrend翻空", 13),
                    ("頂頂低", 12), ("跌破20日低", 10)):
        claims.append((f"{x} 過關格數＝{want}/14",
                       (int(d[d["出場"] == x]["勝"].sum()), len(d[d["出場"] == x])),
                       (want, 14)))
    ma = d[d["出場"] == "跌破MA200"]
    claims.append(("跌破年線 三基礎平均＝1.5332", round(float(ma["獲利因子"].mean()), 4), 1.5332))
    claims.append(("跌破年線 平均持有天＝147.5", round(float(ma["平均持有天"].mean()), 1), 147.5))
    others = d[d["出場"] != "跌破MA200"].groupby("出場")["平均持有天"].mean()
    claims.append(("其餘四條出場的平均持有天＝44.6~89.0",
                   (round(float(others.min()), 1), round(float(others.max()), 1)),
                   (44.6, 89.0)))
    n_top = list(t3["出場"]).index("抱滿60天")
    claims.append(("排行前 4 名都是跌破年線", n_top, 4))
    claims.append(("第 5 名＝均線多頭排列×抱滿60天 1.2511",
                   (f"{t3.iloc[4]['進場濾網']}×{t3.iloc[4]['出場']}",
                    round(float(t3.iloc[4]["平均"]), 4)), ("均線多頭排列×抱滿60天", 1.2511)))
    for f, want in (("均線多頭排列", 1.5859), ("ADX>25", 1.4510),
                    ("收盤>MA200", 1.4348), ("RSI<50且上升", 1.4016)):
        claims.append((f"{f}×跌破年線 三基礎平均＝{want}", round(float(
            t3[(t3["進場濾網"] == f) & (t3["出場"] == "跌破MA200")]["平均"].iloc[0]), 4), want))
    nof = full[(full["進場濾網"] == "無濾網") & (full["出場"] == "跌破MA200")]
    claims.append(("無濾網×跌破年線 三基礎平均＝1.4673",
                   round(float(nof["獲利因子"].mean()), 4), 1.4673))
    claims.append(("交叉 無濾網×跌破年線＝1.5088",
                   round(cell("交叉", "無濾網", "跌破MA200"), 4), 1.5088))
    claims.append(("交叉 基本版交易次數＝139,746",
                   int(cell("交叉", "無濾網", "原生出場", "交易次數")), 139746))
    claims.append(("交叉 基本版持有天＝18.65",
                   round(cell("交叉", "無濾網", "原生出場", "平均持有天"), 2), 18.65))
    claims.append(("交叉 無濾網×跌破年線 交易次數＝40,936",
                   int(cell("交叉", "無濾網", "跌破MA200", "交易次數")), 40936))
    claims.append(("交叉 無濾網×跌破年線 持有天＝130.46",
                   round(cell("交叉", "無濾網", "跌破MA200", "平均持有天"), 2), 130.46))
    for pop, lo, hi in (("交叉", 1.0763, 1.6615), ("零軸", 1.1580, 2.1834),
                        ("背離", 0.9246, 1.6304)):
        x = d[d["母體"] == pop]
        claims.append((f"{pop} 獲利因子範圍＝{lo}~{hi}",
                       (round(float(x["獲利因子"].min()), 4),
                        round(float(x["獲利因子"].max()), 4)), (lo, hi)))
    x = d[(d["母體"] == "交叉") & (d["出場"] == "跌破MA200")]
    claims.append(("交叉 固定跌破年線 五格範圍＝1.4435~1.6615",
                   (round(float(x["獲利因子"].min()), 4),
                    round(float(x["獲利因子"].max()), 4)), (1.4435, 1.6615)))
    claims.append(("交叉 最低那格＝RSI<50且上升×抱滿60天",
                   f"{d[d['母體'] == '交叉'].nsmallest(1, '獲利因子').iloc[0]['進場濾網']}"
                   f"×{d[d['母體'] == '交叉'].nsmallest(1, '獲利因子').iloc[0]['出場']}",
                   "RSI<50且上升×抱滿60天"))
    bad_zero = d[(d["母體"] == "零軸") & (~d["勝"])]
    claims.append(("零軸沒過的 4 格交易次數都在 7~9 千",
                   (len(bad_zero), int(bad_zero["交易次數"].min()),
                    int(bad_zero["交易次數"].max())), (4, 7208, 8199)))
    bad_div = d[(d["母體"] == "背離") & (~d["勝"])]
    claims.append(("背離沒過的 4 格有 3 格是跌破二十日低",
                   (len(bad_div), int((bad_div["出場"] == "跌破20日低").sum())), (4, 3)))
    claims.append(("未平倉率最大值＝3.83", round(float(d["未平倉%"].max()), 2), 3.83))
    r = mx[(mx["母體"] == "零軸") & (mx["進場濾網"] == "創250日新高")
           & (mx["出場"] == "跌破MA200")].iloc[0]
    claims.append(("全表最高 2.1834／635 筆",
                   (round(float(r["獲利因子"]), 4), int(r["交易次數"])), (2.1834, 635)))
    claims.append(("零軸基本版交易次數＝64,240",
                   int(cell("零軸", "無濾網", "原生出場", "交易次數")), 64240))
    r = mx[(mx["母體"] == "背離") & (mx["進場濾網"] == "均線多頭排列")
           & (mx["出場"] == "跌破MA200")].iloc[0]
    claims.append(("背離最佳 1.6304／2,468 筆",
                   (round(float(r["獲利因子"]), 4), int(r["交易次數"])), (1.6304, 2468)))
    m = {r["組合"]: r for _, r in mc.iterrows()}
    claims.append(("MC 小樣本那組平均獲利%＝51.29",
                   round(float(m["零軸 × 創250日新高 × 跌破年線"]["平均獲利%"]), 2), 51.29))
    claims.append(("MC 連敗 P95 範圍＝16~43",
                   (int(mc["最大連敗 P95"].min()), int(mc["最大連敗 P95"].max())), (16, 43)))
    claims.append(("MC 連敗最長那組＝零軸 × 收盤>MA200 × 跌破年線",
                   mc.loc[mc["最大連敗 P95"].idxmax(), "組合"],
                   "零軸 × 收盤>MA200 × 跌破年線"))
    claims.append(("該組勝率 23.48／賺賠比 5.51",
                   (round(float(m["零軸 × 收盤>MA200 × 跌破年線"]["勝率%"]), 2),
                    round(float(m["零軸 × 收盤>MA200 × 跌破年線"]["賺賠比"]), 2)),
                   (23.48, 5.51)))
    claims.append(("連敗最短那組 勝率 50.96／賺賠比 1.51",
                   (round(float(m["背離 × ADX>25 × 跌破年線"]["勝率%"]), 2),
                    round(float(m["背離 × ADX>25 × 跌破年線"]["賺賠比"]), 2)), (50.96, 1.51)))
    claims.append(("MC 回撤 P95 範圍＝4.4~7.8",
                   (round(float(mc["最大回撤% P95"].min()), 1),
                    round(float(mc["最大回撤% P95"].max()), 1)), (4.4, 7.8)))

    print("  正文統計：")
    for name, got, want in claims:
        checked += 1
        ok = got == want
        if not ok:
            fail(f"{name}：實際 {got}")
        print(f"    {'OK' if ok else 'NG'} {name}（實際 {got}）")

    print(f"\n對照 {checked} 個數值，錯誤 {len(errs)} 個")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
