# -*- coding: utf-8 -*-
"""
產第十篇的三張母體表、跨母體排行表與正文統計，全部由 CSV 生成、不手抄。

輸出到 stdout，用標記分段（TABLE:交叉 / TABLE:零軸 / TABLE:背離 / RANK3 / RANK2 / STATS）。

執行：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python \
        _04_analysis/macd/article/build_article10_tables.py
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import sys

import pandas as pd

CSV = os.path.join(common.require_blog_dir(), "reference", "macd", "data", "macd_combo_6x6.csv")
# 兩軸都只取「原生出場下的三母體平均」排序的前五名，依排名擺列。
# 無濾網與原本的出場不佔格子——它們是基本版，數字沿用既有資料、寫在圖例裡當比較基礎。
FILTER_ORDER = ["創250日新高", "均線多頭排列", "ADX>25", "收盤>MA200",
                "RSI<50且上升"]
EXIT_ORDER = ["跌破MA200", "Supertrend翻空", "抱滿60天", "跌破20日低", "頂頂低"]
# CSV 名 → 文章顯示名（跟（七）（八）（九）篇一致）
SHOW_F = {"無濾網": "無濾網", "收盤>MA200": "收盤&gt;MA200", "跳空": "跳空",
          "均線多頭排列": "均線多頭排列", "ADX>25": "ADX&gt;25",
          "RSI<50且上升": "RSI&lt;50 且上升", "創250日新高": "創 250 日新高"}
SHOW_X = {"原生出場": "原本的出場", "跌破MA200": "跌破年線",
          "Supertrend翻空": "超級趨勢", "抱滿60天": "抱滿 60 天",
          "跌破20日低": "跌破二十日低", "頂頂低": "波段高點走低"}
COLS = ["交易次數", "勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%",
        "期望值/筆", "獲利因子", "總獲利(萬)", "未平倉%"]


def fmt(col, v):
    """負數用全形減號，千分位與小數位數比照（九）篇。"""
    if col == "交易次數":
        t = f"{int(round(v)):,}"
    elif col == "期望值/筆":
        t = f"{v:,.0f}"
    elif col == "獲利因子":
        t = f"{v:.4f}"
    elif col == "總獲利(萬)":
        t = f"{v:,.1f}"                       # 比照（七）（八）篇：總獲利一位小數
    elif col in ("勝率%", "平均持有天", "獲利平均%", "虧損平均%", "中位數%",
                 "未平倉%"):
        t = f"{v:,.2f}"
    else:
        t = str(v)
    return t.replace("-", "−")


def main():
    d = pd.read_csv(CSV)
    # 基本版（無濾網 × 原生出場）先取出來當比較基礎，再把矩陣裁到選中的兩軸；
    # 沒裁的話排行表與統計會把落選的濾網／出場一起算進去。
    base = {}
    for pop in ("交叉", "零軸", "背離"):
        r = d[(d["母體"] == pop) & (d["進場濾網"] == "無濾網")
              & (d["出場"] == "原生出場")].iloc[0]
        base[pop] = r["獲利因子"]
    d = d[d["進場濾網"].isin(FILTER_ORDER) & d["出場"].isin(EXIT_ORDER)]

    for pop in ("交叉", "零軸", "背離"):
        print(f"### TABLE:{pop}（基準線 {base[pop]:.4f}）")
        for f in FILTER_ORDER:
            sub = d[(d["母體"] == pop) & (d["進場濾網"] == f)]
            if (sub["交易次數"] == 0).all():
                print(f'<tr><td>{SHOW_F[f]}</td>'
                      f'<td colspan="11">五個出場全部 0 筆（條件互斥）</td></tr>')
                continue
            for x in EXIT_ORDER:
                r = sub[sub["出場"] == x].iloc[0]
                cells = []
                for c in COLS:
                    cls = ""
                    if c == "獲利因子":
                        cls = ' class="up"' if r[c] > base[pop] else ' class="down"'
                    cells.append(f"<td{cls}>{fmt(c, r[c])}</td>")
                print(f"<tr><td>{SHOW_F[f]}</td><td>{SHOW_X[x]}</td>"
                      + "".join(cells) + "</tr>")
        print()

    a = d[d["交易次數"] > 0].copy()
    a["勝基準線"] = a.apply(lambda r: r["獲利因子"] > base[r["母體"]], axis=1)
    g = (a.groupby(["進場濾網", "出場"])
          .agg(母體數=("母體", "count"), 平均=("獲利因子", "mean"),
               勝=("勝基準線", "sum"))
          .reset_index().sort_values("平均", ascending=False))
    for tag, n in (("RANK3", 3), ("RANK2", 2)):
        print(f"### {tag}")
        for _, r in g[g["母體數"] == n].iterrows():
            pf = {p: a[(a["進場濾網"] == r["進場濾網"]) & (a["出場"] == r["出場"])
                       & (a["母體"] == p)] for p in ("交叉", "零軸", "背離")}
            cells = "".join(f"<td>{v.iloc[0]['獲利因子']:.4f}</td>" if len(v)
                            else "<td>—</td>" for v in pf.values())
            print(f"<tr><td>{SHOW_F[r['進場濾網']]}</td><td>{SHOW_X[r['出場']]}</td>"
                  f"{cells}<td>{r['平均']:.4f}</td><td>{int(r['勝'])}/{n}</td></tr>")
        print()

    print("### STATS")
    t3 = g[g["母體數"] == 3].reset_index(drop=True)
    print(f"格數：{len(d)}（5×5×3）；有交易 {len(a)}；0 筆 {int((d['交易次數'] == 0).sum())}")
    print(f"三基礎都有交易的組合數：{len(t3)}")
    ma = t3[t3["出場"] == "跌破MA200"]
    print(f"跌破年線佔排行前 {len(ma)} 名；第 {len(ma) + 1} 名＝"
          f"{t3.iloc[len(ma)]['進場濾網']}×{t3.iloc[len(ma)]['出場']} "
          f"{t3.iloc[len(ma)]['平均']:.4f}")
    for pop in ("交叉", "零軸", "背離"):
        x = a[a["母體"] == pop]
        print(f"  {pop}：有交易 {len(x)} 格、勝基本版 {int(x['勝基準線'].sum())} 格、"
              f"獲利因子 {x['獲利因子'].min():.4f}~{x['獲利因子'].max():.4f}")
    print("各出場 勝/總、平均、平均持有天：")
    for x in EXIT_ORDER:
        s_ = a[a["出場"] == x]
        print(f"  {SHOW_X[x]}：{int(s_['勝基準線'].sum())}/{len(s_)}、"
              f"{s_['獲利因子'].mean():.4f}、{s_['平均持有天'].mean():.1f} 天")
    print("各進場 勝/總、平均：")
    for f in FILTER_ORDER:
        s_ = a[a["進場濾網"] == f]
        print(f"  {SHOW_F[f]}：{int(s_['勝基準線'].sum())}/{len(s_)}、"
              f"{s_['獲利因子'].mean():.4f}")
    print(f"未平倉% 最大：{a['未平倉%'].max():.2f}"
          f"（{a.loc[a['未平倉%'].idxmax(), '母體']}×"
          f"{a.loc[a['未平倉%'].idxmax(), '進場濾網']}×"
          f"{a.loc[a['未平倉%'].idxmax(), '出場']}）")
    for pop in ("交叉", "零軸", "背離"):
        x = a[a["母體"] == pop].nlargest(3, "獲利因子")
        best = "、".join(f"{r['進場濾網']}×{SHOW_X[r['出場']]} {r['獲利因子']:.4f}"
                        f"（{int(r['交易次數']):,} 筆）" for _, r in x.iterrows())
        print(f"{pop} 最佳三組：{best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
