# -*- coding: utf-8 -*-
"""
等額投入 觀念兩張圖（〈買零股還是整張〉用）。
圖一 equal_value_sizing.png：固定金額下，零股 vs 整張 損益等效。
圖二 price_vs_pct_illusion.png：「價差」會騙人、「漲跌幅」才算數；同漲幅→同獲利。
- 全部合成/假設數字，未計費稅，純觀念示意。
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = common.CHART_DIR
os.makedirs(OUT, exist_ok=True)
_fp = font_manager.FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
plt.rcParams["axes.unicode_minus"] = False

LOW = "#4c78a8"     # 低價·整張
HIGH = "#f58518"    # 高價·零股
WINC = "#2ca02c"
LOSSC = "#e45756"
INK = "#33414f"


# ══════════ 圖一：等額投入，零股 vs 整張 損益等效 ══════════
CAP, RATIO = 1_000_000, 1 / 20
INVEST = CAP * RATIO                       # 50,000
pa, pb = 50.0, 200.0
sa, sb = int(INVEST // pa), int(INVEST // pb)   # 1000股(1張) / 250股(零股)
up, dn = 0.08, -0.05
pnl = lambda price, shares, r: shares * (price * (1 + r) - price)
a_up, b_up = pnl(pa, sa, up), pnl(pb, sb, up)
a_dn, b_dn = pnl(pa, sa, dn), pnl(pb, sb, dn)

fig1 = plt.figure(figsize=(12, 5.4))
gs = fig1.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.04, right=0.975,
                       top=0.80, bottom=0.12, wspace=0.16)

axT = fig1.add_subplot(gs[0, 0]); axT.axis("off")
rows = [
    ["", "低價股　50 元", "高價股　200 元"],
    ["投入金額（1/20 × 100 萬）", f"{INVEST:,.0f}", f"{INVEST:,.0f}"],
    ["買進股數", f"{sa:,} 股（1 張）", f"{sb:,} 股（零股）"],
    ["漲 8% → 股價", f"{pa*(1+up):.1f}", f"{pb*(1+up):.1f}"],
    ["　　　　損益", f"+{a_up:,.0f}", f"+{b_up:,.0f}"],
    ["跌 5% → 股價", f"{pa*(1+dn):.1f}", f"{pb*(1+dn):.1f}"],
    ["　　　　損益", f"-{abs(a_dn):,.0f}", f"-{abs(b_dn):,.0f}"],
]
tbl = axT.table(cellText=rows, cellLoc="center", loc="center",
                colWidths=[0.46, 0.27, 0.27])
tbl.auto_set_font_size(False); tbl.scale(1, 1.9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_text_props(fontproperties=_fp, fontsize=11); cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor(INK); cell.set_text_props(fontproperties=_fp, fontsize=11.5, color="white")
    elif c == 0:
        cell.set_facecolor("#f3f4f6"); cell.set_text_props(fontproperties=_fp, fontsize=10.5, color="#333")
    if r in (4, 7) and c >= 1:
        cell.set_facecolor("#eaf4ea" if r == 4 else "#fdeceb")
        cell.set_text_props(fontproperties=_fp, fontsize=12.5, color=WINC if r == 4 else LOSSC)
axT.set_title("同樣投入 5 萬：零股 vs 整張，損益一模一樣",
              fontproperties=_fp, fontsize=13, pad=10)

axB = fig1.add_subplot(gs[0, 1])
x = np.arange(2); w = 0.36
b1 = axB.bar(x - w/2, [a_up, a_dn], w, label="低價 50 元 · 1 張", color=LOW, zorder=3)
b2 = axB.bar(x + w/2, [b_up, b_dn], w, label="高價 200 元 · 250 股(零股)", color=HIGH, zorder=3)
axB.axhline(0, color="#888", lw=1)
axB.set_xticks(x); axB.set_xticklabels(["漲 8%", "跌 5%"], fontproperties=_fp, fontsize=11)
axB.set_ylabel("損益（元）", fontproperties=_fp, fontsize=11)
for lb in axB.get_yticklabels(): lb.set_fontproperties(_fp); lb.set_fontsize(9)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        axB.annotate(f"{'+' if h >= 0 else '-'}{abs(h):,.0f}",
                     (rect.get_x() + rect.get_width()/2, h), textcoords="offset points",
                     xytext=(0, 6 if h >= 0 else -14), ha="center",
                     fontproperties=_fp, fontsize=10, color=WINC if h >= 0 else LOSSC)
axB.legend(prop=_fp, fontsize=9.5, loc="upper right", frameon=False)
axB.set_title("兩檔損益完全等高 → 等效", fontproperties=_fp, fontsize=13, pad=10)
axB.grid(axis="y", alpha=0.25); axB.margins(y=0.22)

fig1.text(0.5, 0.93, "用固定金額下單時，買零股還是買整張，結果完全一樣",
          ha="center", fontproperties=_fp, fontsize=15)
fig1.text(0.5, 0.025,
          "核心：每筆損益 ＝ 投入金額 × 報酬率(%)　——　與股價高低、零股或整張都無關。",
          ha="center", fontproperties=_fp, fontsize=12, color="#222")
p1 = os.path.join(OUT, "equal_value_sizing.png")
fig1.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig1)


# ══════════ 圖二：價差會騙人、漲跌幅才算數；同漲幅→同獲利 ══════════
fig2 = plt.figure(figsize=(12, 5.2))
gs2 = fig2.add_gridspec(1, 2, left=0.07, right=0.975, top=0.80, bottom=0.18, wspace=0.24)

# 左：同樣「漲 10 元」，漲幅天差地遠
axA = fig2.add_subplot(gs2[0, 0])
pct = [10/20*100, 10/200*100]            # +50% vs +5%
bars = axA.bar(["20 元\n(漲到 30)", "200 元\n(漲到 210)"], pct,
               color=[LOW, HIGH], width=0.5, zorder=3)
for rect, v, note in zip(bars, pct, ["約需 5 根漲停", "可能 1 天就到"]):
    axA.annotate(f"+{v:.0f}%", (rect.get_x()+rect.get_width()/2, v),
                 textcoords="offset points", xytext=(0, 6), ha="center",
                 fontproperties=_fp, fontsize=13, color=INK)
    axA.annotate(note, (rect.get_x()+rect.get_width()/2, v/2),
                 ha="center", va="center", fontproperties=_fp, fontsize=10, color="white")
axA.set_ylabel("實際漲幅（%）", fontproperties=_fp, fontsize=11)
for lb in axA.get_xticklabels(): lb.set_fontproperties(_fp); lb.set_fontsize(10.5)
for lb in axA.get_yticklabels(): lb.set_fontproperties(_fp); lb.set_fontsize(9)
axA.set_title("同樣漲 10 元，漲幅天差地遠", fontproperties=_fp, fontsize=13, pad=10)
axA.grid(axis="y", alpha=0.25); axA.margins(y=0.20)

# 右：等成本 20 萬、同樣 +50% → 獲利相同
axC = fig2.add_subplot(gs2[0, 1])
cost = 200_000
g_low = int(cost // 20) * (30 - 20)       # 20元10張漲到30 = +100,000
g_high = int(cost // 200) * (300 - 200)   # 200元1張漲到300 = +100,000
bars2 = axC.bar(["20 元 × 10 張\n漲到 30 (+50%)", "200 元 × 1 張\n漲到 300 (+50%)"],
                [g_low, g_high], color=[LOW, HIGH], width=0.5, zorder=3)
for rect, v in zip(bars2, [g_low, g_high]):
    axC.annotate(f"+{v:,.0f}", (rect.get_x()+rect.get_width()/2, v),
                 textcoords="offset points", xytext=(0, 6), ha="center",
                 fontproperties=_fp, fontsize=13, color=WINC)
axC.set_ylabel("獲利（元）", fontproperties=_fp, fontsize=11)
for lb in axC.get_xticklabels(): lb.set_fontproperties(_fp); lb.set_fontsize(10.5)
for lb in axC.get_yticklabels(): lb.set_fontproperties(_fp); lb.set_fontsize(9)
axC.set_title("只要漲幅相同，獲利就相同（等成本 20 萬）", fontproperties=_fp, fontsize=13, pad=10)
axC.grid(axis="y", alpha=0.25); axC.margins(y=0.20)

fig2.text(0.5, 0.93, "「價差」會騙人，「漲跌幅」才算數",
          ha="center", fontproperties=_fp, fontsize=15)
fig2.text(0.5, 0.04,
          "低價股「漲一塊賺比較多」是錯覺——那是因為漲幅%比較大。換算成同樣的漲幅，價格高低、張數多寡都不影響獲利。",
          ha="center", fontproperties=_fp, fontsize=11, color="#666")
p2 = os.path.join(OUT, "price_vs_pct_illusion.png")
fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2)

print("OK ->", p1)
print("OK ->", p2)
print(f"圖一 投入={INVEST:.0f} 漲8%:{a_up:.0f}/{b_up:.0f} 跌5%:{a_dn:.0f}/{b_dn:.0f}")
print(f"圖二 漲幅%:{pct[0]:.0f}/{pct[1]:.0f} 等成本同漲幅獲利:{g_low:,}/{g_high:,}")
