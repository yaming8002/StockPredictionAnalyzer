# -*- coding: utf-8 -*-
"""
資金控管 觀念圖（〈蒙地卡羅〉§四 資金分配用）。兩塊：
  左：資金底線（90%→50%）→ 本金分成幾份（份數 = 1 / 每筆風險 r，r = 1−底線^(1/S)）
  右：份數 → 中位最終金額（萬，起始100萬；蒙地卡羅一萬條）
沿用假想策略：勝率35%、賺賠比2:1、最壞連敗 S=15。底線 80% ＝ 允許最大虧損 20%。
用「中位」不用算術平均（複利下算術平均被樂透路徑灌爆、會誤導成越激進越高）。
依 chart-minimal 規則，圖內不放標題/結論。
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

P, WIN, LOSS, S = 0.35, 2.0, 1.0, 15
START = 100.0
ACCENT, ACCENT2 = "#4c78a8", "#e45756"

rng = np.random.default_rng(20260721)
N_PATH, N_TRADE = 10000, 500
wins = rng.random((N_PATH, N_TRADE)) < P
floors = np.arange(0.90, 0.4999, -0.05)     # 底線 90%→50%
parts, med = [], []
for fl in floors:
    r = 1 - fl ** (1 / S)
    m = r
    parts.append(1 / m)
    eq = np.cumprod(np.where(wins, 1 + m * WIN, 1 - m * LOSS), axis=1)[:, -1]
    med.append(np.median(eq) * START)
parts, med = np.array(parts), np.array(med)

fig = plt.figure(figsize=(11, 4.4))
gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.92, bottom=0.15, wspace=0.28)

# 左：底線 → 份數
axL = fig.add_subplot(gs[0, 0])
axL.plot(floors * 100, parts, "-o", color=ACCENT, lw=2.2, zorder=3)
axL.invert_xaxis()                            # 90% 在左、50% 在右
axL.set_xlabel("資金底線（%，越低＝允許虧越多）", fontproperties=_fp, fontsize=11)
axL.set_ylabel("本金分成幾份", fontproperties=_fp, fontsize=11)
axL.set_xticks(floors * 100)
for lb in axL.get_xticklabels() + axL.get_yticklabels():
    lb.set_fontproperties(_fp); lb.set_fontsize(9)
axL.grid(alpha=0.25)

# 右：份數 → 中位最終金額
axR = fig.add_subplot(gs[0, 1])
axR.plot(parts, med, "-o", color=ACCENT2, lw=2.2, zorder=3)
axR.axhline(START, color="#bbb", lw=1, ls=":", zorder=0)
axR.set_xlabel("本金分成幾份", fontproperties=_fp, fontsize=11)
axR.set_ylabel("中位最終金額（萬，起始 100 萬）", fontproperties=_fp, fontsize=11)
for lb in axR.get_xticklabels() + axR.get_yticklabels():
    lb.set_fontproperties(_fp); lb.set_fontsize(9)
axR.grid(alpha=0.25)
axR.margins(y=0.12)

p = os.path.join(OUT, "floor_parts_wealth.png")
fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)

print("底線%:", [f"{f*100:.0f}" for f in floors])
print("份數:", [round(v, 1) for v in parts])
print("中位最終(萬):", [round(v, 1) for v in med])
print("OK ->", p)
