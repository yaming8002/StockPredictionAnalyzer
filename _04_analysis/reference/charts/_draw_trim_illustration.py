# -*- coding: utf-8 -*-
"""
去極值（MAD，中位數 ± 3×MAD）示意圖——字典〈回測統計指標說明〉用。
- 合成數據（非特定策略），示範「賺單、賠單各自畫界、界外剔掉、重算平均」。
- 兩面板：賺單（右偏肥尾）/ 賠單（左偏肥尾）；標中位數、界線、保留/剔除、平均前後。
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

OUT = common.CHART_DIR   # 草稿用；發佈時再 copy 到 content/charts/metrics/
os.makedirs(OUT, exist_ok=True)
_fp = font_manager.FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260619)
KEEP = "#4c78a8"   # 保留（界內）
TRIM = "#e45756"   # 剔除（極端值）
MEDC = "#444"      # 中位數
BNDC = "#d62728"   # 界線

# 合成：賺單（多數小賺、少數極端大賺）；賠單（多數小賠、少數極端大賠）
wins = np.concatenate([rng.gamma(2.0, 2200, 460), rng.uniform(28000, 95000, 12)])
loss = -np.concatenate([rng.gamma(2.0, 1100, 460), rng.uniform(14000, 42000, 12)])


def panel(ax, data, title, side):
    """side='win' 剔上界；'loss' 剔下界。"""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    lower, upper = med - 3 * mad, med + 3 * mad
    if side == "win":
        kept = data[data <= upper]; cut = upper; cut_lbl = "上界線"
        trimmed_mask = lambda c: c > upper
    else:
        kept = data[data >= lower]; cut = lower; cut_lbl = "下界線"
        trimmed_mask = lambda c: c < lower

    bins = np.linspace(data.min(), data.max(), 46)
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [TRIM if trimmed_mask(c) else KEEP for c in centers]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=colors, zorder=2)

    ax.axvline(med, color=MEDC, lw=1.4, zorder=3)
    ax.axvline(cut, color=BNDC, lw=1.4, ls="--", zorder=3)
    ax.axvline(data.mean(), color="#888", lw=1.2, ls=":", zorder=3)
    ax.axvline(kept.mean(), color="#2ca02c", lw=1.2, ls=":", zorder=3)

    top = counts.max()
    ax.annotate("中位數", (med, top * 0.98), fontproperties=_fp, fontsize=9,
                color=MEDC, ha="center", va="top")
    ax.annotate(cut_lbl, (cut, top * 0.62), fontproperties=_fp, fontsize=9,
                color=BNDC, ha="center", va="center", rotation=90)
    # 標出被剔除的肥尾
    if side == "win":
        ax.annotate("剔除：極端大賺", (upper, top * 0.30), (upper, top * 0.30),
                    fontproperties=_fp, fontsize=9, color=TRIM, ha="left", va="center")
    else:
        ax.annotate("剔除：極端大賠", (lower, top * 0.30),
                    fontproperties=_fp, fontsize=9, color=TRIM, ha="right", va="center")

    ax.set_title(title, fontproperties=_fp, fontsize=12)
    ax.set_xlabel("每筆淨損益（元）", fontproperties=_fp, fontsize=10)
    ax.set_ylabel("筆數", fontproperties=_fp, fontsize=10)
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_fontproperties(_fp); lb.set_fontsize(8)
    ax.grid(axis="y", alpha=0.25)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
panel(axL, wins, "賺單：界外的「極端大賺」被剔掉", "win")
panel(axR, loss, "賠單：界外的「極端大賠」被剔掉", "loss")

# 共用圖例
handles = [
    plt.Rectangle((0, 0), 1, 1, color=KEEP),
    plt.Rectangle((0, 0), 1, 1, color=TRIM),
    plt.Line2D([0], [0], color=MEDC, lw=1.4),
    plt.Line2D([0], [0], color=BNDC, lw=1.4, ls="--"),
    plt.Line2D([0], [0], color="#888", lw=1.2, ls=":"),
    plt.Line2D([0], [0], color="#2ca02c", lw=1.2, ls=":"),
]
labels = ["保留（界內）", "剔除（極端值）", "中位數", "界線（±3 倍中位數絕對離差）",
          "原始平均", "去極值後平均"]
fig.legend(handles, labels, prop=_fp, fontsize=9, ncol=6,
           loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.text(0.5, 0.93, "去極值：賺、賠各自以「中位數 ± 3 倍中位數絕對離差」畫界，界外的極端單剔掉後再重算平均",
         ha="center", fontproperties=_fp, fontsize=12)
fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.20, wspace=0.18)
path = os.path.join(OUT, "trim_mad_illustration.png")
fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close(fig)
print("OK ->", path)
print(f"賺單 原始平均={wins.mean():.0f} 去極值後={wins[wins<=np.median(wins)+3*np.median(np.abs(wins-np.median(wins)))].mean():.0f}")
print(f"賠單 原始平均={loss.mean():.0f} 去極值後={loss[loss>=np.median(loss)-3*np.median(np.abs(loss-np.median(loss)))].mean():.0f}")
