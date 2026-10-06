# -*- coding: utf-8 -*-
"""指標字典（backtest-metrics-guide）ELI5 範例圖：
- metric_winrate.png：勝率高 不一定 會賺。A、B 各跑 1,000 筆 × 隨機 1,000 次，
  並排比較「結果區間」（P5～P95 + 中位）。
- metric_median.png ：中位數 vs 平均（多數小賠、少數大賺的肥尾）。
全部合成/假想數字，純觀念示意。說明一律放文章正文，圖內不放。"""

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

OUT = common.CHART_DIR
os.makedirs(OUT, exist_ok=True)
_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False
INK = "#33414f"; WINC = "#2ca02c"; LOSSC = "#e45756"; BLUE = "#4c78a8"


# ══════════ 圖一：勝率高 不一定 會賺 —— 結果區間比較 ══════════
N, M, CAP, BET = 1000, 1000, 1_000_000, 10_000


def sim_paths(win_rate, win_pct, loss_pct, seed):
    """M 條路徑、每條 N 筆、固定投入 1 萬非複利，回傳權益矩陣（萬元，含起點）。"""
    r = np.random.default_rng(seed)
    wins = r.random((M, N)) < win_rate
    pnl = np.where(wins, BET * win_pct, -BET * loss_pct)
    eq = CAP + np.cumsum(pnl, axis=1)
    return np.column_stack([np.full(M, CAP), eq]) / 10_000.0


def band(paths):
    return (np.percentile(paths, 5, axis=0),
            np.percentile(paths, 50, axis=0),
            np.percentile(paths, 95, axis=0))


pathsA = sim_paths(0.65, 0.05, 0.10, 101)   # 勝率65%、賺5%賠10%
pathsB = sim_paths(0.35, 0.10, 0.05, 102)   # 勝率35%、賺10%賠5%
xs = np.arange(N + 1)
a5, a50, a95 = band(pathsA)
b5, b50, b95 = band(pathsB)
ylo = min(a5.min(), b5.min()) - 1
yhi = max(a95.max(), b95.max()) + 1

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, (p5, p50, p95), col, ttl in [
        (axA, (a5, a50, a95), LOSSC, "策略 A：勝率 65%"),
        (axB, (b5, b50, b95), WINC, "策略 B：勝率 35%")]:
    ax.fill_between(xs, p5, p95, color=col, alpha=0.20, zorder=2, label="P5～P95")
    ax.plot(xs, p50, color=col, lw=2, zorder=3, label="中位")
    ax.axhline(100, color="#888", lw=1, ls="--")
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("交易筆數", fontproperties=_fp, fontsize=11)
    ax.set_title(ttl, fontproperties=_fp, fontsize=13, pad=8)
    ax.grid(alpha=0.22)
    ax.legend(prop=_fp, fontsize=10, loc="upper left", frameon=False)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontproperties(_fp); lb.set_fontsize(9)
axA.set_ylabel("帳戶權益（萬元，起始 100）", fontproperties=_fp, fontsize=11)
fig.tight_layout()
p1 = os.path.join(OUT, "metric_winrate.png")
fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)


# ══════════ 圖二：中位數 vs 平均（肥尾）══════════
rng = np.random.default_rng(7)
small = rng.normal(-4, 4, 800)      # 多數：小賠/小賺
big = rng.normal(40, 35, 200)       # 少數：大賺
rets = np.concatenate([small, big])
med, mean = np.median(rets), rets.mean()
disp = np.clip(rets, -30, 140)

fig2, ax2 = plt.subplots(figsize=(11, 5.4))
ax2.hist(disp, bins=np.linspace(-30, 140, 60), color=BLUE, alpha=0.75,
         edgecolor="white", linewidth=0.4, zorder=3)
ymax = ax2.get_ylim()[1]
ax2.axvline(med, color=INK, lw=2, ls="--", zorder=5)
ax2.axvline(mean, color=WINC, lw=2.2, zorder=5)
ax2.annotate(f"中位數 {med:.0f}%\n最典型的一筆，是負的",
             xy=(med, ymax * 0.6), xytext=(-28, ymax * 0.78),
             fontproperties=_fp, fontsize=11.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK))
ax2.annotate(f"平均 +{mean:.0f}%\n被少數大賺拉成正的",
             xy=(mean, ymax * 0.85), xytext=(mean + 18, ymax * 0.9),
             fontproperties=_fp, fontsize=11.5, color=WINC,
             arrowprops=dict(arrowstyle="->", color=WINC))
ax2.annotate("少數幾筆大賺（肥尾）→", xy=(105, ymax * 0.16),
             ha="center", fontproperties=_fp, fontsize=11, color="#9a6a00")
ax2.set_xlabel("每筆交易的報酬率（%）", fontproperties=_fp, fontsize=12)
ax2.set_ylabel("有幾筆交易", fontproperties=_fp, fontsize=12)
for lb in ax2.get_xticklabels() + ax2.get_yticklabels():
    lb.set_fontproperties(_fp); lb.set_fontsize(9)
ax2.set_title("多數小賠、少數大賺：中位數 vs 平均", fontproperties=_fp, fontsize=16, pad=12)
ax2.grid(axis="y", alpha=0.25)
p2 = os.path.join(OUT, "metric_median.png")
fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2)

print(f"A: P5/中位/P95 = {a5[-1]:.1f}/{a50[-1]:.1f}/{a95[-1]:.1f} 萬")
print(f"B: P5/中位/P95 = {b5[-1]:.1f}/{b50[-1]:.1f}/{b95[-1]:.1f} 萬")
print(f"中位數={med:.1f} 平均={mean:.1f}")
