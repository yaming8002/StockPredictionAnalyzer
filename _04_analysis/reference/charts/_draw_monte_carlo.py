# -*- coding: utf-8 -*-
"""
蒙地卡羅文章用圖（〈蒙地卡羅：你會連賠幾次？會不會中途陣亡？〉）。
真的跑 10,000 次模擬，全部合成/假設策略，純觀念示意、未計費稅。

假設策略：勝率 35%、賺賠比 2:1（贏 +2、輸 −1），一生 1,000 筆、帳戶 100 萬。
下注：每筆固定拿「本金的 1%（1 萬）」去冒險 → 贏 +2 萬、輸 −1 萬。

產出：
- mc_final_dist.png   一萬個平行時空「最後賺賠」的常態分佈圖
- mc_streak_dist.png  一萬個平行時空「最長連賠幾次」的分佈圖
- formula_ev.png      期望值公式（圖片）
- formula_streak.png  最長連敗經驗公式（圖片）
並印出全部真實統計數字，供文章引用。
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

OUT = common.CHART_DIR
os.makedirs(OUT, exist_ok=True)
_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False

WINC = "#2ca02c"
LOSSC = "#e45756"
INK = "#33414f"
BLUE = "#4c78a8"
ORANGE = "#f58518"

rng = np.random.default_rng(42)   # 固定種子，數字可重現

# ── 共用設定 ─────────────────────────────────────────────
N_SIM = 10_000        # 平行時空數量
N_TRADE = 1_000       # 每個時空的交易筆數
CAP = 1_000_000       # 本金
BET = 10_000          # 每筆固定冒險金額（本金 1%）
WIN_MULT, LOSS_MULT = 2.0, 1.0   # 賺賠比 2:1


def simulate(win_rate):
    """回傳 (輸贏矩陣 wins[N,T] bool, 每時空最長連賠 max_streak[N])。"""
    wins = rng.random((N_SIM, N_TRADE)) < win_rate     # True=贏
    losses = ~wins
    # 逐日累計連賠長度：run[t] = 前一天連賠+1，只要今天贏就歸零
    run = np.zeros(N_SIM, dtype=int)
    max_streak = np.zeros(N_SIM, dtype=int)
    for t in range(N_TRADE):
        run = (run + 1) * losses[:, t]
        max_streak = np.maximum(max_streak, run)
    return wins, max_streak


# ── 模擬：主策略 35%/2:1 與 對照 55%/1:1 ──────────────────
wins35, streak35 = simulate(0.35)
# 對照組賺賠比 1:1，連敗只跟勝率有關，用同函式即可
_, streak55 = simulate(0.55)

# 每筆固定下注（1%）下的「最後賺賠」= 100 萬 + 累積損益
pnl = np.where(wins35, BET * WIN_MULT, -BET * LOSS_MULT)   # +2 萬 / −1 萬
final_profit = pnl.sum(axis=1)                             # 相對本金的總損益（元）
equity = CAP + np.cumsum(pnl, axis=1)                      # 權益曲線
# 最大回撤（相對歷史高點）
running_max = np.maximum.accumulate(np.column_stack([np.full(N_SIM, CAP), equity]), axis=1)
dd = (np.column_stack([np.full(N_SIM, CAP), equity]) - running_max) / running_max
dd_mag = -dd.min(axis=1)                                   # 回撤幅度（正值，越大越慘）
ruin_small = int((equity.min(axis=1) <= 0).sum())          # 小注破產次數

# 重注（複利押 25%）：贏 ×1.5、輸 ×0.75
factor = np.where(wins35, 1.5, 0.75)
equity_heavy = CAP * np.cumprod(factor, axis=1)
ruin_heavy = int((equity_heavy.min(axis=1) < 0.1 * CAP).sum())   # 跌破本金 10% 視為陣亡


def pctl(a, p):
    return np.percentile(a, p)


# ── 印出真實統計 ─────────────────────────────────────────
print("=" * 56)
print(f"模擬 {N_SIM:,} 次 × 每次 {N_TRADE:,} 筆　勝率35%、賺賠2:1")
print("-" * 56)
print("【最後賺賠（每筆固定 1 萬）】相對 100 萬本金")
print(f"  平均獲利 : {final_profit.mean():,.0f} 元（≈ {final_profit.mean()/CAP*100:.1f}%）")
print(f"  標準差   : {final_profit.std():,.0f} 元")
print(f"  最後虧損的比例 : {(final_profit < 0).mean()*100:.1f}%")
print(f"  P5 / 中位 / P95 : {pctl(final_profit,5):,.0f} / {pctl(final_profit,50):,.0f} / {pctl(final_profit,95):,.0f}")
print("-" * 56)
print("【最長連賠次數】35%/2:1")
for p in (50, 95, 99):
    print(f"  P{p} : {pctl(streak35,p):.0f}")
print(f"  極值(max) : {streak35.max()}")
print("  對照 55%/1:1 :",
      f"中位 {pctl(streak55,50):.0f} / P95 {pctl(streak55,95):.0f} / P99 {pctl(streak55,99):.0f} / 極值 {streak55.max()}")
print("-" * 56)
print("【回撤 / 破產】")
print(f"  小注(1%) 最大回撤 中位 {pctl(dd_mag,50)*100:.1f}% / 極端(P99) {pctl(dd_mag,99)*100:.1f}%")
print(f"  小注(1%) 破產次數 : {ruin_small} / {N_SIM}")
print(f"  重注(25%複利) 陣亡次數 : {ruin_heavy} / {N_SIM}（{ruin_heavy/N_SIM*100:.1f}%）")
print("=" * 56)


# ══════════ 圖一：最後賺賠 常態分佈 ══════════
fig1, ax = plt.subplots(figsize=(11, 5.6))
data_wan = final_profit / 10_000.0          # 換成「萬元」
mu, sd = data_wan.mean(), data_wan.std()
loss_ratio = (final_profit < 0).mean() * 100

n, bins, patches = ax.hist(data_wan, bins=60, color=BLUE, alpha=0.75,
                           edgecolor="white", linewidth=0.4, zorder=3)
# 把虧損區間（<0）的長條塗紅
for i in range(len(patches)):
    if bins[i] < 0:
        patches[i].set_facecolor(LOSSC)
# 常態曲線疊上去
xs = np.linspace(data_wan.min(), data_wan.max(), 300)
pdf = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
binw = bins[1] - bins[0]
ax.plot(xs, pdf * N_SIM * binw, color=INK, lw=2.2, zorder=5, label="常態分佈曲線")
ax.axvline(0, color="#555", lw=1.4, ls="--", zorder=4)
ax.axvline(mu, color=WINC, lw=1.8, zorder=4)

ax.set_xlabel("最後賺賠（萬元）", fontproperties=_fp, fontsize=12)
ax.set_ylabel("有幾個平行時空落在這", fontproperties=_fp, fontsize=12)
for lb in ax.get_xticklabels() + ax.get_yticklabels():
    lb.set_fontproperties(_fp); lb.set_fontsize(9)
ax.legend(prop=_fp, fontsize=11, loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.25)
p1 = os.path.join(OUT, "mc_final_dist.png")
fig1.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig1)


# ══════════ 圖二：最長連賠 分佈 ══════════
fig2, ax2 = plt.subplots(figsize=(11, 5.2))
m = max(streak35.max(), streak55.max())
edges = np.arange(0, m + 2) - 0.5
ax2.hist(streak35, bins=edges, color=ORANGE, alpha=0.8, edgecolor="white",
         linewidth=0.4, label="勝率 35%（低勝率）", zorder=3)
ax2.hist(streak55, bins=edges, color=BLUE, alpha=0.55, edgecolor="white",
         linewidth=0.4, label="勝率 55%（對照）", zorder=2)
ax2.set_xlabel("最長連續賠幾次", fontproperties=_fp, fontsize=12)
ax2.set_ylabel("有幾個平行時空", fontproperties=_fp, fontsize=12)
for lb in ax2.get_xticklabels() + ax2.get_yticklabels():
    lb.set_fontproperties(_fp); lb.set_fontsize(9)
ax2.legend(prop=_fp, fontsize=11, loc="upper right", frameon=False)
ax2.grid(axis="y", alpha=0.25)
p2 = os.path.join(OUT, "mc_streak_dist.png")
fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2)


# ══════════ 公式圖：期望值 ══════════
def formula_card(path, title, line_main, line_calc, caption, calc_color=WINC, minimal=False):
    if minimal:
        # 精簡版：只有公式 + 範例計算，無標題、無下方解釋
        fig, ax = plt.subplots(figsize=(9, 1.9))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.01, 0.04), 0.98, 0.92, transform=ax.transAxes,
                     facecolor="#f7f9fc", edgecolor="#d5dde6", linewidth=1.2))
        ax.text(0.5, 0.64, line_main, transform=ax.transAxes, ha="center",
                fontproperties=_fp, fontsize=17, color=INK)
        ax.text(0.5, 0.30, line_calc, transform=ax.transAxes, ha="center",
                fontproperties=_fp, fontsize=16, color=calc_color)
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes,
                 facecolor="#f7f9fc", edgecolor="#d5dde6", linewidth=1.2))
    ax.text(0.5, 0.82, title, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=13, color=INK)
    ax.text(0.5, 0.55, line_main, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=16, color=INK)
    ax.text(0.5, 0.34, line_calc, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=15, color=calc_color)
    ax.text(0.5, 0.13, caption, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=10.5, color="#666")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


formula_card(
    os.path.join(OUT, "formula_ev.png"),
    "",
    "期望值 ＝ 勝率 × 平均賺 ＋ 敗率 × 平均賠",
    "＝ 35% ×(+2) ＋ 65% ×(-1) ＝ ＋0.05",
    "", minimal=True)

formula_card(
    os.path.join(OUT, "formula_streak.png"),
    "",
    "最長連賠 約 ln(交易筆數 × 勝率) ÷ ln(1 ÷ 敗率)",
    "約 ln(1000 × 0.35) ÷ ln(1 ÷ 0.65) ＝ 13.6",
    "", calc_color=INK, minimal=True)

print("OK ->", p1)
print("OK ->", p2)
print("OK ->", os.path.join(OUT, "formula_ev.png"))
print("OK ->", os.path.join(OUT, "formula_streak.png"))
