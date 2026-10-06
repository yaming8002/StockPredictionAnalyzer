# -*- coding: utf-8 -*-
"""指標字典（backtest-metrics-guide）用公式圖：期望值、獲利因子。
minimal：只有公式 + 一個舉例，無標題、無下方解釋。數字為假想示範。"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = common.CHART_DIR
os.makedirs(OUT, exist_ok=True)
_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False
INK = "#33414f"
WINC = "#2ca02c"


def formula_card(path, line_main, line_calc, calc_color=WINC):
    fig, ax = plt.subplots(figsize=(9, 1.9))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.04), 0.98, 0.92, transform=ax.transAxes,
                 facecolor="#f7f9fc", edgecolor="#d5dde6", linewidth=1.2))
    ax.text(0.5, 0.64, line_main, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=17, color=INK)
    ax.text(0.5, 0.30, line_calc, transform=ax.transAxes, ha="center",
            fontproperties=_fp, fontsize=16, color=calc_color)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


formula_card(
    os.path.join(OUT, "metric_ev.png"),
    "期望值 ＝ 勝率 × 平均獲利 ＋ 敗率 × 平均虧損",
    "＝ 40% × 6,000 ＋ 60% × (-2,000) ＝ ＋1,200 元")

formula_card(
    os.path.join(OUT, "metric_pf.png"),
    "獲利因子 ＝ 賺單總獲利 ÷ |賠單總虧損|",
    "＝ (40% × 6,000) ÷ (60% × 2,000) ＝ 2.0", calc_color=INK)

print("OK -> metric_ev.png / metric_pf.png")
