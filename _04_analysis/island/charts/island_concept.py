# -*- coding: utf-8 -*-
"""
〈島型反轉（一）〉示意圖：假想的 K 棒走勢，標出向下缺口、島、向上缺口，以及判定日。

全部是假想數字（不是任何股票）。島＝3 根 K 棒：島前一根的最低價 100，島內最高價 96，
所以 96～100 是向下缺口；判定日最低價 98.5 > 96，形成向上缺口。判定日收盤判定、隔天開盤進場。
圖只放 K 棒、兩個缺口的色帶、判定日標記與圖例；說明文字全部放在文章正文（圖精簡規則）。
判定結果用策略檔的 bottom_island 實際算一次，確認圖跟規則一致。

輸出：blog/drafts/charts/island/island_concept.png，並複製到 blog/site/content/charts/island/
（需設 BLOG_DIR 指向 blog 目錄）。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/island/charts/island_concept.py
"""
import os
import shutil
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.island_strategy.single_island_strategy import bottom_island  # noqa: E402

# 假想 K 棒（開, 高, 低, 收）：下跌 → 向下跳空 → 3 根島 → 向上跳空 → 續漲
BARS = [
    (112, 113, 109, 110), (110, 111, 106.5, 107), (107, 108, 104, 104.5), (104.5, 105, 101, 102),
    (102, 103, 100, 100.5),                                       # 島前一根（最低 100）
    (95, 96, 92, 93), (93, 95, 91, 94.5), (94, 95.5, 92.5, 95),     # 島（最高 96）
    (99.5, 103, 98.5, 102.5),                                     # 判定日（最低 98.5）
    (102.5, 105, 101.5, 104.5), (104.5, 107, 103.5, 106), (106, 108.5, 105, 108),
]
ISLAND = (5, 7)          # 島的第一根、最後一根位置
NAME = "island_concept.png"


def draw_candles(ax, df: pd.DataFrame) -> None:
    """紅漲綠跌（台股慣例）的簡易 K 棒。"""
    for i, (o, h, lo, c) in enumerate(df[["open", "high", "low", "close"]].itertuples(index=False)):
        color = "#d62728" if c >= o else "#2ca02c"
        ax.plot([i, i], [lo, h], color=color, linewidth=1.2, zorder=2)
        ax.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 0.15),
                               color=color, zorder=3))


def main() -> int:
    df = pd.DataFrame(BARS, columns=["open", "high", "low", "close"])
    k, top, _ = bottom_island(df["high"], df["low"])
    signal_day = ISLAND[1] + 1
    if k[signal_day] != ISLAND[1] - ISLAND[0] + 1 or pd.notna(k[:signal_day]).any():
        raise SystemExit("假想 K 棒的判定結果跟圖上標的不一致，請調整 BARS")
    before_low = df["low"][ISLAND[0] - 1]
    island_top = top[signal_day]
    signal_low = df["low"][signal_day]

    fp = common.chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 4.6))
    draw_candles(ax, df)
    ax.add_patch(Rectangle((ISLAND[0] - 1.4, island_top), 1.0, before_low - island_top,
                           color="#1f77b4", alpha=0.25, label=f"向下缺口（{island_top:g}～{before_low:g}）"))
    ax.add_patch(Rectangle((ISLAND[1] + 0.4, island_top), 0.2 + 0.4, signal_low - island_top,
                           color="#ff7f0e", alpha=0.35, label=f"向上缺口（{island_top:g}～{signal_low:g}）"))
    ax.add_patch(Rectangle((ISLAND[0] - 0.45, df["low"][ISLAND[0]:ISLAND[1] + 1].min() - 0.6),
                           ISLAND[1] - ISLAND[0] + 0.9,
                           island_top - df["low"][ISLAND[0]:ISLAND[1] + 1].min() + 1.2,
                           fill=False, edgecolor="#555", linestyle="--", linewidth=1.0, label="島（3 根 K 棒）"))
    ax.scatter([signal_day], [df["high"][signal_day] + 1.2], s=120, marker="v", color="#d62728",
               zorder=4, label="判定日（隔天開盤買進）")
    ax.set_xlabel("交易日", fontproperties=fp)
    ax.set_ylabel("價格", fontproperties=fp)
    ax.set_xlim(-0.8, len(df) - 0.2)
    ax.legend(prop=fp, loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    blog = common.require_blog_dir()
    draft_dir = os.path.join(blog, "drafts", "charts", "island")
    site_dir = os.path.join(blog, "site", "content", "charts", "island")
    os.makedirs(draft_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)
    out = os.path.join(draft_dir, NAME)
    fig.savefig(out, dpi=130)
    shutil.copy2(out, os.path.join(site_dir, NAME))
    print(f"島前一根最低 {before_low}｜島內最高 {island_top}｜判定日最低 {signal_low}｜k={int(k[signal_day])} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
