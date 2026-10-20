# -*- coding: utf-8 -*-
"""
〈雙底（一）〉示意圖：假想的收盤價走勢，標出第一個底、第二個底，以及第二個底「確認」的那一天。

全部是假想數字（不是任何股票）。ZigZag 門檻用 5%：第二個底 92 元要等收盤反彈到 92×1.05＝96.6 元
以上才算確認，確認當天收盤判定、隔天開盤進場。圖只放走勢、三種標記、確認門檻的虛線與圖例；
說明文字全部放在文章正文（圖精簡規則）。

輸出：blog/drafts/charts/double_bottom/double_bottom_concept.png，並複製到 blog/site/content/charts/double_bottom/
（需設 BLOG_DIR 指向 blog 目錄）。

執行：PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python _04_analysis/double_bottom/charts/double_bottom_concept.py
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
import numpy as np  # noqa: E402

from _01_data.indicators_pattern import calculate_zigzag  # noqa: E402
from _02_strategy.base.vbt import common  # noqa: E402

PCT = 0.05
# 假想走勢的轉折（交易日, 收盤價）；中間線性內插再加一點固定的小波動
KNOTS = [(0, 100.0), (12, 88.0), (22, 97.0), (32, 92.0), (40, 99.0), (50, 106.0)]
NAME = "double_bottom_concept.png"


def price_path() -> np.ndarray:
    days = np.arange(KNOTS[-1][0] + 1)
    base = np.interp(days, [k[0] for k in KNOTS], [k[1] for k in KNOTS])
    wiggle = 0.6 * np.sin(days * 1.7)                    # 固定小波動，讓線不要太死板
    wiggle[[k[0] for k in KNOTS]] = 0.0                  # 轉折點保持整數價，正文好對照
    return np.round(base + wiggle, 2)


def main() -> int:
    import pandas as pd
    close = price_path()
    zz = calculate_zigzag(pd.DataFrame({"close": close}), PCT)
    lows = zz["zigzag_turn_low"].dropna()
    # 第一個底、第二個底的「確認日」與值；低點本身落在確認前最近一次等於該值的那天
    (c1, l1), (c2, l2) = list(lows.items())[:2]
    d1 = int(np.flatnonzero(close[:c1] == l1)[-1])
    d2 = int(np.flatnonzero(close[:c2] == l2)[-1])
    if not l2 > l1:
        raise SystemExit("假想走勢的第二個底沒有高於第一個底，請調整 KNOTS")

    fp = common.chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(close, color="#555", linewidth=1.6, label="收盤價")
    ax.scatter([d1], [l1], s=90, color="#1f77b4", zorder=3, label=f"第一個底（{l1:g}）")
    ax.scatter([d2], [l2], s=90, marker="s", color="#ff7f0e", zorder=3, label=f"第二個底（{l2:g}）")
    ax.axhline(l2 * (1 + PCT), color="#ff7f0e", linestyle=":", linewidth=1.2,
               label=f"確認門檻：第二個底 ×1.05（{l2 * (1 + PCT):g}）")
    ax.scatter([c2], [close[c2]], s=130, marker="^", color="#d62728", zorder=4,
               label="第二個底確認日（隔天開盤買進）")
    ax.set_xlabel("交易日", fontproperties=fp)
    ax.set_ylabel("價格", fontproperties=fp)
    ax.legend(prop=fp, loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    blog = common.require_blog_dir()
    draft_dir = os.path.join(blog, "drafts", "charts", "double_bottom")
    site_dir = os.path.join(blog, "site", "content", "charts", "double_bottom")
    os.makedirs(draft_dir, exist_ok=True)
    os.makedirs(site_dir, exist_ok=True)
    out = os.path.join(draft_dir, NAME)
    fig.savefig(out, dpi=130)
    shutil.copy2(out, os.path.join(site_dir, NAME))
    print(f"第一個底 {l1}（第 {d1} 天，第 {c1} 天確認）｜第二個底 {l2}（第 {d2} 天，第 {c2} 天確認，"
          f"當天收盤 {close[c2]}）→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
