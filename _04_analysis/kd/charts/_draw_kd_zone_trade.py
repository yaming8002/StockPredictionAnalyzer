# -*- coding: utf-8 -*-
"""
KD 交叉（二）超買超賣用圖：一筆「低檔黃金交叉買、高檔死叉賣」的實際交易，
並標出「基本版會在中間哪些地方就賣出」以顯示差異。
- 上面板 K 線 + 超買超賣進(▲)/出(▼) + 基本版一般死叉(灰▽，＝基本版會提早賣的點)
- 下面板 KD（K、D 線 + 20/80 線 + 超買/超賣區淡色底）
範例：2330.TW 2008-10-29 → 2009-09-11（+48.7%、抱 317 天，金融海嘯谷底→2009 復甦）
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from _01_data.indicators_momentum_volume import calculate_kd

DATA = common.DATA_DIR
OUT = common.CHART_DIR
_fp = font_manager.FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
plt.rcParams["axes.unicode_minus"] = False
UP, DOWN = "#d62728", "#2ca02c"


def draw(sid, buy, sell, start, end, out_name, title):
    df = pd.read_parquet(os.path.join(DATA, f"{sid}.parquet")).sort_index()
    calculate_kd(df, 9, 3, 3)
    k, d = df["k"], df["d"]
    df["golden"] = (k > d) & (k.shift(1) <= d.shift(1))
    df["death"] = (k < d) & (k.shift(1) >= d.shift(1))
    w = df.loc[start:end].copy()
    x = range(len(w))
    gi = w.index

    fig, (ax, axk) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.07})
    # K 線
    for i, (_, r) in enumerate(w.iterrows()):
        up = r["close"] >= r["open"]; c = UP if up else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.5, zorder=1)
        lo_b, hi_b = (r["open"], r["close"]) if up else (r["close"], r["open"])
        ax.add_patch(plt.Rectangle((i - 0.3, lo_b), 0.6, max(hi_b - lo_b, 1e-6),
                                   facecolor=c, edgecolor=c, linewidth=0.5, zorder=2))
    bx = gi.get_indexer([pd.Timestamp(buy)], method="nearest")[0]
    sx = gi.get_indexer([pd.Timestamp(sell)], method="nearest")[0]
    # 基本版一般死叉（買賣區間內）＝基本版會提早賣出的點
    dcross = [i for i in range(len(w)) if w["death"].iloc[i] and bx < i < sx]
    if dcross:
        ax.scatter(dcross, w["high"].values[dcross] * 1.02, marker="v", s=45,
                   facecolor="none", edgecolor="#888", linewidth=1.0, zorder=4,
                   label="基本版會在此賣出（一般死叉）")
    # 超買超賣進出
    ax.scatter([bx], [w.iloc[bx]["open"]], marker="^", s=170, color=UP,
               edgecolor="black", linewidth=0.7, zorder=6, label="超買超賣進場（低檔黃金交叉）")
    ax.scatter([sx], [w.iloc[sx]["open"]], marker="v", s=170, color=DOWN,
               edgecolor="black", linewidth=0.7, zorder=6, label="超買超賣出場（高檔死叉）")

    ax.set_title(title, fontproperties=_fp, fontsize=14, pad=10)
    ax.set_ylabel("股價（元）", fontproperties=_fp, fontsize=11)
    ax.legend(prop=_fp, fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.25)
    for lb in ax.get_yticklabels():
        lb.set_fontproperties(_fp)

    # KD 面板 + 超買/超賣區
    axk.axhspan(80, 100, color="#d62728", alpha=0.07)
    axk.axhspan(0, 20, color="#2ca02c", alpha=0.07)
    axk.plot(x, w["k"].values, color="#1f77b4", linewidth=1.0, label="K")
    axk.plot(x, w["d"].values, color="#ff7f0e", linewidth=1.0, label="D")
    axk.axhline(80, color="#999", linewidth=0.7, linestyle="--")
    axk.axhline(20, color="#999", linewidth=0.7, linestyle="--")
    axk.set_ylim(0, 100); axk.set_yticks([0, 20, 50, 80, 100])
    axk.set_ylabel("KD", fontproperties=_fp, fontsize=10)
    axk.legend(prop=_fp, fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    axk.grid(True, alpha=0.2)
    for lb in axk.get_yticklabels():
        lb.set_fontproperties(_fp)

    n = len(w); step = max(1, n // 9); ticks = list(range(0, n, step))
    axk.set_xticks(ticks)
    axk.set_xticklabels([w.index[t].strftime("%Y-%m") for t in ticks], fontproperties=_fp, fontsize=9)

    fig.text(0.5, 0.012, "數據來源：Yahoo Finance，僅供教學研究參考",
             ha="center", fontproperties=_fp, fontsize=9, color="#666")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.075)
    fig.savefig(os.path.join(OUT, f"{out_name}.png"), dpi=130)
    plt.close(fig)
    print(f"OK {out_name}  n={n}  buy@{w.iloc[bx]['open']:.1f} sell@{w.iloc[sx]['open']:.1f}  基本死叉數={len(dcross)}")


if __name__ == "__main__":
    draw("2330.TW", "2008-10-29", "2009-09-11", "2008-08-01", "2009-11-15",
         "kd_zone_2330_TW",
         "2330.TW　2008–2009：低檔黃金交叉買進、抱到高檔死叉才賣出")
