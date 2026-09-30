# -*- coding: utf-8 -*-
"""
KD 交叉（一）用圖：KD 黃金/死亡交叉單股示範。
- 資料：data/stock_data/{id}.parquet（OHLCV，全史）；KD 以 calculate_kd(9,3,3) 現算
- 上面板 K 線（台股紅漲綠跌）+ 黃金交叉▲/死亡交叉▼標記
- 下面板可選 KD（K、D 線 + 20/80 線）或成交量（張）
- 圖下方標註資料來源；X 軸 categorical（不留週末空格）
產兩張：
  kd_whipsaw_2330_TW   台積電某段 KD 反覆穿插、每次小賠（雜訊交叉）
  kd_lowliq_9957_TWO   低流動性股，K 棒稀疏、幾乎不可成交
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
import os
import sys

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

UP = "#d62728"     # 台股紅＝漲
DOWN = "#2ca02c"   # 綠＝跌


def load(sid):
    df = pd.read_parquet(os.path.join(DATA, f"{sid}.parquet")).sort_index()
    if "k" not in df.columns or "d" not in df.columns:
        calculate_kd(df, n=9, k_smooth=3, d_smooth=3)
    k, d = df["k"], df["d"]
    df["golden"] = (k > d) & (k.shift(1) <= d.shift(1))
    df["death"] = (k < d) & (k.shift(1) >= d.shift(1))
    return df


def _candles(ax, w):
    for i, (_, r) in enumerate(w.iterrows()):
        up = r["close"] >= r["open"]
        c = UP if up else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.6, zorder=1)
        lo_b, hi_b = (r["open"], r["close"]) if up else (r["close"], r["open"])
        ax.add_patch(plt.Rectangle((i - 0.3, lo_b), 0.6, max(hi_b - lo_b, 1e-6),
                                   facecolor=c, edgecolor=c, linewidth=0.6, zorder=2))


def draw(sid, label, start, end, out_name, lower="kd", title="", mark="cross"):
    """mark='cross'：標窗口內所有黃金/死叉；mark=[(buy,sell),...]：只標指定交易。"""
    df = load(sid)
    w = df.loc[start:end].copy()
    x = range(len(w))

    ratios = [3, 1.1] if lower == "kd" else [3, 1]
    fig, (ax, axl) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": ratios, "hspace": 0.07})
    _candles(ax, w)

    # 買賣標記
    pos = w.index.get_indexer
    if mark == "cross":
        gi = [i for i, g in enumerate(w["golden"].values) if g]
        di = [i for i, g in enumerate(w["death"].values) if g]
        if gi:
            ax.scatter(gi, w["low"].values[gi] * 0.985, marker="^", s=70, color=UP,
                       edgecolor="black", linewidth=0.5, zorder=5, label="黃金交叉（買進）")
        if di:
            ax.scatter(di, w["high"].values[di] * 1.015, marker="v", s=70, color=DOWN,
                       edgecolor="black", linewidth=0.5, zorder=5, label="死亡交叉（賣出）")
    else:
        for (buy, sell) in mark:
            bi = w.index.get_indexer([pd.Timestamp(buy)], method="nearest")[0]
            si = w.index.get_indexer([pd.Timestamp(sell)], method="nearest")[0]
            ax.scatter([bi], [w.iloc[bi]["open"]], marker="^", s=140, color=UP,
                       edgecolor="black", linewidth=0.6, zorder=5, label="黃金交叉買進")
            ax.scatter([si], [w.iloc[si]["open"]], marker="v", s=140, color=DOWN,
                       edgecolor="black", linewidth=0.6, zorder=5, label="死亡交叉賣出")

    ax.set_title(title or label, fontproperties=_fp, fontsize=14, pad=10)
    ax.set_ylabel("股價（元）", fontproperties=_fp, fontsize=11)
    # 去重 legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), prop=_fp, fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    for lb in ax.get_yticklabels():
        lb.set_fontproperties(_fp)

    # 下面板
    if lower == "kd":
        axl.plot(x, w["k"].values, color="#1f77b4", linewidth=1.0, label="K")
        axl.plot(x, w["d"].values, color="#ff7f0e", linewidth=1.0, label="D")
        axl.axhline(80, color="#999", linewidth=0.7, linestyle="--")
        axl.axhline(20, color="#999", linewidth=0.7, linestyle="--")
        axl.set_ylim(0, 100)
        axl.set_ylabel("KD", fontproperties=_fp, fontsize=10)
        axl.legend(prop=_fp, fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    else:
        vol_zhang = w["volume"].values / 1000.0
        colors = [UP if c >= o else DOWN for o, c in zip(w["open"], w["close"])]
        axl.bar(x, vol_zhang, color=colors, width=0.7)
        axl.set_ylabel("成交量（張）", fontproperties=_fp, fontsize=10)
    axl.grid(True, alpha=0.25)
    for lb in axl.get_yticklabels():
        lb.set_fontproperties(_fp)

    n = len(w)
    step = max(1, n // 9)
    ticks = list(range(0, n, step))
    axl.set_xticks(ticks)
    axl.set_xticklabels([w.index[t].strftime("%Y-%m") for t in ticks],
                        fontproperties=_fp, fontsize=9)

    fig.text(0.5, 0.012, "數據來源：Yahoo Finance，僅供教學研究參考",
             ha="center", fontproperties=_fp, fontsize=9, color="#666")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.075)
    path = os.path.join(OUT, f"{out_name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"OK {out_name}  n={n}")


if __name__ == "__main__":
    # 台積電雜訊交叉：2015 年 4–9 月，KD 反覆穿插、幾乎每次小賠
    draw("2330.TW", "2330.TW", "2015-04-01", "2015-09-30", "kd_whipsaw_2330_TW",
         lower="kd", title="2330.TW　2015 年 4–9 月　KD 反覆黃金／死亡交叉", mark="cross")
