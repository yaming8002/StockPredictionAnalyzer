# -*- coding: utf-8 -*-
"""50/200 範例交易 K 線圖（含成交量、50/200 均線疊圖）。"""

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
import pandas as pd

DATA = common.DATA_DIR
OUT = common.CHART_DIR
_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False
UP, DOWN = "#d62728", "#2ca02c"

TRADES = [
    ("2617.TW", "2617.TW", "2020-09-03", "2021-11-24", "trade_win_2617_TW"),
    ("5703.TWO", "5703.TWO", "2018-03-23", "2019-07-05", "trade_loss_5703_TWO"),
]


def draw(file_id, label, buy, sell, out_name, pad=20):
    df = pd.read_parquet(os.path.join(DATA, f"{file_id}.parquet")).sort_index()
    buy_ts, sell_ts = pd.Timestamp(buy), pd.Timestamp(sell)
    idx_all = df.index
    bi = idx_all.get_indexer([buy_ts], method="nearest")[0]
    si = idx_all.get_indexer([sell_ts], method="nearest")[0]
    lo, hi = max(0, bi - pad), min(len(df) - 1, si + pad)
    w = df.iloc[lo:hi + 1].copy()
    x = range(len(w))
    buy_x = w.index.get_indexer([idx_all[bi]])[0]
    sell_x = w.index.get_indexer([idx_all[si]])[0]
    buy_px, sell_px = float(df.iloc[bi]["open"]), float(df.iloc[si]["open"])
    ret = (sell_px - buy_px) / buy_px * 100
    hold = (idx_all[si] - idx_all[bi]).days

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    for i, (_, r) in enumerate(w.iterrows()):
        up = r["close"] >= r["open"]; c = UP if up else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.6, zorder=1)
        lo_b, hi_b = (r["open"], r["close"]) if up else (r["close"], r["open"])
        ax.add_patch(plt.Rectangle((i - 0.3, lo_b), 0.6, max(hi_b - lo_b, 1e-6),
                                   facecolor=c, edgecolor=c, linewidth=0.6, zorder=2))
    if "sma_50" in w.columns:
        ax.plot(x, w["sma_50"].values, color="#1f77b4", linewidth=1.0, label="50 日均線")
    if "sma_200" in w.columns:
        ax.plot(x, w["sma_200"].values, color="#ff7f0e", linewidth=1.0, label="200 日均線")
    ax.scatter([buy_x], [buy_px], marker="^", s=130, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"黃金交叉買進 {buy_px:.2f}")
    ax.scatter([sell_x], [sell_px], marker="v", s=130, color="#2ca02c",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"死亡交叉賣出 {sell_px:.2f}")

    sign = "+" if ret >= 0 else ""
    ax.set_title(f"{label}　{buy}　→　{sell}　（{sign}{ret:.1f}%，抱 {hold} 天）",
                 fontproperties=_fp, fontsize=14, pad=10)
    ax.set_ylabel("股價（元）", fontproperties=_fp, fontsize=11)
    leg = ax.legend(prop=_fp, fontsize=9, loc="best", framealpha=0.9); leg.set_zorder(6)
    ax.grid(True, alpha=0.25)

    vol_zhang = w["volume"].values / 1000.0
    colors = [UP if c >= o else DOWN for o, c in zip(w["open"], w["close"])]
    axv.bar(x, vol_zhang, color=colors, width=0.7)
    axv.set_ylabel("成交量（張）", fontproperties=_fp, fontsize=10)
    axv.grid(True, alpha=0.25)
    for lb in axv.get_yticklabels():
        lb.set_fontproperties(_fp)
    n = len(w); step = max(1, n // 8); ticks = list(range(0, n, step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([w.index[t].strftime("%Y-%m") for t in ticks],
                        fontproperties=_fp, fontsize=9, rotation=0)
    for lb in ax.get_yticklabels():
        lb.set_fontproperties(_fp)
    fig.text(0.5, 0.012, "數據來源：Yahoo Finance，僅供教學研究參考",
             ha="center", fontproperties=_fp, fontsize=9, color="#666")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.075)
    fig.savefig(os.path.join(OUT, f"{out_name}.png"), dpi=130)
    plt.close(fig)
    print(f"OK {out_name}: {label} {buy}->{sell} {ret:+.1f}% {hold}d  avgvol={vol_zhang.mean():.1f}張")


if __name__ == "__main__":
    for t in TRADES:
        draw(*t)
