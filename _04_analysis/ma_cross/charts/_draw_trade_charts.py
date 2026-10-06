# -*- coding: utf-8 -*-
"""
繪製「買入→賣出」單筆交易 K 線圖（含成交量）。
- 資料來源：data/stock_data/{id}.parquet（OHLCV + 指標，全史）
- 交易邏輯對齊 ma_cross baseline：sma_120 上穿 sma_200 隔日開盤買、下穿隔日開盤賣
- 上面板 K 線（台股紅漲綠跌）+ MA120/MA200；下面板成交量（張）；X 軸 categorical（不留週末空格）
- 圖下方標註：數據來源：Yahoo Finance，僅供教學研究參考
"""

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

# 中文字型（微軟正黑）
_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False

UP = "#d62728"     # 台股紅＝漲
DOWN = "#2ca02c"   # 綠＝跌

# 要畫的交易：(檔名, 顯示代號, 買日, 賣日, 標題)
TRADES = [
    ("3059.TW", "3059.TW", "2013-05-03", "2014-10-01", "trade_win_3059_TW"),
    ("6758.TWO", "6758.TWO", "2023-06-16", "2024-12-25", "trade_loss_6758_TWO"),
    ("2108.TW", "2108.TW", "2020-06-23", "2021-12-01", "win2_2108_TW"),
    ("5434.TW", "5434.TW", "2005-04-13", "2006-05-11", "win3_5434_TW"),
    ("3055.TW", "3055.TW", "2007-04-10", "2008-02-13", "loss2_3055_TW"),
    ("1323.TW", "1323.TW", "2015-02-24", "2015-08-06", "loss3_1323_TW"),
]


def draw(file_id, label, buy, sell, out_name, pad=20):
    df = pd.read_parquet(os.path.join(DATA, f"{file_id}.parquet")).sort_index()
    buy_ts, sell_ts = pd.Timestamp(buy), pd.Timestamp(sell)
    # 取買賣區間，前後各補 pad 個交易日
    idx_all = df.index
    bi = idx_all.get_indexer([buy_ts], method="nearest")[0]
    si = idx_all.get_indexer([sell_ts], method="nearest")[0]
    lo, hi = max(0, bi - pad), min(len(df) - 1, si + pad)
    w = df.iloc[lo:hi + 1].copy()
    x = range(len(w))
    buy_x = w.index.get_indexer([idx_all[bi]])[0]
    sell_x = w.index.get_indexer([idx_all[si]])[0]
    buy_px = float(df.iloc[bi]["open"])
    sell_px = float(df.iloc[si]["open"])
    ret = (sell_px - buy_px) / buy_px * 100
    hold = (idx_all[si] - idx_all[bi]).days

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    # ── K 線 ──
    for i, (_, r) in enumerate(w.iterrows()):
        up = r["close"] >= r["open"]
        c = UP if up else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.6, zorder=1)
        lo_b, hi_b = (r["open"], r["close"]) if up else (r["close"], r["open"])
        ax.add_patch(plt.Rectangle((i - 0.3, lo_b), 0.6, max(hi_b - lo_b, 1e-6),
                                   facecolor=c, edgecolor=c, linewidth=0.6, zorder=2))
    # 均線
    if "sma_120" in w.columns:
        ax.plot(x, w["sma_120"].values, color="#1f77b4", linewidth=1.0, label="120 日均線")
    if "sma_200" in w.columns:
        ax.plot(x, w["sma_200"].values, color="#ff7f0e", linewidth=1.0, label="200 日均線")
    # 買賣標記
    ax.scatter([buy_x], [buy_px], marker="^", s=130, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"黃金交叉買進 {buy_px:.2f}")
    ax.scatter([sell_x], [sell_px], marker="v", s=130, color="#2ca02c",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"死亡交叉賣出 {sell_px:.2f}")

    sign = "+" if ret >= 0 else ""
    ax.set_title(f"{label}　{buy}　→　{sell}　（{sign}{ret:.1f}%，抱 {hold} 天）",
                 fontproperties=_fp, fontsize=14, pad=10)
    ax.set_ylabel("股價（元）", fontproperties=_fp, fontsize=11)
    leg = ax.legend(prop=_fp, fontsize=9, loc="best", framealpha=0.9)
    leg.set_zorder(6)
    ax.grid(True, alpha=0.25)

    # ── 成交量（張）──
    vol_zhang = w["volume"].values / 1000.0
    colors = [UP if c >= o else DOWN for o, c in zip(w["open"], w["close"])]
    axv.bar(x, vol_zhang, color=colors, width=0.7)
    axv.set_ylabel("成交量（張）", fontproperties=_fp, fontsize=10)
    axv.grid(True, alpha=0.25)
    for lb in axv.get_yticklabels():
        lb.set_fontproperties(_fp)

    # X 軸：稀疏標日期
    n = len(w)
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([w.index[t].strftime("%Y-%m") for t in ticks],
                        fontproperties=_fp, fontsize=9, rotation=0)
    for lb in ax.get_yticklabels():
        lb.set_fontproperties(_fp)

    fig.text(0.5, 0.012, "數據來源：Yahoo Finance，僅供教學研究參考",
             ha="center", fontproperties=_fp, fontsize=9, color="#666")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.075)
    path = os.path.join(OUT, f"{out_name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"OK {out_name}: {label} {buy}->{sell} {ret:+.1f}% {hold}d  (n={n})")


if __name__ == "__main__":
    for t in TRADES:
        draw(*t)
