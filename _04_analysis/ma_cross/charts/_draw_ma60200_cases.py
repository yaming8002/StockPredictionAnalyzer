# -*- coding: utf-8 -*-
"""
繪製 MA 60/200 規則下、各報酬區間代表交易的 K 線圖。
- 資料來源：data/stock_data/2603.TW.parquet（已還原配股的連續價格）
- 規則：sma_60 上穿 sma_200 隔日開盤買、下穿隔日開盤賣
- 上面板 K 線（台股紅漲綠跌）+ MA60/MA200；下面板成交量（張）
- 沿用 _draw_trade_charts.py 的版式
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

_fp = common.chinese_font()
plt.rcParams["axes.unicode_minus"] = False

UP, DOWN = "#d62728", "#2ca02c"

# (檔名, 顯示代號, 買日, 賣日, 輸出檔名)
TRADES = [
    # 由差到好，每個等差區間挑一筆最接近該區間中位數、且資料乾淨的交易
    # （乾淨＝買賣日無 volume=0 幽靈 K 棒、窗內無連續 3 日以上停牌、無 40% 以上跳空、進場前有 250 根真實 K 棒）
    ("2314.TW", "2314 台揚",   "2024-12-27", "2025-04-07", "bin01_2314"),   #  -70.2%
    ("6669.TW", "6669 緯穎",   "2025-01-09", "2025-04-07", "bin02_6669"),   #  -42.4%
    ("2353.TW", "2353 宏碁",   "2022-01-19", "2022-06-30", "bin03_2353"),   #  -25.7%
    ("2383.TW", "2383 台光電", "2011-01-11", "2011-06-29", "bin04_2383"),   #   -9.6%
    ("1101.TW", "1101 台泥",   "2019-04-16", "2020-04-28", "bin05_1101"),   #   +6.5%
    ("2884.TW", "2884 玉山金", "2019-03-15", "2020-05-05", "bin06_2884"),   #  +28.9%
    ("2881.TW", "2881 富邦金", "2023-02-07", "2025-04-28", "bin07_2881"),   #  +48.3%
    ("2408.TW", "2408 南亞科", "2016-11-10", "2018-08-21", "bin08_2408"),   #  +68.0%
    ("2454.TW", "2454 聯發科", "2020-06-12", "2021-10-13", "bin09_2454"),   #  +88.4%
    ("2376.TW", "2376 技嘉",   "2023-01-03", "2024-08-26", "bin10_2376"),   # +158.1%
    # 附帶：最頂那格內部還有長尾——同一檔連續兩筆，一賠一賺
    ("2603.TW", "2603 長榮",   "2019-05-02", "2019-12-23", "extra_2603_loss"),
    ("2603.TW", "2603 長榮",   "2020-08-25", "2022-07-13", "extra_2603_win"),
]


def draw(file_id, label, buy, sell, out_name, pad=20):
    df = pd.read_parquet(os.path.join(DATA, f"{file_id}.parquet")).sort_index()
    idx_all = df.index
    bi = idx_all.get_indexer([pd.Timestamp(buy)], method="nearest")[0]
    si = idx_all.get_indexer([pd.Timestamp(sell)], method="nearest")[0]
    lo, hi = max(0, bi - pad), min(len(df) - 1, si + pad)
    w = df.iloc[lo:hi + 1].copy()
    x = range(len(w))
    buy_x = w.index.get_indexer([idx_all[bi]])[0]
    sell_x = w.index.get_indexer([idx_all[si]])[0]
    buy_px, sell_px = float(df.iloc[bi]["open"]), float(df.iloc[si]["open"])
    ret = (sell_px - buy_px) / buy_px * 100
    hold = (idx_all[si] - idx_all[bi]).days

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    # K 棒；區間長時收窄棒寬避免糊成一團
    n = len(w)
    bw = 0.6 if n < 400 else 0.9
    lw = 0.6 if n < 400 else 0.35
    for i, (_, r) in enumerate(w.iterrows()):
        up = r["close"] >= r["open"]
        c = UP if up else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=lw, zorder=1)
        lo_b, hi_b = (r["open"], r["close"]) if up else (r["close"], r["open"])
        ax.add_patch(plt.Rectangle((i - bw / 2, lo_b), bw, max(hi_b - lo_b, 1e-6),
                                   facecolor=c, edgecolor=c, linewidth=lw, zorder=2))

    ax.plot(x, w["sma_60"].values, color="#1f77b4", linewidth=1.1, label="60 日均線")
    ax.plot(x, w["sma_200"].values, color="#ff7f0e", linewidth=1.1, label="200 日均線")
    ax.scatter([buy_x], [buy_px], marker="^", s=140, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"黃金交叉買進 {buy_px:.2f}")
    ax.scatter([sell_x], [sell_px], marker="v", s=140, color="#2ca02c",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"死亡交叉賣出 {sell_px:.2f}")

    sign = "+" if ret >= 0 else ""
    ax.set_title(f"{label}　{buy}　→　{sell}　（{sign}{ret:.1f}%，抱 {hold} 天）",
                 fontproperties=_fp, fontsize=14, pad=10)
    ax.set_ylabel("股價（元）", fontproperties=_fp, fontsize=11)
    leg = ax.legend(prop=_fp, fontsize=9, loc="best", framealpha=0.9)
    leg.set_zorder(6)
    ax.grid(True, alpha=0.25)

    vol_zhang = w["volume"].values / 1000.0
    colors = [UP if c >= o else DOWN for o, c in zip(w["open"], w["close"])]
    axv.bar(x, vol_zhang, color=colors, width=0.7 if n < 400 else 1.0)
    axv.set_ylabel("成交量（張）", fontproperties=_fp, fontsize=10)
    axv.grid(True, alpha=0.25)

    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([w.index[t].strftime("%Y-%m") for t in ticks],
                        fontproperties=_fp, fontsize=9)
    for lb in list(ax.get_yticklabels()) + list(axv.get_yticklabels()):
        lb.set_fontproperties(_fp)

    fig.text(0.5, 0.012, "數據來源：Yahoo Finance，價格已還原配股，僅供教學研究參考",
             ha="center", fontproperties=_fp, fontsize=9, color="#666")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.075)
    path = os.path.join(OUT, f"{out_name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"OK {out_name}: {label} {buy}->{sell} {ret:+.1f}% {hold}d (n={n})")


if __name__ == "__main__":
    for t in TRADES:
        draw(*t)
