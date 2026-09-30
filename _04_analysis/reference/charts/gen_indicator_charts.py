# -*- coding: utf-8 -*-
"""產生指標教學文章的示意圖（真實 2330 資料 → 靜態 PNG）。

輸出到 blog/site/content/charts/indicators/，文章內用 /static/charts/indicators/xxx.png 引用。
可重跑：資料更新後重新執行即可刷新圖。

執行：
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 F:/stock-analyzer/.venv/Scripts/python.exe _04_analysis/reference/charts/gen_indicator_charts.py
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字型（Windows 內建 JhengHei）；找不到就退回預設
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

DATA = os.path.join(common.DATA_DIR, "2330.TW.parquet")
OUT = os.path.join(common.CHART_DIR, "indicators")
WINDOW = 420          # 顯示最近約 1.7 年，圖才清楚
UP, DOWN = "#c0392b", "#1e8449"   # 台股紅漲綠跌
ACCENT, MUTED = "#1a73a7", "#888888"

os.makedirs(OUT, exist_ok=True)


def load():
    df = pd.read_parquet(DATA)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


def price_panel(ax, s):
    ax.plot(s.index, s["close"], color="#222", lw=1.1, label="收盤價")
    ax.grid(alpha=0.25)


# ---- 指標計算（與 stock_technical.py 一致）----
def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["sma5"] = c.rolling(5).mean()
    df["sma20"] = c.rolling(20).mean()
    df["sma60"] = c.rolling(60).mean()
    short, long = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["macd"] = short - long
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    df["bb_up"], df["bb_mid"], df["bb_low"] = sma20 + 2 * std20, sma20, sma20 - 2 * std20
    df["bias20"] = (c - sma20) / sma20 * 100
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9) * 100
    df["k"] = rsv.ewm(com=2, adjust=False).mean()
    df["d"] = df["k"].ewm(com=2, adjust=False).mean()
    hl = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / hl
    df["cmf"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum()
    df["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / c * 100
    df["vol20"] = np.log(c / c.shift(1)).rolling(20).std() * 100
    df["dc_up"] = h.rolling(20).max()
    df["dc_low"] = l.rolling(20).min()
    return df


def main():
    df = add_indicators(load())
    s = df.tail(WINDOW)
    x = s.index

    # 1. 均線的多頭 / 空頭排列（SMA5 / 20 / 60）
    fig, ax = plt.subplots(figsize=(9, 4.4))
    price_panel(ax, s)
    ax.plot(x, s["sma5"], color="#d14d8b", lw=1.0, label="SMA5（短）")
    ax.plot(x, s["sma20"], color=ACCENT, lw=1.0, label="SMA20（中）")
    ax.plot(x, s["sma60"], color="#e08a1e", lw=1.0, label="SMA60（長）")
    bull = (s["sma5"] > s["sma20"]) & (s["sma20"] > s["sma60"])   # 短>中>長：多頭排列
    bear = (s["sma5"] < s["sma20"]) & (s["sma20"] < s["sma60"])   # 短<中<長：空頭排列
    ymin, ymax = ax.get_ylim()
    ax.fill_between(x, ymin, ymax, where=bull, color=UP, alpha=0.06, label="多頭排列")
    ax.fill_between(x, ymin, ymax, where=bear, color=DOWN, alpha=0.06, label="空頭排列")
    ax.set_ylim(ymin, ymax)
    ax.set_title("均線的多頭 / 空頭排列（2330.TW）")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, "sma.png")

    # 2. MACD
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("MACD（2330.TW）")
    a2.bar(x, s["macd"] - s["signal"], color=[UP if v >= 0 else DOWN for v in (s["macd"] - s["signal"])], width=1.0)
    a2.plot(x, s["macd"], color=ACCENT, lw=1, label="MACD 快線")
    a2.plot(x, s["signal"], color="#e08a1e", lw=1, label="訊號線")
    a2.axhline(0, color=MUTED, lw=0.8); a2.grid(alpha=0.25); a2.legend(fontsize=8, loc="upper left")
    save(fig, "macd.png")

    # 3. 布林通道
    fig, ax = plt.subplots(figsize=(9, 4.2))
    price_panel(ax, s)
    ax.plot(x, s["bb_mid"], color=ACCENT, lw=0.9, ls="--", label="中軌 SMA20")
    ax.plot(x, s["bb_up"], color=MUTED, lw=0.9, label="上軌")
    ax.plot(x, s["bb_low"], color=MUTED, lw=0.9, label="下軌")
    ax.fill_between(x, s["bb_low"], s["bb_up"], color=ACCENT, alpha=0.07)
    ax.set_title("布林通道（2330.TW）"); ax.legend(fontsize=8, loc="upper left")
    save(fig, "bollinger.png")

    # 4. BIAS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.plot(x, s["sma20"], color=ACCENT, lw=0.9, label="SMA20"); a1.legend(fontsize=8)
    a1.set_title("乖離率 BIAS（2330.TW）")
    a2.bar(x, s["bias20"], color=[UP if v >= 0 else DOWN for v in s["bias20"]], width=1.0)
    a2.axhline(0, color=MUTED, lw=0.8); a2.set_ylabel("BIAS20 %"); a2.grid(alpha=0.25)
    save(fig, "bias.png")

    # 5. RSI
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("RSI 相對強弱（2330.TW）")
    a2.plot(x, s["rsi"], color=ACCENT, lw=1)
    a2.axhline(70, color=UP, lw=0.8, ls="--"); a2.axhline(30, color=DOWN, lw=0.8, ls="--")
    a2.set_ylim(0, 100); a2.set_ylabel("RSI"); a2.grid(alpha=0.25)
    save(fig, "rsi.png")

    # 6. KD
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("KD 隨機指標（2330.TW）")
    a2.plot(x, s["k"], color=ACCENT, lw=1, label="K")
    a2.plot(x, s["d"], color="#e08a1e", lw=1, label="D")
    a2.axhline(80, color=UP, lw=0.8, ls="--"); a2.axhline(20, color=DOWN, lw=0.8, ls="--")
    a2.set_ylim(0, 100); a2.grid(alpha=0.25); a2.legend(fontsize=8, loc="upper left")
    save(fig, "kd.png")

    # 7. CMF
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("CMF 資金流量（2330.TW）")
    a2.fill_between(x, s["cmf"], 0, where=s["cmf"] >= 0, color=UP, alpha=0.6)
    a2.fill_between(x, s["cmf"], 0, where=s["cmf"] < 0, color=DOWN, alpha=0.6)
    a2.axhline(0, color=MUTED, lw=0.8); a2.set_ylabel("CMF20"); a2.grid(alpha=0.25)
    save(fig, "cmf.png")

    # 8. OBV
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("OBV 能量潮（2330.TW）")
    a2.plot(x, s["obv"], color=ACCENT, lw=1); a2.set_ylabel("OBV"); a2.grid(alpha=0.25)
    save(fig, "obv.png")

    # 9. ATR%
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("ATR%（2330.TW）")
    a2.plot(x, s["atr_pct"], color=ACCENT, lw=1); a2.set_ylabel("ATR% (14)"); a2.grid(alpha=0.25)
    save(fig, "atr.png")

    # 10. 報酬率波動率
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), height_ratios=[2, 1], sharex=True)
    price_panel(a1, s); a1.set_title("報酬率波動率（2330.TW）")
    a2.plot(x, s["vol20"], color=ACCENT, lw=1); a2.set_ylabel("波動率 (20)"); a2.grid(alpha=0.25)
    save(fig, "volatility.png")

    # 11. 唐奇安通道
    fig, ax = plt.subplots(figsize=(9, 4.2))
    price_panel(ax, s)
    ax.plot(x, s["dc_up"], color=UP, lw=0.9, label="上軌（20日高）")
    ax.plot(x, s["dc_low"], color=DOWN, lw=0.9, label="下軌（20日低）")
    ax.fill_between(x, s["dc_low"], s["dc_up"], color=ACCENT, alpha=0.06)
    ax.set_title("唐奇安通道（2330.TW）"); ax.legend(fontsize=8, loc="upper left")
    save(fig, "donchian.png")

    print(f"完成，輸出於 {OUT}")


if __name__ == "__main__":
    main()
