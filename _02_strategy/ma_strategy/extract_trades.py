# -*- coding: utf-8 -*-
"""撈出 MA 60/200 黃金交叉進、死亡交叉出（近5日均量>1000張）的全部逐筆交易。
用途：替 blog 挑一筆有代表性的案例，不改任何既有結果。
"""

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START, GLITCH  # noqa: E402
import numpy as np
import pandas as pd

DATA = common.DATA_DIR
SHORT, LONG = "sma_60", "sma_200"
MIN_LOT = 1000 * 1000        # 近 5 日均量 > 1000 張（＝100 萬股）

rows = []
market = common.load_market(DATA, columns=["open", "volume", SHORT, LONG],
                            start=DEFAULT_START, end=DEFAULT_END, exclude=GLITCH,
                            min_rows=250)
for sid, df in market.items():
    if SHORT not in df.columns:
        continue
    s, l = df[SHORT], df[LONG]
    golden = (s > l) & (s.shift(1) <= l.shift(1))
    death = (s < l) & (s.shift(1) >= l.shift(1))
    liq = df["volume"].rolling(5).mean() > MIN_LOT      # 訊號日當天的近5日均量
    buy_sig = golden & liq
    idx = df.index
    o = df["open"].values
    pos = -1
    for i in range(len(df) - 1):
        if pos < 0:
            if buy_sig.iat[i]:
                pos = i + 1                              # 隔日開盤買
        else:
            if death.iat[i]:
                sell = i + 1                             # 隔日開盤賣
                bp, sp = o[pos], o[sell]
                if bp > 0:
                    rows.append((sid, idx[pos], idx[sell], bp, sp,
                                 (sp - bp) / bp * 100, (idx[sell] - idx[pos]).days))
                pos = -1

t = pd.DataFrame(rows, columns=["stock", "buy", "sell", "buy_px", "sell_px", "ret", "hold"])
# 輸出落在 repo 內的 result/（不進版控），不要寫到執行環境的暫存目錄——
# 那種路徑換一場就不存在，等於沒存。
OUT = common.result_dir("ma_strategy", "ma_cross")
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "ma_60_200_trades.csv")
t.to_csv(out_path, index=False, encoding="utf-8-sig")
print("交易數", len(t), "→", out_path)
print("勝率 %.2f%%" % ((t.ret > 0).mean() * 100))
print("獲利平均 %.2f%%  虧損平均 %.2f%%" % (t.ret[t.ret > 0].mean(), t.ret[t.ret <= 0].mean()))
print("中位數 %.2f%%  平均持有 %.1f 天" % (t.ret.median(), t.hold.mean()))
