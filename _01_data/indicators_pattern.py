"""
技術指標（型態類）：價格結構 / 擺動轉折
================================================

獨立於趨勢(SMA家族) / 波動 / 動能量能 三組「數值型指標」之外，
本組處理「**型態 / 結構**」——以價格擺動為基礎的轉折點，供結構判斷（如 CHoCH、波段高低點）使用。

一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了型態欄位」的 DataFrame。

指標：ZigZag 轉折（擺動高/低點）
"""
import numpy as np
import pandas as pd


def calculate_zigzag(data: pd.DataFrame, pct: float = 0.02) -> pd.DataFrame:
    """
    ZigZag 轉折點：以收盤價、回檔/反彈門檻 pct 標出擺動高/低點（結構判斷用，如 CHoCH）。
    - 在「自波段高點回檔達 pct」確認當日，填入該段擺動高點值 → zigzag_turn_high。
    - 在「自波段低點反彈達 pct」確認當日，填入該段擺動低點值 → zigzag_turn_low。
    - 其餘日為 NaN。pct 預設 2%。
    （本質路徑相依、需逐根掃描，故以迴圈實作。）
    """
    close = data["close"].to_numpy(dtype=float)
    n = len(close)
    turn_high = np.full(n, np.nan)
    turn_low = np.full(n, np.nan)
    if n:
        direction = 1          # 1=上升腿(找高點)、-1=下降腿(找低點)
        ext = close[0]
        for i in range(1, n):
            c = close[i]
            if direction == 1:
                if c >= ext:
                    ext = c                          # 續創高
                elif c <= ext * (1 - pct):
                    turn_high[i] = ext               # 確認擺動高點
                    direction = -1
                    ext = c
            else:
                if c <= ext:
                    ext = c                          # 續創低
                elif c >= ext * (1 + pct):
                    turn_low[i] = ext                # 確認擺動低點
                    direction = 1
                    ext = c
    data["zigzag_turn_high"] = turn_high
    data["zigzag_turn_low"] = turn_low
    return data
