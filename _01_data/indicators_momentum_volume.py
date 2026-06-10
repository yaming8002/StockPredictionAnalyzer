"""
技術指標（二）量能與動能：誰在買、超不超買
================================================

對應文章〈常見技術指標（二）量能與動能〉。
一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了指標欄位」的 DataFrame。

指標：RSI / KD / CMF（資金流量）/ OBV（能量潮）
"""
import numpy as np
import pandas as pd


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI（相對強弱）：比較最近漲跌力道，壓進 0~100。> 70 偏超買、< 30 偏超賣。"""
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["rsi"] = (100 - (100 / (1 + rs))).round(4)
    return data


def calculate_kd(data: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """
    KD 隨機指標：看收盤落在最近 n 日高低區間的位置。
    RSV = (收盤 − n日低) / (n日高 − n日低) ×100；K 為 RSV 平滑、D 為 K 平滑。
    （com=smooth−1 ≈ 取 1/3 新值、2/3 舊值，即傳統 KD）
    """
    low_n = data["low"].rolling(n).min()
    high_n = data["high"].rolling(n).max()
    rsv = (data["close"] - low_n) / (high_n - low_n) * 100
    data["k"] = rsv.ewm(com=k_smooth - 1, adjust=False).mean().round(4)
    data["d"] = data["k"].ewm(com=d_smooth - 1, adjust=False).mean().round(4)
    return data


def calculate_cmf(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    CMF（Chaikin Money Flow，資金流量）。
    收盤越靠近當日高點越偏買盤、靠近低點偏賣盤，再用成交量加權。
    > 0 偏買盤、< 0 偏賣盤。
    """
    hl_range = (data["high"] - data["low"]).replace(0, np.nan)
    mfm = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / hl_range
    mfv = mfm * data["volume"]
    cmf = mfv.rolling(window).sum() / data["volume"].rolling(window).sum()
    data[f"cmf_{window}"] = cmf.fillna(0).round(4)
    return data


def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
    """
    OBV（能量潮）：上漲日加當天量、下跌日減量，累積成一條線。
    絕對數值不重要，看的是方向與背離。
    """
    direction = np.sign(data["close"].diff()).fillna(0)   # 漲=+1 跌=−1 平=0
    data["obv"] = (direction * data["volume"]).cumsum()
    return data
