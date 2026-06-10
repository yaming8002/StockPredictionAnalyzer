"""
技術指標（三）波動與突破：風險多大、盤整還是要噴出
====================================================

對應文章〈常見技術指標（三）波動與突破〉。
一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了指標欄位」的 DataFrame。

指標：ATR%（波動幅度比例）/ 報酬率波動率 / 唐奇安通道（Donchian）
"""
import numpy as np
import pandas as pd


def calculate_atr_pct(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR%（平均真實波幅佔股價比例）。
    真實波幅 TR = max(當日高低差, |高−昨收|, |低−昨收|)，後兩項涵蓋跳空。
    用比例而非絕對值，方便跨不同價位的股票比較波動。
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    data[f"atr_pct_{window}"] = (atr / close).fillna(0).round(4)
    return data


def calculate_return_volatility(data: pd.DataFrame, window: int = 20, scale: float = 10.0) -> pd.DataFrame:
    """
    報酬率波動率：每日 log 報酬的標準差（統計角度的風險）。
    scale 為放大倍率方便觀察；要年化乘上 √252。
    """
    close = data["close"]
    log_return = (close / close.shift(1)).apply(np.log)
    vol = log_return.rolling(window=window, min_periods=window).std()
    data[f"volatility_{window}"] = (vol * scale).fillna(0).round(4)
    return data


def calculate_donchian(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    唐奇安通道：最近 window 日的最高 / 最低框成的區間。
    價在箱內 = 盤整；突破上軌 = 趨勢可能啟動（海龜法則核心）。
    """
    data[f"donchian_upper_{window}"] = data["high"].rolling(window).max()
    data[f"donchian_lower_{window}"] = data["low"].rolling(window).min()
    return data
