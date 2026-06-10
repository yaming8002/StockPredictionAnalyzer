"""
技術指標（一）趨勢：以 SMA 為核心的均線家族
================================================

對應文章〈常見技術指標（一）趨勢〉。
一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了指標欄位」的 DataFrame。

指標：SMA / EMA / MACD / 布林通道 / BIAS（乖離率）
"""
import pandas as pd


def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """簡單移動平均（SMA）：最近 window 天收盤價的平均。"""
    data[f"sma_{window}"] = data["close"].rolling(window=window).mean().round(4)
    return data


def calculate_ema(data: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """指數移動平均（EMA），近期權重較高、對轉折反應更快。"""
    data[f"ema_{span}"] = data["close"].ewm(span=span, adjust=False).mean().round(4)
    return data


def calculate_macd(data: pd.DataFrame, short_period: int = 12,
                   long_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """MACD：快線（短 EMA − 長 EMA）與訊號線（MACD 的 EMA）。"""
    short_ema = data["close"].ewm(span=short_period, adjust=False).mean()
    long_ema = data["close"].ewm(span=long_period, adjust=False).mean()
    data["macd"] = (short_ema - long_ema).round(4)
    data["signal_line"] = data["macd"].ewm(span=signal_period, adjust=False).mean().round(4)
    return data


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """布林通道：移動平均（中軌）上下各加減 num_std 倍標準差。"""
    sma = data["close"].rolling(window=window).mean()
    std = data["close"].rolling(window=window).std()
    data["bollinger_upper"] = (sma + num_std * std).round(4)
    data["bollinger_lower"] = (sma - num_std * std).round(4)
    return data


def calculate_bias(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    乖離率（BIAS）：收盤價偏離均線的百分比。
    正乖離過大 = 短線漲過頭、易回檔；負乖離過大 = 跌過頭、易反彈。
    """
    sma = data["close"].rolling(window=window).mean()
    data[f"bias_{window}"] = ((data["close"] - sma) / sma * 100).round(4)
    return data
