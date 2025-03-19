import pandas as pd


def calculate_sma(data: pd.DataFrame, window=20):
    """計算簡單移動平均線 (SMA)"""
    data[f"sma_{window}"] = data["close"].rolling(window=window).mean()
    return data


def calculate_ema(data: pd.DataFrame, span=20):
    """計算指數移動平均線 (EMA)"""
    data[f"ema_{span}"] = data["close"].ewm(span=span, adjust=False).mean()
    return data


def calculate_macd(data: pd.DataFrame, short_period=12, long_period=26, signal_period=9):
    """計算 MACD 指標"""
    short_ema = data["close"].ewm(span=short_period, adjust=False).mean()
    long_ema = data["close"].ewm(span=long_period, adjust=False).mean()
    data["MACD"] = short_ema - long_ema
    data["signal_Line"] = data["MACD"].ewm(span=signal_period, adjust=False).mean()
    return data


def calculate_rsi(data: pd.DataFrame, period=14):
    """計算 RSI 指標"""
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def calculate_bollinger_bands(data: pd.DataFrame, window=20, num_std=2):
    """計算布林通道 (Bollinger Bands)"""
    sma = data["close"].rolling(window=window).mean()
    std = data["close"].rolling(window=window).std()
    data["bollinger_Upper"] = sma + (num_std * std)
    data["bollinger_Lower"] = sma - (num_std * std)
    return data
