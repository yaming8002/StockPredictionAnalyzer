import pandas as pd


def calculate_sma(data, window=20):
    """計算簡單移動平均線 (SMA)"""
    data[f"SMA_{window}"] = data["Close"].rolling(window=window).mean()
    return data


def calculate_ema(data, span=20):
    """計算指數移動平均線 (EMA)"""
    data[f"EMA_{span}"] = data["Close"].ewm(span=span, adjust=False).mean()
    return data


def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """計算 MACD 指標"""
    short_ema = data["Close"].ewm(span=short_period, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_period, adjust=False).mean()
    data["MACD"] = short_ema - long_ema
    data["Signal_Line"] = data["MACD"].ewm(span=signal_period, adjust=False).mean()
    return data


def calculate_rsi(data, period=14):
    """計算 RSI 指標"""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def calculate_bollinger_bands(data, window=20, num_std=2):
    """計算布林通道 (Bollinger Bands)"""
    sma = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    data["Bollinger_Upper"] = sma + (num_std * std)
    data["Bollinger_Lower"] = sma - (num_std * std)
    return data
