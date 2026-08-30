"""
技術指標（三）波動與突破：風險多大、盤整還是要噴出
====================================================

對應文章〈常見技術指標（三）波動與突破〉：https://stockanalyzer.sailforthlab.dev/posts/2026/06/indicators-volatility/
一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了指標欄位」的 DataFrame。

指標：ATR（絕對值）/ ATR%（波動幅度比例）/ 報酬率波動率 / 唐奇安通道（Donchian）/ Supertrend
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


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR（平均真實波幅，絕對金額）。TR 定義與 calculate_atr_pct 相同，差別在不轉成比例、不四捨五入。

    要拿 ATR 去算停損價（例如「進場價 − 2×ATR」）時必須用這個版本：
    atr_pct 是比例且只留 4 位小數，回推成金額會有可觀誤差。
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    data[f"atr_{window}"] = tr.rolling(window=window, min_periods=window).mean()
    return data


def calculate_supertrend(data: pd.DataFrame, period: int = 10,
                         multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend：以「中價 ± multiplier × ATR」畫出的趨勢通道線。

    上下軌只能往「對持倉有利」的方向收（多頭時下軌只升不降），直到價格突破才重設；
    趨勢沿用前一根，收盤穿到另一側才翻向。與唐奇安通道的差別在寬度會隨波動自動調整。
    產出三欄：supertrend（當前那條線）、supertrend_up（趨勢是否為多）、
    supertrend_flip_down（由多翻空的當日＝True，可直接當出場訊號）。

    ATR 用 Wilder 平滑（該指標的通用定義），與 calculate_atr 的簡單平均不同，故在此另算。
    （本質路徑相依、需逐根掃描，故以迴圈實作。）
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr.iloc[:period] = np.nan          # 暖身期不產訊號

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    n = len(c)
    line = np.full(n, np.nan)
    is_up = np.zeros(n, dtype=bool)
    flip_down = np.zeros(n, dtype=bool)
    final_upper, final_lower = np.inf, -np.inf
    trend = 1                            # 1=多、-1=空；起始視為多
    for i in range(n):
        if np.isnan(a[i]):
            continue
        mid = 0.5 * (h[i] + l[i])
        basic_upper = mid + multiplier * a[i]
        basic_lower = mid - multiplier * a[i]
        prev_c = c[i - 1] if i > 0 else c[i]
        # 軌道只能往內收；被昨收突破才重設
        final_upper = basic_upper if (basic_upper < final_upper or prev_c > final_upper) else final_upper
        final_lower = basic_lower if (basic_lower > final_lower or prev_c < final_lower) else final_lower
        new_trend = trend
        if c[i] > final_upper:
            new_trend = 1
        elif c[i] < final_lower:
            new_trend = -1
        if trend == 1 and new_trend == -1:
            flip_down[i] = True
        trend = new_trend
        is_up[i] = trend == 1
        line[i] = final_lower if trend == 1 else final_upper
    data["supertrend"] = line
    data["supertrend_up"] = is_up
    data["supertrend_flip_down"] = flip_down
    return data
