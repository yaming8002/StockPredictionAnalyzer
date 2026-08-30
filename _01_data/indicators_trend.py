"""
技術指標（一）趨勢：以 SMA 為核心的均線家族
================================================

對應文章〈常見技術指標（一）趨勢〉：https://stockanalyzer.sailforthlab.dev/posts/2026/06/indicators-trend/
一組純函式：輸入含 OHLCV 的 DataFrame，回傳「多了指標欄位」的 DataFrame。

指標：SMA / EMA / MACD / 布林通道 / BIAS（乖離率）/ ADX（趨勢強度）/ 拋物線 SAR
"""
import numpy as np
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


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX（平均趨向指標，Wilder）：衡量「趨勢有多強」，不分多空。
    常見用法是設一道門檻（例如 25），低於門檻視為盤整、不進場。

    計算三步：方向變動 DM → 方向指標 DI → 兩者的離散度 DX 再做一次平滑。
    平滑一律用 Wilder 的遞迴平均（等價於 alpha=1/period 的 EMA）。
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    up, down = high.diff(), -low.diff()
    # 只有「漲幅大於跌幅且為正」才算上升方向變動，反之亦然；其餘記 0
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    # 多空方向指標差距越大＝方向越明確；兩者都接近時 DX 趨近 0（盤整）
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
    data["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return data


def calculate_psar(data: pd.DataFrame, af_start: float = 0.02,
                   af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """
    拋物線 SAR（Wilder，Stop And Reverse）：從價格下方（多頭）或上方（空頭）追價的停損線。

    每當波段創新極值，加速因子 af 就加一級（上限 af_max），線越追越緊——
    這是它和「固定寬度移動停損」最大的差別。跌破（多頭）或突破（空頭）即翻向並重設。
    產出兩欄：psar（線的位置）、psar_flip_down（由多翻空的當日＝True，可直接當出場訊號）。

    （本質路徑相依、需逐根掃描，故以迴圈實作。）
    """
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    n = len(high)
    sar_out = np.full(n, np.nan)
    flip_down = np.zeros(n, dtype=bool)
    if n >= 2:
        rising = True          # 起始假設多頭；前幾根的方向會很快被市場修正
        sar = low[0]
        ep = high[0]           # 極值點：多頭時為波段最高、空頭時為波段最低
        af = af_start
        for i in range(1, n):
            sar = sar + af * (ep - sar)
            if rising:
                # Wilder 原著限制：SAR 不得高於前兩根的低點，避免停損穿進近期價區
                sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
                if low[i] < sar:                      # 跌破 → 翻空並重設
                    flip_down[i] = True
                    rising = False
                    sar, ep, af = ep, low[i], af_start
                elif high[i] > ep:                    # 續創新高 → 加速
                    ep = high[i]
                    af = min(af + af_step, af_max)
            else:
                sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
                if high[i] > sar:                     # 突破 → 翻多並重設
                    rising = True
                    sar, ep, af = ep, high[i], af_start
                elif low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
            sar_out[i] = sar
    data["psar"] = sar_out
    data["psar_flip_down"] = flip_down
    return data
