"""
資料優化 + 技術指標計算（聚合入口）
====================================

指標依「衡量什麼」分三組，各自一個模組（對應系列文章三篇）：
  - indicators_trend           趨勢：SMA / EMA / MACD / 布林 / BIAS
  - indicators_momentum_volume 量能動能：RSI / KD / CMF / OBV
  - indicators_volatility      波動：ATR% / 報酬率波動率 / 唐奇安通道

本檔 re-export 全部函式，並提供 add_all_indicators 一次套用，方便單一匯入。

執行範例（讀一檔資料 → 套用全部指標 → 存回）：
  python _01_data/stock_technical.py
"""
import os
import sys

import pandas as pd

# 同目錄的指標模組（讓 `python _01_data/stock_technical.py` 直接執行時可解析）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators_trend import (  # noqa: E402
    calculate_sma, calculate_ema, calculate_macd,
    calculate_bollinger_bands, calculate_bias,
)
from indicators_momentum_volume import (  # noqa: E402
    calculate_rsi, calculate_kd, calculate_cmf, calculate_obv,
)
from indicators_volatility import (  # noqa: E402
    calculate_atr_pct, calculate_return_volatility, calculate_donchian,
)


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """一次套用全部指標（示範用，可依需求挑選）。"""
    # 趨勢
    data = calculate_sma(data, 20)
    data = calculate_sma(data, 60)
    data = calculate_ema(data, 20)
    data = calculate_macd(data)
    data = calculate_bollinger_bands(data)
    data = calculate_bias(data, 20)
    # 量能 / 動能
    data = calculate_rsi(data)
    data = calculate_kd(data)
    data = calculate_cmf(data)
    data = calculate_obv(data)
    # 波動
    data = calculate_atr_pct(data)
    data = calculate_return_volatility(data)
    data = calculate_donchian(data, 20)
    return data


if __name__ == "__main__":
    # 範例：讀一檔下載好的資料 → 套用全部指標 → 存回
    # （需先用 download_stock.py 下載 2330.TW）
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "2330.TW.csv")

    if not os.path.exists(csv_path):
        print(f"找不到 {csv_path}，請先執行 download_stock.py 下載資料")
    else:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df = add_all_indicators(df)
        out_path = os.path.join(data_dir, "2330.TW_with_indicators.csv")
        df.to_csv(out_path, encoding="utf-8-sig")
        print(f"已計算指標並存入 {out_path}")
        print(df.tail())
