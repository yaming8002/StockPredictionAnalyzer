"""
vbt 策略套件 common — 各策略共用的低階工具
================================================

只放「與引擎無關」的純函式：台股 tick 進位、精確費用重建、欄位驗證、summary 組裝。
策略基底（single）以「呼叫這些函式」共用，不靠類別繼承。
"""
import numpy as np
import pandas as pd

# 台股費率
COMMISSION = 0.001425      # 手續費率
DUES = 0.003               # 證交稅（賣方）
MIN_COMMISSION = 20.0      # 單筆最低手續費

# 統一小寫欄位（與資料層約定一致）
OHLCV = ("open", "high", "low", "close", "volume")


def tw_tick_arr(prices) -> np.ndarray:
    """
    台股升降單位「無條件進位」（向量化）。買賣價一律過此函式。
    NaN 會原樣保留（np.ceil(nan)=nan）。
    """
    p = np.asarray(prices, dtype=np.float64)
    tick = np.select(
        [p < 10, p < 50, p < 100, p < 500, p < 1000],
        [0.01, 0.05, 0.1, 0.5, 1.0],
        default=5.0,
    )
    return np.round(np.ceil(p / tick) * tick, 2)


def reconstruct_fees(buy_price, sell_price, qty):
    """
    精確台股費用重建（vbt 純 ratio 表達不了 min 20 + 賣方證交稅，故出 trades 後校正）。
    回傳 (buy_fee, sell_fee)，皆 np.ceil 無條件進位。
    """
    buy_price = np.asarray(buy_price, dtype=np.float64)
    sell_price = np.asarray(sell_price, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)

    buy_amt = buy_price * qty
    sell_amt = sell_price * qty
    buy_fee = np.ceil(np.maximum(buy_amt * COMMISSION, MIN_COMMISSION))
    sell_fee = np.ceil(np.maximum(sell_amt * COMMISSION, MIN_COMMISSION) + sell_amt * DUES)
    return buy_fee, sell_fee


def net_pnl(buy_price, sell_price, qty):
    """每筆已扣台股費用的淨損益。"""
    buy_fee, sell_fee = reconstruct_fees(buy_price, sell_price, qty)
    bp = np.asarray(buy_price, dtype=np.float64)
    sp = np.asarray(sell_price, dtype=np.float64)
    q = np.asarray(qty, dtype=np.float64)
    return (sp - bp) * q - buy_fee - sell_fee


def ensure_columns(df: pd.DataFrame, required=OHLCV) -> None:
    """檢查必要欄位齊全；缺欄直接報錯（不靜默吞）。"""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"資料缺少必要欄位: {missing}（現有: {list(df.columns)}）")


def summarize_trades(records: pd.DataFrame) -> dict:
    """
    從含 buy_price/sell_price/qty/real_pnl 的 trades 表組 summary。
    欄名刻意對齊既有 summary（方便下游分析工具）。
    """
    n = len(records)
    if n == 0:
        return {"交易次數": 0, "勝率(%)": 0.0, "平均獲利金額": 0.0,
                "平均虧損金額": 0.0, "期望報酬值(EV)": 0.0, "總獲利": 0.0}

    pnl = records["real_pnl"].to_numpy(dtype=np.float64)
    win = pnl[pnl > 0]
    lose = pnl[pnl < 0]
    win_rate = len(win) / n
    avg_win = float(win.mean()) if len(win) else 0.0
    avg_lose = float(lose.mean()) if len(lose) else 0.0
    return {
        "交易次數": n,
        "勝率(%)": round(win_rate * 100, 2),
        "平均獲利金額": round(avg_win, 2),
        "平均虧損金額": round(avg_lose, 2),
        "期望報酬值(EV)": round(win_rate * avg_win + (1 - win_rate) * avg_lose, 2),
        "總獲利": round(float(pnl.sum()), 2),
    }
