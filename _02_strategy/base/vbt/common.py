"""
vbt 策略套件 common — 各策略共用的低階工具
================================================

只放「與引擎無關」的純函式：台股 tick 進位、精確費用重建、欄位驗證、summary 組裝。
策略基底（single）以「呼叫這些函式」共用，不靠類別繼承。
"""
import numpy as np
import pandas as pd
from scipy import stats

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


def trim_outliers(series: pd.Series):
    """
    MAD 去極值（k=3）：以中位數 ± 3×MAD 為界，回傳 (filtered, lower, upper)。
    對齊 lab strategy_runner 的 trim_outliers（auto 模式，不做 win/lose 夾擠）。
    """
    if len(series) == 0:
        return series, None, None
    median = series.median()
    mad = (series - median).abs().median()
    lower = median - 3 * mad
    upper = median + 3 * mad
    filtered = series[(series >= lower) & (series <= upper)]
    return filtered, lower, upper


def confidence_interval(series: pd.Series):
    """95% t 分布信賴區間（對齊 lab strategy_runner）。樣本 < 2 回 (nan, nan)。"""
    if len(series) < 2:
        return (np.nan, np.nan)
    mean = np.mean(series)
    std_err = stats.sem(series)
    ci = stats.t.interval(0.95, len(series) - 1, loc=mean, scale=std_err)
    return (round(float(ci[0]), 2), round(float(ci[1]), 2))


def _trim_block(win_df: pd.DataFrame, lose_df: pd.DataFrame, win_rate: float) -> dict:
    """
    去極值（MAD）後的指標。「去極值」定義：在獲利、虧損子集各自移除
    『淨損益落在 中位數 ± 3×MAD 外』的交易，再對存活交易計各項平均
    （金額 / 報酬率 / 持有天數 / EV 同屬一個族群，敘事一致）。
    """
    def keep(sub: pd.DataFrame):
        """回傳 (存活交易 df, 下界, 上界)；空集合回 (sub, None, None)。"""
        if sub.empty:
            return sub, None, None
        _, low, high = trim_outliers(sub["profit"])
        kept = sub[(sub["profit"] >= low) & (sub["profit"] <= high)]
        return kept, low, high

    win_keep, win_low, win_high = keep(win_df)
    lose_keep, lose_low, lose_high = keep(lose_df)
    kept_all = pd.concat([win_keep, lose_keep])

    avg_win_trim = float(win_keep["profit"].mean()) if not win_keep.empty else 0.0
    avg_lose_trim = float(lose_keep["profit"].mean()) if not lose_keep.empty else 0.0
    avg_win_rate_trim = float(win_keep["profit_rate"].mean()) if not win_keep.empty else 0.0
    avg_lose_rate_trim = float(lose_keep["profit_rate"].mean()) if not lose_keep.empty else 0.0
    avg_hold_trim = float(kept_all["hold_days"].mean()) if not kept_all.empty else 0.0
    expect_trim = win_rate * avg_win_trim + (1 - win_rate) * avg_lose_trim

    return {
        "IQR獲利下限": round(win_low, 2) if win_low is not None else np.nan,
        "IQR獲利上限": round(win_high, 2) if win_high is not None else np.nan,
        "IQR虧損下限": round(lose_low, 2) if lose_low is not None else np.nan,
        "IQR虧損上限": round(lose_high, 2) if lose_high is not None else np.nan,
        "排除極值後平均獲利金額": round(avg_win_trim, 2),
        "排除極值後平均虧損金額": round(avg_lose_trim, 2),
        "排除極值後平均獲利報酬率(%)": round(avg_win_rate_trim, 2),
        "排除極值後平均虧損報酬率(%)": round(avg_lose_rate_trim, 2),
        "排除極值後平均持有天數": round(avg_hold_trim, 2),
        "排除極值後期望報酬值(EV,Trim)": round(expect_trim, 2),
    }


def _empty_summary() -> dict:
    """無交易時的零值 summary（欄位與正常情況一致，供下游對齊）。"""
    keys = ["交易次數", "勝率(%)", "平均獲利金額", "平均虧損金額",
            "平均獲利報酬率(%)", "平均虧損報酬率(%)", "最大獲利", "最大虧損",
            "最大獲利報酬率(%)", "最大虧損報酬率(%)", "平均持有天數",
            "期望報酬值(EV)", "總獲利", "IQR獲利下限", "IQR獲利上限",
            "IQR虧損下限", "IQR虧損上限", "排除極值後平均獲利金額",
            "排除極值後平均虧損金額", "排除極值後平均獲利報酬率(%)",
            "排除極值後平均虧損報酬率(%)", "排除極值後平均持有天數",
            "排除極值後期望報酬值(EV,Trim)",
            "獲利信賴區間下限(95%)", "獲利信賴區間上限(95%)",
            "虧損信賴區間下限(95%)", "虧損信賴區間上限(95%)"]
    summary = {k: 0.0 for k in keys}
    summary["交易次數"] = 0
    return summary


def summarize_trades(records: pd.DataFrame) -> dict:
    """
    從 trades 表組完整 summary（指標對齊 lab strategy_runner）。
    金額類用淨損益 real_pnl（已扣台股費稅）；報酬率類用買賣價 gross %。
    排除淨損益=0 的交易（與 lab 一致）。
    """
    if len(records) == 0:
        return _empty_summary()

    df = records.copy()
    df["profit"] = pd.to_numeric(df["real_pnl"], errors="coerce").fillna(0.0)
    df = df[df["profit"] != 0]
    n = len(df)
    if n == 0:
        return _empty_summary()

    # 報酬率：買賣價毛報酬（不含費用，對齊 lab）；持有天數：賣出 − 買入
    df["profit_rate"] = (df["sell_price"] - df["buy_price"]) / df["buy_price"] * 100
    df["hold_days"] = (pd.to_datetime(df["sell_date"]) - pd.to_datetime(df["buy_date"])).dt.days

    win_df = df[df["profit"] > 0]
    lose_df = df[df["profit"] < 0]
    win_rate = len(win_df) / n
    avg_win = float(win_df["profit"].mean()) if not win_df.empty else 0.0
    avg_lose = float(lose_df["profit"].mean()) if not lose_df.empty else 0.0
    avg_win_rate = float(win_df["profit_rate"].mean()) if not win_df.empty else 0.0
    avg_lose_rate = float(lose_df["profit_rate"].mean()) if not lose_df.empty else 0.0
    expect_value = win_rate * avg_win + (1 - win_rate) * avg_lose

    # 去極值（MAD）後各項指標 + 95% 信賴區間
    trim = _trim_block(win_df, lose_df, win_rate)
    win_ci = confidence_interval(win_df["profit"]) if not win_df.empty else (np.nan, np.nan)
    lose_ci = confidence_interval(lose_df["profit"]) if not lose_df.empty else (np.nan, np.nan)

    base = {
        "交易次數": n,
        "勝率(%)": round(win_rate * 100, 2),
        "平均獲利金額": round(avg_win, 2),
        "平均虧損金額": round(avg_lose, 2),
        "平均獲利報酬率(%)": round(avg_win_rate, 2),
        "平均虧損報酬率(%)": round(avg_lose_rate, 2),
        "最大獲利": round(float(df["profit"].max()), 2),
        "最大虧損": round(float(df["profit"].min()), 2),
        "最大獲利報酬率(%)": round(float(df["profit_rate"].max()), 2),
        "最大虧損報酬率(%)": round(float(df["profit_rate"].min()), 2),
        "平均持有天數": round(float(df["hold_days"].mean()), 2),
        "期望報酬值(EV)": round(expect_value, 2),
        "總獲利": round(float(df["profit"].sum()), 2),
    }
    ci = {
        "獲利信賴區間下限(95%)": win_ci[0],
        "獲利信賴區間上限(95%)": win_ci[1],
        "虧損信賴區間下限(95%)": lose_ci[0],
        "虧損信賴區間上限(95%)": lose_ci[1],
    }
    return {**base, **trim, **ci}
