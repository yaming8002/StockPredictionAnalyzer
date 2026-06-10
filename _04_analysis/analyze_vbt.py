"""
vbt 回測輸出的數據分析（整併版）
====================================

吃 VbtSingleStrategy.run() 的輸出：
  - trades: DataFrame（stock_id/buy_date/sell_date/buy_price/sell_price/qty/buy_fee/sell_fee/real_pnl）
  - pf:     vbt Portfolio（可選，取其內建統計）

提供：持有天數分布、年度績效、蒙地卡羅、Portfolio 內建統計。
取代舊的 mongo / 自寫引擎時代分析腳本。
"""
import numpy as np
import pandas as pd


def hold_days_stats(trades: pd.DataFrame) -> dict:
    """持有天數（sell_date − buy_date）分布統計。"""
    if trades.empty:
        return {"交易次數": 0}
    days = (pd.to_datetime(trades["sell_date"]) - pd.to_datetime(trades["buy_date"])).dt.days
    return {
        "交易次數": int(len(days)),
        "平均持有天": round(float(days.mean()), 1),
        "中位持有天": float(days.median()),
        "最短": int(days.min()),
        "最長": int(days.max()),
        "P25": float(days.quantile(0.25)),
        "P75": float(days.quantile(0.75)),
    }


def yearly_performance(trades: pd.DataFrame) -> pd.DataFrame:
    """依買入年份分組，統計每年交易次數 / 勝率 / 總損益 / 平均損益。"""
    cols = ["年份", "交易次數", "勝率(%)", "總損益", "平均損益"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    df = trades.copy()
    df["year"] = pd.to_datetime(df["buy_date"]).dt.year
    rows = []
    for year, group in df.groupby("year"):
        pnl = group["real_pnl"]
        rows.append({
            "年份": int(year),
            "交易次數": int(len(group)),
            "勝率(%)": round(float((pnl > 0).mean()) * 100, 2),
            "總損益": round(float(pnl.sum()), 0),
            "平均損益": round(float(pnl.mean()), 0),
        })
    return pd.DataFrame(rows, columns=cols)


def monte_carlo(trades: pd.DataFrame, initial_cash: float = 100_000,
                n_sims: int = 10_000, ruin_ratio: float = 0.5, seed: int = None) -> dict:
    """
    對每筆 real_pnl 做 bootstrap 蒙地卡羅：每次模擬抽樣 len(trades) 筆損益累加成權益曲線。
    回傳最終資金分位數，與「過程權益跌破 initial_cash×ruin_ratio」的破產機率。
    """
    pnl = trades["real_pnl"].to_numpy(dtype=np.float64)
    n = len(pnl)
    if n == 0:
        return {"交易次數": 0}
    rng = np.random.default_rng(seed)
    ruin_level = initial_cash * ruin_ratio
    draws = rng.choice(pnl, size=(n_sims, n), replace=True)
    equity = initial_cash + np.cumsum(draws, axis=1)
    finals = equity[:, -1]
    ruin_prob = float((equity.min(axis=1) < ruin_level).mean())
    return {
        "模擬次數": n_sims,
        "每次抽樣筆數": n,
        "最終資金_中位": round(float(np.median(finals)), 0),
        "最終資金_平均": round(float(finals.mean()), 0),
        "最終資金_P5": round(float(np.percentile(finals, 5)), 0),
        "最終資金_P95": round(float(np.percentile(finals, 95)), 0),
        f"破產機率(<{ruin_ratio:.0%})": round(ruin_prob * 100, 2),
    }


def portfolio_stats(pf) -> dict:
    """直接取 vbt Portfolio 內建統計（總報酬 / 最大回撤 / Sharpe 等）。"""
    return pf.stats().to_dict()


def full_report(result: dict, initial_cash: float = 100_000, pf=None) -> None:
    """印出整份報告。result = VbtSingleStrategy.run() 的回傳（含 trades / summary）。"""
    trades = result["trades"]
    print("== summary ==", result["summary"])
    print("== 持有天數 ==", hold_days_stats(trades))
    print("== 年度績效 ==")
    print(yearly_performance(trades).to_string(index=False))
    print("== 蒙地卡羅 ==", monte_carlo(trades, initial_cash=initial_cash))
    if pf is not None:
        print("== vbt Portfolio.stats ==", portfolio_stats(pf))
