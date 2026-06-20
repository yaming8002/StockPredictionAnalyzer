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
from numba import njit


@njit(cache=True)
def _mc_core(pnl, n_sims, init_cash, ruin_level, seed):
    """
    逐路徑 bootstrap（省記憶體，不建 n_sims×n 大矩陣）。
    每路徑抽樣 n 筆損益累加成權益，量：最終資金、最大連敗、最大回撤、是否破產。
    """
    np.random.seed(seed)
    n = len(pnl)
    finals = np.empty(n_sims)
    streaks = np.empty(n_sims)        # 每路徑最大連續虧損次數
    max_dds = np.empty(n_sims)        # 每路徑最大回撤（比例）
    ruin = 0
    for s in range(n_sims):
        equity = init_cash
        peak = init_cash
        mdd = 0.0
        cur = 0
        mx = 0
        hit = False
        for _ in range(n):
            x = pnl[np.random.randint(0, n)]
            if x < 0.0:
                cur += 1
                if cur > mx:
                    mx = cur
            else:
                cur = 0
            equity += x
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > mdd:
                mdd = dd
            if equity < ruin_level:
                hit = True
        finals[s] = equity
        streaks[s] = mx
        max_dds[s] = mdd
        if hit:
            ruin += 1
    return finals, streaks, max_dds, ruin


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
                n_sims: int = 10_000, ruin_ratio: float = 0.5, seed: int = 42) -> dict:
    """
    對每筆 real_pnl 做 bootstrap 蒙地卡羅：每次模擬抽樣 len(trades) 筆損益累加成權益曲線。
    逐路徑迴圈（numba，省記憶體；舊版 n_sims×n 大矩陣在大 N 會 OOM）。
    回傳：最終資金分位、破產機率、**最大連敗 S 分布**、**最大回撤分布**。
    """
    pnl = trades["real_pnl"].to_numpy(dtype=np.float64)
    n = len(pnl)
    if n == 0:
        return {"交易次數": 0}
    ruin_level = initial_cash * ruin_ratio
    finals, streaks, max_dds, ruin = _mc_core(pnl, n_sims, float(initial_cash),
                                              float(ruin_level), int(seed))
    return {
        "模擬次數": n_sims,
        "每次抽樣筆數": n,
        "最終資金_中位": round(float(np.median(finals)), 0),
        "最終資金_平均": round(float(finals.mean()), 0),
        "最終資金_P5": round(float(np.percentile(finals, 5)), 0),
        "最終資金_P95": round(float(np.percentile(finals, 95)), 0),
        f"破產機率(<{ruin_ratio:.0%})": round(ruin * 100.0 / n_sims, 2),
        # 最大連敗 S（餵資金管理框架）
        "最大連敗_中位": int(np.median(streaks)),
        "最大連敗_P95": int(np.percentile(streaks, 95)),
        "最大連敗_P99": int(np.percentile(streaks, 99)),
        "最大連敗_極值": int(streaks.max()),
        # 最大回撤（bootstrap 路徑）
        "最大回撤%_中位": round(float(np.median(max_dds)) * 100, 1),
        "最大回撤%_P95": round(float(np.percentile(max_dds, 95)) * 100, 1),
        "最大回撤%_極值": round(float(max_dds.max()) * 100, 1),
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
