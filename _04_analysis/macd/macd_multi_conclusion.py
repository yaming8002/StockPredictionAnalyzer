"""
MACD 系列結論：五組交易策略的多股回測（2015–2025）對 0050 買進持有含息
=========================================================================
**為什麼期間改成 2015–2025，不是回測用的 2002–2025**：0050 只有 2015 年之後的
資料尺度一致、含息算得準（parquet 在 2014-01-02 有一次假分割、更早的配息資料也不
完整），拿 2002 起的策略績效去對一條算不準的基準線沒有意義。指標暖身仍吃全史，
只是把「開始下單」推到 2015-01-01（`build_panel` 先算指標再切期間，暖身不受影響）。

**權益曲線用逐日市值計價**（現金 ＋ 持倉當日市值），不是只看已實現損益——已實現
口徑看不到未平倉部位的浮虧，回撤會被低估、顯得比實際樂觀。基底的 `run_panel`
直接回傳 vbt 的組合權益曲線，這裡不另外重算一條。

**判定基準＝報酬回撤比（年化報酬率% ÷ 最大回撤%）**：單看年化會偏好把風險放大的
做法，單看回撤會偏好不交易，兩者相除才問得出「這套值不值得取代長抱 0050」。

執行：
    python _04_analysis/macd/macd_multi_conclusion.py [--limit N]
輸出：result/macd_multi/macd_multi_conclusion.csv ＋ 終端表。
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common  # noqa: E402
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD
from _04_analysis.macd.macd_multi_driver import (DATA, INIT_CASH, OUT,
                                            PCT_MIN_INVEST, S_BY_STRATEGY,
                                            load_all, units)

WIN_START = "2015-01-01"
BENCH = "0050.TW"
DIVIDENDS = common.DIVIDEND_FILE


def curve_stats(eq: np.ndarray, years: float):
    """資金倍數、年化報酬率%、最大回撤%（都用逐日市值權益）。"""
    mult = eq[-1] / eq[0]
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return round(mult, 2), round(cagr, 2), round(dd, 1)


def bench_0050(cal: pd.DatetimeIndex, years: float):
    """
    0050 買進持有含息：首日收盤買進，配息當日以收盤價再投入，抱到期末。

    ⚠️ 價格與配息**都已經是還原分割後的單位，不要再自己除以 4**。0050 在 2025-06
    真的 1 拆 4，parquet 的價格已還原（2015-01-05 收 16.64 ＝ 當時實際 66.55 的 1/4），
    而 `dividend_actions.parquet` 的金額同樣已還原——2022-01 的實際現金股利是 3.2 元，
    表裡記 0.8000（＝3.2÷4）；2022-07 實際 1.8，表裡記 0.45。兩邊同一套單位。
    先前多除了一次 4，算出 ×4.31／14.23%，與錨點 ×5.51／16.81%／33.8% 差了 28% 才抓到。
    殖利率也可以當快篩：不調整時期間配息合計 9.005、除以 11 年對平均價 ~35 ＝ 每年約
    2.3%（符合 0050 實際），多除一次只剩 0.65%（明顯不合理）。
    """
    px = pd.read_parquet(os.path.join(DATA, f"{BENCH}.parquet"))
    close = px["close"].reindex(cal).ffill().to_numpy(np.float64)
    dv = pd.read_parquet(DIVIDENDS)
    dv = dv[(dv["stock_id"] == BENCH) & (dv["dividend"] > 0)]
    days = cal.to_numpy()
    per_day = np.zeros(len(close))
    for _, r in dv.iterrows():
        k = int(np.searchsorted(days, np.datetime64(pd.Timestamp(r["date"]), "ns")))
        if k < len(days):
            per_day[k] += r["dividend"]
    shares = INIT_CASH / close[0]
    eq = np.empty(len(close))
    for i in range(len(close)):
        if per_day[i] > 0 and close[i] > 0:      # 配息當日以收盤價再投入
            shares += shares * per_day[i] / close[i]
        eq[i] = shares * close[i]
    return curve_stats(eq, years)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜下單期間 {WIN_START} 起｜"
          f"{time.time() - t0:.0f} 秒", flush=True)

    rows, cal = [], None
    for label, base, entry in STRATEGIES:
        n_fixed, n_pct = units(S_BY_STRATEGY[label])
        builder = MultiMACD()
        builder.BASE, builder.ENTRY, builder.PRIO = base, entry, "low_price"
        panel = builder.build_panel(data, start_date=WIN_START)
        cal = panel["close"].index
        years = (cal[-1] - cal[0]).days / 365.25
        print(f"\n{label}｜{years:.2f} 年｜{time.time() - t0:.0f} 秒", flush=True)
        for mname, mode, kwargs, n_units in (
                ("定額", "fixed", {"min_invest": INIT_CASH / n_fixed}, n_fixed),
                ("比例", "percent_floor",
                 {"invest_ratio": 1.0 / n_pct, "min_invest": PCT_MIN_INVEST}, n_pct)):
            inst = MultiMACD(initial_cash=INIT_CASH, sizing_mode=mode, **kwargs)
            res = inst.run_panel(panel)          # 買入排序用預設低價優先
            eq = res["equity"].to_numpy(np.float64)
            mult, cagr, dd = curve_stats(eq, years)
            rows.append({"交易策略": label, "投法": mname, "份數": n_units,
                         "交易次數": res["summary"]["交易次數"],
                         "擋單": res["blocked_orders"], "資金倍數": mult,
                         "年化報酬率%": cagr, "最大回撤%": dd,
                         "報酬回撤比": round(cagr / dd, 3) if dd > 0 else None})
            print(f"  {mname}({n_units})：{rows[-1]['交易次數']:,} 筆｜"
                  f"擋單 {rows[-1]['擋單']:,}｜×{mult}｜年化 {cagr}%｜"
                  f"回撤 {dd}%｜報酬回撤比 {rows[-1]['報酬回撤比']}", flush=True)

    years = (cal[-1] - cal[0]).days / 365.25
    mult, cagr, dd = bench_0050(cal, years)
    rows.append({"交易策略": "0050 買進持有（含息）", "投法": "—", "份數": None,
                 "交易次數": 1, "擋單": 0, "資金倍數": mult, "年化報酬率%": cagr,
                 "最大回撤%": dd, "報酬回撤比": round(cagr / dd, 3)})
    print(f"\n0050 買進持有（含息）：×{mult}｜年化 {cagr}%｜回撤 {dd}%｜"
          f"報酬回撤比 {rows[-1]['報酬回撤比']}")
    print("（錨點：×5.51／16.8%／33.8%；對不上就是含息或分割校準有問題）")

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "macd_multi_conclusion.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
