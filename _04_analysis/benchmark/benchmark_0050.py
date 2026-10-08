"""
0050 買進持有（含息）基準線
============================
所有策略系列的**最終比較數值**：結論篇一律拿策略的多股組合權益曲線，跟同期長抱
0050（含息）比。它不屬於任何一支策略，所以獨立成這個模組，不放進策略共用的 common。

**對照區間＝2015-01-01 ~ 2025-12-31，不沿用回測標準區間的 2002 起點**：
0050 在 2015 以前的資料不可靠——parquet／yfinance 在 2014-01-02 有一次假分割（÷4），
yfinance 又缺 2011~2013 的配息，含息報酬算不準。拿策略績效去對一條算不準的基準線
沒有意義，所以只在 0050 尺度一致、含息可算的這段對照。策略那邊指標暖身仍吃全史，
只是從 BENCHMARK_START 才開始下單（`build_panel(start_date=...)` 先算指標再切期間）。

錨點（2015~2025，首日收盤買進、配息當日再投入）：×5.51／年化 16.81%／最大回撤 33.8%。
對不上就是含息或分割校準出了問題。

單獨執行會印出錨點驗算，並把結果存檔，後續文章可直接讀同一份比對：
    python _04_analysis/benchmark/benchmark_0050.py
輸出（`result/` 不進版控）：
    result/benchmark/0050_2015_2025_equity.csv   逐日：收盤價、配息、持有股數、市值
    result/benchmark/0050_2015_2025_summary.csv  資金倍數、年化報酬率%、最大回撤%、報酬回撤比
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from _02_strategy.base.vbt import common  # noqa: E402

BENCH = "0050.TW"
BENCHMARK_START = "2015-01-01"
BENCHMARK_END = common.DEFAULT_END
OUT_DIR = os.path.join(_root, "result", "benchmark")


def curve_stats(eq: np.ndarray, years: float):
    """
    資金倍數、年化報酬率%、最大回撤%（都用逐日市值權益）。
    策略與 0050 一律用這一支算，兩邊的指標定義才會一致。
    """
    mult = eq[-1] / eq[0]
    cagr = (mult ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max() * 100.0
    return round(mult, 2), round(cagr, 2), round(dd, 1)


def bench_0050(cal: pd.DatetimeIndex, years: float, init_cash: float = 1_000_000.0):
    """0050 買進持有含息的（資金倍數, 年化報酬率%, 最大回撤%）；算法見 equity_0050。"""
    return curve_stats(equity_0050(cal, init_cash)["市值"].to_numpy(np.float64), years)


def equity_0050(cal: pd.DatetimeIndex, init_cash: float = 1_000_000.0) -> pd.DataFrame:
    """
    0050 買進持有含息的逐日明細：首日收盤買進，配息當日以收盤價再投入，抱到期末。
    cal 傳策略面板的交易日曆，兩條曲線才在同一組日子上計價。
    回傳欄位：收盤價、配息（當日每股）、持有股數、市值。

    ⚠️ 價格與配息**都已經是還原分割後的單位，不要再自己除以 4**。0050 在 2025-06
    真的 1 拆 4，parquet 的價格已還原（2015-01-05 收 16.64 ＝ 當時實際 66.55 的 1/4），
    而 `dividend_actions.parquet` 的金額同樣已還原——2022-01 的實際現金股利是 3.2 元，
    表裡記 0.8000（＝3.2÷4）；2022-07 實際 1.8，表裡記 0.45。兩邊同一套單位。
    先前多除了一次 4，算出 ×4.31／14.23%，與錨點 ×5.51／16.81%／33.8% 差了 28% 才抓到。
    殖利率也可以當快篩：不調整時期間配息合計 9.005、除以 11 年對平均價 ~35 ＝ 每年約
    2.3%（符合 0050 實際），多除一次只剩 0.65%（明顯不合理）。
    """
    px = pd.read_parquet(os.path.join(common.DATA_DIR, f"{BENCH}.parquet"))
    close = px["close"].reindex(cal).ffill().to_numpy(np.float64)
    dv = pd.read_parquet(common.require_dividend_file())
    dv = dv[(dv["stock_id"] == BENCH) & (dv["dividend"] > 0)]
    days = cal.to_numpy()
    per_day = np.zeros(len(close))
    for _, r in dv.iterrows():
        when = np.datetime64(pd.Timestamp(r["date"]), "ns")
        # 起日前的配息不屬於這段持有：searchsorted 會把它們全歸到第 0 天，
        # 曾因此在首日多記 2013、2014 兩筆共 2.9 元（起始市值變 117 萬；倍數因等比放大不受影響）
        if when < days[0]:
            continue
        k = int(np.searchsorted(days, when))
        if k < len(days):
            per_day[k] += r["dividend"]
    shares = init_cash / close[0]
    held = np.empty(len(close))
    eq = np.empty(len(close))
    for i in range(len(close)):
        if per_day[i] > 0 and close[i] > 0:      # 配息當日以收盤價再投入
            shares += shares * per_day[i] / close[i]
        held[i] = shares
        eq[i] = shares * close[i]
    return pd.DataFrame({"收盤價": close, "配息": per_day, "持有股數": held, "市值": eq},
                        index=pd.Index(cal, name="日期"))


def main() -> int:
    """以 0050 自己的交易日曆驗算錨點，並把逐日明細與統計存檔（result/ 不進版控）。"""
    px = pd.read_parquet(os.path.join(common.DATA_DIR, f"{BENCH}.parquet")).sort_index()
    cal = px.loc[BENCHMARK_START:BENCHMARK_END].index
    years = (cal[-1] - cal[0]).days / 365.25
    curve = equity_0050(cal)
    mult, cagr, dd = curve_stats(curve["市值"].to_numpy(np.float64), years)
    print(f"0050 買進持有（含息）{cal[0].date()} ~ {cal[-1].date()}："
          f"×{mult}｜年化 {cagr}%｜最大回撤 {dd}%")
    print("（錨點：×5.51／16.81%／33.8%）")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"0050_{cal[0].year}_{cal[-1].year}"
    curve_path = os.path.join(OUT_DIR, f"{tag}_equity.csv")
    summary_path = os.path.join(OUT_DIR, f"{tag}_summary.csv")
    curve.to_csv(curve_path, encoding="utf-8-sig")
    pd.DataFrame([{"起日": cal[0].date(), "迄日": cal[-1].date(), "本金": 1_000_000,
                   "資金倍數": mult, "年化報酬率%": cagr, "最大回撤%": dd,
                   "報酬回撤比": round(cagr / dd, 3)}]).to_csv(summary_path, index=False,
                                                               encoding="utf-8-sig")
    print(f"→ {curve_path}\n→ {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
