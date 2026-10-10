"""
MACD 系列結論的回測段：五組交易策略 × 兩種投法的多股權益曲線（2015–2025）
==========================================================================
只負責跑多股回測、存逐日權益曲線與每組的成交統計；跟 0050 比、算資金倍數／年化／
回撤在分析段 `_04_analysis/macd/macd_multi_conclusion.py`。拆兩段是為了分層：
_03 只做多股回測、_04 只做分析。

**下單期間從 2015-01-01 起**（0050 基準線的對照區間，見 `_04_analysis/benchmark/benchmark_0050.py`）；
指標暖身仍吃 2002 起的資料（`build_panel` 先算指標再切期間）。

**權益曲線用逐日市值計價**（現金 ＋ 持倉當日市值），不是只看已實現損益——已實現
口徑看不到未平倉部位的浮虧，回撤會被低估、顯得比實際樂觀。

引擎用精簡引擎 `run_panel_fast`（2026-10-10 起，原為 vbt `run_panel`）：現金軌跡用台股實際費稅
（最低 20 元、證交稅、無條件進位），權益曲線與逐筆 real_pnl 是同一本帳；vbt 版是純費率，
最終權益會與「本金＋總獲利」對不上。低價平手時按股票代號序（與多股 driver 同一支）。

執行：
    python _03_multi_strategy/macd/macd_conclusion_equity.py [--limit N]
輸出：_02_strategy/macd_strategy/result/macd_multi/
    conclusion_equity.parquet  每欄一組「交易策略｜投法」的逐日權益
    conclusion_runs.csv        每組的份數、規格 9 欄、擋單（另附「<欄名>_精確」未四捨五入值）
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _03_multi_strategy.base.fast_multi import exact_stats, run_panel_fast  # noqa: E402
from _03_multi_strategy.macd.macd_multi_driver import (INIT_CASH, OUT,  # noqa: E402
                                                       PCT_MIN_INVEST, S_BY_STRATEGY,
                                                       load_all, units)
from _03_multi_strategy.macd.multi_macd import STRATEGIES, MultiMACD  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END  # noqa: E402

# 必須等於 0050 對照區間的起點（_04 基準模組的 BENCHMARK_START）。回測層不往上 import 分析層，
# 所以這裡寫值；分析段讀檔時會檢查兩者一致，不一致直接報錯，不會靜默比錯期間。
WIN_START = "2015-01-01"
EQUITY_FILE = os.path.join(OUT, "conclusion_equity.parquet")
RUNS_FILE = os.path.join(OUT, "conclusion_runs.csv")


def equity_key(label: str, method: str) -> str:
    """權益表的欄名（回測段寫、分析段讀，兩邊共用這一支）。"""
    return f"{label}｜{method}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜下單期間 {WIN_START} 起｜{time.time() - t0:.0f} 秒", flush=True)

    runs, curves, cal = [], {}, None
    for label, base, entry in STRATEGIES:
        n_fixed, n_pct = units(S_BY_STRATEGY[label])
        builder = MultiMACD()
        builder.BASE, builder.ENTRY, builder.PRIO = base, entry, "low_price"
        panel = builder.build_panel(data, start_date=WIN_START, end_date=DEFAULT_END)
        this_cal = panel["close"].index
        # 各組共用同一份交易日曆才能放進同一張寬表；不一致就停，不默默對齊補值
        if cal is not None and not this_cal.equals(cal):
            raise ValueError(f"{label} 的交易日曆與前面各組不一致，無法共用一張權益表")
        cal = this_cal
        print(f"\n{label}｜{time.time() - t0:.0f} 秒", flush=True)
        for mname, mode, kwargs, n_units in (
                ("定額", "fixed", {"min_invest": INIT_CASH / n_fixed}, n_fixed),
                ("比例", "percent_floor",
                 {"invest_ratio": 1.0 / n_pct, "min_invest": PCT_MIN_INVEST}, n_pct)):
            inst = MultiMACD(initial_cash=INIT_CASH, sizing_mode=mode, **kwargs)
            res = run_panel_fast(inst, panel)    # 買入排序用面板的低價優先
            curves[equity_key(label, mname)] = res["equity"].to_numpy(np.float64)
            runs.append(common.spec_row(res["summary"], 交易策略=label, 投法=mname,
                                        份數=n_units, 擋單=res["blocked_orders"])
                        | exact_stats(res["trades"]))      # 精確值供文章一次進位
            print(f"  {mname}({n_units})：{runs[-1]['交易次數']:,} 筆｜擋單 {runs[-1]['擋單']:,}",
                  flush=True)

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(curves, index=cal).to_parquet(EQUITY_FILE)
    pd.DataFrame(runs).to_csv(RUNS_FILE, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {EQUITY_FILE}、{RUNS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
