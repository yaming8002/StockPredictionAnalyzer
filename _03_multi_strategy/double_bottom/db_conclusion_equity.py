"""
雙底系列結論的回測段：五組 × 兩種投法 × 三種排序的多股權益曲線（2015–2025）
==========================================================================
只負責跑多股回測、存逐日權益曲線與每組的成交統計；跟 0050 比、算資金倍數／年化／回撤在分析段
`_04_analysis/double_bottom/db_multi_conclusion.py`。結構同 MACD 的 macd_conclusion_equity。

**三種買入排序（低價／高價／成交金額）全跑**：結論篇從〈雙底（九）〉〈雙底（十）〉全期總獲利的前幾名
挑出要跟 0050 比的組合，前幾名不一定只有低價排序，所以 30 組都備妥 2015–2025 的權益，由分析段挑。
引擎、份數、排序依據（判定日收盤，prio_prev）全部沿用多股 driver，口徑與（九）（十）一致。

**下單期間從 2015-01-01 起**（0050 基準線的對照區間，見 `_04_analysis/benchmark/benchmark_0050.py`）；
指標暖身仍吃全史（build_panel 先算指標再切期間）。權益曲線用逐日市值計價（現金＋持倉當日收盤市值）。

執行：
    python _03_multi_strategy/double_bottom/db_conclusion_equity.py [--limit N]
輸出：_02_strategy/double_bottom_strategy/result/double_bottom_multi/
    conclusion_equity.parquet  每欄一組「交易策略｜投法｜排序」的逐日權益
    conclusion_runs.csv        每組的通用欄位、擋單（另附「<欄名>_精確」未四捨五入值）
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

from _02_strategy.base.vbt.common import DEFAULT_END  # noqa: E402
from _03_multi_strategy.double_bottom.db_multi_driver import (OUT, PICKS, load_all,  # noqa: E402
                                                               prio_prev, row_of, sizings)
from _03_multi_strategy.double_bottom.fast_multi_db import build_panel, run_panel  # noqa: E402

# 必須等於 0050 對照區間的起點（_04 基準模組的 BENCHMARK_START）。回測層不往上 import 分析層，
# 所以這裡寫值；分析段讀檔時會檢查兩者一致。
WIN_START = "2015-01-01"
EQUITY_FILE = os.path.join(OUT, "conclusion_equity.parquet")
RUNS_FILE = os.path.join(OUT, "conclusion_runs.csv")
ORDERINGS = ("低價", "高價", "成交金額")


def equity_key(label: str, method: str, kind: str) -> str:
    """權益表的欄名（回測段寫、分析段讀，兩邊共用這一支）。"""
    return f"{label}｜{method}｜{kind}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    data = load_all(a.limit)
    print(f"載入 {len(data)} 檔｜下單期間 {WIN_START} 起｜{time.time() - t0:.0f} 秒", flush=True)

    runs, curves, cal = [], {}, None
    for label, pct, entry, exit_key, s in PICKS:
        panel = build_panel(data, entry, exit_key, pct, WIN_START, DEFAULT_END)
        this_cal = panel["close"].index
        # 各組共用同一份交易日曆才能放進同一張寬表；不一致就停，不默默對齊補值
        if cal is not None and not this_cal.equals(cal):
            raise ValueError(f"{label} 的交易日曆與前面各組不一致，無法共用一張權益表")
        cal = this_cal
        prios = prio_prev(data, panel["close"])
        print(f"\n{label}｜{time.time() - t0:.0f} 秒", flush=True)
        for mname, n_units, sizing in sizings(s):
            for kind in ORDERINGS:
                res = run_panel(sizing, panel, prios[kind], want_equity=True)
                curves[equity_key(label, mname, kind)] = res["equity"].to_numpy(np.float64)
                runs.append(row_of(res, 交易策略=label, 投法=mname, 排序=kind, 份數=n_units)
                            | {"期末清算淨額_精確": res["summary"]["期末清算淨額_精確"]})
                print(f"  {mname}({n_units}) {kind}：{runs[-1]['交易次數']:,} 筆｜擋單 {runs[-1]['擋單']:,}",
                      flush=True)

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(curves, index=cal).to_parquet(EQUITY_FILE)
    pd.DataFrame(runs).to_csv(RUNS_FILE, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {EQUITY_FILE}、{RUNS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
