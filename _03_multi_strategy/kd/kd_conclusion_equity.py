"""
KD 系列結論（八）的回測段：6 交易策略 × 兩種投法的多股權益曲線（2015–2025）
=========================================================================
只負責跑多股回測、存逐日權益曲線與每組的成交統計；挑投法、跟 0050 比、算資金倍數／年化／
回撤在分析段 `_04_analysis/kd/kd_multi_conclusion.py`。拆兩段是為了分層：
_03 只做多股回測、_04 只做分析。

**下單期間從 2015-01-01 起**（0050 基準線的對照區間，見 `_04_analysis/benchmark/benchmark_0050.py`）；
指標暖身仍吃全史（`build_panel` 先算指標再切期間）。買入排序＝低價優先，份數與投法同
`kd_multi_driver.py`（S 讀單股蒙地卡羅表，定額 S/0.2、比例幾何、比例下限 1 萬）。

引擎用精簡引擎 `run_panel_fast`（與多股 driver 同一支，低價平手時按股票代號序，
兩邊數字才一致）；權益曲線＝逐日市值（現金＋持倉當日收盤市值，停牌沿用最後收盤），
不是只看已實現損益——已實現口徑看不到未平倉部位的浮虧，回撤會被低估。

執行：
    python _03_multi_strategy/kd/kd_conclusion_equity.py [--limit N --out <暫存目錄> --mc-csv <MC 表>]
輸出：_02_strategy/kd_strategy/result/kd_multi/
    conclusion_equity.parquet  每欄一組「交易策略｜投法」的逐日權益
    conclusion_runs.csv        每組的規格 9 欄＋份數、擋單（另附「<欄名>_精確」未四捨五入值）
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
from _02_strategy.base.vbt.common import DEFAULT_END  # noqa: E402
from _03_multi_strategy.base.fast_multi import run_panel_fast  # noqa: E402
from _03_multi_strategy.kd.kd_multi_driver import (INIT_CASH, MC_CSV, OUT, entry_name,  # noqa: E402
                                                   exact_stats, load_all, read_s_table, sizings)
from _03_multi_strategy.kd.multi_kd import ANCHORS, ENTRIES, MultiKD  # noqa: E402

# 必須等於 0050 對照區間的起點（_04 基準模組的 BENCHMARK_START）。回測層不往上 import 分析層，
# 所以這裡寫值；分析段讀檔時會檢查兩者一致，不一致直接報錯，不會靜默比錯期間。
WIN_START = "2015-01-01"
EQUITY_NAME = "conclusion_equity.parquet"
RUNS_NAME = "conclusion_runs.csv"


def equity_key(label: str, method: str) -> str:
    """權益表的欄名（回測段寫、分析段讀，兩邊共用這一支）。"""
    return f"{label}｜{method}"


def main() -> int:
    ap = argparse.ArgumentParser(description="KD 結論：多股權益曲線 2015–2025")
    ap.add_argument("--entries", nargs="+", default=ENTRIES, choices=ENTRIES + ANCHORS)
    ap.add_argument("--folder", default=common.DATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（冒煙用，請搭配 --out）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mc-csv", default=MC_CSV)
    a = ap.parse_args()

    s_table = read_s_table(a.mc_csv)
    t0 = time.time()
    data = load_all(a.folder, a.limit)
    print(f"載入 {len(data)} 檔｜下單期間 {WIN_START}~{DEFAULT_END}｜{time.time() - t0:.0f} 秒", flush=True)

    runs, curves, cal = [], {}, None
    for e in a.entries:
        label = entry_name(e)
        m = MultiKD()
        m.ENTRY, m.PRIO = e, "low_price"
        m.initial_cash = INIT_CASH
        panel = m.build_panel(data, start_date=WIN_START, end_date=DEFAULT_END)
        this_cal = panel["close"].index
        # 各組共用同一份交易日曆才能放進同一張寬表；不一致就停，不默默對齊補值
        if cal is not None and not this_cal.equals(cal):
            raise ValueError(f"{label} 的交易日曆與前面各組不一致，無法共用一張權益表")
        cal = this_cal
        print(f"\n{label}｜S={s_table[e]}｜{time.time() - t0:.0f} 秒", flush=True)
        for mname, mode, ratio, floor, n_units in sizings(s_table[e]):
            m.sizing_mode, m.invest_ratio, m.min_invest = mode, ratio, floor
            res = run_panel_fast(m, panel)                 # 面板優先序＝低價（MultiKD.PRIO）
            curves[equity_key(label, mname)] = res["equity"].to_numpy(np.float64)
            runs.append(common.spec_row(res["summary"], 交易策略=label, 進場=e, 投法=mname,
                                        份數=n_units, 擋單=res["blocked_orders"])
                        | exact_stats(res["trades"]))      # 精確值供文章一次進位
            r = runs[-1]
            print(f"  {mname}({n_units})：{r['交易次數']:,} 筆｜擋單 {r['擋單']:,}｜"
                  f"PF {r['獲利因子']}｜最終權益 {res['summary']['最終權益']:,.0f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    eq_path, runs_path = os.path.join(a.out, EQUITY_NAME), os.path.join(a.out, RUNS_NAME)
    pd.DataFrame(curves, index=cal).to_parquet(eq_path)
    pd.DataFrame(runs).to_csv(runs_path, index=False, encoding="utf-8-sig")
    print(f"\n耗時 {time.time() - t0:.0f} 秒 → {eq_path}、{runs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
