"""
4 組候選 × 比例投入 × 三種固定排序：台股實際費稅版（vbt fixed_fees）多股回測
==========================================================================
組合：雙底 7%、8%（CHoCH 或跌破波段低點）、雙底 9%（只用 CHoCH）、海龜系統 2（對照）；全部可成交版。
跟 candidates_multi_driver.py 同流程（全期 2002–2025 統計 ＋ 2015–2025 逐日權益），差在：
  - 只跑比例投入（份數同一張 S 表）；
  - 每格跑兩種費用口徑：rate（純費率，即前一輪）與 tw（台股實際費稅：手續費 0.1425% 最低 20 元、
    證交稅 0.3%，vbt 下單 fees=0＋fixed_fees＝當筆實際費稅、成交價過 tick）。tw 版的權益曲線與
    逐筆 real_pnl 是同一本帳（恆等式與精簡引擎對帳見 verify_tw_fees.py）。

執行：python _03_multi_strategy/candidates/candidates_tw_driver.py [--limit N]
輸出：_02_strategy/double_bottom_strategy/result/candidates_multi/tw/
  tw_runs.csv              4 組 × 三排序 × 兩口徑的全期結果
  equity_2015_tw.parquet   同上的 2015–2025 逐日權益（欄名 key｜費用｜排序）
"""
import argparse
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _02_strategy.base.vbt.common import DEFAULT_END, DEFAULT_START  # noqa: E402
from _03_multi_strategy.candidates import candidates_multi_driver as base  # noqa: E402
from _03_multi_strategy.candidates.cases import CASES, run_with_fees  # noqa: E402
from _03_multi_strategy.macd.macd_multi_driver import prio_panels  # noqa: E402

PICK = ("db_p7_stop", "db_p8_stop", "db_p9_nostop", "turtle_s2")
OUT = os.path.join(base.OUT, "tw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    mc = pd.read_csv(base.MC_FILE, encoding="utf-8-sig").set_index("key")
    t0 = time.time()
    base._init(a.limit)
    data = base._DATA
    print(f"載入 {len(data)} 檔｜{time.time() - t0:.0f} 秒", flush=True)
    rows, curves = [], {}
    for key in PICK:
        case = next(c for c in CASES if c["key"] == key)
        n_pct = int(mc.loc[key, "比例份數"])
        inst = case["make"](initial_cash=base.INIT_CASH, sizing_mode="percent_floor",
                            invest_ratio=1.0 / n_pct, min_invest=base.PCT_MIN_INVEST)
        for start, tag in ((DEFAULT_START, "全期"), (base.WIN_START, "2015")):
            panel = case["make"]().build_panel(data, start, DEFAULT_END)
            prios = prio_panels(data, panel["close"])
            for fees in ("rate", "tw"):
                for kind in base.ORDERINGS:
                    res = run_with_fees(case, inst, panel, prios[kind], True, fees)
                    if tag == "2015":
                        curves[f"{key}｜{fees}｜{kind}"] = res["equity"]
                    else:
                        s = res["summary"]
                        rows.append(common.spec_row(s, key=key, 交易策略=case["label"], 費用=fees, 排序=kind,
                                                    份數=n_pct, 擋單=res["blocked_orders"])
                                    | {"資金倍數": round(s["最終權益"] / base.INIT_CASH, 2),
                                       "最大回撤%": round(s["最大回撤(%)"], 1)})
                    del res
            del panel, prios
            print(f"{case['label']}｜{tag}｜{time.time() - t0:.0f} 秒", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tw_runs.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_parquet(os.path.join(OUT, "equity_2015_tw.parquet"))
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
