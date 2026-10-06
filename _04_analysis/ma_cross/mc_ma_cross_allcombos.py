"""
多股雙均線交叉 × 蒙地卡羅（全 21 組合，分析段）
================================================
讀回測段 `_03_multi_strategy/ma_cross/ma_cross_allcombos_trades.py` 存下的每組多股逐筆交易，
餵 analyze_vbt.monte_carlo（bootstrap 10,000 次）量 最大連敗 S／最大回撤／破產率。這支不跑回測。
勝率由逐筆交易經 `common.summarize_trades` 重算，與多股引擎 summary 用的是同一支函式。

執行（先跑回測段，再跑這支）：
    python _03_multi_strategy/ma_cross/ma_cross_allcombos_trades.py
    python _04_analysis/ma_cross/mc_ma_cross_allcombos.py
輸出：result/mc/ma_cross_mc.csv ＋ 終端表。
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _03_multi_strategy.ma_cross.ma_cross_allcombos_trades import (INIT_CASH,  # noqa: E402
                                                                   combos, trades_path)
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402

N_SIMS = 10_000
RUIN_RATIO = 0.8            # 破產定義：權益跌破本金 80%（資金防線）
OUT = common.result_dir("ma_strategy", "mc")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"逐一讀 {len(combos())} 組合的逐筆交易 → MC {N_SIMS} 次 …", flush=True)
    print("短/長|成交|勝率%|S中位|S_P95|S_P99|S極值|回撤%中位|回撤%P95|回撤%極值|破產%<80", flush=True)

    rows = []
    for s, l in combos():
        path = trades_path(s, l)
        if not os.path.isfile(path):
            raise SystemExit(f"找不到逐筆交易：{path}\n先跑回測段："
                             "python _03_multi_strategy/ma_cross/ma_cross_allcombos_trades.py")
        trades = pd.read_parquet(path)
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=N_SIMS,
                         ruin_ratio=RUIN_RATIO)
        wr = common.summarize_trades(trades)["勝率(%)"]
        row = {"短": s, "長": l, "成交": mc.get("每次抽樣筆數", 0), "勝率%": wr,
               "S中位": mc.get("最大連敗_中位"), "S_P95": mc.get("最大連敗_P95"),
               "S_P99": mc.get("最大連敗_P99"), "S極值": mc.get("最大連敗_極值"),
               "回撤%中位": mc.get("最大回撤%_中位"), "回撤%P95": mc.get("最大回撤%_P95"),
               "回撤%極值": mc.get("最大回撤%_極值"),
               "破產%<80": mc.get(f"破產機率(<{RUIN_RATIO:.0%})")}
        rows.append(row)
        print(f"{s}/{l}|{row['成交']}|{wr:.1f}|{row['S中位']}|{row['S_P95']}|"
              f"{row['S_P99']}|{row['S極值']}|{row['回撤%中位']}|{row['回撤%P95']}|"
              f"{row['回撤%極值']}|{row['破產%<80']}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "ma_cross_mc.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nALL_DONE -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
