"""
新系列候選（海龜 vs 雙底）20 組的單股蒙地卡羅 → S 與兩種投法份數（分析段）
==========================================================================
讀單股回測段輸出的逐筆交易（cases.py 的 trades 路徑，可成交 1,000 萬版），不跑回測。
口徑與 MACD 系列第十篇相同（`_04_analysis/macd/macd_montecarlo.py`）：
  10,000 條路徑、seed 42、起始 100 萬定額累加、破產＝跌破本金五成；
  部署筆數 T ∈ [交易日數×3, 交易日數×4]＝[17,847, 23,796]；歷史筆數 < 下限 → T＝歷史筆數全納。
S＝最大連敗 P95；份數（資金防線 floor 0.8）：
  定額 round(S/0.2)（線性）、比例 round(1/(1−0.8^(1/S)))（幾何）——兩投法份數不同、不可共用。

執行：python _04_analysis/candidates/candidates_montecarlo.py
輸出：_02_strategy/double_bottom_strategy/result/candidates_multi/candidates_montecarlo.csv
（多股 driver 從這份讀 S 與份數）
"""
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from _02_strategy.base.vbt import common  # noqa: E402
from _03_multi_strategy.candidates.cases import CASES  # noqa: E402
from _04_analysis.analyze_vbt import monte_carlo  # noqa: E402

PATHS = 10_000
SEED = 42
INIT_CASH = 1_000_000.0
RUIN_RATIO = 0.5
FLOOR = 0.80
TRADING_DAYS = 5_949          # 2002-01-01 ~ 2025-12-31 的交易日數（同 MACD 第十篇）
T_LOW, T_HIGH = TRADING_DAYS * 3, TRADING_DAYS * 4
OUT = common.result_dir("double_bottom_strategy", "candidates_multi")
MC_FILE = os.path.join(OUT, "candidates_montecarlo.csv")


def units(s: int):
    """(定額份數, 比例份數)。兩式不同，並列避免混用。"""
    return int(round(s / (1.0 - FLOOR))), int(round(1.0 / (1.0 - FLOOR ** (1.0 / s))))


def main() -> int:
    t0 = time.time()
    rows = []
    for case in CASES:
        trades = pd.read_csv(case["trades"], encoding="utf-8-sig", usecols=["real_pnl"])
        trades = trades[trades["real_pnl"] != 0]       # 淨損益 0 不算勝負（與 summarize_trades 同）
        mc = monte_carlo(trades, initial_cash=INIT_CASH, n_sims=PATHS, ruin_ratio=RUIN_RATIO,
                         seed=SEED, t_low=T_LOW, t_high=T_HIGH)
        s = mc["最大連敗_P95"]
        n_fixed, n_pct = units(s)
        rows.append({"key": case["key"], "交易策略": case["label"], "單股筆數": mc["歷史筆數"],
                     "抽樣模式": "抽區間" if mc["歷史筆數"] >= T_LOW else "全納",
                     "最大連敗 P95(S)": s, "定額份數": n_fixed, "定額每筆": round(INIT_CASH / n_fixed),
                     "比例份數": n_pct, "比例每筆(初始)": round(INIT_CASH / n_pct),
                     "破產%": mc[f"破產機率(<{RUIN_RATIO:.0%})"],
                     "最大回撤% P95": mc["最大回撤%_P95"]})
        print(f"{case['label']}：{mc['歷史筆數']:,} 筆｜S={s}｜定額 {n_fixed}／比例 {n_pct} 份｜"
              f"{time.time() - t0:.0f} 秒", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(MC_FILE, index=False, encoding="utf-8-sig")
    print(f"→ {MC_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
