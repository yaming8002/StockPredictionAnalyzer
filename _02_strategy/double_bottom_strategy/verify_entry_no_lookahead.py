"""
截斷測試：雙底進場條件（基本版＋優化 #2～#7）不得用到判定日之後的資料
======================================================================
每檔股票、每個門檻、每個變體：同一套程式分別吃「全史」與「截到某日的資料」，兩邊在重疊期間
（截斷日以前）的進場判定必須逐日相同；任何一天不同＝用到了截斷日之後的資訊。
另外檢查基本版改寫（zigzag_turn_low > db_l1）與原寫法（低點序列 > 前一個低點）逐日相同。

執行：python _02_strategy/double_bottom_strategy/verify_entry_no_lookahead.py [--limit 200]
"""
import argparse
import os
import sys
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from _02_strategy.base.vbt import common
from _02_strategy.base.vbt.common import GLITCH
from _02_strategy.double_bottom_strategy.double_bottom_variants import ENTRY_FILTERS, make_variant

CUTOFFS = ("2008-06-30", "2015-12-31", "2020-03-31")
PCTS = (0.03, 0.06, 0.09)


def old_base(df: pd.DataFrame) -> pd.Series:
    """2026-10-20 以前的基本版寫法（逐字）。"""
    lows = df["zigzag_turn_low"].dropna()
    higher = lows > lows.shift(1)
    signal = pd.Series(False, index=df.index)
    signal.loc[higher.index] = higher.to_numpy()
    return signal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    bad, checks, sig_days = 0, 0, 0
    n_stock = 0
    for sid, raw in common.iter_market(common.DATA_DIR, limit=a.limit, exclude=GLITCH):
        if len(raw) < 300:
            continue
        n_stock += 1
        common.ensure_columns(raw)
        for pct in PCTS:
            for entry in ENTRY_FILTERS:
                v = make_variant(entry, pct, liquidity=True)     # 含可成交門檻（5 日均量也一起測）
                full_df = v.add_columns(raw.copy())
                full = v.buy_signal(full_df)
                if entry == "base":
                    checks += 1
                    if not full.equals(old_base(full_df) & (full_df["turnover"] > 10_000_000)):
                        bad += 1
                        print(f"NG 基本版改寫不等價：{sid} {pct}")
                for cut in CUTOFFS:
                    part = raw.loc[:cut]
                    if len(part) < 50:
                        continue
                    sub = v.buy_signal(v.add_columns(part.copy()))
                    checks += 1
                    sig_days += int(sub.sum())
                    if not sub.equals(full.loc[:cut]):
                        bad += 1
                        diff = sub.index[sub != full.loc[:cut]]
                        print(f"NG {sid} {pct} {entry} 截到 {cut}：{len(diff)} 天不同，首日 {diff[0].date()}")
    print(f"{n_stock} 檔 × {len(PCTS)} 門檻 × {len(ENTRY_FILTERS)} 變體 × {len(CUTOFFS)} 截斷日："
          f"比對 {checks:,} 次（截斷段內進場判定 {sig_days:,} 天）｜不一致 {bad}｜{time.time() - t0:.0f} 秒")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
